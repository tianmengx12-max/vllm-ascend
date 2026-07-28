# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Per-request Active Decode tracking on the Decode Engine Scheduler.

Lifecycle (design §4.2 / §4.2.2):
  - prompt first ready → ``active_inc(kind=prompt, token_ids=...)``
  - ``len(all_token_ids)//block_size`` rises → mint opaque hash → ``active_inc(output)``
  - request finished (before free) → one ``active_dec`` with prompt/all/output hashes

``request_id`` is never placed on the wire.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.logger import logger

from vllm_ascend.distributed.active_decode.publisher import ActiveDecodeZmqPublisher

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.request import Request


def _mint_output_block_hash() -> int:
    """Opaque u64 ledger key (Engine-local; conductor does not remint)."""
    return secrets.randbits(64)


def _token_ids_as_list(token_ids: Any) -> list[int]:
    if token_ids is None:
        return []
    if isinstance(token_ids, list):
        return list(token_ids)
    return list(token_ids)


@dataclass
class _RequestActiveState:
    last_quot: int = 0
    output_block_hashes: list[int] = field(default_factory=list)
    prompt_inc_sent: bool = False


class ActiveDecodeTracker:
    """Tracks prompt/output active edges and publishes via ZMQ."""

    def __init__(
        self,
        publisher: ActiveDecodeZmqPublisher,
        *,
        block_size: int,
        dp_rank: int = 0,
    ) -> None:
        self._publisher = publisher
        self._block_size = max(1, block_size)
        self._dp_rank = dp_rank
        self._states: dict[str, _RequestActiveState] = {}

    @property
    def block_size(self) -> int:
        return self._block_size

    def on_prompt_ready(self, request: "Request") -> None:
        """Single-edge: first time this request attaches prompt blocks."""
        rid = request.request_id
        state = self._states.get(rid)
        if state is not None and state.prompt_inc_sent:
            return

        prompt_token_ids = _token_ids_as_list(getattr(request, "prompt_token_ids", None))
        if not prompt_token_ids:
            logger.debug(
                "ActiveDecodeTracker: skip prompt_inc for %s (empty prompt_token_ids)",
                rid,
            )
            return

        if state is None:
            state = _RequestActiveState()
            self._states[rid] = state

        state.output_block_hashes = []
        state.last_quot = len(prompt_token_ids) // self._block_size
        state.prompt_inc_sent = True

        event = {
            "event_id": self._publisher.next_event_id(),
            "dp_rank": self._dp_rank,
            "type": "active_inc",
            "kind": "prompt",
            "token_ids": prompt_token_ids,
        }
        self._publisher.publish_events([event])
        logger.debug(
            "ActiveDecodeTracker: active_inc(prompt) rid=%s tokens=%d last_quot=%d",
            rid,
            len(prompt_token_ids),
            state.last_quot,
        )

    def on_tokens_updated(self, request: "Request") -> None:
        """After ``all_token_ids`` grew: emit output inc when quotient rises."""
        rid = request.request_id
        state = self._states.get(rid)
        if state is None or not state.prompt_inc_sent:
            # Prompt edge not yet seen; try attach once then re-check.
            self.on_prompt_ready(request)
            state = self._states.get(rid)
            if state is None or not state.prompt_inc_sent:
                return

        all_token_ids = _token_ids_as_list(getattr(request, "all_token_ids", None))
        q = len(all_token_ids) // self._block_size
        if q <= state.last_quot:
            return

        # One edge per integer step of the quotient (spec decode may jump).
        events: list[dict[str, Any]] = []
        while state.last_quot < q:
            state.last_quot += 1
            h = _mint_output_block_hash()
            state.output_block_hashes.append(h)
            events.append(
                {
                    "event_id": self._publisher.next_event_id(),
                    "dp_rank": self._dp_rank,
                    "type": "active_inc",
                    "kind": "output",
                    "output_block_hash": h,
                }
            )
        self._publisher.publish_events(events)
        logger.debug(
            "ActiveDecodeTracker: active_inc(output)x%d rid=%s last_quot=%d",
            len(events),
            rid,
            state.last_quot,
        )

    def on_request_finished(self, request: "Request") -> None:
        """Emit one ``active_dec`` before Request tokens are discarded."""
        rid = request.request_id
        state = self._states.pop(rid, None)
        if state is None or not state.prompt_inc_sent:
            return

        prompt_token_ids = _token_ids_as_list(getattr(request, "prompt_token_ids", None))
        all_token_ids = _token_ids_as_list(getattr(request, "all_token_ids", None))
        event = {
            "event_id": self._publisher.next_event_id(),
            "dp_rank": self._dp_rank,
            "type": "active_dec",
            "prompt_token_ids": prompt_token_ids,
            "all_token_ids": all_token_ids,
            "output_block_hashes": list(state.output_block_hashes),
            "end_of_request": True,
        }
        self._publisher.publish_events([event])
        logger.debug(
            "ActiveDecodeTracker: active_dec rid=%s prompt=%d all=%d output_hashes=%d",
            rid,
            len(prompt_token_ids),
            len(all_token_ids),
            len(state.output_block_hashes),
        )

    def discard(self, request_id: str) -> None:
        """Drop local state without dec (e.g. never successfully inc'd)."""
        self._states.pop(request_id, None)

    def shutdown(self) -> None:
        self._publisher.shutdown()
        self._states.clear()


def maybe_attach_active_decode_tracker(
    scheduler: Any,
    vllm_config: "VllmConfig",
) -> ActiveDecodeTracker | None:
    """Create and attach tracker when env / config enables Active Decode PUB."""
    from vllm_ascend import envs as ascend_envs

    if not ascend_envs.VLLM_ASCEND_ENABLE_ACTIVE_DECODE_EVENTS:
        return None

    # Prefill producers must not emit active_* (cache plane only).
    kv_transfer = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer is not None:
        role = getattr(kv_transfer, "kv_role", None)
        if role == "kv_producer":
            logger.info(
                "Active Decode events disabled on kv_producer (Prefill); use Decode/consumer role"
            )
            return None

    endpoint = ascend_envs.VLLM_ASCEND_ACTIVE_DECODE_ZMQ_ENDPOINT
    if not endpoint:
        logger.warning(
            "VLLM_ASCEND_ENABLE_ACTIVE_DECODE_EVENTS=1 but "
            "VLLM_ASCEND_ACTIVE_DECODE_ZMQ_ENDPOINT is empty; tracker not attached"
        )
        return None

    block_size = getattr(vllm_config.cache_config, "block_size", 128) or 128
    dp_rank = getattr(vllm_config.parallel_config, "data_parallel_rank", 0) or 0
    try:
        publisher = ActiveDecodeZmqPublisher(
            endpoint,
            topic=ascend_envs.VLLM_ASCEND_ACTIVE_DECODE_ZMQ_TOPIC or "",
            instance_id=ascend_envs.VLLM_ASCEND_ACTIVE_DECODE_INSTANCE_ID or "",
            block_size=int(block_size),
            dp_rank=int(dp_rank),
        )
    except Exception:
        logger.exception("Failed to create ActiveDecodeZmqPublisher")
        return None

    tracker = ActiveDecodeTracker(publisher, block_size=int(block_size), dp_rank=int(dp_rank))
    scheduler._active_decode_tracker = tracker  # type: ignore[attr-defined]
    logger.info(
        "Active Decode tracker attached (endpoint=%s, block_size=%d, dp_rank=%d)",
        endpoint,
        block_size,
        dp_rank,
    )
    return tracker
