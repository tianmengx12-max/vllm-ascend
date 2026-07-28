# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Patch Scheduler helpers to emit KV-Aware Decode ``active_inc`` / ``active_dec``.

Design: MindIE-PyMotor ``kv_aware_decode_via_kv_events.md`` Phase 2.

Hooks target **non-overridden** Scheduler helpers so Ascend subclasses
(``SchedulerDynamicBatch``, ``RecomputeScheduler``, ``BalanceScheduler``, …)
still fire events:

  - ``__init__`` → attach ``ActiveDecodeTracker`` (+ wrap connector alloc)
  - ``_try_promote_blocked_waiting_request`` → prompt ready when remote KV clears
  - ``_update_request_with_output`` → output quotient edges after tokens append
  - ``_free_request`` → ``active_dec`` **before** tokens / Request are dropped

``request_id`` never appears on the ZMQ wire.
"""

from __future__ import annotations

from typing import Any

from vllm.logger import logger
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_ascend.distributed.active_decode.tracker import maybe_attach_active_decode_tracker

_ORIGINAL_SCHEDULER_INIT = Scheduler.__init__
_ORIGINAL_UPDATE_REQUEST_WITH_OUTPUT = getattr(Scheduler, "_update_request_with_output", None)
_ORIGINAL_FREE_REQUEST = getattr(Scheduler, "_free_request", None)
_ORIGINAL_TRY_PROMOTE = getattr(Scheduler, "_try_promote_blocked_waiting_request", None)
_patched = False


def _get_tracker(scheduler: Any):
    return getattr(scheduler, "_active_decode_tracker", None)


def _safe_prompt_ready(tracker: Any, request: Any) -> None:
    try:
        tracker.on_prompt_ready(request)
    except Exception:
        logger.exception(
            "ActiveDecodeTracker.on_prompt_ready failed for %s",
            getattr(request, "request_id", "?"),
        )


def _safe_tokens_updated(tracker: Any, request: Any) -> None:
    try:
        tracker.on_tokens_updated(request)
    except Exception:
        logger.exception(
            "ActiveDecodeTracker.on_tokens_updated failed for %s",
            getattr(request, "request_id", "?"),
        )


def _safe_request_finished(tracker: Any, request: Any) -> None:
    try:
        tracker.on_request_finished(request)
    except Exception:
        logger.exception(
            "ActiveDecodeTracker.on_request_finished failed for %s",
            getattr(request, "request_id", "?"),
        )


def _wrap_connector_alloc(scheduler: Any, tracker: Any) -> None:
    """Fire prompt_inc when Decode allocates blocks for remote/local prompt KV."""
    connector = getattr(scheduler, "connector", None)
    if connector is None or not hasattr(connector, "update_state_after_alloc"):
        return
    if getattr(connector, "_active_decode_alloc_wrapped", False):
        return

    original = connector.update_state_after_alloc

    def _wrapped(request: Any, blocks: Any, num_external_tokens: int, *args: Any, **kwargs: Any):
        params = getattr(request, "kv_transfer_params", None) or {}
        # Snapshot before connector clears do_remote_prefill.
        want_prompt = bool(params.get("do_remote_prefill")) or num_external_tokens > 0
        result = original(request, blocks, num_external_tokens, *args, **kwargs)
        if want_prompt:
            _safe_prompt_ready(tracker, request)
        return result

    connector.update_state_after_alloc = _wrapped  # type: ignore[method-assign]
    connector._active_decode_alloc_wrapped = True  # type: ignore[attr-defined]


def _patched_scheduler_init(self: Any, *args: Any, **kwargs: Any) -> None:
    _ORIGINAL_SCHEDULER_INIT(self, *args, **kwargs)
    try:
        tracker = maybe_attach_active_decode_tracker(self, self.vllm_config)
        if tracker is not None:
            _wrap_connector_alloc(self, tracker)
    except Exception:
        logger.exception("Failed to attach Active Decode tracker")


def _patched_update_request_with_output(self: Any, request: Any, *args: Any, **kwargs: Any):
    assert _ORIGINAL_UPDATE_REQUEST_WITH_OUTPUT is not None
    result = _ORIGINAL_UPDATE_REQUEST_WITH_OUTPUT(self, request, *args, **kwargs)
    tracker = _get_tracker(self)
    if tracker is not None:
        _safe_tokens_updated(tracker, request)
    return result


def _patched_free_request(self: Any, request: Any, *args: Any, **kwargs: Any):
    assert _ORIGINAL_FREE_REQUEST is not None
    tracker = _get_tracker(self)
    if tracker is not None:
        # Must run before upstream frees tokens / drops Request.
        _safe_request_finished(tracker, request)
    return _ORIGINAL_FREE_REQUEST(self, request, *args, **kwargs)


def _patched_try_promote(self: Any, request: Any, *args: Any, **kwargs: Any):
    assert _ORIGINAL_TRY_PROMOTE is not None
    promoted = _ORIGINAL_TRY_PROMOTE(self, request, *args, **kwargs)
    tracker = _get_tracker(self)
    # When remote KV becomes ready, ensure prompt_inc was sent (idempotent).
    if tracker is not None and promoted:
        _safe_prompt_ready(tracker, request)
    return promoted


def apply_active_decode_patch() -> None:
    global _patched
    if _patched:
        return

    Scheduler.__init__ = _patched_scheduler_init  # type: ignore[method-assign]

    if _ORIGINAL_UPDATE_REQUEST_WITH_OUTPUT is not None:
        Scheduler._update_request_with_output = _patched_update_request_with_output  # type: ignore[method-assign]
    else:
        logger.warning(
            "Scheduler._update_request_with_output missing; Active Decode output edges disabled"
        )

    if _ORIGINAL_FREE_REQUEST is not None:
        Scheduler._free_request = _patched_free_request  # type: ignore[method-assign]
    else:
        logger.warning("Scheduler._free_request missing; Active Decode active_dec disabled")

    if _ORIGINAL_TRY_PROMOTE is not None:
        Scheduler._try_promote_blocked_waiting_request = _patched_try_promote  # type: ignore[method-assign]

    _patched = True
    logger.info("Active Decode Scheduler patch applied")


apply_active_decode_patch()
