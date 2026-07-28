# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""ZMQ PUB for Active Decode ``active_inc`` / ``active_dec`` events.

Wire format matches MindIE-PyMotor kv-conductor Active Decode SUB
(design §5): multipart ``[topic, seq:u64 BE, msgpack_map]`` where the map
mirrors HTTP ``POST /events`` (``block_size`` + ``events``). Payload must
**not** carry ``request_id``.
"""

from __future__ import annotations

import itertools
import threading
from typing import Any

import msgpack
from vllm.logger import logger

try:
    import zmq
except ImportError:  # pragma: no cover
    zmq = None  # type: ignore[assignment]


class ActiveDecodeZmqPublisher:
    """Fire-and-forget ZMQ PUB for Active Decode events."""

    def __init__(
        self,
        endpoint: str,
        *,
        topic: str = "",
        instance_id: str = "",
        block_size: int = 128,
        dp_rank: int = 0,
    ) -> None:
        if zmq is None:
            raise ImportError("pyzmq is required for Active Decode ZMQ publisher")
        if not endpoint:
            raise ValueError("Active Decode ZMQ endpoint must be non-empty")

        self._endpoint = endpoint
        self._topic = topic.encode("utf-8")
        self._instance_id = instance_id
        self._block_size = block_size
        self._dp_rank = dp_rank
        self._seq = itertools.count()
        self._lock = threading.Lock()
        self._event_id = itertools.count(1)

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        # Bind if wildcard / ipc / inproc; otherwise connect (volatile peer).
        if "*" in endpoint or endpoint.startswith(("ipc://", "inproc://")) or "::" in endpoint:
            self._sock.bind(endpoint)
            logger.info("ActiveDecodeZmqPublisher bound to %s", endpoint)
        else:
            self._sock.connect(endpoint)
            logger.info("ActiveDecodeZmqPublisher connected to %s", endpoint)

    def next_event_id(self) -> int:
        return next(self._event_id)

    def publish_events(self, events: list[dict[str, Any]]) -> None:
        if not events:
            return
        batch: dict[str, Any] = {
            "block_size": self._block_size,
            "events": events,
        }
        if self._instance_id:
            batch["instance_id"] = self._instance_id
        payload = msgpack.packb(batch, use_bin_type=True)
        seq_bytes = next(self._seq).to_bytes(8, "big", signed=False)
        with self._lock:
            try:
                self._sock.send_multipart((self._topic, seq_bytes, payload), flags=zmq.NOBLOCK)
            except zmq.Again:
                logger.warning(
                    "ActiveDecodeZmqPublisher HWM full; dropping %d active event(s)",
                    len(events),
                )
            except Exception:
                logger.exception("ActiveDecodeZmqPublisher failed to send active events")

    def shutdown(self) -> None:
        with self._lock:
            try:
                self._sock.close(linger=0)
            except Exception:
                pass
