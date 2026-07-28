# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Unit tests for Active Decode tracker / publisher (KV-Aware Decode)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm_ascend.distributed.active_decode.publisher import ActiveDecodeZmqPublisher
from vllm_ascend.distributed.active_decode.tracker import (
    ActiveDecodeTracker,
    maybe_attach_active_decode_tracker,
)


class _FakePublisher:
    def __init__(self) -> None:
        self.batches: list[list[dict]] = []
        self._eid = 0

    def next_event_id(self) -> int:
        self._eid += 1
        return self._eid

    def publish_events(self, events: list[dict]) -> None:
        self.batches.append(events)

    def shutdown(self) -> None:
        pass


def _make_request(
    rid: str = "req-1",
    prompt: list[int] | None = None,
    all_tokens: list[int] | None = None,
) -> SimpleNamespace:
    prompt = prompt if prompt is not None else list(range(20))
    all_tokens = all_tokens if all_tokens is not None else list(prompt)
    return SimpleNamespace(
        request_id=rid,
        prompt_token_ids=prompt,
        all_token_ids=all_tokens,
    )


class TestActiveDecodeTracker(unittest.TestCase):
    def setUp(self) -> None:
        self.pub = _FakePublisher()
        self.tracker = ActiveDecodeTracker(self.pub, block_size=16, dp_rank=0)

    def test_prompt_ready_emits_active_inc_once(self) -> None:
        req = _make_request(prompt=list(range(20)))
        self.tracker.on_prompt_ready(req)
        self.tracker.on_prompt_ready(req)  # idempotent
        self.assertEqual(len(self.pub.batches), 1)
        ev = self.pub.batches[0][0]
        self.assertEqual(ev["type"], "active_inc")
        self.assertEqual(ev["kind"], "prompt")
        self.assertEqual(ev["token_ids"], list(range(20)))
        self.assertNotIn("request_id", ev)
        self.assertEqual(ev["dp_rank"], 0)

    def test_output_inc_on_quotient_increase(self) -> None:
        req = _make_request(prompt=list(range(20)), all_tokens=list(range(20)))
        self.tracker.on_prompt_ready(req)
        self.pub.batches.clear()

        # len=31 → quot still 1 → no output inc
        req.all_token_ids = list(range(31))
        self.tracker.on_tokens_updated(req)
        self.assertEqual(self.pub.batches, [])

        # len=32 → quot 2 → one output inc with opaque hash
        req.all_token_ids = list(range(32))
        self.tracker.on_tokens_updated(req)
        self.assertEqual(len(self.pub.batches), 1)
        ev = self.pub.batches[0][0]
        self.assertEqual(ev["type"], "active_inc")
        self.assertEqual(ev["kind"], "output")
        self.assertIn("output_block_hash", ev)
        self.assertIsInstance(ev["output_block_hash"], int)
        self.assertNotIn("token_ids", ev)
        self.assertNotIn("request_id", ev)

    def test_output_inc_jumps_multiple_quotients(self) -> None:
        req = _make_request(prompt=list(range(16)), all_tokens=list(range(16)))
        self.tracker.on_prompt_ready(req)
        self.pub.batches.clear()

        # Jump from quot=1 to quot=3 in one update (spec-decode style).
        req.all_token_ids = list(range(48))
        self.tracker.on_tokens_updated(req)
        flat = [e for batch in self.pub.batches for e in batch]
        self.assertEqual(len(flat), 2)  # quot 2 and 3
        hashes = [e["output_block_hash"] for e in flat]
        self.assertEqual(len(hashes), len(set(hashes)))

    def test_active_dec_returns_output_hashes(self) -> None:
        req = _make_request(prompt=list(range(20)), all_tokens=list(range(20)))
        self.tracker.on_prompt_ready(req)
        req.all_token_ids = list(range(32))
        self.tracker.on_tokens_updated(req)
        hashes = list(self.tracker._states[req.request_id].output_block_hashes)
        self.pub.batches.clear()

        self.tracker.on_request_finished(req)
        self.assertEqual(len(self.pub.batches), 1)
        ev = self.pub.batches[0][0]
        self.assertEqual(ev["type"], "active_dec")
        self.assertEqual(ev["prompt_token_ids"], list(range(20)))
        self.assertEqual(ev["all_token_ids"], list(range(32)))
        self.assertEqual(ev["output_block_hashes"], hashes)
        self.assertTrue(ev["end_of_request"])
        self.assertNotIn("request_id", ev)
        # Second finish is no-op
        self.tracker.on_request_finished(req)
        self.assertEqual(len(self.pub.batches), 1)

    def test_finish_without_prompt_inc_is_noop(self) -> None:
        req = _make_request()
        self.tracker.on_request_finished(req)
        self.assertEqual(self.pub.batches, [])


class TestMaybeAttach(unittest.TestCase):
    def test_disabled_by_default(self) -> None:
        import vllm_ascend.envs as ascend_envs

        scheduler = SimpleNamespace()
        vllm_config = SimpleNamespace(
            kv_transfer_config=None,
            cache_config=SimpleNamespace(block_size=16),
            parallel_config=SimpleNamespace(data_parallel_rank=0),
        )
        with patch.object(ascend_envs, "VLLM_ASCEND_ENABLE_ACTIVE_DECODE_EVENTS", False):
            result = maybe_attach_active_decode_tracker(scheduler, vllm_config)
        self.assertIsNone(result)

    def test_skip_kv_producer(self) -> None:
        import vllm_ascend.envs as ascend_envs

        scheduler = SimpleNamespace()
        vllm_config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_role="kv_producer"),
            cache_config=SimpleNamespace(block_size=16),
            parallel_config=SimpleNamespace(data_parallel_rank=0),
        )
        with (
            patch.object(ascend_envs, "VLLM_ASCEND_ENABLE_ACTIVE_DECODE_EVENTS", True),
            patch.object(ascend_envs, "VLLM_ASCEND_ACTIVE_DECODE_ZMQ_ENDPOINT", "tcp://127.0.0.1:15570"),
        ):
            result = maybe_attach_active_decode_tracker(scheduler, vllm_config)
        self.assertIsNone(result)


class TestPublisherPayload(unittest.TestCase):
    def test_publish_msgpack_map_without_request_id(self) -> None:
        fake_sock = MagicMock()
        fake_ctx = MagicMock()
        fake_ctx.socket.return_value = fake_sock
        fake_zmq = MagicMock()
        fake_zmq.Context.instance.return_value = fake_ctx
        fake_zmq.PUB = object()
        fake_zmq.NOBLOCK = 1
        fake_zmq.Again = type("Again", (Exception,), {})

        with patch("vllm_ascend.distributed.active_decode.publisher.zmq", fake_zmq):
            pub = ActiveDecodeZmqPublisher(
                "tcp://127.0.0.1:15570",
                topic="kv-events",
                instance_id="vllm-decode-3",
                block_size=16,
                dp_rank=0,
            )
            pub.publish_events(
                [
                    {
                        "event_id": 1,
                        "dp_rank": 0,
                        "type": "active_inc",
                        "kind": "prompt",
                        "token_ids": [1, 2, 3],
                    }
                ]
            )

        fake_sock.send_multipart.assert_called_once()
        topic, seq, payload = fake_sock.send_multipart.call_args[0][0]
        self.assertEqual(topic, b"kv-events")
        self.assertEqual(len(seq), 8)
        import msgpack

        batch = msgpack.unpackb(payload, raw=False)
        self.assertEqual(batch["block_size"], 16)
        self.assertEqual(batch["instance_id"], "vllm-decode-3")
        self.assertEqual(batch["events"][0]["type"], "active_inc")
        self.assertNotIn("request_id", batch)
        self.assertNotIn("request_id", batch["events"][0])


if __name__ == "__main__":
    unittest.main()
