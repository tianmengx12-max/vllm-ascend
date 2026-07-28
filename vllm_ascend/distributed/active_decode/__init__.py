# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""KV-Aware Decode: Engine-side active_inc / active_dec ZMQ reporting."""

from vllm_ascend.distributed.active_decode.tracker import (
    ActiveDecodeTracker,
    maybe_attach_active_decode_tracker,
)

__all__ = [
    "ActiveDecodeTracker",
    "maybe_attach_active_decode_tracker",
]
