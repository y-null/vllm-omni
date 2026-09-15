#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A bespoke decode attention for the Talker, in place of the general FIA.

``npu_fused_infer_attention_score`` costs **25 us** per call inside a captured
replay at the Talker's decode shape -- one query row, 12 heads, 64-wide, a
paged bfloat16 cache -- against a 1.7 us per-kernel floor on the same die and a
~2 us roofline for the KV it reads. Raising the declared capacity from 128 to
512 adds only 1.3 us, so essentially all of it is fixed cost, and no argument
we can pass makes it smaller: the layout and block-size sweep moves it by 3 us
at best.

``TalkerDecodeAttention`` is that same computation written for this shape and
nothing else. It measures **6.8 us**, and it is *more* accurate than the op it
replaces -- bit-exact against an fp32 reference where FIA carries 1e-3.
Twenty layers times ~118 codec frames makes the difference ~43 ms a request.

Two things make it cheaper than a general operator can be:

* one AI vector core per (request, head), so a core owns a whole head, needs no
  cross-core reduction and writes its own 64 outputs; and
* the sequence length arrives as an ``int32`` device tensor rather than a
  ``-inf`` bias, so the kernel reads only the pages the sequence occupies.
  Fixed-KV declares 512 to keep the host arguments constant for graph capture,
  but a Talker utterance is ~140 tokens -- the general op reads the declared
  capacity, this one reads the real prefix.

The operator ships in the same payload as the A14 codec sampler and loads
through the same bridge, so it needs no environment variable to be live. When
the payload is absent -- a source checkout, a non-Ascend host, another chip --
:func:`available` is False and the caller keeps FIA.

``VLLM_OMNI_TALKER_DECODE_OP=0`` forces the FIA path, which is how the A/B
control arm is run.
"""

from __future__ import annotations

import os

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# The v1 boundary, mirrored from the operator's own host-side tiling check.
_HEAD_DIM = 64
_MAX_CAPACITY = 4096
_MAX_BLOCK_SIZE = 128
_ENABLE_ENV = "VLLM_OMNI_TALKER_DECODE_OP"

_probed = False
_op = None
_cores: int | None = None


def _vector_cores() -> int:
    """AI vector cores on this chip, which bound the operator's batch x heads.

    The kernel gives one core a whole (request, head) and the tiling refuses a
    blockDim above the core count. That refusal happens during graph capture,
    where it is an exception rather than a fallback, so the count has to be
    known here. 48 on 910C: 12 heads leaves room for four requests.
    """
    global _cores
    if _cores is None:
        try:
            _cores = int(torch.npu.get_device_properties(0).vector_core_num)
        except Exception:  # noqa: BLE001 - a non-Ascend host, or an older API
            _cores = 0
    return _cores


def enabled() -> bool:
    return os.environ.get(_ENABLE_ENV, "1").strip() not in {"0", "false", "off"}


def _resolve():
    """Return the registered operator, loading the shared payload bridge once."""
    global _probed, _op
    if _probed:
        return _op
    _probed = True
    if not enabled():
        return None
    # One bridge registers every operator in the payload, so this is the same
    # load the codec sampler performs; whichever runs first pays for it.
    from vllm_omni.model_executor.models.minicpmo_4_5.talker_codec_sample import (
        load_operator_bridge,
    )

    if not load_operator_bridge():
        return None
    namespace = getattr(torch.ops, "vllm_omni_npu", None)
    _op = getattr(namespace, "talker_decode_attention_out", None) if namespace is not None else None
    if _op is None:
        logger.warning(
            "Operator payload loaded but talker_decode_attention_out is not registered; "
            "the Talker keeps the general decode attention"
        )
    else:
        logger.info("[minicpmo] bespoke Talker decode attention active")
    return _op


def available() -> bool:
    return _resolve() is not None


def applies(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seq_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    num_heads: int,
    num_kv_heads: int,
    rows: int,
    q_len: int,
) -> bool:
    """Whether this step is inside the operator's declared domain.

    Checked here rather than left to the operator's tiling, because a tiling
    that returns GRAPH_FAILED during graph capture is an exception, not a
    fallback.
    """
    if q_len != 1 or num_heads != num_kv_heads:
        return False
    # One core per (request, head), so the step has to fit on the chip.
    if rows * num_heads > _vector_cores():
        return False
    if query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16:
        return False
    if key.shape != value.shape or value.dtype != torch.bfloat16:
        return False
    if query.shape != (rows, num_heads, _HEAD_DIM):
        return False
    if key.dim() != 3 or key.shape[1] != block_size or key.shape[2] != num_kv_heads * _HEAD_DIM:
        return False
    if block_size < 8 or block_size > _MAX_BLOCK_SIZE or block_size & (block_size - 1):
        return False
    if block_size * block_table.shape[1] > _MAX_CAPACITY:
        return False
    if seq_lens is None or seq_lens.dtype != torch.int32 or seq_lens.shape != (rows,):
        return False
    if block_table.dtype != torch.int32 or block_table.shape[0] != rows:
        return False
    if not (query.is_contiguous() and key.is_contiguous() and block_table.is_contiguous()):
        return False
    return True


def emit(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    scale: float,
    out: torch.Tensor,
) -> None:
    """Record the attention into the graph being captured.

    Every argument is a device buffer the capture bakes in, including the
    sequence length -- so a replay needs nothing from the host, which is the
    property the whole fixed-KV decode is built around.
    """
    _op(query, key, value, block_table, seq_lens, num_heads, num_kv_heads, scale, out)
