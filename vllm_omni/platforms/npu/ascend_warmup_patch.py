# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Skip the vllm-ascend Triton warmups that fault the 910_93 vector core.

vllm-ascend warms three Triton kernel families on the worker before graph
capture (``vllm_ascend/model_executor/warmup/kernel_warmup.py``). On 910C/A3
the second one, ``penalties_triton_warmup``, faults inside its dummy-token
setup before any Triton kernel runs:

    tokens[:, -1:] = vocab_size

lowers to ACLNN InplaceCopy over the legacy BroadcastTo kernel, and on
ascend910_93 that kernel raises a vector-core exception -- "The address for
the scalar to access the internal buffer of AICore is out of bounds", aclnn
error 507035 -- which kills the stage worker during initialization and takes
the whole orchestrator down with it.

The identical line runs clean on 910B3 under the same CANN 9.1.0 and
torch_npu 2.10.0.post4 (verified directly, with the same shapes), so this is
a 910_93 operator fault rather than a shape, dtype or memory problem -- the
warmup input is only ``[max_num_seqs, 257]`` token ids. Rewriting the line
would merely relocate the fault into an operator nobody can verify on the
target box, so the warmup is skipped instead.

Skipping is safe by construction: the warmup only pre-compiles Triton kernels
(JIT), and the sampling path runs those kernels over real token tensors, never
through this dummy construction. The cost is a one-off compile on the first
penalised sample of the first request. Correctness is unaffected.

Scoped so the 910B baseline keeps the stock warmup byte-for-byte. Probes that
cannot name the SoC skip the faulting warmup as well: a missing warmup costs
latency, a faulting one costs the whole run.
``VLLM_OMNI_NPU_SKIP_WARMUPS`` overrides the set (comma separated substrings
of the warmup names, or "none" to restore stock behaviour).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False
_ENV = "VLLM_OMNI_NPU_SKIP_WARMUPS"
# Substrings matched against the warmup function names in kernel_warmup.py.
_DEFAULT_SKIP = ("penalties",)
_WARMUP_NAMES = (
    "rejection_sampler_triton_warmup",
    "penalties_triton_warmup",
    "triton_rms_warmup",
)


def _probe_soc_name() -> str:
    """torch_npu device name, or "" when it cannot be read yet."""
    try:
        import torch_npu

        try:
            device = torch_npu.npu.current_device()
        except Exception:
            device = 0
        return str(torch_npu.npu.get_device_name(device))
    except Exception:
        return ""


def _skipped_names() -> set[str]:
    raw = os.environ.get(_ENV, "").strip().lower()
    if raw:
        if raw in ("none", "off", "0", "false", "no"):
            return set()
        return {part.strip() for part in raw.split(",") if part.strip()}
    soc = _probe_soc_name()
    if soc.startswith("Ascend910B"):
        # Verified clean here; keep the stock warmup untouched.
        return set()
    return set(_DEFAULT_SKIP)


def _make_no_op(name: str) -> Callable[[Any], None]:
    def _skip(worker: Any) -> None:  # noqa: ARG001 - signature must match
        logger.info(
            "[npu] %s skipped: the legacy BroadcastTo kernel faults the "
            "910_93 vector core during its dummy-token setup",
            name,
        )

    return _skip


def apply_ascend_warmup_patch() -> None:
    """Replace the warmups selected by ``VLLM_OMNI_NPU_SKIP_WARMUPS``."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    skipped = _skipped_names()
    if not skipped:
        return

    try:
        from vllm_ascend.model_executor.warmup import kernel_warmup as warmup_module
    except Exception as error:  # pragma: no cover - non-ascend builds
        logger.warning("[npu] Ascend warmup patch not applied: %s", error)
        return

    applied: list[str] = []
    for name in _WARMUP_NAMES:
        if not any(fragment in name for fragment in skipped):
            continue
        if not hasattr(warmup_module, name):
            continue
        setattr(warmup_module, name, _make_no_op(name))
        applied.append(name)

    if applied:
        logger.info(
            "[npu] skipped ascend Triton warmups on this SoC: %s "
            "(restore with %s=none)",
            ", ".join(applied),
            _ENV,
        )
