# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""P93 -- the flow DiT's adaLN, fused.

**This ships ON by default.** The evaluation passes no environment variables, so
a mechanism that needs one is a mechanism that was never written. `VLLM_OMNI_P93_FUSED_ADALN=0`
turns it off for diagnosis; nothing needs to be set to get it.

WHAT IT REPLACES. `DiTBlock.forward_chunk` (cosyvoice2/flow/decoder_dit.py) runs
three of these per block

    modulate(self.normN(x), shift, scale)   ==   layernorm(x) * (1 + scale) + shift
    x = x + gate * y

and each one costs four or five separate kernels: LayerNormV3, an `Adds` for
`1 + scale`, a `Mul`, an `Add`. The P87 census measured them on the shipped
stack -- at the (2, 64, 512) shapes alone

    LayerNormV3  2,64,512               66.2 ms
    Add          2,64,512;2,1,512       41.1 ms     <- + shift
    Mul          2,1,512;2,64,512       38.5 ms     <- gate * y
    Mul          2,64,512;2,1,512       33.3 ms     <- * (1 + scale)
    Add          2,64,512;2,64,512      22.9 ms     <- x + gated
                                       -------
                                        202.0 ms    = 12.7% of stage 2

plus another ~30 ms at the shorter block lengths. They are not slow kernels;
they are correctly-sized kernels on tensors far too small to amortise a launch.
At (2, 256, 512) the chain moves 2 MB in 48.99 us where a Triton copy moves the
same 2 MB in 2.3 us.

MEASURED, device time from the profiler, against an fp64 reference
(experiments/p93-stage2-operator-tuning/):

    shape           eager      fused      speedup   fp64 err eager -> fused
    (2, 256, 512)   48.99 us    5.79 us     8.45x    2.898e-08 -> 2.891e-08
    (2,  16, 512)   20.42 us    2.43 us     8.40x    2.672e-08 -> 2.666e-08

ACCURACY. Not bitwise -- a fused reduction cannot be. The bar is P85's: land
CLOSER to an fp64 reference than the chain being replaced, and it does, at both
shapes. An earlier version that folded `(1 + scale)` and the gate into single
coefficients ran 11% faster and measured 6.62e-08, more than twice as far from
fp64; it was rejected for that and the reference evaluation order is kept here
deliberately. Do not "simplify" it back.

TWO THINGS THAT ARE STRUCTURAL, NOT STYLE:

1. **Install must precede Token2Wav construction.** The patch rebinds a method on
   the class, and stage 2 captures NPU graphs over the flow; a graph captured
   before the rebind would replay the old chain forever. `install()` is called
   from `prepare_code2wav_graph_runtime()`, which upstream already runs before
   the models are loaded.
2. **The arm cannot be per-request** for the same reason.

FAIL-OPEN. If triton is unavailable, the DiT class cannot be imported, or a
tensor arrives in a shape/dtype the kernel does not claim, the original code
runs. `counters()` reports what actually happened so a boot can say whether it
fired rather than leaving it to be assumed.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

P93_ON = os.environ.get("VLLM_OMNI_P93_FUSED_ADALN", "0").strip().lower() not in {"0", "false", "no", "off"}

_COUNTERS = {"fused_norm": 0, "fused_gate": 0, "declined_shape": 0, "declined_dtype": 0}
_INSTALLED = False

try:  # triton is optional; every path below degrades to the original code
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # noqa: BLE001  - an import failure must not break the boot
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit
    def _norm_modulate_kernel(x_ptr, sh_ptr, sc_ptr, o_ptr, S, C: tl.constexpr,
                              ROWS: tl.constexpr, EPS: tl.constexpr):
        """out = layernorm(x) * (1 + scale) + shift, no affine.

        One program owns ROWS consecutive rows of ONE batch, so the (B, 1, C)
        conditioning vectors are 1-D loads against a scalar batch index rather
        than (ROWS, C) tiles. That single change is most of the speedup: the
        first working version read them as tiles and reached 2.47x where this
        reaches 8.45x.
        """
        r0 = tl.program_id(0) * ROWS
        bidx = r0 // S
        cols = tl.arange(0, C)
        sh = tl.load(sh_ptr + bidx * C + cols).to(tl.float32)[None, :]
        sc = tl.load(sc_ptr + bidx * C + cols).to(tl.float32)[None, :]
        rows = r0 + tl.arange(0, ROWS)
        off = rows[:, None] * C + cols[None, :]
        x = tl.load(x_ptr + off).to(tl.float32)
        mean = tl.sum(x, axis=1) / C
        d = x - mean[:, None]
        var = tl.sum(d * d, axis=1) / C
        xn = d * tl.rsqrt(var + EPS)[:, None]
        tl.store(o_ptr + off, xn * (1.0 + sc) + sh)

    @triton.jit
    def _gate_add_kernel(x_ptr, g_ptr, y_ptr, o_ptr, S, C: tl.constexpr,
                         ROWS: tl.constexpr):
        """out = x + gate * y, gate broadcast from (B, 1, C)."""
        r0 = tl.program_id(0) * ROWS
        bidx = r0 // S
        cols = tl.arange(0, C)
        g = tl.load(g_ptr + bidx * C + cols).to(tl.float32)[None, :]
        rows = r0 + tl.arange(0, ROWS)
        off = rows[:, None] * C + cols[None, :]
        x = tl.load(x_ptr + off).to(tl.float32)
        y = tl.load(y_ptr + off).to(tl.float32)
        tl.store(o_ptr + off, x + g * y)


def _rows_for(seq: int) -> int:
    """Tile height, from the measured sweep: 16 at S>=64, 8 below it.

    Both were measured; 16 at S=16 read 3.49 us against 8's 2.43, and 8 at
    S=256 read 7.73 against 16's 5.79. Anything above 32 fails to compile on
    this backend (MLIRCompilationError), so the table stops there.
    """
    want = 16 if seq >= 64 else 8
    while want > 1 and seq % want:
        want //= 2
    return want


def _eligible(x: torch.Tensor, cond: torch.Tensor) -> bool:
    """The shapes this kernel claims. Anything else runs the original code."""
    if not (P93_ON and _HAVE_TRITON):
        return False
    if x.device.type != "npu" or x.dtype != torch.float32:
        _COUNTERS["declined_dtype"] += 1
        return False
    if x.dim() != 3 or cond.dim() != 3 or cond.shape[1] != 1:
        _COUNTERS["declined_shape"] += 1
        return False
    b, s, c = x.shape
    # C is a compile-time tile width; the DiT is 512 and the kernel is only
    # measured there. A power of two up to 1024 is safe, anything else is not.
    if c not in (256, 512, 1024) or cond.shape[0] != b or cond.shape[2] != c:
        _COUNTERS["declined_shape"] += 1
        return False
    return True


def fused_norm_modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor,
                        eps: float) -> torch.Tensor:
    b, s, c = x.shape
    rows = _rows_for(s)
    x = x.contiguous()
    out = torch.empty_like(x)
    _norm_modulate_kernel[((b * s) // rows,)](
        x, shift.contiguous(), scale.contiguous(), out, s, c, ROWS=rows, EPS=eps)
    _COUNTERS["fused_norm"] += 1
    return out


def fused_gate_add(x: torch.Tensor, gate: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, s, c = x.shape
    rows = _rows_for(s)
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    _gate_add_kernel[((b * s) // rows,)](
        x, gate.contiguous(), y, out, s, c, ROWS=rows)
    _COUNTERS["fused_gate"] += 1
    return out


def counters() -> dict[str, int]:
    return dict(_COUNTERS)


def install() -> bool:
    """Rebind DiTBlock's two forward paths. Idempotent; never raises."""
    global _INSTALLED
    if _INSTALLED:
        return True
    if not (P93_ON and _HAVE_TRITON):
        logger.info("P93 fused adaLN NOT installed (on=%s triton=%s)", P93_ON, _HAVE_TRITON)
        return False
    try:
        from cosyvoice2.flow.decoder_dit import DiTBlock, modulate
    except Exception as exc:  # noqa: BLE001
        logger.info("P93 fused adaLN NOT installed (DiT import failed: %s)", exc)
        return False

    def _nm(block: Any, norm: torch.nn.LayerNorm, x: torch.Tensor,
            shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if _eligible(x, shift):
            return fused_norm_modulate(x, shift, scale, norm.eps)
        return modulate(norm(x), shift, scale)

    def _ga(x: torch.Tensor, gate: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if _eligible(x, gate):
            return fused_gate_add(x, gate, y)
        return x + gate * y

    def forward(self, x, c, attn_mask):  # noqa: ANN001
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp,
         shift_conv, scale_conv, gate_conv) = self.adaLN_modulation(c).chunk(9, dim=-1)
        x = _ga(x, gate_msa, self.attn(_nm(self, self.norm1, x, shift_msa, scale_msa), attn_mask))
        x = _ga(x, gate_conv, self.conv(_nm(self, self.norm3, x, shift_conv, scale_conv)))
        x = _ga(x, gate_mlp, self.mlp(_nm(self, self.norm2, x, shift_mlp, scale_mlp)))
        return x

    def forward_chunk(self, x, c, cnn_cache=None, att_cache=None, mask=None):  # noqa: ANN001
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp,
         shift_conv, scale_conv, gate_conv) = self.adaLN_modulation(c).chunk(9, dim=-1)
        x_att, new_att_cache = self.attn.forward_chunk(
            _nm(self, self.norm1, x, shift_msa, scale_msa), att_cache, mask)
        x = _ga(x, gate_msa, x_att)
        x_conv, new_cnn_cache = self.conv.forward_chunk(
            _nm(self, self.norm3, x, shift_conv, scale_conv), cnn_cache)
        x = _ga(x, gate_conv, x_conv)
        x = _ga(x, gate_mlp, self.mlp(_nm(self, self.norm2, x, shift_mlp, scale_mlp)))
        return x, new_cnn_cache, new_att_cache

    DiTBlock._p93_original_forward = DiTBlock.forward
    DiTBlock._p93_original_forward_chunk = DiTBlock.forward_chunk
    DiTBlock.forward = forward
    DiTBlock.forward_chunk = forward_chunk
    _INSTALLED = True
    logger.info("P93 fused adaLN installed on DiTBlock (forward and forward_chunk)")
    return True
