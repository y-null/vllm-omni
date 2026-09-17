"""Triton flash-decode decode-attention（q_len=1 GQA，短 KV，online softmax）。

机理：KV 按 slot 表分块加载、online softmax 单 pass 流式累积，避免整段 KV
物化。bf16 输入 fp32 累积。
实现：自研 Triton-Ascend 内核。
集成点：vllm_omni/platforms/npu/attention/fixed_kv_backend.py 的 decode
attention 调用处，用本模块 forward 替换原生 gather+matmul+softmax 链。
env 门：MINICPMO_P85_FLASH_DECODE（默认 "0"，关闭时恒走 torch 回退）。
"""
import os

import torch

_P85_ENABLED = os.environ.get("MINICPMO_P85_FLASH_DECODE", "0") == "1"

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - non-Triton env
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _flash_decode_kernel(
        q_ptr,  # (B*H, D)
        k_ptr,  # (slots, KVH, D)
        v_ptr,  # (slots, KVH, D)
        idx_ptr,  # (B, L) slot table, padded with 0
        lens_ptr,  # (B,)
        out_ptr,  # (B, H, D)
        stride_idx, stride_ks, stride_kh, stride_vs, stride_vh,
        stride_ob, stride_oh, scale,
        D: tl.constexpr, BLOCK_KV: tl.constexpr, GQA: tl.constexpr,
    ):
        pid_h = tl.program_id(0)
        pid_b = tl.program_id(1)
        kv_head = pid_h // GQA
        seq_len = tl.load(lens_ptr + pid_b)
        d = tl.arange(0, D)
        q = tl.load(q_ptr + (pid_b * (tl.num_programs(0)) + pid_h) * D + d).to(tl.float32)
        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros([D], dtype=tl.float32)
        for start in range(0, seq_len, BLOCK_KV):
            offs = start + tl.arange(0, BLOCK_KV)
            mask = offs < seq_len
            idx = tl.load(idx_ptr + pid_b * stride_idx + offs, mask=mask, other=0)
            k = tl.load(
                k_ptr + idx[:, None] * stride_ks + kv_head * stride_kh + d[None, :],
                mask=mask[:, None], other=0.0,
            ).to(tl.float32)
            s = tl.sum(k * q[None, :], axis=1) * scale
            s = tl.where(mask, s, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=0))
            p = tl.exp(s - m_new)
            v = tl.load(
                v_ptr + idx[:, None] * stride_vs + kv_head * stride_vh + d[None, :],
                mask=mask[:, None], other=0.0,
            ).to(tl.float32)
            l_i = l_i * tl.exp(m_i - m_new) + tl.sum(p, axis=0)
            acc = acc * tl.exp(m_i - m_new) + tl.sum(p[:, None] * v, axis=0)
            m_i = m_new
        acc = acc / l_i
        tl.store(out_ptr + pid_b * stride_ob + pid_h * stride_oh + d, acc)


def _flash_decode_torch(q, k_cache, v_cache, kv_indices, seq_lens):
    """原生 torch 回退：gather + bmm + masked softmax。"""
    b, h, d = q.shape
    kvh = k_cache.shape[1]
    gqa = h // kvh
    idx = kv_indices[:, : int(seq_lens.max())].to(torch.long)
    k = k_cache[idx]  # (B, L, KVH, D)
    v = v_cache[idx]
    k = k.repeat_interleave(gqa, dim=2).transpose(1, 2)  # (B, H, L, D)
    v = v.repeat_interleave(gqa, dim=2).transpose(1, 2)
    scores = torch.matmul(q.unsqueeze(2), k.transpose(-1, -2)).squeeze(2) / (d ** 0.5)
    mask = torch.arange(scores.shape[-1], device=q.device)[None, None, :] < seq_lens[:, None, None]
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(probs.unsqueeze(2), v).squeeze(2)  # (B, H, D)


def flash_decode(q, k_cache, v_cache, kv_indices, seq_lens, block_kv=64):
    """q (B,H,D)；k/v_cache (slots,KVH,D)；kv_indices (B,L) slot 表；seq_lens (B,)。"""
    if not (_P85_ENABLED and _HAS_TRITON):
        return _flash_decode_torch(q, k_cache, v_cache, kv_indices, seq_lens)
    b, h, d = q.shape
    kvh = k_cache.shape[1]
    gqa = h // kvh
    out = torch.empty((b, h, d), device=q.device, dtype=q.dtype)
    qf = q.reshape(b * h, d).contiguous()
    grid = (h, b)
    _flash_decode_kernel[grid](
        qf, k_cache, v_cache, kv_indices.to(torch.long), seq_lens.to(torch.int32), out,
        kv_indices.stride(0), k_cache.stride(0), k_cache.stride(1),
        v_cache.stride(0), v_cache.stride(1),
        out.stride(0), out.stride(1),
        1.0 / (d ** 0.5),
        D=d, BLOCK_KV=block_kv, GQA=gqa,
        num_warps=4,
    )
    return out
