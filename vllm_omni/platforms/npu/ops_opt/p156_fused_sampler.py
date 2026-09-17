"""P156: 融合采样器驱动（top-k/top-p 融合 floor + 噪声 argmax 采样尾）。

机理：温度/重复惩罚之后的采样尾收拢为一个融合核调用
（torch_npu.npu_top_k_top_p），候选 floor 用设备常量缓存避免每次 H2D；
回退为等价 torch 参考实现；采样尾提供 multinomial 与 gumbel-argmax 两种。
来源：参考 KuaaMU v8.2 `_npu_top_k_top_p_warp`/`_apply_top_k_top_p` 重写。
集成点：minicpmo_4_5_omni_tts.py::MiniCPMOTTSCodecLogitsProcessor 采样尾，
替换其 if/else 分支（约 686-699 行）。
env 门：MINICPMO_P156_FUSED_SAMPLER（默认 "0"，关闭恒走参考实现）。
"""
import os

import torch

_P156_ENABLED = os.environ.get("MINICPMO_P156_FUSED_SAMPLER", "0") == "1"

try:
    import torch_npu  # noqa: F401

    _HAS_NPU = hasattr(torch_npu, "npu_top_k_top_p")
except Exception:  # pragma: no cover
    torch_npu = None
    _HAS_NPU = False

_CONST_CACHE: dict = {}


def _device_constants(device: str, dtype: torch.dtype, top_p: float, top_k: int):
    key = (device, str(dtype), float(top_p), int(top_k))
    cached = _CONST_CACHE.get(key)
    if cached is None:
        p = torch.full((1,), float(top_p), device=device, dtype=dtype)
        k = torch.full((1,), int(top_k), device=device, dtype=torch.int32)
        _CONST_CACHE[key] = (p, k)
    return _CONST_CACHE[key]


def reference_top_k_top_p(logits, top_k=None, top_p=None, min_tokens_to_keep=3):
    """等价参考实现（与上游 warper 语义一致，浮点级可复现）。"""
    filtered = logits.clone()
    vocab = filtered.shape[-1]
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(filtered, descending=False, dim=-1)
        cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cum <= (1.0 - float(top_p))
        remove[..., -min_tokens_to_keep:] = False
        filtered = filtered.masked_fill(remove.scatter(-1, sorted_idx, remove), float("-inf"))
    if top_k is not None and top_k > 0:
        keep = min(vocab, max(int(top_k), min_tokens_to_keep))
        threshold = torch.topk(filtered, keep, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))
    return filtered


def fused_top_k_top_p(logits, top_k, top_p):
    """融合核路径；失败自动回落参考实现。"""
    if not (_P156_ENABLED and _HAS_NPU):
        return reference_top_k_top_p(logits, top_k=top_k, top_p=top_p)
    p_dev, k_dev = _device_constants(str(logits.device), logits.dtype, top_p, top_k)
    try:
        return torch_npu.npu_top_k_top_p(
            logits, p_dev.expand(logits.shape[0]).contiguous(),
            k_dev.expand(logits.shape[0]).contiguous(),
        )
    except Exception:
        return reference_top_k_top_p(logits, top_k=top_k, top_p=top_p)


def fused_sample(logits, top_k=None, top_p=None, temperature=1.0, mode="multinomial"):
    """完整采样尾：floor -> softmax -> 采样。mode: multinomial | gumbel。"""
    if temperature != 1.0:
        logits = logits / temperature
    logits = fused_top_k_top_p(logits, top_k=top_k, top_p=top_p)
    probs = torch.softmax(logits, dim=-1)
    if mode == "gumbel":
        u = torch.rand_like(probs)
        gumbel = -torch.log(-torch.log(u.clamp_min(1e-20)))
        return (probs.log().clamp_min(-1e20) + gumbel).argmax(dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
