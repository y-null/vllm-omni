"""P47: FIA KV 刷新 C++ 化（slot 填写 + 页表更新）。

机理：把每帧的 KV slot 写入与页表更新从 Python/torch 算子链改为 C++ 扩展
（并行 memcpy，省多次小 kernel launch 与 Python 开销）。本模块只落源码与
驱动：预编译 .so 由 MINICPMO_P47_FIA_SO 指定路径后 importlib 加载，未编译
或未启用时恒走 torch 等价回退（index_copy_ / 高级索引写）。
来源：自行设计（entry_02 P47 描述，标准 torch extension 结构）。
集成点：fixed_kv_backend.py::OmniFixedKVMetadataBuilder.build 的 slot_mapping
捕获处与 runner 逐层 KV 写入点。
env 门：MINICPMO_P47_FIA_CPP（默认 "0"，关闭恒走 torch 回退）。
"""
import importlib.util
import os

import torch

_P47_ENABLED = os.environ.get("MINICPMO_P47_FIA_CPP", "0") == "1"
_CPP_SOURCE = r'''
#include <torch/extension.h>
#include <cstring>

// cache: (num_slots, num_kv_heads, head_dim) 连续；rows: (T, KVH, D)；slot_ids: (T,)
void fill_kv_slots(torch::Tensor cache, torch::Tensor rows, torch::Tensor slot_ids) {
  const int64_t T = slot_ids.size(0);
  const int64_t hd = cache.size(1) * cache.size(2);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, cache.scalar_type(), "fill_kv_slots", [&] {
    scalar_t* dst = cache.data_ptr<scalar_t>();
    const scalar_t* src = rows.data_ptr<scalar_t>();
    const int64_t* ids = slot_ids.data_ptr<int64_t>();
    const int64_t bytes = hd * sizeof(scalar_t);
    at::parallel_for(0, T, 0, [&](int64_t s, int64_t e) {
      for (int64_t t = s; t < e; ++t) {
        std::memcpy(dst + ids[t] * hd, src + t * hd, bytes);
      }
    });
  });
}

// page_table: (B, L) int64；seq_lens: (B,) 当前长度；new_slots: (B,) 本步新 slot
void update_page_table(torch::Tensor page_table, torch::Tensor seq_lens, torch::Tensor new_slots) {
  const int64_t B = page_table.size(0);
  int64_t* pt = page_table.data_ptr<int64_t>();
  const int64_t* sl = seq_lens.data_ptr<int64_t>();
  const int64_t* ns = new_slots.data_ptr<int64_t>();
  const int64_t stride = page_table.stride(0);
  at::parallel_for(0, B, 0, [&](int64_t s, int64_t e) {
    for (int64_t b = s; b < e; ++b) {
      pt[b * stride + sl[b]] = ns[b];
    }
  });
}

PYBIND11_MODULE(_vllm_omni_p47_fia, m) {
  m.def("fill_kv_slots", &fill_kv_slots, "parallel KV slot fill");
  m.def("update_page_table", &update_page_table, "page table append");
}
'''

_EXT = None


def _load_ext():
    """加载预编译扩展（仅 importlib，不触发构建）。"""
    global _EXT
    if _EXT is not None:
        return _EXT
    so_path = os.environ.get("MINICPMO_P47_FIA_SO", "")
    if not so_path or not os.path.exists(so_path):
        return None
    try:
        spec = importlib.util.spec_from_file_location("_vllm_omni_p47_fia", so_path)
        _EXT = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_EXT)
    except Exception:
        _EXT = None
    return _EXT


def save_source(path):
    """把内嵌 C++ 源码落盘，供将来 torch.utils.cpp_extension.load 编译。"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(_CPP_SOURCE)
    return path


def fill_kv_slots(cache, rows, slot_ids):
    """等价 torch 回退：cache.index_copy_(0, slot_ids, rows)。"""
    ext = _load_ext() if _P47_ENABLED else None
    if ext is not None:
        return ext.fill_kv_slots(cache, rows, slot_ids.to(torch.long))
    cache.index_copy_(0, slot_ids.to(torch.long), rows)
    return cache


def update_page_table(page_table, seq_lens, new_slots):
    """等价 torch 回退：page_table[arange(B), seq_lens] = new_slots。"""
    ext = _load_ext() if _P47_ENABLED else None
    if ext is not None:
        return ext.update_page_table(page_table, seq_lens.to(torch.long), new_slots.to(torch.long))
    b = page_table.shape[0]
    page_table[torch.arange(b, device=page_table.device), seq_lens.to(torch.long)] = new_slots.to(torch.long)
    return page_table
