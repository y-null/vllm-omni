# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Numerical parity for cache-width bucketing (pad + key-mask == unpadded).

The cache-bucket design pads each request's attention cache up to an aligned
width and masks the pad columns. Production attention concatenates the cached
keys with the current ones (cache at the head of the kv axis), so pad columns
in the cache tail must be excluded from softmax via the key mask -- exactly
the mechanism ``_decode_batch_once`` already uses for mel-frame padding
(``batched_token2wav.py``: "a zero row still occupies part of the softmax
denominator").

These tests pin that math on CPU with a real softmax attention: with the pad
columns masked, output must match the *unpadded* reference regardless of what
the pad columns contain (zeros or garbage). Also pins that the equivalence
survives CFM-style feedback iterations, where each step's output feeds the
next step's input and the two branches evolve independently.
"""

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _attention(x: torch.Tensor, cache: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
    """Single-head softmax attention with the production cache layout.

    x: (B, T, D) current frames; cache: (B, S, D) history at the head of the
    kv axis; key_mask: (B, T, S + T) bool, True = participates in softmax.
    kv = cat([cache, x]) mirrors ``cat([k, k_cache])``-style chunk attention
    (order is irrelevant for parity as long as mask and kv agree).
    """
    kv = torch.cat([cache, x], dim=1)
    scores = torch.einsum("btd,bsd->bts", x, kv) / x.shape[-1] ** 0.5
    scores = scores.masked_fill(~key_mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("bts,bsd->btd", weights, kv)


def _setup(dtype: torch.dtype, pad_value: float):
    """One valid cache, its padded copy, and the estimator-style mask."""
    torch.manual_seed(0)
    batch, valid_cache, query, dim = 2, 37, 5, 32
    aligned = 64
    x = torch.randn(batch, query, dim, dtype=dtype)
    cache_valid = torch.randn(batch, valid_cache, dim, dtype=dtype)
    # Estimator mask: current frames always valid; cache pad columns are not.
    mask = torch.zeros(batch, query, aligned + query, dtype=torch.bool)
    mask[:, :, :valid_cache] = True
    mask[:, :, aligned:] = True
    cache_padded = torch.full((batch, aligned, dim), pad_value, dtype=dtype)
    cache_padded[:, :valid_cache] = cache_valid
    reference_mask = torch.ones(batch, query, valid_cache + query, dtype=torch.bool)
    return x, cache_valid, cache_padded, mask, reference_mask


_ATOL = {torch.float32: 1e-5, torch.float16: 2e-3}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("pad_value", [0.0, 7.5])
def test_padded_cache_with_mask_matches_unpadded(dtype: torch.dtype, pad_value: float):
    """Masked pad columns must not move the output, whatever they contain."""
    x, cache_valid, cache_padded, mask, reference_mask = _setup(dtype, pad_value)

    reference = _attention(x, cache_valid, reference_mask)
    padded = _attention(x, cache_padded, mask)

    torch.testing.assert_close(padded, reference, rtol=0, atol=_ATOL[dtype])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_equivalence_survives_cfm_feedback_iterations(dtype: torch.dtype):
    """Two independently evolving branches must stay equivalent.

    CFM feeds each step's output back as the next input; pin that a masked
    pad region neither shifts nor amplifies across iterations.
    """
    x0, cache_valid, cache_padded, mask, reference_mask = _setup(dtype, 0.0)
    aligned = mask.shape[2] - x0.shape[1]
    valid_cache = cache_valid.shape[1]

    # The reference kv axis is unpadded; the padded branch carries pad cols.
    x_ref = x0.clone()
    x_pad = x0.clone()
    cache_ref = cache_valid
    dt = 0.02
    for step in range(10):
        out_ref = _attention(x_ref, cache_ref, reference_mask)
        out_pad = _attention(x_pad, cache_padded, mask)
        torch.testing.assert_close(out_pad, out_ref, rtol=0, atol=_ATOL[dtype])
        x_ref = x_ref + dt * out_ref
        x_pad = x_pad + dt * out_pad
        # External cache update (same for both branches over the valid span;
        # pad columns of the padded branch stay masked).
        cache_ref = cache_ref + dt
        cache_padded[:, :valid_cache] = cache_ref
    torch.testing.assert_close(x_pad, x_ref, rtol=0, atol=_ATOL[dtype])


def test_align_estimator_cache_state_machine_matches_ideal_window():
    """The aligned cache carries the same frames as the ideal window, always.

    Simulates setup (pad once to the aligned width) followed by decode chunks
    whose output cache is the padded cache plus the new frames. Frame contents
    are unique ids, so per-position equality pins the time ordering: after
    every chunk the valid prefix must match the ideal (never padded) rolling
    window, and the pad suffix must stay zero until the steady state fills it.
    """
    from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import (
        _align_estimator_cache,
        _cache_align_width,
    )

    prompt_len, bucket, chunk = 304, 64, 80
    cache_width = _cache_align_width(prompt_len + 100, bucket)
    # ceil((304+100)/64) = 7 grid steps
    assert cache_width == 448
    keep = cache_width - prompt_len
    # Real estimator caches are 6-D (depth, batch, cfg, heads, kv, att_width);
    # mock with distinct per-att_width values so any axis confusion shows up.
    frame_ids = torch.arange(4096, dtype=torch.float32).reshape(1, 1, 1, 1, -1, 1)
    frame_ids = frame_ids.expand(1, 1, 1, 1, -1, 3) + torch.arange(3.0).reshape(1, 1, 1, 1, 1, 3)
    cursor = prompt_len

    def next_frames() -> torch.Tensor:
        nonlocal cursor
        frames = frame_ids[..., cursor : cursor + chunk, :]
        cursor += chunk
        return frames

    # Setup: prompt-only cache padded once to the aligned width (kv axis,
    # dim 4; F.pad's tuple starts at the last axis, hence 8 elements).
    att = torch.nn.functional.pad(
        frame_ids[..., :prompt_len, :], (0, 0, 0, cache_width - prompt_len, 0, 0, 0, 0)
    )
    valid = prompt_len
    # Ideal branch: the same rolling window with no padding at all.
    ideal = frame_ids[..., :prompt_len, :]

    for _ in range(4):
        frames = next_frames()
        att_out = torch.cat([att, frames], dim=4)
        att, valid = _align_estimator_cache(
            att_out, prompt_len, cache_width, valid, cache_width
        )
        ideal = torch.cat([ideal, frames], dim=4)
        if ideal.shape[4] > prompt_len + keep:
            ideal = torch.cat([ideal[..., :prompt_len, :], ideal[..., -keep:, :]], dim=4)
        # Physical kv width is constant from the very first chunk.
        assert att.shape[4] == cache_width
        torch.testing.assert_close(att[..., :valid, :], ideal, rtol=0, atol=0)
        assert int(torch.count_nonzero(att[..., valid:, :])) == 0

    assert valid == cache_width
    assert ideal.shape[4] == cache_width


def test_estimator_att_pad_columns():
    """Column mask: True up to each row's valid count, None when all valid."""
    from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import (
        _estimator_att_pad_columns,
    )

    assert _estimator_att_pad_columns(torch.tensor([512]), 512) is None
    ok = _estimator_att_pad_columns(torch.tensor([304]), 512)
    assert ok is not None and ok.shape == (1, 512)
    assert bool(ok[0, :304].all()) and not bool(ok[0, 304:].any())
    per_row = _estimator_att_pad_columns(torch.tensor([304, 512]), 512)
    assert per_row is not None
    assert bool(per_row[0, :304].all()) and not bool(per_row[0, 304:].any())
    assert bool(per_row[1].all())
