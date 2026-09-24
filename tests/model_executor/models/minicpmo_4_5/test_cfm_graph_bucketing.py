# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the CFM mel-frame bucketing decision.

Steady-state chunk lengths vary per request; without bucketing every length
becomes its own CUDA-graph capture shape (428 captures / 13 flushes in the
#6628 regression). ``_cfm_pad_frames`` aligns the frame axis onto a grid so
steady-state calls share one capture shape; ``_decode_cfm`` trims the output
back. These tests pin the decision math on CPU, no CUDA required.

The last three tests pin the cache-width side of the capture-shape explosion:
``_decode_batch_once`` trims each request's attention cache to its own
``prompt_len + 100``, so concurrent requests with distinct reference prompts
own distinct steady-state shapes. ``_cache_align_width`` is the planned
grid-alignment of that trim target (not wired yet): pad the cache once at
setup time to the aligned width, mask the pad columns, and every request
lands on one of a few (chunk bucket, cache bucket) capture shapes.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import (
    _cache_align_width,
    _cfm_pad_frames,
    _zero_padded_cnn_cache,
    _zero_padded_frames,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_pads_partial_chunk_up_to_bucket():
    # 50 frames on a 16-frame grid -> 14 padding frames.
    assert _cfm_pad_frames(mel_frames=50, offset=0, noise_capacity=1000, bucket_frames=16, disabled=False) == 14


def test_aligned_chunk_needs_no_padding():
    assert _cfm_pad_frames(mel_frames=64, offset=0, noise_capacity=1000, bucket_frames=16, disabled=False) == 0


def test_ragged_valid_lengths_path_disables_bucketing():
    """The ragged per-request cache path must never be padded."""
    for ragged in (True,):
        for mel in (50, 64, 96):
            assert (
                _cfm_pad_frames(
                    mel_frames=mel,
                    offset=0,
                    noise_capacity=1000,
                    bucket_frames=16,
                    disabled=ragged,
                )
                == 0
            )


def test_disabled_bucketing_returns_zero():
    """bucket_frames <= 1 (or wrapper absent) means plain eager behavior."""
    for bucket in (0, 1):
        assert (
            _cfm_pad_frames(
                mel_frames=50,
                offset=0,
                noise_capacity=1000,
                bucket_frames=bucket,
                disabled=False,
            )
            == 0
        )


def test_padding_never_overflows_noise_buffer():
    """Padding past the decoder's noise buffer must fall back to zero.

    The padded call slices x from decoder.rand_noise[offset:end]; exceeding
    its width would crash. Bucketing is best-effort, never required.
    """
    # 96 frames + 16 pad = 112 > 100 capacity -> give up, pad 0.
    assert _cfm_pad_frames(mel_frames=96, offset=0, noise_capacity=100, bucket_frames=16, disabled=False) == 0
    # Same mel length fits when un-padded: 96 <= 100.
    assert _cfm_pad_frames(mel_frames=96, offset=0, noise_capacity=100, bucket_frames=1, disabled=False) == 0
    # Cache offset eats into the capacity: 64 + 0 pad would fit, 64 + 48 pad
    # would not, so the pad must be dropped.
    assert _cfm_pad_frames(mel_frames=64, offset=50, noise_capacity=110, bucket_frames=16, disabled=False) == 0
    # The original chunk fits while the padded one does not: 50 + 50 = 100
    # <= 110, but 100 + 14 = 114 > 110, so the pad must be dropped.
    assert _cfm_pad_frames(mel_frames=50, offset=50, noise_capacity=110, bucket_frames=16, disabled=False) == 0
    # Same numbers with room for the padding: 114 <= 120, so it stands.
    assert _cfm_pad_frames(mel_frames=50, offset=50, noise_capacity=120, bucket_frames=16, disabled=False) == 14


def test_padding_applies_with_cache_offset_when_it_fits():
    # offset 50 + mel 50 + pad 14 = 114 <= 200 -> pad stands.
    assert _cfm_pad_frames(mel_frames=50, offset=50, noise_capacity=200, bucket_frames=16, disabled=False) == 14


def test_steady_state_cache_width_saturates_on_bucket_grid():
    """Pin the real recurrence: grow, then trim to ``prompt_len + 100``.

    ``_decode_batch_once`` trims the estimator attention cache after every
    decode, so the width saturates instead of growing without bound. What
    matters for the graph cache is that the steady-state
    ``(chunk_width, cache_width)`` pair settles on a single grid point.
    """
    bucket, prompt_len = 16, 304
    cache_cap = prompt_len + 100
    width = prompt_len
    shapes = set()
    for _ in range(20):
        pad = _cfm_pad_frames(
            mel_frames=50,
            offset=width,
            noise_capacity=30000,
            bucket_frames=bucket,
            disabled=False,
        )
        assert pad == 14
        shapes.add((50 + pad, width))  # (chunk width, cache width) as used
        width = min(width + 50 + pad, cache_cap)
    assert width == cache_cap
    # (64, 304) -> (64, 368) -> (64, 404), then stable.
    assert shapes == {(64, 304), (64, 368), (64, 404)}


def test_varied_chunk_lengths_collapse_onto_few_widths():
    """Bucketing exists for the varied-length calls, not the steady 50-frame one.

    The first/last chunk and the ``plan_token2wav_encode_slices`` splits land on
    arbitrary lengths; those are the calls that would each become their own
    capture shape. Pin that a realistic mix collapses onto a few padded widths.
    """
    varied = (7, 12, 17, 25, 33, 50, 51, 64)
    padded = {
        mel
        + _cfm_pad_frames(
            mel_frames=mel,
            offset=300,
            noise_capacity=30000,
            bucket_frames=16,
            disabled=False,
        )
        for mel in varied
    }
    assert len(padded) == 4, sorted(padded)  # 16 / 32 / 48 / 64
    assert len(padded) < len(varied)


def test_zero_padded_frames_survives_an_integration_step():
    """The padded region must stay zero after ``x = x + dt * velocity``.

    Zeroing once before the CFM loop is not enough: the update touches every
    column, so the padded region becomes non-zero again (0 -> 0.010 -> 0.020
    over the first steps). Pin the helper that re-zeroes it each step.
    """
    mel, pad = 50, 14
    x = torch.zeros(2, 4, mel + pad)
    x[..., :mel] = 0.05
    _zero_padded_frames(x, mel)
    assert torch.all(x[..., mel:] == 0.0)
    for _ in range(3):
        x = x + 0.02 * torch.ones_like(x)  # the integration step
        _zero_padded_frames(x, mel)
        assert torch.all(x[..., mel:] == 0.0)
        assert torch.all(x[..., :mel] > 0.0)


def test_zero_padded_frames_is_a_noop_without_padding():
    x = torch.ones(1, 2, 32)
    _zero_padded_frames(x, None)
    _zero_padded_frames(x, 32)
    assert torch.all(x == 1.0)


def _cnn_cache_estimator(widths):
    return SimpleNamespace(
        blocks=[
            SimpleNamespace(conv=SimpleNamespace(block=[None, SimpleNamespace(causal_padding=(width, 0))]))
            for width in widths
        ]
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize(
    "widths,cache_width,pad_frames,cache_depth",
    [
        ((4, 4), 4, 1, 2),
        ((4, 4), 4, 0, 2),
        ((4, 4), 4, 7, 2),
        ((4, 2), 4, 1, 2),
        ((2, 2), 4, 1, 2),
        ((6, 6), 4, 1, 2),
        ((0, 4, -1), 4, 1, 3),
        ((0, -1), 4, 7, 2),
        ((4, 4), 4, -1, 2),
        ((4, 4), 4, 1, 3),
        ((), 4, 1, 1),
    ],
)
def test_zero_padded_cnn_cache_matches_per_block_writes(
    dtype, noncontiguous, widths, cache_width, pad_frames, cache_depth
):
    # Compare all storage, including the untouched columns of a strided view.
    storage_width = cache_width * (2 if noncontiguous else 1)
    original = torch.arange(cache_depth * 2 * 3 * storage_width, dtype=torch.float32)
    original = (original.reshape(cache_depth, 2, 3, storage_width) + 1).to(dtype)
    expected_storage = original.clone()
    actual_storage = original.clone()
    expected = expected_storage[..., ::2] if noncontiguous else expected_storage
    actual = actual_storage[..., ::2] if noncontiguous else actual_storage
    for index, width in enumerate(widths):
        if width > 0:
            zero_from = max(0, width - pad_frames)
            if zero_from < width:
                expected[index][..., zero_from:] = 0.0

    result = _zero_padded_cnn_cache(actual, _cnn_cache_estimator(widths), pad_frames)

    assert result is None
    assert torch.equal(actual_storage, expected_storage)
    assert actual.dtype == dtype
    assert actual.stride() == expected.stride()


def test_zero_padded_cnn_cache_uniform_blocks_use_one_write():
    cache = torch.ones(16, 2, 3, 4)
    version = cache._version

    _zero_padded_cnn_cache(cache, _cnn_cache_estimator([4] * 16), 1)

    # CPU mutation count pins the mechanism without making a CUDA timing claim.
    assert cache._version - version == 1
    assert torch.equal(cache[..., :3], torch.ones(16, 2, 3, 3))
    assert torch.count_nonzero(cache[..., 3:]) == 0


# --- Cache-width side of the capture-shape explosion -----------------------


def _cache_align_width_disabled(width: int) -> int:
    return width


def _capture_shapes_for_requests(
    prompt_lengths: tuple[int, ...],
    n_chunks: int,
    cache_align,
) -> set[tuple[int, int]]:
    """Simulate the (chunk_width, cache_width) NPUGraph capture shapes.

    Mirrors the ``_decode_batch_once`` recurrence: the cache grows by the
    padded chunk width and trims to ``prompt_len + 100``. With a grid
    ``cache_align`` the trim target is aligned and the cache is padded once
    at setup time (no growth path), so each request lands directly on its
    (chunk bucket, cache bucket) shape.
    """
    shapes: set[tuple[int, int]] = set()
    for prompt_len in prompt_lengths:
        if cache_align is _cache_align_width_disabled:
            cap = prompt_len + 100
            width = prompt_len
            for index in range(n_chunks):
                mel = 7 if index == 0 else 50
                pad = _cfm_pad_frames(
                    mel_frames=mel,
                    offset=width,
                    noise_capacity=30000,
                    bucket_frames=16,
                    disabled=False,
                )
                shapes.add((mel + pad, width))
                width = min(width + mel + pad, cap)
        else:
            cap = cache_align(prompt_len + 100)
            for chunk_width in (16, 64):  # first chunk (7+9), steady (50+14)
                shapes.add((chunk_width, cap))
    return shapes


def test_current_trim_yields_32_capture_shapes_across_8_requests():
    """Pin the c8 NPUGraph-limit mechanism seen in the i70 serve log.

    Eight requests with distinct reference prompt lengths each produce four
    shapes (first chunk + two growth steps + steady state): 8 x 4 = 32, i.e.
    exactly the default max_graphs limit. This is the mechanism behind the
    "reached the 32-entry NPUGraph limit" warning at c8, after which every
    new shape degrades to eager execution.
    """
    prompts = (200, 240, 280, 320, 360, 400, 440, 480)
    shapes = _capture_shapes_for_requests(prompts, n_chunks=30, cache_align=_cache_align_width_disabled)
    assert len(shapes) == 32
    # No cross-request sharing: every request settles on its own steady shape.
    for prompt_len in prompts:
        assert (64, prompt_len + 100) in shapes


def test_cache_align_width_basics():
    assert _cache_align_width(300, 256) == 512
    assert _cache_align_width(512, 256) == 512
    assert _cache_align_width(513, 256) == 768
    assert _cache_align_width(300, 0) == 300
    assert _cache_align_width(300, 1) == 300
    assert _cache_align_width(0, 256) == 0
    assert _cache_align_width(-5, 256) == -5


def test_cache_bucketing_collapses_8_requests_onto_few_shapes():
    """The cache-bucket design: one shape per (chunk bucket, cache bucket).

    Setup pads the cache once to the aligned trim target, so the growth path
    disappears; requests with nearby prompt lengths share a cache bucket.
    """
    prompts = (200, 240, 280, 320, 360, 400, 440, 480)
    shapes = _capture_shapes_for_requests(
        prompts,
        n_chunks=30,
        cache_align=lambda width: _cache_align_width(width, 256),
    )
    # 2 cache buckets (512, 768) x 2 chunk buckets (16, 64).
    assert len(shapes) == 4, sorted(shapes)
    assert len(shapes) < 32
