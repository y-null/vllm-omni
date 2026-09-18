"""No-device regression tests for the MiniCPM-o Talker K-step feedback chain.

Layers pinned down here:

- ``make_omni_output`` must record each frame's codec sample in the request
  state (``last_code``) while the multi-frame loop owns sampling: the rows
  vLLM schedules for the next step carry placeholder continue ids, so
  without the record every frame after the first embeds a placeholder
  instead of the real previous frame (silent audio corruption).
- The decode preprocess must prefer that state record over input_ids.
- The scheduler-side K-frame guard must drop continuation drafts whenever
  prefill work is pending, so no step ever mixes prefill rows with
  multi-row decode spans (the mixed batch that crashed a 64/4 run).
- The multi-frame gate matrix: non-uniform decode spans stay blocked at
  ``applies()``; the runner raise remains the assertion of last resort for
  a combination the guard makes unschedulable.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    MiniCPMO45OmniTTSForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_TALKER_FRAMES_ENV = "VLLM_OMNI_MINICPMO_TALKER_FRAMES"
_NUM_AUDIO_TOKENS = 6562
_EOS_ID = _NUM_AUDIO_TOKENS - 1


def _make_talker(*, k_step_frames: int, scripted_samples: list[int]):
    """Bare talker instance: no config, no weights, deterministic sampler.

    ``emb_code`` is crafted so the placeholder fallback (row 0) is all
    zeros while real codec rows are not -- a sharp contrast for asserting
    which id got embedded.
    """
    model = MiniCPMO45OmniTTSForConditionalGeneration.__new__(
        MiniCPMO45OmniTTSForConditionalGeneration
    )
    nn.Module.__init__(model)
    model._k_step_frames = k_step_frames
    model.supports_multi_frame_decode = k_step_frames > 0
    model._codec_temperature = 0.0
    model._codec_min_tokens = 0
    model._codec_max_tokens = 4032
    model._num_audio_tokens = _NUM_AUDIO_TOKENS
    model._codec_eos_id = _EOS_ID
    model._request_audio_states = {}
    emb = nn.Embedding(_NUM_AUDIO_TOKENS, 4)
    with torch.no_grad():
        emb.weight.zero_()
        emb.weight[42] = 1.0
        emb.weight[43] = 2.0
    model.emb_code = nn.ModuleList([emb])
    queue = iter(scripted_samples)

    def _greedy(_hidden, _codes, _request_id, _step, _min_tokens, _max_tokens):
        return torch.tensor(next(queue), dtype=torch.long)

    model._sample_audio_code_greedy = _greedy
    return model


def _frame_call(model, hidden):
    return model.make_omni_output(
        hidden,
        model_intermediate_buffer=[{"request_id": "r1"}],
        request_token_spans=[(0, 1)],
        request_sample_eligible=[True],
    )


def test_kstep_frame_sample_is_recorded_for_next_frame():
    model = _make_talker(k_step_frames=8, scripted_samples=[42, 43, _EOS_ID])
    state = {"step": 0, "codes": torch.tensor([10, 11])}
    model._request_audio_states["r1"] = state
    hidden = torch.randn(1, 8)

    out0 = _frame_call(model, hidden)
    assert state["last_code"] == 42
    assert state["step"] == 1
    assert out0.multimodal_outputs["codes"]["audio"][0].reshape(-1).tolist() == [42]

    out1 = _frame_call(model, hidden)
    assert state["last_code"] == 43
    assert out1.multimodal_outputs["codes"]["audio"][0].reshape(-1).tolist() == [43]

    # The EOS frame terminates the request: the record keeps the last real
    # sample (there is no next frame to feed), and the delta goes empty.
    out2 = _frame_call(model, hidden)
    assert state["last_code"] == 43
    assert state["finished"] is True
    assert out2.multimodal_outputs["codes"]["audio"][0].numel() == 0


def test_legacy_single_frame_path_does_not_touch_last_code():
    model = _make_talker(k_step_frames=0, scripted_samples=[42])
    state = {"step": 0, "codes": torch.tensor([10, 11])}
    model._request_audio_states["r1"] = state

    _frame_call(model, torch.randn(1, 8))
    assert "last_code" not in state


def test_decode_preprocess_embeds_state_last_code():
    model = _make_talker(k_step_frames=8, scripted_samples=[])
    state = {"step": 3, "codes": torch.tensor([42]), "last_code": 42}
    model._request_audio_states["r1"] = state
    # The scheduled row carries the placeholder continue id (0), not the
    # real previous frame's sample.
    input_ids = torch.zeros(1, dtype=torch.long)

    _, embeds, out = model.preprocess(
        input_ids,
        None,
        request_id="r1",
        audio_state=state,
        _omni_is_prefill=False,
    )
    assert torch.equal(embeds, model.emb_code[0](torch.tensor([42])))
    # Row 0 (the placeholder fallback) is all zeros in this fixture, so a
    # non-zero embed proves the state record won.
    assert torch.count_nonzero(embeds) == embeds.numel()
    assert out["codes"]["audio"].reshape(-1).tolist() == [42]


def _make_scheduler(*, num_spec: int, waiting=(), running=()):
    from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

    sched = OmniARScheduler.__new__(OmniARScheduler)
    sched._omni_talker_kstep_cache = None
    sched.speculative_config = SimpleNamespace(method="ngram", num_speculative_tokens=num_spec)
    sched.waiting = list(waiting)
    sched.running = list(running)
    return sched


def _req(*, computed: int, prompt: int, spec: list[int]):
    return SimpleNamespace(
        num_computed_tokens=computed,
        prompt_token_ids=[0] * prompt,
        spec_token_ids=list(spec),
    )


def test_guard_drops_drafts_when_waiting_request_pending(monkeypatch):
    monkeypatch.setenv(_TALKER_FRAMES_ENV, "8")
    decode_req = _req(computed=100, prompt=100, spec=[0] * 7)
    sched = _make_scheduler(num_spec=7, waiting=[object()], running=[decode_req])

    sched._drop_talker_drafts_if_prefill_pending()
    assert decode_req.spec_token_ids == []


def test_guard_keeps_drafts_when_no_prefill_pending(monkeypatch):
    monkeypatch.setenv(_TALKER_FRAMES_ENV, "8")
    decode_req = _req(computed=100, prompt=100, spec=[0] * 7)
    sched = _make_scheduler(num_spec=7, waiting=[], running=[decode_req])

    sched._drop_talker_drafts_if_prefill_pending()
    assert decode_req.spec_token_ids == [0] * 7


def test_guard_drops_drafts_when_chunked_prefill_in_flight(monkeypatch):
    monkeypatch.setenv(_TALKER_FRAMES_ENV, "8")
    decoding_req = _req(computed=100, prompt=100, spec=[0] * 7)
    chunking_req = _req(computed=50, prompt=100, spec=[])
    sched = _make_scheduler(num_spec=7, waiting=[], running=[decoding_req, chunking_req])

    sched._drop_talker_drafts_if_prefill_pending()
    assert decoding_req.spec_token_ids == []


def test_guard_noop_for_text_stage_spec_config(monkeypatch):
    monkeypatch.setenv(_TALKER_FRAMES_ENV, "8")
    text_req = _req(computed=100, prompt=100, spec=[0] * 15)
    sched = _make_scheduler(num_spec=15, waiting=[object()], running=[text_req])

    sched._drop_talker_drafts_if_prefill_pending()
    assert text_req.spec_token_ids == [0] * 15


def test_multiframe_gate_matrix():
    from vllm_omni.platforms.npu.worker import talker_multiframe

    model = SimpleNamespace(supports_multi_frame_decode=True)
    state = {"step": 5}

    # Uniform K-row decode: the loop engages with K=8.
    uniform = {
        "request_token_spans": [(0, 8), (8, 16)],
        "model_intermediate_buffer": [
            {"audio_state": dict(state)},
            {"audio_state": dict(state)},
        ],
    }
    assert talker_multiframe.applies(model, uniform) == 8

    # Mixed decode spans (an 8-row drafted decode plus a 1-row decode):
    # blocked, and flagged as a multi-token decode step.
    mixed_decode = {
        "request_token_spans": [(0, 8), (8, 9)],
        "model_intermediate_buffer": [
            {"audio_state": dict(state)},
            {"audio_state": dict(state)},
        ],
    }
    assert talker_multiframe.applies(model, mixed_decode) == 0
    assert talker_multiframe.is_multi_token_decode(model, mixed_decode) is True

    # Prefill rows mixed with a drafted decode: blocked the same way.
    mixed_prefill = {
        "request_token_spans": [(0, 515), (515, 523)],
        "model_intermediate_buffer": [
            {"audio_state": dict(state), "_omni_is_prefill": True},
            {"audio_state": dict(state)},
        ],
    }
    assert talker_multiframe.applies(model, mixed_prefill) == 0
    assert talker_multiframe.is_multi_token_decode(model, mixed_prefill) is True
