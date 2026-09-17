# vLLM-Omni Feature Design Doc: MiniCPM-o 4.5 Speech-Generation Performance Optimization Pack (Ascend NPU)

> **Notes:** Follows the vLLM-Omni Feature Design Doc template. The standard skeleton (1 Overview / 2 Design / 3 Test cases) is kept; **2.2 Optimization Item Designs** and **2.3 Marginal Items** are added under Design because this is an optimization *pack*, not a single feature. All code references are verified against the perf-opt branch of this repository.

## 1 Overview

This design specifies six stacked optimizations for MiniCPM-o 4.5 speech generation on vLLM-Omni (Ascend 910B3) that together reduce end-to-end RTF by **49% / 56% / 62%** (concurrency 1 / 4 / 8) and TTFT by **~70%**, with no accuracy regression.

### 1.1 Motivation

MiniCPM-o 4.5 speech generation runs a three-stage pipeline — Thinker (multimodal understanding, text AR), Talker (AR codec-token decoder, ~190M), Code2Wav (CFM/DiT estimator + HiFT vocoder). Serving it on NPU exposes three compounding gaps:

1. **The CFM graph cache never converges.** Each request's reference audio has a different length, which leaks into the graph key `(chunk length, cache width)`; streaming tail chunks land on arbitrary lengths (1..128 frames, uniform). One benchmark run captured ~430 graphs and flushed the cache 13 times — graphs that should have been replayed were re-captured, making stage2 de-eager and full-graph decode modes unusable.
2. **Host overhead dominates the small Talker.** A Talker decode step costs ~4.4 ms of which ~75% is host work (scheduling, tensor prep, kernel launch), not device compute. Paying that per frame caps RTF regardless of device throughput.
3. **Speculation misses exactly where it matters.** The Thinker's answer is a verbatim prompt copy plus the terminator pair `<|tts_eos|><|im_end|>`. The n-gram drafter copies out of the prompt, which never contains `<|tts_eos|>` — every request pays 1–2 eager single-token steps (~37 ms each) at the tail.

These map to serving SLOs: TTFT/audio-TTFP for conversational latency, RTF for streaming sustainability. The pack consolidates three internal optimization entries (entry_02/18/21) and upstream PR #7416, re-measured on 910B3 (baseline vllm-omni main @ `1b6cd282`, vllm-ascend `0.19.1rc2.dev2045`).

### 1.2 Target

#### Feature

- **CFM graph-cache governance** (PR #7416): reference-audio normalization to a fixed window, mel-frame bucketing to a 16-frame grid, valid-length masking, removal of the `active_stream_window` limiter — CFM graph library bounded to ~18 captures, zero flushes.
- **Graph-mode restore**: `FULL_AND_PIECEWISE` + explicit `capture_sizes` on stage1 (restores in-graph sampling), stage-level prefix-cache override, stage2 de-eager.
- **Full-step decode graph (T1)**: `FULL_DECODE_ONLY` — whole decode step (incl. sampling) captured and replayed.
- **Fixed-KV decode backend (T2)**: capacity-bucketed KV with constant host-side graph arguments; graceful fallback outside buckets.
- **K8 runner-local Talker decode**: host overhead amortized across K=8 frames inside the runner loop, gated and bit-exact.
- **Speculative decoding**: ngram draft (1/1, k=15) on stage0 + terminator-pair tail draft (D7).

**Out of Scope:** operator-level R&D (Triton flash-decode, KV affine addressing, full A14 sampler chain; each ≤~3%); TensorRT stepper (CUDA-only); stage-handoff protocol redesign (T44); incremental frequency penalty; first-chunk conditional cache; runtime-dynamic switching of graph modes / speculation (restart-time configuration only).

#### Accuracy

- K8 and weight_norm folding are **bit-exact by construction** (causal mask + per-frame stop; mathematical identity), verified bitwise.
- #7416 masking keeps padded frames out of softmax denominators; cross-chunk caches are zeroed at valid boundaries — seed-tts WER 2.04%→1.42%, SIM/UTMOS bit-identical with unbucketed decode.
- End-to-end gates: seed-tts WER **1.45%** (baseline 1.58%); Daily-Omni full set **77.78%** (931/1197, within CI-nightly range; `total_input_tokens` bit-identical to baseline).

#### Performance

910B3, seed-tts/en, same machine/card/params; tiers 32 req/c=1, 64/c=4, 128/c=8:

| Metric | Baseline | Final | Δ |
| --- | --- | --- | --- |
| RTF (32/1) | 0.940 | **0.479** | −49.0% |
| RTF (64/4) | 2.744 | **1.206** | −56.0% |
| RTF (128/8) | 5.843 | **2.246** | −61.6% |
| TTFT (32/1) | 479 ms | **127 ms** | −73.5% |
| TTFT (64/4) | 531 ms | **156 ms** | −70.6% |
| audio TTFP (32/1) | 2390 ms | **1239 ms** | −48.2% |

**Trade-offs:** async Omni output and ngram speculation are mutually exclusive (`_should_use_async_omni_output()`); this pack chooses ngram — async output is deliberately off. K8 requires sync scheduling on stage1 and rejects non-uniform batches (generic-path fallback: no regression, no gain). Mel bucketing pads up to 16 frames (A/B-validated: 16 beats 25).

---

## 2 Design

### 2.1 Overview of Design

Three principles, applied at different stages:

1. **Make graph keys enumerable** (governance first) — normalize/bucket inputs so the graph library is bounded; only then can graph modes engage.
2. **Make every decode step a replay** — full-step graphs over a fixed-KV backend whose host-side arguments never change.
3. **Pay host cost once, draft ahead** — K-step runner loops amortize scheduling; speculation covers the predictable prompt copy and its terminator tail.

```
request ──> Stage0 Thinker ──────> Stage1 Talker ──────> Stage2 Code2Wav ──> audio
             text AR                codec-token AR         CFM DiT + HiFT
               │                      │                      │
   ngram draft (k=15)         K8 local K-step loop      ref normalization
   tail-draft rewrite         FULL_DECODE_ONLY          mel bucketing (16f)
   FULL_DECODE_ONLY +         fixed-KV capacity         valid-length masking
   fixed-KV, prefix cache     buckets                   graph wrappers + de-eager
```

Dependency chain (enablement order): **#7416 governance → graph-mode restore (T2 → T1) → K8 → speculation**. `FULL_DECODE_ONLY` is a *composite* (decode=FULL, prefill=NONE) resolved via `decode_mode()` to the concrete runtime FULL mode; K8 requires T1's bucket-1 replay path plus the `num_lookahead_tokens` widening from #727.

### 2.2 Optimization Item Designs

#### 2.2.1 CFM Graph-Cache Governance (PR #7416)

**Problem.** Reference-audio length varies per request and enters the CFM graph key via the attention-cache starting width L0; tail chunks land on arbitrary lengths. The key space is effectively unbounded.

**Mechanism.**
- *Normalization*: resample to 24 kHz mono (downmix guard ≤8 channels), zero-pad/truncate to a fixed window (`ref_audio_max_seconds`, default 6 s). All requests share one L0.
- *Mel bucketing*: round each CFM chunk up to the next multiple of 16 frames (`cfm_graph_bucket_frames`), decode, crop back. Chunks collapse to few buckets; frames below 1/3 of the bucket fall back to eager.
- *Masking*: padded frames get a real `attn_mask`; cross-chunk caches are zeroed at valid-frame boundaries before becoming the next chunk's keys — otherwise the integration step `x = x + dt·v` re-inflates zero-padded columns.
- *Remove `active_stream_window`*: the bounded-window limiter serialized concurrent first packets.

**Code touchpoints.**
- `vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_code2wav.py`: `_normalize_reference(ref_audio, sample_rate_hz, ...)` (L96), `_REF_MAX_SECONDS = 6.0` (L62), `_normalized_default_prompt` (folds the shipped default prompt onto the request-reference grid), `_bucket_key(item)` (L624), `extra.get("ref_audio_max_seconds", ...)` (L273)
- `vllm_omni/model_executor/models/minicpmo_4_5/batched_token2wav.py`: mask application, cross-chunk cache boundary zeroing, HiFT fallback
- `vllm_omni/model_executor/models/minicpmo_4_5/cuda_graph_wrapper.py`: graph keys over bucketed shapes

**Configuration.** Connector `extra`: `enable_cfm_graph: true`, `enable_hift_graph: true`, `cfm_max_graphs: 32`, `cfm_graph_bucket_frames: 16`, `ref_audio_max_seconds: 6.0`.

**Result.** L20X stage-wise: captures 430→18, flushes 13→0, audio RTF 2.09→1.21, audio TTFP 5.67 s→2.71 s. On 910B3: RTF **−24.6% / −52.2% / −59.3%**. Prerequisite for everything below.

#### 2.2.2 Restore Graph Mode on Decode Stages

**Problem.** With default PIECEWISE, every decode step is dispatched operator-by-operator and **in-graph sampling silently stops working**; stage2 graph wrappers sit inside an eager engine; repeated prefixes re-prefill.

**Mechanism.** Stage1 `cudagraph_mode: FULL_AND_PIECEWISE` + explicit `cudagraph_capture_sizes: [1,2,4,8,16,24,32]` pre-capture the shapes that occur (a shoe store stocking every half size, so nobody is sent to the eager "factory"). Stage-level prefix caching is enabled while the top-level flag stays `false` (duplex safety unchanged) — only each request's own prefix hits. Stage2 drops `enforce_eager` once 2.2.1 keeps the library under `cfm_max_graphs`.

**Code touchpoints.**
- `vllm_omni/deploy/minicpmo_4_5.yaml` (npu platform block): stage1 `compilation_config.cudagraph_mode: FULL_AND_PIECEWISE`, `cudagraph_capture_sizes`, `max_cudagraph_capture_size: 32`; stage2 `enforce_eager: false`, `additional_config.code2wav_enable_npu_graph: true`, `code2wav_max_npu_graphs: 32`
- `vllm_omni/config/stage_config.py`: graph-mode validation
- Stage-level prefix-cache override in the stage-config path (top-level `enable_prefix_caching: false` untouched)

**Result.** RTF −22.0% / −3.9% / +1.1% (tier 1 gains most; batching already amortizes host overhead at higher tiers).

#### 2.2.3 Full-Step Decode Graph (T1)

**Problem.** Even with 2.2.2, some decode stages still run piecewise; a whole-step graph is only valid if every parameter keeps a stable address across steps.

**Mechanism.** A deploy-path-scoped patch forces `FULL_DECODE_ONLY` on the decode stages (movie set keeps the scenery pinned; only the actors' props change). `FULL_DECODE_ONLY` is a *composite* (decode=FULL, prefill=NONE) and is not itself a valid runtime mode — the runner resolves it via `CUDAGraphMode.FULL_DECODE_ONLY.decode_mode()` into the concrete FULL mode the forward context accepts.

**Code touchpoints.**
- `vllm_omni/platforms/npu/worker/npu_ar_model_runner.py`: mode resolution (`_talker_local_mode`; commits `e9facfa3`/`21b9f6e7`)
- `vllm_omni/config/stage_config.py`: mode whitelist/validation

**Result.** Same-config A/B on 910B3: **−32.6%** — the largest single item on the NPU side. Logs confirm stage0 `PIECEWISE→FULL_DECODE_ONLY`, stage1 `FULL_AND_PIECEWISE→FULL_DECODE_ONLY`.

#### 2.2.4 Fixed-KV Decode Backend with Capacity Buckets (T2)

**Problem.** Full-step graphs require every host-side argument to stay constant across steps; the paged KV backend moves blocks around, and attention takes KV length as a growing host argument (the rebind is ~38% of the small Talker decoder's busy time).

**Mechanism.** Pre-allocate KV **capacity buckets** from `max_model_len`/`block_size`; capture one graph per bucket; within a step only `seq_lens` and write slots change — the op declares full KV capacity and the live length arrives through `pse_shift`, a device tensor the graph refreshes itself. Out-of-bucket requests, chunked/batched prefill, and non-uniform queries refuse capture and fall back. Enabled per stage by `max_model_len` scoping (Talker 4096 engages, Thinker 32768 does not).

**Code touchpoints.**
- `vllm_omni/platforms/npu/attention/fixed_kv_decode.py`: `capacity_for(max_model_len, block_size)`, `buckets_for(capacity, block_size)`, `select_bucket(max_seq_len, buckets)`, `set_runtime_bucket(...)`, `current_capacity()`, `install_into_ascend_aclgraph(...)`
- `vllm_omni/platforms/npu/attention/fixed_kv_backend.py`: `OmniFixedKVMetadataBuilder` (picks the bucket in `build()`), `OmniFixedKVAttentionImpl` (`_fixed_kv_applies`, `_uniform_query_len`, `_fixed_kv_graph_fia(query, ...)`, `update_graph_params(update_stream, forward_context, ...)` — deprecated `num_dcp_pcp_tokens` removed; `enable_hamming_sparse` defaulted `False` for newer vllm-ascend builds), fused `TalkerDecodeAttention` op path
- `vllm_omni/platforms/npu/platform.py`: `get_attn_backend_cls` installs the backend (kill switch `VLLM_OMNI_FIXED_KV_DECODE=0`)
- `vllm_omni/model_executor/models/minicpmo_4_5/talker_codec_sample.py`: sampler payload ported alongside

**Result.** 32/1 **−4.4~5.5%**; 64/4 and 128/8 neutral. The 910B port needed no image change: 910C feature guards, the entry-owned module payload (incl. A14 bindings), and one deprecated API signature were all resolved in code.

#### 2.2.5 K8: Runner-Local K-Step Talker Decode

**Problem.** ~75% of a Talker step is host overhead billed per frame; the scheduler/runner round-trip per 4.4 ms step cannot be hidden by the device.

**Mechanism.** The runner executes a **K-step local loop** (K=8 default) per scheduled step. The scheduler widens allocation by K (`num_lookahead_tokens = K−1`) so the KV window for frames k+1..K exists before they are decoded; a causal mask guarantees row *k* never reads row *k+1*'s not-yet-valid data, and a per-frame stop check terminates exactly as single-step decoding would — **bit-exact by construction**. (Analogy: one tray carries eight dishes per trip, instead of walking back to the counter per dish.)

**Code touchpoints.**
- `vllm_omni/platforms/npu/worker/npu_ar_model_runner.py`: `_init_talker_local_decode_config()`, `_talker_local_decode_eligible(...)` (gate: FULL_DECODE_ONLY bucket-1 replay, pure decode batch, no prefill chunk / spec-decode / encoder / grammar, batch fits captured buckets), `_talker_local_decode_loop(scheduler_output, *, num_reqs, req_ids, ...)`, `_talker_local_tokens_pending` hand-off
- `vllm_omni/core/sched/omni_ar_scheduler.py`: `num_lookahead_tokens = K−1` widening (Talker stage only), `OMNI_TALKER_SCHED_K` override

**Configuration.** Connector `extra`: `talker_local_decode_steps` (K), `talker_local_decode_stage_id`, `talker_local_cpu_slot_mapping`. Env kill switches: `OMNI_TALKER_LOCAL_DECODE=0`, `OMNI_TALKER_LOCAL_STEPS=<K>`. Stage1 must set `async_scheduling: false` (the local window is not async-placeholder aware).

**Result.** The original team reported −28%-order RTF (entry_18); the 910B port (`71a2b332`, 2 files, +1004 lines) is landed and **awaiting formal local benchmark confirmation**. At concurrency > 1 draft tokens advance/retreat as a whole batch (known caveat).

#### 2.2.6 Speculative Decoding: ngram Draft + Terminator Tail Draft

**Problem.** Stage0 output is a verbatim prompt copy — ngram drafting has a high hit rate — but the answer always ends with `<|tts_eos|><|im_end|>` and the prompt never contains `<|tts_eos|>`, so the draft is structurally wrong at the first terminator of every request: 1–2 eager 1-token steps at ~37 ms each (no captured graph for a no-draft step; the uniform verify shape is 16 query tokens at ~7 ms host).

**Mechanism.** Stage0 runs `speculative_config` ngram 1/1 k=15. The tail draft (`stage0_tail_draft.rewrite`) closes the gap with two rewrites, both exact under the rejection sampler (a draft token is only emitted if the model's argmax equals it, so emitted text cannot change):
1. a draft running past the copy into `<|im_end|>` is rewritten at that point to `[<|tts_eos|>, <|im_end|>]` and padded back to full width (padding = repeated `<|im_end|>`; tokens past an accepted stop are dropped, mismatched pads are rejected — padding can never reach the output);
2. an *empty* draft right after the model emitted `<|tts_eos|>` becomes a full-width `<|im_end|>` draft — the terminator rides a 16-token graph replay instead of an eager single-token step.

`VLLM_OMNI_MINICPMO_STAGE0_TAIL_DRAFT=off` restores stock drafts; `=strict` keeps only rewrite 2 (rewrite 1 guesses the future and is off-by-default-risky on free-form generation).

**Code touchpoints.**
- `vllm_omni/platforms/npu/worker/stage0_tail_draft.py`: `enabled()`, `strict()`, `applies(runner)` (ngram && k==15 && enabled), `rewrite(drafts, sampled_token_ids, k)`
- `vllm_omni/platforms/npu/worker/npu_ar_model_runner.py`: `propose_draft_token_ids(self, sampled_token_ids, *args, **kwargs)` — **NPU call-site order differs from upstream** (sampled_token_ids first); a wrong signature silently degrades speculation to eager
- `vllm_omni/config/omni_config.py`: allow `speculative_config` as stage engine override (whitelist, commit `ae3758f6`)
- Deploy yaml stage0: `async_scheduling: false` (VllmConfig validation incompatibility), `speculative_config: {method: ngram, num_speculative_tokens: 15, prompt_lookup_min: 1, prompt_lookup_max: 1}`

**Result.** TTFT 531→165 ms (**−69%**) from ngram; 141→127 ms (**−10%**) from the tail draft; RTF neutral.

### 2.3 Marginal Items (Brief)

**Already in baseline** (upstream main @ `1b6cd282`; not counted toward gains): #1 stop-token injection (#3907), #2 batched_token2wav (#5228+), #3 ref-audio registration cache (#5380; its L0-explosion root cause is what 2.2.1 fixes), #4 prompt feature cache, #5/#6 CFM/HiFT graph capture (#6082/#5869), #7 lookahead infra (#727, K8 prerequisite), #8 async output (off: ngram-exclusive), #9/#11 load-time materialize / fp32 context (#5228), #10 workspace warm-up (#3773), #27 STFT-resident constants (already resident).

**Landed this round, small yield:** #17 weight_norm load-time folding (`_fold_weight_norm_modules` in `step_audio2_token2wav.py`, `8a115ae7`; MiniCPM-o reuses `StepAudio2Token2WavCore` via `minicpmo_4_5_token2wav.py:25`; bit-identical; RTF −1.6/−5.4/−1.1%, audio TTFP −16.7%) · #16 full-chain prewarm (`async_omni_engine.py`, +97 lines, ~20 background synthetic requests; production first-request optimization, benchmark-neutral) · N1 CPU pinning (`utils/cpu_isolation.py`; idle-container-neutral, kept for anti-contention) · N5 `decode_prep_fast` (`worker/decode_prep_fast.py`; gate-gated input reuse, no delta here) · N2 A14 fused-sampler payload (ported with T2, `npu_ops_module()` True on 910B, full enablement pending).

**Not attempted:** #18 TJS (≈0.2%, ragged-cache backfill risk) · #22 TRT (CUDA-only) · #24 T44 (protocol change) · #26 first-chunk cache (per-request refs → low hit rate) · #28 incremental frequency penalty (net-new, small) · #30 hidden-offset (not present locally; daily-omni accuracy clean) · N3/N4 (operator R&D). **Reverted after measurement:** #13 CFM step reduction (≤3% on NPU, voice-quality risk) · #23 chunk resizing (audio TTFP +6~33%).

### 2.4 Use Cases

**1. Low-latency conversational TTS (single stream, c=1)**
- Scenario: a voice assistant answers within ~130 ms first-token / ~1.2 s first-audio on one 910B3.
- Configuration: the full pack (yaml in 2.5): ngram + tail draft + K8 + FULL_DECODE_ONLY + fixed-KV.
- Benefit: TTFT 479→127 ms, audio TTFP 2390→1239 ms, RTF 0.940→0.479.

**2. High-throughput batch TTS (c=8, 128 concurrent)**
- Scenario: offline narration for many documents; RTF / stream sustainability dominate.
- Configuration: same yaml; bucketed graphs + de-eager stage2 carry the load; K8 batch caveat applies.
- Benefit: RTF 5.843→2.246 (−61.6%); captures stay ~18 with zero flushes under mixed reference lengths.

**3. Full-duplex streaming assistant (duplex_session, 4 sessions)**
- Scenario: always-on duplex chat sharing the profile with `/v1/chat/completions`.
- Configuration: `session_mode: duplex`, `codec_chunk_frames: 25`, `codec_left_context_frames: 3`; masking keeps chunk boundaries clean; top-level prefix caching stays off for duplex safety.
- Benefit: removing `active_stream_window` un-serializes concurrent first packets (audio TTFP −23%); stage-level prefix cache shortens Thinker prefill without cross-request leakage.

### 2.5 API Design

#### Current Component Changes

| Component | Change | Why / Impact | Location |
| --- | --- | --- | --- |
| `models/minicpmo_4_5/minicpmo_4_5_code2wav.py` | ref normalization, bucketed work-item keys, configurable ref window | make CFM graph key enumerable; longer refs truncated (logged) | `_normalize_reference`, `_REF_MAX_SECONDS`, `_bucket_key`, `_normalized_default_prompt` |
| `models/minicpmo_4_5/batched_token2wav.py` | valid-length masking, cross-chunk cache zeroing, HiFT fallback | padded frames must not pollute softmax / CNN left-context | mask paths in CFM chunk decode |
| `platforms/npu/platform.py` | install fixed-KV backend by `max_model_len` scope | Talker attention uses capacity buckets; `VLLM_OMNI_FIXED_KV_DECODE=0` restores stock | `get_attn_backend_cls` |
| `platforms/npu/attention/fixed_kv_backend.py` / `fixed_kv_decode.py` | backend classes + bucket algebra + ACLGraphWrapper hook | decode steps capture with constant host args; `update_graph_params` aligned with current vllm-ascend | see New APIs |
| `platforms/npu/worker/npu_ar_model_runner.py` | fast decode prep, tail-draft hook, K8 config/eligibility/loop | gate-failing states take the generic path (bit-exact, only slower) | `_prepare_inputs`, `propose_draft_token_ids`, `_talker_local_*` |
| `core/sched/omni_ar_scheduler.py` | `num_lookahead_tokens = K−1` widening | pre-allocate KV windows for local-loop frames (without it WER 2.12%→1.00% regression at K=1) | scheduler init + schedule path |
| `config/omni_config.py` | allow `speculative_config` as stage engine override | ngram on stage0 via deploy yaml | whitelist (`ae3758f6`) |
| `engine/async_omni_engine.py` | full-chain prewarm | absorb lazy captures before first real request | background synthetic requests |
| `utils/cpu_isolation.py` | NUMA first-touch pinning | variance tightening; anti-contention | startup path |
| `deploy/minicpmo_4_5.yaml` | wire all knobs for the 3-stage single-card profile | reproducible deployment | cuda/npu platform blocks |

#### New APIs

`vllm_omni/platforms/npu/attention/fixed_kv_decode`
```
capacity_for(max_model_len, block_size) -> int | None
buckets_for(capacity, block_size) -> list[int]
select_bucket(max_seq_len, buckets) -> int | None
set_runtime_bucket(bucket) -> None
current_capacity() -> int | None
install_into_ascend_aclgraph(wrapper, ...) -> None
```

`vllm_omni/platforms/npu/worker/stage0_tail_draft`
```
enabled() -> bool                     # env gate (VLLM_OMNI_MINICPMO_STAGE0_TAIL_DRAFT)
strict() -> bool                      # verified-only mode
applies(runner) -> bool               # ngram && k==15 && enabled
rewrite(drafts: list[list[int]] | None,
        sampled_token_ids: list[list[int]], k: int) -> list[list[int]] | None
```

`vllm_omni/platforms/npu/worker/decode_prep_fast`
```
enabled() -> bool
DecodeInputCache.invalidate() -> None
try_fast_prepare(runner, scheduler_output, num_scheduled_tokens) -> inputs | None
note_generic(runner, scheduler_output, num_scheduled_tokens, result) -> None
```

Configuration surface (connector `extra` keys): `ref_audio_max_seconds` · `cfm_graph_bucket_frames` · `cfm_max_graphs` · `enable_cfm_graph` / `enable_hift_graph` · `talker_local_decode_steps` · `talker_local_decode_stage_id` · `talker_local_cpu_slot_mapping` · `code2wav_enable_npu_graph` · `code2wav_max_npu_graphs`. Env knobs: `VLLM_OMNI_FIXED_KV_DECODE=0` · `VLLM_OMNI_MINICPMO_STAGE0_TAIL_DRAFT=off|strict` · `OMNI_TALKER_LOCAL_DECODE=0` · `OMNI_TALKER_LOCAL_STEPS=<K>` · `OMNI_TALKER_SCHED_K`.

### 2.6 API Call Dependency

**Startup sequence**
1. Deploy yaml loads; stage configs validate graph modes (`stage_config.py`) and `speculative_config` override (`omni_config.py`).
2. `platform.get_attn_backend_cls` installs the fixed-KV backend for stages whose `max_model_len` fits the scope.
3. Code2Wav loads flow+HiFT, folds weight_norm, builds CFM/HiFT graph wrappers (buckets derived from `codec_chunk_frames`/`cfm_graph_bucket_frames`).
4. Scheduler sets `num_lookahead_tokens = K−1` for the Talker stage; runner reads K knobs (`_init_talker_local_decode_config`, env overrides win).
5. Engine fires ~20 prewarm requests through the whole chain (lazy captures absorbed).

**Per-request decode (Talker, K8 active)**
1. Scheduler schedules the step; runner checks `_talker_local_decode_eligible` — on reject, the generic path runs (correct, slower).
2. Eligible: `_talker_local_decode_loop` replays K FULL_DECODE_ONLY steps; per-frame stop may exit early; outputs surface via `_talker_local_tokens_pending`.
3. KV for frames k+1..K was pre-allocated via the widening; the causal mask keeps each row local.

**Stage0 speculative step**
1. `propose_draft_token_ids(sampled_token_ids, ...)` builds ngram drafts; `stage0_tail_draft.rewrite` applies the two tail rewrites and pads to k=15.
2. The 16-token verify graph replays; the rejection sampler guarantees emitted text is unchanged.
3. Error paths: ngram off → no rewrite (`applies` False); `strict` mode → only rewrite 2; empty-draft rule fires only after the model itself emitted `<|tts_eos|>`.

---

## 3 Test cases

### 3.1 Unit Test(UT) design

**Existing UT coverage** (all under `tests/model_executor/models/minicpmo_4_5/`):

| File | Covers |
| --- | --- |
| `test_reference_audio_normalization.py` | 24 kHz resample, downmix guard, 6 s window pad/truncate, cache-key stability |
| `test_cfm_graph_bucketing.py` | 16-frame bucket alignment, crop-back exactness, bucket keys |
| `test_cfm_graph_capture_gating.py` | capture gating incl. <⅓-bucket eager fallback |
| `test_code2wav_batching.py` | work-item batching, `_bucket_key` families |
| `test_audio_chunk_mask.py` | valid-length masks, cross-chunk cache boundary zeroing |
| `test_cuda_graph_wrapper.py` | graph key/replay wrapper semantics |
| `test_streaming_audio_cache.py` | streaming chunk cache correctness |
| `test_talker_batching.py`, `test_pipeline.py`, `test_llm2tts.py`, `duplex/` | Talker batching, pipeline, duplex paths |

**UT gaps and proposed new tests** (the NPU worker items currently have no UTs):

1. `tests/platforms/npu/attention/test_fixed_kv_buckets.py` — `test_select_bucket_boundaries()`
   - Purpose: bucket selection never returns a bucket smaller than `max_seq_len`; `None` above capacity.
   - Steps: `select_bucket` at exact bucket edge, edge±1, above capacity; `set_runtime_bucket`/`current_capacity` publish-read consistency.
   - Assertions: monotone selection; out-of-bucket → capture refusal path (`_fixed_kv_applies` False).
2. `tests/platforms/npu/worker/test_stage0_tail_draft.py` — `test_rewrite_rules()`
   - Steps: (a) draft containing `<|im_end|>` → tail rewritten to `[151704, 151645]`, padded to k; (b) empty draft after emitted `151704` → full-width `151645`; (c) strict mode suppresses rule (a); (d) env off → drafts unchanged; (e) padding tokens can never surface (mismatched pads rejected by construction).
   - Assertions: output-draft invariants per rule; no mutation of `sampled_token_ids`.
3. `tests/platforms/npu/worker/test_decode_prep_fast.py` — `test_cache_invalidation_and_gates()`
   - Steps: fast hit on identical consecutive decode; `invalidate()` after metadata-affecting changes; gate failure → generic path recorded via `note_generic`.
   - Assertions: fast path outputs identical to generic `_prepare_inputs` on a stub runner.
4. `tests/platforms/npu/worker/test_talker_local_decode.py` — `test_eligibility_matrix()`
   - Steps: eligible (FULL_DECODE_ONLY, pure decode, K fits buckets) vs rejected (spec decode on, prefill chunk present, mode NONE without S==K, oversized batch).
   - Assertions: rejection reasons tagged; bit-exactness contract documented (causal mask + per-frame stop).
5. `tests/config/test_omni_config_speculative_override.py` — stage-level `speculative_config` passes validation; top-level duplex flag untouched.
6. `tests/utils/test_cpu_isolation.py` — pinning applied pre-engine; failure falls back without aborting startup.

### 3.2 Smoke Test(ST) design

1. `test_minicpmo_perf_pack_serving()` — end-to-end three-tier serving smoke
   - Setup: model MiniCPM-o 4.5; 1× 910B3; `vllm_omni/deploy/minicpmo_4_5.yaml`; server `vllm-omni serve <model> --omni --port 8091`.
   - Steps: run the three-tier seed-tts bench (32/1, 64/4, 128/8) with `--backend openai-chat-omni --ignore-eos --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf --extra_body '{"modalities":["text","audio"]}'`.
   - Assertions: startup logs show `FULL_DECODE_ONLY` on stage0/1, `fast decode prep engaged` or clean fallback, CFM captures ≈18 with 0 flushes, no graph-cap fallback to eager; RTF/TTFT within expected bands (0.48 / 1.21 / 2.25 ±10%; TTFT ≤ 170 ms).
   - Timeout: 30 min (three tiers incl. warmup).
2. `test_minicpmo_wer_smoke()` — accuracy smoke
   - Setup: same server; official WER path (whisper-tiny.en, 20 prompts).
   - Assertions: WER ≤ 1.6%; 20/20 successes; median WER 0.
3. `test_daily_omni_accuracy()` — multimodal gate
   - Setup: `VLLM_DAILY_OMNI_QA_JSON` + `VLLM_DAILY_OMNI_VIDEO_DIR` set; client flags `--daily-omni-pack-mode minicpm-interleave`, `--output-len 512`, `--daily-extra-body-json '{"modalities":["text"],"chat_template_kwargs":{"enable_thinking":false}}'` (missing any one drops accuracy to ~31%).
   - Assertions: full-set accuracy within CI-nightly range (77.6–78.2%); `total_input_tokens` bit-identical to baseline (4349821).
   - Timeout: 2 h.
4. `test_duplex_session_smoke()` — duplex path
   - Setup: `session_mode: duplex`, 4 concurrent chat sessions.
   - Assertions: no cross-session cache pollution (masking), first packets of concurrent sessions do not serialize, outputs within WER gate.

**Environment notes** (repro pitfalls): run `populate_modelinfo_cache.py` on the target tree after `git checkout` (content-hash validated); do **not** set `ASCEND_RT_VISIBLE_DEVICES` for the accuracy framework (it uses physical IDs); exit-time `corrupted size vs. prev_size` is glibc noise — judge by artifact JSONs.
