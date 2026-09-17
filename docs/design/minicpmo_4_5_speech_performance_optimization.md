# Speech Generation on vLLM-Omni: Performance Optimizations for MiniCPM-o 4.5 (Ascend NPU)

## Summary

vLLM-Omni supports end-to-end serving for **MiniCPM-o 4.5**, a speech-generating omni model whose pipeline follows the same multi-stage design as Qwen3-Omni:

- **Thinker**: multimodal understanding and text generation
- **Talker (AR decoder)**: auto-regressively generates codec tokens
- **Code2Wav**: decodes codec tokens into waveform audio (CFM/DiT estimator + HiFT vocoder)

This document describes the six highest-impact optimizations that have been implemented and measured on Ascend 910B3 NPU, ordered by their dependency chain. Stacked together they deliver:

| Metric | Concurrency | Baseline | Final | Improvement |
| --- | --- | --- | --- | --- |
| RTF | 32 req / c=1 | 0.940 | **0.479** | **−49.0%** |
| RTF | 64 req / c=4 | 2.744 | **1.206** | **−56.0%** |
| RTF | 128 req / c=8 | 5.843 | **2.246** | **−61.6%** |
| TTFT | 32/1 | 479 ms | **127 ms** | **−73.5%** |
| TTFT | 64/4 | 531 ms | **156 ms** | **−70.6%** |
| TTFT | 128/8 | 533 ms | **168 ms** | **−68.5%** |
| audio TTFP | 32/1 | 2390 ms | **1239 ms** | **−48.2%** |

Accuracy is unchanged: seed-tts WER **1.45%** (baseline 1.58%), Daily-Omni full set **77.78%** (within CI nightly range 77.6–78.2%).

The optimizations fall into three lines of attack:

1. **CUDA-graph cache governance** (PR #7416): stop the graph cache from exploding, so graphs can actually be captured and replayed.
2. **Full-graph decode replay** (FULL_AND_PIECEWISE + FULL_DECODE_ONLY + fixed-KV backend): make every decode step a graph replay instead of per-operator dispatch.
3. **Host-overhead amortization & speculative decoding** (K8 + ngram draft): pay the per-step host cost once for K frames, and draft likely tokens ahead.

Benchmark environment:

| | Value |
| --- | --- |
| **Hardware** | Ascend 910B3 (single card, shared container) |
| **Model** | MiniCPM-o 4.5 (seed-tts/en benchmark) |
| **vllm-omni** | baseline `1b6cd282` → perf-opt branch |
| **vllm-ascend** | 0.19.1rc2.dev2045 |

---

## 1. CFM Graph-Cache Governance (PR #7416)

**Impact: RTF −24.6% / −52.2% / −59.3% — the single largest item.**

### Why the graph cache was exploding

Two independent regressions stacked up in the same week:

1. **Reference-audio length leakage into the cache key.** Each request carries its own Seed-TTS reference audio; its length L0 (the attention-cache starting width) varies per request. The CFM CUDA-graph key is `(chunk length, cache width)`, so a stream of requests with different reference lengths produces a different key almost every time. A single benchmark run captured ~430 graphs and repeatedly flushed the whole cache — graphs that should have been replayed were re-captured instead.
2. **Uniform tail-chunk lengths.** Streaming chunks end at arbitrary lengths (1..128 frames, uniformly distributed). One captured graph per length means an unbounded graph library.

### The fix: normalize, bucket, mask

- **Reference-audio normalization** — resample to 24 kHz mono and zero-pad/truncate to a fixed window (`ref_audio_max_seconds`, default 6 s). All requests now share the same L0. Analogy: parcels are packed into standard-size boxes before shipping, so the warehouse only stocks a few sizes.
- **Mel-frame bucketing** — before each decode, round the CFM chunk up to the next multiple of 16 frames (`cfm_graph_bucket_frames: 16`), decode, then crop back to the true length. Chunk lengths collapse to a small set of buckets. Bucket size 16 beat 25 in A/B (25 pads more, runs slower).
- **Valid-length masking** — padded frames get a real `attn_mask`; cross-chunk caches are zeroed at valid-frame boundaries before becoming the next chunk's keys, so padding never pollutes the softmax denominator or the CNN/attention left-context. Without this, zero-padded columns would be re-inflated by the integration step `x = x + dt * velocity`.
- **Remove `active_stream_window`** — the bounded-window limiter serialized concurrent first packets; removing it cut TTFP by 23%.

### Results

Stage-by-stage on L20X (seed-tts, 128 requests, c=8):

| Stage | CFM captures | Flushes | Audio RTF | Audio TTFP |
| --- | --- | --- | --- | --- |
| origin/main (broken) | ~430 | 13 | 2.09 | 5.67 s |
| + normalization | 61 | 1 | 1.32 | 3.56 s |
| + bucketing | 15 | 0 | 1.19 | 3.30 s |
| + masking | 18 | 0 | 1.29 | 3.52 s |
| **+ remove active-stream window (final)** | 18 | 0 | **1.21** | **2.71 s** |

On 910B3 the same stack yields the three-tier RTF improvements listed in the summary (−24.6% / −52.2% / −59.3%). Quality is unchanged (WER 2.04% → 1.42% with masking; SIM/UTMOS bit-identical with unbucketed).

### Notes

- Normalization is shared code and benefits NPU directly; bucketing was CUDA-only at PR time and has been enabled on the NPU path in this branch.
- This item is a **prerequisite** for everything below: stage2 de-eager (#21) requires the graph library to stay bounded, otherwise the Code2Wav process falls back to eager for the rest of its life after the graph cap (32) is exceeded.

---

## 2. Restore Graph Mode on Decode Stages (FULL_AND_PIECEWISE + capture sizes)

**Impact: RTF −22.0% / −3.9% / +1.1% (tier 1 gains the most).**

### Why PIECEWISE silently loses the graph

With the default piecewise graph mode, every decode step is dispatched operator-by-operator, and **in-graph sampling silently stops working** — the sampler runs outside the graph, paying a host round-trip each step. Switching stage1 to `FULL_AND_PIECEWISE` with an explicit `capture_sizes` list restores full-graph replay.

### Sizing the graph library

`capture_sizes: [1, 2, 4, 8, 16, 24, 32]` pre-captures the shapes that actually occur. Analogy: a shoe store stocks every half size — every customer finds shoes in stock, nobody is sent to the factory (eager fallback) for a custom pair.

Two companions ride along:

- **Stage-level prefix caching** (`#19`): repeated prefixes hit the KV cache and shorten Thinker prefill. The top-level flag stays `false` (required for duplex safety); only the stage-level override is enabled, which benefits each request's own prefix only.
- **Stage2 de-eager** (`#21`): the CFM/HiFT graph wrappers only pay off when the engine itself is not eager; enabled once #7416 keeps the graph library bounded.

### Results

Tier-1 (single-request) RTF −22.0%; the small high-concurrency delta (+1.1%) reflects that batching already amortizes host overhead there.

---

## 3. Full-Step Decode Graph (FULL_DECODE_ONLY)

**Impact: −32.6% RTF on 910B3 (same-config A/B) — the largest single item on the NPU side.**

### Mechanism

The Talker decode stage is forced to `FULL_DECODE_ONLY` via a deploy-path-scoped patch: the entire decode step (including sampling) is captured as one graph and replayed, instead of piecewise dispatch. Analogy: a movie set keeps the scenery pinned in place and only swaps the actors' props — the shape and address of every parameter stays fixed, so replay is valid.

### Results and overlap notes

- On a pure-PIECEWISE baseline, entry_21 measured −25~33%; on this branch #2 had already restored full-graph decode for stage1, so the incremental gain concentrates on stage0 (`PIECEWISE → FULL_DECODE_ONLY`) and on configurations where #2 does not fully cover decode. Same-config A/B on 910B3: **−32.6%**.
- Logging confirms stage0 `PIECEWISE→FULL_DECODE_ONLY`, stage1 `FULL_AND_PIECEWISE→FULL_DECODE_ONLY`.

---

## 4. Fixed-KV Decode Backend with Capacity Buckets (T2)

**Impact: RTF −4.4~5.5% at low concurrency (32/1); neutral at c=4/c=8.**

### Mechanism

Full-step graphs (items 2–3) require **every parameter to keep the same address across steps** — the scenery must stay pinned. The default paged-attention KV backend moves blocks around, which breaks that invariant. The fixed-KV backend instead:

1. Pre-allocates KV **capacity buckets** sized by concurrency tier, and captures one graph per bucket;
2. Within a step, only updates `seq_lens` and write slots — no block-table reshuffling;
3. Falls back to the generic path when a request outgrows its bucket.

Analogy: reserve fixed parking bays for each bus size; the bus only updates its odometer, it never re-parks.

### Porting notes (910B vs 910C)

Three obstacles surfaced during the port, all resolved in code without changing the runtime image:

- 910C-specific features not present on 910B (guarded off);
- entry-tree self-contained modules (the A14 fused-sampler op payload was ported as a dependency, `64df891d`);
- a deprecated vllm-ascend API signature (`update_graph_params` dropped `num_dcp_pcp_tokens` after PCP removal in MRV1).

### Results

910B3 same-config A/B: 32/1 **−4.4~5.5%**, higher concurrencies neutral (host overhead there is already amortized by batching).

---

## 5. K8: Runner-Local K-Step Talker Decode

**Impact: −28%-order RTF reported by the original team (entry_18); local re-measurement pending.**

### Mechanism

The Talker is a ~190M model whose single decode step takes ~4.4 ms — and **~75% of that is host overhead** (scheduler bookkeeping, tensor prep, kernel launch), not compute. K8 amortizes that host cost across K frames:

- The runner executes a **K-step local loop** per scheduled step (K=8 default, `talker_local_decode_steps`);
- The scheduler **widens token allocation by K** and hands the runner a lookahead window (`num_lookahead_tokens = K−1`, infrastructure from PR #727);
- K graph replays per step each cost ~0.83 ms on device + ~8 µs host, versus K full host round-trips.

Analogy: a waiter used to walk back to the counter to sign for every single dish; now one tray carries eight dishes per trip. Or: milk the cow into a bucket once, not once per cup.

### Correctness

Bit-for-bit equivalence with single-step decoding is guaranteed by construction:

- a **causal mask** ensures row *k* never reads row *k+1*'s (not-yet-valid) data;
- a per-frame **stop check** lets any frame terminate the sequence immediately, exactly as single-step decoding would.

### Dependencies and caveats

- Requires item 2: the K-wide shapes must be covered by `capture_sizes`, otherwise the K-step loop silently falls back to eager.
- At concurrency > 1, draft tokens advance and retreat as a whole batch (a known pitfall from the original team's measurements).
- Port note: the entry_18 `mecha` prebuild subsystem is not ported to 910B and stays disabled by default (`OMNI_TALKER_MECHA=0`); K8's main path does not depend on it. Enabling K8 also requires stage1 `async_scheduling: false` (its gate demands sync scheduling) and `decode_mode()` — not the composite `FULL_DECODE_ONLY` — as the local-window runtime mode.

---

## 6. Speculative Decoding: ngram Draft + EOS-in-Draft (#14 + #25)

**Impact: TTFT 531 → 165 ms (−69%) from #14; a further 141 → 127 ms (−10%) from #25.**

### Why ngram speculation works unusually well for TTS

Stage0 (Thinker) output largely **repeats the prompt text** (the user's spoken question / the text being read aloud). An ngram draft model built from the prompt therefore has a high hit rate — like taking dictation from a textbook you already hold: write the draft ahead, then verify. Config: `speculative_config` with 1 draft token / 1 verify step, k=15; requires `async_scheduling: false`.

Two porting pitfalls worth recording:

- `speculative_config` needed a whitelist registration in the NPU platform config;
- the NPU draft call site has a **different argument order** from upstream GPU (`propose_draft_token_ids` takes `sampled_token_ids` first); getting the signature wrong makes speculation silently degrade to eager.

### EOS-in-draft (#25)

Replies always end with `<|tts_eos|><|im_end|>`, but that pair never appears in the prompt — so the ngram draft is *guaranteed* to miss at the final tokens, forcing 1–2 eager single-token steps (~37.7 ms) per request. The fix rewrites the tail draft: on hitting `151645`, emit `[151704, 151645]` padded to the 15-wide draft; after `151704`, substitute a full-width `151645` draft. Analogy: the closing bow is in the script, so the actor doesn't improvise it at the end.

### Results

| Metric | Baseline | + ngram | + EOS-in-draft |
| --- | --- | --- | --- |
| TTFT (32/1) | 479 ms | 165 ms | **127 ms** |
| RTF (32/1) | 0.940 | −11.8% | ≈ flat (0.480 → 0.479) |

RTF is neutral; the win is entirely first-token latency.

---

## 7. Marginal Items (Brief)

Sections 1–6 cover the optimization chain. This section briefly records everything else so the reader can decide which items deserve a full write-up later.

### 7.1 Upstream optimizations already inside the baseline

main @ `1b6cd282` already ships the following upstream optimizations. They are part of every number in this document's baseline (i.e., there is no "off" starting point for them) and are not counted toward this round's gains:

| # | Item | Upstream | One-line note |
| --- | --- | --- | --- |
| 1 | Stage0 stop-token early termination | PR #3907 | injects 151704/151645, saves 1–2 tail steps |
| 2 | batched_token2wav explicit state batching | #5228 + #6021/#6346/#6397/#6529 | ragged cache / SDPA / RelPos budgeting |
| 3 | Runtime reference-audio registration cache | #5380 | keyed incl. sample_rate; its L0-explosion root cause is what #7416 normalization fixes |
| 4 | Prompt feature cache | #5228 | `prepare_prompt` |
| 5 | CFM graph capture (DiT estimator) | #6082 | `enable_cfm_graph` |
| 6 | HiFT graph capture | #5869 | `enable_hift_graph` |
| 7 | `num_lookahead_tokens` scheduling infra | #727 | prerequisite of K8 (item 5) |
| 8 | Async Omni output stream | #6529 | mutually exclusive with ngram; off in this stack (see Trade-offs) |
| 9 | head_code weight_norm materialize (load time) | #5228 | Talker head only |
| 10 | Attention workspace warm-up | #3773 | per capture size |
| 11 | Token2wav fp32 build context | #5228 | correctness infra |
| 27 | STFT window / harmonic coefficients resident on device | — | already resident in main; entry_21's copy was actually older — no port needed |

(#5228 is the ancestor of 5 of these.)

### 7.2 Small-yield items landed this round

- **#17 HiFT weight_norm folding** (`code2wav.py:110-119`) — `weight_norm` keeps weights factorized as `g·v/‖v‖` at inference, recomputing constants on every forward. After loading, all 82 HiFT/flow modules are folded into plain weights — mathematically identical, bit-identical output (1.52× elementwise). Analogy: a bakery pre-mixes its fixed recipe once instead of every batch. **RTF −1.6% / −5.4% / −1.1%, audio TTFP −16.7%** — the largest of the marginal items.
- **#16 Full-pipeline startup warm-up** (`09045535`) — right after startup, fire ~20 synthetic requests (13–33 chars, matching eval length distribution) through the whole chain so lazy graph capture and workspace allocation happen before the first real customer (heat the wok before the restaurant opens). Benchmark numbers unchanged — the benchmark's own 2 warmup requests already cover the first trip; this is a **production first-request** optimization, kept.
- **N1 CPU pinning / isolation** (`a72f1f46`) — NUMA first-touch core pinning before engine start; the original team reported −16.4% and variance tightening ±0.02→±0.005. On this idle shared container the gain is not visible; kept for production anti-contention.
- **N5 / T6 `decode_prep_fast`** (`9ded6938`) — decode-step input reuse via a `_prepare_inputs` override: across 118 decode steps only 4 tensors actually change, so reuse the diff instead of rebuilding (a form filled in once, only the date updated). Logs `fast decode prep engaged` when gates pass, falls back to the generic path otherwise. No measurable delta on this box; kept as no-regression.
- **N2 A14 fused-sampler payload** (`64df891d`, ported alongside T2) — 53 files / 5.5k lines; `npu_ops_module()` returns True on 910B. The full chain fuses ~50 sampling kernels into 1 (Ascend C). **Full enablement and verification still pending** — counted in Remaining headroom, not in the stacked numbers.

### 7.3 Not attempted / not applicable

| Item | One-line reason |
| --- | --- |
| #18 TJS trajectory skip (CFM analytic tail) | ≈0.2% upside; post-break cache backfill is risky on the ragged CFM cache |
| #22 TensorRT DiT stepper | CUDA-only; N/A on NPU |
| #24 T44 zero-copy stage handoff | requires a connector protocol change (structural); −0.4%-order upside |
| #26 First-chunk conditional result cache | seed-tts uses a distinct reference per request; hit rate too low |
| #28 Incremental frequency penalty (scatter_add) | net-new code, small upside |
| #30 hidden-offset / K8 slot fix | form not present locally; daily-omni full-set accuracy (77.78%, bit-identical input tokens) shows no manifestation |
| N3 Triton flash-decode / N4 KV affine addressing | operator-level R&D, each ≤~3% |

Reverted items (#13 step reduction, #23 chunk sizing) and the async-vs-ngram exclusivity are detailed in Known Trade-offs below.

---

## Stacked Results and Accuracy

Per-item contribution (relative to the previous stack step, seed-tts/en, 910B3):

| Item | RTF change (32/1 / 64/4 / 128/8) | Key gain |
| --- | --- | --- |
| #7416 cache governance | −24.6% / −52.2% / −59.3% | normalization fixes L0 blow-up; bucketing fixes graph-library explosion |
| #15+19+21 graph restore | −22.0% / −3.9% / +1.1% | in-graph sampling restored + prefix cache + stage2 de-eager |
| #3 FULL_DECODE_ONLY | −32.6% (A/B) | whole-step graph replay on decode stages |
| #4 fixed-KV backend | −4.4~5.5% (32/1) | capacity buckets keep graph addresses stable |
| #5 K8 | (entry-reported −28%; local re-measure pending) | host overhead paid once per K frames |
| #6 ngram + EOS draft | TTFT −69% / −10% | first-token latency |

Accuracy gates (all no-regression): seed-tts WER **1.45%** vs 1.58% baseline (20-prompt official WER path, 20/20 success); Daily-Omni full set **77.78%** (931/1197, within CI nightly 77.6–78.2%; `total_input_tokens` bit-identical to baseline); Daily-Omni hard subset 0.30 vs 0.26–0.31 historical.

---

## Deployment Playbook

Key yaml knobs (per stage where applicable):

```yaml
# stage0 (Thinker)
speculative_config:            # item 6
  method: ngram
  num_speculative_tokens: 15
async_scheduling: false        # required by ngram gate AND K8 gate
prefix_caching: false          # keep top-level off (duplex safety)

# stage1 (Talker)
cudagraph_mode: FULL_AND_PIECEWISE   # item 2
capture_sizes: [1, 2, 4, 8, 16, 24, 32]
enforce_eager: false

# Code2Wav (stage2)
enable_cfm_graph: true
enable_hift_graph: true
enforce_eager: false           # item 2 companion (#21), safe after #7416

# connectors extra
cfm_graph_bucket_frames: 16    # item 1
ref_audio_max_seconds: 6       # item 1
# K8 (item 5) defaults are on: talker_local_decode_steps=8 (runner-side);
# kill switches: OMNI_TALKER_LOCAL_DECODE=0, OMNI_TALKER_LOCAL_STEPS=<K>
```

Reproduce the seed-tts benchmark:

```bash
# server
vllm-omni serve <minicpmo_4_5> --omni --port 8091   # yaml as above

# bench (three tiers)
vllm bench serve --dataset-name random --port 8091 \
  --backend openai-chat-omni --max-concurrency 1 \
  --num-prompts 32 --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf \
  --extra_body '{"modalities":["text","audio"]}'
# repeat with --max-concurrency 4 --num-prompts 64 and 8/128
```

## Known Trade-offs

- **Async Omni output vs ngram speculation are mutually exclusive** — `_should_use_async_omni_output()` returns False when `speculative_config` is set. This stack deliberately chooses ngram for the −69% TTFT; async output is therefore off, not forgotten.
- **Rejected after measurement** (kept here to prevent re-tries): CFM step reduction 10→3 (≤3% gain on NPU because graph replay, not compute, dominates; voice-quality risk), and small-first-chunk sizing (audio TTFP regressed +6~33% with no RTF gain).
- **Remaining headroom** is concentrated in kernel-level work (fused sampler full enablement, Triton flash-decode, KV affine addressing) and in re-measuring K8 at all three tiers; each remaining item is individually ≤~3%.
