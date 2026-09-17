# vLLM-Omni Feature Design Doc：MiniCPM-o 4.5 语音生成性能优化包（Ascend NPU）

> **说明：** 本文档遵循 vLLM-Omni Feature Design Doc 模板。保留标准骨架（1 Overview / 2 Design / 3 Test cases）；由于这是一组优化**包**而非单一特性，在 Design 下额外增加了 **2.2 Optimization Item Designs** 和 **2.3 Marginal Items**。所有代码引用均已在仓库 perf-opt 分支上核实。

## 1 Overview

本设计针对 vLLM-Omni（Ascend 910B3）上的 MiniCPM-o 4.5 语音生成，给出六项叠加优化，合并后将端到端 RTF 降低 **49% / 56% / 62%**（并发 1 / 4 / 8），TTFT 降低约 **70%**，且无精度回退。

### 1.1 Motivation

MiniCPM-o 4.5 语音生成是一个三级管线——Thinker（多模态理解、文本 AR）、Talker（AR codec-token 解码器，约 190M）、Code2Wav（CFM/DiT estimator + HiFT vocoder）。在 NPU 上部署时暴露出三个相互放大的问题：

1. **CFM 图缓存永不收敛。** 每个请求的参考音频长度不同，长度会泄漏进图 key `(chunk length, cache width)`；流式尾部 chunk 落在任意长度上（1..128 帧，均匀分布）。一次 benchmark 捕获了约 430 个图、flush 缓存 13 次——本应重放的图被反复重新捕获，导致 stage2 去 eager 和整图 decode 模式都不可用。
2. **主机开销主导小模型 Talker。** Talker 一步解码约 4.4 ms，其中约 75% 是主机工作（调度、张量准备、kernel 启动），而非设备计算。逐帧支付这笔开销使 RTF 被封顶，与设备吞吐无关。
3. **投机解码恰好错在最关键处。** Thinker 的回答是逐字复述 prompt 加上终止符对 `<|tts_eos|><|im_end|>`。ngram drafter 从 prompt 里抄写，而 prompt 中从不包含 `<|tts_eos|>`——每个请求都要在尾部支付 1–2 个 eager 单 token 步（每步约 37 ms）。

这些对应到服务 SLO：TTFT/audio-TTFP 决定对话时延，RTF 决定流式可持续性。本优化包整合三个内部优化条目（entry_02/18/21）和上游 PR #7416，并在 910B3 上重新测量（基线 vllm-omni main @ `1b6cd282`，vllm-ascend `0.19.1rc2.dev2045`）。

### 1.2 Target

#### Feature

- **CFM 图缓存治理**（PR #7416）：参考音频归一化到固定窗口、mel 帧分桶到 16 帧网格、有效长度掩码、移除 `active_stream_window` 限流——CFM 图库收敛到约 18 个 captures，零 flush。
- **图模式恢复**：stage1 配置 `FULL_AND_PIECEWISE` + 显式 `capture_sizes`（恢复图内采样），stage 级 prefix cache 覆盖，stage2 去 eager。
- **整步解码图（T1）**：`FULL_DECODE_ONLY`——整个解码步（含采样）捕获并重放。
- **fixed-KV 解码后端（T2）**：按容量分桶的 KV，主机侧图参数恒定；桶外请求优雅回退。
- **K8 runner 本地 Talker 解码**：主机开销在 runner 循环内摊到 K=8 帧，带门控且逐位一致。
- **投机解码**：stage0 上的 ngram draft（1/1，k=15）+ 终止符对尾部 draft（D7）。

**Out of Scope（不在范围内）：** 算子级研发（Triton flash-decode、KV affine addressing、A14 采样链全量启用；每项 ≤~3%）；TensorRT stepper（仅 CUDA）；stage 交接协议重构（T44）；增量 frequency penalty；首 chunk 条件缓存；图模式 / 投机解码的运行时动态切换（仅支持重启时配置）。

#### Accuracy

- K8 与 weight_norm 折叠是**构造上逐位一致**的（因果掩码 + 逐帧 stop；数学恒等），并经逐位校验。
- #7416 的掩码保证 padded 帧不进入 softmax 分母；跨 chunk 内存在有效帧边界处清零——seed-tts WER 2.04%→1.42%，SIM/UTMOS 与未分桶解码逐位一致。
- 端到端门槛：seed-tts WER **1.45%**（基线 1.58%）；Daily-Omni 全集 **77.78%**（931/1197，处于 CI-nightly 区间内；`total_input_tokens` 与基线逐位一致）。

#### Performance

910B3，seed-tts/en，同机/同卡/同参数；三档 32 req/c=1、64/c=4、128/c=8：

| 指标 | 基线 | 最终 | Δ |
| --- | --- | --- | --- |
| RTF (32/1) | 0.940 | **0.479** | −49.0% |
| RTF (64/4) | 2.744 | **1.206** | −56.0% |
| RTF (128/8) | 5.843 | **2.246** | −61.6% |
| TTFT (32/1) | 479 ms | **127 ms** | −73.5% |
| TTFT (64/4) | 531 ms | **156 ms** | −70.6% |
| audio TTFP (32/1) | 2390 ms | **1239 ms** | −48.2% |

**Trade-offs（取舍）：** async Omni output 与 ngram 投机互斥（`_should_use_async_omni_output()`）；本包选择 ngram——async output 是有意关闭。K8 要求 stage1 同步调度，且拒绝非均匀 batch（走通用路径回退：无回退损失，也无增益）。Mel 分桶会 pad 到 16 帧（A/B 验证：16 优于 25）。

---

## 2 Design

### 2.1 Overview of Design

三条原则，作用于不同 stage：

1. **让图 key 可枚举**（治理先行）——对输入做归一化/分桶，使图库有界；之后图模式才能生效。
2. **让每个解码步都变成重放**——在 fixed-KV 后端之上做整步图，该后端的主机侧参数永不变化。
3. **主机成本一次付清、提前起草**——K 步 runner 循环摊薄调度开销；投机解码覆盖可预测的 prompt 复述及其终止符尾部。

```
request ──> Stage0 Thinker ──────> Stage1 Talker ──────> Stage2 Code2Wav ──> audio
             text AR                codec-token AR         CFM DiT + HiFT
               │                      │                      │
   ngram draft (k=15)         K8 local K-step loop      ref normalization
   tail-draft rewrite         FULL_DECODE_ONLY          mel bucketing (16f)
   FULL_DECODE_ONLY +         fixed-KV capacity         valid-length masking
   fixed-KV, prefix cache     buckets                   graph wrappers + de-eager
```

依赖链（启用顺序）：**#7416 治理 → 图模式恢复（T2 → T1）→ K8 → 投机解码**。`FULL_DECODE_ONLY` 是一个*组合值*（decode=FULL，prefill=NONE），需经 `decode_mode()` 解析为运行时可用的 FULL 模式；K8 依赖 T1 的 bucket-1 重放路径以及 #727 的 `num_lookahead_tokens` 扩展。

### 2.2 Optimization Item Designs

#### 2.2.1 CFM 图缓存治理（PR #7416）

**问题。** 参考音频长度随请求变化，并经由 attention 缓存起始宽度 L0 进入 CFM 图 key；尾部 chunk 落在任意长度上。key 空间实际上无界。

**机理。**
- *归一化*：重采样到 24 kHz 单声道（混音保护，≤8 通道），零填充/截断到固定窗口（`ref_audio_max_seconds`，默认 6 s）。所有请求共享同一个 L0。
- *Mel 分桶*：每个 CFM chunk 向上对齐到 16 帧的倍数（`cfm_graph_bucket_frames`），解码后裁剪回原长。chunk 收敛到少数几个桶；不足桶 1/3 的帧回退 eager。
- *掩码*：padded 帧使用真实的 `attn_mask`；跨 chunk 内存在有效帧边界处清零后再作为下一 chunk 的 keys——否则积分步 `x = x + dt·v` 会把零填充列重新放大。
- *移除 `active_stream_window`*：有界窗口限流器使并发首包串行化。

**代码落点。**
- `vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_code2wav.py`：`_normalize_reference(ref_audio, sample_rate_hz, ...)`（L96）、`_REF_MAX_SECONDS = 6.0`（L62）、`_normalized_default_prompt`（把随包默认 prompt 折算到请求参考音频的网格上）、`_bucket_key(item)`（L624）、`extra.get("ref_audio_max_seconds", ...)`（L273）
- `vllm_omni/model_executor/models/minicpmo_4_5/batched_token2wav.py`：掩码施加、跨 chunk 缓存边界清零、HiFT 回退
- `vllm_omni/model_executor/models/minicpmo_4_5/cuda_graph_wrapper.py`：图 key 基于分桶后的形状

**配置。** Connector `extra`：`enable_cfm_graph: true`、`enable_hift_graph: true`、`cfm_max_graphs: 32`、`cfm_graph_bucket_frames: 16`、`ref_audio_max_seconds: 6.0`。

**结果。** L20X 分阶段：captures 430→18，flushes 13→0，audio RTF 2.09→1.21，audio TTFP 5.67 s→2.71 s。在 910B3 上：RTF **−24.6% / −52.2% / −59.3%**。是后续所有项的前提。

#### 2.2.2 在解码 stage 恢复图模式

**问题。** 在默认 PIECEWISE 下，每个解码步逐算子下发，**图内采样静默失效**；stage2 的图 wrapper 运行在 eager 引擎里；重复前缀被反复 prefill。

**机理。** stage1 配置 `cudagraph_mode: FULL_AND_PIECEWISE` + 显式 `cudagraph_capture_sizes: [1,2,4,8,16,24,32]`，预先捕获实际会出现的形状（鞋店把每个半码都备上库存，没人被送去 eager「工厂」）。stage 级 prefix caching 开启，同时顶层开关保持 `false`（duplex 安全性不变）——只命中各请求自己的前缀。一旦 2.2.1 把图库控制在 `cfm_max_graphs` 之内，stage2 即可去掉 `enforce_eager`。

**代码落点。**
- `vllm_omni/deploy/minicpmo_4_5.yaml`（npu platform 块）：stage1 `compilation_config.cudagraph_mode: FULL_AND_PIECEWISE`、`cudagraph_capture_sizes`、`max_cudagraph_capture_size: 32`；stage2 `enforce_eager: false`、`additional_config.code2wav_enable_npu_graph: true`、`code2wav_max_npu_graphs: 32`
- `vllm_omni/config/stage_config.py`：图模式校验
- stage 级 prefix-cache 覆盖在 stage-config 路径中实现（顶层 `enable_prefix_caching: false` 不动）

**结果。** RTF −22.0% / −3.9% / +1.1%（第一档收益最大；更高并发下 batching 已摊薄主机开销）。

#### 2.2.3 整步解码图（T1）

**问题。** 即使做了 2.2.2，部分解码 stage 仍走 piecewise；整步图只有在**每个参数跨步保持稳定地址**时才有效。

**机理。** 一个 deploy-path 范围内的 patch 强制解码 stage 使用 `FULL_DECODE_ONLY`（影视布景保持钉死，只换演员手里的道具）。`FULL_DECODE_ONLY` 是*组合值*（decode=FULL，prefill=NONE），本身不是合法运行时模式——runner 通过 `CUDAGraphMode.FULL_DECODE_ONLY.decode_mode()` 将其解析为 forward context 接受的具体 FULL 模式。

**代码落点。**
- `vllm_omni/platforms/npu/worker/npu_ar_model_runner.py`：模式解析（`_talker_local_mode`；commits `e9facfa3`/`21b9f6e7`）
- `vllm_omni/config/stage_config.py`：模式白名单/校验

**结果。** 910B3 同配置 A/B：**−32.6%**——NPU 侧单项收益最大。日志确认 stage0 `PIECEWISE→FULL_DECODE_ONLY`、stage1 `FULL_AND_PIECEWISE→FULL_DECODE_ONLY`。

#### 2.2.4 带容量桶的 fixed-KV 解码后端（T2）

**问题。** 整步图要求所有主机侧参数跨步恒定；paged KV 后端会搬动 block，而 attention 把 KV 长度当作不断增长的主机参数（对小型 Talker 解码器，这一重绑定约占忙时的 38%）。

**机理。** 从 `max_model_len`/`block_size` 预分配 KV **容量桶**；每桶捕获一张图；步内只有 `seq_lens` 和写入槽位变化——算子声明全部 KV 容量，实际长度经 `pse_shift`（图自行刷新的设备张量）传入。桶外请求、chunked/batched prefill、非均匀 query 拒绝捕获并回退。按 `max_model_len` 范围逐 stage 启用（Talker 4096 命中，Thinker 32768 不命中）。

**代码落点。**
- `vllm_omni/platforms/npu/attention/fixed_kv_decode.py`：`capacity_for(max_model_len, block_size)`、`buckets_for(capacity, block_size)`、`select_bucket(max_seq_len, buckets)`、`set_runtime_bucket(...)`、`current_capacity()`、`install_into_ascend_aclgraph(...)`
- `vllm_omni/platforms/npu/attention/fixed_kv_backend.py`：`OmniFixedKVMetadataBuilder`（在 `build()` 中选桶）、`OmniFixedKVAttentionImpl`（`_fixed_kv_applies`、`_uniform_query_len`、`_fixed_kv_graph_fia(query, ...)`、`update_graph_params(update_stream, forward_context, ...)`——移除已废弃的 `num_dcp_pcp_tokens`；为较新的 vllm-ascend 将 `enable_hamming_sparse` 默认 `False`）、融合算子 `TalkerDecodeAttention` 路径
- `vllm_omni/platforms/npu/platform.py`：`get_attn_backend_cls` 安装后端（kill 开关 `VLLM_OMNI_FIXED_KV_DECODE=0`）
- `vllm_omni/model_executor/models/minicpmo_4_5/talker_codec_sample.py`：随同移植的采样器 payload

**结果。** 32/1 **−4.4~5.5%**；64/4 与 128/8 持平。910B 移植无需更换镜像：910C 特性守卫、entry 自有模块 payload（含 A14 绑定）、一处废弃 API 签名，全部在代码内解决。

#### 2.2.5 K8：runner 本地 K 步 Talker 解码

**问题。** Talker 一步约 75% 是逐帧计费的主机开销；调度器/runner 每 4.4 ms 一次往返，设备侧无法掩盖。

**机理。** runner 在每个被调度步内执行 **K 步本地循环**（K=8 默认）。调度器将分配拓宽 K 倍（`num_lookahead_tokens = K−1`），使第 k+1..K 帧的 KV 窗口在其被解码前就已存在；因果掩码保证第 *k* 行永远读不到第 *k+1* 行尚未有效的数据，逐帧 stop 检查与单步解码完全等价地终止序列——**构造上逐位一致**。（类比：一趟托盘端八道菜，而不是每道菜都回柜台取。）

**代码落点。**
- `vllm_omni/platforms/npu/worker/npu_ar_model_runner.py`：`_init_talker_local_decode_config()`、`_talker_local_decode_eligible(...)`（门控：FULL_DECODE_ONLY bucket-1 重放、纯 decode batch、无 prefill chunk / spec-decode / encoder / grammar、batch 落在已捕获桶内）、`_talker_local_decode_loop(scheduler_output, *, num_reqs, req_ids, ...)`、`_talker_local_tokens_pending` 交接
- `vllm_omni/core/sched/omni_ar_scheduler.py`：`num_lookahead_tokens = K−1` 拓宽（仅 Talker stage）、`OMNI_TALKER_SCHED_K` 覆盖

**配置。** Connector `extra`：`talker_local_decode_steps`（K）、`talker_local_decode_stage_id`、`talker_local_cpu_slot_mapping`。环境变量 kill 开关：`OMNI_TALKER_LOCAL_DECODE=0`、`OMNI_TALKER_LOCAL_STEPS=<K>`。stage1 必须设置 `async_scheduling: false`（本地窗口不感知 async 占位）。

**结果。** 原团队报告 −28% 量级的 RTF（entry_18）；910B 移植（`71a2b332`，2 个文件，+1004 行）已落地，**待本机正式 benchmark 确认**。并发 >1 时 draft token 作为整批前进/回退（已知注意事项）。

#### 2.2.6 投机解码：ngram draft + 终止符尾部 draft

**问题。** stage0 输出是对 prompt 的逐字复述——ngram draft 命中率很高——但回答总是以 `<|tts_eos|><|im_end|>` 结尾，而 prompt 中从不包含 `<|tts_eos|>`，因此每个请求在第一个终止符处 draft 结构性失误：1–2 个 eager 单 token 步、每步约 37 ms（无 draft 的步没有已捕获的图；均匀 verify 形状为 16 个 query token，主机约 7 ms）。

**机理。** stage0 运行 `speculative_config` ngram 1/1 k=15。尾部 draft（`stage0_tail_draft.rewrite`）用两条改写规则补齐缺口，二者在拒绝采样器下都是精确的（draft token 只有在模型 argmax 等于它时才会被采纳发出，因此发出的文本不会改变）：
1. draft 从复述越过进入 `<|im_end|>` 时，在该点改写为 `[<|tts_eos|>, <|im_end|>]` 并补齐到全宽（padding = 重复 `<|im_end|>`；已接受 stop 之后的 token 被丢弃，不匹配的 pad 会被拒绝——padding 永远不会进入输出）；
2. 模型已发出 `<|tts_eos|>` 之后的*空* draft，改写为全宽 `<|im_end|>` draft——终止符搭上 16 token 图重放，而不是走 eager 单 token 步。

`VLLM_OMNI_MINICPMO_STAGE0_TAIL_DRAFT=off` 恢复原版 draft；`=strict` 只保留规则 2（规则 1 猜测未来，在自由生成场景默认关闭有风险）。

**代码落点。**
- `vllm_omni/platforms/npu/worker/stage0_tail_draft.py`：`enabled()`、`strict()`、`applies(runner)`（ngram && k==15 && enabled）、`rewrite(drafts, sampled_token_ids, k)`
- `vllm_omni/platforms/npu/worker/npu_ar_model_runner.py`：`propose_draft_token_ids(self, sampled_token_ids, *args, **kwargs)`——**NPU 调用点参数顺序与上游不同**（sampled_token_ids 在前）；签名写错会使投机解码静默退化为 eager
- `vllm_omni/config/omni_config.py`：允许 `speculative_config` 作为 stage 引擎覆盖（白名单，commit `ae3758f6`）
- Deploy yaml stage0：`async_scheduling: false`（VllmConfig 校验不兼容）、`speculative_config: {method: ngram, num_speculative_tokens: 15, prompt_lookup_min: 1, prompt_lookup_max: 1}`

**结果。** ngram 带来 TTFT 531→165 ms（**−69%**）；尾部 draft 再降 141→127 ms（**−10%**）；RTF 持平。

### 2.3 Marginal Items（简述）

**已在基线内**（上游 main @ `1b6cd282`；不计入收益）：#1 stop-token 注入（#3907）、#2 batched_token2wav（#5228+）、#3 ref-audio 注册缓存（#5380；其 L0 膨胀的根因正是 2.2.1 所修复）、#4 prompt 特征缓存、#5/#6 CFM/HiFT 图捕获（#6082/#5869）、#7 lookahead 基建（#727，K8 前提）、#8 async output（关闭：与 ngram 互斥）、#9/#11 加载期 materialize / fp32 上下文（#5228）、#10 workspace 预热（#3773）、#27 STFT 常驻系数（已常驻）。

**本轮落地、收益小：** #17 weight_norm 加载期折叠（`step_audio2_token2wav.py` 中的 `_fold_weight_norm_modules`，`8a115ae7`；MiniCPM-o 经 `minicpmo_4_5_token2wav.py:25` 复用 `StepAudio2Token2WavCore`；逐位一致；RTF −1.6/−5.4/−1.1%，audio TTFP −16.7%）· #16 全链路预热（`async_omni_engine.py`，+97 行，约 20 条后台合成请求；生产首请求优化，对 benchmark 持平）· N1 CPU 绑核（`utils/cpu_isolation.py`；空载容器无可见收益，保留用于抗资源竞争）· N5 `decode_prep_fast`（`worker/decode_prep_fast.py`；带门控的输入复用，本机无增量）· N2 A14 融合采样 payload（随 T2 移植，910B 上 `npu_ops_module()` 为 True，全量启用待验证）。

**未尝试：** #18 TJS（≈0.2%，ragged-cache 回填风险）· #22 TRT（仅 CUDA）· #24 T44（需协议变更）· #26 首 chunk 缓存（每请求参考音频不同 → 命中率低）· #28 增量 frequency penalty（全新代码，收益小）· #30 hidden-offset（本地未复现；daily-omni 精度无异常）· N3/N4（算子研发）。**实测后回退：** #13 CFM 步数缩减（NPU 上 ≤3%，有音质风险）· #23 chunk 尺寸调整（audio TTFP +6~33%）。

### 2.4 Use Cases

**1. 低时延对话 TTS（单流，c=1）**
- 场景：语音助手在单卡 910B3 上以约 130 ms 首 token / 约 1.2 s 首包音频响应。
- 配置：完整优化包（yaml 见 2.5）：ngram + tail draft + K8 + FULL_DECODE_ONLY + fixed-KV。
- 收益：TTFT 479→127 ms，audio TTFP 2390→1239 ms，RTF 0.940→0.479。

**2. 高吞吐批量 TTS（c=8，128 并发）**
- 场景：大量文档的离线旁白合成；RTF / 流式可持续性主导。
- 配置：同一 yaml；分桶图 + stage2 去 eager 承担负载；K8 的 batch 注意事项适用。
- 收益：RTF 5.843→2.246（−61.6%）；混合参考长度下 captures 保持约 18、零 flush。

**3. 全双工流式助手（duplex_session，4 会话）**
- 场景：常驻双工对话，与 `/v1/chat/completions` 共用同一部署档案。
- 配置：`session_mode: duplex`、`codec_chunk_frames: 25`、`codec_left_context_frames: 3`；掩码保证 chunk 边界干净；顶层 prefix caching 为 duplex 安全保持关闭。
- 收益：移除 `active_stream_window` 使并发首包不再串行（audio TTFP −23%）；stage 级 prefix cache 缩短 Thinker prefill 且无跨请求泄漏。

### 2.5 API Design

#### Current Component Changes

| 组件 | 变更 | 原因 / 影响 | 位置 |
| --- | --- | --- | --- |
| `models/minicpmo_4_5/minicpmo_4_5_code2wav.py` | 参考音频归一化、分桶 work-item key、可配置参考窗口 | 使 CFM 图 key 可枚举；过长参考被截断（有日志） | `_normalize_reference`、`_REF_MAX_SECONDS`、`_bucket_key`、`_normalized_default_prompt` |
| `models/minicpmo_4_5/batched_token2wav.py` | 有效长度掩码、跨 chunk 缓存清零、HiFT 回退 | padded 帧不得污染 softmax / CNN 左上下文 | CFM chunk 解码中的掩码路径 |
| `platforms/npu/platform.py` | 按 `max_model_len` 范围安装 fixed-KV 后端 | Talker attention 使用容量桶；`VLLM_OMNI_FIXED_KV_DECODE=0` 恢复原版 | `get_attn_backend_cls` |
| `platforms/npu/attention/fixed_kv_backend.py` / `fixed_kv_decode.py` | 后端类 + 桶代数 + ACLGraphWrapper 钩子 | 解码步以恒定主机参数捕获；`update_graph_params` 与当前 vllm-ascend 对齐 | 见 New APIs |
| `platforms/npu/worker/npu_ar_model_runner.py` | fast decode prep、tail-draft 钩子、K8 配置/门控/循环 | 门控不通过时走通用路径（逐位一致，只是更慢） | `_prepare_inputs`、`propose_draft_token_ids`、`_talker_local_*` |
| `core/sched/omni_ar_scheduler.py` | `num_lookahead_tokens = K−1` 拓宽 | 为本地循环的帧预分配 KV 窗口（缺它时 K=1 下 WER 2.12%→1.00% 回退） | 调度器初始化 + 调度路径 |
| `config/omni_config.py` | 允许 `speculative_config` 作为 stage 引擎覆盖 | stage0 经 deploy yaml 启用 ngram | 白名单（`ae3758f6`） |
| `engine/async_omni_engine.py` | 全链路预热 | 在首个真实请求前吸收惰性捕获 | 后台合成请求 |
| `utils/cpu_isolation.py` | NUMA first-touch 绑核 | 方差收紧；抗资源竞争 | 启动路径 |
| `deploy/minicpmo_4_5.yaml` | 为三 stage 单卡档案接好全部旋钮 | 可复现部署 | cuda/npu platform 块 |

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
enabled() -> bool                     # env 门控 (VLLM_OMNI_MINICPMO_STAGE0_TAIL_DRAFT)
strict() -> bool                      # 仅验证模式
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

配置面（connector `extra` 键）：`ref_audio_max_seconds` · `cfm_graph_bucket_frames` · `cfm_max_graphs` · `enable_cfm_graph` / `enable_hift_graph` · `talker_local_decode_steps` · `talker_local_decode_stage_id` · `talker_local_cpu_slot_mapping` · `code2wav_enable_npu_graph` · `code2wav_max_npu_graphs`。环境变量旋钮：`VLLM_OMNI_FIXED_KV_DECODE=0` · `VLLM_OMNI_MINICPMO_STAGE0_TAIL_DRAFT=off|strict` · `OMNI_TALKER_LOCAL_DECODE=0` · `OMNI_TALKER_LOCAL_STEPS=<K>` · `OMNI_TALKER_SCHED_K`。

### 2.6 API Call Dependency

**启动序列**
1. 加载 deploy yaml；stage 配置校验图模式（`stage_config.py`）与 `speculative_config` 覆盖（`omni_config.py`）。
2. `platform.get_attn_backend_cls` 为 `max_model_len` 落在范围内的 stage 安装 fixed-KV 后端。
3. Code2Wav 加载 flow+HiFT，折叠 weight_norm，构建 CFM/HiFT 图 wrapper（桶由 `codec_chunk_frames`/`cfm_graph_bucket_frames` 推导）。
4. 调度器为 Talker stage 设置 `num_lookahead_tokens = K−1`；runner 读取 K 旋钮（`_init_talker_local_decode_config`，环境变量优先）。
5. 引擎向全链路发出约 20 条预热请求（吸收惰性捕获）。

**逐请求解码（Talker，K8 生效）**
1. 调度器调度该步；runner 检查 `_talker_local_decode_eligible`——拒绝时走通用路径（结果正确，只是更慢）。
2. 通过门控：`_talker_local_decode_loop` 重放 K 个 FULL_DECODE_ONLY 步；逐帧 stop 可提前退出；输出经 `_talker_local_tokens_pending` 交还。
3. 第 k+1..K 帧的 KV 已由拓宽预分配；因果掩码保证每行只读本地。

**stage0 投机步**
1. `propose_draft_token_ids(sampled_token_ids, ...)` 构建 ngram draft；`stage0_tail_draft.rewrite` 施加两条尾部改写并补齐到 k=15。
2. 16-token verify 图重放；拒绝采样器保证发出的文本不变。
3. 错误路径：ngram 关闭 → 不做改写（`applies` 为 False）；`strict` 模式 → 只保留规则 2；空 draft 规则仅在模型自身已发出 `<|tts_eos|>` 后触发。

---

## 3 Test cases

### 3.1 Unit Test(UT) design

**已有 UT 覆盖**（均在 `tests/model_executor/models/minicpmo_4_5/` 下）：

| 文件 | 覆盖内容 |
| --- | --- |
| `test_reference_audio_normalization.py` | 24 kHz 重采样、混音保护、6 s 窗口填充/截断、cache key 稳定性 |
| `test_cfm_graph_bucketing.py` | 16 帧桶对齐、裁剪回原长精确性、桶 key |
| `test_cfm_graph_capture_gating.py` | 捕获门控，含 <⅓ 桶回退 eager |
| `test_code2wav_batching.py` | work-item batching、`_bucket_key` 家族 |
| `test_audio_chunk_mask.py` | 有效长度掩码、跨 chunk 缓存边界清零 |
| `test_cuda_graph_wrapper.py` | 图 key/重放 wrapper 语义 |
| `test_streaming_audio_cache.py` | 流式 chunk 缓存正确性 |
| `test_talker_batching.py`、`test_pipeline.py`、`test_llm2tts.py`、`duplex/` | Talker batching、管线、duplex 路径 |

**UT 缺口与建议新增**（NPU worker 相关项目前无 UT）：

1. `tests/platforms/npu/attention/test_fixed_kv_buckets.py` — `test_select_bucket_boundaries()`
   - 目的：桶选择永不返回小于 `max_seq_len` 的桶；超过容量返回 `None`。
   - 步骤：在桶边界精确值、边界 ±1、超过容量处调用 `select_bucket`；校验 `set_runtime_bucket`/`current_capacity` 的写读一致性。
   - 断言：选择单调；桶外 → 走拒绝捕获路径（`_fixed_kv_applies` 为 False）。
2. `tests/platforms/npu/worker/test_stage0_tail_draft.py` — `test_rewrite_rules()`
   - 步骤：(a) 含 `<|im_end|>` 的 draft → 尾部改写为 `[151704, 151645]`，补齐到 k；(b) 已发出 `151704` 后的空 draft → 全宽 `151645`；(c) strict 模式抑制规则 (a)；(d) 环境变量关闭 → draft 原样；(e) padding token 永远不会出现在输出（不匹配的 pad 被构造性拒绝）。
   - 断言：每条规则的输出 draft 不变量；不修改 `sampled_token_ids`。
3. `tests/platforms/npu/worker/test_decode_prep_fast.py` — `test_cache_invalidation_and_gates()`
   - 步骤：连续相同 decode 步的快速命中；元数据变化后调用 `invalidate()`；门控失败 → 经 `note_generic` 记录走通用路径。
   - 断言：stub runner 上快速路径输出与通用 `_prepare_inputs` 完全一致。
4. `tests/platforms/npu/worker/test_talker_local_decode.py` — `test_eligibility_matrix()`
   - 步骤：可入选（FULL_DECODE_ONLY、纯 decode、K 落在桶内）与被拒（spec decode 开启、存在 prefill chunk、模式为 NONE 且 S≠K、batch 超大）。
   - 断言：拒绝原因带标签；逐位一致性契约成文（因果掩码 + 逐帧 stop）。
5. `tests/config/test_omni_config_speculative_override.py` — stage 级 `speculative_config` 通过校验；顶层 duplex 开关不受影响。
6. `tests/utils/test_cpu_isolation.py` — 绑核在引擎启动前施加；失败时回退且不中断启动。

### 3.2 Smoke Test(ST) design

1. `test_minicpmo_perf_pack_serving()` — 端到端三档 serving 冒烟
   - 环境：模型 MiniCPM-o 4.5；1× 910B3；`vllm_omni/deploy/minicpmo_4_5.yaml`；服务端 `vllm-omni serve <model> --omni --port 8091`。
   - 步骤：跑三档 seed-tts bench（32/1、64/4、128/8），参数 `--backend openai-chat-omni --ignore-eos --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf --extra_body '{"modalities":["text","audio"]}'`。
   - 断言：启动日志显示 stage0/1 为 `FULL_DECODE_ONLY`、`fast decode prep engaged` 或干净回退、CFM captures ≈18 且 0 flush、无图容量回退 eager；RTF/TTFT 落在预期带内（0.48 / 1.21 / 2.25 ±10%；TTFT ≤ 170 ms）。
   - 超时：30 分钟（三档含预热）。
2. `test_minicpmo_wer_smoke()` — 精度冒烟
   - 环境：同一服务；官方 WER 路径（whisper-tiny.en，20 条 prompt）。
   - 断言：WER ≤ 1.6%；20/20 成功；WER 中位数为 0。
3. `test_daily_omni_accuracy()` — 多模态门槛
   - 环境：设置 `VLLM_DAILY_OMNI_QA_JSON` + `VLLM_DAILY_OMNI_VIDEO_DIR`；客户端参数 `--daily-omni-pack-mode minicpm-interleave`、`--output-len 512`、`--daily-extra-body-json '{"modalities":["text"],"chat_template_kwargs":{"enable_thinking":false}}'`（缺任一项精度会掉到约 31%）。
   - 断言：全集精度处于 CI-nightly 区间（77.6–78.2%）；`total_input_tokens` 与基线逐位一致（4349821）。
   - 超时：2 小时。
4. `test_duplex_session_smoke()` — duplex 路径
   - 环境：`session_mode: duplex`，4 个并发 chat 会话。
   - 断言：无跨会话缓存污染（掩码生效）、并发会话首包不串行、输出在 WER 门槛内。

**环境注意事项**（复现避坑）：`git checkout` 后在目标代码树上运行 `populate_modelinfo_cache.py`（按 content-hash 校验）；精度评测框架**不要**设置 `ASCEND_RT_VISIBLE_DEVICES`（它使用物理 ID）；退出时的 `corrupted size vs. prev_size` 是 glibc 噪音——以产物 JSON 为准。
