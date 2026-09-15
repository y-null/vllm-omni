#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t VOCAB_SIZE = 6562;
constexpr uint32_t CHUNK0_INNER = 4096;
constexpr uint32_t CHUNK1_INNER = 2496;
constexpr uint32_t PADDED_VOCAB_SIZE = CHUNK0_INNER + CHUNK1_INNER;
constexpr uint32_t HISTORY_WINDOW = 16;
constexpr uint32_t MAX_TOP_K = 100;
constexpr uint32_t MAX_TOPK_OUTPUT_PAD = 104;
constexpr uint32_t MAX_MERGE_INNER = 224;
constexpr uint32_t REDUCE_MASK = 64;
constexpr uint32_t REDUCE_REPEATS = PADDED_VOCAB_SIZE / REDUCE_MASK;

class KernelTalkerCodecLogitsPrepare {
public:
    __aicore__ inline void Init(GM_ADDR rawLogits, GM_ADDR history, GM_ADDR historyLen,
                               GM_ADDR step, GM_ADDR minTokens, GM_ADDR temperature,
                               GM_ADDR repetitionPenalty, GM_ADDR warpedLogits,
                               const TalkerCodecLogitsPrepareTilingData &tiling)
    {
        vocabSize_ = tiling.vocabSize;
        eosTokenId_ = tiling.eosTokenId;
        historyWindow_ = tiling.historyWindow;
        topK_ = tiling.topK;
        topP_ = tiling.topP;
        minTokensToKeep_ = tiling.minTokensToKeep;
        rawLogitsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(rawLogits), vocabSize_);
        historyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(history), historyWindow_);
        historyLenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(historyLen), 1);
        stepGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(step), 1);
        minTokensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(minTokens), 1);
        temperatureGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(temperature), 1);
        repetitionPenaltyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(repetitionPenalty), 1);
        warpedLogitsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(warpedLogits), vocabSize_);
        pipe_.InitBuffer(logitsBuf_, PADDED_VOCAB_SIZE * sizeof(float));
        pipe_.InitBuffer(historyBuf_, HISTORY_WINDOW * sizeof(int32_t));
        pipe_.InitBuffer(candidateValuesBuf_, MAX_MERGE_INNER * sizeof(float));
        pipe_.InitBuffer(candidateIndicesBuf_, MAX_MERGE_INNER * sizeof(int32_t));
        pipe_.InitBuffer(topValuesBuf_, MAX_TOPK_OUTPUT_PAD * sizeof(float));
        pipe_.InitBuffer(topIndicesBuf_, MAX_TOPK_OUTPUT_PAD * sizeof(int32_t));
        pipe_.InitBuffer(reduceBuf_, 128 * sizeof(float));
        pipe_.InitBuffer(topkTmpBuf_, tiling.topkTmpBytes);
        chunk0Topk_ = tiling.chunk0Topk;
        chunk1Topk_ = tiling.chunk1Topk;
        mergeTopk_ = tiling.mergeTopk;
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> logits = logitsBuf_.Get<float>();
        LocalTensor<int32_t> history = historyBuf_.Get<int32_t>();
        const DataCopyExtParams logitsCopy{1, static_cast<uint32_t>(VOCAB_SIZE * sizeof(float)), 0, 0, 0};
        const DataCopyPadExtParams<float> logitsPad{true, 0,
            static_cast<uint8_t>(8 - VOCAB_SIZE % 8), -__builtin_inff()};
        DataCopyPad(logits, rawLogitsGm_, logitsCopy, logitsPad);
        DataCopy(history, historyGm_, HISTORY_WINDOW);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        for (uint32_t index = (VOCAB_SIZE + 7) / 8 * 8;
             index < PADDED_VOCAB_SIZE; ++index) {
            logits.SetValue(index, -__builtin_inff());
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);

        const float temperature = temperatureGm_.GetValue(0);
        Muls(logits, logits, 1.0F / temperature, static_cast<int32_t>(PADDED_VOCAB_SIZE));
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);

        int32_t historyLen = historyLenGm_.GetValue(0);
        if (historyLen < 0) historyLen = 0;
        if (historyLen > static_cast<int32_t>(historyWindow_)) historyLen = historyWindow_;
        ApplyRepetitionPenalty(logits, history, historyLen, repetitionPenaltyGm_.GetValue(0));
        if (stepGm_.GetValue(0) < minTokensGm_.GetValue(0)) {
            logits.SetValue(eosTokenId_, -__builtin_inff());
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);

        LocalTensor<float> candidateValues = candidateValuesBuf_.Get<float>();
        LocalTensor<int32_t> candidateIndices = candidateIndicesBuf_.Get<int32_t>();
        LocalTensor<float> topValues = topValuesBuf_.Get<float>();
        LocalTensor<int32_t> topIndices = topIndicesBuf_.Get<int32_t>();
        LocalTensor<uint8_t> topkTmp = topkTmpBuf_.Get<uint8_t>();
        LocalTensor<int32_t> unusedIndices;
        LocalTensor<bool> unusedFinish;
        const uint32_t candidateK = topK_ > minTokensToKeep_ ? topK_ : minTokensToKeep_;
        const uint32_t topKOutputPad = (candidateK + 7U) / 8U * 8U;
        const uint32_t mergeN = 2U * candidateK;
        const uint32_t mergeInner = (mergeN + 31U) / 32U * 32U;
        const TopKInfo chunk0Info{1, static_cast<int32_t>(CHUNK0_INNER),
                                  static_cast<int32_t>(CHUNK0_INNER)};
        const TopKInfo chunk1Info{1, static_cast<int32_t>(CHUNK1_INNER),
                                  static_cast<int32_t>(VOCAB_SIZE - CHUNK0_INNER)};
        TopK<float, false, false, false, TopKMode::TOPK_NORMAL>(
            candidateValues, candidateIndices, logits, unusedIndices, unusedFinish,
            topkTmp, candidateK, chunk0Topk_, chunk0Info, true);
        PipeBarrier<PIPE_V>();
        TopK<float, false, false, false, TopKMode::TOPK_NORMAL>(
            candidateValues[topKOutputPad], candidateIndices[topKOutputPad],
            logits[CHUNK0_INNER], unusedIndices, unusedFinish, topkTmp,
            candidateK, chunk1Topk_, chunk1Info, true);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (uint32_t index = 0; index < candidateK; ++index) {
            candidateValues.SetValue(candidateK + index,
                                     candidateValues.GetValue(topKOutputPad + index));
            candidateIndices.SetValue(candidateK + index,
                                      candidateIndices.GetValue(topKOutputPad + index) +
                                          static_cast<int32_t>(CHUNK0_INNER));
        }
        for (uint32_t index = mergeN; index < mergeInner; ++index) {
            candidateValues.SetValue(index, -__builtin_inff());
            candidateIndices.SetValue(index, static_cast<int32_t>(index));
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        const TopKInfo mergeInfo{1, static_cast<int32_t>(mergeInner),
                                 static_cast<int32_t>(mergeN)};
        TopK<float, true, false, false, TopKMode::TOPK_NORMAL>(
            topValues, topIndices, candidateValues, candidateIndices, unusedFinish,
            topkTmp, candidateK, mergeTopk_, mergeInfo, true);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);

        const float maximum = topValues.GetValue(0);
        Adds(logits, logits, -maximum, static_cast<int32_t>(PADDED_VOCAB_SIZE));
        PipeBarrier<PIPE_V>();
        Exp(logits, logits, static_cast<int32_t>(PADDED_VOCAB_SIZE));
        PipeBarrier<PIPE_V>();
        LocalTensor<float> reduced = reduceBuf_.Get<float>();
        WholeReduceSum<float>(reduced, logits, REDUCE_MASK, REDUCE_REPEATS, 1, 1, 8);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float denominator = 0.0F;
        for (uint32_t index = 0; index < REDUCE_REPEATS; ++index) {
            denominator += reduced.GetValue(index);
        }

        uint32_t keepCount = 0;
        float cumulative = 0.0F;
        const bool topPEnabled = topP_ > 0.0F && topP_ < 1.0F;
        for (uint32_t index = 0; index < candidateK; ++index) {
            const bool forced = index < minTokensToKeep_;
            if (!forced && topPEnabled && cumulative / denominator >= topP_) break;
            const int32_t token = topIndices.GetValue(index);
            cumulative += logits.GetValue(static_cast<uint32_t>(token));
            ++keepCount;
        }

        Duplicate(logits, -__builtin_inff(), static_cast<int32_t>(PADDED_VOCAB_SIZE));
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (uint32_t index = 0; index < keepCount; ++index) {
            logits.SetValue(static_cast<uint32_t>(topIndices.GetValue(index)),
                            topValues.GetValue(index));
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        const DataCopyExtParams outputCopy{1, static_cast<uint32_t>(VOCAB_SIZE * sizeof(float)), 0, 0, 0};
        DataCopyPad(warpedLogitsGm_, logits, outputCopy);
        PipeBarrier<PIPE_ALL>();
    }

private:
    __aicore__ inline void ApplyRepetitionPenalty(LocalTensor<float> logits,
                                                  LocalTensor<int32_t> history,
                                                  int32_t historyLen, float penalty)
    {
        if (historyLen == 0 || penalty == 1.0f) return;
        int32_t uniqueTokens[HISTORY_WINDOW];
        int32_t counts[HISTORY_WINDOW];
        uint32_t uniqueCount = 0;
        for (int32_t index = 0; index < historyLen; ++index) {
            const int32_t token = history.GetValue(static_cast<uint32_t>(index));
            if (token < 0 || token >= static_cast<int32_t>(vocabSize_)) continue;
            uint32_t owner = uniqueCount;
            for (uint32_t unique = 0; unique < uniqueCount; ++unique) {
                if (uniqueTokens[unique] == token) { owner = unique; break; }
            }
            if (owner == uniqueCount) {
                uniqueTokens[uniqueCount] = token;
                counts[uniqueCount] = 1;
                ++uniqueCount;
            } else {
                ++counts[owner];
            }
        }
        if (uniqueCount == 0) return;
        for (uint32_t index = 0; index < uniqueCount; ++index) {
            const uint32_t token = static_cast<uint32_t>(uniqueTokens[index]);
            const float value = logits.GetValue(token);
            float alpha = 1.0f;
            for (int32_t repeat = 0; repeat < counts[index]; ++repeat) alpha *= penalty;
            logits.SetValue(token, value < 0.0f ? value * alpha : value / alpha);
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> logitsBuf_;
    TBuf<TPosition::VECCALC> historyBuf_;
    TBuf<TPosition::VECCALC> candidateValuesBuf_;
    TBuf<TPosition::VECCALC> candidateIndicesBuf_;
    TBuf<TPosition::VECCALC> topValuesBuf_;
    TBuf<TPosition::VECCALC> topIndicesBuf_;
    TBuf<TPosition::VECCALC> reduceBuf_;
    TBuf<TPosition::VECCALC> topkTmpBuf_;
    GlobalTensor<float> rawLogitsGm_;
    GlobalTensor<int32_t> historyGm_;
    GlobalTensor<int32_t> historyLenGm_;
    GlobalTensor<int32_t> stepGm_;
    GlobalTensor<int32_t> minTokensGm_;
    GlobalTensor<float> temperatureGm_;
    GlobalTensor<float> repetitionPenaltyGm_;
    GlobalTensor<float> warpedLogitsGm_;
    uint32_t vocabSize_ = VOCAB_SIZE;
    uint32_t eosTokenId_ = VOCAB_SIZE - 1;
    uint32_t historyWindow_ = HISTORY_WINDOW;
    uint32_t topK_ = MAX_TOP_K;
    float topP_ = 0.95F;
    uint32_t minTokensToKeep_ = 3;
    AscendC::tiling::TopkTiling chunk0Topk_;
    AscendC::tiling::TopkTiling chunk1Topk_;
    AscendC::tiling::TopkTiling mergeTopk_;
};
}  // namespace

extern "C" __global__ __aicore__ void talker_codec_logits_prepare(
    GM_ADDR raw_logits, GM_ADDR history, GM_ADDR history_len, GM_ADDR step,
    GM_ADDR min_tokens, GM_ADDR temperature, GM_ADDR repetition_penalty,
    GM_ADDR warped_logits, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    (void)workspace;
    KernelTalkerCodecLogitsPrepare kernel;
    kernel.Init(raw_logits, history, history_len, step, min_tokens, temperature,
                repetition_penalty, warped_logits, tilingData);
    kernel.Process();
}
