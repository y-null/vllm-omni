#include "kernel_operator.h"

using namespace AscendC;

namespace {
// v1 boundary, checked on the host: K a multiple of one fp32 repeat, x small
// enough to sit in UB whole, weights row-major [N, K] so every output element
// reads one contiguous K-row.
constexpr uint32_t FP32_PER_REPEAT = 64;
constexpr uint32_t MAX_K = 4096;
// Rows staged per chunk: the double-buffered working set is
// 2 x CHUNK_ELEMS x 2B (halves in the queue) + CHUNK_ELEMS x 4B (floats).
constexpr uint32_t CHUNK_ELEMS = 8192;
// Worst-case per-row partial count is MAX_K / 64, padded to a whole 32-byte
// block so every row's slice starts aligned.
constexpr uint32_t MAX_PARTIALS = MAX_K / FP32_PER_REPEAT;

class KernelTalkerGemv {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR y,
                                const TalkerGemvTilingData &tiling)
    {
        k_ = tiling.k;
        n_ = tiling.n;
        rowsPerCore_ = tiling.rowsPerCore;
        rowsPerChunk_ = tiling.rowsPerChunk;
        // Row partials at an aligned stride wide enough for this K. The first
        // version used a fixed stride of 16, which K=3072's 48 partials
        // overran -- rows clobbered each other and `down` came back wrong.
        partialStride_ = ((k_ / FP32_PER_REPEAT) + 7U) & ~7U;

        const uint32_t core = GetBlockIdx();
        rowStart_ = core * rowsPerCore_;
        rowCount_ = rowsPerCore_;
        if (rowStart_ >= n_) {
            rowCount_ = 0;
        } else if (rowStart_ + rowCount_ > n_) {
            rowCount_ = n_ - rowStart_;
        }

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(x), k_);
        weightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(weight),
                                  static_cast<uint64_t>(n_) * k_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(y), n_);

        pipe_.InitBuffer(inQue_, 2, CHUNK_ELEMS * sizeof(bfloat16_t));
        pipe_.InitBuffer(xHalfBuf_, MAX_K * sizeof(bfloat16_t));
        pipe_.InitBuffer(xBuf_, MAX_K * sizeof(float));
        pipe_.InitBuffer(rowsBuf_, CHUNK_ELEMS * sizeof(float));
        pipe_.InitBuffer(sumBuf_, (CHUNK_ELEMS / FP32_PER_REPEAT) * MAX_PARTIALS * sizeof(float) / 8);
        pipe_.InitBuffer(outBuf_, (CHUNK_ELEMS / FP32_PER_REPEAT) * sizeof(float));
        pipe_.InitBuffer(outHalfBuf_, (CHUNK_ELEMS / FP32_PER_REPEAT) * sizeof(bfloat16_t));
    }

    __aicore__ inline void Process()
    {
        if (rowCount_ == 0) {
            return;
        }
        LocalTensor<bfloat16_t> xHalf = xHalfBuf_.Get<bfloat16_t>();
        DataCopy(xHalf, xGm_, k_);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        LocalTensor<float> x = xBuf_.Get<float>();
        Cast(x, xHalf, RoundMode::CAST_NONE, k_);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> rows = rowsBuf_.Get<float>();
        LocalTensor<float> sums = sumBuf_.Get<float>();
        LocalTensor<float> out = outBuf_.Get<float>();
        LocalTensor<bfloat16_t> outHalf = outHalfBuf_.Get<bfloat16_t>();

        const uint32_t repeatsPerRow = k_ / FP32_PER_REPEAT;
        const uint32_t chunks = (rowCount_ + rowsPerChunk_ - 1) / rowsPerChunk_;

        // The copy of chunk c+1 is issued before the vector work of chunk c,
        // so MTE2 streams the next rows while the vector unit reduces the
        // current ones.
        CopyChunk(0);
        for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
            if (chunk + 1 < chunks) {
                CopyChunk(chunk + 1);
            }
            const uint32_t done = chunk * rowsPerChunk_;
            uint32_t take = rowsPerChunk_;
            if (done + take > rowCount_) {
                take = rowCount_ - done;
            }
            LocalTensor<bfloat16_t> rowsHalf = inQue_.DeQue<bfloat16_t>();
            Cast(rows, rowsHalf, RoundMode::CAST_NONE, take * k_);
            inQue_.FreeTensor(rowsHalf);
            PipeBarrier<PIPE_V>();
            for (uint32_t row = 0; row < take; ++row) {
                Mul(rows[row * k_], rows[row * k_], x, k_);
            }
            PipeBarrier<PIPE_V>();
            // Fold 1, per row: K collapses to `repeatsPerRow` partials at an
            // aligned per-row stride. (Per-row calls because the source
            // stride between rows, K/8 blocks, overflows the uint8 repeat
            // stride at K=3072.)
            for (uint32_t row = 0; row < take; ++row) {
                WholeReduceSum<float>(sums[row * partialStride_], rows[row * k_],
                                      FP32_PER_REPEAT, repeatsPerRow, 1, 1,
                                      FP32_PER_REPEAT / 8);
            }
            PipeBarrier<PIPE_V>();
            // Fold 2, one call: each repeat reduces one row's partials into
            // consecutive outputs.
            WholeReduceSum<float>(out, sums, repeatsPerRow, take, 1, 1,
                                  partialStride_ / 8);
            PipeBarrier<PIPE_V>();
            Cast(outHalf, out, RoundMode::CAST_RINT, take);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyPad(yGm_[rowStart_ + done], outHalf,
                        DataCopyExtParams{1, static_cast<uint32_t>(take * sizeof(bfloat16_t)), 0, 0, 0});
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

private:
    __aicore__ inline void CopyChunk(uint32_t chunk)
    {
        const uint32_t done = chunk * rowsPerChunk_;
        uint32_t take = rowsPerChunk_;
        if (done + take > rowCount_) {
            take = rowCount_ - done;
        }
        LocalTensor<bfloat16_t> staging = inQue_.AllocTensor<bfloat16_t>();
        const uint64_t srcOffset = (static_cast<uint64_t>(rowStart_) + done) * k_;
        DataCopy(staging, weightGm_[srcOffset], take * k_);
        inQue_.EnQue(staging);
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, 2> inQue_;
    TBuf<TPosition::VECCALC> xHalfBuf_, xBuf_, rowsBuf_, sumBuf_, outBuf_, outHalfBuf_;
    GlobalTensor<bfloat16_t> xGm_, weightGm_, yGm_;
    uint32_t k_ = 0;
    uint32_t n_ = 0;
    uint32_t rowsPerCore_ = 0;
    uint32_t rowsPerChunk_ = 0;
    uint32_t partialStride_ = 0;
    uint32_t rowStart_ = 0;
    uint32_t rowCount_ = 0;
};
}  // namespace

extern "C" __global__ __aicore__ void talker_gemv(GM_ADDR x, GM_ADDR weight, GM_ADDR y,
                                                  GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);
    KernelTalkerGemv op;
    op.Init(x, weight, y, tilingData);
    op.Process();
}
