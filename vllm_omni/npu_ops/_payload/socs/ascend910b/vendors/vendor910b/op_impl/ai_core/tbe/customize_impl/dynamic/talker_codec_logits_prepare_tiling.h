#ifndef TALKER_CODEC_LOGITS_PREPARE_TILING_H
#define TALKER_CODEC_LOGITS_PREPARE_TILING_H
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

// The kernel reads these fields straight out of the GET_TILING_DATA blob, and
// the three TopK tiling structs are filled on the host by TopKTilingFunc.
struct TalkerCodecLogitsPrepareTilingData {
    uint32_t vocabSize;
    uint32_t eosTokenId;
    uint32_t historyWindow;
    uint32_t topK;
    float topP;
    uint32_t minTokensToKeep;
    uint32_t topkTmpBytes;
    AscendC::tiling::TopkTiling chunk0Topk;
    AscendC::tiling::TopkTiling chunk1Topk;
    AscendC::tiling::TopkTiling mergeTopk;
};

#endif  // TALKER_CODEC_LOGITS_PREPARE_TILING_H
