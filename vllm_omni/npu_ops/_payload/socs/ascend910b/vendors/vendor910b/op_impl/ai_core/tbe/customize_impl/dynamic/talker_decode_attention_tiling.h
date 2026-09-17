#ifndef TALKER_DECODE_ATTENTION_TILING_H
#define TALKER_DECODE_ATTENTION_TILING_H
#include <cstdint>

// Mirrors the fields KernelTalkerDecodeAttention reads out of the tiling blob.
// Order and width matter: the kernel reinterprets this struct straight from the
// GET_TILING_DATA payload the host wrote, so both sides must agree byte for byte.
struct TalkerDecodeAttentionTilingData {
    uint32_t batch;
    uint32_t numHeads;
    uint32_t headDim;
    uint32_t blockSize;
    uint32_t maxBlocks;
    uint32_t kvStride;
    uint32_t kvCapacity;
    uint32_t numBlocks;
    float scale;
};

#endif  // TALKER_DECODE_ATTENTION_TILING_H
