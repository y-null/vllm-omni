#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(TalkerCodecLogitsPrepare)
    .INPUT(raw_logits, ge::TensorType::ALL())
    .INPUT(history, ge::TensorType::ALL())
    .INPUT(history_len, ge::TensorType::ALL())
    .INPUT(step, ge::TensorType::ALL())
    .INPUT(min_tokens, ge::TensorType::ALL())
    .INPUT(temperature, ge::TensorType::ALL())
    .INPUT(repetition_penalty, ge::TensorType::ALL())
    .OUTPUT(warped_logits, ge::TensorType::ALL())
    .REQUIRED_ATTR(vocab_size, Int)
    .REQUIRED_ATTR(eos_token_id, Int)
    .REQUIRED_ATTR(history_window, Int)
    .REQUIRED_ATTR(top_k, Int)
    .REQUIRED_ATTR(top_p, Float)
    .REQUIRED_ATTR(min_tokens_to_keep, Int)
    .OP_END_FACTORY_REG(TalkerCodecLogitsPrepare);

REG_OP(TalkerDecodeAttention)
    .INPUT(query, ge::TensorType::ALL())
    .INPUT(key_cache, ge::TensorType::ALL())
    .INPUT(value_cache, ge::TensorType::ALL())
    .INPUT(block_table, ge::TensorType::ALL())
    .INPUT(seq_lens, ge::TensorType::ALL())
    .OUTPUT(attn_out, ge::TensorType::ALL())
    .REQUIRED_ATTR(num_heads, Int)
    .REQUIRED_ATTR(num_kv_heads, Int)
    .REQUIRED_ATTR(scale, Float)
    .OP_END_FACTORY_REG(TalkerDecodeAttention);

}

#endif
