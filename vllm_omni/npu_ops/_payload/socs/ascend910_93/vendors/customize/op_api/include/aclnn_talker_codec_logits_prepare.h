
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_TALKER_CODEC_LOGITS_PREPARE_H_
#define ACLNN_TALKER_CODEC_LOGITS_PREPARE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnTalkerCodecLogitsPrepareGetWorkspaceSize
 * parameters :
 * rawLogits : required
 * history : required
 * historyLen : required
 * step : required
 * minTokens : required
 * temperature : required
 * repetitionPenalty : required
 * vocabSize : required
 * eosTokenId : required
 * historyWindow : required
 * topK : required
 * topP : required
 * minTokensToKeep : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnTalkerCodecLogitsPrepareGetWorkspaceSize(
    const aclTensor *rawLogits,
    const aclTensor *history,
    const aclTensor *historyLen,
    const aclTensor *step,
    const aclTensor *minTokens,
    const aclTensor *temperature,
    const aclTensor *repetitionPenalty,
    int64_t vocabSize,
    int64_t eosTokenId,
    int64_t historyWindow,
    int64_t topK,
    double topP,
    int64_t minTokensToKeep,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnTalkerCodecLogitsPrepare
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnTalkerCodecLogitsPrepare(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
