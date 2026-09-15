
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_TALKER_CODEC_SAMPLE_ADVANCE_H_
#define ACLNN_TALKER_CODEC_SAMPLE_ADVANCE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnTalkerCodecSampleAdvanceGetWorkspaceSize
 * parameters :
 * logits : required
 * noise : required
 * history : required
 * historyLen : required
 * step : required
 * finished : required
 * maxTokens : required
 * eosTokenId : required
 * sampledOut : required
 * nextHistoryOut : required
 * nextHistoryLenOut : required
 * nextStepOut : required
 * finishedOutOut : required
 * emitOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnTalkerCodecSampleAdvanceGetWorkspaceSize(
    const aclTensor *logits,
    const aclTensor *noise,
    const aclTensor *history,
    const aclTensor *historyLen,
    const aclTensor *step,
    const aclTensor *finished,
    const aclTensor *maxTokens,
    int64_t eosTokenId,
    const aclTensor *sampledOut,
    const aclTensor *nextHistoryOut,
    const aclTensor *nextHistoryLenOut,
    const aclTensor *nextStepOut,
    const aclTensor *finishedOutOut,
    const aclTensor *emitOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnTalkerCodecSampleAdvance
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnTalkerCodecSampleAdvance(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
