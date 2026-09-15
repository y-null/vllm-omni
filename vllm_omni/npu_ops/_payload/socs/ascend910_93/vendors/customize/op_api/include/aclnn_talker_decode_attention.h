
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_TALKER_DECODE_ATTENTION_H_
#define ACLNN_TALKER_DECODE_ATTENTION_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnTalkerDecodeAttentionGetWorkspaceSize
 * parameters :
 * query : required
 * keyCache : required
 * valueCache : required
 * blockTable : required
 * seqLens : required
 * numHeads : required
 * numKvHeads : required
 * scale : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnTalkerDecodeAttentionGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *keyCache,
    const aclTensor *valueCache,
    const aclTensor *blockTable,
    const aclTensor *seqLens,
    int64_t numHeads,
    int64_t numKvHeads,
    double scale,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnTalkerDecodeAttention
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnTalkerDecodeAttention(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
