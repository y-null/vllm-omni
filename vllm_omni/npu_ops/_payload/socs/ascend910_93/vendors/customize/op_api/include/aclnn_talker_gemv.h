
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_TALKER_GEMV_H_
#define ACLNN_TALKER_GEMV_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnTalkerGemvGetWorkspaceSize
 * parameters :
 * x : required
 * weight : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnTalkerGemvGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnTalkerGemv
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnTalkerGemv(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
