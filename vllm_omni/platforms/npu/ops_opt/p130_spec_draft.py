"""p130: 投机采样草稿链（ngram 提议 + 目标模型并行验证）。

机理：draft 侧用多尺寸 ngram 前缀表（后缀 tuple -> 候选 Counter）按最长
匹配提议 K 个音频码；目标模型把 [context + draft[:-1]] 单次前向并行打分，
贪心接受最长一致前缀，全不中时用末位 logits argmax 回退修正（每步至少
出 1 token）。接口函数式，可被 talker/codec 逐帧循环或 vllm spec_decode
钩子复用。
来源：自行设计（entry_02 p130 描述）。
集成点：minicpmo_4_5_omni_tts.py 的 talker/codec 逐帧生成循环；
DraftProposer 可特化替换 vllm/v1/spec_decode/ngram_proposer.py。
env 门：MINICPMO_P130_SPEC_DRAFT（默认 "0"，关闭时 propose 恒返 []，
即退化为纯贪心，但序列仍持续入表供将来开启）。
"""
import os
from collections import Counter
from typing import Callable, List, Sequence, Tuple

_P130_ENABLED = os.environ.get("MINICPMO_P130_SPEC_DRAFT", "0") == "1"


class DraftProposer:
    """多尺寸 ngram 前缀查表提议器。"""

    def __init__(self, max_suffix: int = 10, min_suffix: int = 1, topk_per_suffix: int = 1):
        self.max_suffix = max_suffix
        self.min_suffix = min_suffix
        self.topk = topk_per_suffix
        self._table: dict = {}

    def observe(self, ids: Sequence[int]) -> None:
        """把一段序列的全部 (suffix -> next) 对吸收进表。"""
        ids = list(ids)
        for i in range(1, len(ids)):
            for n in range(self.min_suffix, self.max_suffix + 1):
                if i - n < 0:
                    break
                key = tuple(ids[i - n : i])
                self._table.setdefault(key, Counter()).update([ids[i]])

    def propose(self, context: Sequence[int], k: int) -> List[int]:
        """最长匹配后缀优先；env 关闭恒返 []。"""
        if not _P130_ENABLED or k <= 0 or not self._table:
            return []
        ctx = list(context)
        for n in range(min(self.max_suffix, len(ctx)), self.min_suffix - 1, -1):
            counter = self._table.get(tuple(ctx[-n:]))
            if counter:
                return [tid for tid, _ in counter.most_common(self.topk * k)[:k]]
        return []

    def rollback(self, ids: Sequence[int]) -> None:
        """（预留）修正后可对被否决提议降权；当前实现为无操作。"""
        return None


def verify(
    target_forward: Callable[[List[int]], "object"],
    context: Sequence[int],
    draft: Sequence[int],
) -> Tuple[List[int], object]:
    """单次前向并行验证 draft，返回 (被接受 token 序列, 末位 logits)。

    target_forward(ids) -> logits (len(ids), vocab)：对每个位置给出下一 token 分布。
    """
    ctx = list(context)
    draft = list(draft)
    if not draft:
        logits = target_forward(ctx[-1:])
        return [], logits[-1]
    probe = ctx + draft[:-1]
    logits = target_forward(probe)
    greedy = logits.argmax(dim=-1).tolist()
    # logits[j] 是"给定 probe[:j+1] 时的 next 分布"；验证 draft[i] 需取
    # 位置 len(ctx)-1+i（ctx 末位 + i）的预测，否则与 draft 错位。
    base = len(ctx) - 1
    accepted: List[int] = []
    for i, tid in enumerate(draft):
        if greedy[base + i] != tid:
            # 首个不一致处：目标模型的意见优先（回退修正，保每步 >=1 token）
            return accepted, logits[base + i]
        accepted.append(tid)
    # 全部命中：附赠末位 argmax
    return accepted, logits[-1]


def spec_decode_step(
    proposer: DraftProposer,
    target_forward: Callable[[List[int]], "object"],
    context: List[int],
    k: int = 4,
) -> List[int]:
    """一个完整投机步：propose -> verify -> 提交 -> 回填 ngram 表。"""
    draft = proposer.propose(context, k)
    accepted, last_logits = verify(target_forward, context, draft)
    emitted = list(accepted)
    if len(emitted) < len(draft):  # 提前否决：补上目标模型的修正 token
        emitted.append(int(last_logits.argmax()))
    else:  # 全中：bonus token 取末位 logits argmax
        emitted.append(int(last_logits.argmax()))
    proposer.observe(context + emitted)
    return emitted
