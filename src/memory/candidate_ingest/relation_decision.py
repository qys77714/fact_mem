"""
LME 候选事实更新 — relation_decision：pairwise 五类关系 → 桶内聚合 → 确定性写库。

桶内优先级：``EQV`` > ``OSN`` > ``NSO`` > ``CON`` > ``IND``。
不物理删除；``CON`` 产出 ``conflict_update``（ADD primary ``m_new`` + 对 CON 旧条 UPDATE 边）；
``EQV``/``NSO`` 为弱侧新行；``OSN`` 为 ADD primary + 弱侧旧行 ATTACH。

检索结果为空 → 视为仅 IND → ``fresh_primary``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

from memory.base import RetrievedMemory

LmeRelationOutcome = Literal[
    "equivalent_evidence",
    "stronger_primary",
    "weaker_evidence",
    "conflict_update",
    "fresh_primary",
]


@dataclass(frozen=True)
class LmeRelationDecision:
    """Structured plan for one candidate fact after pairwise labels on retrieved primaries."""

    outcome: LmeRelationOutcome
    content: str
    representative_id: Optional[str] = None  # EQV: weak m_new attaches under this primary
    parent_id: Optional[str] = None  # NSO: weak m_new attaches under this primary
    demote_ids: Tuple[str, ...] = ()  # OSN: these primaries become weak (ATTACH to new primary)
    con_update_ids: Tuple[str, ...] = ()  # CON: weak UPDATE edge to the new row id (after ADD)
    reason: str = ""


# Backward-compatible alias
RelationDecision = LmeRelationDecision


def partition_label_list_into_buckets(
    label_list: Sequence[Tuple[RetrievedMemory, str]],
) -> List[List[Tuple[RetrievedMemory, str]]]:
    """
    ``core_algorithm.md`` 中 ``B ← Bucket(m_new, label_list)``。

    当前版本：不解析 entity_anchor / slot / temporal，整表作为**一个桶**。
    """
    if not label_list:
        return []
    return [list(label_list)]


def _ordered_by_label(
    labeled: Sequence[Tuple[RetrievedMemory, str]],
    label: str,
) -> List[RetrievedMemory]:
    seen: set[str] = set()
    out: List[RetrievedMemory] = []
    for m, lab in labeled:
        if lab != label or m.memory_id in seen:
            continue
        seen.add(m.memory_id)
        out.append(m)
    return out


def select_representative(candidates: Sequence[RetrievedMemory]) -> RetrievedMemory:
    """Prefer the first candidate in retrieval order (dense rank)."""
    if not candidates:
        raise ValueError("empty representative set")
    return candidates[0]


def select_primary_parent(nso: Sequence[RetrievedMemory]) -> RetrievedMemory:
    """Among NSO olds (strictly stronger than m_new), prefer the most specific text."""
    if not nso:
        raise ValueError("empty NSO set")
    return max(nso, key=lambda m: len((m.text or "").strip()))


def decide_lme_update_relation_decision_for_bucket(
    m_new: str,
    labeled: Sequence[Tuple[RetrievedMemory, str]],
) -> LmeRelationDecision:
    """
    顺序：EQV > OSN > NSO；若仅 CON → ``conflict_update``；否则 ``fresh_primary``。
    ``con_update_ids`` 收集所有标为 CON 的 ``m_old``，在写库阶段统一打 UPDATE 弱边（见 memory_system）。
    """
    m_new = (m_new or "").strip()
    eqv = _ordered_by_label(labeled, "EQV")
    nso = _ordered_by_label(labeled, "NSO")
    osn = _ordered_by_label(labeled, "OSN")
    con = _ordered_by_label(labeled, "CON")
    con_ids = tuple(m.memory_id for m in con)

    if eqv:
        rep = select_representative(eqv)
        return LmeRelationDecision(
            outcome="equivalent_evidence",
            content=m_new,
            representative_id=rep.memory_id,
            con_update_ids=con_ids,
            reason="eqv",
        )
    if osn:
        return LmeRelationDecision(
            outcome="stronger_primary",
            content=m_new,
            demote_ids=tuple(m.memory_id for m in osn),
            con_update_ids=con_ids,
            reason="osn",
        )
    if nso:
        parent = select_primary_parent(nso)
        return LmeRelationDecision(
            outcome="weaker_evidence",
            content=m_new,
            parent_id=parent.memory_id,
            con_update_ids=con_ids,
            reason="nso",
        )
    if con:
        return LmeRelationDecision(
            outcome="conflict_update",
            content=m_new,
            con_update_ids=con_ids,
            reason="con",
        )
    return LmeRelationDecision(
        outcome="fresh_primary",
        content=m_new,
        con_update_ids=(),
        reason="ind",
    )


def decide_lme_update_relation_decision(
    m_new: str,
    label_list: Sequence[Tuple[RetrievedMemory, str]],
) -> LmeRelationDecision:
    """
    对 ``m_new`` 的全部分桶依次决策；当前仅一块桶，等价于 ``decide_lme_update_relation_decision_for_bucket``。
    """
    buckets = partition_label_list_into_buckets(label_list)
    if not buckets:
        return LmeRelationDecision(
            outcome="fresh_primary",
            content=(m_new or "").strip(),
            con_update_ids=(),
            reason="empty_bucket_list_treat_as_ind",
        )
    return decide_lme_update_relation_decision_for_bucket(m_new, buckets[0])


def decide_relation_decision_for_pairs(
    bucket_pairs: Sequence[Tuple[RetrievedMemory, str]],
    m_new: str = "",
) -> LmeRelationDecision:
    """兼容旧名：单桶决策。"""
    return decide_lme_update_relation_decision_for_bucket(m_new, bucket_pairs)
