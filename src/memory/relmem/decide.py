"""Deterministic RelMem decision (bucket-local aggregation), see 02_design/architecture/system_overview.md."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

from memory.base import RetrievedMemory


@dataclass(frozen=True)
class RelMemDecision:
    """Planned write semantics for one new fact after pairwise labels on retrieved primaries."""

    invalidate_ids: Tuple[str, ...]
    outcome: Literal["equivalent_evidence", "stronger_primary", "weaker_evidence", "fresh_primary"]
    content: str
    representative_id: Optional[str] = None  # EQV: attach m_new as evidence to this primary
    parent_id: Optional[str] = None  # NSO: attach m_new as evidence under this primary
    demote_ids: Tuple[str, ...] = ()  # OSN: these primaries become evidence under the new primary


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
    """Prefer the first candidate in retrieval order (caller passes ordered lists)."""
    if not candidates:
        raise ValueError("empty representative set")
    return candidates[0]


def select_primary_parent(nso: Sequence[RetrievedMemory]) -> RetrievedMemory:
    """Among NSO olds (stronger / more specific than m_new), prefer the most specific text."""
    if not nso:
        raise ValueError("empty NSO set")
    return max(nso, key=lambda m: len((m.text or "").strip()))


def decide_relmem(
    m_new: str,
    labeled: Sequence[Tuple[RetrievedMemory, str]],
) -> RelMemDecision:
    """
    Pairwise labels are intermediate evidence; order follows system_overview:
    invalidate CON, then EQV > OSN > NSO > default ADD primary.
    """
    m_new = (m_new or "").strip()
    eqv = _ordered_by_label(labeled, "EQV")
    nso = _ordered_by_label(labeled, "NSO")
    osn = _ordered_by_label(labeled, "OSN")
    con = _ordered_by_label(labeled, "CON")

    inv = tuple(m.memory_id for m in con)

    if eqv:
        rep = select_representative(eqv)
        return RelMemDecision(
            invalidate_ids=inv,
            outcome="equivalent_evidence",
            content=m_new,
            representative_id=rep.memory_id,
        )
    if osn:
        return RelMemDecision(
            invalidate_ids=inv,
            outcome="stronger_primary",
            content=m_new,
            demote_ids=tuple(m.memory_id for m in osn),
        )
    if nso:
        parent = select_primary_parent(nso)
        return RelMemDecision(
            invalidate_ids=inv,
            outcome="weaker_evidence",
            content=m_new,
            parent_id=parent.memory_id,
        )
    return RelMemDecision(invalidate_ids=inv, outcome="fresh_primary", content=m_new)
