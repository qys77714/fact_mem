"""Optional filtering of benchmark items / jsonl rows by ``question_type``."""

import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from benchmark.base import QuestionItem


def answer_row_key(row: Dict[str, Any]) -> Tuple[str, str]:
    """Match ``pipeline_generate`` resume keys: (history_name, question_id)."""
    h = str(row.get("history_name", ""))
    qid = row.get("question_id")
    return (h, str(qid if qid is not None else h))


def parse_question_types_arg(arg: Optional[str]) -> Optional[Set[str]]:
    """
    Parse ``--question-types`` CLI value: comma-separated labels, stripped.
    Empty / omitted means no filter.
    """
    if arg is None:
        return None
    s = str(arg).strip()
    if not s:
        return None
    out = {p.strip() for p in s.split(",") if p.strip()}
    return out or None


def filter_question_items(
    items: List[QuestionItem],
    types: Optional[Set[str]],
) -> List[QuestionItem]:
    if not types:
        return items
    return [q for q in items if q.question_type in types]


def filter_jsonl_rows_by_question_type(
    rows: List[Dict[str, Any]],
    types: Optional[Set[str]],
) -> List[Dict[str, Any]]:
    if not types:
        return rows
    return [r for r in rows if r.get("question_type") in types]


def _dedupe_question_keys(
    items: List[Tuple[Tuple[str, str], Optional[str]]],
) -> List[Tuple[Tuple[str, str], Optional[str]]]:
    """Keep first occurrence per (history_name, question_id)."""
    seen: Dict[Tuple[str, str], Optional[str]] = {}
    for key, qt in items:
        if key not in seen:
            seen[key] = qt
    return [(k, seen[k]) for k in seen]


def stratified_sample_by_question_type(
    items: List[Tuple[Tuple[str, str], Optional[str]]],
    n: int,
    seed: int,
) -> Set[Tuple[str, str]]:
    """
    Sample ``n`` question keys so that per-``question_type`` counts follow the
    largest-remainder allocation of the full pool (approximate proportions).

    Used for multi-category QA workloads: single-hop / multi-hop / temporal / …
    """
    if n <= 0:
        return {k for k, _ in _dedupe_question_keys(items)}
    deduped = _dedupe_question_keys(items)
    if len(deduped) <= n:
        return {k for k, _ in deduped}

    buckets: Dict[str, List[Tuple[str, str]]] = {}
    for (key, qt) in deduped:
        label = str(qt) if qt else "unknown"
        buckets.setdefault(label, []).append(key)

    types_sorted = sorted(buckets.keys())
    counts = {t: len(buckets[t]) for t in types_sorted}
    total = sum(counts.values())
    assert total == len(deduped)

    # Largest remainder allocation for integer targets per type
    ideals = {t: n * counts[t] / total for t in types_sorted}
    floors = {t: int(math.floor(ideals[t])) for t in types_sorted}
    rem = n - sum(floors.values())
    targets = dict(floors)
    frac_order = sorted(
        types_sorted,
        key=lambda t: (-(ideals[t] - floors[t]), t),
    )
    for i in range(rem):
        targets[frac_order[i]] += 1

    rng = random.Random(int(seed))
    selected: Set[Tuple[str, str]] = set()
    for t in types_sorted:
        pool = sorted(buckets[t])
        k = min(targets[t], len(pool))
        if k <= 0:
            continue
        picked = rng.sample(pool, k)
        selected.update(picked)

    # If rounding left us short (e.g. type had fewer rows than target), top up
    if len(selected) < n:
        remaining = [k for k, _ in deduped if k not in selected]
        remaining.sort()
        need = n - len(selected)
        extra = rng.sample(remaining, min(need, len(remaining)))
        selected.update(extra)

    return selected
