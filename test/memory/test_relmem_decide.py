"""Unit tests for RelMem deterministic decision (no LLM / DB)."""
from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from memory.base import RetrievedMemory
from memory.relmem.decide import decide_relmem


def _m(mid: str, text: str) -> RetrievedMemory:
    return RetrievedMemory(
        memory_id=mid,
        text=text,
        source_index="s",
        time="t",
        score=0.0,
        metadata={},
    )


def test_decide_eqv_attaches_evidence():
    a = _m("a", "Alice lives in NYC")
    plan = decide_relmem("Alice lives in New York", [(a, "EQV")])
    assert plan.outcome == "equivalent_evidence"
    assert plan.representative_id == "a"
    assert plan.invalidate_ids == ()


def test_decide_osn_promotes_and_demotes():
    old = _m("o", "Alice lives in France")
    plan = decide_relmem("Alice lives in Paris", [(old, "OSN")])
    assert plan.outcome == "stronger_primary"
    assert plan.demote_ids == ("o",)
    assert plan.invalidate_ids == ()


def test_decide_nso_evidence_under_parent():
    old = _m("o", "Alice lives in Paris")
    plan = decide_relmem("Alice lives in France", [(old, "NSO")])
    assert plan.outcome == "weaker_evidence"
    assert plan.parent_id == "o"


def test_decide_con_invalidates_then_fresh_primary():
    c = _m("c", "Alice lives in London")
    plan = decide_relmem("Alice lives in Paris", [(c, "CON")])
    assert plan.outcome == "fresh_primary"
    assert plan.invalidate_ids == ("c",)


def test_decide_priority_eqv_over_osn():
    o = _m("o", "x")
    plan = decide_relmem("y", [(o, "OSN"), (o, "EQV")])
    assert plan.outcome == "equivalent_evidence"
