"""Unit tests for LME relation_decision bucket aggregation."""

from memory.base import RetrievedMemory
from memory.candidate_ingest.relation_decision import (
    decide_lme_update_relation_decision,
    decide_lme_update_relation_decision_for_bucket,
)


def _mem(mid: str, text: str = "x") -> RetrievedMemory:
    return RetrievedMemory(
        memory_id=mid,
        text=text,
        source_index="s",
        time="t",
        score=0.0,
        metadata={},
    )


def test_empty_label_list_is_fresh_primary():
    p = decide_lme_update_relation_decision("new fact", [])
    assert p.outcome == "fresh_primary"
    assert p.con_update_ids == ()


def test_only_ind():
    p = decide_lme_update_relation_decision_for_bucket(
        "n",
        [(_mem("a"), "IND")],
    )
    assert p.outcome == "fresh_primary"
    assert p.reason == "ind"


def test_eqv_wins_over_osn():
    m1, m2 = _mem("1"), _mem("2")
    p = decide_lme_update_relation_decision_for_bucket(
        "n",
        [
            (m1, "EQV"),
            (m2, "OSN"),
        ],
    )
    assert p.outcome == "equivalent_evidence"
    assert p.representative_id == "1"


def test_osn_stronger_primary():
    m1 = _mem("1", "old specific")
    p = decide_lme_update_relation_decision_for_bucket("weaker new", [(m1, "OSN")])
    assert p.outcome == "stronger_primary"
    assert p.demote_ids == ("1",)


def test_nso_weaker_evidence():
    m1 = _mem("1", "strong old text here")
    p = decide_lme_update_relation_decision_for_bucket("weak", [(m1, "NSO")])
    assert p.outcome == "weaker_evidence"
    assert p.parent_id == "1"


def test_con_conflict_update():
    m1 = _mem("1")
    p = decide_lme_update_relation_decision_for_bucket("conflict", [(m1, "CON")])
    assert p.outcome == "conflict_update"
    assert p.con_update_ids == ("1",)


def test_con_ids_carried_when_osn_wins():
    m_con = _mem("c")
    m_osn = _mem("o")
    p = decide_lme_update_relation_decision_for_bucket(
        "n",
        [
            (m_con, "CON"),
            (m_osn, "OSN"),
        ],
    )
    assert p.outcome == "stronger_primary"
    assert set(p.con_update_ids) == {"c"}
    assert set(p.demote_ids) == {"o"}
