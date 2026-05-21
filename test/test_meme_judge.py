from eval.meme_judge import (
    _er_substring_match,
    aggregate_meme_metrics,
    classify_trivial_pass,
    counts_toward_meme_score,
)


def test_er_substring_match():
    ok = _er_substring_match("hello world", "The answer is hello world today")
    assert ok["u_pass"] is True
    bad = _er_substring_match("hello world", "I don't know")
    assert bad["u_pass"] is False


def test_classify_trivial_pass():
    assert classify_trivial_pass(
        task_type="Cas",
        entity_key="medication",
        before_pass_by_entity={"medication": True},
        after_u_pass=True,
    ) == "real"
    assert classify_trivial_pass(
        task_type="Del",
        entity_key="allergy",
        before_pass_by_entity={"allergy": False},
        after_u_pass=True,
    ) == "trivial"
    assert classify_trivial_pass(
        task_type="ER",
        entity_key="x",
        before_pass_by_entity={},
        after_u_pass=True,
    ) is None


def test_counts_toward_meme_score_cas_abs_del_after():
    assert counts_toward_meme_score(
        {"phase": "after", "question_type": "Cas", "u_pass": True, "pass_type": "real"}
    )
    assert not counts_toward_meme_score(
        {"phase": "after", "question_type": "Del", "u_pass": True, "pass_type": "trivial"}
    )


def test_aggregate_meme_metrics():
    rows = [
        {"phase": "before", "question_type": "Cas", "u_pass": True, "judge_api_failed": False},
        {"phase": "after", "question_type": "Cas", "u_pass": True, "pass_type": "real", "judge_api_failed": False},
        {"phase": "before", "question_type": "Del", "u_pass": False, "judge_api_failed": False},
        {"phase": "after", "question_type": "Del", "u_pass": True, "pass_type": "trivial", "judge_api_failed": False},
        {"phase": "after", "question_type": "ER", "u_pass": True, "pass_type": None, "judge_api_failed": False},
    ]
    m = aggregate_meme_metrics(rows)
    assert m["before_total"] == 2
    assert m["before_pass"] == 1
    assert m["after_total"] == 3
    assert m["after_pass"] == 2  # Cas real + ER; Del trivial excluded
    assert m["after_pass_raw"] == 3
    assert m["meme_score"] == 2 / 3  # after-only denominator
    assert m["meme_score_raw"] == 3 / 3
    assert m["meme_score_judge_totals"] == 4 / 5  # judge.py style
    assert m["trivial_analysis"]["Cas"]["real_pass"] == 1
    assert m["trivial_analysis"]["Del"]["trivial_pass"] == 1
