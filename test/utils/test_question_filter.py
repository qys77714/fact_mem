"""Tests for utils.question_filter (no heavy pipeline imports)."""

from benchmark.base import QuestionItem
from utils.question_filter import (
    filter_jsonl_rows_by_question_type,
    filter_question_items,
    parse_question_types_arg,
    stratified_sample_by_question_type,
)


def test_parse_question_types_arg():
    assert parse_question_types_arg(None) is None
    assert parse_question_types_arg("  ") is None
    assert parse_question_types_arg("a,b") == {"a", "b"}


def test_filter_jsonl_rows():
    rows = [
        {"history_name": "h", "question_type": "knowledge-update"},
        {"history_name": "h2", "question_type": "multi-session"},
    ]
    assert len(filter_jsonl_rows_by_question_type(rows, {"knowledge-update"})) == 1


def test_filter_question_items():
    items = [
        QuestionItem(question="a", answer="1", question_time="", question_type="x"),
        QuestionItem(question="b", answer="2", question_time="", question_type="y"),
    ]
    assert len(filter_question_items(items, {"x"})) == 1


def test_stratified_sample_keeps_all_when_n_large():
    items = [
        (("h", "1"), "a"),
        (("h", "2"), "a"),
        (("h", "3"), "b"),
    ]
    out = stratified_sample_by_question_type(items, 100, seed=0)
    assert len(out) == 3


def test_stratified_sample_proportions_and_determinism():
    # 50 A, 50 B — take 20 → expect 10+10 with fixed seed
    items = [(("h", str(i)), "a" if i < 50 else "b") for i in range(100)]
    for seed in (0, 1, 42):
        s = stratified_sample_by_question_type(items, 20, seed)
        assert len(s) == 20
        a_keys = {("h", str(i)) for i in range(50)}
        b_keys = {("h", str(i)) for i in range(50, 100)}
        assert len(s & a_keys) == 10
        assert len(s & b_keys) == 10
    s1 = stratified_sample_by_question_type(items, 20, seed=7)
    s2 = stratified_sample_by_question_type(items, 20, seed=7)
    assert s1 == s2


def test_stratified_sample_unknown_type():
    items = [(("h", "1"), None), (("h", "2"), "a")]
    s = stratified_sample_by_question_type(items, 1, seed=0)
    assert len(s) == 1
