"""Tests for Judge 阶段产物隔离：--output / --metrics-output / 纯函数抽取.

TDD：这些测试先于实现编写，实现落地前应全部失败（ImportError 或 AttributeError）。
不调用 LLM/API：所有涉及 evaluate_one_input 的测试都通过 monkeypatch 替换
``pipeline_lme_evaluate.evaluate`` 为确定性 stub。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

import pipeline_lme_evaluate as plev
from pipeline_lme_evaluate import (
    JudgeOutcome,
    evaluate_one_input,
    merge_judge_outcomes,
    parse_args,
    write_json_atomic,
    write_jsonl_atomic,
)


def _row(history_name: str, question_id: str, **extra: Any) -> Dict[str, Any]:
    base = {
        "history_name": history_name,
        "question_id": question_id,
        "question": f"q-{question_id}",
        "answer": f"a-{question_id}",
        "model_answer": f"m-{question_id}",
        "question_type": extra.pop("question_type", "single-session-user"),
    }
    base.update(extra)
    return base


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# 1. merge_judge_outcomes：纯函数，无 IO
# ---------------------------------------------------------------------------


class TestMergeJudgeOutcomes:
    def test_full_run_merge_keys_false_adds_fields_in_order(self):
        samples_all = [_row("h1", "q1"), _row("h1", "q2"), _row("h1", "q3")]
        samples = samples_all
        outcomes: List[JudgeOutcome] = [
            JudgeOutcome(api_failed=False, is_correct=True),
            JudgeOutcome(api_failed=True, is_correct=None),
            JudgeOutcome(api_failed=False, is_correct=False),
        ]
        merged = merge_judge_outcomes(samples_all, samples, outcomes, merge_keys=False)

        assert [r["question_id"] for r in merged] == ["q1", "q2", "q3"]
        assert merged[0]["is_correct"] is True
        assert merged[0]["judge_api_failed"] is False
        assert merged[1]["is_correct"] is None
        assert merged[1]["judge_api_failed"] is True
        assert merged[2]["is_correct"] is False
        assert merged[2]["judge_api_failed"] is False
        # 原有字段全部保留
        for orig, out in zip(samples_all, merged):
            for k, v in orig.items():
                assert out[k] == v

    def test_subset_merge_keys_true_preserves_unjudged_rows(self):
        samples_all = [_row("h1", "q1"), _row("h1", "q2"), _row("h1", "q3")]
        samples = [samples_all[0], samples_all[2]]  # only q1, q3 judged
        outcomes: List[JudgeOutcome] = [
            JudgeOutcome(api_failed=False, is_correct=True),
            JudgeOutcome(api_failed=False, is_correct=False),
        ]
        merged = merge_judge_outcomes(samples_all, samples, outcomes, merge_keys=True)

        # order matches samples_all, not samples
        assert [r["question_id"] for r in merged] == ["q1", "q2", "q3"]

        assert merged[0]["is_correct"] is True
        assert merged[0]["judge_api_failed"] is False

        # unjudged row q2: no result fields added at all
        assert "is_correct" not in merged[1]
        assert "judge_api_failed" not in merged[1]
        # but all original fields preserved
        assert merged[1]["question"] == "q-q2"
        assert merged[1]["model_answer"] == "m-q2"

        assert merged[2]["is_correct"] is False
        assert merged[2]["judge_api_failed"] is False

    def test_subset_does_not_overwrite_unjudged_existing_result_fields(self):
        """未被 Judge 的行如已带有旧的 is_correct/judge_api_failed，必须原样保留不被覆盖。"""
        stale_row = _row("h1", "q2", is_correct=True, judge_api_failed=False)
        samples_all = [_row("h1", "q1"), stale_row, _row("h1", "q3")]
        samples = [samples_all[0], samples_all[2]]
        outcomes: List[JudgeOutcome] = [
            JudgeOutcome(api_failed=False, is_correct=False),
            JudgeOutcome(api_failed=False, is_correct=False),
        ]
        merged = merge_judge_outcomes(samples_all, samples, outcomes, merge_keys=True)
        assert merged[1]["is_correct"] is True
        assert merged[1]["judge_api_failed"] is False

    def test_overwrites_existing_result_fields_for_judged_rows(self):
        stale_row = _row("h1", "q1", is_correct=True, judge_api_failed=True)
        samples_all = [stale_row]
        samples = [stale_row]
        outcomes: List[JudgeOutcome] = [JudgeOutcome(api_failed=False, is_correct=False)]
        merged = merge_judge_outcomes(samples_all, samples, outcomes, merge_keys=False)
        assert merged[0]["is_correct"] is False
        assert merged[0]["judge_api_failed"] is False

    def test_does_not_mutate_input_rows(self):
        row = _row("h1", "q1")
        snapshot = dict(row)
        samples_all = [row]
        outcomes: List[JudgeOutcome] = [JudgeOutcome(api_failed=False, is_correct=True)]
        merge_judge_outcomes(samples_all, samples_all, outcomes, merge_keys=False)
        assert row == snapshot
        assert "is_correct" not in row

    def test_raises_on_length_mismatch(self):
        samples_all = [_row("h1", "q1"), _row("h1", "q2")]
        outcomes: List[JudgeOutcome] = [JudgeOutcome(api_failed=False, is_correct=True)]
        with pytest.raises(ValueError):
            merge_judge_outcomes(samples_all, samples_all, outcomes, merge_keys=False)


# ---------------------------------------------------------------------------
# 2. write_jsonl_atomic / write_json_atomic
# ---------------------------------------------------------------------------


class TestWriteAtomicHelpers:
    def test_write_jsonl_atomic_creates_parent_dir_and_content(self, tmp_path: Path):
        target = tmp_path / "nested" / "out.jsonl"
        rows = [{"a": 1}, {"a": 2}]
        write_jsonl_atomic(target, rows)
        assert target.exists()
        assert _read_jsonl(target) == rows

    def test_write_jsonl_atomic_no_leftover_tmp_files(self, tmp_path: Path):
        target = tmp_path / "out.jsonl"
        write_jsonl_atomic(target, [{"a": 1}])
        leftovers = [p for p in tmp_path.iterdir() if p != target]
        assert leftovers == []

    def test_write_jsonl_atomic_overwrites_existing_file(self, tmp_path: Path):
        target = tmp_path / "out.jsonl"
        write_jsonl_atomic(target, [{"a": 1}, {"a": 2}, {"a": 3}])
        write_jsonl_atomic(target, [{"b": 1}])
        assert _read_jsonl(target) == [{"b": 1}]

    def test_write_jsonl_atomic_failure_leaves_original_untouched(self, tmp_path: Path):
        target = tmp_path / "out.jsonl"
        original_rows = [{"a": 1}]
        write_jsonl_atomic(target, original_rows)

        class Unserializable:
            pass

        with pytest.raises(TypeError):
            write_jsonl_atomic(target, [{"a": 1}, {"bad": Unserializable()}])

        # original file must remain intact, no leftover tmp files
        assert _read_jsonl(target) == original_rows
        leftovers = [p for p in tmp_path.iterdir() if p != target]
        assert leftovers == []

    def test_write_json_atomic_creates_parent_dir_and_content(self, tmp_path: Path):
        target = tmp_path / "nested" / "metrics.json"
        obj = {"n": 3, "overall_accuracy": 0.5}
        write_json_atomic(target, obj)
        assert json.loads(target.read_text(encoding="utf-8")) == obj

    def test_write_json_atomic_overwrites_old_value(self, tmp_path: Path):
        target = tmp_path / "metrics.json"
        write_json_atomic(target, {"n": 1})
        write_json_atomic(target, {"n": 2, "extra": True})
        assert json.loads(target.read_text(encoding="utf-8")) == {"n": 2, "extra": True}

    def test_write_json_atomic_no_leftover_tmp_files(self, tmp_path: Path):
        target = tmp_path / "metrics.json"
        write_json_atomic(target, {"n": 1})
        leftovers = [p for p in tmp_path.iterdir() if p != target]
        assert leftovers == []


# ---------------------------------------------------------------------------
# 3. parser: --output 与 --write_back 互斥
# ---------------------------------------------------------------------------


class TestParserMutualExclusion:
    def test_output_and_write_back_together_raises(self):
        argv = [
            "--input",
            "foo.jsonl",
            "--judge_model",
            "m",
            "--write_back",
            "--output",
            "bar.jsonl",
        ]
        with pytest.raises(SystemExit):
            parse_args(argv)

    def test_output_alone_is_accepted(self):
        argv = [
            "--input",
            "foo.jsonl",
            "--judge_model",
            "m",
            "--output",
            "bar.jsonl",
        ]
        args = parse_args(argv)
        assert args.output == "bar.jsonl"
        assert args.write_back is False

    def test_write_back_alone_is_accepted(self):
        argv = ["--input", "foo.jsonl", "--judge_model", "m", "--write_back"]
        args = parse_args(argv)
        assert args.write_back is True
        assert args.output is None

    def test_metrics_output_default_none(self):
        argv = ["--input", "foo.jsonl", "--judge_model", "m"]
        args = parse_args(argv)
        assert args.metrics_output is None

    def test_metrics_output_parsed(self):
        argv = [
            "--input",
            "foo.jsonl",
            "--judge_model",
            "m",
            "--metrics-output",
            "metrics.json",
        ]
        args = parse_args(argv)
        assert args.metrics_output == "metrics.json"


# ---------------------------------------------------------------------------
# 4. evaluate_one_input 集成：mock evaluate()，不调用 LLM/API
# ---------------------------------------------------------------------------


def _make_args(**overrides: Any) -> argparse.Namespace:
    defaults = dict(
        judge_model="dummy",
        benchmark=None,
        use_cot=False,
        judge_oqa_template="pipeline_eval_oqa.jinja",
        judge_mcq_template="pipeline_eval_mcq.jinja",
        judge_system_template="pipeline_eval_system.jinja",
        judge_qwen_thinking=False,
        max_concurrency=20,
        max_new_tokens=2048,
        append_result=None,
        csv=None,
        write_back=False,
        output=None,
        metrics_output=None,
        question_types=None,
        print_one_sample=False,
        stratified_sample_n=0,
        stratified_sample_seed=42,
        files_concurrency=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def stub_evaluate(monkeypatch):
    """将 pipeline_lme_evaluate.evaluate 替换为确定性 stub：按输入顺序交替 True/False。"""

    async def fake_evaluate(samples, **kwargs):
        outcomes = [
            JudgeOutcome(api_failed=False, is_correct=(i % 2 == 0))
            for i in range(len(samples))
        ]
        judged = len(outcomes)
        correct = sum(1 for o in outcomes if o["is_correct"])
        metrics = {
            "n_samples": len(samples),
            "api_failure_count": 0,
            "judged_count": judged,
            "overall_accuracy": (correct / judged) if judged else 0.0,
            "per_type": {},
        }
        return metrics, outcomes

    monkeypatch.setattr(plev, "evaluate", fake_evaluate)
    return fake_evaluate


class TestEvaluateOneInputOutputIsolation:
    def test_output_writes_full_set_and_leaves_input_untouched(self, tmp_path, stub_evaluate):
        input_path = tmp_path / "pred.jsonl"
        rows = [_row("h1", "q1"), _row("h1", "q2"), _row("h1", "q3")]
        _write_jsonl(input_path, rows)
        original_bytes = input_path.read_bytes()

        output_path = tmp_path / "judged" / "pred.judged.jsonl"
        args = _make_args(output=str(output_path))

        _run(evaluate_one_input(str(input_path), args, print_one_sample=False))

        # input file must remain byte-identical
        assert input_path.read_bytes() == original_bytes

        out_rows = _read_jsonl(output_path)
        assert [r["question_id"] for r in out_rows] == ["q1", "q2", "q3"]
        assert out_rows[0]["is_correct"] is True
        assert out_rows[1]["is_correct"] is False
        assert out_rows[2]["is_correct"] is True
        for r in out_rows:
            assert "judge_api_failed" in r

    def test_output_with_question_types_filter_preserves_unjudged_rows(
        self, tmp_path, stub_evaluate
    ):
        input_path = tmp_path / "pred.jsonl"
        rows = [
            _row("h1", "q1", question_type="single-session-user"),
            _row("h1", "q2", question_type="multi-session"),
            _row("h1", "q3", question_type="single-session-user"),
        ]
        _write_jsonl(input_path, rows)
        original_bytes = input_path.read_bytes()

        output_path = tmp_path / "out.jsonl"
        args = _make_args(output=str(output_path), question_types="single-session-user")

        _run(evaluate_one_input(str(input_path), args, print_one_sample=False))

        assert input_path.read_bytes() == original_bytes

        out_rows = _read_jsonl(output_path)
        assert [r["question_id"] for r in out_rows] == ["q1", "q2", "q3"]
        # q1, q3 judged (indices 0, 1 within the filtered subset -> True, False)
        assert out_rows[0]["is_correct"] is True
        assert out_rows[2]["is_correct"] is False
        # q2 filtered out of judging: no result fields added, other fields intact
        assert "is_correct" not in out_rows[1]
        assert "judge_api_failed" not in out_rows[1]
        assert out_rows[1]["question_type"] == "multi-session"

    def test_write_back_still_replaces_input_in_place(self, tmp_path, stub_evaluate):
        input_path = tmp_path / "pred.jsonl"
        rows = [_row("h1", "q1"), _row("h1", "q2")]
        _write_jsonl(input_path, rows)

        args = _make_args(write_back=True)
        _run(evaluate_one_input(str(input_path), args, print_one_sample=False))

        out_rows = _read_jsonl(input_path)
        assert [r["question_id"] for r in out_rows] == ["q1", "q2"]
        assert out_rows[0]["is_correct"] is True
        assert out_rows[1]["is_correct"] is False

    def test_metrics_output_writes_single_overwritable_json_object(
        self, tmp_path, stub_evaluate
    ):
        input_path = tmp_path / "pred.jsonl"
        rows = [_row("h1", "q1"), _row("h1", "q2")]
        _write_jsonl(input_path, rows)

        metrics_path = tmp_path / "metrics" / "run.json"
        append_result_path = tmp_path / "eval_judge.json"
        args = _make_args(
            metrics_output=str(metrics_path), append_result=str(append_result_path)
        )

        _run(evaluate_one_input(str(input_path), args, print_one_sample=False))

        # metrics-output: single JSON object (not an array, not jsonl)
        metrics_obj = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert isinstance(metrics_obj, dict)
        assert metrics_obj["n"] == 2
        assert metrics_obj["eval_type"] == "judge"

        # append_result must still work (coexistence)
        append_data = json.loads(append_result_path.read_text(encoding="utf-8"))
        assert isinstance(append_data, list)
        assert len(append_data) == 1
        assert append_data[0]["n"] == 2

        # run again: metrics-output must be overwritten, not appended to
        _run(evaluate_one_input(str(input_path), args, print_one_sample=False))
        metrics_obj_2 = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert isinstance(metrics_obj_2, dict)

        # append_result on the other hand keeps growing (unchanged existing behavior)
        append_data_2 = json.loads(append_result_path.read_text(encoding="utf-8"))
        assert len(append_data_2) == 2

    def test_no_metrics_output_keeps_default_eval_judge_json_append_behavior(
        self, tmp_path, stub_evaluate
    ):
        input_path = tmp_path / "pred.jsonl"
        rows = [_row("h1", "q1")]
        _write_jsonl(input_path, rows)

        args = _make_args()
        _run(evaluate_one_input(str(input_path), args, print_one_sample=False))

        default_append_path = input_path.resolve().parent / "eval_judge.json"
        assert default_append_path.exists()
        data = json.loads(default_append_path.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) == 1

    def test_neither_output_nor_write_back_does_not_touch_input(self, tmp_path, stub_evaluate):
        input_path = tmp_path / "pred.jsonl"
        rows = [_row("h1", "q1")]
        _write_jsonl(input_path, rows)
        original_bytes = input_path.read_bytes()

        args = _make_args()
        _run(evaluate_one_input(str(input_path), args, print_one_sample=False))

        assert input_path.read_bytes() == original_bytes
