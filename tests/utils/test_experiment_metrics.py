"""Tests for src/utils/experiment_metrics.py (重复实验 Judge metrics 聚合).

TDD：本测试文件先于实现编写，实现落地前应全部失败（ImportError）。
"""

from __future__ import annotations

import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

from utils.experiment_metrics import (
    SCHEMA_VERSION,
    aggregate_metrics,
    atomic_write_json,
    load_metrics,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLI_SCRIPT = _REPO_ROOT / "script" / "aggregate_experiment_metrics.py"


def _write_metrics(
    path: Path,
    *,
    overall_accuracy=0.5,
    judged_count=100,
    api_failure_count=2,
    benchmark="lme_s",
    judge_model="qwen3-max",
    per_type=None,
    extra=None,
) -> Path:
    data = {
        "overall_accuracy": overall_accuracy,
        "judged_count": judged_count,
        "api_failure_count": api_failure_count,
        "benchmark": benchmark,
        "judge_model": judge_model,
        "per_type": per_type if per_type is not None else {"single-session-user": {"accuracy": 0.5}},
    }
    if extra:
        data.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# load_metrics
# ---------------------------------------------------------------------------


class TestLoadMetrics:
    def test_loads_a_valid_metrics_object(self, tmp_path):
        path = _write_metrics(tmp_path / "metrics.json", overall_accuracy=0.75)
        data = load_metrics(path)
        assert data["overall_accuracy"] == 0.75
        assert data["benchmark"] == "lme_s"
        assert data["judge_model"] == "qwen3-max"

    def test_missing_file_raises_with_path(self, tmp_path):
        missing = tmp_path / "does_not_exist.json"
        with pytest.raises(ValueError) as excinfo:
            load_metrics(missing)
        assert str(missing) in str(excinfo.value)

    def test_invalid_json_raises_with_path(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ValueError) as excinfo:
            load_metrics(bad)
        assert str(bad) in str(excinfo.value)

    def test_non_object_top_level_raises_with_path(self, tmp_path):
        arr = tmp_path / "array.json"
        arr.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError) as excinfo:
            load_metrics(arr)
        assert str(arr) in str(excinfo.value)

    def test_does_not_mutate_the_file(self, tmp_path):
        path = _write_metrics(tmp_path / "metrics.json")
        before = path.read_text(encoding="utf-8")
        load_metrics(path)
        after = path.read_text(encoding="utf-8")
        assert before == after


# ---------------------------------------------------------------------------
# aggregate_metrics: rejections
# ---------------------------------------------------------------------------


class TestAggregateMetricsRejections:
    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="(?i)empty|no input|at least one"):
            aggregate_metrics([])

    def test_rejects_missing_overall_accuracy_with_path(self, tmp_path):
        good = _write_metrics(tmp_path / "good.json")
        bad_path = tmp_path / "missing_field.json"
        bad_path.write_text(
            json.dumps(
                {
                    "judged_count": 10,
                    "api_failure_count": 0,
                    "benchmark": "lme_s",
                    "judge_model": "qwen3-max",
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError) as excinfo:
            aggregate_metrics([good, bad_path])
        message = str(excinfo.value)
        assert str(bad_path.resolve()) in message
        assert "overall_accuracy" in message

    def test_rejects_non_numeric_overall_accuracy_with_path(self, tmp_path):
        good = _write_metrics(tmp_path / "good.json")
        bad_path = _write_metrics(tmp_path / "bad.json", overall_accuracy="not-a-number")
        with pytest.raises(ValueError) as excinfo:
            aggregate_metrics([good, bad_path])
        message = str(excinfo.value)
        assert str(bad_path.resolve()) in message
        assert "overall_accuracy" in message

    def test_rejects_nan_overall_accuracy(self, tmp_path):
        bad_path = tmp_path / "nan.json"
        bad_path.write_text(
            json.dumps(
                {
                    "overall_accuracy": None,
                    "judged_count": 10,
                    "api_failure_count": 0,
                    "benchmark": "lme_s",
                    "judge_model": "qwen3-max",
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError) as excinfo:
            aggregate_metrics([bad_path])
        assert "overall_accuracy" in str(excinfo.value)

    def test_rejects_inconsistent_benchmark(self, tmp_path):
        a = _write_metrics(tmp_path / "a.json", benchmark="lme_s")
        b = _write_metrics(tmp_path / "b.json", benchmark="lme_hybrid")
        with pytest.raises(ValueError) as excinfo:
            aggregate_metrics([a, b])
        message = str(excinfo.value)
        assert "benchmark" in message
        assert str(a.resolve()) in message
        assert str(b.resolve()) in message
        assert "lme_s" in message and "lme_hybrid" in message

    def test_rejects_inconsistent_judge_model(self, tmp_path):
        a = _write_metrics(tmp_path / "a.json", judge_model="qwen3-max")
        b = _write_metrics(tmp_path / "b.json", judge_model="gpt-4o-mini")
        with pytest.raises(ValueError) as excinfo:
            aggregate_metrics([a, b])
        message = str(excinfo.value)
        assert "judge_model" in message
        assert str(a.resolve()) in message
        assert str(b.resolve()) in message

    def test_rejects_missing_benchmark_field_with_path(self, tmp_path):
        bad_path = tmp_path / "no_benchmark.json"
        bad_path.write_text(
            json.dumps(
                {
                    "overall_accuracy": 0.5,
                    "judged_count": 10,
                    "api_failure_count": 0,
                    "judge_model": "qwen3-max",
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError) as excinfo:
            aggregate_metrics([bad_path])
        message = str(excinfo.value)
        assert str(bad_path.resolve()) in message
        assert "benchmark" in message

    def test_does_not_treat_attempt_directory_names_as_repeats(self, tmp_path):
        """只以显式 paths 为准；同目录下多余的兄弟文件不得被隐式纳入。"""
        attempt_dir = tmp_path / "attempts" / "20260101-abcd"
        a = _write_metrics(attempt_dir / "metrics.json", overall_accuracy=0.2)
        # A sibling file that looks like another repeat but is NOT passed in explicitly.
        _write_metrics(attempt_dir.parent / "20260102-efgh" / "metrics.json", overall_accuracy=0.9)

        result = aggregate_metrics([a])
        assert result["n_repeats"] == 1
        assert result["overall_accuracy"]["values"] == [0.2]


# ---------------------------------------------------------------------------
# aggregate_metrics: correctness
# ---------------------------------------------------------------------------


class TestAggregateMetricsCorrectness:
    def test_three_repeats_exact_mean_and_sample_std(self, tmp_path):
        paths = [
            _write_metrics(
                tmp_path / f"r{i}.json",
                overall_accuracy=acc,
                judged_count=jc,
                api_failure_count=fc,
            )
            for i, (acc, jc, fc) in enumerate([(0.1, 90, 1), (0.2, 100, 2), (0.3, 110, 3)])
        ]

        result = aggregate_metrics(paths)

        expected_values = [0.1, 0.2, 0.3]
        expected_mean = statistics.mean(expected_values)
        expected_std = statistics.stdev(expected_values)

        assert result["schema_version"] == SCHEMA_VERSION
        assert result["benchmark"] == "lme_s"
        assert result["judge_model"] == "qwen3-max"
        assert result["n_repeats"] == 3
        assert result["overall_accuracy"]["values"] == pytest.approx(expected_values)
        assert result["overall_accuracy"]["mean"] == pytest.approx(expected_mean)
        assert result["overall_accuracy"]["sample_std"] == pytest.approx(expected_std)

        assert result["judged_count"]["sum"] == 300
        assert result["judged_count"]["mean"] == pytest.approx(100.0)
        assert result["api_failure_count"]["sum"] == 6
        assert result["api_failure_count"]["mean"] == pytest.approx(2.0)

        # input_paths: absolute, stable order matching the call order.
        assert result["input_paths"] == [str(p.resolve()) for p in paths]
        for p in result["input_paths"]:
            assert Path(p).is_absolute()

    def test_single_sample_has_null_sample_std(self, tmp_path):
        path = _write_metrics(tmp_path / "only.json", overall_accuracy=0.42)
        result = aggregate_metrics([path])

        assert result["n_repeats"] == 1
        assert result["overall_accuracy"]["values"] == [0.42]
        assert result["overall_accuracy"]["mean"] == pytest.approx(0.42)
        assert result["overall_accuracy"]["sample_std"] is None

    def test_result_is_json_serializable(self, tmp_path):
        paths = [_write_metrics(tmp_path / f"r{i}.json", overall_accuracy=0.1 * i) for i in range(1, 4)]
        result = aggregate_metrics(paths)
        # Must round-trip through json without error (no NaN/Infinity/non-primitive values).
        text = json.dumps(result)
        reparsed = json.loads(text)
        assert reparsed["n_repeats"] == 3

    def test_input_path_order_is_preserved_not_sorted(self, tmp_path):
        paths = [
            _write_metrics(tmp_path / "zzz.json", overall_accuracy=0.9),
            _write_metrics(tmp_path / "aaa.json", overall_accuracy=0.1),
        ]
        result = aggregate_metrics(paths)
        assert result["input_paths"] == [str(p.resolve()) for p in paths]
        assert result["overall_accuracy"]["values"] == [0.9, 0.1]

    def test_does_not_mutate_input_files(self, tmp_path):
        paths = [_write_metrics(tmp_path / f"r{i}.json", overall_accuracy=0.1 * i) for i in range(1, 4)]
        before = [p.read_text(encoding="utf-8") for p in paths]
        aggregate_metrics(paths)
        after = [p.read_text(encoding="utf-8") for p in paths]
        assert before == after

    def test_accepts_pathlib_and_string_paths_interchangeably(self, tmp_path):
        p1 = _write_metrics(tmp_path / "r1.json", overall_accuracy=0.2)
        p2 = _write_metrics(tmp_path / "r2.json", overall_accuracy=0.4)
        result = aggregate_metrics([str(p1), p2])
        assert result["n_repeats"] == 2
        assert result["overall_accuracy"]["values"] == [0.2, 0.4]


# ---------------------------------------------------------------------------
# atomic_write_json
# ---------------------------------------------------------------------------


class TestAtomicWriteJson:
    def test_writes_valid_json_readable_back(self, tmp_path):
        out = tmp_path / "out.json"
        atomic_write_json(out, {"a": 1, "b": [1, 2, 3]})
        assert json.loads(out.read_text(encoding="utf-8")) == {"a": 1, "b": [1, 2, 3]}

    def test_pretty_output_is_indented(self, tmp_path):
        out = tmp_path / "out.json"
        atomic_write_json(out, {"a": 1}, pretty=True)
        text = out.read_text(encoding="utf-8")
        assert "\n" in text
        assert json.loads(text) == {"a": 1}

    def test_no_leftover_temp_files(self, tmp_path):
        out = tmp_path / "out.json"
        atomic_write_json(out, {"a": 1})
        remaining = list(tmp_path.iterdir())
        assert remaining == [out]

    def test_overwrites_existing_output(self, tmp_path):
        out = tmp_path / "out.json"
        atomic_write_json(out, {"a": 1})
        atomic_write_json(out, {"a": 2})
        assert json.loads(out.read_text(encoding="utf-8")) == {"a": 2}

    def test_creates_parent_directories(self, tmp_path):
        out = tmp_path / "nested" / "dir" / "out.json"
        atomic_write_json(out, {"a": 1})
        assert out.exists()


# ---------------------------------------------------------------------------
# CLI: script/aggregate_experiment_metrics.py
# ---------------------------------------------------------------------------


class TestCli:
    def _run_cli(self, args, cwd=None):
        return subprocess.run(
            [sys.executable, str(_CLI_SCRIPT), *args],
            cwd=cwd,
            capture_output=True,
            text=True,
        )

    def test_cli_writes_aggregated_output(self, tmp_path):
        paths = [
            _write_metrics(tmp_path / f"r{i}.json", overall_accuracy=acc)
            for i, acc in enumerate([0.1, 0.2, 0.3])
        ]
        out_path = tmp_path / "aggregated.json"

        proc = self._run_cli(
            ["--input", *[str(p) for p in paths], "--output", str(out_path)]
        )

        assert proc.returncode == 0, proc.stderr
        assert out_path.exists()
        result = json.loads(out_path.read_text(encoding="utf-8"))
        assert result["n_repeats"] == 3
        assert result["overall_accuracy"]["values"] == pytest.approx([0.1, 0.2, 0.3])
        assert result["overall_accuracy"]["sample_std"] == pytest.approx(statistics.stdev([0.1, 0.2, 0.3]))

    def test_cli_pretty_flag_produces_indented_output(self, tmp_path):
        paths = [_write_metrics(tmp_path / f"r{i}.json", overall_accuracy=0.1 * i) for i in range(1, 3)]
        out_path = tmp_path / "aggregated.json"

        proc = self._run_cli(
            ["--input", *[str(p) for p in paths], "--output", str(out_path), "--pretty"]
        )

        assert proc.returncode == 0, proc.stderr
        text = out_path.read_text(encoding="utf-8")
        assert "\n" in text
        assert json.loads(text)["n_repeats"] == 2

    def test_cli_does_not_modify_input_files(self, tmp_path):
        paths = [_write_metrics(tmp_path / f"r{i}.json", overall_accuracy=0.1 * i) for i in range(1, 3)]
        before = [p.read_text(encoding="utf-8") for p in paths]
        out_path = tmp_path / "aggregated.json"

        proc = self._run_cli(
            ["--input", *[str(p) for p in paths], "--output", str(out_path)]
        )

        assert proc.returncode == 0, proc.stderr
        after = [p.read_text(encoding="utf-8") for p in paths]
        assert before == after

    def test_cli_reports_error_and_nonzero_exit_on_inconsistent_benchmark(self, tmp_path):
        a = _write_metrics(tmp_path / "a.json", benchmark="lme_s")
        b = _write_metrics(tmp_path / "b.json", benchmark="lme_hybrid")
        out_path = tmp_path / "aggregated.json"

        proc = self._run_cli(["--input", str(a), str(b), "--output", str(out_path)])

        assert proc.returncode != 0
        assert not out_path.exists()
        assert "benchmark" in proc.stderr

    def test_cli_output_is_atomic_no_leftover_temp_files(self, tmp_path):
        paths = [_write_metrics(tmp_path / f"r{i}.json", overall_accuracy=0.1 * i) for i in range(1, 3)]
        out_path = tmp_path / "aggregated.json"

        proc = self._run_cli(
            ["--input", *[str(p) for p in paths], "--output", str(out_path)]
        )

        assert proc.returncode == 0, proc.stderr
        remaining = {p.name for p in tmp_path.iterdir()}
        expected = {p.name for p in paths} | {out_path.name}
        assert remaining == expected
