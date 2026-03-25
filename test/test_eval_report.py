import csv
import json
from pathlib import Path

from utils.eval_report import CSV_FIELDNAMES, append_csv_row, append_jsonl, utc_timestamp_iso


def test_utc_timestamp_iso_format():
    s = utc_timestamp_iso()
    assert len(s) == 20 and s.endswith("Z")


def test_append_jsonl_roundtrip(tmp_path: Path):
    p = tmp_path / "m.jsonl"
    append_jsonl(p, {"a": 1, "b": "x"})
    append_jsonl(p, {"a": 2})
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(lines[0]) == {"a": 1, "b": "x"}
    assert json.loads(lines[1]) == {"a": 2}


def test_append_csv_row_header_and_merge(tmp_path: Path):
    p = tmp_path / "s.csv"
    append_csv_row(
        p,
        {
            "timestamp": "t1",
            "eval_type": "judge",
            "input_path": "/a.jsonl",
            "benchmark": "lme",
            "n": 3,
            "judge_model": "m",
            "use_cot": False,
            "max_concurrency": 4,
            "overall_accuracy": 0.5,
            "api_failure_count": 0,
            "judged_count": 3,
            "mean_f1": "",
            "mean_exact_match": "",
            "token_mode": "",
            "per_type_json": {"x": 1},
        },
    )
    append_csv_row(
        p,
        {
            "timestamp": "t2",
            "eval_type": "f1",
            "input_path": "/b.jsonl",
            "benchmark": "locomo",
            "n": 10,
            "judge_model": "",
            "use_cot": "",
            "max_concurrency": "",
            "overall_accuracy": "",
            "api_failure_count": "",
            "judged_count": "",
            "mean_f1": 0.2,
            "mean_exact_match": 0.1,
            "token_mode": "char",
            "per_type_json": {},
        },
    )
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    assert len(rows) == 2
    assert set(rows[0].keys()) == set(CSV_FIELDNAMES)
    assert rows[0]["eval_type"] == "judge"
    assert rows[0]["use_cot"] == "false"
    assert rows[1]["mean_f1"] == "0.2"
