"""Append evaluation summaries as JSONL and optional flattened CSV."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

CSV_FIELDNAMES: List[str] = [
    "timestamp",
    "eval_type",
    "input_path",
    "benchmark",
    "n",
    "judge_model",
    "use_cot",
    "max_concurrency",
    "overall_accuracy",
    "api_failure_count",
    "judged_count",
    "mean_f1",
    "mean_exact_match",
    "token_mode",
    "per_type_json",
]


def utc_timestamp_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def append_csv_row(path: Path, row: Mapping[str, Any], fieldnames: Sequence[str] = CSV_FIELDNAMES) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=names, extrasaction="ignore")
        if not file_exists:
            w.writeheader()
        out = {k: _csv_cell(row.get(k)) for k in names}
        w.writerow(out)
