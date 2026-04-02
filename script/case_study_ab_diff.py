#!/usr/bin/env python3
"""筛选「方法 A 答对、方法 B 答错」的样本；可选合并 answer_agent_trace 中的 question_answer 事件。

两份输入应为同一 benchmark 下、经 pipeline_evaluate --write_back 后的 JSONL（含 is_correct）。
Agent trace 目录通常为 logs/answer_agent_trace/<experiment_name>/，内含 agent_*.jsonl。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

KeyT = Tuple[str, str]


def _row_key(row: Dict[str, Any]) -> KeyT:
    h = str(row.get("history_name", ""))
    qid = row.get("question_id")
    return (h, str(qid if qid is not None else h))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _pick_latest_agent_trace_file(directory: Path) -> Path:
    files = sorted(
        directory.glob("agent_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(f"No agent_*.jsonl under {directory}")
    return files[0]


def load_agent_trace_index(path: Path) -> Dict[KeyT, Dict[str, Any]]:
    """Map (history_name, question_id) -> trace event for question_answer rows."""
    if path.is_dir():
        path = _pick_latest_agent_trace_file(path)
    index: Dict[KeyT, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            evt = json.loads(line)
            if evt.get("event_type") != "question_answer":
                continue
            h = str(evt.get("history_name", ""))
            qid = evt.get("question_id")
            k = (h, str(qid if qid is not None else h))
            index[k] = evt
    return index


def _build_output_row(
    ra: Dict[str, Any],
    rb: Dict[str, Any],
    trace_a: Optional[Dict[str, Any]],
    trace_b: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "history_name": ra.get("history_name"),
        "question_id": ra.get("question_id"),
        "question": ra.get("question"),
        "benchmark": ra.get("benchmark"),
        "answer": ra.get("answer"),
        "model_answer_a": ra.get("model_answer"),
        "model_answer_b": rb.get("model_answer"),
        "is_correct_a": ra.get("is_correct"),
        "is_correct_b": rb.get("is_correct"),
        "judge_api_failed_a": ra.get("judge_api_failed"),
        "judge_api_failed_b": rb.get("judge_api_failed"),
    }
    if trace_a is not None:
        out["agent_trace_a"] = {
            "prompt": trace_a.get("prompt"),
            "response": trace_a.get("response"),
            "retrieved": trace_a.get("retrieved"),
            "question": trace_a.get("question"),
        }
    if trace_b is not None:
        out["agent_trace_b"] = {
            "prompt": trace_b.get("prompt"),
            "response": trace_b.get("response"),
            "retrieved": trace_b.get("retrieved"),
            "question": trace_b.get("question"),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Case study: rows where method A is correct and method B is wrong (requires is_correct on both JSONL).",
    )
    parser.add_argument("--input-a", required=True, type=Path, help="JSONL evaluated with A (write_back)")
    parser.add_argument("--input-b", required=True, type=Path, help="JSONL evaluated with B (write_back)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: stdout)",
    )
    parser.add_argument(
        "--agent-trace-a",
        type=Path,
        default=None,
        help="Optional: file or directory (latest agent_*.jsonl) for method A",
    )
    parser.add_argument(
        "--agent-trace-b",
        type=Path,
        default=None,
        help="Optional: file or directory (latest agent_*.jsonl) for method B",
    )
    args = parser.parse_args()

    rows_a = load_jsonl(args.input_a)
    rows_b = load_jsonl(args.input_b)

    map_a = {_row_key(r): r for r in rows_a}
    map_b = {_row_key(r): r for r in rows_b}

    idx_a: Optional[Dict[KeyT, Dict[str, Any]]] = None
    idx_b: Optional[Dict[KeyT, Dict[str, Any]]] = None
    if args.agent_trace_a is not None:
        idx_a = load_agent_trace_index(args.agent_trace_a)
    if args.agent_trace_b is not None:
        idx_b = load_agent_trace_index(args.agent_trace_b)

    keys_common = set(map_a) & set(map_b)
    selected: List[KeyT] = []
    for k in sorted(keys_common):
        ra, rb = map_a[k], map_b[k]
        if ra.get("is_correct") is True and rb.get("is_correct") is False:
            selected.append(k)

    out_lines: List[str] = []
    for k in selected:
        ra, rb = map_a[k], map_b[k]
        ta = idx_a.get(k) if idx_a is not None else None
        tb = idx_b.get(k) if idx_b is not None else None
        row = _build_output_row(ra, rb, ta, tb)
        out_lines.append(json.dumps(row, ensure_ascii=False))

    text = "\n".join(out_lines) + ("\n" if out_lines else "")

    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")

    print(f"Matched {len(selected)} rows (A correct, B wrong) out of {len(keys_common)} common keys.", file=sys.stderr)


if __name__ == "__main__":
    main()
