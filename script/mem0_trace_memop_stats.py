#!/usr/bin/env python3
"""Aggregate mem0 memory_operation events from a memory_trace run directory (JSONL per file)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Not counted toward operation mix / ratios (internal consolidation steps).
_SKIPPED_OPERATIONS = frozenset({"DEDUPE_TEXT"})


def _scan_trace_dir(trace_dir: Path) -> Tuple[Dict[str, Any], List[str]]:
    jsonl_files = sorted(trace_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise SystemExit(f"No *.jsonl files under {trace_dir}")

    op_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    run_ids: set[str] = set()
    methods: set[str] = set()
    lines_total = 0
    parse_errors = 0
    memop_lines_raw = 0
    skipped_by_operation: Counter[str] = Counter()

    for path in jsonl_files:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, start=1):
                lines_total += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    parse_errors += 1
                    continue
                if not isinstance(rec, dict):
                    parse_errors += 1
                    continue
                if rec.get("event_type") != "memory_operation":
                    continue
                memop_lines_raw += 1
                op = rec.get("operation")
                op_key = op.upper() if isinstance(op, str) else "<missing>"
                if op_key in _SKIPPED_OPERATIONS:
                    skipped_by_operation[op_key] += 1
                    continue
                if isinstance(op, str):
                    op_counts[op_key] += 1
                else:
                    op_counts["<missing>"] += 1
                st = rec.get("status")
                status_counts[str(st) if st is not None else "<missing>"] += 1
                rid = rec.get("run_id")
                if isinstance(rid, str) and rid:
                    run_ids.add(rid)
                m = rec.get("method")
                if isinstance(m, str) and m:
                    methods.add(m)

    total_ops = sum(op_counts.values())
    return {
        "trace_dir": str(trace_dir.resolve()),
        "jsonl_files": [p.name for p in jsonl_files],
        "lines_total": lines_total,
        "memory_operation_events_raw": memop_lines_raw,
        "memory_operation_events": total_ops,
        "excluded_operations": dict(sorted(skipped_by_operation.items(), key=lambda x: (-x[1], x[0]))),
        "operation_counts": dict(sorted(op_counts.items(), key=lambda x: (-x[1], x[0]))),
        "status_counts": dict(sorted(status_counts.items(), key=lambda x: (-x[1], x[0]))),
        "run_ids": sorted(run_ids),
        "methods": sorted(methods),
        "parse_errors": parse_errors,
        "total_memory_operations_counted": total_ops,
    }, jsonl_files


def _build_report(stats: Dict[str, Any]) -> Dict[str, Any]:
    total = stats["total_memory_operations_counted"]
    op_counts: Dict[str, int] = stats["operation_counts"]
    fractions = {op: c / total for op, c in op_counts.items()}
    pct = {op: round(100.0 * f, 4) for op, f in fractions.items()}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **stats,
        "operation_fraction": fractions,
        "operation_percent": pct,
    }


def _print_summary(report: Dict[str, Any]) -> None:
    total = report["total_memory_operations_counted"]
    print(f"Trace dir: {report['trace_dir']}")
    print(f"JSONL files: {len(report['jsonl_files'])}")
    print(f"Memory operations (for ratios): {total}")
    excl = report.get("excluded_operations") or {}
    if excl:
        parts = [f"{k}={v}" for k, v in sorted(excl.items())]
        print(f"Excluded from ratios: {', '.join(parts)}")
    for op in sorted(report["operation_counts"], key=lambda k: (-report["operation_counts"][k], k)):
        c = report["operation_counts"][op]
        p = report["operation_percent"][op]
        print(f"  {op}: {c} ({p}%)")
    if report.get("methods"):
        print(f"method field(s): {', '.join(report['methods'])}")
    if len(report.get("run_ids", [])) > 1:
        print(f"run_id: multiple ({len(report['run_ids'])} distinct)")
    elif report.get("run_ids"):
        print(f"run_id: {report['run_ids'][0]}")
    if report["parse_errors"]:
        print(f"parse_errors: {report['parse_errors']}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trace_dir",
        type=Path,
        help="Directory containing per-episode *.jsonl memory traces (e.g. logs/memory_trace/<experiment_name>)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write JSON report here (default: logs/mem0_memop_stats_<basename>_<timestamp>.json)",
    )
    args = parser.parse_args()
    trace_dir = args.trace_dir
    if not trace_dir.is_dir():
        raise SystemExit(f"Not a directory: {trace_dir}")

    stats, _ = _scan_trace_dir(trace_dir)
    total_ops = stats["total_memory_operations_counted"]
    if total_ops == 0:
        raise SystemExit(
            "No memory_operation events left after excluding DEDUPE_TEXT (and similar). "
            "Nothing to compute ratios for."
        )

    report = _build_report(stats)

    out_path = args.output
    if out_path is None:
        logs = Path("logs")
        logs.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_base = trace_dir.name.replace("/", "_").replace("\\", "_")
        out_path = logs / f"mem0_memop_stats_{safe_base}_{ts}.json"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["output_path"] = str(out_path.resolve())

    _print_summary(report)
    print(f"Wrote: {report['output_path']}")


if __name__ == "__main__":
    main()
