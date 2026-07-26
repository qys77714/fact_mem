#!/usr/bin/env python3
"""CLI：把一组重复实验的 Judge ``metrics.json`` 聚合为一份汇总统计。

用法::

    python script/aggregate_experiment_metrics.py --input A.json B.json C.json --output OUT.json
    python script/aggregate_experiment_metrics.py --input A.json B.json --output OUT.json --pretty

只读取 ``--input`` 指向的文件（不做任何目录扫描/推断，不修改输入），把
``src/utils/experiment_metrics.py::aggregate_metrics`` 的结果原子写到 ``--output``。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.experiment_metrics import aggregate_metrics, atomic_write_json  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate repeated Judge metrics.json files into one summary (mean/sample_std)."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        metavar="PATH",
        help="explicit list of metrics.json paths, one per repeat run (order preserved)",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="path to write the aggregated JSON summary to (atomic write)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="pretty-print (indented) the output JSON instead of compact single-line JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        result = aggregate_metrics(args.input)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    atomic_write_json(args.output, result, pretty=args.pretty)
    return 0


if __name__ == "__main__":
    sys.exit(main())
