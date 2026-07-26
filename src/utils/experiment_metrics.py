"""重复实验 Judge metrics 聚合（Experiment Metrics Aggregation）。

用于把同一实验配置下重复运行 N 次的 Judge ``metrics.json``（每个文件是单个
JSON object，含 ``overall_accuracy`` / ``judged_count`` / ``api_failure_count`` /
``per_type`` 等字段）聚合为一份汇总统计（mean / sample std），用于稳定性/方差报告。

设计要点：
- 纯标准库实现，不依赖项目其它模块，也不发起任何网络请求。
- 完全以调用方显式传入的 ``paths`` 为 repeat 样本集合；不扫描目录、不推断
  attempt/文件名模式，避免把无关的兄弟文件误当成重复样本。
- 拒绝空输入、缺失或非数值的 ``overall_accuracy``，以及跨输入不一致的
  ``benchmark`` / ``judge_model``；错误信息中总是包含出问题的具体路径与字段名，
  便于定位。
- ``aggregate_metrics`` 只读取输入文件，不做任何写入；输出是可以直接
  ``json.dumps`` 的普通 dict（不含 NaN/Infinity）。
"""

from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

__all__ = [
    "SCHEMA_VERSION",
    "load_metrics",
    "aggregate_metrics",
    "atomic_write_json",
]

SCHEMA_VERSION = 1

_REQUIRED_STRING_FIELDS: Sequence[str] = ("benchmark", "judge_model")
_REQUIRED_NUMERIC_FIELDS: Sequence[str] = ("overall_accuracy", "judged_count", "api_failure_count")

PathLike = Union[str, "os.PathLike[str]"]


def load_metrics(path: PathLike) -> Dict[str, Any]:
    """读取并解析单个 Judge ``metrics.json`` 文件，返回其顶层 JSON object。

    只读取、不修改文件。任何失败（文件不存在/不可读、JSON 语法错误、顶层不是
    JSON object）都会抛出 ``ValueError``，错误信息中包含 ``path``。
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"failed to read metrics file at {p}: {exc}") from exc

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in metrics file at {p}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"metrics file at {p} must contain a single JSON object at the top level, "
            f"got {type(data).__name__}"
        )
    return data


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _require_field(data: Mapping[str, Any], field: str, path: str) -> Any:
    if field not in data:
        raise ValueError(f"metrics file at {path} is missing required field '{field}'")
    return data[field]


def _require_numeric_field(data: Mapping[str, Any], field: str, path: str) -> float:
    value = _require_field(data, field, path)
    if not _is_finite_number(value):
        raise ValueError(
            f"metrics file at {path} has a missing/non-numeric value for required field "
            f"'{field}': {value!r}"
        )
    return value


def _require_string_field(data: Mapping[str, Any], field: str, path: str) -> str:
    value = _require_field(data, field, path)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"metrics file at {path} has a missing/invalid value for required field "
            f"'{field}': {value!r}"
        )
    return value


def _check_consistent(field: str, entries: Sequence[tuple]) -> None:
    """entries: 按输入顺序排列的 (path, value) 列表；不一致时报出全部取值分布。"""
    distinct = sorted({value for _, value in entries})
    if len(distinct) <= 1:
        return
    detail = ", ".join(f"{path}={value!r}" for path, value in entries)
    raise ValueError(
        f"aggregate_metrics requires all inputs to share the same '{field}', "
        f"but found {len(distinct)} distinct values across inputs: {detail}"
    )


def aggregate_metrics(paths: Sequence[PathLike]) -> Dict[str, Any]:
    """把 ``paths`` 指向的重复 Judge metrics.json 聚合为一份汇总统计。

    ``paths`` 的顺序即 repeat 顺序（不排序、不去重、不做任何目录扫描/推断），
    完全由调用方显式决定哪些文件算作同一组 repeat 样本。

    返回的 mapping 是 JSON-safe 的（可直接 ``json.dumps``），结构：

    ``{schema_version, benchmark, judge_model, input_paths, n_repeats,
    overall_accuracy: {values, mean, sample_std}, judged_count: {sum, mean},
    api_failure_count: {sum, mean}}``

    ``n_repeats == 1`` 时 ``overall_accuracy.sample_std`` 为 ``None``。
    """
    path_list = list(paths)
    if not path_list:
        raise ValueError("aggregate_metrics requires at least one input path, got an empty list")

    resolved_paths = [str(Path(p).resolve()) for p in path_list]

    loaded: List[Dict[str, Any]] = [
        load_metrics(resolved) for resolved in resolved_paths
    ]

    benchmarks = []
    judge_models = []
    accuracies: List[float] = []
    judged_counts: List[float] = []
    api_failure_counts: List[float] = []

    for resolved, data in zip(resolved_paths, loaded):
        benchmarks.append((resolved, _require_string_field(data, "benchmark", resolved)))
        judge_models.append((resolved, _require_string_field(data, "judge_model", resolved)))
        accuracies.append(_require_numeric_field(data, "overall_accuracy", resolved))
        judged_counts.append(_require_numeric_field(data, "judged_count", resolved))
        api_failure_counts.append(_require_numeric_field(data, "api_failure_count", resolved))

    _check_consistent("benchmark", benchmarks)
    _check_consistent("judge_model", judge_models)

    n = len(path_list)
    mean_accuracy = statistics.mean(accuracies)
    sample_std_accuracy: Optional[float] = statistics.stdev(accuracies) if n > 1 else None

    judged_sum = sum(judged_counts)
    api_failure_sum = sum(api_failure_counts)

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": benchmarks[0][1],
        "judge_model": judge_models[0][1],
        "input_paths": resolved_paths,
        "n_repeats": n,
        "overall_accuracy": {
            "values": accuracies,
            "mean": mean_accuracy,
            "sample_std": sample_std_accuracy,
        },
        "judged_count": {
            "sum": judged_sum,
            "mean": judged_sum / n,
        },
        "api_failure_count": {
            "sum": api_failure_sum,
            "mean": api_failure_sum / n,
        },
    }


def atomic_write_json(
    path: PathLike,
    data: Any,
    *,
    pretty: bool = False,
) -> None:
    """以「临时文件 + os.replace」的方式原子写出 JSON，不留下半写文件。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if pretty:
        text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    else:
        text = json.dumps(data, sort_keys=True, ensure_ascii=False)

    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
