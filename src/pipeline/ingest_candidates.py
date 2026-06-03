#!/usr/bin/env python3
"""
将 ``pipeline/extract_candidates.py`` 产出的 per-episode JSON 写入向量库。

默认路径：
  - 候选：``MemDB/candidates/{benchmark}/{candidate_extract_model}/{candidate_suffix}/``
  - 灌库：``MemDB/ingest/{benchmark}/{manager_model}/{update_method}/``
  - trace：``logs/memory_trace/ingest/{benchmark}/{manager_model}/{update_method}/``

显式传入 ``--candidates-dir`` / ``--database-root`` / ``--trace-log-dir`` 时覆盖上述默认。

可选 ``--question-types``：与 benchmark 中 episode 的 ``question_type`` 对齐，仅灌库匹配的
``history_name``；候选目录可含全集 JSON，未匹配的 episode 会被跳过（与 extract 子集一致）。
可选 ``--benchmark-file``：与 extract 相同，覆盖默认数据路径。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, FrozenSet

from tqdm import tqdm

from benchmark import get_benchmark
from benchmark.datasets import resolve_benchmark_data_path
from utils.question_filter import parse_question_types_arg

from memory.candidate_ingest import (
    LmeCandidateAddAllMemorySystem,
    LmeCandidateRelationDecisionMemorySystem,
    apply_candidate_file,
    apply_candidate_file_mem0,
    apply_candidate_file_zep,
    load_candidate_json,
    LmeCandidateAmacMemorySystem,
    EverMemOSMemorySystem,
)
from memory.mem0 import Mem0MemorySystem
from memory.zep import ZepMemorySystem
from memory.tracing import remove_episode_trace_jsonl_files_for_logger
from pipeline.paths import (
    default_candidates_dir,
    default_ingest_dir,
    default_memory_trace_dir,
    safe_model_tag,
)
from utils.env import load_env
from utils.llm_api import load_api_chat_completion


def _default_embedding_base_url() -> str:
    return os.getenv("EMBEDDING_BASE_URL", "http://localhost:7110/v1/")


LME_APPLY_MEMORY_READY_VERSION = 1
LME_APPLY_MARKER_KIND = "lme_candidate_apply"

# ``LmeCandidateAddAllMemorySystem`` 灌库时不做 dense 检索；父类仍要求该参数，此处仅占位。
_ADD_ALL_RELATED_TOP_K_PLACEHOLDER = 1


def _print_apply_error(path: Path, exc: Exception) -> None:
    print(f"ERROR {path}: {exc}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)


def _episode_apply_marker_path(database_root: Path, history_name: str) -> Path:
    return database_root / history_name / ".memory_ready.json"


def _read_apply_marker(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _write_apply_marker_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _file_entry_rel(path: Path, candidates_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(candidates_dir.resolve()))
    except ValueError:
        return str(path.resolve())


def history_names_for_question_type_filter(
    *,
    benchmark: str,
    benchmark_file: str | None,
    question_types_arg: str | None,
) -> FrozenSet[str] | None:
    """
    若 ``question_types_arg`` 非空：返回 benchmark 中「至少有一道该类型题」的 ``history_name`` 集合；
    否则返回 ``None``（表示不筛选）。
    """
    q_types = parse_question_types_arg(question_types_arg)
    if not q_types:
        return None
    data_path, lang = resolve_benchmark_data_path(benchmark, benchmark_file)
    bm = get_benchmark(benchmark, file_path=data_path, lang=lang)
    return frozenset(
        str(ep.history_name)
        for ep in bm
        if any((q.question_type or "") in q_types for q in ep.qas)
    )


def _apply_config_fingerprint_block(args: argparse.Namespace) -> dict[str, Any]:
    block: dict[str, Any] = {
        "candidates_dir": str(args.candidates_dir.resolve()),
        "glob": args.glob,
        "update_method": args.update_method,
        "embedding_model": args.embedding_model,
        "language": args.language,
        "relation_concurrency": args.relation_concurrency,
        "relation_max_new_tokens": args.relation_max_new_tokens,
        "manager_max_new_tokens": args.manager_max_new_tokens,
        "relation_llm": args.relation_llm,
    }
    qt = parse_question_types_arg(getattr(args, "question_types", None))
    if qt is not None:
        block["question_types"] = sorted(qt)
    bf = getattr(args, "benchmark_file", None)
    if bf:
        block["benchmark_file"] = str(Path(str(bf)).resolve())
    if args.update_method == "relation_decision":
        block["related_top_k"] = args.related_top_k
        if max(1, int(args.relation_episode_concurrency)) != 1:
            block["relation_episode_concurrency"] = args.relation_episode_concurrency
        block["relation_system_en_template"] = (getattr(args, "relation_system_en_template", "") or "").strip()
        block["relation_system_zh_template"] = (getattr(args, "relation_system_zh_template", "") or "").strip()
        block["relation_user_template"] = (getattr(args, "relation_user_template", "") or "").strip()
    if args.update_method == "mem0":
        block["mem0_related_top_k"] = args.mem0_related_top_k
        block["mem0_related_aggregate_max"] = args.mem0_related_aggregate_max
    if args.update_method == "add_all":
        if max(1, int(args.add_all_episode_concurrency)) != 1:
            block["add_all_episode_concurrency"] = args.add_all_episode_concurrency
    if args.update_method == "zep":
        if max(1, int(args.zep_episode_concurrency)) != 1:
            block["zep_episode_concurrency"] = args.zep_episode_concurrency
    if args.update_method == "amac":
        block["amac_weights"] = str(getattr(args, "amac_weights", "") or "")
        block["amac_threshold"] = float(getattr(args, "amac_threshold", 0.5))
        block["amac_skip_utility"] = bool(getattr(args, "amac_skip_utility", False))
        block["amac_recency_decay_per_step"] = float(getattr(args, "amac_recency_decay_per_step", 0.12))
        block["amac_novelty_max_existing"] = int(getattr(args, "amac_novelty_max_existing", 64))
        block["ingest_obs_granularity"] = str(getattr(args, "ingest_obs_granularity", "all"))
        block["ingest_obs_turn_overlap"] = int(getattr(args, "ingest_obs_turn_overlap", 0))
        block["ingest_obs_dialogue_format"] = str(getattr(args, "ingest_obs_dialogue_format", "user_assistant"))
        if max(1, int(args.amac_episode_concurrency)) != 1:
            block["amac_episode_concurrency"] = args.amac_episode_concurrency
    if args.update_method == "evermemos":
        block["evermemos_similarity_threshold"] = float(getattr(args, "evermemos_similarity_threshold", 0.65))
        block["evermemos_max_time_gap_days"] = float(getattr(args, "evermemos_max_time_gap_days", 7.0))
        if max(1, int(args.evermemos_episode_concurrency)) != 1:
            block["evermemos_episode_concurrency"] = args.evermemos_episode_concurrency
    return block


def _lme_apply_fingerprint_for_file(path: Path, args: argparse.Namespace) -> str:
    st = path.stat()
    file_part = [_file_entry_rel(path, args.candidates_dir), st.st_mtime_ns, st.st_size]
    raw = json.dumps(
        {"file": file_part, "config": _apply_config_fingerprint_block(args)},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _episode_apply_marker_payload(
    history_name: str, path: Path, args: argparse.Namespace
) -> dict[str, Any]:
    return {
        "version": LME_APPLY_MEMORY_READY_VERSION,
        "kind": LME_APPLY_MARKER_KIND,
        "history_name": history_name,
        "source_relpath": _file_entry_rel(path, args.candidates_dir),
        "fingerprint": _lme_apply_fingerprint_for_file(path, args),
        "update_method": args.update_method,
        "embedding_model": args.embedding_model,
    }


def _episode_marker_up_to_date(
    database_root: Path, history_name: str, path: Path, args: argparse.Namespace
) -> bool:
    marker_path = _episode_apply_marker_path(database_root, history_name)
    data = _read_apply_marker(marker_path)
    if not data:
        return False
    if int(data.get("version", -1)) != LME_APPLY_MEMORY_READY_VERSION:
        return False
    if str(data.get("kind", "")) != LME_APPLY_MARKER_KIND:
        return False
    if getattr(args, "trust_apply_marker", False):
        if str(data.get("history_name", "")).strip() != history_name.strip():
            return False
        if str(data.get("update_method", "")) != args.update_method:
            return False
        return True
    return str(data.get("fingerprint", "")) == _lme_apply_fingerprint_for_file(path, args)


def _resolve_paths(args: argparse.Namespace) -> None:
    """Fill default MemDB paths when dirs omitted."""
    if getattr(args, "candidates_dir", None) is None:
        args.candidates_dir = default_candidates_dir(
            benchmark=args.benchmark,
            extract_model=args.candidate_extract_model,
            suffix=args.candidate_suffix,
        )
    if getattr(args, "database_root", None) is None:
        mgr = safe_model_tag(str(args.relation_llm or "").strip() or "unnamed_manager")
        args.database_root = default_ingest_dir(
            benchmark=args.benchmark,
            manager_model=mgr,
            method_name=args.update_method,
        )
    trace_raw = getattr(args, "trace_log_dir", None)
    if trace_raw is None:
        mgr = safe_model_tag(str(args.relation_llm or "").strip() or "unnamed_manager")
        args.trace_log_dir = str(
            default_memory_trace_dir(
                benchmark=args.benchmark,
                manager_model=mgr,
                method_name=args.update_method,
            )
        )


def _build_observation_map_for_history_name(
    *,
    benchmark: str,
    benchmark_file: str | None,
    history_name: str,
    memory_granularity: Any,
    dialogue_format: str,
    overlap_turns: int,
) -> dict[int, str]:
    """Rebuild chunk observation text from benchmark (same as extract_candidates chunks)."""
    from pipeline.extract_candidates import episode_to_observation_chunks

    data_path, lang = resolve_benchmark_data_path(benchmark, benchmark_file)
    bm = get_benchmark(benchmark, file_path=data_path, lang=lang)
    ep = next(
        (e for e in bm if str(e.history_name).strip() == history_name.strip()),
        None,
    )
    if ep is None:
        return {}
    pairs = episode_to_observation_chunks(
        ep,
        memory_granularity,
        dialogue_format,
        overlap_turns=int(overlap_turns),
    )
    return {int(meta["chunk_index"]): text for text, meta in pairs}


def _run_parallel_ingest(
    to_process: list[tuple[Path, str]],
    workers: int,
    apply_fn,
    desc: str,
    *,
    accumulate_errors: bool = False,
) -> int:
    """Run ``apply_fn(path, history_name)`` over ``to_process`` with ``workers`` threads.

    When ``accumulate_errors=False`` (default), returns 1 immediately on the first error.
    When ``accumulate_errors=True`` (used for zep), collects all errors and returns 1 if any.
    """
    if workers <= 1:
        n_errors = 0
        for path, history_name in tqdm(to_process, desc=desc, unit="ep"):
            try:
                apply_fn(path, history_name)
            except Exception as e:
                _print_apply_error(path, e)
                if not accumulate_errors:
                    return 1
                n_errors += 1
        return 1 if n_errors else 0

    future_to_path: dict = {}
    n_errors = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for path, history_name in to_process:
            fut = pool.submit(apply_fn, path, history_name)
            future_to_path[fut] = path
        for fut in tqdm(as_completed(future_to_path), total=len(future_to_path), desc=desc, unit="ep"):
            try:
                fut.result()
            except Exception as e:
                path = future_to_path[fut]
                _print_apply_error(path, e)
                n_errors += 1
                if not accumulate_errors:
                    return 1
    return 1 if n_errors else 0


def main() -> int:
    load_env()
    parser = argparse.ArgumentParser(
        description="候选事实 → 记忆库更新（relation_decision / mem0 / add_all / zep / amac）"
    )
    parser.add_argument(
        "--benchmark",
        default="lme_s",
        help="用于默认 MemDB 路径分段（与 extract 脚本的 --benchmark 一致）",
    )
    parser.add_argument(
        "--benchmark-file",
        default=None,
        help="与 extract / pipeline_generate 一致：覆盖默认 benchmark 数据文件",
    )
    parser.add_argument(
        "--question-types",
        default=None,
        metavar="TYPES",
        help=(
            "逗号分隔 question_type；仅灌库在 benchmark 中至少含一道该类型题的 episode。"
            "候选目录可含全集 JSON，与本参数不一致的 episode 会被跳过。"
        ),
    )
    parser.add_argument(
        "--candidate-extract-model",
        default="default",
        help="与 extract 时模型目录一致（默认 candidates 路径中的一段）",
    )
    parser.add_argument(
        "--candidate-suffix",
        default="default",
        help="与 extract 时 --suffix 一致",
    )
    parser.add_argument(
        "--update-method",
        choices=("relation_decision", "mem0", "add_all", "zep", "amac", "evermemos"),
        default="relation_decision",
        help="relation_decision：五类关系+桶聚合；mem0：Mem0 更新管线；add_all：候选全量直接入库；zep：graphiti；"
        "amac：A-MAC 加权准入后 add_all 式写 primary；evermemos：EverMemOS 增量语义聚类+LLM合并",
    )
    parser.add_argument(
        "--candidates-dir",
        type=Path,
        default=None,
        help="per-episode JSON 目录；默认 MemDB/candidates/{benchmark}/{candidate_extract_model}/{candidate_suffix}/",
    )
    parser.add_argument(
        "--glob",
        default="*.json",
        help="匹配候选文件（默认 *.json）",
    )
    parser.add_argument(
        "--database-root",
        type=Path,
        default=None,
        help="LocalFaiss 根目录；默认 MemDB/ingest/{benchmark}/{manager_model}/{update_method}/",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        help="Embedding 模型名（默认 EMBEDDING_MODEL 或 text-embedding-3-small）",
    )
    parser.add_argument(
        "--relation-llm",
        "--manager-model",
        dest="relation_llm",
        default="",
        help="灌库管理用 LLM（OpenAI 兼容端点）",
    )
    parser.add_argument(
        "--related-top-k",
        type=int,
        default=5,
        help="relation_decision：每条新事实 dense 检索条数（related_memory_top_k；add_all 灌库不用检索，可忽略）",
    )
    parser.add_argument(
        "--mem0-related-top-k",
        type=int,
        default=5,
        dest="mem0_related_top_k",
        help="mem0：更新决策时每条事实检索的条数（Mem0 related_memory_top_k；与 --related-top-k 独立）",
    )
    parser.add_argument(
        "--mem0-related-aggregate-max",
        type=int,
        default=10,
        dest="mem0_related_aggregate_max",
        help="mem0：多条事实检索结果合并去重后，按 score 截断保留的最大条数（默认 10）",
    )
    parser.add_argument(
        "--mem0-episode-concurrency",
        type=int,
        default=4,
        dest="mem0_episode_concurrency",
        help="mem0：跨 episode 并行灌库的线程数（候选 JSON 已含事实，不做 extract；默认 4，设为 1 则顺序执行）",
    )
    parser.add_argument(
        "--relation-episode-concurrency",
        type=int,
        default=1,
        dest="relation_episode_concurrency",
        help="relation_decision：跨 episode 并行灌库的线程数（默认 1 顺序；与 --mem0-episode-concurrency 类似）",
    )
    parser.add_argument(
        "--add-all-episode-concurrency",
        type=int,
        default=1,
        dest="add_all_episode_concurrency",
        help="add_all：跨 episode 并行灌库的线程数（默认 1 顺序）",
    )
    parser.add_argument(
        "--zep-episode-concurrency",
        type=int,
        default=1,
        dest="zep_episode_concurrency",
        help="zep：跨 episode 并行灌库的线程数（每个 episode 独立 Kuzu DB，线程安全；默认 1 顺序）",
    )
    parser.add_argument(
        "--amac-episode-concurrency",
        type=int,
        default=1,
        dest="amac_episode_concurrency",
        help="amac：跨 episode 并行灌库线程数（默认 1 顺序）",
    )
    parser.add_argument(
        "--amac-weights",
        type=str,
        default="0.1,0.1,0.1,0.1,0.6",
        help="A-MAC 权重 U,C,N,R,T（逗号分隔五数，或 JSON 列表）；自动归一化为和 1",
    )
    parser.add_argument(
        "--amac-threshold",
        type=float,
        default=0.55,
        help="A-MAC 准入阈值 S>=threshold 则写入",
    )
    parser.add_argument(
        "--amac-skip-utility",
        action="store_true",
        help="跳过 Utility LLM，U 固定为 0.5",
    )
    parser.add_argument(
        "--amac-recency-decay-per-step",
        type=float,
        default=0.12,
        help="A-MAC 块序 Recency：exp(-decay * steps_from_end)",
    )
    parser.add_argument(
        "--amac-novelty-max-existing",
        type=int,
        default=64,
        help="Novelty 最多与最近 N 条已入库 primary 比嵌入",
    )
    parser.add_argument(
        "--evermemos-similarity-threshold",
        type=float,
        default=0.65,
        dest="evermemos_similarity_threshold",
        help="evermemos：聚类匹配余弦相似度阈值（默认 0.65）",
    )
    parser.add_argument(
        "--evermemos-max-time-gap-days",
        type=float,
        default=7.0,
        dest="evermemos_max_time_gap_days",
        help="evermemos：同一聚类的最大时间跨度天数（默认 7 天；无日期信息则忽略时间约束）",
    )
    parser.add_argument(
        "--evermemos-episode-concurrency",
        type=int,
        default=10,
        dest="evermemos_episode_concurrency",
        help="evermemos：跨 episode 并行灌库线程数（默认 10）",
    )
    parser.add_argument(
        "--ingest-obs-granularity",
        type=str,
        default="all",
        help="amac：从 benchmark 重建 observation 的 memory_granularity（须与 extract 一致，默认 all）",
    )
    parser.add_argument(
        "--ingest-obs-turn-overlap",
        type=str,
        default="0",
        help="amac：重建 observation 的 turn overlap 字符串（须与 extract 一致；granularity=all 时为 0）",
    )
    parser.add_argument(
        "--ingest-obs-dialogue-format",
        type=str,
        default="user_assistant",
        help="amac：user_assistant 或 named_speakers（须与 extract 一致）",
    )
    parser.add_argument(
        "--relation-concurrency",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--relation-max-new-tokens",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--language",
        default="en",
        help="提示语种：en / zh（与 extract_candidates 保持一致）",
    )
    parser.add_argument(
        "--relation-system-template-en",
        default="",
        dest="relation_system_en_template",
        metavar="NAME.jinja",
        help="relation_decision：英文 system 模板（覆盖默认 lme_relation_classification_system_en_v2.jinja）",
    )
    parser.add_argument(
        "--relation-system-template-zh",
        default="",
        dest="relation_system_zh_template",
        metavar="NAME.jinja",
        help="relation_decision：中文 system 模板（覆盖默认 lme_relation_classification_system_zh_v2.jinja）",
    )
    parser.add_argument(
        "--relation-user-template",
        default="",
        dest="relation_user_template",
        metavar="NAME.jinja",
        help="relation_decision：成对比较 user 模板（覆盖默认 lme_relation_classification_user.jinja）",
    )
    parser.add_argument(
        "--manager-max-new-tokens",
        type=int,
        default=2048,
        dest="manager_max_new_tokens",
        help="mem0：更新决策 max_tokens（默认 2048）",
    )
    parser.add_argument(
        "--trace-log-dir",
        type=str,
        default=None,
        help="Memory trace；默认 logs/memory_trace/ingest/...；空字符串关闭",
    )
    parser.add_argument(
        "--trust-apply-marker",
        action="store_true",
        help=(
            "续传时若 episode 目录下已有合法 .memory_ready.json（version/kind/history_name/"
            "update_method 一致）则跳过，不再比对 fingerprint。默认会校验 fingerprint，"
            "以便候选 JSON 或灌库配置变更后自动重跑该 episode。"
        ),
    )
    args = parser.parse_args()
    _resolve_paths(args)

    from pipeline.extract_candidates import _normalize_memory_granularity, _normalize_turn_overlap

    try:
        args.ingest_obs_granularity_norm = _normalize_memory_granularity(str(args.ingest_obs_granularity))
        args.ingest_obs_overlap_norm = _normalize_turn_overlap(
            str(args.ingest_obs_turn_overlap),
            args.ingest_obs_granularity_norm,
        )
    except ValueError as e:
        print(f"ERROR: ingest observation chunk args: {e}", file=sys.stderr)
        return 1

    paths = sorted(args.candidates_dir.glob(args.glob))
    if not paths:
        print(f"No files matched {args.candidates_dir}/{args.glob}", file=sys.stderr)
        return 1

    allowed_histories = history_names_for_question_type_filter(
        benchmark=str(args.benchmark),
        benchmark_file=(str(args.benchmark_file).strip() or None) if args.benchmark_file else None,
        question_types_arg=args.question_types,
    )
    if allowed_histories is not None and len(allowed_histories) == 0:
        print(
            "WARNING: --question-types matched no episodes in the benchmark; "
            "all candidate files will be skipped.",
            file=sys.stderr,
        )

    db_root = args.database_root.resolve()
    file_jobs: list[tuple[Path, str]] = []
    skipped_question_filter = 0
    for path in paths:
        try:
            payload = load_candidate_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            print(f"ERROR loading {path}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return 1
        history_name = str(payload.get("history_name") or "").strip()
        if not history_name:
            print(f"ERROR {path}: candidate json missing history_name", file=sys.stderr)
            return 1
        if allowed_histories is not None and history_name not in allowed_histories:
            skipped_question_filter += 1
            continue
        file_jobs.append((path, history_name))

    if skipped_question_filter:
        print(
            f"Ingest: skipped {skipped_question_filter} candidate file(s) "
            f"(episode not in --question-types selection for benchmark {args.benchmark!r}).",
            flush=True,
        )
    if not file_jobs:
        print(
            "No candidate files to ingest after --question-types filter "
            f"(benchmark={args.benchmark!r}).",
            file=sys.stderr,
        )
        return 0

    to_process: list[tuple[Path, str]] = []
    skipped = 0
    for path, history_name in file_jobs:
        if _episode_marker_up_to_date(db_root, history_name, path, args):
            skipped += 1
            continue
        to_process.append((path, history_name))

    if not to_process:
        print(
            f"Skip apply ({args.update_method}): all {skipped} episode(s) up-to-date "
            f"under {db_root} (per-directory .memory_ready.json).",
            flush=True,
        )
        return 0

    if skipped:
        print(
            f"Partial skip ({args.update_method}): {skipped} episode(s) up-to-date, "
            f"{len(to_process)} to apply.",
            flush=True,
        )

    api_key = os.getenv("EMBEDDING_API_KEY")
    if not api_key:
        print("ERROR: EMBEDDING_API_KEY is not set.", file=sys.stderr)
        return 1

    try:
        from openai import OpenAI
    except ImportError as e:
        print("ERROR: openai package required.", e, file=sys.stderr)
        return 1

    embed_client = OpenAI(api_key=api_key, base_url=_default_embedding_base_url())
    llm_client = None
    if args.update_method in ("relation_decision", "mem0", "add_all", "zep", "amac", "evermemos"):
        if not str(args.relation_llm or "").strip():
            print(
                "ERROR: --relation-llm / --manager-model is required for this --update-method.",
                file=sys.stderr,
            )
            return 1
        llm_client = load_api_chat_completion(args.relation_llm, async_=False)

    trace_dir = (args.trace_log_dir or "").strip() or None
    lme_candidate_base_kw = dict(
        embed_model_name=args.embedding_model,
        llm_client=llm_client,
        embed_client=embed_client,
        database_root=str(args.database_root),
        language=args.language,
        granularity="all",
        trace_log_dir=trace_dir,
        dialogue_format="user_assistant",
        relation_concurrency=args.relation_concurrency,
        relation_max_new_tokens=args.relation_max_new_tokens,
    )
    if args.update_method == "relation_decision":
        rel_kw = {
            "relation_system_en_template": (args.relation_system_en_template or "").strip() or None,
            "relation_system_zh_template": (args.relation_system_zh_template or "").strip() or None,
            "relation_user_template": (args.relation_user_template or "").strip() or None,
        }
        memory = LmeCandidateRelationDecisionMemorySystem(
            **lme_candidate_base_kw,
            **rel_kw,
            related_memory_top_k=args.related_top_k,
        )
    elif args.update_method == "add_all":
        memory = LmeCandidateAddAllMemorySystem(
            **lme_candidate_base_kw,
            related_memory_top_k=_ADD_ALL_RELATED_TOP_K_PLACEHOLDER,
        )
    elif args.update_method == "amac":
        memory = LmeCandidateAmacMemorySystem(
            **lme_candidate_base_kw,
            related_memory_top_k=_ADD_ALL_RELATED_TOP_K_PLACEHOLDER,
            amac_weights=args.amac_weights,
            amac_threshold=float(args.amac_threshold),
            amac_skip_utility=bool(args.amac_skip_utility),
            amac_recency_decay_per_step=float(args.amac_recency_decay_per_step),
            amac_novelty_max_existing=int(args.amac_novelty_max_existing),
        )
    elif args.update_method == "zep":
        memory = ZepMemorySystem(
            embed_model_name=args.embedding_model,
            llm_client=llm_client,
            embed_client=embed_client,
            database_root=str(args.database_root),
            language=args.language,
            granularity="all",
            trace_log_dir=trace_dir,
            dialogue_format="user_assistant",
            manager_max_new_tokens=args.manager_max_new_tokens,
        )
    elif args.update_method == "evermemos":

        def _make_evermemos_memory() -> EverMemOSMemorySystem:
            # 每个 episode 独立实例：_cluster_state / _pending 不可跨线程共享。
            return EverMemOSMemorySystem(
                **lme_candidate_base_kw,
                related_memory_top_k=_ADD_ALL_RELATED_TOP_K_PLACEHOLDER,
                manager_max_new_tokens=args.manager_max_new_tokens,
                similarity_threshold=float(args.evermemos_similarity_threshold),
                max_time_gap_days=float(args.evermemos_max_time_gap_days),
            )

        memory = None  # evermemos 在 _apply_one_episode 内按 episode 构造
    else:
        # Candidate ingest only uses retrieve / decide / apply; it does not call _extract_facts.
        # extract_concurrency applies to dialogue-driven store_episode only.
        memory = Mem0MemorySystem(
            embed_model_name=args.embedding_model,
            llm_client=llm_client,
            embed_client=embed_client,
            database_root=str(args.database_root),
            related_memory_top_k=args.mem0_related_top_k,
            related_memory_aggregate_max=args.mem0_related_aggregate_max,
            language=args.language,
            granularity="all",
            trace_log_dir=trace_dir,
            dialogue_format="user_assistant",
            manager_max_new_tokens=args.manager_max_new_tokens,
        )

    desc = f"candidates→{args.update_method}"

    def _apply_one_episode(path: Path, history_name: str) -> None:
        # 重跑未完成 episode：先清空残留的数据库目录和 trace JSONL，再从零灌库
        # （已完成的 episode 有 .memory_ready.json marker，不会进入此分支）
        if args.update_method == "evermemos":
            ep_memory = _make_evermemos_memory()
        else:
            ep_memory = memory
        db = ep_memory._get_database(history_name)
        db.clear_all()
        remove_episode_trace_jsonl_files_for_logger(ep_memory.trace, history_name)
        if args.update_method == "mem0":
            apply_candidate_file_mem0(ep_memory, path)
        elif args.update_method == "zep":
            apply_candidate_file_zep(ep_memory, path)
        elif args.update_method == "amac":
            obs_map = _build_observation_map_for_history_name(
                benchmark=str(args.benchmark),
                benchmark_file=(str(args.benchmark_file).strip() or None) if args.benchmark_file else None,
                history_name=history_name,
                memory_granularity=args.ingest_obs_granularity_norm,
                dialogue_format=str(args.ingest_obs_dialogue_format or "user_assistant"),
                overlap_turns=int(args.ingest_obs_overlap_norm),
            )
            apply_candidate_file(ep_memory, path, observation_by_chunk_index=obs_map)
        elif args.update_method == "evermemos":
            apply_candidate_file(ep_memory, path)
            ep_memory.finalize_episode(db, history_name)
        else:
            apply_candidate_file(ep_memory, path)
        _write_apply_marker_atomic(
            _episode_apply_marker_path(db_root, history_name),
            _episode_apply_marker_payload(history_name, path, args),
        )

    _episode_concurrency_map = {
        "mem0": args.mem0_episode_concurrency,
        "relation_decision": args.relation_episode_concurrency,
        "add_all": args.add_all_episode_concurrency,
        "zep": args.zep_episode_concurrency,
        "amac": args.amac_episode_concurrency,
        "evermemos": args.evermemos_episode_concurrency,
    }
    workers = min(max(1, _episode_concurrency_map[args.update_method]), len(to_process))
    # zep accumulates errors (kuzu may crash mid-run); all other methods fail fast
    accumulate_errors = (args.update_method == "zep")
    return _run_parallel_ingest(to_process, workers, _apply_one_episode, desc,
                                accumulate_errors=accumulate_errors)


if __name__ == "__main__":
    raise SystemExit(main())
