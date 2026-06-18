"""
MEME 4-Phase Pipeline — 与 MEME-public run_episode 协议完全对齐。

每个 episode 按四阶段执行：
  Phase 1: ingest sessions 0 → before_pos → MemDB/{hn}_before
  Phase 2: answer before_questions
  Phase 3: copy {hn}_before → {hn}_after; 增量 ingest sessions before_pos+1 → after_pos
  Phase 4: answer after_questions

relation_decision：灌库时就地增量融合答题记忆 C（同库，role=answer），答题用 answer_mode
检索（C + 未被覆盖的孤立原子）。不再有独立的事后整树 fuse 阶段 / _fused 目录。

pred.jsonl 格式与 pipeline_lme_generate.py 一致，含 phase/entity_key/entity_values/hop 字段。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from benchmark import get_benchmark
from benchmark.datasets import resolve_benchmark_data_path
from benchmark.base import MemoryEpisode, QuestionItem
from memory.candidate_ingest import (
    LmeCandidateAddAllMemorySystem,
    LmeCandidateAmacMemorySystem,
    LmeCandidateRelationDecisionMemorySystem,
    EverMemOSMemorySystem,
    apply_candidate_episode_json,
    apply_candidate_episode_mem0,
    apply_candidate_episode_zep,
    load_candidate_json,
)
from memory.mem0 import Mem0MemorySystem
from memory.storage.local_faiss import LocalFaissDatabase
from memory.zep import ZepMemorySystem
from memory.prebuilt import PrebuiltMemorySystem
from memory.tracing import remove_episode_trace_jsonl_files_for_logger
from agent.standard_agent import StandardAgent
from utils.env import load_env
from utils.eval_report import append_jsonl, utc_timestamp_iso
from utils.llm_api import load_api_chat_completion

_ADD_ALL_RELATED_TOP_K_PLACEHOLDER = 1


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class MemePhaseConfig:
    benchmark: str
    benchmark_file: Optional[str]
    candidates_dir: Path
    update_method: str
    database_root: Path             # unfused root; episodes stored as {hn}_before / {hn}_after
    fused_database_root: Optional[Path]  # only for relation_decision
    output: Path
    answer_model: str
    embedding_model: str
    embedding_base_url: str
    embedding_api_key: str
    manager_model: str
    language: str
    retrieve_topk: int
    memory_token_limit: int
    answer_concurrency: int
    parallel_episodes: int
    show_memory_time: bool
    # hybrid retrieval
    hybrid_bm25_dense: bool
    hybrid_dense_weight: float
    hybrid_bm25_weight: float
    hybrid_pool_mult: int
    # ingest-specific
    relation_concurrency: int
    relation_max_new_tokens: int
    manager_max_new_tokens: int
    related_top_k: int
    mem0_related_top_k: int
    mem0_related_aggregate_max: int
    relation_system_en_template: str
    relation_system_zh_template: str
    relation_user_template: str
    # fusion (relation_decision)
    fuse_max_new_tokens: int
    fusion_bundle_template_en: str
    fusion_bundle_template_zh: str
    fusion_edge_labels_template_en: str
    fusion_edge_labels_template_zh: str
    fusion_package_concurrency: int
    # amac
    amac_weights: str
    amac_threshold: float
    amac_skip_utility: bool
    amac_recency_decay_per_step: float
    amac_novelty_max_existing: int
    # evermemos
    evermemos_similarity_threshold: float
    evermemos_max_time_gap_days: float
    evermemos_cluster_concurrency: int
    # cascade (relation_decision)
    cascade_enabled: bool
    deletion_enabled: bool
    topic_aggregation_enabled: bool
    condition_sim_threshold: float
    pairwise_sim_threshold: float
    # trace
    trace_log_dir: Optional[str]


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------

def _filter_payload(payload: Dict[str, Any], hn_phase: str,
                    min_si: Optional[int] = None,
                    max_si: Optional[int] = None) -> Dict[str, Any]:
    """Return a shallow copy of payload with history_name overridden and chunks filtered."""
    p = dict(payload)
    p["history_name"] = hn_phase
    chunks = list(payload.get("chunks") or [])
    if max_si is not None:
        chunks = [c for c in chunks if int(c.get("session_index") or 1) <= max_si]
    if min_si is not None:
        chunks = [c for c in chunks if int(c.get("session_index") or 1) > min_si]
    p["chunks"] = chunks
    return p


# ---------------------------------------------------------------------------
# DB copy / fuse helpers
# ---------------------------------------------------------------------------

def _copy_episode_db(root: Path, hn_src: str, hn_dst: str) -> None:
    src = root / hn_src
    dst = root / hn_dst
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


# ---------------------------------------------------------------------------
# Memory system builders
# ---------------------------------------------------------------------------

def _build_ingest_memory(cfg: MemePhaseConfig, database_root: Path):
    """Build the ingest-side memory system (points at database_root)."""
    api_key = cfg.embedding_api_key
    from openai import OpenAI
    embed_client = OpenAI(api_key=api_key, base_url=cfg.embedding_base_url)
    llm_client = load_api_chat_completion(cfg.manager_model, async_=False)

    base_kw = dict(
        embed_model_name=cfg.embedding_model,
        llm_client=llm_client,
        embed_client=embed_client,
        database_root=str(database_root),
        language=cfg.language,
        granularity="all",
        trace_log_dir=cfg.trace_log_dir,
        dialogue_format="user_assistant",
        relation_concurrency=cfg.relation_concurrency,
        relation_max_new_tokens=cfg.relation_max_new_tokens,
    )

    m = cfg.update_method
    if m == "relation_decision":
        return LmeCandidateRelationDecisionMemorySystem(
            **base_kw,
            relation_system_en_template=cfg.relation_system_en_template or None,
            relation_system_zh_template=cfg.relation_system_zh_template or None,
            relation_user_template=cfg.relation_user_template or None,
            related_memory_top_k=cfg.related_top_k,
            cascade_enabled=cfg.cascade_enabled,
            deletion_enabled=cfg.deletion_enabled,
            condition_sim_threshold=cfg.condition_sim_threshold,
            pairwise_sim_threshold=cfg.pairwise_sim_threshold,
            answer_fuse_max_new_tokens=cfg.fuse_max_new_tokens,
            topic_aggregation_enabled=cfg.topic_aggregation_enabled,
        )
    if m == "add_all":
        return LmeCandidateAddAllMemorySystem(
            **base_kw, related_memory_top_k=_ADD_ALL_RELATED_TOP_K_PLACEHOLDER
        )
    if m == "amac":
        return LmeCandidateAmacMemorySystem(
            **base_kw,
            related_memory_top_k=_ADD_ALL_RELATED_TOP_K_PLACEHOLDER,
            amac_weights=cfg.amac_weights,
            amac_threshold=cfg.amac_threshold,
            amac_skip_utility=cfg.amac_skip_utility,
            amac_recency_decay_per_step=cfg.amac_recency_decay_per_step,
            amac_novelty_max_existing=cfg.amac_novelty_max_existing,
        )
    if m == "zep":
        return ZepMemorySystem(
            embed_model_name=cfg.embedding_model,
            llm_client=llm_client,
            embed_client=embed_client,
            database_root=str(database_root),
            language=cfg.language,
            granularity="all",
            trace_log_dir=cfg.trace_log_dir,
            dialogue_format="user_assistant",
            manager_max_new_tokens=cfg.manager_max_new_tokens,
        )
    if m == "evermemos":
        return EverMemOSMemorySystem(
            **base_kw,
            related_memory_top_k=_ADD_ALL_RELATED_TOP_K_PLACEHOLDER,
            manager_max_new_tokens=cfg.manager_max_new_tokens,
            similarity_threshold=cfg.evermemos_similarity_threshold,
            max_time_gap_days=cfg.evermemos_max_time_gap_days,
            cluster_concurrency=cfg.evermemos_cluster_concurrency,
        )
    # mem0 (default)
    return Mem0MemorySystem(
        embed_model_name=cfg.embedding_model,
        llm_client=llm_client,
        embed_client=embed_client,
        database_root=str(database_root),
        related_memory_top_k=cfg.mem0_related_top_k,
        related_memory_aggregate_max=cfg.mem0_related_aggregate_max,
        language=cfg.language,
        granularity="all",
        trace_log_dir=cfg.trace_log_dir,
        dialogue_format="user_assistant",
        manager_max_new_tokens=cfg.manager_max_new_tokens,
    )


def _build_answer_memory(cfg: MemePhaseConfig, answer_db_root: Path) -> PrebuiltMemorySystem:
    api_key = cfg.embedding_api_key
    from openai import OpenAI
    embed_client = OpenAI(api_key=api_key, base_url=cfg.embedding_base_url)
    return PrebuiltMemorySystem(
        embed_model_name=cfg.embedding_model,
        embed_client=embed_client,
        database_root=str(answer_db_root),
        use_hybrid_retrieval=cfg.hybrid_bm25_dense,
        hybrid_dense_weight=cfg.hybrid_dense_weight,
        hybrid_bm25_weight=cfg.hybrid_bm25_weight,
        hybrid_pool_mult=cfg.hybrid_pool_mult,
        # relation_decision：答题只检索融合记忆 C + 未被覆盖的孤立原子（排除 evidence/被覆盖原子）
        answer_mode=(cfg.update_method == "relation_decision"),
        language=cfg.language,
    )


# ---------------------------------------------------------------------------
# Apply helpers (episode-level)
# ---------------------------------------------------------------------------

async def _close_async_chat_client(answer_client: Any) -> None:
    """Close httpx-backed AsyncOpenAI so asyncio.run() does not leave dangling aclose tasks."""
    inner = getattr(answer_client, "client", None)
    close = getattr(inner, "close", None) if inner is not None else None
    if close is not None:
        await close()


async def _batch_answer_and_close(
    agent: StandardAgent,
    history_name: str,
    questions: List[QuestionItem],
    top_k: int,
    answer_client: Any,
) -> List[str]:
    try:
        return await agent.batch_answer_questions(
            history_name=history_name,
            questions=questions,
            top_k=top_k,
        )
    finally:
        await _close_async_chat_client(answer_client)


def _apply_phase(ingest_memory, payload_phase: Dict[str, Any],
                 update_method: str, incremental: bool = False) -> None:
    """Apply a phase payload to ingest_memory. If not incremental, clear DB first."""
    hn = str(payload_phase["history_name"])
    db = ingest_memory._get_database(hn)
    if not incremental:
        db.clear_all()
        remove_episode_trace_jsonl_files_for_logger(ingest_memory.trace, hn)

    if update_method == "mem0":
        apply_candidate_episode_mem0(ingest_memory, payload_phase)
    elif update_method == "zep":
        apply_candidate_episode_zep(
            ingest_memory, payload_phase, incremental=incremental
        )
    elif update_method == "evermemos":
        if not incremental:
            ingest_memory._reset_episode_state()
        apply_candidate_episode_json(ingest_memory, payload_phase)
        ingest_memory.finalize_episode(db, hn)
    else:
        apply_candidate_episode_json(ingest_memory, payload_phase)


# ---------------------------------------------------------------------------
# pred.jsonl helpers
# ---------------------------------------------------------------------------

def _build_record(benchmark: str, history_name: str, q: QuestionItem,
                  model_answer: Optional[str]) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "benchmark": benchmark,
        "history_name": history_name,
        "question_id": q.metadata.get("question_id", history_name),
        "question": q.question,
        "answer": q.answer,
        "model_answer": model_answer or "",
        "question_type": q.question_type,
        "question_time": q.question_time,
    }
    if q.options is not None:
        record["options"] = q.options
    for f in ("phase", "entity_key", "entity_values", "max_session_index", "hop"):
        if f in q.metadata:
            record[f] = q.metadata[f]
    return record


def _load_answered_keys(output_path: Path) -> Set[Tuple[str, str]]:
    if not output_path.exists():
        return set()
    answered: Set[Tuple[str, str]] = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                h = str(r.get("history_name", ""))
                qid = r.get("question_id")
                answered.add((h, str(qid if qid is not None else h)))
            except json.JSONDecodeError:
                pass
    return answered


# ---------------------------------------------------------------------------
# Per-episode 4-phase logic (runs in thread pool)
# ---------------------------------------------------------------------------

def run_episode_4phase(
    episode: MemoryEpisode,
    cfg: MemePhaseConfig,
    cand_path: Path,
    answered: Set[Tuple[str, str]],
    output_path: Path,
    output_lock: threading.Lock,
) -> None:
    hn = str(episode.history_name)
    phase_blocks = episode.metadata.get("phase_blocks", {})
    before_block = phase_blocks.get("before", {})
    after_block = phase_blocks.get("after", {})
    before_max = int(before_block.get("max_session_index", len(episode.sessions)))
    after_max = int(after_block.get("max_session_index", len(episode.sessions)))

    hn_before = f"{hn}_before"
    hn_after = f"{hn}_after"

    payload = load_candidate_json(cand_path)

    # relation_decision 现在灌库时就地融合答题记忆 C（同库），答题直接读 database_root；
    # 不再需要事后 fuse 到 fused_database_root。
    answer_db_root = cfg.database_root

    # Build per-episode ingest memory system pointing at the shared database_root.
    # All namespaces (hn_before / hn_after) are subdirs under this root.
    ingest_memory = _build_ingest_memory(cfg, cfg.database_root)

    # -----------------------------------------------------------------------
    # Phase 1: ingest sessions 0 → before_max
    # -----------------------------------------------------------------------
    print(f"  [{hn}] Phase 1: ingest sessions 1-{before_max} → {hn_before}", flush=True)
    payload_before = _filter_payload(payload, hn_before, max_si=before_max)
    _apply_phase(ingest_memory, payload_before, cfg.update_method, incremental=False)

    # -----------------------------------------------------------------------
    # Phase 2: answer before_questions
    # -----------------------------------------------------------------------
    before_qas = [q for q in episode.qas if q.metadata.get("phase") == "before"]
    pending_before = [
        q for q in before_qas
        if (hn, str(q.metadata.get("question_id", hn))) not in answered
    ]
    if pending_before:
        print(f"  [{hn}] Phase 2: {len(pending_before)} before questions", flush=True)
        answer_mem_before = _build_answer_memory(cfg, answer_db_root)
        answer_client = load_api_chat_completion(cfg.answer_model, async_=True)
        agent = StandardAgent(
            memory_system=answer_mem_before,
            chat_model=answer_client,
            memory_token_limit=cfg.memory_token_limit,
            language=cfg.language,
            answer_concurrency=cfg.answer_concurrency,
            show_time=cfg.show_memory_time,
        )
        before_answers = asyncio.run(
            _batch_answer_and_close(
                agent,
                hn_before,
                pending_before,
                cfg.retrieve_topk,
                answer_client,
            )
        )
        records = [_build_record(cfg.benchmark, hn, q, ans)
                   for q, ans in zip(pending_before, before_answers)]
        with output_lock:
            with output_path.open("a", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()

    # -----------------------------------------------------------------------
    # Phase 3: copy _before → _after; ingest sessions before_max+1 → after_max
    # -----------------------------------------------------------------------
    print(f"  [{hn}] Phase 3: copy {hn_before} → {hn_after}; "
          f"ingest sessions {before_max+1}-{after_max}", flush=True)
    _copy_episode_db(cfg.database_root, hn_before, hn_after)
    payload_after = _filter_payload(payload, hn_after,
                                    min_si=before_max, max_si=after_max)
    _apply_phase(ingest_memory, payload_after, cfg.update_method, incremental=True)

    # -----------------------------------------------------------------------
    # Phase 4: answer after_questions
    # -----------------------------------------------------------------------
    after_qas = [q for q in episode.qas if q.metadata.get("phase") == "after"]
    pending_after = [
        q for q in after_qas
        if (hn, str(q.metadata.get("question_id", hn))) not in answered
    ]
    if pending_after:
        print(f"  [{hn}] Phase 4: {len(pending_after)} after questions", flush=True)
        answer_mem_after = _build_answer_memory(cfg, answer_db_root)
        answer_client = load_api_chat_completion(cfg.answer_model, async_=True)
        agent = StandardAgent(
            memory_system=answer_mem_after,
            chat_model=answer_client,
            memory_token_limit=cfg.memory_token_limit,
            language=cfg.language,
            answer_concurrency=cfg.answer_concurrency,
            show_time=cfg.show_memory_time,
        )
        after_answers = asyncio.run(
            _batch_answer_and_close(
                agent,
                hn_after,
                pending_after,
                cfg.retrieve_topk,
                answer_client,
            )
        )
        records = [_build_record(cfg.benchmark, hn, q, ans)
                   for q, ans in zip(pending_after, after_answers)]
        with output_lock:
            with output_path.open("a", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()

    print(f"  [{hn}] Done", flush=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(cfg: MemePhaseConfig) -> int:
    load_env()

    benchmark_file, language = (
        (cfg.benchmark_file, cfg.language or "en")
        if cfg.benchmark_file
        else resolve_benchmark_data_path(cfg.benchmark, None)
    )
    if cfg.language:
        language = cfg.language

    benchmark = get_benchmark(cfg.benchmark, file_path=benchmark_file, lang=language)

    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.database_root.mkdir(parents=True, exist_ok=True)

    answered = _load_answered_keys(cfg.output)

    # Build per-episode job list
    jobs: List[Tuple[MemoryEpisode, Path]] = []
    for ep in benchmark:
        hn = str(ep.history_name)
        cand_path = cfg.candidates_dir / f"{hn}.json"
        if not cand_path.exists():
            print(f"  WARNING: candidate file not found: {cand_path}", file=sys.stderr)
            continue
        all_qids = {(hn, str(q.metadata.get("question_id", hn))) for q in ep.qas}
        if all_qids and all_qids.issubset(answered):
            print(f"  [{hn}] skip (all answered)", flush=True)
            continue
        jobs.append((ep, cand_path))

    if not jobs:
        print("All episodes already processed. Nothing to do.", flush=True)
        return 0

    print(f"Processing {len(jobs)} episode(s) with method={cfg.update_method}, "
          f"parallel={cfg.parallel_episodes}", flush=True)

    output_lock = threading.Lock()
    workers = max(1, min(cfg.parallel_episodes, len(jobs)))

    def _run_one(ep_path: Tuple[MemoryEpisode, Path]) -> None:
        ep, cand_path = ep_path
        try:
            run_episode_4phase(
                episode=ep,
                cfg=cfg,
                cand_path=cand_path,
                answered=answered,
                output_path=cfg.output,
                output_lock=output_lock,
            )
        except Exception as exc:
            print(f"  ERROR [{ep.history_name}]: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            raise

    if workers == 1:
        for item in tqdm(jobs, desc="4phase", unit="ep"):
            _run_one(item)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_ep = {pool.submit(_run_one, item): item[0].history_name
                            for item in jobs}
            for fut in tqdm(as_completed(future_to_ep),
                            total=len(future_to_ep), desc="4phase", unit="ep"):
                fut.result()  # re-raise on error

    print(f"Done. Output: {cfg.output}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> MemePhaseConfig:
    load_env()
    p = argparse.ArgumentParser(
        description="MEME 4-Phase aligned ingest + answer pipeline"
    )
    p.add_argument("--benchmark", default="meme_filler32k")
    p.add_argument("--benchmark-file", default=None)
    p.add_argument(
        "--candidates-dir", type=Path, required=True,
        help="Per-episode candidate JSON directory (output of extract_candidates.py)"
    )
    p.add_argument(
        "--update-method",
        choices=("relation_decision", "mem0", "add_all", "zep", "amac", "evermemos"),
        required=True,
    )
    p.add_argument(
        "--database-root", type=Path, required=True,
        help="Unfused ingest root; episodes stored as {hn}_before / {hn}_after"
    )
    p.add_argument(
        "--fused-database-root", type=Path, default=None,
        help="Fused output root for relation_decision (auto-derived if omitted)"
    )
    p.add_argument("--output", type=Path, required=True, help="pred.jsonl output path")
    p.add_argument("--answer-model", required=True)
    p.add_argument("--embedding-model",
                   default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    p.add_argument("--embedding-base-url",
                   default=os.getenv("EMBEDDING_BASE_URL", "http://localhost:7110/v1/"))
    p.add_argument("--manager-model", "--relation-llm", dest="manager_model", default="")
    p.add_argument("--language", default="en")
    p.add_argument("--retrieve-topk", type=int, default=50)
    p.add_argument("--memory-token-limit", type=int, default=512)
    p.add_argument("--answer-concurrency", type=int, default=2)
    p.add_argument("--parallel-episodes", type=int, default=1,
                   help="Parallel episode workers (default 1; use with caution — each worker "
                        "builds its own ingest + answer memory instances)")
    p.add_argument("--no-memory-time", action="store_true")
    # hybrid retrieval
    p.add_argument("--hybrid-bm25-dense", action="store_true")
    p.add_argument("--hybrid-dense-weight", type=float, default=0.8)
    p.add_argument("--hybrid-bm25-weight", type=float, default=0.2)
    p.add_argument("--hybrid-pool-mult", type=int, default=4)
    # ingest
    p.add_argument("--relation-concurrency", type=int, default=8)
    p.add_argument("--relation-max-new-tokens", type=int, default=256)
    p.add_argument("--manager-max-new-tokens", type=int, default=2048,
                   dest="manager_max_new_tokens")
    p.add_argument("--related-top-k", type=int, default=3)
    p.add_argument("--mem0-related-top-k", type=int, default=3)
    p.add_argument("--mem0-related-aggregate-max", type=int, default=10)
    p.add_argument("--relation-system-template-en", default="",
                   dest="relation_system_en_template")
    p.add_argument("--relation-system-template-zh", default="",
                   dest="relation_system_zh_template")
    p.add_argument("--relation-user-template", default="",
                   dest="relation_user_template")
    # fusion
    p.add_argument("--fuse-max-new-tokens", type=int, default=512)
    p.add_argument("--fusion-bundle-template-en", default="",
                   dest="fusion_bundle_template_en")
    p.add_argument("--fusion-bundle-template-zh", default="",
                   dest="fusion_bundle_template_zh")
    p.add_argument("--fusion-edge-labels-template-en", default="",
                   dest="fusion_edge_labels_template_en")
    p.add_argument("--fusion-edge-labels-template-zh", default="",
                   dest="fusion_edge_labels_template_zh")
    p.add_argument("--fusion-package-concurrency", type=int, default=4,
                   dest="fusion_package_concurrency")
    # amac
    p.add_argument("--amac-weights", default="0.1,0.1,0.1,0.1,0.6")
    p.add_argument("--amac-threshold", type=float, default=0.55)
    p.add_argument("--amac-skip-utility", action="store_true")
    p.add_argument("--amac-recency-decay-per-step", type=float, default=0.12)
    p.add_argument("--amac-novelty-max-existing", type=int, default=64)
    # evermemos
    p.add_argument("--evermemos-similarity-threshold", type=float, default=0.65)
    p.add_argument("--evermemos-max-time-gap-days", type=float, default=7.0)
    p.add_argument("--evermemos-cluster-concurrency", type=int, default=8,
                   dest="evermemos_cluster_concurrency",
                   help="evermemos: cluster 合并并行线程数（默认 8）")
    # trace
    p.add_argument("--trace-log-dir", default=None)
    # cascade (relation_decision)
    p.add_argument("--cascade-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--deletion-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--topic-aggregation", action=argparse.BooleanOptionalAction, default=True,
                   help="同主题 profile 聚合（基于 candidate_topics 平行数组）")
    p.add_argument("--condition-sim-threshold", type=float, default=0.5)
    p.add_argument("--pairwise-sim-threshold", type=float, default=0.7)

    args = p.parse_args()

    api_key = os.getenv("EMBEDDING_API_KEY", "")
    if not api_key:
        print("ERROR: EMBEDDING_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # relation_decision 现在灌库时就地融合答题记忆 C（同库），不再产出独立的 _fused 目录。
    # 保留 --fused-database-root CLI 以向后兼容，但不再使用。
    fused_root = None

    return MemePhaseConfig(
        benchmark=args.benchmark,
        benchmark_file=args.benchmark_file,
        candidates_dir=args.candidates_dir,
        update_method=args.update_method,
        database_root=args.database_root,
        fused_database_root=fused_root,
        output=args.output,
        answer_model=args.answer_model,
        embedding_model=args.embedding_model,
        embedding_base_url=args.embedding_base_url,
        embedding_api_key=api_key,
        manager_model=args.manager_model,
        language=args.language,
        retrieve_topk=args.retrieve_topk,
        memory_token_limit=args.memory_token_limit,
        answer_concurrency=args.answer_concurrency,
        parallel_episodes=args.parallel_episodes,
        show_memory_time=not bool(args.no_memory_time),
        hybrid_bm25_dense=bool(args.hybrid_bm25_dense),
        hybrid_dense_weight=float(args.hybrid_dense_weight),
        hybrid_bm25_weight=float(args.hybrid_bm25_weight),
        hybrid_pool_mult=int(args.hybrid_pool_mult),
        relation_concurrency=args.relation_concurrency,
        relation_max_new_tokens=args.relation_max_new_tokens,
        manager_max_new_tokens=args.manager_max_new_tokens,
        related_top_k=args.related_top_k,
        mem0_related_top_k=args.mem0_related_top_k,
        mem0_related_aggregate_max=args.mem0_related_aggregate_max,
        relation_system_en_template=args.relation_system_en_template,
        relation_system_zh_template=args.relation_system_zh_template,
        relation_user_template=args.relation_user_template,
        fuse_max_new_tokens=args.fuse_max_new_tokens,
        fusion_bundle_template_en=args.fusion_bundle_template_en,
        fusion_bundle_template_zh=args.fusion_bundle_template_zh,
        fusion_edge_labels_template_en=args.fusion_edge_labels_template_en,
        fusion_edge_labels_template_zh=args.fusion_edge_labels_template_zh,
        fusion_package_concurrency=args.fusion_package_concurrency,
        amac_weights=args.amac_weights,
        amac_threshold=args.amac_threshold,
        amac_skip_utility=args.amac_skip_utility,
        amac_recency_decay_per_step=args.amac_recency_decay_per_step,
        amac_novelty_max_existing=args.amac_novelty_max_existing,
        evermemos_similarity_threshold=args.evermemos_similarity_threshold,
        evermemos_max_time_gap_days=args.evermemos_max_time_gap_days,
        evermemos_cluster_concurrency=args.evermemos_cluster_concurrency,
        cascade_enabled=bool(args.cascade_enabled),
        deletion_enabled=bool(args.deletion_enabled),
        topic_aggregation_enabled=bool(args.topic_aggregation),
        condition_sim_threshold=float(args.condition_sim_threshold),
        pairwise_sim_threshold=float(args.pairwise_sim_threshold),
        trace_log_dir=(args.trace_log_dir or "").strip() or None,
    )


def main() -> None:
    cfg = parse_args()
    sys.exit(run_pipeline(cfg))


if __name__ == "__main__":
    main()
