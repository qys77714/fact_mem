import argparse
import asyncio
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from benchmark import get_benchmark
from benchmark.base import MemoryEpisode, QuestionItem
from agent.standard_agent import StandardAgent
from memory import get_memory_system
from memory.tracing import (
    MemoryTraceLogger,
    _sanitize_for_filename,
    remove_episode_trace_jsonl_files,
)
from utils.env import load_env
from utils.llm_api import load_api_chat_completion
from utils.question_filter import (
    filter_question_items,
    parse_question_types_arg,
    stratified_sample_by_question_type,
)

from benchmark.datasets import DEFAULT_BENCHMARK_DATASETS


@dataclass
class GenerateConfig:
    benchmark: str
    benchmark_file: Optional[str]
    output: str
    method: str
    answer_model: str
    embedding_model: str
    retrieve_topk: int
    memory_token_limit: int
    database_root: str
    embedding_base_url: str
    embedding_api_key: Optional[str]
    language: Optional[str]
    agent_trace_dir: Optional[str]
    parallel_episodes: int
    answer_concurrency: int
    question_types: Optional[Set[str]]
    hybrid_bm25_dense: bool
    hybrid_dense_weight: float
    hybrid_bm25_weight: float
    hybrid_pool_mult: int
    answer_stratified_sample: int
    answer_sample_seed: int
    show_memory_time: bool
    require_lme_ingest_marker: bool
    ingest_marker_update_method: str


def parse_args() -> GenerateConfig:
    load_env()

    parser = argparse.ArgumentParser(description="生成流水线：预灌向量库 → Agent 答题 → JSONL")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--benchmark_file", default=None)
    parser.add_argument("--output", required=True, help="输出 JSONL 文件路径")
    parser.add_argument(
        "--method",
        required=True,
        help="记忆方法：目前仅支持 prebuilt（预灌向量库）",
    )
    parser.add_argument("--answer_model", required=True, help="答题 LLM")
    parser.add_argument("--embedding_model", required=True, help="Embedding 模型名")
    parser.add_argument("--retrieve_topk", type=int, default=5, help="召回记忆条数")
    parser.add_argument("--memory_token_limit", type=int, default=8192, help="传入 Agent 的记忆 token 上限")
    parser.add_argument(
        "--hybrid-bm25-dense",
        action="store_true",
        help="BM25 + dense 混合检索",
    )
    parser.add_argument("--hybrid-dense-weight", type=float, default=0.5)
    parser.add_argument("--hybrid-bm25-weight", type=float, default=0.5)
    parser.add_argument(
        "--hybrid-pool-mult",
        type=int,
        default=4,
        help="混合检索每路候选数：max(retrieve_topk * mult, 50)",
    )
    parser.add_argument(
        "--database_root",
        required=True,
        help="预灌向量库根目录（ingest_candidates.py 的产出目录）",
    )
    parser.add_argument(
        "--embedding_base_url",
        default=os.getenv("EMBEDDING_BASE_URL", "http://localhost:7110/v1/"),
    )
    parser.add_argument("--embedding_api_key", default=os.getenv("EMBEDDING_API_KEY"))
    parser.add_argument("--language", default=None, help="覆盖语言：zh / en")
    parser.add_argument(
        "--agent_trace_dir",
        default="logs/agent_trace",
        help="Agent 答题 trace 目录（传空字符串可禁用）",
    )
    parser.add_argument(
        "--parallel_episodes",
        type=int,
        default=2,
        help="并行处理的 episode 数量",
    )
    parser.add_argument(
        "--answer-concurrency",
        type=int,
        default=2,
        help="同一 episode 内多道问题并发调用的上限",
    )
    parser.add_argument(
        "--no-memory-time",
        action="store_true",
        help="召回记忆不展示时间信息",
    )
    parser.add_argument(
        "--question-types",
        default=None,
        metavar="TYPES",
        help="只评测这些 question_type（逗号分隔）",
    )
    parser.add_argument(
        "--answer-stratified-sample",
        type=int,
        default=0,
        metavar="N",
        help="按 question_type 分层抽样 N 道题（0 = 全量）",
    )
    parser.add_argument("--answer-sample-seed", type=int, default=42)
    parser.add_argument(
        "--require-lme-ingest-marker",
        action="store_true",
        help="仅对 ingest_candidates.py 已完成的 episode 生成预测（跳过未灌库的 episode）",
    )
    parser.add_argument(
        "--ingest-marker-update-method",
        default="zep",
        metavar="METHOD",
        help="配合 --require-lme-ingest-marker：ingest 时使用的 --update-method",
    )

    args = parser.parse_args()

    if args.method != "prebuilt":
        raise ValueError(f"--method 目前仅支持 prebuilt，收到 {args.method!r}")

    ingest_marker_method = str(args.ingest_marker_update_method or "").strip()
    if not ingest_marker_method:
        raise ValueError("--ingest-marker-update-method 不能为空")

    return GenerateConfig(
        benchmark=args.benchmark,
        benchmark_file=args.benchmark_file,
        output=args.output,
        method=args.method,
        answer_model=args.answer_model,
        embedding_model=args.embedding_model,
        retrieve_topk=args.retrieve_topk,
        memory_token_limit=args.memory_token_limit,
        database_root=args.database_root,
        embedding_base_url=args.embedding_base_url,
        embedding_api_key=args.embedding_api_key,
        language=args.language,
        agent_trace_dir=(args.agent_trace_dir.strip() or None),
        parallel_episodes=args.parallel_episodes,
        answer_concurrency=max(1, int(args.answer_concurrency)),
        question_types=parse_question_types_arg(args.question_types),
        hybrid_bm25_dense=bool(args.hybrid_bm25_dense),
        hybrid_dense_weight=float(args.hybrid_dense_weight),
        hybrid_bm25_weight=float(args.hybrid_bm25_weight),
        hybrid_pool_mult=max(1, int(args.hybrid_pool_mult)),
        answer_stratified_sample=max(0, int(args.answer_stratified_sample)),
        answer_sample_seed=int(args.answer_sample_seed),
        show_memory_time=not bool(args.no_memory_time),
        require_lme_ingest_marker=bool(args.require_lme_ingest_marker),
        ingest_marker_update_method=ingest_marker_method,
    )


def _resolve_benchmark(cfg: GenerateConfig) -> Tuple[str, str]:
    if cfg.benchmark_file:
        return cfg.benchmark_file, (cfg.language or "en")

    if cfg.benchmark not in DEFAULT_BENCHMARK_DATASETS:
        supported = ", ".join(sorted(DEFAULT_BENCHMARK_DATASETS.keys()))
        raise ValueError(
            f"Unknown benchmark '{cfg.benchmark}'. Please provide --benchmark_file, "
            f"or choose one of: {supported}"
        )

    file_path, default_lang = DEFAULT_BENCHMARK_DATASETS[cfg.benchmark]
    return file_path, (cfg.language or default_lang)


def _resolve_agent_trace_dir(cfg: GenerateConfig) -> Optional[str]:
    if not cfg.agent_trace_dir:
        return None
    return str(Path(cfg.agent_trace_dir))


def _resolve_agent_trace_method(cfg: GenerateConfig) -> str:
    stem = Path(cfg.output).stem
    pred_prefix = "pred_"
    if stem.startswith(pred_prefix) and len(stem) > len(pred_prefix):
        return _sanitize_for_filename(stem[len(pred_prefix):])
    return _sanitize_for_filename(cfg.method)


def _build_memory_system(cfg: GenerateConfig, language: str):
    if not cfg.database_root:
        raise ValueError("--database_root is required")
    if not cfg.embedding_api_key:
        raise ValueError("EMBEDDING_API_KEY must be set (via env or --embedding_api_key)")

    from openai import OpenAI  # type: ignore

    embed_client = OpenAI(api_key=cfg.embedding_api_key, base_url=cfg.embedding_base_url)

    return get_memory_system(
        method_name=cfg.method,
        embed_model_name=cfg.embedding_model,
        embed_client=embed_client,
        database_root=cfg.database_root,
        use_hybrid_retrieval=cfg.hybrid_bm25_dense,
        hybrid_dense_weight=cfg.hybrid_dense_weight,
        hybrid_bm25_weight=cfg.hybrid_bm25_weight,
        hybrid_pool_mult=cfg.hybrid_pool_mult,
        language=language,
        granularity="all",
        llm_client=None,
        related_memory_top_k=cfg.retrieve_topk,
        retrieve_topk=cfg.retrieve_topk,
        trace_log_dir=None,
    )


# ---------------------------------------------------------------------------
# Ingest marker check (safety: skip episodes whose DB wasn't built yet)
# ---------------------------------------------------------------------------

_LME_INGEST_MARKER_EXPECT_VERSION = 1
_LME_INGEST_MARKER_EXPECT_KIND = "lme_candidate_apply"


def _read_marker(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _episode_has_lme_ingest_marker(
    database_root: Path,
    history_name: str,
    *,
    update_method: str,
) -> bool:
    data = _read_marker(database_root / history_name / ".memory_ready.json")
    if not data:
        return False
    if int(data.get("version", -1)) != _LME_INGEST_MARKER_EXPECT_VERSION:
        return False
    if str(data.get("kind", "")) != _LME_INGEST_MARKER_EXPECT_KIND:
        return False
    if str(data.get("update_method", "")).strip() != update_method.strip():
        return False
    return str(data.get("history_name", "")).strip() == str(history_name).strip()


# ---------------------------------------------------------------------------
# Answer key tracking
# ---------------------------------------------------------------------------

def _question_id_for_episode(history_name: str, question: QuestionItem) -> str:
    return str(question.metadata.get("question_id", history_name))


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
                record = json.loads(line)
                if "history_name" not in record:
                    continue
                h = str(record["history_name"])
                qid = record.get("question_id")
                answered.add((h, str(qid if qid is not None else h)))
            except json.JSONDecodeError:
                continue
    return answered


# ---------------------------------------------------------------------------
# Agent trace cleanup on resume
# ---------------------------------------------------------------------------

def _should_clear_agent_trace_for_resume(
    cfg: GenerateConfig,
    history_name: str,
    episode: MemoryEpisode,
    pending_qas: List[QuestionItem],
    answered: Set[Tuple[str, str]],
    scope_keys: Optional[Set[Tuple[str, str]]] = None,
) -> bool:
    if not (cfg.agent_trace_dir or "").strip():
        return False
    h = str(history_name)
    all_qs = filter_question_items(episode.qas, cfg.question_types)
    if scope_keys is not None:
        all_qs = [q for q in all_qs if (h, _question_id_for_episode(h, q)) in scope_keys]
    if not all_qs or not pending_qas:
        return False
    answered_in_scope = [
        q for q in all_qs if (h, _question_id_for_episode(h, q)) in answered
    ]
    if answered_in_scope:
        return True
    agent_trace_dir = Path(cfg.agent_trace_dir)
    method = _resolve_agent_trace_method(cfg)
    root = MemoryTraceLogger(method=method, log_dir=str(agent_trace_dir), use_experiment_naming=True)
    primary = root.get_trace_path(history_name)
    if primary.exists() and primary.stat().st_size > 0:
        return True
    safe = _sanitize_for_filename(history_name)
    for p in agent_trace_dir.glob(f"{safe}*.jsonl"):
        if p.is_file() and p.stat().st_size > 0:
            return True
    return False


# ---------------------------------------------------------------------------
# Record building
# ---------------------------------------------------------------------------

def _build_record(
    benchmark_name: str, history_name: str, question: QuestionItem, model_answer: Optional[str]
) -> Dict:
    record = {
        "benchmark": benchmark_name,
        "history_name": history_name,
        "question_id": question.metadata.get("question_id", history_name),
        "question": question.question,
        "answer": question.answer,
        "model_answer": model_answer or "",
        "question_type": question.question_type,
        "question_time": question.question_time,
    }
    if question.options is not None:
        record["options"] = question.options
    if "golden_option" in question.metadata:
        record["golden_option"] = question.metadata["golden_option"]
    for field in ("phase", "entity_key", "entity_values", "max_session_index", "hop"):
        if field in question.metadata:
            record[field] = question.metadata[field]
    return record


# ---------------------------------------------------------------------------
# Core async episode processor
# ---------------------------------------------------------------------------

async def _process_episode(
    cfg: GenerateConfig,
    episode: MemoryEpisode,
    agent: StandardAgent,
    semaphore: asyncio.Semaphore,
    pbar: tqdm,
    output_path: Path,
    output_lock: threading.Lock,
    pending_qas: List[QuestionItem],
    answered: Set[Tuple[str, str]],
    scope_keys: Optional[Set[Tuple[str, str]]] = None,
) -> None:
    history_name = str(episode.history_name)

    async with semaphore:
        if _should_clear_agent_trace_for_resume(
            cfg, history_name, episode, pending_qas, answered, scope_keys
        ):
            remove_episode_trace_jsonl_files(
                log_dir=Path(cfg.agent_trace_dir),
                method=_resolve_agent_trace_method(cfg),
                history_name=history_name,
                use_experiment_naming=True,
            )

        responses = await agent.batch_answer_questions(
            history_name=history_name,
            questions=pending_qas,
            top_k=cfg.retrieve_topk,
        )

        records = [
            _build_record(cfg.benchmark, history_name, q, ans)
            for q, ans in zip(pending_qas, responses)
        ]

        with output_lock:
            with output_path.open("a", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

        pbar.update(1)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

async def run_pipeline(cfg: GenerateConfig) -> None:
    benchmark_file, language = _resolve_benchmark(cfg)

    benchmark = get_benchmark(
        task_name=cfg.benchmark,
        file_path=benchmark_file,
        lang=language,
    )

    output_path = Path(cfg.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    answered = _load_answered_keys(output_path)
    sample_keys: Optional[Set[Tuple[str, str]]] = None
    if cfg.answer_stratified_sample > 0:
        keyed: List[Tuple[Tuple[str, str], Optional[str]]] = []
        for episode in benchmark:
            h = str(episode.history_name)
            for q in filter_question_items(episode.qas, cfg.question_types):
                qid = _question_id_for_episode(h, q)
                keyed.append(((h, qid), q.question_type))
        sample_keys = stratified_sample_by_question_type(
            keyed,
            cfg.answer_stratified_sample,
            cfg.answer_sample_seed,
        )

    episodes_to_process: List[Tuple[int, MemoryEpisode, List[QuestionItem]]] = []
    for idx, episode in enumerate(benchmark):
        h = str(episode.history_name)
        pending_qas = [q for q in episode.qas if (h, _question_id_for_episode(h, q)) not in answered]
        pending_qas = filter_question_items(pending_qas, cfg.question_types)
        if sample_keys is not None:
            pending_qas = [
                q for q in pending_qas
                if (h, _question_id_for_episode(h, q)) in sample_keys
            ]
        if pending_qas:
            episodes_to_process.append((idx, episode, pending_qas))

    if cfg.require_lme_ingest_marker:
        db_root = Path(cfg.database_root)
        meth = cfg.ingest_marker_update_method
        n_before = len(episodes_to_process)
        episodes_to_process = [
            item
            for item in episodes_to_process
            if _episode_has_lme_ingest_marker(db_root, str(item[1].history_name), update_method=meth)
        ]
        n_skip = n_before - len(episodes_to_process)
        if n_skip:
            print(
                f"Generating: skipped {n_skip} episode(s) without LME ingest marker "
                f"({meth!r}) under {db_root}",
                flush=True,
            )

    if not episodes_to_process:
        return

    memory_system = _build_memory_system(cfg, language=language)
    answer_chat_model = load_api_chat_completion(cfg.answer_model, async_=True)

    agent = StandardAgent(
        memory_system=memory_system,
        chat_model=answer_chat_model,
        memory_token_limit=cfg.memory_token_limit,
        language=language,
        trace_log_dir=_resolve_agent_trace_dir(cfg),
        trace_method=_resolve_agent_trace_method(cfg),
        answer_concurrency=cfg.answer_concurrency,
        show_time=cfg.show_memory_time,
    )

    semaphore = asyncio.Semaphore(cfg.parallel_episodes)
    output_lock = threading.Lock()
    pbar = tqdm(
        total=len(episodes_to_process),
        desc="Generating (episodes with pending questions)",
        position=0,
        leave=True,
    )

    tasks = [
        _process_episode(
            cfg=cfg,
            episode=episode,
            agent=agent,
            semaphore=semaphore,
            pbar=pbar,
            output_path=output_path,
            output_lock=output_lock,
            pending_qas=pending_qas,
            answered=answered,
            scope_keys=sample_keys,
        )
        for idx, episode, pending_qas in episodes_to_process
    ]
    await asyncio.gather(*tasks)


def main() -> None:
    cfg = parse_args()
    asyncio.run(run_pipeline(cfg))


if __name__ == "__main__":
    main()
