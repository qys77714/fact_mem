import argparse
import asyncio
import bisect
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from benchmark import get_benchmark
from benchmark.base import MemoryEpisode, QuestionItem
from agent.standard_agent import StandardAgent
from memory import get_memory_system
from memory.base import BaseMemorySystem
from memory.tracing import (
    MemoryTraceLogger,
    _sanitize_for_filename,
    remove_episode_trace_jsonl_files,
    remove_episode_trace_jsonl_files_for_logger,
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
    extractor_model: Optional[str]
    manager_model: Optional[str]
    answer_model: str
    embedding_model: str
    retrieve_topk: int
    memory_token_limit: int
    memory_granularity: str
    database_root: Optional[str]
    embedding_base_url: str
    embedding_api_key: Optional[str]
    language: Optional[str]
    agent_trace_dir: Optional[str]
    parallel_episodes: int
    rebuild_memory: bool
    dialogue_format: str
    manager_max_new_tokens: int
    fact_extract_concurrency: int
    answer_concurrency: int
    question_types: Optional[Set[str]]
    prebuilt_memory: bool
    agent_trace_label: Optional[str]
    hybrid_bm25_dense: bool
    hybrid_dense_weight: float
    hybrid_bm25_weight: float
    hybrid_pool_mult: int
    hybrid_full_corpus_pool: bool
    unfused_rank_database_root: Optional[str]
    rerank_qwen3_vllm: bool
    rerank_qwen3_vllm_base_url: Optional[str]
    rerank_qwen3_vllm_api_key: Optional[str]
    rerank_qwen3_vllm_model: str
    rerank_qwen3_vllm_timeout_s: float
    rerank_top_k: Optional[int]
    answer_stratified_sample: int
    answer_sample_seed: int
    show_memory_time: bool
    require_lme_ingest_marker: bool
    ingest_marker_update_method: str


def _normalize_memory_granularity(value: str) -> str:
    v = str(value).strip().lower()
    if v == "all":
        return "all"
    if v.isdigit() and int(v) > 0:
        return str(int(v))
    raise ValueError("--memory_granularity must be 'all' or a positive integer (e.g., 1/2/3).")


def parse_args() -> GenerateConfig:
    load_env()

    parser = argparse.ArgumentParser(description="统一生成流水线：Benchmark -> Memory -> Agent -> JSONL")
    parser.add_argument("--benchmark", required=True, help="如: lme_oracle / lme_oracle_ku / locomo / lmb_event")
    parser.add_argument("--benchmark_file", default=None, help="可选：自定义 benchmark 数据文件")
    parser.add_argument("--output", required=True, help="输出 jsonl 文件路径")

    parser.add_argument(
        "--method",
        required=True,
        help="记忆方法：lme_prebuilt（预灌向量库，需配合 --prebuilt-memory）",
    )
    parser.add_argument("--extractor_model", default=None, help="记忆抽取模型（预留，当前优先使用 manager_model）")
    parser.add_argument(
        "--manager_model",
        default=None,
        help="记忆管理 LLM（预留；预灌库评测通常不需要）",
    )
    parser.add_argument("--answer_model", required=True, help="回答问题模型")
    parser.add_argument("--embedding_model", required=True, help="向量模型")

    parser.add_argument("--retrieve_topk", type=int, default=5)
    parser.add_argument("--memory_token_limit", type=int, default=8192)
    parser.add_argument(
        "--hybrid-bm25-dense",
        action="store_true",
        help="答题检索：BM25 + dense（FAISS）线性融合（仅 lme_prebuilt 预灌路径）",
    )
    parser.add_argument("--hybrid-dense-weight", type=float, default=0.5, help="混合检索 dense 权重（与 bm25 归一后和为 1）")
    parser.add_argument("--hybrid-bm25-weight", type=float, default=0.5, help="混合检索 BM25 权重")
    parser.add_argument(
        "--hybrid-pool-mult",
        type=int,
        default=4,
        help="混合检索每路候选数：max(retrieve_topk * mult, 50)，上限为库内条数（与 --hybrid-full-corpus-pool 互斥语义：后者为全库）",
    )
    parser.add_argument(
        "--hybrid-full-corpus-pool",
        action="store_true",
        help="混合检索时 dense/BM25 候选池为全库所有记忆（再按分数取 retrieve_topk），不设 top_k*mult 上限",
    )
    parser.add_argument(
        "--memory_granularity",
        default="all",
        help="记忆粒度：'all' 或正整数（如 1/2/3，表示每 N turn 一组）",
    )

    parser.add_argument("--database_root", default=None, help="向量库根目录，默认自动拼接")
    parser.add_argument(
        "--unfused-rank-database-root",
        default=None,
        metavar="DIR",
        help="预灌库评测：在未融合向量库上 hybrid/dense 排序，映射到 --database-root 融合库（去重）",
    )
    parser.add_argument("--embedding_base_url", default=os.getenv("EMBEDDING_BASE_URL", "http://localhost:7110/v1/"))
    parser.add_argument("--embedding_api_key", default=os.getenv("EMBEDDING_API_KEY"))
    parser.add_argument("--language", default=None, help="可选覆盖语言: zh/en")
    parser.add_argument(
        "--agent_trace_dir",
        default="logs/agent_trace",
        help="Agent 答题 tracing 目录，直接在该路径下写 JSONL（不再自动追加子目录）；传空字符串可禁用",
    )
    parser.add_argument(
        "--agent-trace-label",
        default=None,
        metavar="NAME",
        help=(
            "Agent trace 日志文件名前缀（传给 MemoryTraceLogger.method）。"
            "缺省时：若 --output 为 pred_<name>.jsonl 则用 <name>，否则用 --method。"
        ),
    )
    parser.add_argument(
        "--parallel_episodes",
        type=int,
        default=2,
        help="并行处理的 episode 数量，设为 1 时退化为串行",
    )
    parser.add_argument(
        "--rebuild-memory",
        action="store_true",
        help="忽略 .memory_ready.json，强制 clear + 全量重灌向量库",
    )
    parser.add_argument(
        "--prebuilt-memory",
        action="store_true",
        help=(
            "跳过写库：不向 memory 灌入 episode 对话（向量库须已存在，如候选 ingest 产物）。"
            "需同时指定 --method lme_prebuilt 且显式 --database-root。"
        ),
    )
    parser.add_argument(
        "--dialogue-format",
        default="auto",
        choices=["auto", "user_assistant", "named_speakers"],
        help="对话转写模板；auto 对 locomo benchmark 使用 named_speakers",
    )
    parser.add_argument(
        "--manager_max_new_tokens",
        type=int,
        default=2048,
        help="记忆管理 / 事实抽取 LLM 的 max_new_tokens（OpenAI 兼容 API；预灌库评测多未使用）",
    )
    parser.add_argument(
        "--fact-extract-concurrency",
        type=int,
        default=8,
        help="episode 内事实抽取 LLM 的最大并发（1=串行）；与 --parallel-episodes 相乘影响总 QPS",
    )
    parser.add_argument(
        "--answer-concurrency",
        type=int,
        default=2,
        help="答题阶段：同一 episode 内多道问题并发调用回答模型的上限（传给 get_response_chat 的 max_concurrency）",
    )
    parser.add_argument(
        "--no-memory-time",
        action="store_true",
        default=False,
        help="召回记忆不展示时间信息（对应模板变量 show_time=False）",
    )
    parser.add_argument(
        "--question-types",
        default=None,
        metavar="TYPES",
        help=(
            "可选：只评测这些 question_type（逗号分隔，与数据字段一致）。"
            "例 LongMemEval: knowledge-update,temporal-reasoning,multi-session 等"
        ),
    )
    parser.add_argument(
        "--rerank-qwen3-vllm",
        action="store_true",
        help=(
            "答题检索：粗排（dense/BM25 混合或 dense）取 --retrieve_topk 条后，"
            "用本地 vLLM Qwen3-Reranker /v1/score 精排（需 RERANKER_BASE_URL / RERANKER_API_KEY，"
            "见 script/0_run_reranker_ppu.sh）"
        ),
    )
    parser.add_argument(
        "--rerank-qwen3-vllm-base-url",
        default=os.getenv("RERANKER_BASE_URL", "http://localhost:7114/v1/"),
        help="精排服务 OpenAI 兼容 base URL（默认读 RERANKER_BASE_URL）",
    )
    parser.add_argument(
        "--rerank-qwen3-vllm-api-key",
        default=os.getenv("RERANKER_API_KEY"),
        help="精排 API Key（默认读 RERANKER_API_KEY）",
    )
    parser.add_argument(
        "--rerank-qwen3-vllm-model",
        default=os.getenv("RERANKER_MODEL", "Qwen3-Reranker-0.6B"),
        help="精排 served-model-name（默认读 RERANKER_MODEL）",
    )
    parser.add_argument(
        "--rerank-qwen3-vllm-timeout-s",
        type=float,
        default=120.0,
        help="单次精排 HTTP 超时（秒）",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=None,
        metavar="K",
        help="精排后保留条数；默认与 --retrieve_topk 相同（粗排 topK → 精排 topK）",
    )
    parser.add_argument(
        "--answer-stratified-sample",
        type=int,
        default=0,
        metavar="N",
        help=(
            "可选：只评测 N 道题，按 question_type 在题库中的比例分层抽样（最大余数法 + 各层随机）。"
            "0 表示不限制。与 --question-types 同时使用时，先按题型过滤再抽样。"
        ),
    )
    parser.add_argument(
        "--answer-sample-seed",
        type=int,
        default=42,
        help="--answer-stratified-sample 的随机种子（默认可复现）",
    )
    parser.add_argument(
        "--require-lme-ingest-marker",
        action="store_true",
        help=(
            "仅在与 --prebuilt-memory / --database-root 连用时生效："
            "只对 ingest_candidates.py 已成功写入的 episode（目录下存在合法的 "
            ".memory_ready.json，kind=lme_candidate_apply）生成答题预测；"
            "未完成灌库的 episode 跳过（可与分层抽样等筛选同时使用）。"
        ),
    )
    parser.add_argument(
        "--ingest-marker-update-method",
        default="zep",
        metavar="METHOD",
        help=(
            "配合 --require-lme-ingest-marker：匹配的 ingest Candidates update_method "
            "（默认 zep，须与灌库时 --update-method 一致）。"
        ),
    )

    args = parser.parse_args()
    granularity = _normalize_memory_granularity(args.memory_granularity)

    if args.prebuilt_memory:
        if args.method != "lme_prebuilt":
            raise ValueError("--prebuilt-memory 仅支持与 --method lme_prebuilt 连用")
        if not args.database_root:
            raise ValueError("--prebuilt-memory 时必须显式传入 --database_root（预灌库根目录）")
        if args.rebuild_memory:
            raise ValueError("--prebuilt-memory 与 --rebuild-memory 互斥")

    ingest_marker_method = str(args.ingest_marker_update_method or "").strip()
    if not ingest_marker_method:
        raise ValueError("--ingest-marker-update-method 不能为空")
    if args.require_lme_ingest_marker and not args.prebuilt_memory:
        raise ValueError("--require-lme-ingest-marker 仅支持与 --prebuilt-memory 连用")

    return GenerateConfig(
        benchmark=args.benchmark,
        benchmark_file=args.benchmark_file,
        output=args.output,
        method=args.method,
        extractor_model=args.extractor_model,
        manager_model=args.manager_model,
        answer_model=args.answer_model,
        embedding_model=args.embedding_model,
        retrieve_topk=args.retrieve_topk,
        memory_token_limit=args.memory_token_limit,
        memory_granularity=granularity,
        database_root=args.database_root,
        embedding_base_url=args.embedding_base_url,
        embedding_api_key=args.embedding_api_key,
        language=args.language,
        agent_trace_dir=args.agent_trace_dir.strip() or None,
        parallel_episodes=args.parallel_episodes,
        rebuild_memory=bool(args.rebuild_memory),
        dialogue_format=str(args.dialogue_format),
        manager_max_new_tokens=int(args.manager_max_new_tokens),
        fact_extract_concurrency=max(1, int(args.fact_extract_concurrency)),
        answer_concurrency=max(1, int(args.answer_concurrency)),
        question_types=parse_question_types_arg(args.question_types),
        prebuilt_memory=bool(args.prebuilt_memory),
        agent_trace_label=(args.agent_trace_label.strip() or None) if args.agent_trace_label else None,
        hybrid_bm25_dense=bool(args.hybrid_bm25_dense),
        hybrid_dense_weight=float(args.hybrid_dense_weight),
        hybrid_bm25_weight=float(args.hybrid_bm25_weight),
        hybrid_pool_mult=max(1, int(args.hybrid_pool_mult)),
        hybrid_full_corpus_pool=bool(args.hybrid_full_corpus_pool),
        unfused_rank_database_root=(args.unfused_rank_database_root.strip() or None)
        if args.unfused_rank_database_root
        else None,
        rerank_qwen3_vllm=bool(args.rerank_qwen3_vllm),
        rerank_qwen3_vllm_base_url=(
            str(args.rerank_qwen3_vllm_base_url).strip() or None
        )
        if args.rerank_qwen3_vllm_base_url
        else None,
        rerank_qwen3_vllm_api_key=(
            str(args.rerank_qwen3_vllm_api_key).strip() or None
        )
        if args.rerank_qwen3_vllm_api_key
        else None,
        rerank_qwen3_vllm_model=str(args.rerank_qwen3_vllm_model or "Qwen3-Reranker-0.6B").strip(),
        rerank_qwen3_vllm_timeout_s=float(args.rerank_qwen3_vllm_timeout_s),
        rerank_top_k=int(args.rerank_top_k) if args.rerank_top_k is not None else None,
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


def _build_experiment_name(cfg: GenerateConfig) -> str:
    """Build experiment dir name: {benchmark}_gran{gran}_{method}_{model}."""
    model = cfg.answer_model
    safe_model = _sanitize_for_filename(str(model))
    return f"{cfg.benchmark}_gran{cfg.memory_granularity}_{cfg.method}_{safe_model}"


def _resolve_agent_trace_dir(cfg: GenerateConfig) -> Optional[str]:
    """Resolve directory for StandardAgent JSONL traces (writes directly under ``agent_trace_dir``)."""
    if not cfg.agent_trace_dir:
        return None
    return str(Path(cfg.agent_trace_dir))


def _resolve_agent_trace_method(cfg: GenerateConfig) -> str:
    """Filename prefix for agent JSONL trace (MemoryTraceLogger.method)."""
    label = (cfg.agent_trace_label or "").strip()
    if label:
        return _sanitize_for_filename(label)
    stem = Path(cfg.output).stem
    pred_prefix = "pred_"
    if stem.startswith(pred_prefix) and len(stem) > len(pred_prefix):
        return _sanitize_for_filename(stem[len(pred_prefix) :])
    return _sanitize_for_filename(cfg.method)


def _resolve_dialogue_format(cfg: GenerateConfig) -> str:
    choice = (cfg.dialogue_format or "auto").strip().lower()
    if choice in ("user_assistant", "named_speakers"):
        return choice
    b = cfg.benchmark.strip().lower()
    if b == "locomo" or b.startswith("locomo"):
        return "named_speakers"
    return "user_assistant"


def _build_memory_system(cfg: GenerateConfig, language: str):
    method = cfg.method

    if cfg.unfused_rank_database_root and not cfg.prebuilt_memory:
        raise ValueError("--unfused-rank-database-root requires --prebuilt-memory")

    if cfg.prebuilt_memory:
        if not cfg.database_root:
            raise ValueError("prebuilt_memory requires database_root")
        database_root = cfg.database_root
    else:
        database_root = cfg.database_root or f"MemDB/{_build_experiment_name(cfg)}"

    if not cfg.embedding_api_key:
        raise ValueError("EMBEDDING_API_KEY must be set (via env or --embedding_api_key)")

    from openai import OpenAI  # type: ignore

    embed_client = OpenAI(api_key=cfg.embedding_api_key, base_url=cfg.embedding_base_url)

    kwargs: Dict[str, Any] = {
        "granularity": cfg.memory_granularity,
        "llm_client": None,
        "related_memory_top_k": cfg.retrieve_topk,
        "retrieve_topk": cfg.retrieve_topk,
        "trace_log_dir": None,
    }

    return get_memory_system(
        method_name=method,
        embed_model_name=cfg.embedding_model,
        embed_client=embed_client,
        database_root=database_root,
        use_hybrid_retrieval=cfg.hybrid_bm25_dense,
        hybrid_dense_weight=cfg.hybrid_dense_weight,
        hybrid_bm25_weight=cfg.hybrid_bm25_weight,
        hybrid_pool_mult=cfg.hybrid_pool_mult,
        hybrid_full_corpus_pool=cfg.hybrid_full_corpus_pool,
        unfused_rank_database_root=cfg.unfused_rank_database_root,
        language=language,
        rerank_qwen3_vllm=cfg.rerank_qwen3_vllm,
        rerank_qwen3_vllm_base_url=cfg.rerank_qwen3_vllm_base_url,
        rerank_qwen3_vllm_api_key=cfg.rerank_qwen3_vllm_api_key,
        rerank_qwen3_vllm_model=cfg.rerank_qwen3_vllm_model,
        rerank_qwen3_vllm_timeout_s=cfg.rerank_qwen3_vllm_timeout_s,
        rerank_top_k=cfg.rerank_top_k,
        **kwargs,
    )


MEMORY_READY_VERSION = 1


def _episode_memory_fingerprint(episode: MemoryEpisode) -> str:
    """Stable hash: ordered list of (session_date, turn count) per session."""
    payload = [[str(s.session_date), len(s.turns)] for s in episode.sessions]
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _memory_ready_payload(cfg: GenerateConfig, episode: MemoryEpisode) -> Dict[str, Any]:
    return {
        "version": MEMORY_READY_VERSION,
        "num_sessions": len(episode.sessions),
        "fingerprint": _episode_memory_fingerprint(episode),
        "method": cfg.method,
        "memory_granularity": cfg.memory_granularity,
    }


def _read_memory_ready_marker(marker_path: Path) -> Optional[Dict[str, Any]]:
    if not marker_path.is_file():
        return None
    try:
        return json.loads(marker_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ingest_candidates.py 灌库成功后写入的标记（LME_APPLY_*）；与此保持校验字段一致。
_LME_INGEST_MARKER_EXPECT_VERSION = 1
_LME_INGEST_MARKER_EXPECT_KIND = "lme_candidate_apply"


def _episode_has_lme_ingest_marker(
    database_root: Path,
    history_name: str,
    *,
    update_method: str,
) -> bool:
    """True iff candidate ingest wrote a valid ``.memory_ready.json`` for this episode."""
    marker_path = database_root / history_name / ".memory_ready.json"
    data = _read_memory_ready_marker(marker_path)
    if not data:
        return False
    if int(data.get("version", -1)) != _LME_INGEST_MARKER_EXPECT_VERSION:
        return False
    if str(data.get("kind", "")) != _LME_INGEST_MARKER_EXPECT_KIND:
        return False
    if str(data.get("update_method", "")).strip() != update_method.strip():
        return False
    return str(data.get("history_name", "")).strip() == str(history_name).strip()


def _write_memory_ready_marker_atomic(marker_path: Path, payload: Dict[str, Any]) -> None:
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = marker_path.with_suffix(marker_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(marker_path)


def _is_memory_ready(
    memory_system: BaseMemorySystem,
    episode: MemoryEpisode,
) -> bool:
    history_name = str(episode.history_name)
    marker_path = memory_system.memory_ready_marker_path(history_name)
    if marker_path is None:
        return True
    data = _read_memory_ready_marker(marker_path)
    if not data:
        return False
    if int(data.get("version", -1)) != MEMORY_READY_VERSION:
        return False
    if int(data.get("num_sessions", -1)) != len(episode.sessions):
        return False
    if str(data.get("fingerprint", "")) != _episode_memory_fingerprint(episode):
        return False
    return True


def _question_id_for_episode(history_name: str, question) -> str:
    return str(question.metadata.get("question_id", history_name))


def _load_answered_keys(output_path: Path) -> Set[Tuple[str, str]]:
    """Keys (history_name, question_id) for lines already in the output JSONL."""
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


def _cleanup_interrupted_episode(
    memory_system: BaseMemorySystem,
    history_name: str,
) -> None:
    """Clear MemDB and memory trace for an interrupted episode (do not touch Agent trace)."""
    if hasattr(memory_system, "clear"):
        memory_system.clear(history_name)
    if hasattr(memory_system, "trace"):
        remove_episode_trace_jsonl_files_for_logger(memory_system.trace, history_name)


def _should_clear_agent_trace_for_resume(
    cfg: GenerateConfig,
    history_name: str,
    episode: MemoryEpisode,
    pending_qas: List[QuestionItem],
    answered: Set[Tuple[str, str]],
    scope_keys: Optional[Set[Tuple[str, str]]] = None,
) -> bool:
    """True if this episode should drop existing agent JSONL before answering pending questions."""
    if not (cfg.agent_trace_dir or "").strip():
        return False
    h = str(history_name)
    all_qs = filter_question_items(episode.qas, cfg.question_types)
    if scope_keys is not None:
        all_qs = [
            q for q in all_qs if (h, _question_id_for_episode(h, q)) in scope_keys
        ]
    if not all_qs or not pending_qas:
        return False
    answered_in_scope = [
        q
        for q in all_qs
        if (h, _question_id_for_episode(h, q)) in answered
    ]
    if len(answered_in_scope) > 0:
        return True
    agent_trace_dir = Path(cfg.agent_trace_dir)
    method = _resolve_agent_trace_method(cfg)
    root = MemoryTraceLogger(
        method=method,
        log_dir=str(agent_trace_dir),
        use_experiment_naming=True,
    )
    primary = root.get_trace_path(history_name)
    if primary.exists() and primary.stat().st_size > 0:
        return True
    safe = _sanitize_for_filename(history_name)
    for p in agent_trace_dir.glob(f"{safe}*.jsonl"):
        if p.is_file() and p.stat().st_size > 0:
            return True
    return False


def _build_record(benchmark_name: str, history_name: str, question, model_answer: Optional[str]) -> Dict:
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

    return record


class _TqdmSlotPool:
    """Assign distinct tqdm ``position`` values so parallel episode workers don't garble bars."""

    def __init__(self, n_slots: int):
        n = max(1, int(n_slots))
        self._lock = threading.Lock()
        self._free: List[int] = list(range(1, n + 1))

    def acquire(self) -> int:
        with self._lock:
            if not self._free:
                return 1
            return self._free.pop(0)

    def release(self, slot: int) -> None:
        with self._lock:
            bisect.insort(self._free, slot)


def _store_sessions_sync(
    memory_system: BaseMemorySystem,
    history_name: str,
    sessions: list,
    tqdm_slot_pool: _TqdmSlotPool,
) -> None:
    """Synchronous helper for run_in_executor: store all sessions for an episode."""
    if not sessions:
        return
    label = str(history_name)
    if len(label) > 28:
        label = label[:27] + "…"
    slot = tqdm_slot_pool.acquire()
    try:
        with tqdm(
            total=len(sessions),
            desc=f"Memory [{label}]",
            position=slot,
            leave=False,
            unit="session",
        ) as pbar:
            memory_system.store_episode(history_name, sessions, session_progress=pbar)
    finally:
        tqdm_slot_pool.release(slot)


async def _process_episode(
    cfg: GenerateConfig,
    episode: MemoryEpisode,
    episode_idx: int,
    agent: StandardAgent,
    memory_system: BaseMemorySystem,
    loop: asyncio.AbstractEventLoop,
    executor: ThreadPoolExecutor,
    semaphore: asyncio.Semaphore,
    pbar: tqdm,
    tqdm_slot_pool: _TqdmSlotPool,
    output_path: Path,
    output_lock: threading.Lock,
    pending_qas: List[QuestionItem],
    answered: Set[Tuple[str, str]],
    scope_keys: Optional[Set[Tuple[str, str]]] = None,
) -> None:
    """Store sessions when memory not ready (or --rebuild-memory), then answer pending questions only."""
    history_name = str(episode.history_name)
    if cfg.prebuilt_memory:
        need_store = False
    else:
        need_store = cfg.rebuild_memory or not _is_memory_ready(memory_system, episode)

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
        if need_store:
            _cleanup_interrupted_episode(memory_system, history_name)
            await loop.run_in_executor(
                executor,
                _store_sessions_sync,
                memory_system,
                history_name,
                episode.sessions,
                tqdm_slot_pool,
            )
            marker_path = memory_system.memory_ready_marker_path(history_name)
            if marker_path is not None:
                _write_memory_ready_marker_atomic(marker_path, _memory_ready_payload(cfg, episode))

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
                q
                for q in pending_qas
                if (h, _question_id_for_episode(h, q)) in sample_keys
            ]
        if pending_qas:
            episodes_to_process.append((idx, episode, pending_qas))

    if cfg.require_lme_ingest_marker:
        root_str = (cfg.database_root or "").strip()
        if not root_str:
            raise ValueError(
                "--require-lme-ingest-marker 需要有效的 --database_root（预灌库根目录）"
            )
        db_root = Path(root_str)
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

    loop = asyncio.get_running_loop()
    semaphore = asyncio.Semaphore(cfg.parallel_episodes)
    output_lock = threading.Lock()
    pbar = tqdm(
        total=len(episodes_to_process),
        desc="Generating (episodes with pending questions)",
        position=0,
        leave=True,
    )
    tqdm_slot_pool = _TqdmSlotPool(cfg.parallel_episodes)

    with ThreadPoolExecutor(max_workers=cfg.parallel_episodes) as executor:
        tasks = [
            _process_episode(
                cfg=cfg,
                episode=episode,
                episode_idx=idx,
                agent=agent,
                memory_system=memory_system,
                loop=loop,
                executor=executor,
                semaphore=semaphore,
                pbar=pbar,
                tqdm_slot_pool=tqdm_slot_pool,
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
