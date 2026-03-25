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
from memory.tracing import _sanitize_for_filename
from utils.env import load_env
from utils.llm_api import load_api_chat_completion


BENCHMARK_TO_DATASET: Dict[str, Tuple[str, str]] = {
    "test": ("data/preprocessed/test.json", "zh"),
    "lme_o": ("data/preprocessed/longmemeval_oracle_converted.json", "en"),
    "lme_s": ("data/preprocessed/longmemeval_s_cleaned_converted.json", "en"),
    "locomo": ("data/raw_data/locomo10.json", "en"),
    "lmb_event": ("data/preprocessed/LifeMemBench_event.json", "zh"),
    "emb_event": ("data/preprocessed/EgoMemBench_event_half.json", "en"),
}


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
    mem0_dialogue_format: str
    manager_max_new_tokens: int
    mem0_extract_concurrency: int


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
        help="记忆方法，如 amem / mem0 / mem0_nodel / rag / full_context",
    )
    parser.add_argument("--extractor_model", default=None, help="记忆抽取模型（预留，当前优先使用 manager_model）")
    parser.add_argument(
        "--manager_model",
        default=None,
        help="记忆管理模型（amem / mem0 / mem0_nodel 必填其一）",
    )
    parser.add_argument("--answer_model", required=True, help="回答问题模型")
    parser.add_argument("--embedding_model", required=True, help="向量模型")

    parser.add_argument("--retrieve_topk", type=int, default=5)
    parser.add_argument("--memory_token_limit", type=int, default=8192)
    parser.add_argument(
        "--memory_granularity",
        default="all",
        help="记忆粒度：'all' 或正整数（如 1/2/3，表示每 N turn 一组）",
    )

    parser.add_argument("--database_root", default=None, help="向量库根目录，默认自动拼接")
    parser.add_argument("--embedding_base_url", default=os.getenv("EMBEDDING_BASE_URL", "http://localhost:7110/v1/"))
    parser.add_argument("--embedding_api_key", default=os.getenv("EMBEDDING_API_KEY"))
    parser.add_argument("--language", default=None, help="可选覆盖语言: zh/en")
    parser.add_argument(
        "--agent_trace_dir",
        default="logs/agent_trace",
        help="Agent 答题 tracing 日志目录，默认开启；传空字符串可禁用",
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
        "--mem0-dialogue-format",
        default="auto",
        choices=["auto", "user_assistant", "named_speakers"],
        help="mem0：对话转写与事实抽取模板；auto 对 locomo benchmark 使用 named_speakers",
    )
    parser.add_argument(
        "--manager_max_new_tokens",
        type=int,
        default=2048,
        help="amem/mem0 记忆管理 LLM 的 max_new_tokens（传给 OpenAI 兼容 API 的 max_tokens）",
    )
    parser.add_argument(
        "--mem0-extract-concurrency",
        type=int,
        default=8,
        help="mem0/mem0_nodel：episode 内事实抽取 LLM 调用的最大并发（1=串行）；与 --parallel-episodes 相乘影响总 QPS",
    )

    args = parser.parse_args()
    granularity = _normalize_memory_granularity(args.memory_granularity)

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
        mem0_dialogue_format=str(args.mem0_dialogue_format),
        manager_max_new_tokens=int(args.manager_max_new_tokens),
        mem0_extract_concurrency=max(1, int(args.mem0_extract_concurrency)),
    )


def _resolve_benchmark(cfg: GenerateConfig) -> Tuple[str, str]:
    if cfg.benchmark_file:
        return cfg.benchmark_file, (cfg.language or "en")

    if cfg.benchmark not in BENCHMARK_TO_DATASET:
        supported = ", ".join(sorted(BENCHMARK_TO_DATASET.keys()))
        raise ValueError(
            f"Unknown benchmark '{cfg.benchmark}'. Please provide --benchmark_file, "
            f"or choose one of: {supported}"
        )

    file_path, default_lang = BENCHMARK_TO_DATASET[cfg.benchmark]
    return file_path, (cfg.language or default_lang)


def _build_experiment_name(cfg: GenerateConfig) -> str:
    """Build experiment dir name: {benchmark}_gran{gran}_{method}_{model}."""
    if cfg.method in {"amem", "mem0", "mem0_nodel"}:
        model = cfg.manager_model or cfg.extractor_model or "default"
    else:
        model = cfg.answer_model
    safe_model = _sanitize_for_filename(str(model))
    return f"{cfg.benchmark}_gran{cfg.memory_granularity}_{cfg.method}_{safe_model}"


def _build_memory_trace_dir(cfg: GenerateConfig) -> str:
    """Build memory trace dir: logs/memory_trace/{experiment_name}."""
    return f"logs/memory_trace/{_build_experiment_name(cfg)}"


def _resolve_mem0_dialogue_format(cfg: GenerateConfig) -> str:
    choice = (cfg.mem0_dialogue_format or "auto").strip().lower()
    if choice in ("user_assistant", "named_speakers"):
        return choice
    b = cfg.benchmark.strip().lower()
    if b == "locomo" or b.startswith("locomo"):
        return "named_speakers"
    return "user_assistant"


def _build_memory_system(cfg: GenerateConfig, language: str):
    method = cfg.method

    database_root = cfg.database_root or f"MemDB/{_build_experiment_name(cfg)}"

    if not cfg.embedding_api_key:
        raise ValueError("EMBEDDING_API_KEY must be set (via env or --embedding_api_key)")

    from openai import OpenAI  # type: ignore

    embed_client = OpenAI(api_key=cfg.embedding_api_key, base_url=cfg.embedding_base_url)

    llm_client = None
    trace_log_dir = None
    if method in {"amem", "mem0", "mem0_nodel"}:
        manager_or_extractor_model = cfg.manager_model or cfg.extractor_model
        if not manager_or_extractor_model:
            raise ValueError(
                "For method 'amem'/'mem0'/'mem0_nodel', please provide --manager_model (or --extractor_model)."
            )
        llm_client = load_api_chat_completion(manager_or_extractor_model, async_=False)
        trace_log_dir = _build_memory_trace_dir(cfg)

    kwargs: Dict[str, Any] = {
        "granularity": cfg.memory_granularity,
        "llm_client": llm_client,
        "related_memory_top_k": cfg.retrieve_topk,
        "retrieve_topk": cfg.retrieve_topk,
        "language": language,
        "trace_log_dir": trace_log_dir,
    }
    if method in ("mem0", "mem0_nodel"):
        kwargs["dialogue_format"] = _resolve_mem0_dialogue_format(cfg)
        kwargs["extract_concurrency"] = cfg.mem0_extract_concurrency
    if method in {"amem", "mem0", "mem0_nodel"}:
        kwargs["manager_max_new_tokens"] = cfg.manager_max_new_tokens

    return get_memory_system(
        method_name=method,
        embed_model_name=cfg.embedding_model,
        embed_client=embed_client,
        database_root=database_root,
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
    if hasattr(memory_system, "trace") and hasattr(memory_system.trace, "get_trace_path"):
        trace_path = memory_system.trace.get_trace_path(history_name)
        if trace_path.exists():
            trace_path.unlink()


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
) -> None:
    """Store sessions when memory not ready (or --rebuild-memory), then answer pending questions only."""
    history_name = str(episode.history_name)
    need_store = cfg.rebuild_memory or not _is_memory_ready(memory_system, episode)

    async with semaphore:
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
    episodes_to_process: List[Tuple[int, MemoryEpisode, List[QuestionItem]]] = []
    for idx, episode in enumerate(benchmark):
        h = str(episode.history_name)
        pending_qas = [q for q in episode.qas if (h, _question_id_for_episode(h, q)) not in answered]
        if pending_qas:
            episodes_to_process.append((idx, episode, pending_qas))

    if not episodes_to_process:
        return

    memory_system = _build_memory_system(cfg, language=language)
    answer_chat_model = load_api_chat_completion(cfg.answer_model, async_=True)

    agent = StandardAgent(
        memory_system=memory_system,
        chat_model=answer_chat_model,
        memory_token_limit=cfg.memory_token_limit,
        language=language,
        trace_log_dir=cfg.agent_trace_dir,
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
            )
            for idx, episode, pending_qas in episodes_to_process
        ]
        await asyncio.gather(*tasks)


def main() -> None:
    cfg = parse_args()
    asyncio.run(run_pipeline(cfg))


if __name__ == "__main__":
    main()
