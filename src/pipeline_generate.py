import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from tqdm import tqdm

from benchmark import get_benchmark
from agent.standard_agent import StandardAgent
from memory import get_memory_system
from utils.env import load_env
from utils.llm_api import load_api_chat_completion


BENCHMARK_TO_DATASET: Dict[str, Tuple[str, str]] = {
    "test": ("data/preprocessed/test.json", "zh"),
    "lme_oracle": ("data/preprocessed/longmemeval_oracle_converted.json", "en"),
    "lme_s": ("data/preprocessed/longmemeval_s_cleaned.json", "en"),
    "locomo": ("data/preprocessed/locomo10_converted.json", "en"),
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

    parser.add_argument("--method", required=True, help="记忆方法，如 amem / mem0 / rag / full_context")
    parser.add_argument("--extractor_model", default=None, help="记忆抽取模型（预留，当前优先使用 manager_model）")
    parser.add_argument("--manager_model", default=None, help="记忆管理模型（amem/mem0 必填其一）")
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


def _build_memory_system(cfg: GenerateConfig, language: str):
    method = cfg.method

    database_root = cfg.database_root or f"MemDB/{cfg.benchmark}/{cfg.answer_model}_{cfg.memory_granularity}/{method}"

    if not cfg.embedding_api_key:
        raise ValueError("EMBEDDING_API_KEY must be set (via env or --embedding_api_key)")

    from openai import OpenAI  # type: ignore

    embed_client = OpenAI(api_key=cfg.embedding_api_key, base_url=cfg.embedding_base_url)

    llm_client = None
    if method in {"amem", "mem0"}:
        manager_or_extractor_model = cfg.manager_model or cfg.extractor_model
        if not manager_or_extractor_model:
            raise ValueError("For method 'amem'/'mem0', please provide --manager_model (or --extractor_model).")
        llm_client = load_api_chat_completion(manager_or_extractor_model, async_=False)

    return get_memory_system(
        method_name=method,
        embed_model_name=cfg.embedding_model,
        embed_client=embed_client,
        database_root=database_root,
        granularity=cfg.memory_granularity,
        llm_client=llm_client,
        related_memory_top_k=cfg.retrieve_topk,
        retrieve_topk=cfg.retrieve_topk,
        language=language,
    )


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


async def run_pipeline(cfg: GenerateConfig) -> None:
    benchmark_file, language = _resolve_benchmark(cfg)

    benchmark = get_benchmark(
        task_name=cfg.benchmark,
        file_path=benchmark_file,
        lang=language,
    )

    memory_system = _build_memory_system(cfg, language=language)
    answer_chat_model = load_api_chat_completion(cfg.answer_model, async_=True)

    agent = StandardAgent(
        memory_system=memory_system,
        chat_model=answer_chat_model,
        memory_token_limit=cfg.memory_token_limit,
        language=language,
        trace_log_dir=cfg.agent_trace_dir,
    )

    output_path = Path(cfg.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for episode in tqdm(benchmark, total=len(benchmark), desc="Generating"):
            # 1) 自动记忆抽取/管理：按 session 逐个写入
            for session_idx, session in enumerate(episode.sessions, start=1):
                memory_system.store_session(
                    history_name=str(episode.history_name),
                    session_idx=session_idx,
                    session=session,
                )

            # 2) 自动回答问题：统一从 memory 检索后作答
            responses = await agent.batch_answer_questions(
                history_name=str(episode.history_name),
                questions=episode.qas,
                top_k=cfg.retrieve_topk,
            )

            for q, ans in zip(episode.qas, responses):
                record = _build_record(cfg.benchmark, str(episode.history_name), q, ans)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            f.flush()


def main() -> None:
    cfg = parse_args()
    asyncio.run(run_pipeline(cfg))


if __name__ == "__main__":
    main()
