#!/usr/bin/env python3
"""
MEME oracle baseline: use dataset gold_facts as the sole memory bank, answer
after_questions, and judge with an LLM.

Requires vLLM chat (e.g. vllm_model_runner_4090/script/0_run_model.sh) and
embedding (0_run_embedding.sh) to be running; ports from fact_memory/.env.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import faiss
from openai import OpenAI
from tqdm import tqdm

_MEME_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MEME_DIR.parent.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from memory.base import RetrievedMemory  # noqa: E402
from pipeline_evaluate import evaluate  # noqa: E402
from prompts import render_prompt  # noqa: E402
from utils.embed_utils import embed_texts  # noqa: E402
from utils.env import load_env  # noqa: E402
from utils.eval_report import append_eval_json, utc_timestamp_iso  # noqa: E402
from utils.llm_api import load_api_chat_completion  # noqa: E402


DEFAULT_DATASET = _PROJECT_ROOT / "data/raw_data/MEME/meme_nofiller.json"
DEFAULT_OUTPUT_DIR = _MEME_DIR / "output"


@dataclass(frozen=True)
class MemeQuestion:
    episode_id: str
    domain: str
    phase: str  # "after" | "before"
    task_type: str
    question: str
    reference: str
    question_time: str
    position_after_session: int
    hop: Optional[int] = None
    entities: Optional[List[str]] = None


@dataclass(frozen=True)
class GoldFactRecord:
    fact_id: int
    entity: str
    value: str
    fact_text: str
    session_index: int
    session_id: str
    timestamp: str
    evidence_type: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MEME gold_facts memory oracle eval")
    p.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="MEME JSON path (default: meme_nofiller.json)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for pred.jsonl and eval_judge.json",
    )
    p.add_argument("--answer-model", default="gemma4-26B")
    p.add_argument("--judge-model", default="gemma4-26B")
    p.add_argument("--embedding-model", default="qwen3-embedding-8b")
    p.add_argument(
        "--embedding-base-url",
        default=None,
        help="Default: EMBEDDING_BASE_URL from .env",
    )
    p.add_argument(
        "--embedding-api-key",
        default=None,
        help="Default: EMBEDDING_API_KEY from .env",
    )
    p.add_argument(
        "--phases",
        default="after",
        help="Comma-separated: after, before, or both",
    )
    p.add_argument(
        "--retrieve-topk",
        type=int,
        default=20,
        help="Dense retrieval top-k from gold_facts bank (ignored if --use-all-facts)",
    )
    p.add_argument(
        "--use-all-facts",
        action="store_true",
        help="Put all gold_facts up to the question cutoff into context (no retrieval)",
    )
    p.add_argument(
        "--memory-token-limit",
        type=int,
        default=8192,
        help="Reserved for future context trimming",
    )
    p.add_argument("--answer-concurrency", type=int, default=8)
    p.add_argument("--judge-concurrency", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--judge-max-new-tokens", type=int, default=512)
    p.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Debug: limit episodes (0 = all)",
    )
    p.add_argument(
        "--task-types",
        default=None,
        help="Comma-separated MEME task types, e.g. ER,Agg,Tr,Del,Cas,Abs",
    )
    p.add_argument("--use-cot", action="store_true", help="Judge with brief CoT")
    p.add_argument(
        "--db-root",
        type=Path,
        default=None,
        help="FAISS cache root (default: <output-dir>/faiss_cache)",
    )
    p.add_argument(
        "--skip-judge",
        action="store_true",
        help="Only write pred.jsonl",
    )
    p.add_argument(
        "--html",
        action="store_true",
        help="After eval, write qa_viewer.html in the output directory",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip (episode_id, phase, question) rows already in pred.jsonl",
    )
    p.add_argument(
        "--embed-batch-size",
        type=int,
        default=64,
        help="Batch size for question embedding in retrieval mode",
    )
    p.add_argument(
        "--skip-api-check",
        action="store_true",
        help="Skip startup health check for embedding/chat APIs",
    )
    return p.parse_args()


def load_episodes(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of episodes in {path}")
    return data


def extract_gold_facts(
    episode: Dict[str, Any], max_session_index: Optional[int] = None
) -> List[GoldFactRecord]:
    out: List[GoldFactRecord] = []
    for sess_idx, sess in enumerate(episode.get("sessions") or []):
        if max_session_index is not None and sess_idx > max_session_index:
            continue
        if sess.get("type") != "evidence":
            continue
        for gf in sess.get("gold_facts") or []:
            out.append(
                GoldFactRecord(
                    fact_id=int(gf["fact_id"]),
                    entity=str(gf["entity"]),
                    value=str(gf["value"]),
                    fact_text=str(gf.get("fact_text") or gf.get("original_seed", "")),
                    session_index=sess_idx,
                    session_id=str(sess.get("session_id", "")),
                    timestamp=str(sess.get("timestamp", "")),
                    evidence_type=str(sess.get("evidence_type", "")),
                )
            )
    return out


def iter_questions(
    episode: Dict[str, Any], phases: List[str]
) -> List[MemeQuestion]:
    rows: List[MemeQuestion] = []
    eid = episode["episode_id"]
    domain = episode.get("domain", "")
    for phase in phases:
        block = episode.get(f"{phase}_questions")
        if not block or not isinstance(block, dict):
            continue
        pos = int(block.get("position_after_session", -1))
        q_time = str(block.get("timestamp", ""))
        for q in block.get("questions") or []:
            ref = q.get("gold_answer")
            if ref is None:
                ref = q.get("expected_answer", "")
            rows.append(
                MemeQuestion(
                    episode_id=eid,
                    domain=domain,
                    phase=phase,
                    task_type=str(q.get("task_type", "")),
                    question=str(q.get("question", "")),
                    reference=str(ref),
                    question_time=q_time,
                    position_after_session=pos,
                    hop=q.get("hop"),
                    entities=list(q.get("entity") or []),
                )
            )
    return rows


def _question_key(q: MemeQuestion) -> Tuple[str, str, str]:
    return (q.episode_id, q.phase, q.question)


@dataclass
class _EpisodeVectorIndex:
    facts: List[GoldFactRecord]
    index: faiss.IndexFlatIP


def _probe_services(
    answer_model: str,
    embed_client: Optional[OpenAI],
    embedding_model: str,
) -> None:
    sync = load_api_chat_completion(answer_model, async_=False)
    sync.get_response_chat(
        [{"role": "user", "content": "Reply with OK."}],
        max_new_tokens=8,
        temperature=0.0,
    )
    print(f"[ok] chat: {sync.model_name}")
    if embed_client is not None:
        embed_client.embeddings.create(input=["health check"], model=embedding_model)
        print(f"[ok] embedding: {embedding_model}")


class GoldFactsMemoryBank:
    """gold_facts memory bank: in-memory list (--use-all-facts) or in-RAM FAISS retrieval."""

    def __init__(
        self,
        embed_client: Optional[OpenAI],
        embed_model: str,
        db_root: Path,
        use_all_facts: bool,
    ) -> None:
        self._embed_client = embed_client
        self._embed_model = embed_model
        self._db_root = db_root  # kept for API compat; retrieval no longer writes disk
        self._use_all_facts = use_all_facts
        self._episode_facts: Dict[str, List[GoldFactRecord]] = {}
        self._episode_indexes: Dict[str, _EpisodeVectorIndex] = {}
        self._built: set[str] = set()

    def build_episode(self, episode_id: str, facts: List[GoldFactRecord]) -> None:
        if episode_id in self._built:
            return
        self._episode_facts[episode_id] = list(facts)
        if self._use_all_facts:
            self._built.add(episode_id)
            return
        if self._embed_client is None:
            raise ValueError("embedding client required when not using --use-all-facts")
        if not facts:
            self._built.add(episode_id)
            return
        texts = [f.fact_text for f in facts]
        embs = embed_texts(self._embed_client, texts, self._embed_model)
        embs = np.ascontiguousarray(embs.astype(np.float32))
        faiss.normalize_L2(embs)
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        self._episode_indexes[episode_id] = _EpisodeVectorIndex(facts=list(facts), index=index)
        self._built.add(episode_id)

    def _facts_to_retrieved(
        self, facts: List[GoldFactRecord], cutoff_session_index: int
    ) -> List[RetrievedMemory]:
        eligible = [f for f in facts if f.session_index <= cutoff_session_index]
        eligible.sort(key=lambda f: (f.session_index, f.fact_id))
        return [
            RetrievedMemory(
                memory_id=f"{f.session_id}#fact{f.fact_id}",
                text=f.fact_text,
                source_index=f"{f.session_id}#fact{f.fact_id}",
                time=f.timestamp,
                score=1.0,
                metadata={
                    "entity": f.entity,
                    "value": f.value,
                    "session_index": f.session_index,
                    "evidence_type": f.evidence_type,
                },
            )
            for f in eligible
        ]

    def retrieve(
        self,
        episode_id: str,
        query: str,
        cutoff_session_index: int,
        top_k: int,
        use_all_facts: bool,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[RetrievedMemory]:
        facts = self._episode_facts.get(episode_id, [])
        if use_all_facts:
            return self._facts_to_retrieved(facts, cutoff_session_index)

        if not facts:
            return []

        if query_embedding is None:
            if not query.strip():
                return []
            if self._embed_client is None:
                raise ValueError("embedding client required for retrieval mode")
            query_embedding = embed_texts(self._embed_client, [query], self._embed_model)[0]

        ep_index = self._episode_indexes.get(episode_id)
        if ep_index is None or ep_index.index.ntotal == 0:
            return []

        q = np.ascontiguousarray(query_embedding.astype(np.float32).reshape(1, -1))
        faiss.normalize_L2(q)
        k = min(max(top_k * 3, top_k), ep_index.index.ntotal)
        scores, indices = ep_index.index.search(q, k)

        out: List[RetrievedMemory] = []
        for score, row in zip(scores[0], indices[0]):
            if row < 0:
                continue
            fact = ep_index.facts[int(row)]
            if fact.session_index > cutoff_session_index:
                continue
            out.append(
                RetrievedMemory(
                    memory_id=f"{fact.session_id}#fact{fact.fact_id}",
                    text=fact.fact_text,
                    source_index=f"{fact.session_id}#fact{fact.fact_id}",
                    time=fact.timestamp,
                    score=float(score),
                    metadata={
                        "entity": fact.entity,
                        "value": fact.value,
                        "session_index": fact.session_index,
                        "evidence_type": fact.evidence_type,
                    },
                )
            )
            if len(out) >= top_k:
                break
        return out


def format_context(retrieved: List[RetrievedMemory]) -> str:
    if not retrieved:
        return render_prompt("agent_context_empty_en.jinja")
    lines = [
        render_prompt(
            "agent_context_unit_en.jinja",
            index=i + 1,
            text=item.text,
            time=item.time,
            metadata=item.metadata or {},
        )
        for i, item in enumerate(retrieved)
    ]
    return "\n\n".join(lines)


def build_answer_prompt(question: MemeQuestion, context_block: str) -> str:
    return render_prompt(
        "agent_prompt_en_open.jinja",
        context_block=context_block,
        question_time=question.question_time,
        question=question.question,
    )


def serialize_retrieved(retrieved: List[RetrievedMemory]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in retrieved:
        meta = dict(r.metadata or {})
        rows.append(
            {
                "text": r.text,
                "time": r.time,
                "score": r.score,
                "source_index": r.source_index,
                "entity": meta.get("entity"),
                "value": meta.get("value"),
                "session_index": meta.get("session_index"),
                "evidence_type": meta.get("evidence_type"),
            }
        )
    return rows


def load_done_keys(pred_path: Path) -> set[Tuple[str, str, str]]:
    if not pred_path.is_file():
        return set()
    done: set[Tuple[str, str, str]] = set()
    with pred_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.add(
                (
                    str(row.get("episode_id", "")),
                    str(row.get("phase", "")),
                    str(row.get("question", "")),
                )
            )
    return done


def _load_pred_rows(pred_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not pred_path.is_file():
        return rows
    with pred_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _batch_embed_queries(
    embed_client: OpenAI,
    model: str,
    queries: List[str],
    batch_size: int = 64,
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for i in range(0, len(queries), batch_size):
        chunk = queries[i : i + batch_size]
        embs = embed_texts(embed_client, chunk, model)
        for row in embs:
            out.append(row)
    return out


async def run_answer_phase(
    questions: List[MemeQuestion],
    episodes_by_id: Dict[str, Dict[str, Any]],
    memory_bank: GoldFactsMemoryBank,
    answer_client,
    *,
    retrieve_topk: int,
    use_all_facts: bool,
    concurrency: int,
    max_new_tokens: int,
    pred_path: Path,
    resume: bool,
    embed_client: Optional[OpenAI] = None,
    embed_model: str = "",
    embed_batch_size: int = 64,
) -> List[Dict[str, Any]]:
    done = load_done_keys(pred_path) if resume else set()
    pred_path.parent.mkdir(parents=True, exist_ok=True)

    episode_ids = list(episodes_by_id.keys())
    if not use_all_facts:
        print(f"Embedding gold_facts & building in-memory index for {len(episode_ids)} episodes …")
    for eid in tqdm(episode_ids, desc="build memory", unit="ep"):
        facts = extract_gold_facts(episodes_by_id[eid], max_session_index=None)
        memory_bank.build_episode(eid, facts)

    pending = [q for q in questions if _question_key(q) not in done]
    if not pending:
        return _load_pred_rows(pred_path)

    messages_list: List[List[dict]] = []
    meta: List[Tuple[MemeQuestion, List[RetrievedMemory]]] = []

    query_embs: Optional[List[np.ndarray]] = None
    if not use_all_facts and embed_client is not None:
        print(f"Embedding {len(pending)} questions for retrieval …")
        query_embs = _batch_embed_queries(
            embed_client,
            embed_model,
            [q.question for q in pending],
            batch_size=embed_batch_size,
        )

    for i, q in enumerate(tqdm(pending, desc="retrieve+prompt", unit="q")):
        q_emb = query_embs[i] if query_embs is not None else None
        retrieved = memory_bank.retrieve(
            q.episode_id,
            q.question,
            q.position_after_session,
            retrieve_topk,
            use_all_facts,
            query_embedding=q_emb,
        )
        ctx = format_context(retrieved)
        prompt = build_answer_prompt(q, ctx)
        messages_list.append([{"role": "user", "content": prompt}])
        meta.append((q, retrieved))

    print(
        f"Calling answer model ({answer_client.model_name}) for {len(messages_list)} questions "
        f"(concurrency={concurrency}) …"
    )
    responses = await answer_client.get_response_chat(
        messages_list,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        max_concurrency=max(1, concurrency),
        use_tqdm=True,
        verbose=False,
    )

    with pred_path.open("a", encoding="utf-8") as out_f:
        for (q, retrieved), model_answer in zip(meta, responses):
            row = {
                "episode_id": q.episode_id,
                "domain": q.domain,
                "phase": q.phase,
                "task_type": q.task_type,
                "question_type": q.task_type,
                "question": q.question,
                "answer": q.reference,
                "model_answer": model_answer if model_answer is not None else "",
                "question_time": q.question_time,
                "position_after_session": q.position_after_session,
                "hop": q.hop,
                "entities": q.entities,
                "retrieved_count": len(retrieved),
                "retrieved_memories": serialize_retrieved(retrieved),
                "use_all_facts": use_all_facts,
                "memory_source": "gold_facts",
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return _load_pred_rows(pred_path)


async def async_main() -> None:
    args = parse_args()
    load_env(str(_PROJECT_ROOT / ".env"))

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    task_filter: Optional[set[str]] = None
    if args.task_types:
        task_filter = {t.strip() for t in args.task_types.split(",") if t.strip()}

    episodes = load_episodes(args.dataset)
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]

    episodes_by_id = {ep["episode_id"]: ep for ep in episodes}
    all_questions: List[MemeQuestion] = []
    for ep in episodes:
        for q in iter_questions(ep, phases):
            if task_filter and q.task_type not in task_filter:
                continue
            all_questions.append(q)

    tag = args.dataset.stem
    if args.use_all_facts:
        tag += "_allfacts"
    else:
        tag += f"_topk{args.retrieve_topk}"
    out_dir = args.output_dir / f"{tag}_{args.answer_model}_{args.judge_model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "pred.jsonl"
    if not args.resume and pred_path.exists():
        pred_path.unlink()

    embed_client: Optional[OpenAI] = None
    if not args.use_all_facts:
        embed_base = args.embedding_base_url or os.getenv(
            "EMBEDDING_BASE_URL", "http://localhost:7110/v1/"
        )
        embed_key = args.embedding_api_key or os.getenv("EMBEDDING_API_KEY", "zjj")
        embed_client = OpenAI(api_key=embed_key, base_url=embed_base)
    db_root = args.db_root or (out_dir / "faiss_cache")
    memory_bank = GoldFactsMemoryBank(
        embed_client,
        args.embedding_model,
        Path(db_root),
        use_all_facts=args.use_all_facts,
    )

    answer_client = load_api_chat_completion(args.answer_model, async_=True)

    if not args.skip_api_check:
        _probe_services(args.answer_model, embed_client, args.embedding_model)

    print(
        f"Dataset={args.dataset} episodes={len(episodes)} "
        f"questions={len(all_questions)} use_all_facts={args.use_all_facts}"
    )
    print(f"Output: {out_dir}")

    pred_rows = await run_answer_phase(
        all_questions,
        episodes_by_id,
        memory_bank,
        answer_client,
        retrieve_topk=args.retrieve_topk,
        use_all_facts=args.use_all_facts,
        concurrency=args.answer_concurrency,
        max_new_tokens=args.max_new_tokens,
        pred_path=pred_path,
        resume=args.resume,
        embed_client=embed_client,
        embed_model=args.embedding_model,
        embed_batch_size=args.embed_batch_size,
    )

    if args.skip_judge:
        print(f"Wrote {len(pred_rows)} predictions to {pred_path}")
        if args.html:
            from build_meme_gold_facts_html import build_html_report

            html_path = out_dir / "qa_viewer.html"
            build_html_report(
                pred_path=pred_path,
                dataset_path=args.dataset,
                eval_path=None,
                output_path=html_path,
                retrieve_topk=args.retrieve_topk,
                use_all_facts=args.use_all_facts,
                db_root=db_root,
                embedding_model=args.embedding_model,
            )
            print(f"HTML viewer: {html_path}")
        return

    judge_samples = [
        {
            "question": r["question"],
            "answer": r["answer"],
            "model_answer": r.get("model_answer", ""),
            "question_time": r.get("question_time", ""),
            "question_type": r.get("task_type", ""),
        }
        for r in pred_rows
    ]

    metrics, outcomes = await evaluate(
        judge_samples,
        judge_model=args.judge_model,
        use_cot=args.use_cot,
        max_concurrency=args.judge_concurrency,
        max_new_tokens=args.judge_max_new_tokens,
        judge_qwen_thinking=False,
        print_one_sample=False,
        judge_oqa_template="pipeline_eval_oqa.jinja",
        judge_mcq_template="pipeline_eval_mcq.jinja",
        judge_system_template="pipeline_eval_system.jinja",
    )

    for row, outcome in zip(pred_rows, outcomes):
        row["is_correct"] = outcome.get("is_correct")
        row["judge_api_failed"] = outcome.get("api_failed", False)
    with pred_path.open("w", encoding="utf-8") as f:
        for row in pred_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    eval_record = {
        "timestamp": utc_timestamp_iso(),
        "eval_type": "meme_gold_facts_oracle",
        "dataset": str(args.dataset),
        "pred_path": str(pred_path),
        "answer_model": args.answer_model,
        "judge_model": args.judge_model,
        "embedding_model": args.embedding_model,
        "phases": phases,
        "use_all_facts": args.use_all_facts,
        "retrieve_topk": args.retrieve_topk,
        "n_episodes": len(episodes),
        "n_questions": len(pred_rows),
        **metrics,
    }
    eval_path = out_dir / "eval_judge.json"
    append_eval_json(str(eval_path), eval_record)

    print(json.dumps(eval_record, indent=2, ensure_ascii=False))
    print(f"\nResults: {eval_path}")

    if args.html:
        from build_meme_gold_facts_html import build_html_report

        html_path = out_dir / "qa_viewer.html"
        build_html_report(
            pred_path=pred_path,
            dataset_path=args.dataset,
            eval_path=eval_path,
            output_path=html_path,
            retrieve_topk=args.retrieve_topk,
            use_all_facts=args.use_all_facts,
            db_root=db_root,
            embedding_model=args.embedding_model,
        )
        print(f"HTML viewer: {html_path}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
