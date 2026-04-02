"""
Append-Only / Recency-Only baselines for fact-memory experiments.

- Append-Only: same fact extraction as mem0 (`_extract_facts`), then only ADD each fact
  (no manager / merge / dedupe). Retrieval inherits mem0 (dense search + context template).
- Recency-Only: same storage as full_context (whole session per row); retrieve all sessions
  newest-first for agent prefix trim under memory_token_limit.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

from benchmark.base import ChatSession
from memory.baselines.full_context import FullContextMemorySystem
from memory.base import RetrievedMemory, _session_progress_tick
from memory.mem0 import Mem0MemorySystem
from memory.tracing import MemoryTraceLogger

if TYPE_CHECKING:
    from openai import OpenAI


class AppendOnlyMemorySystem(Mem0MemorySystem):
    """
    Append-only baseline: mem0-aligned chunking and `_extract_facts`, then embed + add
    each fact with no UPDATE/DELETE/manager merge or post-hoc dedupe.
    """

    def __init__(
        self,
        embed_model_name: str,
        llm_client=None,
        embed_client: Optional["OpenAI"] = None,
        database_root: Optional[str] = None,
        related_memory_top_k: int = 5,
        language: str = "en",
        granularity: Union[str, int] = "all",
        trace_log_dir: Optional[str] = None,
        dialogue_format: str = "user_assistant",
        manager_max_new_tokens: int = 2048,
        extract_concurrency: int = 8,
    ) -> None:
        super().__init__(
            embed_model_name=embed_model_name,
            llm_client=llm_client,
            embed_client=embed_client,
            database_root=database_root,
            related_memory_top_k=related_memory_top_k,
            language=language,
            granularity=granularity,
            trace_log_dir=trace_log_dir,
            dialogue_format=dialogue_format,
            allow_memory_delete=True,
            manager_max_new_tokens=manager_max_new_tokens,
            extract_concurrency=extract_concurrency,
        )
        self.trace = MemoryTraceLogger(
            method="append_only",
            log_dir=trace_log_dir or "logs/memory_trace",
            use_experiment_naming=trace_log_dir is not None,
        )

    def _store_planar_entries(
        self,
        history_name: str,
        session_entries: List[Tuple[int, ChatSession]],
        *,
        session_progress: Optional[Any] = None,
    ) -> None:
        trace = self.trace.get_logger_for(history_name)
        database = self._get_database(history_name)

        work_items: List[Dict[str, Any]] = []
        for session_idx, session in session_entries:
            session_date = session.session_date
            session_scope = trace.create_scope(
                "append_only_store_session",
                metadata={
                    "history_name": history_name,
                    "session_idx": session_idx,
                    "session_date": str(session_date),
                    "granularity": self.granularity,
                },
            )
            chunks = self._iter_turn_chunks(session.turns)
            for turn_start, turn_end, chunk_turns in chunks:
                chunk_scope = trace.create_scope(
                    "append_only_store_chunk",
                    parent_scope_id=session_scope,
                    metadata={
                        "turn_start": turn_start,
                        "turn_end": turn_end,
                        "turn_count": len(chunk_turns),
                    },
                )
                transcript = self._build_chunk_transcript(chunk_turns)
                work_items.append(
                    {
                        "session_idx": session_idx,
                        "session_date": session_date,
                        "turn_start": turn_start,
                        "turn_end": turn_end,
                        "session_scope": session_scope,
                        "chunk_scope": chunk_scope,
                        "transcript": transcript,
                    }
                )

        if not work_items:
            return

        n = len(work_items)
        if self._extract_concurrency <= 1 or n <= 1:
            facts_per_chunk = [
                self._extract_facts(
                    it["transcript"],
                    user_name="user",
                    trace_scope_id=it["chunk_scope"],
                    trace=trace,
                )
                for it in work_items
            ]
        else:
            workers = min(self._extract_concurrency, n)
            pending: List[Optional[List[str]]] = [None] * n
            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_to_i = {
                    pool.submit(
                        self._extract_facts,
                        work_items[idx]["transcript"],
                        "user",
                        work_items[idx]["chunk_scope"],
                        trace,
                    ): idx
                    for idx in range(n)
                }
                for fut in as_completed(future_to_i):
                    idx = future_to_i[fut]
                    pending[idx] = fut.result()
            facts_per_chunk = [p if p is not None else [] for p in pending]

        for i, item in enumerate(work_items):
            if i > 0 and item["session_scope"] != work_items[i - 1]["session_scope"]:
                trace.close_scope(work_items[i - 1]["session_scope"], status="ok")
                _session_progress_tick(session_progress, 1)

            facts = facts_per_chunk[i]
            session_idx = item["session_idx"]
            session_date = item["session_date"]
            chunk_scope = item["chunk_scope"]
            turn_start = item["turn_start"]
            turn_end = item["turn_end"]

            if not facts:
                trace.close_scope(chunk_scope, status="ok", metadata={"skipped": "no_facts"})
                continue

            metadata_base: Dict[str, Any] = {
                "method": "append_only",
                "history_name": history_name,
                "session": session_idx,
                "date": session_date,
                "granularity": self.granularity,
            }
            if turn_start is not None:
                metadata_base["turn_start"] = turn_start
                metadata_base["turn_end"] = turn_end

            clean_facts = [f.strip() for f in facts if f.strip()]
            if not clean_facts:
                trace.close_scope(chunk_scope, status="ok", metadata={"skipped": "no_facts"})
                continue

            texts_to_embed = [
                self.build_text_for_embedding(ft, metadata=metadata_base) for ft in clean_facts
            ]
            embeddings = self._embed_texts(texts_to_embed)

            for fi, mem_text in enumerate(clean_facts):
                memory_id = database.add(
                    text=mem_text,
                    source_index=f"session_{session_idx}",
                    time=str(metadata_base["date"]),
                    metadata=dict(metadata_base),
                    embedding=embeddings[fi],
                )
                trace.log_memory_operation(
                    operation="ADD",
                    memory_id=memory_id,
                    scope_id=chunk_scope,
                    metadata={"fact_index": fi},
                    after={
                        "text": mem_text,
                        "source_index": f"session_{session_idx}",
                        "time": str(metadata_base["date"]),
                        "metadata": dict(metadata_base),
                    },
                    status="ok",
                )

            trace.close_scope(
                chunk_scope, status="ok", metadata={"fact_count": len(clean_facts)}
            )

        trace.close_scope(work_items[-1]["session_scope"], status="ok")
        _session_progress_tick(session_progress, 1)


class RecencyOnlyMemorySystem(FullContextMemorySystem):
    """
    Recency-only baseline: same writes as full_context; retrieve all session rows
    newest-first (query/top_k ignored) so agent prefix trim keeps the latest dialogue.
    """

    def retrieve(
        self,
        history_name: str,
        query: str,
        current_time: str,
        top_k: int = 5,
    ) -> List[RetrievedMemory]:
        database = self._get_database(history_name)
        return database.list_all_memories(sort_by_time=True, descending=True)
