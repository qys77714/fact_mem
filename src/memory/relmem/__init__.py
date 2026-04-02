from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

from memory.mem0 import Mem0MemorySystem
from memory.relmem.decide import decide_relmem
from memory.relmem.prompts import (
    build_relation_classification_user_prompt,
    relation_system_prompt_for_language,
)
from memory.relmem.schemas import RELATION_CLASSIFICATION_RESPONSE_FORMAT
from memory.storage.local_faiss import LocalFaissDatabase
from memory.tracing import MemoryTraceLogger
from benchmark.base import ChatSession
from memory.base import RetrievedMemory, _session_progress_tick

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from openai import OpenAI


class RelMemMemorySystem(Mem0MemorySystem):
    """
    RelMem: mem0-aligned extraction + dense top-K on valid primaries +
    pairwise 5-way relation classification + bucket-local Decide (EQV/OSN/NSO/CON/IND).
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
        relation_concurrency: int = 8,
        relation_max_new_tokens: int = 256,
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
            method="relmem",
            log_dir=trace_log_dir or "logs/memory_trace",
            use_experiment_naming=trace_log_dir is not None,
        )
        self._relation_concurrency = max(1, int(relation_concurrency))
        self._relation_max_new_tokens = max(32, int(relation_max_new_tokens))

    def retrieve(self, history_name: str, query: str, current_time: str, top_k: int = 5) -> List[RetrievedMemory]:
        database = self._get_database(history_name)
        query_embedding = self._embed_texts([query])
        if query_embedding.size == 0:
            return []
        primaries = database.search(
            query_embedding[0],
            top_k,
            only_valid=True,
            only_primary=True,
        )
        for mem in primaries:
            mem.attached_evidence = database.collect_evidence_descendants(mem.memory_id)
        return primaries

    def format_retrieved_for_context(
        self, retrieved: List[RetrievedMemory], language: str = "zh"
    ) -> str:
        """Mem0 风格 + 每条 primary 下附带嵌套 evidence。"""
        from prompts import render_prompt

        if not retrieved:
            template = "agent_context_empty_zh.jinja" if language == "zh" else "agent_context_empty_en.jinja"
            return render_prompt(template)

        unit_template = "relmem_context_unit_zh.jinja" if language == "zh" else "relmem_context_unit_en.jinja"
        context_lines = [
            render_prompt(
                unit_template,
                index=idx + 1,
                text=item.text,
                time=item.time,
                metadata=item.metadata or {},
                attached_evidence=item.attached_evidence or [],
            )
            for idx, item in enumerate(retrieved)
        ]
        return "\n\n".join(context_lines)

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
                "relmem_store_session",
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
                    "relmem_store_chunk",
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

            metadata_base = {
                "method": "relmem",
                "history_name": history_name,
                "session": session_idx,
                "date": session_date,
                "granularity": self.granularity,
            }
            if turn_start is not None:
                metadata_base["turn_start"] = turn_start
                metadata_base["turn_end"] = turn_end

            op_count = 0
            for m_new in facts:
                op_count += self._process_one_new_fact(
                    database,
                    m_new,
                    dict(metadata_base),
                    session_idx,
                    chunk_scope,
                    trace,
                )

            removed = database.deduplicate_identical_text()
            if removed:
                trace.log_memory_operation(
                    operation="DEDUPE_TEXT",
                    memory_id=None,
                    scope_id=chunk_scope,
                    metadata={"removed_count": removed},
                    status="ok",
                )

            trace.close_scope(chunk_scope, status="ok", metadata={"operation_count": op_count})

        trace.close_scope(work_items[-1]["session_scope"], status="ok")
        _session_progress_tick(session_progress, 1)

    def _dense_candidates(self, database: LocalFaissDatabase, m_new: str) -> List[RetrievedMemory]:
        fact_emb = self._embed_texts([m_new])[0]
        return database.search(
            fact_emb,
            self.related_memory_top_k,
            only_valid=True,
            only_primary=True,
        )

    def _classify_relation(
        self,
        m_old_text: str,
        m_new: str,
        trace_scope_id: Optional[str],
        trace: MemoryTraceLogger,
    ) -> str:
        system = relation_system_prompt_for_language(self.language)
        user = build_relation_classification_user_prompt(m_old_text, m_new)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        raw_response = None
        try:
            raw_response = self.llm_client.get_response_chat(
                messages,
                max_new_tokens=self._relation_max_new_tokens,
                temperature=0,
                response_format=RELATION_CLASSIFICATION_RESPONSE_FORMAT,
                verbose=False,
            )
            t = trace
            t.log_llm_interaction(
                purpose="relmem_classify_relation",
                messages=messages,
                response=raw_response,
                scope_id=trace_scope_id,
                metadata={"temperature": 0},
            )
        except Exception as exc:
            t = trace
            t.log_llm_interaction(
                purpose="relmem_classify_relation",
                messages=messages,
                response=None,
                scope_id=trace_scope_id,
                metadata={"temperature": 0},
                error=str(exc),
            )
            logger.warning("RelMem relation classification failed: %s", exc)
            return "IND"

        rel = self._parse_relation_label(raw_response)
        return rel if rel in ("IND", "EQV", "NSO", "OSN", "CON") else "IND"

    def _parse_relation_label(self, raw_response: Any) -> str:
        if isinstance(raw_response, dict):
            r = raw_response.get("relation")
            if isinstance(r, str):
                return r.strip().upper()
        if hasattr(raw_response, "__iter__") and not isinstance(raw_response, str):
            raw_response = raw_response[0]
        if not raw_response:
            return "IND"
        payload = self._safe_json_loads(raw_response) if isinstance(raw_response, str) else None
        if isinstance(payload, dict):
            r = payload.get("relation")
            if isinstance(r, str):
                return r.strip().upper()
        return "IND"

    def _label_candidates(
        self,
        m_new: str,
        candidates: List[RetrievedMemory],
        chunk_scope: str,
        trace: MemoryTraceLogger,
    ) -> List[Tuple[RetrievedMemory, str]]:
        if not candidates:
            return []
        if self._relation_concurrency <= 1 or len(candidates) == 1:
            out: List[Tuple[RetrievedMemory, str]] = []
            for mem in candidates:
                lab = self._classify_relation(mem.text, m_new, chunk_scope, trace)
                out.append((mem, lab))
            return out

        workers = min(self._relation_concurrency, len(candidates))
        out_map: Dict[str, Tuple[RetrievedMemory, str]] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_mem = {
                pool.submit(
                    self._classify_relation,
                    mem.text,
                    m_new,
                    chunk_scope,
                    trace,
                ): mem
                for mem in candidates
            }
            for fut in as_completed(future_to_mem):
                mem = future_to_mem[fut]
                lab = fut.result()
                out_map[mem.memory_id] = (mem, lab)
        return [out_map[m.memory_id] for m in candidates]

    def _process_one_new_fact(
        self,
        database: LocalFaissDatabase,
        m_new: str,
        metadata_base: Dict[str, Any],
        session_idx: int,
        chunk_scope: str,
        trace: MemoryTraceLogger,
    ) -> int:
        m_new = (m_new or "").strip()
        if not m_new:
            return 0

        candidates = self._dense_candidates(database, m_new)
        labeled = self._label_candidates(m_new, candidates, chunk_scope, trace)
        plan = decide_relmem(m_new, labeled)

        for mid in plan.invalidate_ids:
            ok = database.invalidate_memory(mid)
            trace.log_memory_operation(
                operation="INVALIDATE",
                memory_id=mid,
                scope_id=chunk_scope,
                metadata={"m_new": m_new},
                status="ok" if ok else "failed",
            )

        metadata_base["memory_status"] = "valid"

        if plan.outcome == "equivalent_evidence":
            rid = plan.representative_id
            if not rid:
                return 0
            return self._add_evidence_row(
                database,
                m_new,
                dict(metadata_base),
                parent_id=rid,
                session_idx=session_idx,
                chunk_scope=chunk_scope,
                trace=trace,
                operation="ATTACH_EQUIVALENT",
            )

        if plan.outcome == "stronger_primary":
            meta = dict(metadata_base)
            meta["memory_role"] = "primary"
            emb = self._embed_texts([self.build_text_for_embedding(m_new, metadata=meta)])[0]
            new_id = database.add(
                text=m_new,
                source_index=f"session_{session_idx}",
                time=str(metadata_base["date"]),
                metadata=meta,
                embedding=emb,
            )
            trace.log_memory_operation(
                operation="ADD",
                memory_id=new_id,
                scope_id=chunk_scope,
                metadata={"m_new": m_new, "reason": "OSN"},
                after={
                    "text": m_new,
                    "source_index": f"session_{session_idx}",
                    "time": str(metadata_base["date"]),
                    "metadata": dict(meta),
                },
                status="ok",
            )
            for oid in plan.demote_ids:
                ok = database.update_memory(
                    oid,
                    metadata_updates={"memory_role": "evidence", "parent_primary": new_id},
                )
                trace.log_memory_operation(
                    operation="DEMOTE",
                    memory_id=oid,
                    scope_id=chunk_scope,
                    metadata={"m_new": m_new, "new_primary_id": new_id},
                    status="ok" if ok else "failed",
                )
            return 1

        if plan.outcome == "weaker_evidence":
            pid = plan.parent_id
            if not pid:
                return 0
            return self._add_evidence_row(
                database,
                m_new,
                dict(metadata_base),
                parent_id=pid,
                session_idx=session_idx,
                chunk_scope=chunk_scope,
                trace=trace,
                operation="ATTACH_EVIDENCE",
            )

        meta = dict(metadata_base)
        meta["memory_role"] = "primary"
        emb = self._embed_texts([self.build_text_for_embedding(m_new, metadata=meta)])[0]
        memory_id = database.add(
            text=m_new,
            source_index=f"session_{session_idx}",
            time=str(metadata_base["date"]),
            metadata=meta,
            embedding=emb,
        )
        trace.log_memory_operation(
            operation="ADD",
            memory_id=memory_id,
            scope_id=chunk_scope,
            metadata={"m_new": m_new},
            after={
                "text": m_new,
                "source_index": f"session_{session_idx}",
                "time": str(metadata_base["date"]),
                "metadata": dict(meta),
            },
            status="ok",
        )
        return 1

    def _add_evidence_row(
        self,
        database: LocalFaissDatabase,
        text: str,
        metadata_base: Dict[str, Any],
        *,
        parent_id: str,
        session_idx: int,
        chunk_scope: str,
        trace: MemoryTraceLogger,
        operation: str,
    ) -> int:
        meta = dict(metadata_base)
        meta["memory_role"] = "evidence"
        meta["parent_primary"] = parent_id
        emb = self._embed_texts([self.build_text_for_embedding(text, metadata=meta)])[0]
        eid = database.add(
            text=text,
            source_index=f"session_{session_idx}",
            time=str(metadata_base["date"]),
            metadata=meta,
            embedding=emb,
        )
        trace.log_memory_operation(
            operation=operation,
            memory_id=eid,
            scope_id=chunk_scope,
            metadata={"m_new": text, "parent_primary": parent_id},
            after={
                "text": text,
                "source_index": f"session_{session_idx}",
                "time": str(metadata_base["date"]),
                "metadata": dict(meta),
            },
            status="ok",
        )
        return 1
