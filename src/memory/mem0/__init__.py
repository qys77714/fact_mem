from __future__ import annotations

import json
import logging
import shutil
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union
import numpy as np

from .prompts import (
    build_fact_retrieval_system_prompt,
    build_update_memory_messages,
)
from .schemas import (
    FACT_RETRIEVAL_RESPONSE_FORMAT,
    UPDATE_MEMORY_RESPONSE_FORMAT,
    UPDATE_MEMORY_RESPONSE_FORMAT_NO_DELETE,
)
from memory.base import BaseMemorySystem, RetrievedMemory, _session_progress_tick
from memory.storage.local_faiss import LocalFaissDatabase
from memory.tracing import MemoryTraceLogger
from benchmark.base import ChatSession

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from openai import OpenAI

class Mem0MemorySystem(BaseMemorySystem):
    def __init__(
        self,
        embed_model_name: str,
        llm_client = None,
        embed_client: Optional["OpenAI"] = None,
        database_root: Optional[str] = None,
        related_memory_top_k: int = 5,
        language: str = "en",
        granularity: Union[str, int] = "all",
        trace_log_dir: Optional[str] = None,
        dialogue_format: str = "user_assistant",
        allow_memory_delete: bool = True,
        manager_max_new_tokens: int = 2048,
        extract_concurrency: int = 8,
    ) -> None:
        super().__init__(
            embed_client=embed_client,
            embed_model_name=embed_model_name,
            llm_client=llm_client,
            database_root=database_root,
        )
        if llm_client is None:
            raise ValueError("llm_client must be provided for Mem0MemorySystem.")
        if embed_client is None:
            raise ValueError("embed_client must be provided for Mem0MemorySystem.")
            
        self.related_memory_top_k = max(1, related_memory_top_k)
        self.language = language
        df = (dialogue_format or "user_assistant").strip().lower()
        if df not in ("user_assistant", "named_speakers"):
            raise ValueError(
                "dialogue_format must be 'user_assistant' or 'named_speakers', "
                f"got {dialogue_format!r}"
            )
        self.dialogue_format = df
        self.granularity = self._parse_granularity(granularity)
        self._allow_memory_delete = allow_memory_delete
        self._manager_max_new_tokens = max(1, int(manager_max_new_tokens))
        self._update_memory_response_format = (
            UPDATE_MEMORY_RESPONSE_FORMAT
            if allow_memory_delete
            else UPDATE_MEMORY_RESPONSE_FORMAT_NO_DELETE
        )
        trace_method = "mem0" if allow_memory_delete else "mem0_nodel"
        self.trace = MemoryTraceLogger(
            method=trace_method,
            log_dir=trace_log_dir or "logs/memory_trace",
            use_experiment_naming=trace_log_dir is not None,
        )
        self._databases = {}
        self._extract_concurrency = max(1, int(extract_concurrency))

    @staticmethod
    def _parse_granularity(granularity: Union[str, int]) -> Union[str, int]:
        if isinstance(granularity, str):
            g = granularity.strip().lower()
            if g == "all":
                return "all"
            if g.isdigit():
                granularity = int(g)
            else:
                raise ValueError("Mem0 granularity must be 'all' or a positive integer.")

        if isinstance(granularity, int) and granularity > 0:
            return granularity
        raise ValueError("Mem0 granularity must be 'all' or a positive integer.")

    def _build_chunk_transcript(self, chunk_turns: List[Any]) -> str:
        lines: List[str] = []
        for turn in chunk_turns:
            content = turn.content.strip()
            if not content:
                continue
            if self.dialogue_format == "named_speakers":
                speaker = (turn.speaker or "Unknown").strip()
                lines.append(f"{speaker}: {content}")
                continue
            role = turn.speaker.lower()
            if role not in ("user", "assistant"):
                role = "user" if role in ("human", "人") else "assistant"
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _get_database(self, history_name: str) -> LocalFaissDatabase:
        namespace = history_name
        if namespace not in self._databases:
            self._databases[namespace] = LocalFaissDatabase(
                namespace=namespace,
                database_root=self.database_root
            )
        return self._databases[namespace]

    def episode_storage_path(self, history_name: str) -> Optional[Path]:
        return self.persisted_data_root() / history_name

    def clear(self, history_name: str) -> None:
        """Remove all memory for this episode (for resume after interrupt)."""
        namespace = history_name
        if namespace in self._databases:
            db = self._databases[namespace]
            db.clear_all()
            del self._databases[namespace]
        # Also remove from disk if not in cache (e.g. from previous run)
        if self.database_root:
            ns_dir = Path(self.database_root) / namespace
            if ns_dir.exists():
                shutil.rmtree(ns_dir)

    def _iter_turn_chunks(self, turns: List[Any]) -> List[Tuple[Optional[int], Optional[int], List[Any]]]:
        if self.granularity == "all":
            return [(None, None, turns)]

        chunk_size = int(self.granularity)
        chunks: List[Tuple[Optional[int], Optional[int], List[Any]]] = []
        for i in range(0, len(turns), chunk_size):
            chunk_turns = turns[i : i + chunk_size]
            if chunk_turns:
                chunks.append((i, i + len(chunk_turns) - 1, chunk_turns))
        return chunks

    def _embed_texts(self, inputs: List[str]) -> np.ndarray:
        from utils.embed_utils import embed_texts
        return embed_texts(self.embed_client, inputs, self.embed_model_name)

    def build_text_for_embedding(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Mem0: 仅对纯 text 做 embedding，不拼接 metadata。"""
        return text

    def store_episode(
        self,
        history_name: str,
        sessions: List[ChatSession],
        *,
        session_progress: Optional[Any] = None,
    ) -> None:
        if not sessions:
            return
        entries = [(idx, sess) for idx, sess in enumerate(sessions, start=1)]
        self._store_planar_entries(history_name, entries, session_progress=session_progress)

    def store_session(self, history_name: str, session_idx: int, session: ChatSession) -> None:
        self._store_planar_entries(history_name, [(session_idx, session)], session_progress=None)

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
                "mem0_store_session",
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
                    "mem0_store_chunk",
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
                "method": "mem0_nodel" if not self._allow_memory_delete else "mem0",
                "history_name": history_name,
                "session": session_idx,
                "date": session_date,
                "granularity": self.granularity,
            }
            if turn_start is not None:
                metadata_base["turn_start"] = turn_start
                metadata_base["turn_end"] = turn_end

            old_memory_json, temp_uuid_mapping = self._collect_related_memories(database, facts, session_date)
            operations = self._decide_memory_operations(
                facts, old_memory_json, trace_scope_id=chunk_scope, trace=trace
            )

            self._apply_memory_changes(
                database,
                operations,
                temp_uuid_mapping,
                metadata_base,
                session_idx,
                trace_scope_id=chunk_scope,
                trace=trace,
            )
            trace.close_scope(chunk_scope, status="ok", metadata={"operation_count": len(operations)})

        trace.close_scope(work_items[-1]["session_scope"], status="ok")
        _session_progress_tick(session_progress, 1)

    def retrieve(self, history_name: str, query: str, current_time: str, top_k: int = 5) -> List[RetrievedMemory]:
        database = self._get_database(history_name)
        query_embedding = self._embed_texts([query])
        if query_embedding.size == 0:
            return []
        return database.search(query_embedding[0], top_k)

    def format_retrieved_for_context(
        self, retrieved: List[RetrievedMemory], language: str = "zh"
    ) -> str:
        """Mem0 自定义组装：text + time + metadata（如 turn_start/turn_end）"""
        from prompts import render_prompt

        if not retrieved:
            template = "agent_context_empty_zh.jinja" if language == "zh" else "agent_context_empty_en.jinja"
            return render_prompt(template)

        unit_template = "mem0_context_unit_zh.jinja" if language == "zh" else "mem0_context_unit_en.jinja"
        context_lines = [
            render_prompt(
                unit_template,
                index=idx + 1,
                text=item.text,
                time=item.time,
                metadata=item.metadata or {},
            )
            for idx, item in enumerate(retrieved)
        ]
        return "\n\n".join(context_lines)

    def _extract_facts(
        self,
        transcript: str,
        user_name: str,
        trace_scope_id: Optional[str] = None,
        trace: Optional[MemoryTraceLogger] = None,
    ) -> List[str]:
        system_prompt = build_fact_retrieval_system_prompt(
            user_name=user_name,
            language=self.language,
            dialogue_format=self.dialogue_format,
        )
        user_prompt = f"Input:\n{transcript}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            raw_response = self.llm_client.get_response_chat(
                messages,
                max_new_tokens=self._manager_max_new_tokens,
                temperature=0,
                response_format=FACT_RETRIEVAL_RESPONSE_FORMAT,
                verbose=False
            )
            t = trace or self.trace
            t.log_llm_interaction(
                purpose="mem0_extract_facts",
                messages=messages,
                response=raw_response,
                scope_id=trace_scope_id,
                metadata={"temperature": 0},
            )
        except Exception as exc:
            t = trace or self.trace
            t.log_llm_interaction(
                purpose="mem0_extract_facts",
                messages=messages,
                response=None,
                scope_id=trace_scope_id,
                metadata={"temperature": 0},
                error=str(exc),
            )
            logger.warning("Mem0 fact extraction failed: %s", exc)
            return []

        if hasattr(raw_response, '__iter__') and not isinstance(raw_response, str):
            # 处理可能的批量返回值
            raw_response = raw_response[0]

        if not raw_response:
            return []
        facts = self._parse_fact_response(raw_response)
        return list(dict.fromkeys(facts))

    def _collect_related_memories(
        self,
        database: LocalFaissDatabase,
        facts: List[str],
        session_date: Any,
    ) -> Tuple[Optional[str], Dict[str, str]]:
        aggregated: "OrderedDict[str, RetrievedMemory]" = OrderedDict()
        
        # 批量 embedding 所有的 fact query（与官方一致：仅用 fact 文本检索）
        fact_texts = list(facts)
        
        if fact_texts:
            fact_embeddings = self._embed_texts(fact_texts)
            for fact_emb in fact_embeddings:
                for memory in database.search(fact_emb, self.related_memory_top_k):
                    memory_id = memory.memory_id
                    if not memory_id or memory_id in aggregated:
                        continue
                    aggregated[memory_id] = memory

        if not aggregated:
            return None, {}
                
        if len(aggregated) > 20:
            # 根据 memory.score 降序排序，取前20个
            sorted_items = sorted(aggregated.items(), key=lambda item: item[1].score, reverse=True)
            selected_items = sorted_items[:20]
            aggregated = OrderedDict(selected_items)

        temp_uuid_mapping = {str(idx): memory_id for idx, memory_id in enumerate(aggregated.keys())}

        serialized_payload = []
        for idx, memory_id in enumerate(aggregated.keys()):
            text_val = aggregated[memory_id].text
            serialized_payload.append({"id": str(idx), "text": text_val})

        return json.dumps(serialized_payload, ensure_ascii=False, indent=2), temp_uuid_mapping

    def _decide_memory_operations(
        self,
        facts: List[str],
        retrieved_old_memory_json: Optional[str],
        trace_scope_id: Optional[str] = None,
        trace: Optional[MemoryTraceLogger] = None,
    ) -> List[Dict[str, Any]]:
        if not facts:
            return []
        response_content = json.dumps(facts, ensure_ascii=False)
        update_prompt = build_update_memory_messages(
            retrieved_old_memory_json,
            response_content,
            language=self.language,
            allow_delete=self._allow_memory_delete,
        )
        messages = [{"role": "user", "content": update_prompt}]
        raw_response = None
        try:
            raw_response = self.llm_client.get_response_chat(
                messages,
                max_new_tokens=self._manager_max_new_tokens,
                temperature=0,
                response_format=self._update_memory_response_format,
                verbose=False
            )
            t = trace or self.trace
            t.log_llm_interaction(
                purpose="mem0_decide_memory_operations",
                messages=messages,
                response=raw_response,
                scope_id=trace_scope_id,
                metadata={"temperature": 0, "fact_count": len(facts)},
            )
        except Exception as exc:
            t = trace or self.trace
            t.log_llm_interaction(
                purpose="mem0_decide_memory_operations",
                messages=messages,
                response=None,
                scope_id=trace_scope_id,
                metadata={"temperature": 0, "fact_count": len(facts)},
                error=str(exc),
            )
            logger.warning("Mem0 decide operations failed: %s", exc)
            return []

        if not raw_response or not str(raw_response).strip():
            return []
        return self._parse_memory_changes(raw_response)

    def _apply_memory_changes(
        self,
        database: LocalFaissDatabase,
        operations: List[Dict[str, Any]],
        temp_uuid_mapping: Dict[str, str],
        metadata_base: Dict[str, Any],
        session_idx: int,
        trace_scope_id: Optional[str] = None,
        trace: Optional[MemoryTraceLogger] = None,
    ) -> None:
        # 分组准备需要 embed 的文本
        texts_to_embed = []
        op_mapping = []

        for operation in operations:
            event = (operation.get("event") or "").upper()
            text = (operation.get("text") or "").strip()
            op_id = operation.get("id")

            if event == "NONE":
                continue

            if event in ("ADD", "UPDATE"):
                texts_to_embed.append(
                    self.build_text_for_embedding(text, metadata=metadata_base)
                )
                old_mem = operation.get("old_memory") if event == "UPDATE" else None
                op_mapping.append((event, op_id, text, old_mem))
            elif event == "DELETE":
                if self._allow_memory_delete:
                    op_mapping.append((event, op_id, None, None))
                else:
                    logger.debug(
                        "Mem0 nodel: ignoring DELETE from model op_id=%s", op_id
                    )

        if texts_to_embed:
            embeddings = self._embed_texts(texts_to_embed)
        else:
            embeddings = []

        emb_idx = 0
        for event, op_id, mem_text, old_memory in op_mapping:
            if event == "ADD":
                memory_id = database.add(
                    text=mem_text,
                    source_index=f"session_{session_idx}",
                    time=str(metadata_base['date']),
                    metadata=dict(metadata_base),
                    embedding=embeddings[emb_idx]
                )
                t = trace or self.trace
                t.log_memory_operation(
                    operation="ADD",
                    memory_id=memory_id,
                    scope_id=trace_scope_id,
                    metadata={"op_id": op_id},
                    after={
                        "text": mem_text,
                        "source_index": f"session_{session_idx}",
                        "time": str(metadata_base['date']),
                        "metadata": dict(metadata_base),
                    },
                    status="ok",
                )
                emb_idx += 1

            elif event == "UPDATE":
                target_id = temp_uuid_mapping.get(str(op_id))
                if target_id and mem_text:
                    success = database.update_memory(
                        memory_id=target_id,
                        new_text=mem_text,
                        new_source_index=f"session_{session_idx}_updated",
                        new_time=str(metadata_base['date']),
                        metadata_updates=dict(metadata_base),
                        new_embedding=embeddings[emb_idx]
                    )
                    if not success:
                        logger.warning("Mem0 update failed: memory_id=%s", target_id)
                    t = trace or self.trace
                    t.log_memory_operation(
                        operation="UPDATE",
                        memory_id=target_id,
                        scope_id=trace_scope_id,
                        metadata={"op_id": op_id},
                        before={"text": old_memory} if old_memory else None,
                        after={
                            "text": mem_text,
                            "source_index": f"session_{session_idx}_updated",
                            "time": str(metadata_base['date']),
                            "metadata_updates": dict(metadata_base),
                        },
                        status="ok" if success else "failed",
                    )
                emb_idx += 1
                
            elif event == "DELETE":
                if not self._allow_memory_delete:
                    continue
                target_id = temp_uuid_mapping.get(str(op_id))
                if target_id:
                    success = database.delete(target_id)
                    if not success:
                        logger.warning("Mem0 delete failed: memory_id=%s", target_id)
                    t = trace or self.trace
                    t.log_memory_operation(
                        operation="DELETE",
                        memory_id=target_id,
                        scope_id=trace_scope_id,
                        metadata={"op_id": op_id},
                        status="ok" if success else "failed",
                    )

        removed = database.deduplicate_identical_text()
        if removed:
            t = trace or self.trace
            t.log_memory_operation(
                operation="DEDUPE_TEXT",
                memory_id=None,
                scope_id=trace_scope_id,
                metadata={"removed_count": removed},
                status="ok",
            )

    def _parse_fact_response(self, raw_response: str) -> List[str]:
        payload = self._safe_json_loads(raw_response)
        if not payload:
            logger.warning("Mem0 fact extraction returned empty or invalid JSON.")
            return []
        facts = payload.get("facts")
        if not isinstance(facts, list):
            return []
        normalized: List[str] = []
        for fact in facts:
            if isinstance(fact, str):
                cleaned = fact.strip()
                if cleaned:
                    normalized.append(cleaned)
        return normalized

    def _parse_memory_changes(self, raw_response: str) -> List[Dict[str, Any]]:
        payload = self._safe_json_loads(raw_response)
        if not payload:
            logger.warning("Mem0 memory update response is empty or invalid.")
            return []
        memory_ops = payload.get("memory")
        if not isinstance(memory_ops, list):
            return []
        normalized_ops: List[Dict[str, Any]] = []
        for op in memory_ops:
            if not isinstance(op, dict):
                continue
            normalized_ops.append(
                {
                    "id": op.get("id"),
                    "text": op.get("text"),
                    "event": (op.get("event") or "").upper(),
                    "old_memory": op.get("old_memory"),
                }
            )
        return normalized_ops

    def _safe_json_loads(self, value: Any) -> Optional[Dict[str, Any]]:
        if isinstance(value, dict):
            return value
        if not isinstance(value, str):
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            logger.warning("Mem0 failed to parse JSON payload: %s", value)
            return None

