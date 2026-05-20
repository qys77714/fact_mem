"""LME 候选灌库共用基类：dense 检索、成对分类、evidence 行写入；不含从对话 session 在线抽取事实的管线。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

from memory.base import RetrievedMemory
from memory.mem0 import Mem0MemorySystem
from memory.storage.local_faiss import LocalFaissDatabase
from memory.tracing import MemoryTraceLogger

if TYPE_CHECKING:
    from openai import OpenAI


class LmeCandidateMemorySystemBase(Mem0MemorySystem):
    """Mem0 存储 + 仅 primary 的 dense 检索与 evidence 子树（供 relation_decision / add_all 灌库）。"""

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
            only_primary=True,
        )
        for mem in primaries:
            mem.attached_evidence = database.collect_evidence_descendants(mem.memory_id)
        return primaries

    def format_retrieved_for_context(
        self, retrieved: List[RetrievedMemory], language: str = "zh", show_time: bool = True
    ) -> str:
        from prompts import render_prompt

        if not retrieved:
            template = "agent_context_empty_zh.jinja" if language == "zh" else "agent_context_empty_en.jinja"
            return render_prompt(template)

        unit_template = "lme_memory_context_unit_zh.jinja" if language == "zh" else "lme_memory_context_unit_en.jinja"
        context_lines = [
            render_prompt(
                unit_template,
                index=idx + 1,
                text=item.text,
                time=item.time,
                metadata=item.metadata or {},
                show_time=show_time,
            )
            for idx, item in enumerate(retrieved)
        ]
        return "\n\n".join(context_lines)

    def _dense_candidates(self, database: LocalFaissDatabase, m_new: str) -> List[RetrievedMemory]:
        fact_emb = self._embed_texts([m_new])[0]
        return database.search(
            fact_emb,
            self.related_memory_top_k,
            only_primary=True,
        )

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
        metadata_extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        meta = dict(metadata_base)
        meta["memory_role"] = "evidence"
        meta["parent_primary"] = parent_id
        if metadata_extra:
            meta.update(metadata_extra)
        emb = self._embed_texts([self.build_text_for_embedding(text, metadata=meta)])[0]
        date_s = str(metadata_base.get("date", ""))
        eid = database.add(
            text=text,
            source_index=f"session_{session_idx}",
            time=date_s,
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
                "time": date_s,
                "metadata": dict(meta),
            },
            status="ok",
        )
        return eid
