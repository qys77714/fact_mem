"""LME 候选事实更新方法 3：不做关系判断，所有候选事实直接入库（ADD ALL）。"""

from __future__ import annotations

from typing import Any, Optional

from .memory_system_base import LmeCandidateMemorySystemBase
from memory.storage.local_faiss import LocalFaissDatabase
from memory.tracing import MemoryTraceLogger


class LmeCandidateAddAllMemorySystem(LmeCandidateMemorySystemBase):
    """
    直接将每条候选事实写入库中，不执行关系分类、冲突判断或删除旧记忆。
    灌库路径不调用 dense 检索，故 ``related_memory_top_k`` 对行为无影响（仅满足父类构造）。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        trace_log_dir: Optional[str] = kwargs.get("trace_log_dir")
        super().__init__(*args, **kwargs)
        self.trace = MemoryTraceLogger(
            method="lme_candidate_add_all",
            log_dir=trace_log_dir or "logs/memory_trace",
            use_experiment_naming=trace_log_dir is not None,
        )

    def _process_one_new_fact(
        self,
        database: LocalFaissDatabase,
        m_new: str,
        metadata_base: dict[str, Any],
        session_idx: int,
        chunk_scope: str,
        trace: MemoryTraceLogger,
    ) -> int:
        m_new = (m_new or "").strip()
        if not m_new:
            return 0

        meta = dict(metadata_base)
        meta["memory_role"] = "primary"
        meta["lme_update_method"] = "add_all"

        emb = self._embed_texts([self.build_text_for_embedding(m_new, metadata=meta)])[0]
        memory_id = database.add(
            text=m_new,
            source_index=f"session_{session_idx}",
            time=str(metadata_base.get("date", "")),
            metadata=meta,
            embedding=emb,
        )
        trace.log_memory_operation(
            operation="ADD",
            memory_id=memory_id,
            scope_id=chunk_scope,
            metadata={"m_new": m_new, "strategy": "add_all"},
            after={
                "text": m_new,
                "source_index": f"session_{session_idx}",
                "time": str(metadata_base.get("date", "")),
                "metadata": dict(meta),
            },
            status="ok",
        )
        return 1
