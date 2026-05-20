"""LME 抽取候选 + 五类关系分类；写库：桶内聚合后的有向弱边（EQUIV / ATTACH / UPDATE），不物理删除。"""

from __future__ import annotations

import logging
from typing import Any, Optional

from memory.storage.local_faiss import LocalFaissDatabase
from memory.tracing import MemoryTraceLogger

from .memory_system_base import LmeCandidateMemorySystemBase
from .schemas import LME_RELATION_CLASSIFICATION_RESPONSE_FORMAT
from .relation_decision import LmeRelationDecision, decide_lme_update_relation_decision
from .prompts import (
    build_lme_relation_classification_user_prompt,
    lme_relation_system_prompt_for_language,
)

logger = logging.getLogger(__name__)

_VALID_RELATIONS = frozenset({"IND", "EQV", "NSO", "OSN", "CON"})


class LmeCandidateRelationDecisionMemorySystem(LmeCandidateMemorySystemBase):
    """
    - **成对分类**：dense Top-K 后，对与 ``m_new`` 相关的每条**已在库中**的 primary 各调一次 LLM。
      结构化输出仅五类标签 ``IND``/``EQV``/``NSO``/``OSN``/``CON``；**EQUIV** 等名称表示写库时的 **操作语义**
      （弱侧指向强侧），由 ``EQV`` 等标签映射到 ``lme_edge``，不单独出现在 LLM JSON 里。
    - **写库**：与 ``decide_lme_update_relation_decision`` 对齐 — 弱记忆为 ``memory_role=evidence``，
      ``lme_edge`` 区分 EQUIV / ATTACH / UPDATE；ingest dense 使用 ``only_primary=True`` 不召回弱行。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._relation_system_en_template = (kwargs.pop("relation_system_en_template", None) or "").strip() or None
        self._relation_system_zh_template = (kwargs.pop("relation_system_zh_template", None) or "").strip() or None
        self._relation_user_template = (kwargs.pop("relation_user_template", None) or "").strip() or None
        trace_log_dir: Optional[str] = kwargs.get("trace_log_dir")
        super().__init__(*args, **kwargs)
        self.trace = MemoryTraceLogger(
            method="lme_candidate_relation_decision",
            log_dir=trace_log_dir or "logs/memory_trace",
            use_experiment_naming=trace_log_dir is not None,
        )

    def _classify_relation(
        self,
        m_old_text: str,
        m_new: str,
        trace_scope_id: Optional[str],
        trace: MemoryTraceLogger,
    ) -> str:
        system = lme_relation_system_prompt_for_language(
            self.language,
            template_en=self._relation_system_en_template,
            template_zh=self._relation_system_zh_template,
        )
        user = build_lme_relation_classification_user_prompt(
            m_old_text, m_new, template=self._relation_user_template
        )
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
                response_format=LME_RELATION_CLASSIFICATION_RESPONSE_FORMAT,
                verbose=False,
            )
            trace.log_llm_interaction(
                purpose="lme_candidate_relation_decision_classify_relation",
                messages=messages,
                response=raw_response,
                scope_id=trace_scope_id,
                metadata={"temperature": 0},
            )
        except Exception as exc:
            trace.log_llm_interaction(
                purpose="lme_candidate_relation_decision_classify_relation",
                messages=messages,
                response=None,
                scope_id=trace_scope_id,
                metadata={"temperature": 0},
                error=str(exc),
            )
            logger.warning("LME relation classification failed: %s", exc)
            return "IND"

        rel = self._parse_relation_label(raw_response)
        return rel if rel in _VALID_RELATIONS else "IND"

    def _apply_con_weak_updates(
        self,
        database: LocalFaissDatabase,
        new_row_id: str,
        con_update_ids: tuple[str, ...],
        m_new: str,
        chunk_scope: str,
        trace: MemoryTraceLogger,
    ) -> None:
        """CON：弱侧旧记忆保留，打 UPDATE 边指向新行（primary 或 weak）。"""
        for mid in con_update_ids:
            ok = database.update_memory(
                mid,
                metadata_updates={
                    "memory_role": "evidence",
                    "parent_primary": new_row_id,
                    "lme_edge": "UPDATE",
                },
            )
            trace.log_memory_operation(
                operation="UPDATE_REL",
                memory_id=mid,
                scope_id=chunk_scope,
                metadata={"m_new": m_new, "update_target": new_row_id},
                status="ok" if ok else "failed",
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

        candidates = self._dense_candidates(database, m_new)
        labeled = self._label_candidates(m_new, candidates, chunk_scope, trace)
        plan = decide_lme_update_relation_decision(m_new, labeled)

        mb = dict(metadata_base)
        mb["lme_update_method"] = "relation_decision"

        return self._execute_lme_plan(
            database,
            m_new,
            mb,
            session_idx,
            chunk_scope,
            trace,
            plan,
        )

    def _execute_lme_plan(
        self,
        database: LocalFaissDatabase,
        m_new: str,
        metadata_base: dict[str, Any],
        session_idx: int,
        chunk_scope: str,
        trace: MemoryTraceLogger,
        plan: LmeRelationDecision,
    ) -> int:
        con_ids = plan.con_update_ids
        date_s = str(metadata_base.get("date", ""))

        if plan.outcome == "equivalent_evidence":
            rid = plan.representative_id
            if not rid:
                return 0
            weak_id = self._add_evidence_row(
                database,
                m_new,
                dict(metadata_base),
                parent_id=rid,
                session_idx=session_idx,
                chunk_scope=chunk_scope,
                trace=trace,
                operation="ATTACH_EQUIVALENT",
                metadata_extra={"lme_edge": "EQUIV"},
            )
            self._apply_con_weak_updates(database, weak_id, con_ids, m_new, chunk_scope, trace)
            return 1

        if plan.outcome == "stronger_primary":
            meta = dict(metadata_base)
            meta["memory_role"] = "primary"
            emb = self._embed_texts([self.build_text_for_embedding(m_new, metadata=meta)])[0]
            new_id = database.add(
                text=m_new,
                source_index=f"session_{session_idx}",
                time=date_s,
                metadata=meta,
                embedding=emb,
            )
            trace.log_memory_operation(
                operation="ADD",
                memory_id=new_id,
                scope_id=chunk_scope,
                metadata={"m_new": m_new, "reason": plan.reason},
                after={
                    "text": m_new,
                    "source_index": f"session_{session_idx}",
                    "time": date_s,
                    "metadata": dict(meta),
                },
                status="ok",
            )
            self._apply_con_weak_updates(database, new_id, con_ids, m_new, chunk_scope, trace)
            for oid in plan.demote_ids:
                ok = database.update_memory(
                    oid,
                    metadata_updates={
                        "memory_role": "evidence",
                        "parent_primary": new_id,
                        "lme_edge": "ATTACH",
                    },
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
            weak_id = self._add_evidence_row(
                database,
                m_new,
                dict(metadata_base),
                parent_id=pid,
                session_idx=session_idx,
                chunk_scope=chunk_scope,
                trace=trace,
                operation="ATTACH_EVIDENCE",
                metadata_extra={"lme_edge": "ATTACH"},
            )
            self._apply_con_weak_updates(database, weak_id, con_ids, m_new, chunk_scope, trace)
            return 1

        if plan.outcome == "conflict_update":
            meta = dict(metadata_base)
            meta["memory_role"] = "primary"
            emb = self._embed_texts([self.build_text_for_embedding(m_new, metadata=meta)])[0]
            new_id = database.add(
                text=m_new,
                source_index=f"session_{session_idx}",
                time=date_s,
                metadata=meta,
                embedding=emb,
            )
            trace.log_memory_operation(
                operation="ADD",
                memory_id=new_id,
                scope_id=chunk_scope,
                metadata={"m_new": m_new, "reason": plan.reason},
                after={
                    "text": m_new,
                    "source_index": f"session_{session_idx}",
                    "time": date_s,
                    "metadata": dict(meta),
                },
                status="ok",
            )
            self._apply_con_weak_updates(database, new_id, con_ids, m_new, chunk_scope, trace)
            return 1

        meta = dict(metadata_base)
        meta["memory_role"] = "primary"
        emb = self._embed_texts([self.build_text_for_embedding(m_new, metadata=meta)])[0]
        memory_id = database.add(
            text=m_new,
            source_index=f"session_{session_idx}",
            time=date_s,
            metadata=meta,
            embedding=emb,
        )
        trace.log_memory_operation(
            operation="ADD",
            memory_id=memory_id,
            scope_id=chunk_scope,
            metadata={"m_new": m_new, "reason": plan.reason},
            after={
                "text": m_new,
                "source_index": f"session_{session_idx}",
                "time": date_s,
                "metadata": dict(meta),
            },
            status="ok",
        )
        self._apply_con_weak_updates(database, memory_id, con_ids, m_new, chunk_scope, trace)
        return 1
