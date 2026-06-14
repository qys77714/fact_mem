"""LME 抽取候选 + 五类关系分类；写库：桶内聚合后的有向弱边（EQUIV / ATTACH / UPDATE），不物理删除。"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from memory.base import RetrievedMemory
from memory.storage.local_faiss import LocalFaissDatabase
from memory.tracing import MemoryTraceLogger

from .cas_update import (
    apply_if_then_enrich,
    build_cascade_metadata,
    decide_cas_cascade_sync,
    find_primary_for_if_then,
    get_pending_rules,
    is_if_then_text,
    match_prior_conditions,
    metadata_for_new_primary,
    normalize_primary_from_context,
    skip_condition_match,
)
from .deletion_update import (
    apply_user_deletion,
    find_deletion_target,
    is_user_deletion_request,
)
from .memory_system_base import LmeCandidateMemorySystemBase
from .schemas import LME_RELATION_CLASSIFICATION_RESPONSE_FORMAT
from .relation_decision import LmeRelationDecision, decide_lme_update_relation_decision
from .relation_classifier_backend import RelationClassifierBackend
from .prompts import (
    build_lme_relation_classification_user_prompt,
    lme_relation_system_prompt_for_language,
)

logger = logging.getLogger(__name__)

_VALID_RELATIONS = frozenset({"IND", "EQV", "NSO", "OSN", "CON"})


def _check_relation_language(relation_backend: str, language: str) -> None:
    """Raise ValueError if the backend/language combination is unsupported."""
    if relation_backend == "classifier" and language != "en":
        raise ValueError(
            "relation_backend='classifier' 只支持英文（language='en'），"
            f"当前 language={language!r}。请用 relation_backend='llm' 或英文输入。"
        )


class LmeCandidateRelationDecisionMemorySystem(LmeCandidateMemorySystemBase):
    """
    relation_decision with optional cascade-first layer (cas_update_condition).

    When cascade_enabled:
      - all memories: match all prior cas_update_conditions above threshold → two-step cas cascade LLM per match
      - any non-NO_ACTION action skips pairwise relation_decision
      - all NO_ACTION or no match → standard pairwise relation_decision
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._relation_system_en_template = (kwargs.pop("relation_system_en_template", None) or "").strip() or None
        self._relation_system_zh_template = (kwargs.pop("relation_system_zh_template", None) or "").strip() or None
        self._relation_user_template = (kwargs.pop("relation_user_template", None) or "").strip() or None
        self._relation_backend = (kwargs.pop("relation_backend", "classifier") or "classifier")
        self._cascade_enabled = bool(kwargs.pop("cascade_enabled", True))
        self._deletion_enabled = bool(kwargs.pop("deletion_enabled", True))
        self._condition_sim_threshold = float(kwargs.pop("condition_sim_threshold", 0.5))
        self._pairwise_sim_threshold = float(kwargs.pop("pairwise_sim_threshold", 0.7))
        self._cascade_max_new_tokens = int(kwargs.pop("cascade_max_new_tokens", 512))
        trace_log_dir: Optional[str] = kwargs.get("trace_log_dir")
        super().__init__(*args, **kwargs)
        _check_relation_language(self._relation_backend, self.language)
        if self._relation_backend == "classifier":
            self._rc_backend = RelationClassifierBackend()
        else:
            self._rc_backend = None
        self.trace = MemoryTraceLogger(
            method="lme_candidate_relation_decision",
            log_dir=trace_log_dir or "logs/memory_trace",
            use_experiment_naming=trace_log_dir is not None,
        )

    def build_text_for_embedding(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        meta = metadata or {}
        primary = str(meta.get("primary_text") or text).strip()
        return primary or text

    def _classify_relation(
        self,
        m_old_text: str,
        m_new: str,
        trace_scope_id: Optional[str],
        trace: MemoryTraceLogger,
    ) -> str:
        if self._relation_backend == "classifier":
            label = self._rc_backend.classify(m_old_text, m_new)
            trace.log_llm_interaction(
                purpose="lme_candidate_relation_decision_classify_relation",
                messages=[{"role": "user", "content": f"old: {m_old_text}\nnew: {m_new}"}],
                response={"relation": label, "backend": "classifier"},
                scope_id=trace_scope_id,
                metadata={"backend": "classifier"},
            )
            return label if label in _VALID_RELATIONS else "IND"

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
        for mid in con_update_ids:
            ok = database.update_memory(
                mid,
                metadata_updates={
                    "memory_role": "evidence",
                    "parent_primary": new_row_id,
                    "edge": "UPDATE",
                },
            )
            trace.log_memory_operation(
                operation="UPDATE_REL",
                memory_id=mid,
                scope_id=chunk_scope,
                metadata={"m_new": m_new, "update_target": new_row_id},
                status="ok" if ok else "failed",
            )

    def _apply_cascade_decision_on_match(
        self,
        database: LocalFaissDatabase,
        m_new: str,
        match: Any,
        metadata_base: dict[str, Any],
        session_idx: int,
        chunk_scope: str,
        trace: MemoryTraceLogger,
    ) -> Optional[int]:
        """Execute cascade LLM action on one match. Returns op count or None."""
        mem = match.memory
        meta = mem.metadata or {}
        decision = decide_cas_cascade_sync(
            self.llm_client,
            new_memory=m_new,
            matched_condition=str(meta.get("cas_update_condition") or ""),
            linked_primary=mem.text,
            max_new_tokens=self._cascade_max_new_tokens,
            trace=trace,
            trace_scope_id=chunk_scope,
        )
        action = str(decision.get("action", "NO_ACTION")).upper()
        if action == "NO_ACTION":
            return None

        new_text = str(decision.get("new_primary_text") or "").strip()
        if not new_text:
            return 1
        self._cas_apply_update_primary(
            database,
            mem.memory_id,
            mem.text,
            meta,
            new_text,
            metadata_base,
            session_idx,
            chunk_scope,
            trace,
        )
        return 1

    def _try_cascade_update(
        self,
        database: LocalFaissDatabase,
        m_new: str,
        metadata_base: dict[str, Any],
        session_idx: int,
        chunk_scope: str,
        trace: MemoryTraceLogger,
    ) -> Optional[int]:
        """All threshold matches cascade layer (single pass). None → fallback to pairwise."""
        matches = match_prior_conditions(
            database,
            m_new,
            self._embed_texts,
            self._condition_sim_threshold,
        )
        if not matches:
            return None

        total_ops = 0
        for match in matches:
            if skip_condition_match(m_new, match.memory):
                continue
            ops = self._apply_cascade_decision_on_match(
                database,
                m_new,
                match,
                metadata_base,
                session_idx,
                chunk_scope,
                trace,
            )
            if ops is None:
                continue
            total_ops += ops

        if total_ops == 0:
            return None
        return total_ops

    def _dense_candidates(self, database: LocalFaissDatabase, m_new: str) -> List[RetrievedMemory]:
        """Retrieve primaries with score >= pairwise threshold, then cap at related_top_k."""
        fact_emb = self._embed_texts([m_new])[0]
        search_k = max(self.related_memory_top_k * 8, 32)
        raw = database.search(
            fact_emb,
            search_k,
            only_primary=True,
        )
        filtered = [m for m in raw if m.score >= self._pairwise_sim_threshold]
        return filtered[: self.related_memory_top_k]

    def _run_pairwise_relation_decision(
        self,
        database: LocalFaissDatabase,
        m_new: str,
        metadata_base: dict[str, Any],
        session_idx: int,
        chunk_scope: str,
        trace: MemoryTraceLogger,
    ) -> int:
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

    def _cas_add_primary(
        self,
        database: LocalFaissDatabase,
        primary_text: str,
        cas_update_condition: Optional[str],
        metadata_base: dict[str, Any],
        session_idx: int,
        chunk_scope: str,
        trace: MemoryTraceLogger,
        *,
        pending_rules: Optional[List[str]] = None,
    ) -> str:
        date_s = str(metadata_base.get("date", ""))
        meta = build_cascade_metadata(
            metadata_base,
            primary_text=primary_text,
            cas_update_condition=cas_update_condition,
            pending_rules=pending_rules,
        )
        emb = self._embed_texts([self.build_text_for_embedding(primary_text, metadata=meta)])[0]
        memory_id = database.add(
            text=primary_text,
            source_index=f"session_{session_idx}",
            time=date_s,
            metadata=meta,
            embedding=emb,
        )
        trace.log_memory_operation(
            operation="ADD_CASCADE",
            memory_id=memory_id,
            scope_id=chunk_scope,
            metadata={"primary_text": primary_text, "has_condition": bool(cas_update_condition)},
            after={
                "text": primary_text,
                "source_index": f"session_{session_idx}",
                "time": date_s,
                "metadata": dict(meta),
            },
            status="ok",
        )
        return memory_id

    def _cas_apply_update_primary(
        self,
        database: LocalFaissDatabase,
        old_id: str,
        old_text: str,
        old_meta: Dict[str, Any],
        new_primary_text: str,
        metadata_base: dict[str, Any],
        session_idx: int,
        chunk_scope: str,
        trace: MemoryTraceLogger,
    ) -> str:
        normalized = normalize_primary_from_context(old_text, new_primary_text)
        pending = get_pending_rules(old_meta)
        new_id = self._cas_add_primary(
            database,
            normalized,
            old_meta.get("cas_update_condition"),
            metadata_base,
            session_idx,
            chunk_scope,
            trace,
            pending_rules=pending,
        )
        database.update_memory(
            old_id,
            metadata_updates={
                "memory_role": "evidence",
                "parent_primary": new_id,
                "edge": "UPDATE",
                "stale": True,
            },
        )
        trace.log_memory_operation(
            operation="CASCADE_UPDATE_PRIMARY",
            memory_id=old_id,
            scope_id=chunk_scope,
            metadata={"new_primary_id": new_id, "new_primary_text": normalized},
            status="ok",
        )
        return new_id

    def _cas_apply_invalidate(
        self,
        database: LocalFaissDatabase,
        old_id: str,
        old_text: str,
        old_meta: Dict[str, Any],
        metadata_base: dict[str, Any],
        session_idx: int,
        chunk_scope: str,
        trace: MemoryTraceLogger,
    ) -> str:
        cond = str(old_meta.get("cas_update_condition") or "")
        if cond:
            cond += " [INVALIDATED: upstream change, value uncertain]"
        database.update_memory(
            old_id,
            metadata_updates={"stale": True, "cas_update_condition": cond},
        )
        uncertain = (
            f"This fact is uncertain — previously '{old_text}', "
            f"but an upstream dependency changed and I do not know the current value."
        )
        new_id = self._cas_add_primary(
            database,
            uncertain,
            None,
            metadata_base,
            session_idx,
            chunk_scope,
            trace,
        )
        trace.log_memory_operation(
            operation="CASCADE_INVALIDATE",
            memory_id=old_id,
            scope_id=chunk_scope,
            metadata={"uncertainty_primary_id": new_id},
            status="ok",
        )
        return new_id

    def _handle_if_then_enrich(
        self,
        database: LocalFaissDatabase,
        m_new: str,
        session_idx: int,
        chunk_scope: str,
        trace: MemoryTraceLogger,
    ) -> int:
        match = find_primary_for_if_then(database, m_new, self._embed_texts)
        if match is None:
            trace.log_memory_operation(
                operation="CASCADE_IF_THEN_ORPHAN",
                memory_id="",
                scope_id=chunk_scope,
                metadata={"rule": m_new},
                status="ok",
            )
            return 0

        mem = match.memory
        meta = mem.metadata or {}
        condition_after = apply_if_then_enrich(
            database,
            mem.memory_id,
            meta,
            m_new,
        )
        trace.log_memory_operation(
            operation="CASCADE_IF_THEN_ENRICH",
            memory_id=mem.memory_id,
            scope_id=chunk_scope,
            metadata={
                "rule": m_new,
                "linked_primary": mem.text,
                "condition_after": condition_after,
            },
            status="ok",
        )
        return 1

    def _try_user_deletion(
        self,
        database: LocalFaissDatabase,
        m_new: str,
        metadata_base: dict[str, Any],
        session_idx: int,
        chunk_scope: str,
        trace: MemoryTraceLogger,
    ) -> int:
        target, match_debug = find_deletion_target(
            database,
            m_new,
            self._embed_texts,
            self._pairwise_sim_threshold,
            language=self.language,
        )
        return apply_user_deletion(
            database,
            m_new,
            target,
            metadata_base,
            session_idx,
            chunk_scope,
            trace,
            self._embed_texts,
            match_debug=match_debug,
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

        if self._deletion_enabled and is_user_deletion_request(m_new, language=self.language):
            return self._try_user_deletion(
                database, m_new, metadata_base, session_idx, chunk_scope, trace
            )

        if self._cascade_enabled and is_if_then_text(m_new):
            return self._handle_if_then_enrich(
                database, m_new, session_idx, chunk_scope, trace
            )

        if self._cascade_enabled:
            cascade_ops = self._try_cascade_update(
                database, m_new, metadata_base, session_idx, chunk_scope, trace
            )
            if cascade_ops is not None:
                return cascade_ops

        return self._run_pairwise_relation_decision(
            database, m_new, metadata_base, session_idx, chunk_scope, trace
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
                metadata_extra={"edge": "EQUIV"},
            )
            self._apply_con_weak_updates(database, weak_id, con_ids, m_new, chunk_scope, trace)
            return 1

        if plan.outcome == "stronger_primary":
            meta = metadata_for_new_primary(metadata_base, m_new)
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
                        "edge": "ATTACH",
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
                metadata_extra={"edge": "ATTACH"},
            )
            self._apply_con_weak_updates(database, weak_id, con_ids, m_new, chunk_scope, trace)
            return 1

        if plan.outcome == "conflict_update":
            meta = metadata_for_new_primary(metadata_base, m_new)
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

        meta = metadata_for_new_primary(metadata_base, m_new)
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
