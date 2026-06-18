"""LME 抽取候选 + 五类关系分类；写库：桶内聚合后的有向弱边（EQUIV / ATTACH / UPDATE），不物理删除。"""

from __future__ import annotations

import logging
import time
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
from .topics import MISC_TOPIC, VALID_TOPICS, normalize_topic
from .schemas import (
    LME_RELATION_CLASSIFICATION_RESPONSE_FORMAT,
    LME_RELATION_VERIFY_RESPONSE_FORMAT,
)
from .relation_decision import LmeRelationDecision, decide_lme_update_relation_decision
from .relation_classifier_backend import RelationClassifierBackend, get_shared_backend
from .prompts import (
    build_lme_answer_fuse_prompt,
    build_lme_relation_classification_user_prompt,
    build_lme_relation_verify_user_prompt,
    lme_relation_system_prompt_for_language,
    lme_relation_verify_system_prompt_for_language,
)

logger = logging.getLogger(__name__)

_VALID_RELATIONS = frozenset({"IND", "EQV", "NSO", "OSN", "CON"})

# plan.reason → 融合提示语用的关系标签
_REASON_TO_RELATION = {"con": "CON", "osn": "OSN", "nso": "NSO", "eqv": "EQV"}


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
        # 只有真正启用级联/删除时才消费平行栏条件标注；两者都关(消融)时退化为
        # 与 baseline 同输入(条件 merge 回文本)，保证对比公平。
        self.consumes_cas_rules = self._cascade_enabled or self._deletion_enabled
        # 同主题 profile 聚合：消费 apply.py 透传的 metadata["topic"]，把同 episode 同主题、
        # 互不冲突的事实融进一条答题记忆 C，使 Agg「列出关于 X 的一切」一次检索带全 slot。
        self._topic_aggregation_enabled = bool(kwargs.pop("topic_aggregation_enabled", True))
        self.consumes_topics = self._topic_aggregation_enabled
        self._condition_sim_threshold = float(kwargs.pop("condition_sim_threshold", 0.5))
        self._pairwise_sim_threshold = float(kwargs.pop("pairwise_sim_threshold", 0.7))
        self._cascade_max_new_tokens = int(kwargs.pop("cascade_max_new_tokens", 512))
        self._answer_fuse_max_new_tokens = int(kwargs.pop("answer_fuse_max_new_tokens", 512))
        trace_log_dir: Optional[str] = kwargs.get("trace_log_dir")
        super().__init__(*args, **kwargs)
        _check_relation_language(self._relation_backend, self.language)
        if self._relation_backend == "classifier":
            # 进程级共享单例：N 个并发 episode 复用一份 backbone，避免显存 N 倍 OOM
            self._rc_backend = get_shared_backend()
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
            _t0 = time.perf_counter()
            label = self._rc_backend.classify(m_old_text, m_new)
            _latency_ms = (time.perf_counter() - _t0) * 1000.0
            trace.log_llm_interaction(
                purpose="lme_candidate_relation_decision_classify_relation",
                messages=[{"role": "user", "content": f"old: {m_old_text}\nnew: {m_new}"}],
                response={"relation": label, "backend": "classifier"},
                scope_id=trace_scope_id,
                metadata={"backend": "classifier", "latency_ms": round(_latency_ms, 2)},
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
        _t0 = time.perf_counter()
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
                metadata={"temperature": 0, "latency_ms": round((time.perf_counter() - _t0) * 1000.0, 2)},
            )
        except Exception as exc:
            trace.log_llm_interaction(
                purpose="lme_candidate_relation_decision_classify_relation",
                messages=messages,
                response=None,
                scope_id=trace_scope_id,
                metadata={"temperature": 0, "latency_ms": round((time.perf_counter() - _t0) * 1000.0, 2)},
                error=str(exc),
            )
            logger.warning("LME relation classification failed: %s", exc)
            return "IND"

        rel = self._parse_relation_label(raw_response)
        return rel if rel in _VALID_RELATIONS else "IND"

    def _verify_relation(
        self,
        m_old_text: str,
        m_new: str,
        relation: str,
        trace_scope_id: Optional[str],
        trace: MemoryTraceLogger,
    ) -> bool:
        """LLM 复核 classifier 预测的非 IND 标签是否成立；失败/拿不准一律否决(False→退回 IND)。"""
        system = lme_relation_verify_system_prompt_for_language(self.language, relation)
        user = build_lme_relation_verify_user_prompt(m_old_text, m_new, relation)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        _t0 = time.perf_counter()
        try:
            raw_response = self.llm_client.get_response_chat(
                messages,
                max_new_tokens=self._relation_max_new_tokens,
                temperature=0,
                response_format=LME_RELATION_VERIFY_RESPONSE_FORMAT,
                verbose=False,
            )
            trace.log_llm_interaction(
                purpose="lme_candidate_relation_decision_verify_relation",
                messages=messages,
                response=raw_response,
                scope_id=trace_scope_id,
                metadata={"relation": relation, "temperature": 0, "latency_ms": round((time.perf_counter() - _t0) * 1000.0, 2)},
            )
        except Exception as exc:
            trace.log_llm_interaction(
                purpose="lme_candidate_relation_decision_verify_relation",
                messages=messages,
                response=None,
                scope_id=trace_scope_id,
                metadata={"relation": relation, "temperature": 0, "latency_ms": round((time.perf_counter() - _t0) * 1000.0, 2)},
                error=str(exc),
            )
            logger.warning("LME relation verify failed: %s", exc)
            return False
        return self._parse_verify_correct(raw_response)

    @staticmethod
    def _parse_verify_correct(raw_response: Any) -> bool:
        payload = raw_response
        if isinstance(payload, (list, tuple)):
            payload = payload[0] if payload else None
        if isinstance(payload, dict):
            return bool(payload.get("correct"))
        if isinstance(payload, str):
            import json as _json
            import re as _re

            m = _re.search(r"\{.*\}", payload, _re.DOTALL)
            if m:
                try:
                    obj = _json.loads(m.group(0))
                    return bool(obj.get("correct"))
                except (ValueError, TypeError):
                    return False
        return False

    def _verify_labels(
        self,
        m_new: str,
        labeled: List[tuple[RetrievedMemory, str]],
        chunk_scope: str,
        trace: MemoryTraceLogger,
    ) -> List[tuple[RetrievedMemory, str]]:
        """对每个非 IND 标签 LLM 复核；否决的退回 IND。IND 直接放行。"""
        out: List[tuple[RetrievedMemory, str]] = []
        for mem, lab in labeled:
            if lab == "IND" or lab not in _VALID_RELATIONS:
                out.append((mem, "IND"))
                continue
            if self._verify_relation(mem.text, m_new, lab, chunk_scope, trace):
                out.append((mem, lab))
            else:
                out.append((mem, "IND"))
        return out

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
        # classifier/LLM 判出非 IND 后，逐对 LLM 复核；否决的退回 IND（只用确认的关系建边）
        labeled = self._verify_labels(m_new, labeled, chunk_scope, trace)
        plan = decide_lme_update_relation_decision(m_new, labeled)
        mb = dict(metadata_base)
        mb["lme_update_method"] = "relation_decision"
        ops, new_row_id = self._execute_lme_plan(
            database,
            m_new,
            mb,
            session_idx,
            chunk_scope,
            trace,
            plan,
        )
        # 关系成立 → 就地增量融合主 cluster 的答题记忆 C（仅用于回答问题）
        if new_row_id is not None and plan.outcome != "fresh_primary":
            self._update_answer_memory(
                database,
                m_new,
                new_row_id,
                plan,
                mb,
                session_idx,
                chunk_scope,
                trace,
            )
        # 同主题 profile 聚合：与关系决策正交，独立把本条事实并入「同 episode 同主题」的
        # profile C，使 Agg「列出关于 X 的一切」一次命中带全 slot。fresh_primary 也聚合
        # （正交 slot 正是被判 IND 才散落的，必须纳入）。
        if new_row_id is not None and self._topic_aggregation_enabled:
            self._aggregate_topic_profile(
                database, m_new, new_row_id, mb, session_idx, chunk_scope, trace
            )
        return ops

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
    ) -> tuple[int, Optional[str]]:
        """写库执行计划。返回 (写库行数, m_new 落地行 id)；行 id 供答题记忆 C 增量融合定位。"""
        con_ids = plan.con_update_ids
        date_s = str(metadata_base.get("date", ""))

        if plan.outcome == "equivalent_evidence":
            rid = plan.representative_id
            if not rid:
                return 0, None
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
            return 1, weak_id

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
            return 1, new_id

        if plan.outcome == "weaker_evidence":
            pid = plan.parent_id
            if not pid:
                return 0, None
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
            return 1, weak_id

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
            return 1, new_id

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
        return 1, memory_id

    # ------------------------------------------------------------------
    # 答题记忆 C：关系成立后就地增量融合（仅供答题检索，不参与后续关系判断）
    # ------------------------------------------------------------------
    def _plan_anchor_and_hide(
        self, plan: LmeRelationDecision, new_row_id: str
    ) -> Optional[tuple[str, str, str]]:
        """返回 (anchor_old_id, hide_primary_id, relation)。

        - CON/OSN：m_new 成主、old 降证据 → 隐藏 m_new(new_row_id)；anchor 取首个降级 old。
        - NSO/EQV：m_new 挂证据、old 仍是主 → 隐藏 old(=anchor)。
        其中 hide_primary_id 是融合后仍为 primary、但内容已并入 C 的那条原子，答题需隐藏它。
        """
        if plan.outcome == "conflict_update":
            if not plan.con_update_ids:
                return None
            return plan.con_update_ids[0], new_row_id, "CON"
        if plan.outcome == "stronger_primary":
            if not plan.demote_ids:
                return None
            return plan.demote_ids[0], new_row_id, "OSN"
        if plan.outcome == "weaker_evidence":
            if not plan.parent_id:
                return None
            return plan.parent_id, plan.parent_id, "NSO"
        if plan.outcome == "equivalent_evidence":
            if not plan.representative_id:
                return None
            return plan.representative_id, plan.representative_id, "EQV"
        return None

    def _update_answer_memory(
        self,
        database: LocalFaissDatabase,
        m_new: str,
        new_row_id: str,
        plan: LmeRelationDecision,
        metadata_base: dict[str, Any],
        session_idx: int,
        chunk_scope: str,
        trace: MemoryTraceLogger,
    ) -> None:
        anchor = self._plan_anchor_and_hide(plan, new_row_id)
        if anchor is None:
            return
        anchor_old_id, hide_primary_id, relation = anchor

        anchor_mem = database.get_memory(anchor_old_id)
        if anchor_mem is None:
            return

        # 主 cluster 现有 C：经 anchor old 的 answer_id 定位；没有则以 anchor old 文本起步
        existing_c_id = str((anchor_mem.metadata or {}).get("answer_id") or "").strip() or None
        existing_c = database.get_memory(existing_c_id) if existing_c_id else None
        current_memory = existing_c.text if existing_c is not None else anchor_mem.text
        # 两侧发生时间：供 EQV 模板区分「同一事件重复提及」与「事件多次发生」。
        current_memory_time = (existing_c.time if existing_c is not None else anchor_mem.time) or ""
        new_fact_time = str(metadata_base.get("date", "") or "")

        fused_text = self._fuse_answer_memory(
            current_memory, m_new, relation, chunk_scope, trace,
            current_memory_time=current_memory_time,
            new_fact_time=new_fact_time,
        )
        if not fused_text:
            return

        emb = self._embed_texts([fused_text])[0]
        members = list((existing_c.metadata or {}).get("fused_member_ids", [])) if existing_c else []
        for mid in (anchor_old_id, new_row_id):
            if mid not in members:
                members.append(mid)

        if existing_c is not None:
            database.update_memory(
                existing_c_id,
                new_text=fused_text,
                new_embedding=emb,
                metadata_updates={
                    "cluster_root": hide_primary_id,
                    "fused_member_ids": members,
                    "fused_member_count": len(members),
                },
            )
            c_id = existing_c_id
            op = "ANSWER_FUSE_UPDATE"
        else:
            c_meta = {
                **{k: v for k, v in metadata_base.items()
                   if k not in ("memory_role", "parent_primary", "edge", "answer_hidden", "answer_id")},
                "memory_role": "answer",
                "answer_fused": True,
                "cluster_root": hide_primary_id,
                "fused_member_ids": members,
                "fused_member_count": len(members),
                "lme_update_method": "relation_decision_answer_fuse",
            }
            c_id = database.add(
                text=fused_text,
                source_index=f"session_{session_idx}",
                time=str(metadata_base.get("date", "")),
                metadata=c_meta,
                embedding=emb,
            )
            op = "ANSWER_FUSE_ADD"

        # 内容已进 C 的那条原子在答题检索隐藏；后续关系判断仍可见（answer_hidden≠stale）
        database.update_memory(
            hide_primary_id,
            metadata_updates={"answer_hidden": True, "answer_id": c_id},
        )
        # anchor old 也指向同一条 C（CON/OSN 下它已是 evidence，仅作可追溯）
        if anchor_old_id != hide_primary_id:
            database.update_memory(
                anchor_old_id,
                metadata_updates={"answer_id": c_id},
            )
        trace.log_memory_operation(
            operation=op,
            memory_id=c_id,
            scope_id=chunk_scope,
            metadata={
                "relation": relation,
                "hide_primary_id": hide_primary_id,
                "anchor_old_id": anchor_old_id,
                "members": members,
            },
            status="ok",
        )

    def _fuse_answer_memory(
        self,
        current_memory: str,
        new_fact: str,
        relation: str,
        chunk_scope: str,
        trace: MemoryTraceLogger,
        *,
        current_memory_time: str = "",
        new_fact_time: str = "",
    ) -> str:
        """LLM 增量融合 current_memory + new_fact → 新 C 文本；失败回退拼接。"""
        prompt = build_lme_answer_fuse_prompt(
            current_memory,
            new_fact,
            relation,
            language=self.language,
            current_memory_time=current_memory_time,
            new_fact_time=new_fact_time,
        )
        messages = [{"role": "user", "content": prompt}]
        _t0 = time.perf_counter()
        try:
            raw = self.llm_client.get_response_chat(
                messages,
                max_new_tokens=self._answer_fuse_max_new_tokens,
                temperature=0,
                verbose=False,
            )
            trace.log_llm_interaction(
                purpose="lme_candidate_relation_decision_answer_fuse",
                messages=messages,
                response=raw,
                scope_id=chunk_scope,
                metadata={"relation": relation, "temperature": 0, "latency_ms": round((time.perf_counter() - _t0) * 1000.0, 2)},
            )
        except Exception as exc:
            trace.log_llm_interaction(
                purpose="lme_candidate_relation_decision_answer_fuse",
                messages=messages,
                response=None,
                scope_id=chunk_scope,
                metadata={"relation": relation, "temperature": 0, "latency_ms": round((time.perf_counter() - _t0) * 1000.0, 2)},
                error=str(exc),
            )
            logger.warning("LME answer fuse failed: %s", exc)
            return f"{current_memory}\n{new_fact}".strip()
        text = raw[0] if isinstance(raw, (list, tuple)) and raw else raw
        text = str(text or "").strip()
        return text or f"{current_memory}\n{new_fact}".strip()

    # ------------------------------------------------------------------
    # 同主题 profile 聚合（与关系决策正交）
    # ------------------------------------------------------------------
    def _find_topic_profile(
        self, database: LocalFaissDatabase, topic: str
    ) -> Optional[RetrievedMemory]:
        """同 episode 内找已存在的该主题 profile C（memory_role=answer 且 topic_profile=True）。"""
        for mem in database.list_all_memories(sort_by_time=False):
            meta = mem.metadata or {}
            if (
                meta.get("memory_role") == "answer"
                and bool(meta.get("topic_profile"))
                and str(meta.get("topic") or "") == topic
            ):
                return mem
        return None

    def _aggregate_topic_profile(
        self,
        database: LocalFaissDatabase,
        m_new: str,
        new_row_id: str,
        metadata_base: dict[str, Any],
        session_idx: int,
        chunk_scope: str,
        trace: MemoryTraceLogger,
    ) -> None:
        """把本条事实并入「同 episode 同主题」的 profile C（答题可见），一次检索带全 slot。

        profile C 与关系 cluster 的 C 是两条独立的答题可见行；成员原子保留可见（Agg 靠
        profile、ER/Tr 靠原子兜底）。``misc`` / 非法主题不聚合。
        """
        topic = normalize_topic(metadata_base.get("topic"))
        if topic == MISC_TOPIC or topic not in VALID_TOPICS:
            return

        existing = self._find_topic_profile(database, topic)
        current_memory = existing.text if existing is not None else ""
        fused_text = (
            self._fuse_answer_memory(current_memory, m_new, "AGG", chunk_scope, trace)
            if current_memory
            else m_new
        )
        if not fused_text:
            return

        emb = self._embed_texts([fused_text])[0]
        members = list((existing.metadata or {}).get("fused_member_ids", [])) if existing else []
        if new_row_id not in members:
            members.append(new_row_id)

        if existing is not None:
            database.update_memory(
                existing.memory_id,
                new_text=fused_text,
                new_embedding=emb,
                metadata_updates={
                    "fused_member_ids": members,
                    "fused_member_count": len(members),
                },
            )
            c_id = existing.memory_id
            op = "TOPIC_PROFILE_UPDATE"
        else:
            c_meta = {
                **{k: v for k, v in metadata_base.items()
                   if k not in ("memory_role", "parent_primary", "edge", "answer_hidden", "answer_id")},
                "memory_role": "answer",
                "answer_fused": True,
                "topic_profile": True,
                "topic": topic,
                "fused_member_ids": members,
                "fused_member_count": len(members),
                "lme_update_method": "relation_decision_topic_profile",
            }
            c_id = database.add(
                text=fused_text,
                source_index=f"session_{session_idx}",
                time=str(metadata_base.get("date", "")),
                metadata=c_meta,
                embedding=emb,
            )
            op = "TOPIC_PROFILE_ADD"

        trace.log_memory_operation(
            operation=op,
            memory_id=c_id,
            scope_id=chunk_scope,
            metadata={"topic": topic, "new_row_id": new_row_id, "members": members},
            status="ok",
        )
