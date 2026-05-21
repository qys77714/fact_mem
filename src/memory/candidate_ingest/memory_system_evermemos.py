"""EverMemOS 灌库适配：增量语义聚类 + 聚类内 LLM 合并。

核心算法源自 EverMemOS ClusterManager（EverOS/src/memory_layer/cluster_manager/manager.py），
已去除 async / MongoDB 依赖，改用项目已有的 _embed_texts() 和 llm_client。

两阶段处理：
  Phase 1（_process_one_new_fact）：embed + 余弦相似度 + 时间窗口分配聚类，事实暂存内存，不写 DB。
  Phase 2（finalize_episode）：对每个聚类调用 LLM 合并 → primary 写 DB；原始成员写为 evidence 行。
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from memory.storage.local_faiss import LocalFaissDatabase
from memory.tracing import MemoryTraceLogger
from prompts import render_prompt

from .memory_system_base import LmeCandidateMemorySystemBase

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

_DEFAULT_CONSOLIDATION_TEMPLATE_EN = "evermemos_consolidate_en.jinja"
_DEFAULT_CONSOLIDATION_TEMPLATE_ZH = "evermemos_consolidate_zh.jinja"


# ---------------------------------------------------------------------------
# In-memory cluster state (simplified ClusterState from EverOS)
# ---------------------------------------------------------------------------

@dataclass
class _ClusterState:
    """Per-episode in-memory cluster state (no DB persistence)."""
    cluster_centroids: Dict[str, np.ndarray] = field(default_factory=dict)
    cluster_counts: Dict[str, int] = field(default_factory=dict)
    cluster_last_ts: Dict[str, Optional[float]] = field(default_factory=dict)
    next_cluster_idx: int = 0

    def new_cluster_id(self) -> str:
        cid = f"evermemos_cluster_{self.next_cluster_idx:04d}"
        self.next_cluster_idx += 1
        return cid

    def add(self, cluster_id: str, vector: np.ndarray, ts: Optional[float]) -> None:
        count = self.cluster_counts.get(cluster_id, 0)
        if count == 0:
            self.cluster_centroids[cluster_id] = vector.astype(np.float32, copy=False)
        else:
            centroid = self.cluster_centroids[cluster_id]
            new_centroid = (centroid * float(count) + vector) / float(count + 1)
            self.cluster_centroids[cluster_id] = new_centroid.astype(np.float32, copy=False)
        self.cluster_counts[cluster_id] = count + 1
        if ts is not None:
            prev = self.cluster_last_ts.get(cluster_id)
            self.cluster_last_ts[cluster_id] = max(prev, ts) if prev is not None else ts


# ---------------------------------------------------------------------------
# EverMemOSMemorySystem
# ---------------------------------------------------------------------------

class EverMemOSMemorySystem(LmeCandidateMemorySystemBase):
    """
    Adapts EverMemOS memory management to the fact_memory ingest pipeline.

    Each candidate fact is:
      1. Embedded and assigned to a semantic cluster (incremental cosine clustering).
      2. Deferred until finalize_episode() is called for the current episode.
      3. LLM-merged per cluster → consolidated primary + original members as evidence rows.

    Embedding:  self._embed_texts()        → embedding_model
    LLM merge:  self.llm_client            → manager_model
    """

    def __init__(
        self,
        *args: Any,
        similarity_threshold: float = 0.65,
        max_time_gap_days: float = 7.0,
        consolidation_template_en: Optional[str] = None,
        consolidation_template_zh: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        trace_log_dir: Optional[str] = kwargs.get("trace_log_dir")
        super().__init__(*args, **kwargs)
        self._similarity_threshold = float(similarity_threshold)
        self._max_time_gap_seconds = float(max_time_gap_days) * 86400.0
        self._consolidation_template_en = (
            (consolidation_template_en or "").strip() or _DEFAULT_CONSOLIDATION_TEMPLATE_EN
        )
        self._consolidation_template_zh = (
            (consolidation_template_zh or "").strip() or _DEFAULT_CONSOLIDATION_TEMPLATE_ZH
        )
        self.trace = MemoryTraceLogger(
            method="lme_candidate_evermemos",
            log_dir=trace_log_dir or "logs/memory_trace",
            use_experiment_naming=trace_log_dir is not None,
        )
        # Per-episode mutable state (reset by _reset_episode_state before each episode)
        self._cluster_state = _ClusterState()
        self._pending: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def _reset_episode_state(self) -> None:
        """Clear in-memory cluster state before processing a new episode."""
        self._cluster_state = _ClusterState()
        self._pending = []

    # ------------------------------------------------------------------
    # Phase 1: incremental clustering (called per-fact by apply.py)
    # ------------------------------------------------------------------

    def _process_one_new_fact(
        self,
        database: LocalFaissDatabase,
        m_new: str,
        metadata_base: Dict[str, Any],
        session_idx: int,
        chunk_scope: str,
        trace: MemoryTraceLogger,
    ) -> int:
        """Embed the fact and assign it to a cluster. DB writes are deferred to finalize_episode()."""
        m_new = (m_new or "").strip()
        if not m_new:
            return 0

        embed_text = self.build_text_for_embedding(m_new, metadata=metadata_base)
        emb_arr = self._embed_texts([embed_text])
        if emb_arr.size == 0:
            logger.warning("evermemos: embedding failed for fact, skipping: %r", m_new[:80])
            return 0
        emb = emb_arr[0].astype(np.float32)

        ts = _parse_date_to_timestamp(str(metadata_base.get("date") or ""))
        cluster_id = self._assign_cluster(emb, ts)

        self._pending.append({
            "text": m_new,
            "embedding": emb,
            "metadata": dict(metadata_base),
            "session_idx": session_idx,
            "cluster_id": cluster_id,
            "chunk_scope": chunk_scope,
        })
        return 0  # DB ops are deferred

    def _assign_cluster(self, vector: np.ndarray, ts: Optional[float]) -> str:
        """Find best matching cluster by cosine similarity + time gap, or create new one."""
        state = self._cluster_state
        best_sim = -1.0
        best_cid: Optional[str] = None
        v_norm = float(np.linalg.norm(vector)) + 1e-9

        for cid, centroid in state.cluster_centroids.items():
            if centroid is None or centroid.size == 0:
                continue
            # Time gap check
            if ts is not None:
                last_ts = state.cluster_last_ts.get(cid)
                if last_ts is not None and abs(ts - last_ts) > self._max_time_gap_seconds:
                    continue
            c_norm = float(np.linalg.norm(centroid)) + 1e-9
            sim = float(np.dot(centroid, vector) / (c_norm * v_norm))
            if sim > best_sim:
                best_sim = sim
                best_cid = cid

        if best_sim >= self._similarity_threshold and best_cid is not None:
            state.add(best_cid, vector, ts)
            return best_cid

        # Create new cluster
        new_cid = state.new_cluster_id()
        state.add(new_cid, vector, ts)
        return new_cid

    # ------------------------------------------------------------------
    # Phase 2: consolidation + DB writes (called once per episode)
    # ------------------------------------------------------------------

    def finalize_episode(
        self,
        database: LocalFaissDatabase,
        history_name: str,
    ) -> int:
        """
        Consolidate clusters and write primaries + evidence rows to the database.
        Returns total number of DB row writes.
        """
        if not self._pending:
            return 0

        trace = self.trace.get_logger_for(history_name)
        ep_scope = trace.create_scope(
            "evermemos_finalize_episode",
            metadata={"history_name": history_name, "pending_facts": len(self._pending)},
        )

        # Group pending items by cluster
        clusters: Dict[str, List[Dict[str, Any]]] = {}
        for item in self._pending:
            clusters.setdefault(item["cluster_id"], []).append(item)

        total_ops = 0
        try:
            # Parallel consolidation across clusters (reuse relation_concurrency setting)
            workers = min(self._relation_concurrency, len(clusters))
            if workers <= 1:
                for cid, members in clusters.items():
                    ops = self._write_cluster(database, cid, members, ep_scope, trace)
                    total_ops += ops
            else:
                futures: Dict[Any, str] = {}
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    for cid, members in clusters.items():
                        fut = pool.submit(
                            self._write_cluster, database, cid, members, ep_scope, trace
                        )
                        futures[fut] = cid
                    for fut in as_completed(futures):
                        try:
                            total_ops += fut.result()
                        except Exception as exc:
                            logger.warning("evermemos: cluster write failed (%s): %s", futures[fut], exc)

            trace.close_scope(
                ep_scope,
                status="ok",
                metadata={"total_db_ops": total_ops, "num_clusters": len(clusters)},
            )
        except Exception:
            trace.close_scope(ep_scope, status="error")
            raise

        return total_ops

    def _write_cluster(
        self,
        database: LocalFaissDatabase,
        cluster_id: str,
        members: List[Dict[str, Any]],
        ep_scope: str,
        trace: MemoryTraceLogger,
    ) -> int:
        """Write one cluster to the database. Returns number of rows written."""
        if not members:
            return 0

        # Representative metadata and session from the first member
        rep = members[0]
        session_idx = rep["session_idx"]
        date_s = str(rep["metadata"].get("date", ""))
        source_index = f"session_{session_idx}"

        if len(members) == 1:
            # Single-member cluster → write directly as primary
            item = members[0]
            meta = dict(item["metadata"])
            meta["memory_role"] = "primary"
            meta["lme_update_method"] = "evermemos"
            meta["evermemos_cluster_id"] = cluster_id
            mid = database.add(
                text=item["text"],
                source_index=source_index,
                time=date_s,
                metadata=meta,
                embedding=item["embedding"],
            )
            trace.log_memory_operation(
                operation="ADD",
                memory_id=mid,
                scope_id=ep_scope,
                metadata={"strategy": "evermemos_singleton", "cluster_id": cluster_id},
                after={"text": item["text"], "source_index": source_index, "time": date_s},
                status="ok",
            )
            return 1

        # Multi-member cluster → LLM consolidation
        consolidated_text = self._consolidate_cluster(members, ep_scope, trace)

        # Embed consolidated text and write as primary
        primary_meta: Dict[str, Any] = dict(rep["metadata"])
        primary_meta["memory_role"] = "primary"
        primary_meta["lme_update_method"] = "evermemos"
        primary_meta["evermemos_cluster_id"] = cluster_id
        primary_meta["evermemos_cluster_size"] = len(members)
        primary_emb = self._embed_texts(
            [self.build_text_for_embedding(consolidated_text, metadata=primary_meta)]
        )[0]

        primary_id = database.add(
            text=consolidated_text,
            source_index=source_index,
            time=date_s,
            metadata=primary_meta,
            embedding=primary_emb,
        )
        trace.log_memory_operation(
            operation="ADD",
            memory_id=primary_id,
            scope_id=ep_scope,
            metadata={
                "strategy": "evermemos_consolidated_primary",
                "cluster_id": cluster_id,
                "cluster_size": len(members),
            },
            after={"text": consolidated_text, "source_index": source_index, "time": date_s},
            status="ok",
        )
        ops = 1

        # Write original members as evidence rows
        for item in members:
            ev_meta = dict(item["metadata"])
            ev_meta["memory_role"] = "evidence"
            ev_meta["parent_primary"] = primary_id
            ev_meta["lme_update_method"] = "evermemos"
            ev_meta["evermemos_cluster_id"] = cluster_id
            ev_source = f"session_{item['session_idx']}"
            ev_time = str(item["metadata"].get("date", ""))
            ev_id = database.add(
                text=item["text"],
                source_index=ev_source,
                time=ev_time,
                metadata=ev_meta,
                embedding=item["embedding"],
            )
            trace.log_memory_operation(
                operation="ADD_EVIDENCE",
                memory_id=ev_id,
                scope_id=ep_scope,
                metadata={
                    "strategy": "evermemos_original_member",
                    "parent_primary": primary_id,
                    "cluster_id": cluster_id,
                },
                after={"text": item["text"], "source_index": ev_source, "time": ev_time},
                status="ok",
            )
            ops += 1

        return ops

    def _consolidate_cluster(
        self,
        members: List[Dict[str, Any]],
        scope_id: str,
        trace: MemoryTraceLogger,
    ) -> str:
        """Call LLM to merge cluster members into a single comprehensive memory text."""
        texts = [m["text"] for m in members]
        facts_text = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))

        template = (
            self._consolidation_template_zh
            if self.language == "zh"
            else self._consolidation_template_en
        )
        user_content = render_prompt(template, facts_text=facts_text)
        messages = [{"role": "user", "content": user_content}]

        try:
            raw = self.llm_client.get_response_chat(
                messages,
                max_new_tokens=self._manager_max_new_tokens,
                temperature=0,
                verbose=False,
            )
            trace.log_llm_interaction(
                purpose="evermemos_consolidate_cluster",
                messages=messages,
                response=raw,
                scope_id=scope_id,
                metadata={"cluster_size": len(members)},
            )
        except Exception as exc:
            trace.log_llm_interaction(
                purpose="evermemos_consolidate_cluster",
                messages=messages,
                response=None,
                scope_id=scope_id,
                metadata={"cluster_size": len(members)},
                error=str(exc),
            )
            logger.warning("evermemos: LLM consolidation failed, falling back to concatenation: %s", exc)
            return " ".join(t.strip() for t in texts if t.strip())

        result = (raw or "").strip()
        return result if result else " ".join(t.strip() for t in texts if t.strip())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date_to_timestamp(date_s: str) -> Optional[float]:
    """Parse a session date string (e.g. '2023-01-15') to a Unix timestamp float.

    Returns None if the string is empty or unparseable.
    """
    s = (date_s or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s).timestamp()
    except (ValueError, TypeError):
        pass
    return None
