"""LME candidate ingest: A-MAC admission + add_all-style primary writes."""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from memory.admission import AmacAdmissionScorer, AmacCandidate, parse_amac_weights_arg
from memory.storage.local_faiss import LocalFaissDatabase
from memory.tracing import MemoryTraceLogger

from .memory_system_base import LmeCandidateMemorySystemBase


class LmeCandidateAmacMemorySystem(LmeCandidateMemorySystemBase):
    """Score each candidate with A-MAC; on accept, ADD primary (same as add_all)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        amac_weights: Union[str, np.ndarray] = kwargs.pop("amac_weights", "0.2,0.2,0.2,0.2,0.2")
        amac_threshold: float = float(kwargs.pop("amac_threshold", 0.5))
        amac_skip_utility: bool = bool(kwargs.pop("amac_skip_utility", False))
        amac_recency_decay: float = float(kwargs.pop("amac_recency_decay_per_step", 0.12))
        amac_novelty_max: int = int(kwargs.pop("amac_novelty_max_existing", 64))
        trace_log_dir: Optional[str] = kwargs.get("trace_log_dir")
        super().__init__(*args, **kwargs)
        if isinstance(amac_weights, str):
            w = parse_amac_weights_arg(amac_weights)
        else:
            w = np.asarray(amac_weights, dtype=np.float64).reshape(5)
            if w.shape[0] != 5 or np.any(w < 0) or float(np.sum(w)) <= 0:
                raise ValueError("amac_weights array must be length 5, non-negative, positive sum")
            w = w / float(np.sum(w))

        self._amac_scorer = AmacAdmissionScorer(
            w,
            amac_threshold,
            llm_client=self.llm_client,
            embed_texts=self._embed_texts,
            max_new_tokens=self._manager_max_new_tokens,
            language=self.language,
            skip_utility=amac_skip_utility,
            recency_decay_per_step=amac_recency_decay,
            novelty_max_existing=amac_novelty_max,
        )
        self.trace = MemoryTraceLogger(
            method="lme_candidate_amac",
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

        obs = str(metadata_base.get("amac_observation") or "")
        chunk_ordinal = int(metadata_base.get("amac_chunk_ordinal") or 0)
        chunk_count = max(1, int(metadata_base.get("amac_chunk_count") or 1))
        existing = database.list_primary_texts_ordered()

        cand = AmacCandidate(content=m_new, metadata=dict(metadata_base))
        should_admit, score, feats = self._amac_scorer.admit(
            cand,
            obs,
            existing,
            chunk_ordinal=chunk_ordinal,
            chunk_count=chunk_count,
        )

        if not should_admit:
            trace.log_memory_operation(
                operation="AMAC_REJECT",
                memory_id=None,
                scope_id=chunk_scope,
                metadata={
                    "m_new": m_new,
                    "score": score,
                    "utility": feats.utility,
                    "confidence": feats.confidence,
                    "novelty": feats.novelty,
                    "recency": feats.recency,
                    "type_prior": feats.type_prior,
                },
                status="skip",
            )
            return 0

        meta = dict(metadata_base)
        meta["memory_role"] = "primary"
        meta["lme_update_method"] = "amac"
        meta["amac_score"] = score
        meta["amac_utility"] = feats.utility
        meta["amac_confidence"] = feats.confidence
        meta["amac_novelty"] = feats.novelty
        meta["amac_recency"] = feats.recency
        meta["amac_type_prior"] = feats.type_prior

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
            metadata={"m_new": m_new, "strategy": "amac", "amac_score": score},
            after={
                "text": m_new,
                "source_index": f"session_{session_idx}",
                "time": str(metadata_base.get("date", "")),
                "metadata": dict(meta),
            },
            status="ok",
        )
        return 1
