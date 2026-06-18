"""
预灌库评测：向量库已由 ``pipeline/ingest_candidates`` 等方法写入，
本系统 **不写** 对话 session，仅按查询做 dense 检索。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from benchmark.base import ChatSession
from memory.base import BaseMemorySystem, RetrievedMemory
from memory.storage.local_faiss import LocalFaissDatabase

logger = logging.getLogger(__name__)


class PrebuiltMemorySystem(BaseMemorySystem):
    """``store_*`` 为空操作；``retrieve`` 与 RAG 相同（不按 only_primary 过滤）。"""

    def __init__(
        self,
        *,
        use_hybrid_retrieval: bool = False,
        hybrid_dense_weight: float = 0.5,
        hybrid_bm25_weight: float = 0.5,
        hybrid_pool_mult: int = 4,
        hybrid_full_corpus_pool: bool = False,
        unfused_rank_database_root: Optional[str] = None,
        answer_mode: bool = False,
        language: str = "en",
        **kwargs: Any,
    ) -> None:
        # ``pipeline_lme_generate`` passes mem0-style kwargs (granularity, trace_log_dir, …);
        # only BaseMemorySystem's constructor args may go to ``super()`` .
        _base_keys = ("embed_client", "embed_model_name", "llm_client", "database_root")
        super().__init__(**{k: kwargs[k] for k in _base_keys if k in kwargs})
        self._databases: dict[str, LocalFaissDatabase] = {}
        self._unfused_databases: dict[str, LocalFaissDatabase] = {}
        self._use_hybrid_retrieval = bool(use_hybrid_retrieval)
        self._hybrid_dense_weight = float(hybrid_dense_weight)
        self._hybrid_bm25_weight = float(hybrid_bm25_weight)
        self._hybrid_pool_mult = max(1, int(hybrid_pool_mult))
        self._hybrid_full_corpus_pool = bool(hybrid_full_corpus_pool)
        self._unfused_rank_database_root = (
            str(Path(unfused_rank_database_root).resolve()) if unfused_rank_database_root else None
        )
        self._answer_mode = bool(answer_mode)
        self._fused_maps_cache: dict[str, Tuple[Dict[str, str], Dict[str, RetrievedMemory]]] = {}
        self._language = (language or "en").strip() or "en"

    def episode_storage_path(self, history_name: str) -> Optional[Path]:
        return self.persisted_data_root() / history_name

    def _get_database(self, history_name: str) -> LocalFaissDatabase:
        if history_name not in self._databases:
            self._databases[history_name] = LocalFaissDatabase(
                namespace=history_name,
                database_root=self.database_root,
            )
        return self._databases[history_name]

    def _get_unfused_database(self, history_name: str) -> LocalFaissDatabase:
        if not self._unfused_rank_database_root:
            raise RuntimeError("unfused rank database root not configured")
        if history_name not in self._unfused_databases:
            self._unfused_databases[history_name] = LocalFaissDatabase(
                namespace=history_name,
                database_root=self._unfused_rank_database_root,
            )
        return self._unfused_databases[history_name]

    def _get_fused_member_maps(
        self, db_f: LocalFaissDatabase, history_name: str
    ) -> Tuple[Dict[str, str], Dict[str, RetrievedMemory]]:
        if history_name not in self._fused_maps_cache:
            from memory.fusion.bundle_fusion import build_pre_fusion_member_to_fused_maps

            self._fused_maps_cache[history_name] = build_pre_fusion_member_to_fused_maps(db_f)
        return self._fused_maps_cache[history_name]

    def _embed_texts(self, inputs: Iterable[str]) -> np.ndarray:
        from utils.embed_utils import embed_texts

        return embed_texts(self.embed_client, inputs, self.embed_model_name)

    def store_session(
        self,
        history_name: str,
        session_idx: int,
        session: ChatSession,
    ) -> None:
        return

    def retrieve(
        self,
        history_name: str,
        query: str,
        current_time: str,
        top_k: int = 5,
    ) -> List[RetrievedMemory]:
        db_f = self._get_database(history_name)
        query_embedding = self._embed_texts([query])
        if query_embedding.size == 0:
            return []

        if self._unfused_rank_database_root:
            return self._retrieve_unfused_rank_fused_content(
                history_name=history_name,
                query=query,
                query_embedding=query_embedding[0],
                top_k=top_k,
            )

        if self._use_hybrid_retrieval:
            return db_f.search_hybrid(
                query,
                query_embedding[0],
                top_k,
                dense_weight=self._hybrid_dense_weight,
                bm25_weight=self._hybrid_bm25_weight,
                only_primary=False,
                answer_mode=self._answer_mode,
                pool_mult=self._hybrid_pool_mult,
                full_corpus_pool=self._hybrid_full_corpus_pool,
            )
        return db_f.search(
            query_embedding[0],
            top_k,
            only_primary=False,
            answer_mode=self._answer_mode,
        )

    def _retrieve_unfused_rank_fused_content(
        self,
        *,
        history_name: str,
        query: str,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> List[RetrievedMemory]:
        db_u = self._get_unfused_database(history_name)
        db_f = self._get_database(history_name)
        member_to_fused, fused_by_id = self._get_fused_member_maps(db_f, history_name)
        pm = self._hybrid_pool_mult
        if self._use_hybrid_retrieval:
            ranked = db_u.search_hybrid(
                query,
                query_embedding,
                top_k,
                dense_weight=self._hybrid_dense_weight,
                bm25_weight=self._hybrid_bm25_weight,
                only_primary=False,
                pool_mult=pm,
                return_full_ranked_pool=True,
                full_corpus_pool=self._hybrid_full_corpus_pool,
            )
        else:
            n_all = db_u.memory_row_count()
            if n_all == 0 or top_k <= 0:
                return []
            M = min(n_all, max(top_k * pm, 50))
            ranked = db_u.search(
                query_embedding,
                M,
                only_primary=False,
            )

        out: List[RetrievedMemory] = []
        seen_fused: set[str] = set()
        for hit in ranked:
            fid = member_to_fused.get(hit.memory_id)
            if fid is None:
                logger.debug(
                    "unfused hit id %s has no fused mapping (episode %s)",
                    hit.memory_id,
                    history_name,
                )
                continue
            if fid in seen_fused:
                continue
            seen_fused.add(fid)
            base = fused_by_id.get(fid)
            if base is None:
                logger.warning(
                    "fused id %s missing from fused_by_id map (episode %s)",
                    fid,
                    history_name,
                )
                continue
            out.append(
                RetrievedMemory(
                    memory_id=base.memory_id,
                    text=base.text,
                    source_index=base.source_index,
                    time=base.time,
                    score=float(hit.score),
                    metadata=dict(base.metadata or {}),
                )
            )
            if len(out) >= top_k:
                break
        return out
