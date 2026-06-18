import json
import logging
import os
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from memory.base import RetrievedMemory

logger = logging.getLogger(__name__)


def _memory_entry_is_primary(metadata: Dict[str, Any]) -> bool:
    """结构主条目。``evidence`` 弱边行与 ``answer`` 答题融合行(C)都不算 primary。

    缺省(无 memory_role)视为 primary，向后兼容 baseline 数据。``answer`` 是
    relation_decision 就地融合产出的答题专用记忆，不参与灌库判断/弱边结构。
    """
    role = metadata.get("memory_role")
    return role != "evidence" and role != "answer"


def _memory_entry_is_searchable_primary(metadata: Dict[str, Any]) -> bool:
    """灌库判断检索可见的原子 primary（排除 stale；保留 answer_hidden——后续仍用它判断）。"""
    return _memory_entry_is_primary(metadata) and not bool(metadata.get("stale"))


def _memory_entry_is_answer_visible(metadata: Dict[str, Any]) -> bool:
    """答题检索可见行：融合记忆 C(role=answer) + 未被 C 覆盖的孤立原子 primary。

    排除：stale、evidence 弱边、以及已被某条 C 覆盖的原子(answer_hidden=True)。
    """
    if bool(metadata.get("stale")):
        return False
    role = metadata.get("memory_role")
    if role == "evidence":
        return False
    if role == "answer":
        return True
    # 原子 primary：仅当未被融合记忆覆盖时才在答题检索可见
    return not bool(metadata.get("answer_hidden"))



def _tokenize_bm25(text: str) -> List[str]:
    """Whitespace tokenization + lowercasing; for CJK-heavy corpora consider a dedicated tokenizer."""
    return text.lower().split()


@dataclass
class _HistoryStore:
    index: Optional[faiss.Index]
    ids: List[str]
    texts: List[str]
    source_indices: List[str]  # 存储"原文索引"
    times: List[str]           # 存储"代表时间"
    metadatas: List[Dict[str, Any]]
    embeddings: List[np.ndarray]
    # FAISS 第 i 行向量对应 memory_id；与 ids 列表顺序无关（允许存在无向量的记忆行）
    indexed_memory_ids: List[str]

class LocalFaissDatabase:
    """
    一个极其纯粹的底层存储器。
    满足五要素约束：id/text/source_index/time/metadata/embedding
    只负责存储传入的 embedding 进行向量检索，不再依赖具体的 OpenAI 客户端进行文本转向量。
    """
    def __init__(
        self,
        namespace: str,            # 通常是 method_name 和 history_name 的结合，用作存储子目录
        database_root: Optional[str] = None
    ) -> None:
        self.namespace = namespace
        self.base_dir = Path(database_root or "MemDB/LocalStore")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self._store_loaded = False
        self._bm25_revision: int = 0
        self._bm25_cache_revision: Optional[int] = None
        self._bm25_okapi: Any = None
        # Reentrant lock: serialises all mutations (add/delete/update/dedup/clear)
        # and protects _ensure_loaded + _persist from concurrent access.
        self._lock = threading.RLock()
        self._reset_store()

    def _bump_bm25_revision(self) -> None:
        self._bm25_revision += 1

    def _get_bm25_okapi(self):
        """Lazy BM25 over ``store.texts``; invalidated when store mutates or reloads."""
        from rank_bm25 import BM25Okapi

        self._ensure_loaded()
        store = self._store
        n = len(store.texts)
        if n == 0:
            return None
        if self._bm25_cache_revision == self._bm25_revision and self._bm25_okapi is not None:
            return self._bm25_okapi
        corpus = [_tokenize_bm25(t) for t in store.texts]
        self._bm25_okapi = BM25Okapi(corpus)
        self._bm25_cache_revision = self._bm25_revision
        return self._bm25_okapi

    def add(self, text: str, source_index: str, time: str, metadata: Dict[str, Any], embedding: Optional[np.ndarray] = None) -> str:
        """
        向数据库中添加一条记忆。
        如果传入了 embedding，则会进行向量索引存储。
        """
        with self._lock:
            self._ensure_loaded()
            store = self._store
            normalized: Optional[np.ndarray] = None

            if embedding is not None and embedding.size > 0:
                # 确保 embedding 是 2D 以符合 faiss 要求 (1, dim)
                if embedding.ndim == 1:
                    embedding = embedding.reshape(1, -1)

                if store.index is None:
                    self._initialize_history(embedding.shape[1])

                normalized = np.ascontiguousarray(embedding.astype(np.float32))
                faiss.normalize_L2(normalized)
                store.index.add(normalized)

            memory_id = str(uuid.uuid4())

            store.ids.append(memory_id)
            store.texts.append(text)
            store.source_indices.append(source_index)
            store.times.append(time)
            store.metadatas.append(metadata)

            if normalized is not None:
                store.embeddings.append(normalized[0].copy())
                store.indexed_memory_ids.append(memory_id)

            self._bump_bm25_revision()
            self._persist()
            return memory_id

    def list_primary_texts_ordered(self) -> List[str]:
        """Primary rows in insertion order (excludes ``memory_role=evidence``)."""
        self._ensure_loaded()
        store = self._store
        out: List[str] = []
        n = min(len(store.ids), len(store.texts), len(store.metadatas))
        for i in range(n):
            if _memory_entry_is_primary(store.metadatas[i]):
                out.append(store.texts[i])
        return out

    def invalidate_memory(self, memory_id: str) -> bool:
        """从索引中移除条目（与 delete 同语义，保留方法名供调用方）。"""
        return self.delete(memory_id)

    def delete(self, memory_id: str) -> bool:
        with self._lock:
            self._ensure_loaded()
            store = self._store
            try:
                idx = store.ids.index(memory_id)
            except ValueError:
                return False

            store.ids.pop(idx)
            store.texts.pop(idx)
            store.source_indices.pop(idx)
            store.times.pop(idx)
            store.metadatas.pop(idx)
            if memory_id in store.indexed_memory_ids:
                emb_idx = store.indexed_memory_ids.index(memory_id)
                store.indexed_memory_ids.pop(emb_idx)
                store.embeddings.pop(emb_idx)

            if not store.ids:
                self._clear_dataset()
            else:
                self._rebuild_index()
                self._bump_bm25_revision()
                self._persist()
            return True

    def update_memory(
        self,
        memory_id: str,
        new_text: Optional[str] = None,
        new_source_index: Optional[str] = None,
        new_time: Optional[str] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
        new_embedding: Optional[np.ndarray] = None,
    ) -> bool:
        """
        更新对应的记忆。如果更新了文本且需要更新向量，可以传入新的 new_embedding。
        """
        with self._lock:
            self._ensure_loaded()
            store = self._store
            try:
                idx = store.ids.index(memory_id)
            except ValueError:
                return False

            if new_text is not None:
                store.texts[idx] = new_text

            if new_embedding is not None and new_embedding.size > 0:
                if new_embedding.ndim == 1:
                    new_embedding = new_embedding.reshape(1, -1)

                normalized = np.ascontiguousarray(new_embedding.astype(np.float32))
                faiss.normalize_L2(normalized)
                row = normalized[0].copy()

                if memory_id in store.indexed_memory_ids:
                    emb_idx = store.indexed_memory_ids.index(memory_id)
                    store.embeddings[emb_idx] = row
                else:
                    store.embeddings.append(row)
                    store.indexed_memory_ids.append(memory_id)
                self._rebuild_index()

            if new_source_index is not None:
                store.source_indices[idx] = new_source_index
            if new_time is not None:
                store.times[idx] = new_time
            if metadata_updates:
                store.metadatas[idx].update(metadata_updates)

            self._bump_bm25_revision()
            self._persist()
            return True

    def _faiss_row_to_list_row(self, store: _HistoryStore, fidx: int) -> Optional[int]:
        """Map FAISS internal row index → parallel lists (ids/texts/...) index."""
        if fidx < 0 or fidx >= len(store.indexed_memory_ids):
            logger.warning(
                "FAISS row %s out of range for indexed_memory_ids (len=%s)",
                fidx,
                len(store.indexed_memory_ids),
            )
            return None
        memory_id = store.indexed_memory_ids[fidx]
        try:
            row = store.ids.index(memory_id)
        except ValueError:
            logger.warning("indexed_memory_ids[%s]=%r missing from ids", fidx, memory_id)
            return None
        n = min(
            len(store.ids),
            len(store.texts),
            len(store.source_indices),
            len(store.times),
            len(store.metadatas),
        )
        if row < 0 or row >= n:
            logger.warning(
                "parallel lists out of sync: row=%s n_parallel=%s (ids=%s texts=%s metadatas=%s)",
                row,
                n,
                len(store.ids),
                len(store.texts),
                len(store.metadatas),
            )
            return None
        return row

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        *,
        only_primary: bool = False,
        answer_mode: bool = False,
    ) -> List[RetrievedMemory]:
        """
        直接接收 query_embedding (1D 或 2D numpy array) 并召回最近似的 K 条结果。
        only_primary=True 时跳过 metadata['memory_role'] == 'evidence'（主检索路径）。
        answer_mode=True 时只返回答题可见行：融合记忆 C(role=answer) + 未被 C 覆盖的孤立原子；
        优先级高于 only_primary（relation_decision 答题路径用）。
        """
        if top_k <= 0 or query_embedding is None or query_embedding.size == 0:
            return []

        self._ensure_loaded()
        store = self._store
        if store.index is None or store.index.ntotal == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        normalized_query = np.ascontiguousarray(query_embedding.astype(np.float32))
        faiss.normalize_L2(normalized_query)

        # answer_mode 走与 only_primary 相同的「轮次扩召回 + 过滤」循环，故并入同一路径
        filter_restrictive = bool(only_primary or answer_mode)

        def _passes_filters(meta: Dict[str, Any]) -> bool:
            if answer_mode:
                return _memory_entry_is_answer_visible(meta)
            if bool(meta.get("stale")):
                return False
            if only_primary and not _memory_entry_is_searchable_primary(meta):
                return False
            return True

        total = store.index.ntotal
        if not filter_restrictive:
            k = min(top_k, total)
            scores, indices = store.index.search(normalized_query, k)
            results: List[RetrievedMemory] = []
            for score, fidx in zip(scores[0], indices[0]):
                if fidx == -1:
                    continue
                row = self._faiss_row_to_list_row(store, int(fidx))
                if row is None:
                    continue
                meta = store.metadatas[row]
                if not _passes_filters(meta):
                    continue
                results.append(
                    RetrievedMemory(
                        memory_id=store.ids[row],
                        text=store.texts[row],
                        source_index=store.source_indices[row],
                        time=store.times[row],
                        score=float(score),
                        metadata=store.metadatas[row],
                    )
                )
            return results

        fetch_k = min(total, top_k)
        results = []
        seen_rounds = 0
        max_rounds = 8
        while seen_rounds < max_rounds:
            seen_rounds += 1
            k_req = min(total, fetch_k)
            scores, indices = store.index.search(normalized_query, k_req)
            for score, fidx in zip(scores[0], indices[0]):
                if fidx == -1:
                    continue
                row = self._faiss_row_to_list_row(store, int(fidx))
                if row is None:
                    continue
                meta = store.metadatas[row]
                if not _passes_filters(meta):
                    continue
                results.append(
                    RetrievedMemory(
                        memory_id=store.ids[row],
                        text=store.texts[row],
                        source_index=store.source_indices[row],
                        time=store.times[row],
                        score=float(score),
                        metadata=meta,
                    )
                )
                if len(results) >= top_k:
                    return results
            if k_req >= total:
                return results
            fetch_k = min(total, max(fetch_k * 2, top_k * 4))
        return results

    def search_hybrid(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int,
        *,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
        only_primary: bool = False,
        answer_mode: bool = False,
        pool_mult: int = 4,
        return_full_ranked_pool: bool = False,
        full_corpus_pool: bool = False,
    ) -> List[RetrievedMemory]:
        """BM25 + dense (FAISS inner product) linear fusion with per-modality max normalization.

        If ``return_full_ranked_pool`` is True, return all candidates in the fusion pool sorted by
        combined score (not truncated to ``top_k``), for downstream mapping (e.g. unfused rank → fused rows).

        If ``full_corpus_pool`` is True, dense/BM25 each consider every indexed row when building the
        fusion pool (then truncate to ``top_k``), instead of capping at
        ``min(n, max(top_k * pool_mult, 50))``.

        ``answer_mode``：只保留答题可见行（融合记忆 C + 未被覆盖的孤立原子），过滤 evidence/被覆盖原子。
        """
        if only_primary:
            return self.search(query_embedding, top_k, only_primary=True)

        # answer_mode 下用「答题可见」谓词替代单纯的 stale 过滤
        def _row_dropped(meta: Dict[str, Any]) -> bool:
            if answer_mode:
                return not _memory_entry_is_answer_visible(meta)
            return bool(meta.get("stale"))

        w_d = max(0.0, float(dense_weight))
        w_b = max(0.0, float(bm25_weight))
        w_sum = w_d + w_b
        if w_sum <= 0:
            return self.search(query_embedding, top_k, only_primary=False, answer_mode=answer_mode)

        w_d /= w_sum
        w_b /= w_sum

        self._ensure_loaded()
        store = self._store
        n_all = len(store.ids)
        if n_all == 0 or top_k <= 0:
            return []

        pm = max(1, int(pool_mult))
        if full_corpus_pool:
            M = n_all
        else:
            M = min(n_all, max(top_k * pm, 50))

        dense_scores: Dict[str, float] = {}
        if (
            w_d > 0
            and query_embedding is not None
            and query_embedding.size > 0
            and store.index is not None
            and store.index.ntotal > 0
        ):
            qe = query_embedding
            if qe.ndim == 1:
                qe = qe.reshape(1, -1)
            normalized_query = np.ascontiguousarray(qe.astype(np.float32))
            faiss.normalize_L2(normalized_query)
            k_d = min(M, store.index.ntotal)
            scores, indices = store.index.search(normalized_query, k_d)
            for score, fidx in zip(scores[0], indices[0]):
                if fidx == -1:
                    continue
                row = self._faiss_row_to_list_row(store, int(fidx))
                if row is None:
                    continue
                if _row_dropped(store.metadatas[row]):
                    continue
                dense_scores[store.ids[row]] = float(score)

        bm25_top: Dict[str, float] = {}
        if w_b > 0:
            bm25 = self._get_bm25_okapi()
            if bm25 is not None:
                q_tokens = _tokenize_bm25(query_text)
                raw = np.asarray(bm25.get_scores(q_tokens), dtype=np.float64)
                take_m = min(M, len(raw))
                if take_m > 0:
                    if len(raw) <= take_m:
                        top_ix = np.argsort(-raw)[:take_m]
                    else:
                        part = np.argpartition(-raw, take_m - 1)[:take_m]
                        top_ix = part[np.argsort(-raw[part])]
                    for i in top_ix:
                        ii = int(i)
                        if _row_dropped(store.metadatas[ii]):
                            continue
                        bm25_top[store.ids[ii]] = float(raw[ii])

        max_d = max(dense_scores.values()) if dense_scores else 0.0
        max_b = max(bm25_top.values()) if bm25_top else 0.0

        pool_ids = set(dense_scores) | set(bm25_top)
        if not pool_ids:
            return []

        scored: List[tuple[float, float, float, str]] = []
        for mid in pool_ids:
            rd = dense_scores.get(mid, 0.0)
            rb = bm25_top.get(mid, 0.0)
            nd = (rd / max_d) if max_d > 0 else 0.0
            nb = (rb / max_b) if max_b > 0 else 0.0
            comb = w_d * nd + w_b * nb
            scored.append((comb, rb, rd, mid))

        # Tie-break combined scores with raw BM25 / dense so near-zero ties are deterministic.
        scored.sort(key=lambda t: (-t[0], -t[1], -t[2], t[3]))
        rank_cap = len(scored) if return_full_ranked_pool else top_k
        scored = scored[:rank_cap]

        out: List[RetrievedMemory] = []
        for comb, _rb, _rd, mid in scored:
            try:
                row = store.ids.index(mid)
            except ValueError:
                continue
            if _row_dropped(store.metadatas[row]):
                continue
            out.append(
                RetrievedMemory(
                    memory_id=mid,
                    text=store.texts[row],
                    source_index=store.source_indices[row],
                    time=store.times[row],
                    score=float(comb),
                    metadata=dict(store.metadatas[row]),
                )
            )
        return out

    def collect_evidence_descendants(self, root_primary_id: str) -> List[RetrievedMemory]:
        """
        从某条 primary 出发，按 parent_primary 链收集所有 evidence（BFS，支持 evidence 再挂子 evidence）。
        仅包含 memory_role == evidence 的条目；返回顺序为 BFS。metadata 中写入 evidence_depth（1 起）。
        """
        self._ensure_loaded()
        store = self._store
        if not store.ids:
            return []

        children: Dict[str, List[int]] = {}
        for i in range(len(store.ids)):
            meta = store.metadatas[i]
            if _memory_entry_is_primary(meta):
                continue
            pid = meta.get("parent_primary")
            if not pid or not isinstance(pid, str):
                continue
            children.setdefault(pid, []).append(i)

        out: List[RetrievedMemory] = []
        seen: set[str] = set()
        queue: List[tuple[str, int]] = [(root_primary_id, 0)]
        while queue:
            parent_id, parent_depth = queue.pop(0)
            for idx in children.get(parent_id, []):
                mid = store.ids[idx]
                if mid in seen:
                    continue
                seen.add(mid)
                depth = parent_depth + 1
                meta = dict(store.metadatas[idx])
                meta["evidence_depth"] = depth
                out.append(
                    RetrievedMemory(
                        memory_id=mid,
                        text=store.texts[idx],
                        source_index=store.source_indices[idx],
                        time=store.times[idx],
                        score=0.0,
                        metadata=meta,
                    )
                )
                queue.append((mid, depth))
        return out

    def memory_row_count(self) -> int:
        """Number of memory rows in this namespace (including rows without embeddings)."""
        self._ensure_loaded()
        return len(self._store.ids)

    def get_memory(self, memory_id: str) -> Optional[RetrievedMemory]:
        """按 memory_id 取单条记忆；不存在返回 None。"""
        self._ensure_loaded()
        store = self._store
        try:
            idx = store.ids.index(memory_id)
        except ValueError:
            return None
        return RetrievedMemory(
            memory_id=store.ids[idx],
            text=store.texts[idx],
            source_index=store.source_indices[idx],
            time=store.times[idx],
            score=0.0,
            metadata=store.metadatas[idx],
        )

    def list_all_memories(self, sort_by_time: bool = True, descending: bool = False) -> List[RetrievedMemory]:
        """
        返回所有保存的记忆，可选择按时间排序（如果 time 包含有效可比格式）。
        """
        self._ensure_loaded()
        store = self._store

        def _safe_parse_date(time_str: str) -> datetime:
            from utils.date_utils import parse_chat_time
            return parse_chat_time(time_str)

        entries = []
        for i in range(len(store.ids)):
            mem = RetrievedMemory(
                memory_id=store.ids[i],
                text=store.texts[i],
                source_index=store.source_indices[i],
                time=store.times[i],
                score=0.0,
                metadata=store.metadatas[i]
            )
            parsed_time = _safe_parse_date(store.times[i]) if sort_by_time else datetime.min
            entries.append((parsed_time, mem))

        if sort_by_time:
            entries.sort(key=lambda x: x[0], reverse=descending)

        return [mem for _, mem in entries]

    def deduplicate_identical_text(self) -> int:
        """
        Remove memories whose stripped text is identical to another entry **with the same
        ``memory_role``**.  Rows with different roles (e.g. ``primary`` vs ``answer``) serve
        distinct purposes — primary for relation decisions, answer for question-answering
        retrieval — so they are NOT deduplicated across roles.

        Within each (text, role) group, keeps the earliest by ``time``, then smallest list index.
        Returns the number of removed memories.
        """
        with self._lock:
            self._ensure_loaded()
            store = self._store
            n = len(store.ids)
            if n <= 1:
                return 0

            from utils.date_utils import parse_chat_time

            # 按 (text, memory_role) 分组，不同角色的相同文本互不去重
            groups: Dict[tuple, List[int]] = {}
            for i in range(n):
                text_key = store.texts[i].strip()
                if not text_key:
                    continue
                meta_i = store.metadatas[i]
                role = meta_i.get("memory_role", "primary") if isinstance(meta_i, dict) else "primary"
                groups.setdefault((text_key, role), []).append(i)

            remove_idx: set[int] = set()
            for indices in groups.values():
                if len(indices) < 2:
                    continue
                keeper = min(indices, key=lambda i: (parse_chat_time(store.times[i]), i))
                for i in indices:
                    if i != keeper:
                        remove_idx.add(i)

            if not remove_idx:
                return 0

            for idx in sorted(remove_idx, reverse=True):
                memory_id = store.ids[idx]
                store.ids.pop(idx)
                store.texts.pop(idx)
                store.source_indices.pop(idx)
                store.times.pop(idx)
                store.metadatas.pop(idx)
                if memory_id in store.indexed_memory_ids:
                    emb_idx = store.indexed_memory_ids.index(memory_id)
                    store.indexed_memory_ids.pop(emb_idx)
                    store.embeddings.pop(emb_idx)

            if not store.ids:
                self._clear_dataset()
            else:
                self._rebuild_index()
                self._persist()

            return len(remove_idx)

    def _initialize_history(self, dim: int) -> None:
        self._store.index = faiss.IndexFlatIP(dim)

    def _rebuild_index(self) -> None:
        store = self._store
        if not store.embeddings:
            store.index = None
            return
        embeddings = np.vstack(store.embeddings).astype(np.float32)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(np.ascontiguousarray(embeddings))
        store.index = index

    def _align_parallel_lists_to_ids(self) -> None:
        """Ensure ids/texts/source_indices/times/metadatas have the same length (on-disk JSON can diverge)."""
        store = self._store
        n = len(store.ids)
        if n == 0:
            store.texts = []
            store.source_indices = []
            store.times = []
            store.metadatas = []
            return
        for name, default in (
            ("texts", ""),
            ("source_indices", "unknown"),
            ("times", "unknown_time"),
            ("metadatas", None),
        ):
            lst: List[Any] = getattr(store, name)
            if len(lst) < n:
                while len(lst) < n:
                    lst.append({} if name == "metadatas" else default)
                setattr(store, name, lst)
            elif len(lst) > n:
                setattr(store, name, lst[:n])

    def _repair_indexed_memory_ids_after_load(self) -> None:
        """Align indexed_memory_ids with embeddings; migrate legacy layouts where FAISS row == ids index."""
        store = self._store
        ne = len(store.embeddings)
        ni = len(store.indexed_memory_ids)
        nid = len(store.ids)

        if ne == 0:
            store.indexed_memory_ids = []
            return

        if ni == 0:
            if ne == nid:
                store.indexed_memory_ids = list(store.ids)
            elif ne <= nid:
                store.indexed_memory_ids = list(store.ids[:ne])
            else:
                logger.warning(
                    "More embeddings (%s) than ids (%s); truncating embeddings to ids length",
                    ne,
                    nid,
                )
                store.embeddings = store.embeddings[:nid]
                store.indexed_memory_ids = list(store.ids)
                ne = len(store.embeddings)

        if ne != len(store.indexed_memory_ids):
            m = min(ne, len(store.indexed_memory_ids))
            store.embeddings = store.embeddings[:m]
            store.indexed_memory_ids = store.indexed_memory_ids[:m]
            logger.warning(
                "Truncated embeddings/indexed_memory_ids to %s to recover consistent store",
                m,
            )
            ne = len(store.embeddings)

        n_idx = store.index.ntotal if store.index is not None else 0
        if store.index is not None and n_idx != ne:
            self._rebuild_index()

    def _dataset_dir(self) -> Path:
        return self.base_dir / self.namespace

    def _reset_store(self) -> None:
        self._store = _HistoryStore(
            index=None,
            ids=[],
            texts=[],
            source_indices=[],
            times=[],
            metadatas=[],
            embeddings=[],
            indexed_memory_ids=[],
        )

    def _ensure_loaded(self) -> None:
        if self._store_loaded:
            return

        self._reset_store()
        store = self._store
        dataset_dir = self._dataset_dir()
        if not dataset_dir.exists():
            self._store_loaded = True
            return

        # Core loads
        for name in ["ids", "texts", "source_indices", "times", "metadatas"]:
            path = dataset_dir / f"{name}.json"
            if path.exists():
                try:
                    setattr(store, name, json.loads(path.read_text(encoding="utf-8")))
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning("Failed to load %s from %s: %s", name, path, e)
                    setattr(store, name, [])

        im_path = dataset_dir / "indexed_memory_ids.json"
        if im_path.exists():
            try:
                raw_im = json.loads(im_path.read_text(encoding="utf-8"))
                store.indexed_memory_ids = raw_im if isinstance(raw_im, list) else []
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load indexed_memory_ids from %s: %s", im_path, e)
                store.indexed_memory_ids = []
        else:
            store.indexed_memory_ids = []

        index_path = dataset_dir / "index.faiss"
        emb_path = dataset_dir / "embeddings.npy"
        if index_path.exists():
            try:
                store.index = faiss.read_index(str(index_path))
            except Exception as e:
                logger.warning("Failed to load FAISS index from %s: %s", index_path, e)
                store.index = None
        if emb_path.exists():
            try:
                arr = np.load(emb_path, allow_pickle=False)
                store.embeddings = [row.astype(np.float32) for row in arr]
            except (EOFError, ValueError, OSError) as e:
                logger.warning("Failed to load embeddings from %s: %s – starting with empty embeddings", emb_path, e)
                store.embeddings = []
        if store.index is None and store.embeddings:
            self._rebuild_index()

        self._align_parallel_lists_to_ids()

        self._repair_indexed_memory_ids_after_load()

        # 兼容补齐老数据（times 可从 metadatas 推断）
        total_len = len(store.ids)
        if len(store.source_indices) != total_len:
            store.source_indices = ["unknown"] * total_len
        if len(store.times) != total_len:
            store.times = [m.get("date", "unknown_time") if isinstance(m, dict) else "unknown_time" for m in store.metadatas]
            if len(store.times) != total_len:
                store.times = ["unknown_time"] * total_len

        self._store_loaded = True

    @staticmethod
    def _atomic_write_bytes(target: Path, data: bytes) -> None:
        """Write *data* to *target* via temp-file + rename to avoid partial / corrupt files."""
        fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
        try:
            os.write(fd, data)
            os.fsync(fd)
            os.close(fd)
            os.replace(tmp, target)
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @staticmethod
    def _atomic_write_text(target: Path, text: str) -> None:
        fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
        try:
            os.write(fd, text.encode("utf-8"))
            os.fsync(fd)
            os.close(fd)
            os.replace(tmp, target)
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _persist(self) -> None:
        store = self._store
        dataset_dir = self._dataset_dir()
        dataset_dir.mkdir(parents=True, exist_ok=True)

        if store.index is not None and store.embeddings:
            # FAISS index → atomic write via temp file
            fd, tmp = tempfile.mkstemp(dir=dataset_dir, suffix=".tmp")
            os.close(fd)
            try:
                faiss.write_index(store.index, tmp)
                os.replace(tmp, dataset_dir / "index.faiss")
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
            import io
            buf = io.BytesIO()
            np.save(buf, np.vstack(store.embeddings).astype(np.float32))
            self._atomic_write_bytes(dataset_dir / "embeddings.npy", buf.getvalue())

        for name in ["ids", "texts", "source_indices", "times", "metadatas"]:
            path = dataset_dir / f"{name}.json"
            val = getattr(store, name)
            self._atomic_write_text(path, json.dumps(val, ensure_ascii=False, indent=2))

        self._atomic_write_text(
            dataset_dir / "indexed_memory_ids.json",
            json.dumps(store.indexed_memory_ids, ensure_ascii=False, indent=2),
        )

    def _clear_dataset(self) -> None:
        dataset_dir = self._dataset_dir()
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        self._reset_store()
        self._bump_bm25_revision()
        self._store_loaded = True

    def clear_all(self) -> None:
        """Remove all data for this namespace (for resume/cleanup)."""
        with self._lock:
            self._clear_dataset()
