import json
import logging
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from memory.base import RetrievedMemory

logger = logging.getLogger(__name__)


@dataclass
class _HistoryStore:
    index: Optional[faiss.Index]
    ids: List[str]
    texts: List[str]
    source_indices: List[str]  # 存储"原文索引"
    times: List[str]           # 存储"代表时间"
    metadatas: List[Dict[str, Any]]
    embeddings: List[np.ndarray]

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
        self._reset_store()

    def add(self, text: str, source_index: str, time: str, metadata: Dict[str, Any], embedding: Optional[np.ndarray] = None) -> str:
        """
        向数据库中添加一条记忆。
        如果传入了 embedding，则会进行向量索引存储。
        """
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

        self._persist()
        return memory_id

    def delete(self, memory_id: str) -> bool:
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
        if len(store.embeddings) > idx:
            store.embeddings.pop(idx)

        if not store.ids:
            self._clear_dataset()
        else:
            self._rebuild_index()
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
            
            # 补齐长度如果前面没有 embedding 却中途更新了
            while len(store.embeddings) <= idx:
                store.embeddings.append(np.zeros_like(normalized[0]))
                
            store.embeddings[idx] = normalized[0].copy()
            self._rebuild_index()

        if new_source_index is not None:
            store.source_indices[idx] = new_source_index
        if new_time is not None:
            store.times[idx] = new_time
        if metadata_updates:
            store.metadatas[idx].update(metadata_updates)

        self._persist()
        return True

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[RetrievedMemory]:
        """
        直接接收 query_embedding (1D 或 2D numpy array) 并召回最近似的 K 条结果。
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
        
        k = min(top_k, store.index.ntotal)
        scores, indices = store.index.search(normalized_query, k)

        results: List[RetrievedMemory] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(
                RetrievedMemory(
                    memory_id=store.ids[idx],
                    text=store.texts[idx],
                    source_index=store.source_indices[idx],
                    time=store.times[idx],
                    score=float(score),
                    metadata=store.metadatas[idx]
                )
            )
        return results

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
        Remove memories whose stripped text is identical to another entry.
        Keeps one per text: earliest by `time` (parse_chat_time), then smaller list index as tie-breaker.
        Returns the number of removed memories.
        """
        self._ensure_loaded()
        store = self._store
        n = len(store.ids)
        if n <= 1:
            return 0

        from utils.date_utils import parse_chat_time

        groups: Dict[str, List[int]] = {}
        for i in range(n):
            key = store.texts[i].strip()
            if not key:
                continue
            groups.setdefault(key, []).append(i)

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
            store.ids.pop(idx)
            store.texts.pop(idx)
            store.source_indices.pop(idx)
            store.times.pop(idx)
            store.metadatas.pop(idx)
            if len(store.embeddings) > idx:
                store.embeddings.pop(idx)

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
            embeddings=[]
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

        index_path = dataset_dir / "index.faiss"
        emb_path = dataset_dir / "embeddings.npy"
        if index_path.exists():
            store.index = faiss.read_index(str(index_path))
        if emb_path.exists():
            arr = np.load(emb_path, allow_pickle=False)
            store.embeddings = [row.astype(np.float32) for row in arr]
        if store.index is None and store.embeddings:
            self._rebuild_index()

        # 兼容补齐老数据
        total_len = len(store.ids)
        if len(store.source_indices) != total_len:
            store.source_indices = ["unknown"] * total_len
        if len(store.times) != total_len:
            store.times = [m.get("date", "unknown_time") if isinstance(m, dict) else "unknown_time" for m in store.metadatas]
            if len(store.times) != total_len:
                store.times = ["unknown_time"] * total_len

        self._store_loaded = True

    def _persist(self) -> None:
        store = self._store
        dataset_dir = self._dataset_dir()
        dataset_dir.mkdir(parents=True, exist_ok=True)

        if store.index is not None and store.embeddings:
            faiss.write_index(store.index, str(dataset_dir / "index.faiss"))
            np.save(dataset_dir / "embeddings.npy", np.vstack(store.embeddings).astype(np.float32))

        for name in ["ids", "texts", "source_indices", "times", "metadatas"]:
            path = dataset_dir / f"{name}.json"
            val = getattr(store, name)
            path.write_text(json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8")

    def _clear_dataset(self) -> None:
        dataset_dir = self._dataset_dir()
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        self._reset_store()
        self._store_loaded = True

    def clear_all(self) -> None:
        """Remove all data for this namespace (for resume/cleanup)."""
        self._clear_dataset()
