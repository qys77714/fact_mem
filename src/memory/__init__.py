from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json
import shutil
import uuid

import faiss
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime


@dataclass
class RetrievedMemory:
    text: str
    metadata: Dict[str, Any]
    score: float


@dataclass
class _HistoryStore:
    index: Optional[faiss.Index]
    texts: List[str]
    metadatas: List[Dict[str, Any]]
    embeddings: List[np.ndarray]
    ids: List[str]


class MemoryDatabase:
    def __init__(
        self,
        embed_model_name: str,
        history_name: str,
        method_name: str = "default",
        embed_client: OpenAI = None,
        database_root: Optional[str] = None,
    ) -> None:
        self.embed_model_name = embed_model_name
        self.history_name = history_name
        self.method_name = method_name
        self.embed_client = embed_client
        self.base_dir = Path(database_root or "MemDB/longMemEval")
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self._reset_store()
        self._store_loaded = False
        self._requires_embeddings = self.method_name != "full_context"

        self._texts: Dict[str, List[str]] = {}
        self._metadatas: Dict[str, List[Dict[str, Any]]] = {}
        self._embeddings: Dict[str, List[np.ndarray]] = {}
        self._ids: Dict[str, List[str]] = {}

    def add(self, text: str, metadata: Dict[str, Any]) -> str:
        self._ensure_loaded()

        store = self._store
        normalized: Optional[np.ndarray] = None

        if self._requires_embeddings:
            embedding = self._embed_texts([text])
            if embedding.size == 0:
                raise ValueError("failed to compute embedding for input text")

            if store.index is None:
                self._initialize_history(embedding.shape[1])

            normalized = np.ascontiguousarray(embedding.astype(np.float32))
            faiss.normalize_L2(normalized)
            store.index.add(normalized)

        memory_id = str(uuid.uuid4())
        stored_metadata = dict(metadata)
        stored_metadata["memory_id"] = memory_id

        store.texts.append(text)
        store.metadatas.append(stored_metadata)
        if self._requires_embeddings and normalized is not None:
            store.embeddings.append(normalized[0].copy())
        store.ids.append(memory_id)

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
        store.metadatas.pop(idx)
        if self._requires_embeddings and len(store.embeddings) > idx:
            store.embeddings.pop(idx)

        if not store.ids:
            self._clear_dataset()
        else:
            if self._requires_embeddings:
                self._rebuild_index()
            self._persist()
        return True

    def update_memory(
        self,
        memory_id: str,
        new_text: str,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not new_text:
            raise ValueError("new_text must not be empty when updating memory.")

        self._ensure_loaded()
        store = self._store
        try:
            idx = store.ids.index(memory_id)
        except ValueError:
            return False

        store.texts[idx] = new_text

        if metadata_updates:
            sanitized_updates = dict(metadata_updates)
            sanitized_updates.pop("memory_id", None)
            store.metadatas[idx].update(sanitized_updates)
        if "memory_id" not in store.metadatas[idx]:
            store.metadatas[idx]["memory_id"] = memory_id

        if self._requires_embeddings:
            embedding = self._embed_texts([new_text])
            if embedding.size == 0:
                raise ValueError("failed to compute embedding for updated memory text")

            normalized = np.ascontiguousarray(embedding.astype(np.float32))
            faiss.normalize_L2(normalized)
            if len(store.embeddings) <= idx:
                store.embeddings.append(normalized[0].copy())
            else:
                store.embeddings[idx] = normalized[0].copy()
            self._rebuild_index()

        self._persist()
        return True

    def search(self, query: str, top_k: int) -> List[RetrievedMemory]:
        if not query or top_k <= 0:
            return []

        self._ensure_loaded()
        store = self._store
        if store.index is None or store.index.ntotal == 0:
            return []

        query_embedding = self._embed_texts([query])
        if query_embedding.size == 0:
            return []

        faiss.normalize_L2(query_embedding)
        k = min(top_k, store.index.ntotal)
        scores, indices = store.index.search(query_embedding, k)

        results: List[RetrievedMemory] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(
                RetrievedMemory(
                    text=store.texts[idx],
                    metadata=store.metadatas[idx],
                    score=float(score),
                )
            )
        return results

    def list_all_memories(self, descending: bool = False) -> List[RetrievedMemory]:
        """
        Return all memories sorted by metadata['date'] ascending or descending.
        """
        self._ensure_loaded()
        store = self._store

        def _parse_date(metadata: Dict[str, Any]) -> datetime:
            value = metadata.get("date")
            if not value:
                return datetime.min
            try:
                return datetime.strptime(value, "%Y/%m/%d (%a) %H:%M")
            except ValueError:
                return datetime.min

        entries = [
            (_parse_date(metadata), text, metadata)
            for text, metadata in zip(store.texts, store.metadatas)
        ]
        entries.sort(key=lambda item: item[0], reverse=descending)

        return [
            RetrievedMemory(text=text, metadata=metadata, score=0.0)
            for _, text, metadata in entries
        ]

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

    def _embed_texts(self, inputs: Iterable[str]) -> np.ndarray:
        batch_inputs = list(inputs)
        if not batch_inputs:
            return np.empty((0, 0), dtype=np.float32)

        response = self.embed_client.embeddings.create(
            input=batch_inputs,
            model=self.embed_model_name,
        )
        indexed_embeddings = {
            item.index: np.asarray(item.embedding, dtype=np.float32)
            for item in response.data
        }
        ordered_embeddings = [
            indexed_embeddings[i] for i in range(len(batch_inputs))
        ]
        return np.vstack(ordered_embeddings).astype(np.float32)

    def _dataset_dir(self) -> Path:
        return self.base_dir / self.history_name

    def _reset_store(self) -> None:
        self._store = _HistoryStore(
            index=None,
            texts=[],
            metadatas=[],
            embeddings=[],
            ids=[],
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

        index_path = dataset_dir / "index.faiss"
        texts_path = dataset_dir / "texts.json"
        metadata_path = dataset_dir / "metadatas.json"
        embeddings_path = dataset_dir / "embeddings.npy"
        ids_path = dataset_dir / "ids.json"

        if self._requires_embeddings and index_path.exists():
            store.index = faiss.read_index(str(index_path))

        if texts_path.exists():
            store.texts = json.loads(texts_path.read_text(encoding="utf-8"))

        if metadata_path.exists():
            store.metadatas = json.loads(metadata_path.read_text(encoding="utf-8"))

        if ids_path.exists():
            store.ids = json.loads(ids_path.read_text(encoding="utf-8"))

        if self._requires_embeddings and embeddings_path.exists():
            embeddings_array = np.load(embeddings_path, allow_pickle=False)
            store.embeddings = [row.astype(np.float32) for row in embeddings_array]

        if self._requires_embeddings and store.index is None and store.embeddings:
            self._rebuild_index()

        self._store_loaded = True

    def _persist(self) -> None:
        store = self._store
        dataset_dir = self._dataset_dir()
        dataset_dir.mkdir(parents=True, exist_ok=True)

        index_path = dataset_dir / "index.faiss"
        embeddings_path = dataset_dir / "embeddings.npy"

        if self._requires_embeddings and store.index is not None and store.embeddings:
            faiss.write_index(store.index, str(index_path))
            embeddings = np.vstack(store.embeddings).astype(np.float32)
            np.save(embeddings_path, embeddings)
        else:
            for path in (index_path, embeddings_path):
                if path.exists():
                    path.unlink()

        texts_path = dataset_dir / "texts.json"
        texts_path.write_text(
            json.dumps(store.texts, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )

        metadata_path = dataset_dir / "metadatas.json"
        metadata_path.write_text(
            json.dumps(store.metadatas, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )

        ids_path = dataset_dir / "ids.json"
        ids_path.write_text(
            json.dumps(store.ids, ensure_ascii=False),
            encoding="utf-8",
        )

    def _clear_dataset(self) -> None:
        dataset_dir = self._dataset_dir()
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        self._reset_store()
        self._store_loaded = True


class BaseMemoryMethod:
    def __init__(
        self,
        embed_model_name: str,
        embed_client: OpenAI = None,
        llm_client = None,
        storage_namespace: str = "default",
        database_root: Optional[str] = None,
    ) -> None:
        self.embed_model_name = embed_model_name
        self.storage_namespace = storage_namespace
        self.embed_client = embed_client
        self.llm_client = llm_client
        self.database_root = database_root
        self._databases: Dict[str, MemoryDatabase] = {}

    def _get_database(self, history_name: str) -> MemoryDatabase:
        database = self._databases.get(history_name)
        if database is None:
            database = MemoryDatabase(
                embed_model_name=self.embed_model_name,
                history_name=history_name,
                method_name=self.storage_namespace,
                embed_client=self.embed_client,
                database_root=self.database_root,
            )
            self._databases[history_name] = database
        return database

# 下面是三个MemoryMethod: 
class FullContextMemoryMethod(BaseMemoryMethod):
    def __init__(
        self,
        embed_model_name: str,
        embed_client: OpenAI = None,
        llm_client = None,
        database_root: Optional[str] = None
    ) -> None:
        super().__init__(
            embed_model_name,
            storage_namespace="full_context",
            embed_client=embed_client,
            llm_client=None,
            database_root=database_root,
        )

    def store_history(
        self,
        history_name: str,
        chat_history: List[List[Dict[str, str]]],
        chat_dates: List,
        *args, **kwargs
    ) -> None:
        database = self._get_database(history_name)
        dataset_dir = database._dataset_dir()
        if dataset_dir.exists() and any(dataset_dir.iterdir()):
            print(f"[FULL CONTEXT] 跳过历史 '{history_name}'，数据已存在。")
            return
        
        for session_idx, session in tqdm(enumerate(chat_history), desc="Storing Session"):
            mem_content = "\n\n".join(
                f"**{turn['speaker']}**: {turn['content']}" for turn in session
            )
            if not mem_content:
                continue
            metadata = {
                "session": session_idx,
                "date": chat_dates[session_idx],
                "turns": len(session),
                "method": "rag",
                "granularity": "session",
            }

            mem_text = (
                f"- Memory Date: {metadata['date']}\n"
                f"- Memory Content: \n"
                f"<MemoryContent>\n"
                f"{mem_content.strip()}\n"
                f"</MemoryContent>"
            )

            database.add(mem_text, metadata)
              
    def retrieve(
        self,
        history_name: str,
        *args, **kwargs
    ) -> List[RetrievedMemory]:
        database = self._get_database(history_name)
        return database.list_all_memories(descending=True)

class RagMemoryMethod(BaseMemoryMethod):
    def __init__(
        self,
        embed_model_name: str,
        embed_client: OpenAI = None,
        llm_client = None,
        database_root: Optional[str] = None,
    ) -> None:
        super().__init__(
            embed_model_name,
            storage_namespace="rag",
            embed_client=embed_client,
            llm_client=None,
            database_root=database_root,
        )

    def store_history(
        self,
        history_name: str,
        chat_history: List[List[Dict[str, str]]],
        chat_dates: List,
        granularity: str = "session",
    ) -> None:
        database = self._get_database(history_name)
        dataset_dir = database._dataset_dir()
        if dataset_dir.exists() and any(dataset_dir.iterdir()):
            print(f"[RAG] 跳过历史 '{history_name}'，数据已存在。")
            return

        if granularity not in {"session", "turn"}:
            raise ValueError("granularity must be 'session' or 'turn'")

        if granularity == "session":
            for session_idx, session in tqdm(enumerate(chat_history), desc="Storing Session"):
                mem_content = "\n\n".join(
                    f"**{turn['speaker']}**: {turn['content']}" for turn in session
                )
                if not mem_content:
                    continue
                metadata = {
                    "session": session_idx,
                    "date": chat_dates[session_idx],
                    "turns": len(session),
                    "method": "rag",
                    "granularity": "session",
                }

                mem_text = (
                    f"- Memory Date: {metadata['date']}\n"
                    f"- Memory Content: \n"
                    f"<MemoryContent>\n"
                    f"{mem_content.strip()}\n"
                    f"</MemoryContent>"
                )
                database.add(mem_text, metadata)
        else:
            for session_idx, session in tqdm(enumerate(chat_history), total=len(chat_history), desc="Storing Session"):
                for turn_idx, turn in tqdm(enumerate(session), desc="Storing Turn", leave=False):
                    mem_content = f"**{turn['speaker']}**: {turn['content']}"
                    if not mem_content:
                        continue
                    metadata = {
                        "session": session_idx,
                        "date": chat_dates[session_idx],
                        "turn": turn_idx,
                        "speaker": turn.get("speaker"),
                        "method": "rag",
                        "granularity": "turn",
                    }

                    mem_text = (
                        f"- Memory Date: {metadata['date']}\n"
                        f"- Memory Content: \n"
                        f"<MemoryContent>\n"
                        f"{mem_content.strip()}\n"
                        f"</MemoryContent>"
                    )
                    database.add(mem_text, metadata)

    def retrieve(
        self,
        history_name: str,
        question_text: str,
        top_k: int,
    ) -> List[RetrievedMemory]:
        database = self._get_database(history_name)
        return database.search(question_text, top_k)
    
class OnlyQueryMemoryMethod(BaseMemoryMethod):
    def __init__(
        self,
        embed_model_name: str,
        embed_client: OpenAI = None,
        llm_client = None,
        database_root: Optional[str] = None,
    ) -> None:
        super().__init__(
            embed_model_name,
            storage_namespace="only_query",
            embed_client=embed_client,
            llm_client=None,
            database_root=database_root,
        )

    def store_history(
        self,
        *args, **kwargs
    ) -> None:
        pass

    def retrieve(
        self,
        *args, **kwargs
    ) -> List[RetrievedMemory]:
        return []