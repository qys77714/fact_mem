"""ZepMemorySystem: graphiti-backed memory management over pre-extracted candidates.

Facts from the candidate JSON (already extracted by gemma4-26B) are fed into
graphiti's entity-deduplication and temporal-validity pipeline (via a Kuzu
embedded graph DB), and the resulting currently-valid EntityEdge facts are
exported to LocalFaissDatabase so that the unchanged lme_prebuilt
generate/judge pipeline can evaluate them alongside mem0 and relation_decision.

This is a controlled-variable experiment: the **input** is always the same
pre-extracted candidates; only the **management algorithm** differs.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from memory.base import BaseMemorySystem, RetrievedMemory
from memory.storage.local_faiss import LocalFaissDatabase
from memory.tracing import MemoryTraceLogger

from .adapters import _NoCrossEncoder, _SyncEmbedderAdapter, _SyncLLMAdapter

logger = logging.getLogger(__name__)


def _remove_path(path: Path) -> None:
    """Remove a file or directory tree (Kuzu uses a single db file, not a folder)."""
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


async def _close_kuzu_driver(driver: Any) -> None:
    """Drain in-flight kuzu queries, then release all Python references.

    Two bugs must be avoided simultaneously:

    1. Original segfault — kuzu.AsyncConnection's internal ThreadPoolExecutor
       keeps worker threads alive after process_chunks() returns.  When
       asyncio later shuts down its own default executor those threads are
       reaped while holding live kuzu C++ handles; the C++ destructors then
       run outside any valid context → segfault.  Fix: drain the executor
       with shutdown(wait=True) before returning.

    2. double-free (kuzu 0.11.x bug) — kuzu's Python Connection.close() and
       Database.close() each call the underlying C++ close() method AND then
       set the pybind11 reference to None.  Setting the reference to None
       triggers pybind11's tp_dealloc which calls the C++ *destructor* on the
       same (already close()'d) object → double-free / heap corruption.
       Fix: never call close() explicitly.  Just drain the executor so no C++
       code is running, then release Python references.  Python's refcounting
       will call the C++ destructors exactly once in the correct order
       (Connection objects first, then Database).
    """
    try:
        await driver.close()  # KuzuDriver.close() is a no-op
    finally:
        kuzu_conn = getattr(driver, "client", None)
        if kuzu_conn is not None:
            # Step 1: drain the internal ThreadPoolExecutor so no kuzu C++
            # callbacks are pending when we release the objects below.
            executor = getattr(kuzu_conn, "executor", None)
            if executor is not None:
                try:
                    await asyncio.to_thread(executor.shutdown, True)
                except Exception as _exc:
                    logger.warning("Failed to shutdown kuzu executor: %s", _exc)

            # Step 2: release kuzu.Connection Python objects WITHOUT calling
            # conn.close().  Clearing the list drops refcounts to 0; pybind11
            # tp_dealloc calls the C++ Connection destructor exactly once.
            # (Calling conn.close() first would call C++ close() + destructor
            # → double-free.)
            try:
                kuzu_conn.connections = []
            except Exception:
                pass

        # Step 3: release all driver-level references.  Do NOT call
        # kuzu_db.close() — same double-free reason as above.
        # Python refcounting frees AsyncConnection (+ its Database ref) first,
        # then the Database itself, guaranteeing correct C++ destructor order.
        try:
            driver.client = None
        except Exception:
            pass
        try:
            driver.db = None
        except Exception:
            pass


def _shutdown_ephemeral_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Drain/cancel pending tasks before closing a one-shot event loop."""
    try:
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(loop.shutdown_default_executor())
    finally:
        loop.close()

# Kuzu 的默认 max_db_size 是 8 TiB。在多线程并发灌库时（每个 episode 一个 KuzuDriver），
# 每个实例都会 mmap 8 TiB 虚拟地址空间，10 个并发 = 80 TiB，超出内核限制导致
# "Mmap for size 8796093022208 failed"。
# 对于 episode 级别的临时小型 Kuzu DB（几十个节点/边），1 GiB 绰绰有余。
# 在模块导入时一次性打补丁，对整个进程生效，无线程安全问题。
_KUZU_MAX_DB_SIZE = 1 << 30  # 1 GiB
try:
    import kuzu as _kuzu_mod

    _kuzu_orig_db_init = _kuzu_mod.Database.__init__

    def _kuzu_bounded_db_init(self, database_path=None, **kwargs):
        kwargs.setdefault("max_db_size", _KUZU_MAX_DB_SIZE)
        _kuzu_orig_db_init(self, database_path, **kwargs)

    _kuzu_mod.Database.__init__ = _kuzu_bounded_db_init
    del _kuzu_mod, _kuzu_bounded_db_init
except Exception as _e:
    logger.warning("Failed to patch kuzu.Database max_db_size: %s", _e)


def _parse_session_date(date_str: str) -> Optional[datetime]:
    """Return a timezone-aware datetime from common date string formats, or None."""
    if not date_str:
        return None
    for fmt in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


class ZepMemorySystem(BaseMemorySystem):
    """graphiti-backed memory system that consumes pre-extracted candidate facts.

    Memory management is delegated to graphiti's entity-deduplication and
    temporal-edge-validity pipeline (backed by a per-episode Kuzu embedded
    graph).  After ingest the currently-valid EntityEdge.fact strings are
    exported to LocalFaissDatabase so that the existing generate/evaluate
    pipeline can be used without modification.

    The constructor signature mirrors ``Mem0MemorySystem`` so that
    ``ingest_candidates.py`` can instantiate both in exactly the same way.
    """

    def __init__(
        self,
        embed_model_name: str,
        llm_client=None,
        embed_client=None,
        database_root: Optional[str] = None,
        language: str = "en",
        granularity: Union[str, int] = "all",
        trace_log_dir: Optional[str] = None,
        dialogue_format: str = "user_assistant",
        manager_max_new_tokens: int = 2048,
        # Accepted for API compatibility with ingest_candidates; not used by zep
        extract_concurrency: int = 8,
        relation_concurrency: int = 8,
        relation_max_new_tokens: int = 256,
    ) -> None:
        super().__init__(
            embed_client=embed_client,
            embed_model_name=embed_model_name,
            llm_client=llm_client,
            database_root=database_root,
        )
        if llm_client is None:
            raise ValueError("llm_client must be provided for ZepMemorySystem.")
        if embed_client is None:
            raise ValueError("embed_client must be provided for ZepMemorySystem.")

        self.language = language
        self._manager_max_new_tokens = max(1, int(manager_max_new_tokens))
        self._databases: Dict[str, LocalFaissDatabase] = {}

        self.trace = MemoryTraceLogger(
            method="zep",
            log_dir=trace_log_dir or "logs/memory_trace",
            use_experiment_naming=trace_log_dir is not None,
        )

        # Build adapters once; they wrap the same sync clients as mem0/relation_decision
        self._llm_adapter = _SyncLLMAdapter(
            sync_llm_client=llm_client,
            model_name=llm_client.model_name,
            max_tokens=self._manager_max_new_tokens,
        )
        self._embed_adapter = _SyncEmbedderAdapter(
            embed_client=embed_client,
            embed_model_name=embed_model_name,
        )

    # ------------------------------------------------------------------
    # Internal helpers (same pattern as Mem0MemorySystem)
    # ------------------------------------------------------------------

    def _get_database(self, history_name: str) -> LocalFaissDatabase:
        if history_name not in self._databases:
            self._databases[history_name] = LocalFaissDatabase(
                namespace=history_name,
                database_root=self.database_root,
            )
        return self._databases[history_name]

    def _embed_texts(self, inputs: List[str]) -> np.ndarray:
        from utils.embed_utils import embed_texts

        return embed_texts(self.embed_client, inputs, self.embed_model_name)

    def _get_kuzu_path(self, history_name: str) -> Path:
        return Path(self.database_root or "MemDB/LocalStore") / history_name / "kuzu_db"

    def episode_storage_path(self, history_name: str) -> Optional[Path]:
        return self.persisted_data_root() / history_name

    def clear(self, history_name: str) -> None:
        if history_name in self._databases:
            self._databases[history_name].clear_all()
            del self._databases[history_name]
        if self.database_root:
            ns_dir = Path(self.database_root) / history_name
            if ns_dir.exists():
                shutil.rmtree(ns_dir)

    def store_session(self, history_name: str, session_idx: int, session) -> None:
        raise NotImplementedError(
            "ZepMemorySystem does not support online store_session(). "
            "Use process_chunks() via the candidate ingest pipeline."
        )

    # ------------------------------------------------------------------
    # Core ingest
    # ------------------------------------------------------------------

    def process_chunks(
        self,
        history_name: str,
        chunks: List[Dict[str, Any]],
        *,
        incremental: bool = False,
    ) -> None:
        """Synchronous entry point; drives the async graphiti pipeline.

        Uses a manually managed event loop so that cleanup order is explicit:
        we shut down async generators first, then drain the default executor
        (which backs asyncio.to_thread LLM/embed calls) before closing the
        loop.  Draining the executor is critical — without it the to_thread
        worker threads outlive the loop and may hold live Kuzu handles; when
        those threads are finally reaped the native destructor runs outside any
        valid loop context and causes a segfault.
        """
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                self._async_process_chunks(history_name, chunks, incremental=incremental)
            )
        finally:
            _shutdown_ephemeral_loop(loop)

    async def _async_process_chunks(
        self,
        history_name: str,
        chunks: List[Dict[str, Any]],
        *,
        incremental: bool = False,
    ) -> None:
        from graphiti_core import Graphiti
        from graphiti_core.driver.kuzu_driver import KuzuDriver
        from graphiti_core.nodes import EpisodeType

        kuzu_path = self._get_kuzu_path(history_name)
        # clear_all() in ingest_candidates already removes the history_name dir;
        # we still guard here in case process_chunks is called standalone.
        # Phase-3 incremental ingest copies an existing kuzu db file — keep it.
        if not incremental and kuzu_path.exists():
            _remove_path(kuzu_path)
        kuzu_path.parent.mkdir(parents=True, exist_ok=True)

        database = self._get_database(history_name)

        driver = KuzuDriver(db=str(kuzu_path))
        graphiti_client = Graphiti(
            graph_driver=driver,
            llm_client=self._llm_adapter,
            embedder=self._embed_adapter,
            cross_encoder=_NoCrossEncoder(),
        )

        try:
            # Copied kuzu db from *_before already has schema + FTS indexes (Phase 1).
            if not incremental:
                await graphiti_client.build_indices_and_constraints()

            # Stable entity aliases for this episode: candidate facts often use
            # generic pronouns ("the user", "the assistant") as subjects.  Graphiti
            # needs a concrete, episode-scoped name so it can form entity nodes and
            # extract subject→object edges.  We replace the pronouns in-place before
            # feeding the text to graphiti; the original facts stored in FAISS are
            # NOT modified (they come from the raw candidate_memories).
            user_alias = f"User_{history_name}"
            asst_alias = f"Asst_{history_name}"
            _PRONOUN_PAIRS = [
                ("the user", user_alias),
                ("the assistant", asst_alias),
                ("The user", user_alias),
                ("The assistant", asst_alias),
            ]

            def _normalize_fact(fact: str) -> str:
                for src, dst in _PRONOUN_PAIRS:
                    fact = fact.replace(src, dst)
                return fact

            sorted_chunks = sorted(chunks, key=lambda c: int(c.get("chunk_index", 0)))
            for chunk in sorted_chunks:
                facts = [
                    str(m).strip()
                    for m in (chunk.get("candidate_memories") or [])
                    if str(m).strip()
                ]
                if not facts:
                    continue

                session_date = str(chunk.get("session_date") or "").strip()
                reference_time = _parse_session_date(session_date) or datetime.now(
                    timezone.utc
                )
                chunk_idx = chunk.get("chunk_index", 0)
                normalized_facts = [_normalize_fact(f) for f in facts]
                episode_body = "\n".join(f"- {f}" for f in normalized_facts)

                _add_episode_max_retries = 3
                for _attempt in range(_add_episode_max_retries):
                    try:
                        await graphiti_client.add_episode(
                            name=f"{history_name}_chunk_{chunk_idx}",
                            episode_body=episode_body,
                            source_description="extracted memory facts",
                            reference_time=reference_time,
                            source=EpisodeType.text,
                            group_id=history_name,
                        )
                        break
                    except Exception as _exc:
                        # graphiti internally validates the LLM response with pydantic;
                        # if the LLM returns an incomplete edge (missing relation_type /
                        # fact), a ValidationError is raised.  Retry up to
                        # _add_episode_max_retries times before giving up on this chunk.
                        if _attempt < _add_episode_max_retries - 1:
                            logger.warning(
                                "add_episode failed for %s chunk %s (attempt %d/%d): %s — retrying",
                                history_name,
                                chunk_idx,
                                _attempt + 1,
                                _add_episode_max_retries,
                                _exc,
                            )
                        else:
                            logger.error(
                                "add_episode failed for %s chunk %s after %d attempts, skipping: %s",
                                history_name,
                                chunk_idx,
                                _add_episode_max_retries,
                                _exc,
                            )

            await self._export_to_faiss(
                graphiti_client, database, history_name, incremental=incremental
            )
        finally:
            await _close_kuzu_driver(graphiti_client.driver)

    async def _export_to_faiss(
        self,
        graphiti_client,
        database: LocalFaissDatabase,
        history_name: str,
        *,
        incremental: bool = False,
    ) -> None:
        """Export all currently-valid EntityEdge facts to LocalFaissDatabase."""
        from graphiti_core.edges import EntityEdge
        from graphiti_core.errors import GroupsEdgesNotFoundError

        try:
            all_edges = await EntityEdge.get_by_group_ids(
                graphiti_client.driver,
                [history_name],
            )
        except GroupsEdgesNotFoundError:
            # graphiti raises instead of returning [] when the graph has no edges
            # (e.g. LLM extracted 0 facts from this episode's chunks)
            logger.warning(
                "[ZepMemorySystem] %s: no edges in graph, exporting 0 facts to FAISS",
                history_name,
            )
            all_edges = []
        valid_edges = [
            e for e in all_edges if e.invalid_at is None and (e.fact or "").strip()
        ]
        logger.info(
            "[ZepMemorySystem] %s: exporting %d valid edges to FAISS (total=%d)",
            history_name,
            len(valid_edges),
            len(all_edges),
        )

        existing_edge_uuids: set[str] = set()
        if incremental:
            for mem in database.list_all_memories(sort_by_time=False):
                uid = mem.metadata.get("entity_edge_uuid")
                if uid:
                    existing_edge_uuids.add(str(uid))

        for edge in valid_edges:
            if incremental and edge.uuid in existing_edge_uuids:
                continue
            text = edge.fact.strip()
            time_str = edge.valid_at.strftime("%Y-%m-%d") if edge.valid_at else ""
            metadata: Dict[str, Any] = {
                "method": "zep",
                "history_name": history_name,
                "source": "graphiti_entity_edge",
                "granularity": "all",
                "date": time_str,
                "entity_edge_uuid": edge.uuid,
            }
            emb = self._embed_texts([text])
            database.add(
                text=text,
                source_index="zep_export",
                time=time_str,
                metadata=metadata,
                embedding=emb[0],
            )

    # ------------------------------------------------------------------
    # Retrieval (LocalFaissDatabase, same pattern as mem0)
    # ------------------------------------------------------------------

    def retrieve(
        self,
        history_name: str,
        query: str,
        current_time: str,
        top_k: int = 5,
    ) -> List[RetrievedMemory]:
        database = self._get_database(history_name)
        query_embedding = self._embed_texts([query])
        if query_embedding.size == 0:
            return []
        return database.search(query_embedding[0], top_k)


__all__ = ["ZepMemorySystem"]
