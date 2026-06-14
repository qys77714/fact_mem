"""
User-requested memory deletion helpers for relation_decision ingest.

Deletion-first: detect explicit remove/delete requests, match a prior memory row
(primary or evidence), mark stale, add a value-free tombstone; skip cascade/pairwise.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from memory.base import RetrievedMemory
from memory.storage.local_faiss import LocalFaissDatabase, _memory_entry_is_primary
from memory.tracing import MemoryTraceLogger

from .cas_update import cosine_sim, metadata_for_new_primary

TOMBSTONE_TEXT = (
    "This information was explicitly deleted by the user and must not be recalled."
)

_DELETION_EN_PATTERNS = (
    re.compile(r"\bplease\s+remove\b", re.I),
    re.compile(r"\bremove\s+(?:that|this|it)\b", re.I),
    re.compile(r"\bremove\b.+\bfrom\s+(?:your\s+)?memory\b", re.I),
    re.compile(r"\bdelete\s+(?:that|this|it)\b", re.I),
    re.compile(r"\bdelete\b.+\bfrom\s+(?:your\s+)?memory\b", re.I),
    re.compile(r"\bforget\s+(?:that|this|it)\b", re.I),
)

_DELETION_ZH_PATTERNS = (
    re.compile(r"请删除"),
    re.compile(r"从记忆中移除"),
    re.compile(r"从.*?记忆.*?删除"),
    re.compile(r"忘记"),
)

_STRIP_EN_PATTERNS = (
    re.compile(r"\.\s*please\s+remove\b.*$", re.I | re.S),
    re.compile(
        r"\.\s*remove\s+(?:that|this|it)\s+from\s+(?:your\s+)?memory\b.*$",
        re.I | re.S,
    ),
    re.compile(
        r"\.\s*delete\s+(?:that|this|it)\s+from\s+(?:your\s+)?memory\b.*$",
        re.I | re.S,
    ),
    re.compile(r"\s+please\s+remove\b.*$", re.I | re.S),
    re.compile(
        r"\s+remove\s+(?:that|this|it)\s+from\s+(?:your\s+)?memory\b.*$",
        re.I | re.S,
    ),
)

_STRIP_ZH_PATTERNS = (
    re.compile(r"[。．]\s*请删除.*$", re.S),
    re.compile(r"[。．]\s*从记忆中移除.*$", re.S),
    re.compile(r"\s*请删除.*$", re.S),
    re.compile(r"\s*从记忆中移除.*$", re.S),
)


def is_user_deletion_request(text: str, *, language: str = "en") -> bool:
    t = (text or "").strip()
    if not t:
        return False
    patterns = _DELETION_ZH_PATTERNS if str(language or "en").lower().startswith("zh") else _DELETION_EN_PATTERNS
    return any(p.search(t) for p in patterns)


def strip_deletion_clause(text: str, *, language: str = "en") -> str:
    """Return the factual主体 fragment used for embedding-based target match."""
    t = (text or "").strip()
    if not t:
        return ""
    patterns = _STRIP_ZH_PATTERNS if str(language or "en").lower().startswith("zh") else _STRIP_EN_PATTERNS
    for pat in patterns:
        stripped = pat.sub("", t).strip().rstrip(".")
        if stripped and stripped != t:
            return stripped
    return t


def _memory_role_label(meta: Optional[Dict[str, Any]]) -> str:
    if not meta:
        return "primary"
    return "evidence" if meta.get("memory_role") == "evidence" else "primary"


def _sort_key_for_match(item: Tuple[RetrievedMemory, float]) -> Tuple[float, int, str]:
    mem, sim = item
    is_primary = 1 if _memory_entry_is_primary(mem.metadata or {}) else 0
    return (-sim, -is_primary, mem.memory_id)


def find_deletion_target(
    db: LocalFaissDatabase,
    m_new: str,
    embed_fn: Callable[[List[str]], np.ndarray],
    sim_threshold: float,
    *,
    language: str = "en",
) -> Tuple[Optional[RetrievedMemory], List[Dict[str, Any]]]:
    """
    Argmax cosine similarity over all non-stale rows (primary + evidence).
    Returns (best_match_or_None, top3_debug_records).
    """
    query_text = strip_deletion_clause(m_new, language=language)
    if not query_text:
        return None, []

    query_emb = embed_fn([query_text])[0]
    candidates = [
        m
        for m in db.list_all_memories(sort_by_time=False)
        if not bool((m.metadata or {}).get("stale"))
    ]
    if not candidates:
        return None, []

    texts = [(m.text or "").strip() for m in candidates]
    cand_embs = embed_fn(texts)
    scored: List[Tuple[RetrievedMemory, float]] = []
    for i, mem in enumerate(candidates):
        row_emb = cand_embs[i] if cand_embs.ndim == 2 else cand_embs
        sim = cosine_sim(query_emb, row_emb)
        scored.append((mem, sim))

    scored.sort(key=_sort_key_for_match)
    top_debug = [
        {
            "memory_id": mem.memory_id,
            "role": _memory_role_label(mem.metadata),
            "similarity": round(sim, 4),
            "text_prefix": (mem.text or "")[:120],
        }
        for mem, sim in scored[:3]
    ]

    if not scored or scored[0][1] < sim_threshold:
        return None, top_debug

    return scored[0][0], top_debug


def _add_deletion_tombstone(
    database: LocalFaissDatabase,
    metadata_base: Dict[str, Any],
    session_idx: int,
    chunk_scope: str,
    trace: MemoryTraceLogger,
    embed_fn: Callable[[List[str]], np.ndarray],
    *,
    matched: bool,
    target_id: Optional[str] = None,
) -> str:
    meta = metadata_for_new_primary(
        metadata_base,
        TOMBSTONE_TEXT,
        lme_update_method="relation_decision_deletion",
    )
    meta["user_deleted"] = True
    meta["deletion_tombstone"] = True
    if target_id:
        meta["deletion_target_id"] = target_id
    emb = embed_fn([TOMBSTONE_TEXT])[0]
    date_s = str(metadata_base.get("date", ""))
    memory_id = database.add(
        text=TOMBSTONE_TEXT,
        source_index=f"session_{session_idx}",
        time=date_s,
        metadata=meta,
        embedding=emb,
    )
    trace.log_memory_operation(
        operation="ADD_DELETION_TOMBSTONE",
        memory_id=memory_id,
        scope_id=chunk_scope,
        metadata={"matched_target": matched, "deletion_target_id": target_id},
        after={
            "text": TOMBSTONE_TEXT,
            "source_index": f"session_{session_idx}",
            "time": date_s,
            "metadata": dict(meta),
        },
        status="ok",
    )
    return memory_id


def apply_user_deletion(
    database: LocalFaissDatabase,
    m_new: str,
    target: Optional[RetrievedMemory],
    metadata_base: Dict[str, Any],
    session_idx: int,
    chunk_scope: str,
    trace: MemoryTraceLogger,
    embed_fn: Callable[[List[str]], np.ndarray],
    *,
    match_debug: Sequence[Dict[str, Any]],
) -> int:
    """Handle one deletion request. Always returns >= 1 (tombstone added)."""
    trace.log_memory_operation(
        operation="USER_DELETE",
        memory_id=target.memory_id if target else "",
        scope_id=chunk_scope,
        metadata={
            "m_new": m_new,
            "query_text": strip_deletion_clause(m_new),
            "matched": target is not None,
            "match_top": list(match_debug),
        },
        status="ok",
    )

    if target is not None:
        meta = target.metadata or {}
        is_evidence = meta.get("memory_role") == "evidence"
        updates: Dict[str, Any] = {
            "stale": True,
            "user_deleted": True,
            "lme_update_method": "relation_decision_deletion",
        }
        if is_evidence:
            updates["memory_role"] = "primary"
            updates["parent_primary"] = ""
            updates["edge"] = ""
            database.update_memory(target.memory_id, metadata_updates=updates)
            trace.log_memory_operation(
                operation="USER_DELETE_DETACH_EVIDENCE",
                memory_id=target.memory_id,
                scope_id=chunk_scope,
                metadata={"m_new": m_new},
                status="ok",
            )
        else:
            database.update_memory(target.memory_id, metadata_updates=updates)
        trace.log_memory_operation(
            operation="USER_DELETE_STALE",
            memory_id=target.memory_id,
            scope_id=chunk_scope,
            metadata={"m_new": m_new, "was_evidence": is_evidence},
            status="ok",
        )

    _add_deletion_tombstone(
        database,
        metadata_base,
        session_idx,
        chunk_scope,
        trace,
        embed_fn,
        matched=target is not None,
        target_id=target.memory_id if target else None,
    )
    return 1
