"""Apply candidate JSON → Zep/graphiti update pipeline (facts from JSON, no dialogue extraction)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from memory.zep import ZepMemorySystem

from .apply import _sorted_chunks, load_candidate_json
from .cas_update import chunk_with_merged_candidates


def apply_candidate_episode_zep(
    memory: ZepMemorySystem,
    payload: Dict[str, Any],
    *,
    source_path: Optional[Path] = None,
    incremental: bool = False,
) -> Dict[str, Any]:
    """
    Walk chunks in order; feed ``candidate_memories`` strings through graphiti's
    entity-deduplication and temporal-validity pipeline via
    ``memory.process_chunks()``.  Valid EntityEdge facts are then exported to
    ``LocalFaissDatabase`` (inside ``process_chunks``).

    Mirrors the structure of ``apply_candidate_episode_mem0``.
    """
    history_name = str(payload.get("history_name") or "").strip()
    if not history_name:
        raise ValueError("candidate json: missing history_name")

    sorted_c = _sorted_chunks(payload)
    # Fold cas_update_rules back into candidate_memories so zep sees the same
    # information ours consumes via the parallel column (input parity across methods).
    sorted_c = [chunk_with_merged_candidates(c) for c in sorted_c]

    trace = memory.trace.get_logger_for(history_name)
    ep_scope = trace.create_scope(
        "zep_candidate_episode",
        metadata={
            "history_name": history_name,
            "source_file": str(source_path) if source_path else None,
            "num_chunks": len(sorted_c),
        },
    )

    stats: Dict[str, Any] = {
        "history_name": history_name,
        "chunks": len(sorted_c),
        "facts_submitted": sum(
            len([m for m in (c.get("candidate_memories") or []) if str(m).strip()])
            for c in sorted_c
        ),
    }

    try:
        memory.process_chunks(history_name, sorted_c, incremental=incremental)
    finally:
        trace.close_scope(ep_scope, status="ok", metadata=stats)

    return stats


def apply_candidate_file_zep(
    memory: ZepMemorySystem,
    path: Path,
) -> Dict[str, Any]:
    payload = load_candidate_json(path)
    return apply_candidate_episode_zep(memory, payload, source_path=path)
