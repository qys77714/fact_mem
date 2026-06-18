"""Apply candidate JSON → Mem0 update pipeline (facts from JSON, no dialogue extraction)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from memory.mem0 import Mem0MemorySystem

from .apply import _sorted_chunks, load_candidate_json
from .cas_update import merged_candidate_texts


def _facts_from_chunk(chunk: Dict[str, Any]) -> List[str]:
    # Fold cas_update_rules back into the text so mem0 sees the same information
    # ours consumes via the parallel column (input parity across methods).
    return merged_candidate_texts(chunk)


def apply_candidate_episode_mem0(
    memory: Mem0MemorySystem,
    payload: Dict[str, Any],
    *,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Walk chunks in order; per chunk, feed ``candidate_memories`` strings through Mem0's
    ``_collect_related_memories`` → ``_decide_memory_operations`` → ``_apply_memory_changes``
    (same batch semantics as dialogue-driven Mem0 per chunk).
    """
    history_name = str(payload.get("history_name") or "").strip()
    if not history_name:
        raise ValueError("candidate json: missing history_name")

    database = memory._get_database(history_name)
    trace = memory.trace.get_logger_for(history_name)
    sorted_c = _sorted_chunks(payload)

    ep_scope = trace.create_scope(
        "mem0_candidate_episode",
        metadata={
            "history_name": history_name,
            "source_file": str(source_path) if source_path else None,
            "num_chunks": len(sorted_c),
        },
    )

    stats: Dict[str, Any] = {
        "history_name": history_name,
        "chunks": len(sorted_c),
        "facts_submitted": 0,
        "operation_batches": 0,
    }

    try:
        for chunk in sorted_c:
            ci = chunk.get("chunk_index")
            ch_scope = trace.create_scope(
                "mem0_candidate_chunk",
                parent_scope_id=ep_scope,
                metadata={
                    "chunk_index": ci,
                    "session_index": chunk.get("session_index"),
                    "turn_start": chunk.get("turn_start"),
                    "turn_end": chunk.get("turn_end"),
                },
            )
            facts = _facts_from_chunk(chunk)
            if not facts:
                trace.close_scope(ch_scope, status="ok", metadata={"skipped": "no_facts"})
                continue

            stats["facts_submitted"] += len(facts)
            session_idx = int(chunk.get("session_index") or 1)
            session_date = str(chunk.get("session_date") or "").strip()

            metadata_base: Dict[str, Any] = {
                "method": "mem0_nodel" if not memory._allow_memory_delete else "mem0",
                "history_name": history_name,
                "session": session_idx,
                "date": session_date,
                "granularity": memory.granularity,
                "source": "candidate_extract",
                "candidate_chunk_index": ci,
            }
            ts = chunk.get("turn_start")
            te = chunk.get("turn_end")
            if ts is not None:
                metadata_base["turn_start"] = ts
            if te is not None:
                metadata_base["turn_end"] = te

            old_memory_json, temp_uuid_mapping = memory._collect_related_memories(
                database, facts, session_date
            )
            operations = memory._decide_memory_operations(
                facts,
                old_memory_json,
                trace_scope_id=ch_scope,
                trace=trace,
            )
            memory._apply_memory_changes(
                database,
                operations,
                temp_uuid_mapping,
                metadata_base,
                session_idx,
                trace_scope_id=ch_scope,
                trace=trace,
            )
            stats["operation_batches"] += 1
            trace.close_scope(
                ch_scope,
                status="ok",
                metadata={"operation_count": len(operations)},
            )
    finally:
        trace.close_scope(ep_scope, status="ok", metadata=stats)

    return stats


def apply_candidate_file_mem0(
    memory: Mem0MemorySystem,
    path: Path,
) -> Dict[str, Any]:
    payload = load_candidate_json(path)
    return apply_candidate_episode_mem0(memory, payload, source_path=path)
