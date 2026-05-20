"""Turn observation text → conversation_history dicts for confidence / utility."""

from __future__ import annotations

from typing import Any, Dict, List


def observation_to_history_turns(observation: str) -> List[Dict[str, Any]]:
    """Split non-empty lines into ``[{"text": ...}, ...]`` for feature extractors."""
    obs = (observation or "").strip()
    if not obs:
        return []
    turns: List[Dict[str, Any]] = []
    for raw in obs.splitlines():
        line = raw.strip()
        if line:
            turns.append({"text": line})
    return turns if turns else [{"text": obs}]


def recency_score_chunk_order(chunk_ordinal: int, chunk_count: int, decay_per_step: float = 0.12) -> float:
    """Later chunks score higher (same session order as ingest)."""
    import math

    if chunk_count <= 0:
        return 1.0
    co = max(0, min(chunk_ordinal, chunk_count - 1))
    steps_back = (chunk_count - 1) - co
    r = math.exp(-decay_per_step * float(steps_back))
    return float(max(0.0, min(1.0, r)))
