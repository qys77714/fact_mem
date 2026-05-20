"""Novelty (N): 1 - max cosine similarity to existing primaries (same embedding API as ingest)."""

from __future__ import annotations

from typing import Callable, List

import numpy as np

EmbedFn = Callable[[List[str]], np.ndarray]


def novelty_score(embed_texts: EmbedFn, candidate: str, existing_texts: List[str], max_existing: int = 64) -> float:
    cand = (candidate or "").strip()
    if not cand:
        return 0.0
    if not existing_texts:
        return 1.0
    tail = existing_texts[-max_existing:] if len(existing_texts) > max_existing else list(existing_texts)
    emb = embed_texts([cand] + tail)
    if emb.size == 0 or emb.shape[0] < 1:
        return 0.5
    v = emb[0:1].astype(np.float64)
    if emb.shape[0] == 1:
        return 1.0
    rest = emb[1:].astype(np.float64)
    v_norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    r_norm = np.linalg.norm(rest, axis=1, keepdims=True) + 1e-12
    v_u = v / v_norm
    r_u = rest / r_norm
    sims = (r_u @ v_u.T).ravel()
    max_sim = float(np.max(sims)) if sims.size else 0.0
    return float(max(0.0, min(1.0, 1.0 - max_sim)))
