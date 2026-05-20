"""Weighted linear admission: S = w·[U,C,N,R,T]."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .confidence import AmacConfidenceExtractor
from .novelty import novelty_score
from .observation import observation_to_history_turns, recency_score_chunk_order
from .type_prior import AmacTypePriorExtractor
from .utility import utility_llm_score

EmbedFn = Callable[[List[str]], np.ndarray]


@dataclass(frozen=True)
class AmacCandidate:
    content: str
    turn_id: int = 0
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class AmacFeatures:
    utility: float
    confidence: float
    novelty: float
    recency: float
    type_prior: float

    def as_vector(self) -> np.ndarray:
        return np.array(
            [self.utility, self.confidence, self.novelty, self.recency, self.type_prior],
            dtype=np.float64,
        )


def parse_amac_weights_arg(s: str) -> np.ndarray:
    """Parse ``0.1,0.1,0.1,0.1,0.6`` or JSON list into normalized length-5 weights."""
    raw = (s or "").strip()
    if not raw:
        return np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float64)
    if raw.startswith("["):
        arr = json.loads(raw)
        if not isinstance(arr, list) or len(arr) != 5:
            raise ValueError("amac weights JSON must be a list of 5 numbers")
        w = np.array([float(x) for x in arr], dtype=np.float64)
    else:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) != 5:
            raise ValueError("--amac-weights must be 5 comma-separated floats [U,C,N,R,T]")
        w = np.array([float(p) for p in parts], dtype=np.float64)
    if np.any(w < 0) or np.any(np.isnan(w)):
        raise ValueError("amac weights must be non-negative finite")
    ssum = float(np.sum(w))
    if ssum <= 0:
        raise ValueError("amac weights sum must be positive")
    w = w / ssum
    return w


class AmacAdmissionScorer:
    """S(m) = dot(w, features); admit if S >= threshold."""

    def __init__(
        self,
        weights: np.ndarray,
        threshold: float,
        *,
        llm_client: Any,
        embed_texts: EmbedFn,
        max_new_tokens: int = 256,
        language: str = "en",
        skip_utility: bool = False,
        recency_decay_per_step: float = 0.12,
        novelty_max_existing: int = 64,
    ) -> None:
        w = np.asarray(weights, dtype=np.float64).reshape(5)
        if w.shape[0] != 5:
            raise ValueError("weights must have length 5 [U,C,N,R,T]")
        self._weights = w / np.sum(w) if np.sum(w) > 0 else w
        self._threshold = float(threshold)
        self._llm = llm_client
        self._embed_texts = embed_texts
        self._max_new_tokens = int(max_new_tokens)
        self._language = (language or "en").strip() or "en"
        self._skip_utility = bool(skip_utility)
        self._recency_decay = float(recency_decay_per_step)
        self._novelty_max = int(novelty_max_existing)
        self._conf_ext = AmacConfidenceExtractor()
        self._type_ext = AmacTypePriorExtractor()

    def compute_features(
        self,
        candidate: AmacCandidate,
        observation_text: str,
        existing_primary_texts: List[str],
        *,
        chunk_ordinal: int,
        chunk_count: int,
    ) -> AmacFeatures:
        hist = observation_to_history_turns(observation_text)
        text = (candidate.content or "").strip()

        if self._skip_utility:
            u = 0.5
        else:
            u = utility_llm_score(
                self._llm,
                candidate=text,
                conversation_history=hist,
                max_new_tokens=self._max_new_tokens,
                language=self._language,
            )

        c = self._conf_ext.score(text, hist)
        n = novelty_score(self._embed_texts, text, existing_primary_texts, max_existing=self._novelty_max)
        r = recency_score_chunk_order(chunk_ordinal, chunk_count, decay_per_step=self._recency_decay)
        t = self._type_ext.score(text, candidate.metadata)

        return AmacFeatures(utility=u, confidence=c, novelty=n, recency=r, type_prior=t)

    def score(self, feats: AmacFeatures) -> float:
        return float(np.dot(self._weights, feats.as_vector()))

    def admit(
        self,
        candidate: AmacCandidate,
        observation_text: str,
        existing_primary_texts: List[str],
        *,
        chunk_ordinal: int,
        chunk_count: int,
    ) -> tuple[bool, float, AmacFeatures]:
        feats = self.compute_features(
            candidate,
            observation_text,
            existing_primary_texts,
            chunk_ordinal=chunk_ordinal,
            chunk_count=chunk_count,
        )
        s = self.score(feats)
        return s >= self._threshold, s, feats
