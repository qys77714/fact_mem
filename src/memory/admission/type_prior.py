"""Type prior (T): lightweight rule-based category score."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


class AmacTypePriorExtractor:
    """Keyword-based type classification and fixed priors (A-MAC-style)."""

    def __init__(self) -> None:
        self._type_priors: Dict[str, float] = {
            "preference": 0.9,
            "identity": 0.9,
            "belief": 0.9,
            "value": 0.9,
            "fact": 0.7,
            "knowledge": 0.7,
            "information": 0.7,
            "plan": 0.5,
            "goal": 0.5,
            "intention": 0.5,
            "task": 0.5,
            "temporary": 0.2,
            "ephemeral": 0.2,
            "transient": 0.2,
            "state": 0.3,
            "unknown": 0.5,
        }
        self._type_keywords: Dict[str, set[str]] = {
            "preference": {
                "prefer", "like", "dislike", "hate", "love", "favorite",
                "enjoy", "appreciate", "avoid", "want", "wish", "喜欢", "讨厌", "偏好",
            },
            "identity": {
                "i am", "i'm", "my name is", "called", "live in", "from",
                "work as", "job", "profession", "我叫", "我是", "住在",
            },
            "fact": {
                "is", "are", "was", "were", "always", "never", "true",
                "false", "known", "是", "不是", "位于",
            },
            "plan": {
                "will", "going to", "plan to", "intend to", "scheduled",
                "planning", "tomorrow", "next", "打算", "计划", "将要",
            },
            "goal": {
                "want to", "hope to", "aim to", "goal", "objective",
                "target", "希望", "目标",
            },
            "temporary": {
                "currently", "right now", "at the moment", "today",
                "temporarily", "目前", "现在", "正在", "临时",
            },
        }
        self._patterns: Dict[str, list[str]] = {
            "identity": [
                r"\b(my name is|i am|i'm)\s+\w+",
                r"\b(live in|from|based in)\s+[A-Z]\w+",
            ],
            "preference": [
                r"\b(prefer|like|love|favorite)\s+\w+",
            ],
            "plan": [
                r"\b(will|going to|planning to)\s+\w+",
            ],
        }

    def classify(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        if metadata and isinstance(metadata.get("type"), str):
            return str(metadata["type"]).strip().lower() or "unknown"
        text = (content or "").lower()
        for mem_type, patterns in self._patterns.items():
            for pattern in patterns:
                if re.search(pattern, content or "", re.IGNORECASE):
                    return mem_type
        type_scores: Dict[str, int] = {}
        for mem_type, keywords in self._type_keywords.items():
            score = sum(1 for kw in keywords if kw in text)
            if score:
                type_scores[mem_type] = score
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]
        return "unknown"

    def score(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> float:
        t = self.classify(content, metadata)
        return float(self._type_priors.get(t, self._type_priors["unknown"]))
