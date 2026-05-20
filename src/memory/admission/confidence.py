"""Confidence (C): ROUGE-L style alignment with conversation context."""

from __future__ import annotations

from typing import Any, Dict, List

from rouge_score import rouge_scorer


class AmacConfidenceExtractor:
    """ROUGE-L between candidate and overlapping context lines."""

    def __init__(self, rouge_metric: str = "rougeL") -> None:
        self._rouge_metric = rouge_metric
        self._scorer = rouge_scorer.RougeScorer([rouge_metric], use_stemmer=True)

    def score(self, memory_text: str, conversation_history: List[Dict[str, Any]]) -> float:
        memory_text = (memory_text or "").strip()
        if not memory_text:
            return 0.0
        spans = self._supporting_spans(memory_text, conversation_history)
        if not spans:
            return 0.0
        best = 0.0
        for span in spans:
            try:
                scores = self._scorer.score(span, memory_text)
                best = max(best, float(scores[self._rouge_metric].fmeasure))
            except Exception:
                best = max(best, self._overlap_f1(memory_text, span))
        return max(0.0, min(1.0, best))

    def _supporting_spans(self, memory_text: str, conversation_history: List[Dict[str, Any]]) -> List[str]:
        memory_words = set(memory_text.lower().split())
        spans: List[str] = []
        for turn in conversation_history:
            text = str(turn.get("text") or "").strip()
            if not text:
                continue
            turn_words = set(text.lower().split())
            if memory_words and turn_words and memory_words.intersection(turn_words):
                spans.append(text)
        return spans

    def _overlap_f1(self, memory_text: str, evidence_span: str) -> float:
        memory_words = set(memory_text.lower().split())
        evidence_words = set(evidence_span.lower().split())
        if not memory_words or not evidence_words:
            return 0.0
        overlap = memory_words.intersection(evidence_words)
        precision = len(overlap) / len(memory_words)
        recall = len(overlap) / len(evidence_words)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
