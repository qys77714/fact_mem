"""Token-level exact match and F1 for QA evaluation (SQuAD-style normalization)."""

from __future__ import annotations

import re
import string
from typing import List, Literal, Tuple

TokenMode = Literal["whitespace", "char"]

_PUNCT = set(string.punctuation)


def normalize_answer(text: str) -> str:
    """Lowercase, drop punctuation, remove English articles, collapse whitespace."""

    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def remove_punc(s: str) -> str:
        return "".join(ch for ch in s if ch not in _PUNCT)

    s = text.lower()
    s = remove_punc(s)
    s = remove_articles(s)
    return re.sub(r"\s+", " ", s).strip()


def _tokens(normalized: str, mode: TokenMode) -> List[str]:
    if mode == "char":
        compact = normalized.replace(" ", "")
        return list(compact) if compact else []
    return normalized.split()


def get_tokens_for_f1(text: str, mode: TokenMode) -> List[str]:
    return _tokens(normalize_answer(text), mode)


def compute_exact(prediction: str, ground_truth: str, mode: TokenMode) -> bool:
    return get_tokens_for_f1(prediction, mode) == get_tokens_for_f1(ground_truth, mode)


def compute_f1(prediction: str, ground_truth: str, mode: TokenMode) -> float:
    pred_tokens = get_tokens_for_f1(prediction, mode)
    gold_tokens = get_tokens_for_f1(ground_truth, mode)

    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0

    pred_counter: dict[str, int] = {}
    for t in pred_tokens:
        pred_counter[t] = pred_counter.get(t, 0) + 1
    gold_counter: dict[str, int] = {}
    for t in gold_tokens:
        gold_counter[t] = gold_counter.get(t, 0) + 1

    overlap = 0
    for t, gc in gold_counter.items():
        overlap += min(gc, pred_counter.get(t, 0))

    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_f1_em(
    prediction: str, ground_truth: str, mode: TokenMode
) -> Tuple[float, bool]:
    return compute_f1(prediction, ground_truth, mode), compute_exact(prediction, ground_truth, mode)
