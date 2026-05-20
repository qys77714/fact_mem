"""
Reranker landscape (common choices and how they are used)

**Commercial API (widely used in production RAG)**  
- **Cohere Rerank**: ``cohere.Client(api_key=...).rerank(model="rerank-v3.5"``, or newer
  ``rerank-v4.0-pro`` / ``rerank-v4.0-fast``), ``query=...``, ``documents=[...]`` → ordered
  relevance scores. Good multilingual coverage; no local GPU.

**Open weights (local cross-encoders)**  
- **BGE rerankers** (BAAI): ``BAAI/bge-reranker-base``, ``BAAI/bge-reranker-v2-m3`` (multilingual).
  Typical usage: ``FlagEmbedding.FlagReranker`` (``compute_score`` / ``compute_score`` batch),
  or Hugging Face ``AutoModelForSequenceClassification`` on ``[CLS]`` logits for (query, passage) pairs.
- **MS MARCO–style cross-encoders**: ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (small baseline),
  ``cross-encoder/ms-marco-electra-base`` (stronger). Convenient API:
  ``sentence_transformers.CrossEncoder.predict([[q, d], ...])``.

**Typical retrieval pattern**  
1) Bi-encoder / BM25 retrieves ``top_k * pool`` candidates (e.g. ``k=5``, ``pool=4`` → 20).  
2) Reranker scores each ``(query, doc)`` pair.  
3) Keep top ``k`` by reranker score for the LLM context.

This module tests a small **CrossEncoder** path (optional deps), a **live vLLM /score**
check when ``RERANKER_API_KEY`` is set, plus a **pure rerank helper** that stays runnable
without ``torch`` / model download.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence
from urllib.error import HTTPError, URLError

import pytest

from utils.env import load_env
from utils.qwen3_reranker_vllm import (
    DEFAULT_INSTRUCTION_EN,
    build_qwen3_reranker_vllm_score_texts,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class ScoredDoc:
    index: int
    text: str
    score: float


def rerank_documents(
    query: str,
    documents: Sequence[str],
    pair_scores: Sequence[float],
) -> List[ScoredDoc]:
    """
    Map parallel ``pair_scores`` (one per document, same order) to sorted ``ScoredDoc`` list.

    ``pair_scores`` should come from any cross-encoder / reranker that outputs a scalar
    relevance score per (query, document) pair.
    """
    if len(documents) != len(pair_scores):
        raise ValueError("documents and pair_scores must have the same length")
    ranked = [
        ScoredDoc(index=i, text=documents[i], score=float(pair_scores[i]))
        for i in range(len(documents))
    ]
    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked


def test_rerank_documents_sorts_by_score() -> None:
    query = "ignored here"
    docs = ["a", "b", "c"]
    scores = [0.1, 0.9, 0.3]
    out = rerank_documents(query, docs, scores)
    assert [x.text for x in out] == ["b", "c", "a"]
    assert [x.index for x in out] == [1, 2, 0]


def test_rerank_documents_length_mismatch() -> None:
    with pytest.raises(ValueError):
        rerank_documents("q", ["a", "b"], [0.1])


def _default_reranker_base_url() -> str:
    return os.getenv("RERANKER_BASE_URL", "http://localhost:7114/v1").rstrip("/") + "/"


def _reranker_integration_ready() -> bool:
    load_env(str(PROJECT_ROOT / ".env"))
    return bool((os.getenv("RERANKER_API_KEY") or "").strip())


def _vllm_score_many_vs_one(
    *,
    base_url: str,
    api_key: str,
    model: str,
    query: str,
    documents: List[str],
    timeout_s: float = 120.0,
    instruction: str = DEFAULT_INSTRUCTION_EN,
) -> List[float]:
    """
    Call vLLM OpenAI-compatible ``POST .../v1/score`` (``ScoreRequest``: one query, many docs).

    vLLM cross-encoder rerankers concatenate ``text_1 + text_2``; we send the same
    query/document fragments as upstream ``qwen3_reranker.py``. ``text_1`` is
    broadcast when it is a single string and ``text_2`` is a list.
    """
    text_1, text_2 = build_qwen3_reranker_vllm_score_texts(
        query, documents, instruction=instruction
    )
    url = base_url.rstrip("/") + "/score"
    payload = {"model": model, "text_1": text_1, "text_2": text_2}
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    items = sorted(data["data"], key=lambda x: int(x["index"]))
    return [float(x["score"]) for x in items]


@pytest.mark.slow
def test_qwen3_reranker_vllm_score_api_orders_relevant_passage() -> None:
    """
    Live check against local vLLM (``script/0_run_reranker_ppu.sh``, ``--task score``).

    Requires ``RERANKER_API_KEY`` and optional ``RERANKER_BASE_URL`` / ``RERANKER_MODEL``
    in project root ``.env`` (same style as embedding). Skips if unset or service down.
    """
    if not _reranker_integration_ready():
        pytest.skip("Set RERANKER_API_KEY in .env (and start vLLM reranker) to run this test.")

    api_key = os.environ["RERANKER_API_KEY"]
    base = _default_reranker_base_url()
    model = (os.getenv("RERANKER_MODEL") or "Qwen3-Reranker-0.6B").strip()

    query = "What is the capital of France?"
    documents = [
        "Paris is the capital and largest city of France.",
        "Berlin is the capital of Germany.",
        "Pasta is common in Italian cuisine.",
    ]

    try:
        scores = _vllm_score_many_vs_one(
            base_url=base,
            api_key=api_key,
            model=model,
            query=query,
            documents=documents,
        )
    except (URLError, HTTPError, TimeoutError, KeyError, json.JSONDecodeError) as e:
        pytest.skip(f"Reranker HTTP score call failed ({type(e).__name__}: {e}). Is vLLM up on {base}?")

    assert len(scores) == len(documents)
    ranked = rerank_documents(query, documents, scores)
    assert "Paris" in ranked[0].text


def test_cross_encoder_ms_marco_mini_orders_capital_query() -> None:
    """
    Integration check using a small public cross-encoder (first run downloads weights).

    Install extras for local neural reranking, e.g.::

        uv add sentence-transformers

    (pulls PyTorch; CPU is enough for this tiny model and test batch).
    """
    pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")
    from sentence_transformers import CrossEncoder

    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ce = CrossEncoder(model_name)

    query = "What is the capital of France?"
    documents = [
        "Paris is the capital and largest city of France.",
        "Berlin is the capital of Germany.",
        "Pasta is common in Italian cuisine.",
    ]
    pairs: List[List[str]] = [[query, d] for d in documents]
    raw = ce.predict(pairs)
    scores = raw.reshape(-1).tolist() if hasattr(raw, "reshape") else list(raw)

    ranked = rerank_documents(query, documents, scores)
    assert "Paris" in ranked[0].text


def mock_two_stage_retrieve(
    query: str,
    corpus: Sequence[str],
    coarse_ranking: Sequence[int],
    pair_scorer: Callable[[str, str], float],
    final_k: int,
) -> List[ScoredDoc]:
    """
    Illustrate retrieve-then-rerank without FAISS: ``coarse_ranking`` is candidate indices.
    """
    candidates = [corpus[i] for i in coarse_ranking]
    scores = [pair_scorer(query, corpus[i]) for i in coarse_ranking]
    reranked = rerank_documents(query, candidates, scores)
    return reranked[:final_k]


def test_mock_two_stage_retrieve_respects_reranker() -> None:
    corpus = [
        "alpha",
        "beta matches query keyword",
        "gamma",
    ]
    # Pretend coarse stage returned worst-first
    coarse = [0, 2, 1]

    def scorer(q: str, doc: str) -> float:
        return float("keyword" in doc)

    out = mock_two_stage_retrieve("any query keyword", corpus, coarse, scorer, final_k=1)
    assert out[0].text == "beta matches query keyword"
