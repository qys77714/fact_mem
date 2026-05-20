"""LmePrebuiltMemorySystem Qwen3 vLLM rerank path (mocked HTTP)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from memory.base import RetrievedMemory
from memory.baselines.lme_prebuilt import LmePrebuiltMemorySystem


def _mem(i: int, text: str, score: float = 0.0) -> RetrievedMemory:
    return RetrievedMemory(
        memory_id=str(i),
        text=text,
        source_index=f"s-{i}",
        time="t",
        score=score,
        metadata={},
    )


def test_apply_qwen3_vllm_rerank_reorders_by_score() -> None:
    mem = object.__new__(LmePrebuiltMemorySystem)
    mem._rerank_qwen3_vllm = True
    mem._rerank_base_url = "http://localhost:7114/v1/"
    mem._rerank_api_key = "k"
    mem._rerank_model = "Qwen3-Reranker-0.6B"
    mem._rerank_timeout_s = 30.0
    mem._rerank_top_k = None
    mem._language = "en"

    coarse = [_mem(0, "low", 9.0), _mem(1, "high", 1.0)]
    with patch(
        "memory.baselines.lme_prebuilt.qwen3_vllm_score_documents",
        return_value=[0.1, 0.9],
    ):
        out = LmePrebuiltMemorySystem._apply_qwen3_vllm_rerank(mem, "q", coarse, retrieve_top_k=2)
    assert [m.memory_id for m in out] == ["1", "0"]
    assert out[0].score == 0.9
    assert out[1].score == 0.1


def test_apply_qwen3_vllm_rerank_respects_rerank_top_k() -> None:
    mem = object.__new__(LmePrebuiltMemorySystem)
    mem._rerank_qwen3_vllm = True
    mem._rerank_base_url = "http://localhost:7114/v1/"
    mem._rerank_api_key = "k"
    mem._rerank_model = "Qwen3-Reranker-0.6B"
    mem._rerank_timeout_s = 30.0
    mem._rerank_top_k = 1
    mem._language = "en"

    coarse = [_mem(0, "a", 1.0), _mem(1, "b", 2.0)]
    with patch(
        "memory.baselines.lme_prebuilt.qwen3_vllm_score_documents",
        return_value=[0.5, 0.8],
    ):
        out = LmePrebuiltMemorySystem._apply_qwen3_vllm_rerank(mem, "q", coarse, retrieve_top_k=2)
    assert len(out) == 1
    assert out[0].memory_id == "1"


@pytest.mark.parametrize("lang,expect_zh", [("zh", True), ("en", False), (None, False)])
def test_rerank_instruction_switch(lang: str | None, expect_zh: bool) -> None:
    from utils.qwen3_reranker_vllm import DEFAULT_INSTRUCTION_ZH, rerank_instruction_for_language

    ins = rerank_instruction_for_language(lang)
    if expect_zh:
        assert ins == DEFAULT_INSTRUCTION_ZH
    else:
        assert "web search query" in ins.lower()
