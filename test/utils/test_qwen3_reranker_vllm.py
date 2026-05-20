"""Unit tests for Qwen3-Reranker vLLM ``/score`` prompt fragments."""

from __future__ import annotations

from utils.qwen3_reranker_vllm import (
    QWEN3_RERANKER_SCORE_PREFIX,
    QWEN3_RERANKER_SCORE_SUFFIX,
    build_qwen3_reranker_vllm_score_texts,
    format_qwen3_reranker_score_text_1,
    format_qwen3_reranker_score_text_2,
)


def test_prefix_matches_upstream_literal() -> None:
    expected = (
        '<|im_start|>system\n'
        'Judge whether the Document meets the requirements based on the Query and the '
        'Instruct provided. Note that the answer can only be "yes" or "no".'
        "<|im_end|>\n"
        "<|im_start|>user\n"
    )
    assert QWEN3_RERANKER_SCORE_PREFIX == expected


def test_suffix_matches_upstream_literal() -> None:
    assert QWEN3_RERANKER_SCORE_SUFFIX == (
        "<|im_end|>\n"
        "<|im_start|>assistant\n \n\n \n\n"
    )


def test_query_side_contains_instruction_and_query_lines() -> None:
    s = format_qwen3_reranker_score_text_1("What is X?")
    assert s.startswith(QWEN3_RERANKER_SCORE_PREFIX)
    assert "Given a web search query" in s
    assert "\n: What is X?\n" in s


def test_document_side_leading_colon_and_suffix() -> None:
    s = format_qwen3_reranker_score_text_2("doc body")
    assert s.startswith(": doc body")
    assert s.endswith(QWEN3_RERANKER_SCORE_SUFFIX)


def test_build_one_query_three_docs_lengths() -> None:
    t1, t2 = build_qwen3_reranker_vllm_score_texts("q", ["a", "b", "c"])
    assert "q" in t1
    assert len(t2) == 3
    assert all(x.startswith(":") for x in t2)
