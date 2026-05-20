"""
Qwen3-Reranker inputs for vLLM ``POST .../score`` (``ScoreRequest``).

For cross-encoder / ``llm as reranker`` models, vLLM concatenates ``text_1 + text_2``
before tokenization; it does **not** apply ``--chat-template`` to these fields. Use the
same ``query_template`` / ``document_template`` as upstream
``examples/offline_inference/qwen3_reranker.py`` (v0.10.1).
"""

from __future__ import annotations

import json
from typing import List, Sequence
from urllib.request import Request, urlopen

# Mirrors vLLM v0.10.1 examples/offline_inference/qwen3_reranker.py
DEFAULT_INSTRUCTION_EN = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
DEFAULT_INSTRUCTION_ZH = "给定一个查询，检索能够回答该查询的相关文档段落"

QWEN3_RERANKER_SCORE_PREFIX = (
    '<|im_start|>system\n'
    'Judge whether the Document meets the requirements based on the Query and the '
    'Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n"
    "<|im_start|>user\n"
)

QWEN3_RERANKER_SCORE_SUFFIX = (
    "<|im_end|>\n"
    "<|im_start|>assistant\n \n\n \n\n"
)

_QUERY_TEMPLATE = "{prefix}: {instruction}\n: {query}\n"
_DOCUMENT_TEMPLATE = ": {doc}{suffix}"


def format_qwen3_reranker_score_text_1(
    query: str,
    *,
    instruction: str = DEFAULT_INSTRUCTION_EN,
) -> str:
    """Query-side fragment for one ``(query, document)`` score (becomes ``text_1`` or part of a list)."""
    return _QUERY_TEMPLATE.format(
        prefix=QWEN3_RERANKER_SCORE_PREFIX,
        instruction=instruction,
        query=query,
    )


def format_qwen3_reranker_score_text_2(document: str) -> str:
    """Document-side fragment (becomes ``text_2`` for that pair)."""
    return _DOCUMENT_TEMPLATE.format(doc=document, suffix=QWEN3_RERANKER_SCORE_SUFFIX)


def build_qwen3_reranker_vllm_score_texts(
    query: str,
    documents: Sequence[str],
    *,
    instruction: str = DEFAULT_INSTRUCTION_EN,
) -> tuple[str, list[str]]:
    """
    Build ``(text_1, text_2)`` for one query vs many documents.

    vLLM accepts one ``text_1`` string and a list ``text_2`` and broadcasts ``text_1``
    when lengths are 1:N.
    """
    t1 = format_qwen3_reranker_score_text_1(query, instruction=instruction)
    t2 = [format_qwen3_reranker_score_text_2(d) for d in documents]
    return t1, t2


def qwen3_vllm_score_documents(
    *,
    base_url: str,
    api_key: str,
    model: str,
    query: str,
    documents: Sequence[str],
    timeout_s: float = 120.0,
    instruction: str = DEFAULT_INSTRUCTION_EN,
) -> List[float]:
    """
    Call vLLM OpenAI-compatible ``POST {base_url}/score`` (one ``text_1``, many ``text_2``).

    ``base_url`` should look like ``http://host:7114/v1/`` (trailing slash optional).
    """
    if not documents:
        return []
    text_1, text_2 = build_qwen3_reranker_vllm_score_texts(
        query, documents, instruction=instruction
    )
    url = base_url.rstrip("/") + "/score"
    payload = {"model": model, "text_1": text_1, "text_2": text_2}
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    items = sorted(data["data"], key=lambda x: int(x["index"]))
    return [float(x["score"]) for x in items]


def rerank_instruction_for_language(language: str | None) -> str:
    lang = (language or "en").strip().lower()
    if lang.startswith("zh"):
        return DEFAULT_INSTRUCTION_ZH
    return DEFAULT_INSTRUCTION_EN
