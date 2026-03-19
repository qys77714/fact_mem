"""Prompt templates and response format exports for the AMem memory method."""
from __future__ import annotations

from prompts import render_prompt

from .schemas import (
    METADATA_RESPONSE_FORMAT,
    EVOLUTION_RESPONSE_FORMAT,
    QUERY_RESPONSE_FORMAT,
)


def build_metadata_prompt(transcript: str, language: str) -> str:
    template = "amem_metadata_zh.jinja" if language == "zh" else "amem_metadata_en.jinja"
    return render_prompt(template, transcript=transcript.strip())


def build_evolution_prompt(
    context: str,
    summary: str,
    keywords: str,
    tags: str,
    neighbor_summary: str,
    language: str,
) -> str:
    template = "amem_evolution_zh.jinja" if language == "zh" else "amem_evolution_en.jinja"
    return render_prompt(
        template,
        context=context.strip(),
        summary=summary.strip(),
        keywords=keywords.strip(),
        tags=tags.strip(),
        neighbor_summary=neighbor_summary.strip() or "（无相关记忆）",
    )


def build_query_prompt(question: str, language: str) -> str:
    template = "amem_query_zh.jinja" if language == "zh" else "amem_query_en.jinja"
    return render_prompt(template, question=question.strip())


__all__ = [
    "build_metadata_prompt",
    "build_evolution_prompt",
    "build_query_prompt",
    "METADATA_RESPONSE_FORMAT",
    "EVOLUTION_RESPONSE_FORMAT",
    "QUERY_RESPONSE_FORMAT",
]
