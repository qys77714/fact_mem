"""Prompt templates and response formats for the AMem memory method."""
from __future__ import annotations

from typing import Dict, Any

METADATA_ANALYSIS_PROMPT = """
你是一个对话记忆标注助手。请分析下面的对话内容，并输出 JSON：
1. context：一句话概述（场景 + 关键信息）。
2. keywords：3~8个关键词，按重要性排序。
3. tags：3~6个分类标签，兼顾主题与场景。
4. summary：不超过30字的核心摘要。

对话内容：
{transcript}
""".strip()

EVOLUTION_DECISION_PROMPT = """
你负责管理记忆库，需要判断新记忆与已存记忆的关系。

JSON 字段说明：
- should_evolve: 是否需要对新记忆或邻居进行调整。
- new_note_context: 若需要，给出更新后的新记忆 context（字符串），否则留空。
- new_note_tags: 若需要，给出新的标签列表，否则为空数组。
- neighbor_updates: 针对相关记忆的修改列表。

新记忆信息：
- Context: {context}
- Summary: {summary}
- Keywords: {keywords}
- Tags: {tags}

相关记忆：
{neighbor_summary}

请输出 JSON：
{{
  "should_evolve": true/false,
  "new_note_context": "如需更新新记忆 context，否则留空",
  "new_note_tags": ["如需更新新记忆标签，否则为空"],
  "neighbor_updates": [
    {{
      "memory_id": "需更新的记忆ID",
      "context": "邻居的新 context，可选",
      "tags": ["邻居的新标签，可选"]
    }}
  ]
}}
""".strip()

QUERY_KEYWORD_PROMPT = """
请根据下面的问题生成 3~6 个检索关键词，按重要性排序，只输出 JSON：
{question}
""".strip()


def build_metadata_prompt(transcript: str) -> str:
    return METADATA_ANALYSIS_PROMPT.format(transcript=transcript.strip())


def build_evolution_prompt(
    context: str,
    summary: str,
    keywords: str,
    tags: str,
    neighbor_summary: str,
) -> str:
    return EVOLUTION_DECISION_PROMPT.format(
        context=context.strip(),
        summary=summary.strip(),
        keywords=keywords.strip(),
        tags=tags.strip(),
        neighbor_summary=neighbor_summary.strip() or "（无相关记忆）",
    )


def build_query_prompt(question: str) -> str:
    return QUERY_KEYWORD_PROMPT.format(question=question.strip())


METADATA_RESPONSE_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "amem_metadata",
        "schema": {
            "type": "object",
            "properties": {
                "context": {"type": "string"},
                "keywords": {"type": "array", "items": {"type": "string"}},
                "tags": {"type": "array", "items": {"type": "string"}},
                "summary": {"type": "string"},
            },
            "required": ["context", "keywords", "tags"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

EVOLUTION_RESPONSE_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "amem_evolution",
        "schema": {
            "type": "object",
            "properties": {
                "should_evolve": {"type": "boolean"},
                "new_note_context": {"type": "string"},
                "new_note_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "neighbor_updates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "memory_id": {"type": "string"},
                            "context": {"type": "string"},
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["memory_id"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["should_evolve"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

QUERY_RESPONSE_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "amem_query_keywords",
        "schema": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["keywords"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}