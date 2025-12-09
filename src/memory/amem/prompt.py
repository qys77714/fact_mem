"""Prompt templates and response formats for the AMem memory method."""
from __future__ import annotations

from typing import Dict, Any


def build_metadata_prompt(transcript: str, language: str) -> str:
    METADATA_ANALYSIS_PROMPT = METADATA_ANALYSIS_PROMPT_ZH if language == "zh" else METADATA_ANALYSIS_PROMPT_EN
    return METADATA_ANALYSIS_PROMPT.format(transcript=transcript.strip())


def build_evolution_prompt(
    context: str,
    summary: str,
    keywords: str,
    tags: str,
    neighbor_summary: str,
    language: str,
) -> str:
    EVOLUTION_DECISION_PROMPT = EVOLUTION_DECISION_PROMPT_ZH if language == "zh" else EVOLUTION_DECISION_PROMPT_EN
    return EVOLUTION_DECISION_PROMPT.format(
        context=context.strip(),
        summary=summary.strip(),
        keywords=keywords.strip(),
        tags=tags.strip(),
        neighbor_summary=neighbor_summary.strip() or "（无相关记忆）",
    )


def build_query_prompt(question: str, language: str) -> str:
    QUERY_KEYWORD_PROMPT = QUERY_KEYWORD_PROMPT_ZH if language == "zh" else QUERY_KEYWORD_PROMPT_EN
    return QUERY_KEYWORD_PROMPT.format(question=question.strip())


METADATA_ANALYSIS_PROMPT_ZH = """
你是一个对话记忆标注助手。请分析下面的对话内容，并输出 JSON：
1. context：一句话概述（场景 + 关键信息）。
2. keywords：3~8个关键词，按重要性排序。
3. tags：3~6个分类标签，兼顾主题与场景。
4. summary：不超过30字的核心摘要。

对话内容：
{transcript}
""".strip()

EVOLUTION_DECISION_PROMPT_ZH = """
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

QUERY_KEYWORD_PROMPT_ZH = """
请根据下面的问题生成 3~6 个检索关键词，按重要性排序，只输出 JSON：
{question}
""".strip()


METADATA_ANALYSIS_PROMPT_EN = """
You are a dialogue memory annotation assistant. Please analyze the following dialogue content and output JSON:
1. context: A one-sentence overview (scene + key information).
2. keywords: 3~8 keywords, sorted by importance.
3. tags: 3~6 classification tags, covering both themes and scenarios.
4. summary: A core summary not exceeding 30 Chinese characters.

Dialogue Content:
{transcript}
""".strip()

EVOLUTION_DECISION_PROMPT_EN = """
You are responsible for managing the memory base and need to determine the relationship between new memories and stored memories.

JSON Field Descriptions:
- should_evolve: Whether adjustments to the new memory or related memories are needed.
- new_note_context: If needed, provide the updated context of the new memory (string); otherwise, leave blank.
- new_note_tags: If needed, provide the updated tag list; otherwise, an empty array.
- neighbor_updates: A list of modifications for related memories.

New Memory Information:
- Context: {context}
- Summary: {summary}
- Keywords: {keywords}
- Tags: {tags}

Related Memories:
{neighbor_summary}

Please output JSON:
{{
  "should_evolve": true/false,
  "new_note_context": "Updated context for the new memory if needed, otherwise leave blank",
  "new_note_tags": ["Updated tags for the new memory if needed, otherwise empty"],
  "neighbor_updates": [
    {{
      "memory_id": "Memory ID to be updated",
      "context": "Updated context for the related memory, optional",
      "tags": ["Updated tags for the related memory, optional"]
    }}
  ]
}}
""".strip()

QUERY_KEYWORD_PROMPT_EN = """
Please generate 3~6 retrieval keywords based on the following question, sorted by importance, and output only JSON:
{question}
""".strip()


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
            "required": ["context", "keywords", "tags", "summary"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

EVOLUTION_RESPONSE_FORMAT = {
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
                        # 关键修复：包含所有 properties 字段
                        "required": ["memory_id", "context", "tags"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["should_evolve", "new_note_context", "new_note_tags", "neighbor_updates"],
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