"""Pydantic models and JSON-schema response format for atomic memory extraction."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


def _build_response_format(model_cls: type[BaseModel], schema_name: str) -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": model_cls.model_json_schema(),
            "strict": False,
        },
    }


# ---- LME / 标准格式：memories 为字符串列表 ----

class MemExtractResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memories: list[str] = Field(
        default_factory=list,
        description="Atomic, self-contained fact sentences; one string per fact.",
    )


MEM_EXTRACT_RESPONSE_FORMAT: Dict[str, Any] = _build_response_format(
    MemExtractResponse,
    "mem_extract",
)


# ---- LoCoMo 格式：memories 为对象列表（含 evidence 追踪） ----

class LoCoMoMemoryItem(BaseModel):
    """LoCoMo 单条 memory candidate。"""
    model_config = ConfigDict(extra="forbid")

    memory: str = Field(..., description="Self-contained atomic memory string.")
    evidence: List[str] = Field(
        default_factory=list,
        description="Supporting dialogue IDs (e.g. ['D2:1']).",
    )


class LoCoMoMemExtractResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memories: list[LoCoMoMemoryItem] = Field(
        default_factory=list,
        description="List of memory objects, each with memory text and evidence.",
    )


LOCOMO_MEM_EXTRACT_RESPONSE_FORMAT: Dict[str, Any] = _build_response_format(
    LoCoMoMemExtractResponse,
    "locomo_mem_extract",
)


# ---- 通用工具 ----

def normalize_extract_memories(
    parsed: Dict[str, Any],
    *,
    locomo_mode: bool = False,
) -> tuple[List[str], Optional[List[List[str]]]]:
    """将 LLM 返回的 parsed JSON 中的 memories 标准化为字符串列表 + 可选 evidence。

    Args:
        parsed: 已解析的 JSON dict。
        locomo_mode: True 时按 LoCoMo 格式（对象列表）解析；False 时按字符串列表。

    Returns:
        (memories_strings, evidence_lists) — ``evidence_lists`` 仅在
        ``locomo_mode=True`` 且解析成功时不为 ``None``，与 ``memories_strings``
        等长。
    """
    raw_mems = parsed.get("memories")
    if not isinstance(raw_mems, list):
        return [], None

    if not locomo_mode:
        # LME 格式：字符串列表
        strings: List[str] = []
        for m in raw_mems:
            if isinstance(m, str):
                s = m.strip()
                if s:
                    strings.append(s)
            elif isinstance(m, dict):
                # 宽容：dict 取 memory 字段
                s = str(m.get("memory", "")).strip()
                if s:
                    strings.append(s)
        return strings, None

    # LoCoMo 格式：对象列表
    strings = []
    evidence: List[List[str]] = []
    for m in raw_mems:
        if isinstance(m, dict):
            s = str(m.get("memory", "")).strip()
            ev = m.get("evidence", [])
            if isinstance(ev, list):
                ev = [str(e).strip() for e in ev if str(e).strip()]
            else:
                ev = []
        elif isinstance(m, str):
            s = m.strip()
            ev = []
        else:
            continue
        if s:
            strings.append(s)
            evidence.append(ev)

    return strings, evidence if evidence else None


__all__ = [
    "MemExtractResponse",
    "MEM_EXTRACT_RESPONSE_FORMAT",
    "LoCoMoMemExtractResponse",
    "LoCoMoMemoryItem",
    "LOCOMO_MEM_EXTRACT_RESPONSE_FORMAT",
    "normalize_extract_memories",
]
