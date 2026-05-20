"""Pydantic models and JSON-schema response format for atomic memory extraction."""

from __future__ import annotations

from typing import Any, Dict

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

__all__ = [
    "MemExtractResponse",
    "MEM_EXTRACT_RESPONSE_FORMAT",
]
