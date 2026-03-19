"""Pydantic models and response formats for AMem structured outputs."""
from __future__ import annotations

from typing import Dict, Any

from pydantic import BaseModel, ConfigDict


def _build_response_format(model_cls: type[BaseModel], schema_name: str) -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": model_cls.model_json_schema(),
            "strict": True,
        },
    }


class AMemMetadataResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context: str
    keywords: list[str]
    tags: list[str]
    summary: str


class AMemNeighborUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str
    context: str
    tags: list[str]


class AMemEvolutionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    should_evolve: bool
    new_note_context: str
    new_note_tags: list[str]
    neighbor_updates: list[AMemNeighborUpdate]


class AMemQueryKeywordsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keywords: list[str]


METADATA_RESPONSE_FORMAT: Dict[str, Any] = _build_response_format(
    AMemMetadataResponse,
    "amem_metadata",
)

EVOLUTION_RESPONSE_FORMAT: Dict[str, Any] = _build_response_format(
    AMemEvolutionResponse,
    "amem_evolution",
)

QUERY_RESPONSE_FORMAT: Dict[str, Any] = _build_response_format(
    AMemQueryKeywordsResponse,
    "amem_query_keywords",
)
