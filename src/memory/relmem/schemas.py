"""Structured output for RelMem pairwise relation classification."""
from __future__ import annotations

from typing import Any, Dict, Literal

from pydantic import BaseModel, ConfigDict

from memory.mem0.schemas import _build_response_format


class RelMemRelationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relation: Literal["IND", "EQV", "NSO", "OSN", "CON"]


RELATION_CLASSIFICATION_RESPONSE_FORMAT: Dict[str, Any] = _build_response_format(
    RelMemRelationResponse,
    "relmem_relation",
)
