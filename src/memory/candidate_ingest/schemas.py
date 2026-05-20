"""Structured output for LME pairwise relation classification."""
from __future__ import annotations

from typing import Any, Dict, Literal

from pydantic import BaseModel, ConfigDict

from memory.mem0.schemas import _build_response_format


class LmeRelationClassificationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relation: Literal["IND", "EQV", "NSO", "OSN", "CON"]


LME_RELATION_CLASSIFICATION_RESPONSE_FORMAT: Dict[str, Any] = _build_response_format(
    LmeRelationClassificationResponse,
    "lme_relation_classification",
)
