"""Pydantic models and response formats for Mem0 structured outputs."""
from __future__ import annotations

from typing import Annotated, Any, Dict, Literal, Optional, Union

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


class Mem0FactRetrievalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facts: list[str]


class Mem0AddOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    event: Literal["ADD"]


class Mem0UpdateOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    event: Literal["UPDATE"]
    old_memory: str


class Mem0DeleteOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    event: Literal["DELETE"]


class Mem0NoneOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    event: Literal["NONE"]
    text: Optional[str] = None


Mem0MemoryOperation = Annotated[
    Union[Mem0AddOperation, Mem0UpdateOperation, Mem0DeleteOperation, Mem0NoneOperation],
    Field(discriminator="event"),
]


class Mem0UpdateMemoryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory: list[Mem0MemoryOperation]


Mem0MemoryOperationNoDelete = Annotated[
    Union[Mem0AddOperation, Mem0UpdateOperation, Mem0NoneOperation],
    Field(discriminator="event"),
]


class Mem0UpdateMemoryResponseNoDelete(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory: list[Mem0MemoryOperationNoDelete]


FACT_RETRIEVAL_RESPONSE_FORMAT: Dict[str, Any] = _build_response_format(
    Mem0FactRetrievalResponse,
    "fact_retrieval",
)

UPDATE_MEMORY_RESPONSE_FORMAT: Dict[str, Any] = _build_response_format(
    Mem0UpdateMemoryResponse,
    "update_memory_actions",
)

UPDATE_MEMORY_RESPONSE_FORMAT_NO_DELETE: Dict[str, Any] = _build_response_format(
    Mem0UpdateMemoryResponseNoDelete,
    "update_memory_actions_no_delete",
)
