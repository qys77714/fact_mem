"""Adapters wrapping fact_memory's sync OpenAI clients for graphiti's async interfaces.

These are private implementation details of ZepMemorySystem and should not be
imported directly outside this package.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

from pydantic import BaseModel

from graphiti_core.llm_client.client import DEFAULT_MAX_TOKENS, LLMClient
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.prompts.models import Message
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.cross_encoder.client import CrossEncoderClient

logger = logging.getLogger(__name__)


def _try_close_truncated_json(text: str) -> dict | None:
    """Attempt to recover a truncated JSON object by closing open brackets/braces.

    When the LLM hits its max_tokens limit, it may emit valid JSON up to the
    cut-off point.  We try increasingly aggressive truncation + bracket-closing
    to recover whatever was complete.  Returns None if recovery fails.
    """
    openers = {"{": "}", "[": "]"}
    closers = set(openers.values())
    stack: list[str] = []
    in_string = False
    escape_next = False
    last_complete = 0  # last position after a complete top-level value

    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in openers:
            stack.append(openers[ch])
        elif ch in closers:
            if stack and stack[-1] == ch:
                stack.pop()
                if not stack:
                    last_complete = i + 1

    # Try closing with suffix
    for cut in range(len(text), last_complete - 1, -1):
        candidate = text[:cut].rstrip().rstrip(",") + "".join(reversed(stack))
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue
        # Only try a few truncations to avoid O(n^2)
        if cut < len(text) - 200:
            break
    return None


class _SyncLLMAdapter(LLMClient):
    """Wraps fact_memory's sync OpenAIClient as graphiti's async LLMClient.

    The underlying sync client is the same one used by mem0/relation_decision
    (created by ``load_api_chat_completion(manager_model, async_=False)``).
    Sync calls are offloaded to a thread pool via ``asyncio.to_thread`` so that
    the graphiti async pipeline can drive them without blocking the event loop.
    """

    def __init__(self, sync_llm_client, model_name: str, max_tokens: int = 2048) -> None:
        config = LLMConfig(model=model_name, max_tokens=max_tokens)
        super().__init__(config, cache=False)
        self._sync_client = sync_llm_client
        self._model_name = model_name
        self._max_tokens = max_tokens
        # graphiti's extract_edges() requests up to 16k completion tokens.  vLLM enforces
        # prompt_tokens + max_tokens <= max_model_len, so with max_model_len=16384 a request
        # for 16k completion fails as soon as the prompt is non-empty → noisy halving retries.
        # Cap completion budget by default; set ZEP_MAX_COMPLETION_TOKENS (e.g. 8192 when
        # max_model_len=16384) to leave headroom for long graphiti prompts.
        self._completion_cap = int(os.getenv("ZEP_MAX_COMPLETION_TOKENS", "2048"))

    # ------------------------------------------------------------------
    # Schema → example helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _schema_to_example(schema: dict, defs: dict, depth: int = 0) -> Any:
        """Recursively build a minimal concrete example from a JSON Schema node."""
        if depth > 5:
            return None
        if "$ref" in schema:
            ref_name = schema["$ref"].split("/")[-1]
            return _SyncLLMAdapter._schema_to_example(defs.get(ref_name, {}), defs, depth + 1)
        schema_type = schema.get("type")
        if schema_type == "object":
            result: dict = {}
            for prop, prop_schema in schema.get("properties", {}).items():
                result[prop] = _SyncLLMAdapter._schema_to_example(prop_schema, defs, depth + 1)
            return result
        if schema_type == "array":
            item_schema = schema.get("items", {})
            return [_SyncLLMAdapter._schema_to_example(item_schema, defs, depth + 1)]
        if schema_type == "string":
            desc = schema.get("description", "")
            return desc[:30] if desc else "value"
        if schema_type == "integer":
            return 0
        if schema_type == "number":
            return 0.0
        if schema_type == "boolean":
            return True
        if "anyOf" in schema:
            for s in schema["anyOf"]:
                if s.get("type") != "null":
                    return _SyncLLMAdapter._schema_to_example(s, defs, depth + 1)
        return None

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Override base class to inject a concrete example instead of raw JSON Schema.

        graphiti's base generate_response appends the full JSON Schema definition
        to the last message, but open-weight models (e.g. gemma4-26B via vLLM)
        tend to echo the schema object rather than filling it in.  We replace it
        with a concrete example built from the schema so the model sees exactly
        what structure to produce.
        """
        if response_model is not None:
            schema = response_model.model_json_schema()
            defs = schema.get("$defs", {})
            example = self._schema_to_example(schema, defs)
            example_str = json.dumps(example, ensure_ascii=False)
            messages[-1].content += (
                f"\n\nRespond with a JSON object in this exact format (replace placeholder "
                f"values with real data):\n{example_str}"
            )
            response_model = None  # skip base class schema injection

        return await super().generate_response(
            messages,
            response_model=response_model,
            max_tokens=max_tokens,
            model_size=model_size,
        )

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        openai_messages = [{"role": m.role, "content": m.content} for m in messages]
        # Honour graphiti's max_tokens when set; else use ingest default.
        effective_max_tokens = max_tokens if max_tokens else self._max_tokens
        if self._completion_cap > 0:
            before = effective_max_tokens
            effective_max_tokens = min(effective_max_tokens, self._completion_cap)
            if before > effective_max_tokens:
                logger.debug(
                    "[ZepLLMAdapter] capping max_tokens %d -> %d "
                    "(ZEP_MAX_COMPLETION_TOKENS=%d)",
                    before,
                    effective_max_tokens,
                    self._completion_cap,
                )

        def _call():
            return self._sync_client.client.chat.completions.create(
                model=self._model_name,
                messages=openai_messages,
                max_tokens=effective_max_tokens,
                response_format={"type": "json_object"},
            )

        # Retry with halved max_tokens if model rejects due to context length.
        # When the input itself is too long (beyond context window even with minimum
        # output tokens), we exhaust retries and fall back to an empty dict so that
        # graphiti can continue processing the remaining episodes.
        for attempt in range(3):
            try:
                response = await asyncio.to_thread(_call)
                break
            except Exception as exc:
                msg = str(exc)
                if "context" in msg.lower() or "tokens" in msg.lower():
                    if attempt < 2:
                        effective_max_tokens = max(64, effective_max_tokens // 2)
                        logger.warning(
                            "[ZepLLMAdapter] Context too long, retrying with max_tokens=%d",
                            effective_max_tokens,
                        )
                    else:
                        logger.warning(
                            "[ZepLLMAdapter] Input prompt too long after %d retries, "
                            "returning empty dict. Error: %s",
                            attempt + 1,
                            exc,
                        )
                        return {}
                else:
                    logger.error("[ZepLLMAdapter] LLM call failed: %s", exc)
                    raise

        raw = (response.choices[0].message.content or "").strip()
        # Strip markdown code fences (```json ... ``` or ``` ... ```)
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        content = fence_match.group(1).strip() if fence_match else raw
        if not content:
            content = "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Response may be truncated (output token limit). Try to close open
            # brackets/braces so the partial JSON can be recovered.
            recovered = _try_close_truncated_json(content)
            if recovered is not None:
                return recovered
            logger.warning(
                "[ZepLLMAdapter] JSON parse failed, returning empty dict. Content: %.300s",
                raw,
            )
            return {}


class _SyncEmbedderAdapter(EmbedderClient):
    """Wraps fact_memory's sync embed_client as graphiti's async EmbedderClient.

    Uses the same ``embed_utils.embed_texts`` helper as mem0/relation_decision so
    that embeddings are byte-for-byte identical across methods.
    """

    def __init__(self, embed_client, embed_model_name: str) -> None:
        self._embed_client = embed_client
        self._model_name = embed_model_name

    async def create(
        self, input_data: str | list[str]
    ) -> list[float]:
        from utils.embed_utils import embed_texts

        inputs = [input_data] if isinstance(input_data, str) else list(input_data)
        embeddings = await asyncio.to_thread(
            embed_texts, self._embed_client, inputs, self._model_name
        )
        return embeddings[0].tolist()

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        from utils.embed_utils import embed_texts

        embeddings = await asyncio.to_thread(
            embed_texts, self._embed_client, input_data_list, self._model_name
        )
        return [e.tolist() for e in embeddings]


class _NoCrossEncoder(CrossEncoderClient):
    """Stub cross-encoder; ingest-only usage never calls rank()."""

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        return [(p, 1.0) for p in passages]
