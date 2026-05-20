"""OpenAI-compatible chat completions with ``tools`` / ``tool_calls``."""

from __future__ import annotations

from typing import Any, List, Optional

from utils.openai_client import OpenAIClient, _merge_extra_body_qwen_thinking


def chat_completion_with_tool_calls(
    client: OpenAIClient,
    messages: List[dict],
    tools: List[dict],
    *,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    tool_choice: Any = "auto",
    extra_body: Optional[dict] = None,
) -> Any:
    """
    返回 ``completion.choices[0].message``。

    **单次 completion** 的 ``message.tool_calls`` 可为 **列表**，即一轮内多条工具调用（与 Agent 行为一致）；
    由调用方遍历解析。失败返回 ``None``。
    """
    kargs: dict = {
        "model": client.model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "tools": tools,
        "tool_choice": tool_choice,
    }
    if client.model_name in ("gpt-4o-mini",):
        completion = client.client.chat.completions.create(**kargs)
    else:
        kargs["extra_body"] = _merge_extra_body_qwen_thinking(extra_body, enable_thinking=False)
        completion = client.client.chat.completions.create(**kargs)

    return completion.choices[0].message
