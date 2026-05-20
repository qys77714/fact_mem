"""Utility (U): LLM score via same ``get_response_chat`` as relation_decision."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Protocol


class _ChatClient(Protocol):
    def get_response_chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Optional[str]: ...


def _parse_json_loose(text: str) -> Optional[Dict[str, Any]]:
    if not text or not str(text).strip():
        return None
    t = str(text).strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", t)
    if fence:
        t = fence.group(1).strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    i = t.find("{")
    j = t.rfind("}")
    if i != -1 and j > i:
        try:
            obj = json.loads(t[i : j + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def utility_llm_score(
    llm_client: _ChatClient,
    *,
    candidate: str,
    conversation_history: List[Dict[str, Any]],
    max_new_tokens: int = 256,
    language: str = "en",
) -> float:
    """Return utility in [0,1] from structured JSON ``{\"utility\": <float>}``."""
    hist_lines: List[str] = []
    for turn in conversation_history[-40:]:
        hist_lines.append(str(turn.get("text") or ""))
    hist_blob = "\n".join(hist_lines).strip() or "(empty context)"
    cand = (candidate or "").strip()
    if not cand:
        return 0.0
    use_zh = (language or "en").lower().startswith("zh")
    if use_zh:
        user_msg = (
            "你将评估一条「待写入长期记忆」的陈述，在未来对话/任务中可能的作用。\n"
            "对话上下文（节选）：\n"
            f"{hist_blob}\n\n"
            "候选记忆：\n"
            f"{cand}\n\n"
            "仅输出一个 JSON 对象，键 utility，值为 0 到 1 的浮点数（越高表示越值得持久化）。"
            "示例：{\"utility\": 0.72}\n"
        )
    else:
        user_msg = (
            "Rate how useful the following candidate memory would be for future tasks "
            "in this conversation context.\n\n"
            "Context (excerpt):\n"
            f"{hist_blob}\n\n"
            "Candidate memory:\n"
            f"{cand}\n\n"
            "Reply with JSON only: {\"utility\": <float in [0,1]>}"
        )
    messages = [
        {"role": "system", "content": "You output compact JSON only."},
        {"role": "user", "content": user_msg},
    ]
    raw = llm_client.get_response_chat(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        verbose=False,
    )
    payload = _parse_json_loose("" if raw is None else str(raw))
    if not payload:
        return 0.5
    u = payload.get("utility")
    if isinstance(u, (int, float)):
        return float(max(0.0, min(1.0, float(u))))
    if isinstance(u, str):
        try:
            return float(max(0.0, min(1.0, float(u.strip()))))
        except ValueError:
            pass
    return 0.5
