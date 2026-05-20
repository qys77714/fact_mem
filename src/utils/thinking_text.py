"""Helpers for Qwen / vLLM assistant text where chain-of-thought may live inside XML-style tags."""

from __future__ import annotations

import re

# vLLM may place CoT inside the assistant message string while reasoning_content is empty.
_EMBEDDED_THINK_RE = re.compile(
    r"<\s*(?:redacted_thinking|redacted_reasoning|think)\s*>(.*?)<\s*/\s*(?:redacted_thinking|redacted_reasoning|think)\s*>",
    re.IGNORECASE | re.DOTALL,
)


def split_embedded_thinking(content: str) -> tuple[str, str]:
    """Split *content* when reasoning is wrapped in think / redacted_thinking tags.

    Returns *(thinking, remainder)*: *thinking* is concatenated block bodies; *remainder*
    is text after the last closing tag. If no block matches, returns ("", stripped content).
    """
    if not isinstance(content, str):
        return "", ""
    s = content.strip()
    if not s:
        return "", ""
    parts: list[str] = []
    pos = 0
    while True:
        m = _EMBEDDED_THINK_RE.search(s, pos)
        if not m:
            break
        inner = (m.group(1) or "").strip()
        if inner:
            parts.append(inner)
        pos = m.end()
    if not parts:
        return "", s
    return "\n\n".join(parts), s[pos:].strip()


__all__ = ["split_embedded_thinking"]
