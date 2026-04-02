"""RelMem prompts live under src/prompts/templates/; this module wires language + render."""
from __future__ import annotations

from prompts import render_prompt


def build_relation_classification_user_prompt(m_old: str, m_new: str) -> str:
    return render_prompt("relmem_relation_classification_user.jinja", m_old=m_old, m_new=m_new)


def relation_system_prompt_for_language(language: str) -> str:
    lang = (language or "en").strip().lower()
    if lang.startswith("zh"):
        return render_prompt("relmem_relation_classification_system_zh.jinja")
    return render_prompt("relmem_relation_classification_system_en.jinja")
