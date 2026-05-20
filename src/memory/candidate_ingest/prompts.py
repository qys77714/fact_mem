"""
成对关系分类（relation_decision）所用模板，均在 ``src/prompts/templates/``：

- **System（中文）**：``lme_relation_classification_system_zh_v2.jinja``（默认）
- **System（非中文）**：``lme_relation_classification_system_en_v2.jinja``（默认）
- **User**：``lme_relation_classification_user.jinja``

v2 在 v1 基础上补充了每类标签的 one-liner 示例（EQV/NSO/OSN/CON/IND），
其余决策流程、标签语义与输出格式与 v1 保持一致。历史版本
``lme_relation_classification_system_{en,zh}.jinja`` 仍保留用于回归对照。
"""

from __future__ import annotations

from prompts import render_prompt


def lme_relation_system_prompt_for_language(
    language: str,
    *,
    template_en: str | None = None,
    template_zh: str | None = None,
) -> str:
    lang = (language or "en").strip().lower()
    if lang.startswith("zh"):
        t = (template_zh or "").strip() or "lme_relation_classification_system_zh_v2.jinja"
        return render_prompt(t)
    t = (template_en or "").strip() or "lme_relation_classification_system_en_v2.jinja"
    return render_prompt(t)


def build_lme_relation_classification_user_prompt(
    m_old: str, m_new: str, *, template: str | None = None
) -> str:
    t = (template or "").strip() or "lme_relation_classification_user.jinja"
    return render_prompt(t, m_old=m_old, m_new=m_new)
