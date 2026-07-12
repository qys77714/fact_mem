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



def build_lme_answer_fuse_prompt(
    current_memory: str,
    new_fact: str,
    relation: str,
    *,
    language: str,
    current_memory_time: str = "",
    new_fact_time: str = "",
) -> str:
    """增量融合答题记忆 C：current_memory + new_fact → 新 C（单条 user 消息）。

    按 ``relation`` 选用各标签专属融合模板（CON/OSN/NSO/EQV 融合策略不同）；
    未知 relation 回退到通用融合模板。

    ``current_memory_time`` / ``new_fact_time`` 为两侧事实的发生时间（会话日期），
    供 EQV 模板判断「同一事件被重复提及」与「事件多次发生」，从而谨慎计数。
    """
    lang = (language or "en").strip().lower()
    suffix = "zh" if lang.startswith("zh") else "en"
    rel = (relation or "").strip().upper()
    per_label = {
        "CON": f"lme_answer_fuse_con_{suffix}.jinja",
        "OSN": f"lme_answer_fuse_osn_{suffix}.jinja",
        "NSO": f"lme_answer_fuse_nso_{suffix}.jinja",
        "EQV": f"lme_answer_fuse_eqv_{suffix}.jinja",
    }
    t = per_label.get(rel, f"lme_answer_fuse_{suffix}.jinja")
    return render_prompt(
        t,
        current_memory=current_memory,
        new_fact=new_fact,
        relation=relation,
        current_memory_time=current_memory_time,
        new_fact_time=new_fact_time,
    )
