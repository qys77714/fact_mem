"""
成对关系分类（relation_decision）所用模板：``RD_0_relation_classify.jinja``（默认），
包含任务说明 + OLD FACT / NEW FACT 占位，作为单条 user message 发送。
"""

from __future__ import annotations

from prompts import render_prompt


def build_relation_classification_prompt(
    m_old: str,
    m_new: str,
    *,
    language: str = "en",
    template_en: str | None = None,
    template_zh: str | None = None,
) -> str:
    """构建关系分类 user prompt（单条，含任务说明 + 待比较事实）。"""
    lang = (language or "en").strip().lower()
    if lang.startswith("zh"):
        t = (template_zh or "").strip() or "RD_0_relation_classify.jinja"
    else:
        t = (template_en or "").strip() or "RD_0_relation_classify.jinja"
    return render_prompt(t, m_old=m_old, m_new=m_new)



def build_lme_answer_fuse_prompt(
    current_memory: str,
    new_fact: str,
    relation: str,
    *,
    current_memory_time: str = "",
    new_fact_time: str = "",
) -> str:
    """增量融合答题记忆 C：current_memory + new_fact → 新 C（单条 user 消息）。

    按 ``relation`` 选用各标签专属融合模板（CON/OSN/NSO/EQV 融合策略不同）。

    ``current_memory_time`` / ``new_fact_time`` 为两侧事实的发生时间（会话日期），
    供 EQV 模板判断「同一事件被重复提及」与「事件多次发生」，从而谨慎计数。
    """
    rel = (relation or "").strip().upper()
    per_label = {
        "CON": "RD_1_fuse_CON.jinja",
        "OSN": "RD_1_fuse_OSN.jinja",
        "NSO": "RD_1_fuse_NSO.jinja",
        "EQV": "RD_1_fuse_EQV.jinja",
    }
    if rel not in per_label:
        raise ValueError(f"未知融合关系类型: {rel}，仅支持 {sorted(per_label.keys())}")
    t = per_label[rel]
    return render_prompt(
        t,
        current_memory=current_memory,
        new_fact=new_fact,
        relation=relation,
        current_memory_time=current_memory_time,
        new_fact_time=new_fact_time,
    )
