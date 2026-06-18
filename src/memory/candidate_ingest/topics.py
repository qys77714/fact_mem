"""Predefined, benchmark-agnostic topic taxonomy for candidate-fact aggregation.

主题在抽取后由独立打标 pass 赋给每条 atomic fact（平行数组 ``candidate_topics``，
与 ``cas_update_rules`` 同样的透传模式）。灌库期按「同 episode 同 topic」把互不冲突、
侧面正交的事实融进同一条答题记忆 C（profile），让 Agg「列出关于 X 的一切」类问题
一次检索命中即可带出全部 slot。

枚举刻意按「人 / 团队 / 工程」等**通用生活与工作维度**切分，不含任何 benchmark
专有的 slot 名，避免「overfit 到测试集」的质疑。``misc`` 为兜底，不参与聚合。
"""

from __future__ import annotations

from typing import Dict, List

# slug -> 一句话说明（注入打标 prompt，帮助 LLM 选类）
TOPIC_TAXONOMY: Dict[str, str] = {
    "personal_interests": "Hobbies, sports, games, clubs, entertainment, and leisure activities.",
    "personal_logistics": "Commute, travel plans, family events, residence, and personal schedule.",
    "team_process": "On-call rotation, meetings, stand-ups, alert channels, and team rituals.",
    "engineering_standards": "Code review, testing, coverage, design system, branching, and release policy.",
    "project_resources": "Documentation, runbooks, changelogs, onboarding guides, and reference links.",
    "work_profile": "Job role, team, reporting lines, skills, and professional background.",
    "health_wellbeing": "Health conditions, diet, allergies, sleep, and fitness routines.",
    "misc": "Anything that does not clearly fit the categories above (not aggregated).",
}

# 兜底类：打标命中它时不建/不并 profile
MISC_TOPIC = "misc"

VALID_TOPICS: frozenset[str] = frozenset(TOPIC_TAXONOMY)

# 参与聚合的主题（排除兜底）
AGGREGATABLE_TOPICS: frozenset[str] = frozenset(t for t in TOPIC_TAXONOMY if t != MISC_TOPIC)


def normalize_topic(raw: object) -> str:
    """把任意返回规整成合法 slug；无法识别一律落 ``misc``（绝不抛异常，保证灌库稳健）。"""
    s = str(raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    return s if s in VALID_TOPICS else MISC_TOPIC


def topic_menu_lines() -> List[str]:
    """``- slug: description`` 形式的菜单行，供打标 prompt 注入。"""
    return [f"- {slug}: {desc}" for slug, desc in TOPIC_TAXONOMY.items()]


__all__ = [
    "TOPIC_TAXONOMY",
    "MISC_TOPIC",
    "VALID_TOPICS",
    "AGGREGATABLE_TOPICS",
    "normalize_topic",
    "topic_menu_lines",
]
