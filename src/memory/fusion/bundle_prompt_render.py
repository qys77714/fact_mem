"""Fusion user prompt text (no LocalFaiss / faiss import). Used by HTML build scripts and bundle_fusion.

Default templates: ``fuse_memory_bundle_{en,zh}_v3.jinja``. Optional ``bundle_template_en`` /
``bundle_template_zh`` override the filename for that language (e.g. ``fuse_memory_bundle_en_v3.jinja``).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Sequence, Tuple

from memory.base import RetrievedMemory
from prompts import render_prompt


def _is_primary_meta(metadata: Dict[str, Any]) -> bool:
    return (metadata or {}).get("memory_role") != "evidence"


def _member_tag(mem: RetrievedMemory) -> str:
    meta = mem.metadata or {}
    if _is_primary_meta(meta):
        return "primary"
    return str(meta.get("edge") or meta.get("memory_role") or "evidence")


def _zh_parent_desc(parent_idx: int | None) -> str:
    """指代直接父节点，带上编号。"""
    if parent_idx is None:
        return "父节点"
    return f"父节点（第 {parent_idx} 条）"


@lru_cache(maxsize=32)
def _zh_edge_label_lines_for(template: str) -> tuple[str, str, str, str]:
    rendered = render_prompt(
        template,
        equivalent_label="【等价】",
        attached_label="【附属】",
        superseded_label="【被更新】",
        parent_desc="父节点",
    )
    lines = [ln.strip() for ln in rendered.splitlines() if ln.strip()]
    if len(lines) != 4:
        raise ValueError("edge label template zh must contain exactly 4 non-empty lines")
    return tuple(lines)  # type: ignore[return-value]


def _zh_edge_line_prefix(tag: str, *, parent_idx: int | None, edge_labels_template_zh: str) -> str:
    primary_line, eqv_t, attach_t, update_t = _zh_edge_label_lines_for(edge_labels_template_zh)
    t = (tag or "").strip().upper()
    if t == "PRIMARY":
        return primary_line
    p = _zh_parent_desc(parent_idx)
    if t == "EQUIV":
        return eqv_t.replace("父节点", p)
    if t == "ATTACH":
        return attach_t.replace("父节点", p)
    if t == "UPDATE":
        return update_t.replace("父节点", p)
    return f"【{tag or 'unknown'}】"


def _en_parent_desc(parent_idx: int | None) -> str:
    """Refer to the immediate parent by row number."""
    if parent_idx is None:
        return "parent node"
    return f"parent node (row {parent_idx})"


@lru_cache(maxsize=32)
def _en_edge_label_lines_for(template: str) -> tuple[str, str, str, str]:
    rendered = render_prompt(
        template,
        equivalent_label="[equivalent]",
        attached_label="[attached]",
        superseded_label="[superseded]",
        parent_desc="parent node",
    )
    lines = [ln.strip() for ln in rendered.splitlines() if ln.strip()]
    if len(lines) != 4:
        raise ValueError("edge label template en must contain exactly 4 non-empty lines")
    return tuple(lines)  # type: ignore[return-value]


def _en_edge_line_prefix(tag: str, *, parent_idx: int | None, edge_labels_template_en: str) -> str:
    primary_line, eqv_t, attach_t, update_t = _en_edge_label_lines_for(edge_labels_template_en)
    t = (tag or "").strip().upper()
    if t == "PRIMARY":
        return primary_line
    p = _en_parent_desc(parent_idx)
    if t == "EQUIV":
        return eqv_t.replace("parent node", p)
    if t == "ATTACH":
        return attach_t.replace("parent node", p)
    if t == "UPDATE":
        return update_t.replace("parent node", p)
    return f"[{tag}]"


def _resolve_parent_index(
    mem: RetrievedMemory,
    id_to_idx: Dict[str, int],
) -> Tuple[int | None, str | None]:
    """Return (1-based parent index in this bundle, or None) and optional unresolved parent id snippet."""
    meta = mem.metadata or {}
    if _is_primary_meta(meta):
        return None, None
    pid = meta.get("parent_primary")
    if not pid or not isinstance(pid, str):
        return None, None
    if pid in id_to_idx:
        return id_to_idx[pid], None
    short = pid if len(pid) <= 16 else f"{pid[:12]}…"
    return None, short


def _tree_edge_summary(members: Sequence[RetrievedMemory], id_to_idx: Dict[str, int], *, language: str) -> str:
    lang = (language or "en").strip().lower()
    edges: List[Tuple[int, int]] = []
    for i, m in enumerate(members, start=1):
        meta = m.metadata or {}
        if _is_primary_meta(meta):
            continue
        pid = meta.get("parent_primary")
        if isinstance(pid, str) and pid in id_to_idx:
            edges.append((i, id_to_idx[pid]))
    if not edges:
        return ""
    parts = [f"{c}→{p}" for c, p in edges]
    if lang.startswith("zh"):
        return f"【包内树】父子边（子→父）：{', '.join(parts)}。"
    return f"[bundle tree] Parent–child edges (child→parent): {', '.join(parts)}."


def _format_bundle_for_prompt(
    members: Sequence[RetrievedMemory],
    *,
    language: str,
    edge_labels_en: str = "fuse_memory_bundle_edge_labels_en_v2.jinja",
    edge_labels_zh: str = "fuse_memory_bundle_edge_labels_zh_v2.jinja",
) -> str:
    lang = (language or "en").strip().lower()
    id_to_idx: Dict[str, int] = {m.memory_id: i for i, m in enumerate(members, start=1)}
    lines: list[str] = []
    summary = _tree_edge_summary(members, id_to_idx, language=lang)
    if summary:
        lines.append(summary)
        lines.append("")
    for i, m in enumerate(members, start=1):
        tag = _member_tag(m)
        time_s = (m.time or "").strip()
        body = (m.text or "").strip()
        meta = m.metadata or {}
        depth_s = meta.get("evidence_depth")
        parent_idx, unresolved = _resolve_parent_index(m, id_to_idx)
        if tag == "primary":
            prefix = (
                _zh_edge_line_prefix("PRIMARY", parent_idx=None, edge_labels_template_zh=edge_labels_zh)
                if lang.startswith("zh")
                else _en_edge_line_prefix("PRIMARY", parent_idx=None, edge_labels_template_en=edge_labels_en)
            )
            extra_lines: list[str] = []
        else:
            prefix = (
                _zh_edge_line_prefix(tag, parent_idx=parent_idx, edge_labels_template_zh=edge_labels_zh)
                if lang.startswith("zh")
                else _en_edge_line_prefix(tag, parent_idx=parent_idx, edge_labels_template_en=edge_labels_en)
            )
            extra_lines = []
            if lang.startswith("zh"):
                if isinstance(depth_s, int):
                    extra_lines.append(f"   深度（相对根）：{depth_s}")
                if parent_idx is None and unresolved:
                    extra_lines.append(f"   父：不在本包内或未解析（parent_primary≈{unresolved}）")
            else:
                if isinstance(depth_s, int):
                    extra_lines.append(f"   depth (from root): {depth_s}")
                if parent_idx is None and unresolved:
                    extra_lines.append(f"   parent: not in bundle / unresolved (parent_primary≈{unresolved})")

        if lang.startswith("zh"):
            t_disp = time_s if time_s else "未知时间"
            block = [f"{i}. {prefix}", *extra_lines, f"   记录时间：{t_disp}", f"   内容：{body}"]
            lines.append("\n".join(block))
        else:
            t_disp = time_s if time_s else "unknown time"
            block = [f"{i}. {prefix}", *extra_lines, f"   recorded time: {t_disp}", f"   text: {body}"]
            lines.append("\n".join(block))
    return "\n".join(lines)


def render_fusion_user_prompt(
    members: Sequence[RetrievedMemory],
    *,
    language: str = "en",
    bundle_template_en: str | None = None,
    bundle_template_zh: str | None = None,
    edge_labels_en: str | None = None,
    edge_labels_zh: str | None = None,
) -> str:
    """User message sent to the LLM for bundle fusion (same as ``_fuse_bundle_with_llm`` input)."""
    lang = (language or "en").strip().lower()
    el_en = (edge_labels_en or "").strip() or "fuse_memory_bundle_edge_labels_en_v2.jinja"
    el_zh = (edge_labels_zh or "").strip() or "fuse_memory_bundle_edge_labels_zh_v2.jinja"
    if lang.startswith("zh"):
        template = bundle_template_zh or "fuse_memory_bundle_zh_v3.jinja"
    else:
        template = bundle_template_en or "fuse_memory_bundle_en_v3.jinja"
    return render_prompt(
        template,
        bundle_lines=_format_bundle_for_prompt(
            members, language=lang, edge_labels_en=el_en, edge_labels_zh=el_zh
        ),
        num_facts=len(members),
    ).strip()


__all__ = ["render_fusion_user_prompt", "_is_primary_meta"]
