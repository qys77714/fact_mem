#!/usr/bin/env python3
"""
Build HTML report for MEME Cas questions: per-golden-memory similarity rankings
(within the same episode, up to question cutoff).

Cas golden memories = full cascade dependency chain (see meme_nofiller.json):
  - Walk dependency_edges_used backwards from question.entity for question.hop steps
  - e.g. medication hop=1 → [health_condition, medication]
  - e.g. fitness_facility hop=2 → [health_condition, exercise_routine, fitness_facility]
  - All gold_facts whose entity is on that chain (up to cutoff) are golden for the question

For EACH golden memory, compute similarity only vs memories ingested BEFORE it
(same order as pipeline ingest: chunk_index asc, then fact order within chunk).
Similarity is within-episode only; top-K per golden memory.
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from openai import OpenAI
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.embed_utils import embed_texts  # noqa: E402
from utils.env import load_env  # noqa: E402


@dataclass
class MemoryItem:
    text: str
    source: str  # evidence_gold_facts | filler
    session_index: int
    chunk_index: int
    order_index: int  # global ingest order (0-based), aligned with apply_candidate_episode_json
    entity: Optional[str] = None
    session_date: str = ""


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.empty((a.shape[0], b.shape[0]), dtype=np.float32)
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    a_norm = np.linalg.norm(a64, axis=1, keepdims=True) + 1e-12
    b_norm = np.linalg.norm(b64, axis=1, keepdims=True) + 1e-12
    return ((a64 / a_norm) @ (b64 / b_norm).T).astype(np.float32)


def _build_entity_map(episode: Dict[str, Any]) -> Dict[str, str]:
    """Map fact_text -> entity from dataset gold_facts."""
    out: Dict[str, str] = {}
    for sess in episode.get("sessions") or []:
        if sess.get("type") != "evidence":
            continue
        for gf in sess.get("gold_facts") or []:
            text = str(gf.get("fact_text") or gf.get("original_seed") or "").strip()
            entity = str(gf.get("entity") or "").strip()
            if text and entity:
                out[text] = entity
    return out


def _load_candidate_chunks(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("chunks") or [])


def _memories_up_to_cutoff(
    chunks: List[Dict[str, Any]],
    entity_map: Dict[str, str],
    cutoff_0based: int,
) -> List[MemoryItem]:
    """cutoff_0based = position_after_session (0-based inclusive session index)."""
    max_sess_1based = cutoff_0based + 1
    items: List[MemoryItem] = []
    seen: Set[str] = set()
    order = 0
    sorted_chunks = sorted(chunks, key=lambda c: int(c.get("chunk_index", 0)))
    for ch in sorted_chunks:
        sess_idx = int(ch.get("session_index") or 0)
        if sess_idx > max_sess_1based:
            continue
        source = str(ch.get("source") or "filler")
        ci = int(ch.get("chunk_index", 0))
        session_date = str(ch.get("session_date") or "")
        for raw in ch.get("candidate_memories") or []:
            text = str(raw).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            items.append(
                MemoryItem(
                    text=text,
                    source=source,
                    session_index=sess_idx,
                    chunk_index=ci,
                    order_index=order,
                    entity=entity_map.get(text),
                    session_date=session_date,
                )
            )
            order += 1
    return items


def _cascade_chain_entities(
    episode: Dict[str, Any],
    target_entity: str,
    hop: int,
) -> List[str]:
    """Collect entity chain for a Cas question by walking dependency_edges_used."""
    edges = episode.get("dependency_edges_used") or []
    chain = [target_entity]
    cur = target_entity
    for h in range(int(hop), 0, -1):
        for edge in edges:
            if edge.get("target") == cur and int(edge.get("hop", 0)) == h:
                chain.insert(0, edge["source"])
                cur = edge["source"]
                break
    return chain


def _build_logic_chain(
    episode: Dict[str, Any],
    target_entity: str,
    hop: int,
    phase: str,
    chain: List[str],
) -> Dict[str, Any]:
    """Human-readable cascade logic for the question."""
    edges = episode.get("dependency_edges_used") or []
    entities_meta = episode.get("entities") or {}
    root_change = episode.get("root_change") or {}

    steps: List[Dict[str, Any]] = []
    for i, ent in enumerate(chain):
        meta = entities_meta.get(ent) or {}
        val_key = "before" if phase == "before" else "after"
        value = meta.get(val_key)
        if value is None:
            value = meta.get("before")
        step: Dict[str, Any] = {
            "entity": ent,
            "value": value,
            "task": meta.get("task"),
        }
        if i > 0:
            src = chain[i - 1]
            for e in edges:
                if e.get("source") == src and e.get("target") == ent:
                    step["edge_from"] = src
                    step["edge_hop"] = e.get("hop")
                    step["pattern"] = e.get("pattern")
                    if e.get("is_2hop_middle"):
                        step["is_2hop_middle"] = True
                    break
        steps.append(step)

    edge_lines: List[str] = []
    for i in range(1, len(chain)):
        src, tgt = chain[i - 1], chain[i]
        for e in edges:
            if e.get("source") == src and e.get("target") == tgt:
                edge_lines.append(
                    f"{src} → {tgt} (hop={e.get('hop')}, {e.get('pattern', '')})"
                )
                break

    return {
        "root": episode.get("root"),
        "root_change": {
            "before": root_change.get("before"),
            "after": root_change.get("after"),
        },
        "target_entity": target_entity,
        "hop": hop,
        "chain": chain,
        "edges": edge_lines,
        "steps": steps,
    }


def _split_for_question(
    memories: List[MemoryItem],
    chain_entities: Set[str],
) -> Tuple[List[MemoryItem], List[MemoryItem]]:
    golden: List[MemoryItem] = []
    other: List[MemoryItem] = []
    for m in memories:
        if m.entity and m.entity in chain_entities:
            golden.append(m)
        else:
            other.append(m)
    return golden, other


def _classify_target(
    anchor: MemoryItem,
    target: MemoryItem,
    chain_entities: Set[str],
) -> Optional[str]:
    if target.text == anchor.text:
        return None
    if target.entity and target.entity in chain_entities:
        return "gold_same_chain"
    if target.entity:
        return "gold_other_entity"
    if target.source == "evidence_gold_facts":
        return "gold_unmapped"
    return "filler"


def _per_golden_rankings(
    golden: List[MemoryItem],
    pool: List[MemoryItem],
    golden_emb: np.ndarray,
    pool_emb: np.ndarray,
    chain_entities: Set[str],
    *,
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    """For each golden memory, rank similarities vs only PRIOR memories in ingest order."""
    if not golden or not pool:
        return []

    out: List[Dict[str, Any]] = []
    for gi, g in enumerate(golden):
        sim_row = _cosine_sim_matrix(golden_emb[gi : gi + 1], pool_emb)[0]
        candidates: List[Tuple[float, int]] = []
        for j, target in enumerate(pool):
            if target.order_index >= g.order_index:
                continue
            kind = _classify_target(g, target, chain_entities)
            if kind is None:
                continue
            candidates.append((float(sim_row[j]), j))
        candidates.sort(key=lambda x: x[0], reverse=True)

        pairs: List[Dict[str, Any]] = []
        for rank, (score, j) in enumerate(candidates[:top_k], start=1):
            target = pool[j]
            pairs.append({
                "rank": rank,
                "similarity": round(score, 4),
                "other_text": target.text,
                "other_kind": _classify_target(g, target, chain_entities),
                "other_entity": target.entity,
                "other_order": target.order_index,
            })
        out.append({
            "golden_text": g.text,
            "golden_entity": g.entity,
            "golden_order": g.order_index,
            "prior_count": sum(1 for t in pool if t.order_index < g.order_index),
            "top_pairs": pairs,
        })
    return out


def _embed_texts_cached(
    client: OpenAI,
    texts: List[str],
    model: str,
    cache: Dict[str, np.ndarray],
    batch_size: int,
) -> np.ndarray:
    missing = [t for t in texts if t not in cache]
    for i in range(0, len(missing), batch_size):
        batch = missing[i : i + batch_size]
        embs = embed_texts(client, batch, model)
        for t, e in zip(batch, embs):
            cache[t] = e.astype(np.float32)
    return np.vstack([cache[t] for t in texts]).astype(np.float32)


def iter_cas_questions(
    episodes: List[Dict[str, Any]],
    phases: List[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ep in episodes:
        eid = ep["episode_id"]
        for phase in phases:
            block = ep.get(f"{phase}_questions")
            if not block or not isinstance(block, dict):
                continue
            pos = int(block.get("position_after_session", -1))
            q_time = str(block.get("timestamp", ""))
            for q in block.get("questions") or []:
                if str(q.get("task_type", "")) != "Cas":
                    continue
                entities = [str(e) for e in (q.get("entity") or []) if str(e).strip()]
                ref = q.get("gold_answer")
                if ref is None:
                    ref = q.get("expected_answer", "")
                rows.append({
                    "episode_id": eid,
                    "domain": ep.get("domain", ""),
                    "phase": phase,
                    "question": str(q.get("question", "")),
                    "entities": entities,
                    "hop": q.get("hop"),
                    "answer": str(ref or ""),
                    "question_time": q_time,
                    "position_after_session": pos,
                })
    return rows


def build_report_rows(
    episodes_by_id: Dict[str, Dict[str, Any]],
    candidates_dir: Path,
    client: OpenAI,
    embedding_model: str,
    batch_size: int,
    top_k: int,
    phases: List[str],
) -> List[Dict[str, Any]]:
    embed_cache: Dict[str, np.ndarray] = {}
    chunk_cache: Dict[str, List[Dict[str, Any]]] = {}
    entity_cache: Dict[str, Dict[str, str]] = {}

    questions = iter_cas_questions(list(episodes_by_id.values()), phases)
    report_rows: List[Dict[str, Any]] = []

    for q in tqdm(questions, desc="Cas questions"):
        eid = q["episode_id"]
        ep = episodes_by_id[eid]

        if eid not in entity_cache:
            entity_cache[eid] = _build_entity_map(ep)
        if eid not in chunk_cache:
            cand_path = candidates_dir / f"{eid}.json"
            chunk_cache[eid] = _load_candidate_chunks(cand_path) if cand_path.is_file() else []

        entity_map = entity_cache[eid]
        memories = _memories_up_to_cutoff(
            chunk_cache[eid],
            entity_map,
            int(q["position_after_session"]),
        )
        target_entity = q["entities"][0] if q.get("entities") else ""
        hop = int(q.get("hop") or 1)
        chain = _cascade_chain_entities(ep, target_entity, hop)
        chain_set = set(chain)
        logic_chain = _build_logic_chain(ep, target_entity, hop, q["phase"], chain)
        golden, other = _split_for_question(memories, chain_set)

        by_golden: List[Dict[str, Any]] = []
        if golden and memories:
            pool_texts = [m.text for m in memories]
            gold_texts = [m.text for m in golden]
            pool_emb = _embed_texts_cached(client, pool_texts, embedding_model, embed_cache, batch_size)
            gold_emb = _embed_texts_cached(client, gold_texts, embedding_model, embed_cache, batch_size)
            by_golden = _per_golden_rankings(
                golden, memories, gold_emb, pool_emb, chain_set, top_k=top_k,
            )

        other_gold = sum(1 for m in other if m.entity)
        other_filler = len(other) - other_gold

        golden_timeline = [
            {
                "order": m.order_index,
                "entity": m.entity,
                "session_index": m.session_index,
                "chunk_index": m.chunk_index,
                "session_date": m.session_date,
                "text": m.text,
            }
            for m in golden
        ]

        report_rows.append({
            **q,
            "cascade_chain": chain,
            "logic_chain": logic_chain,
            "golden_memories": [m.text for m in golden],
            "golden_timeline": golden_timeline,
            "golden_entities": sorted({m.entity for m in golden if m.entity}),
            "memory_pool_size": len(memories),
            "counts": {
                "golden": len(golden),
                "other": len(other),
                "other_gold_entity": other_gold,
                "other_filler": other_filler,
                "pool": len(memories),
            },
            "by_golden": by_golden,
        })

    return report_rows


_KIND_SHORT = {
    "gold_same_chain": "sc",
    "gold_other_entity": "og",
    "gold_unmapped": "gu",
    "filler": "f",
}


def _compact_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compact: by_golden[i].pairs = [rank, sim, other_text, kind, entity]."""
    out: List[Dict[str, Any]] = []
    for row in rows:
        gold = row.get("golden_memories") or []
        by_golden: List[List[List[Any]]] = []
        for block in row.get("by_golden") or []:
            pairs: List[List[Any]] = []
            for p in block.get("top_pairs") or []:
                pairs.append([
                    p["rank"],
                    p["similarity"],
                    p["other_text"],
                    _KIND_SHORT.get(p["other_kind"], "f"),
                    p.get("other_entity"),
                    p.get("other_order"),
                ])
            by_golden.append(pairs)
        logic = row.get("logic_chain") or {}
        out.append({
            "ep": row["episode_id"],
            "ph": row["phase"],
            "q": row["question"],
            "ent": row.get("entities") or [],
            "hop": row.get("hop"),
            "chain": row.get("cascade_chain") or [],
            "logic": logic,
            "ans": row.get("answer", ""),
            "pos": row.get("position_after_session"),
            "gold": gold,
            "gold_tl": row.get("golden_timeline") or [],
            "cnt": row.get("counts") or {},
            "by_gold": by_golden,
        })
    return out


def _render_html(json_filename: str, meta: Dict[str, Any]) -> str:
    title = "MEME Cas · Golden vs Other Similarity"
    json_esc = html_lib.escape(json_filename)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html_lib.escape(title)}</title>
  <style>
    :root {{
      --bg:#f6f7fb; --card:#fff; --border:#e2e6ef; --text:#1a1d26;
      --muted:#667085; --accent:#3b6df6; --gold:#b45309; --gold-bg:#fffbeb;
      --og:#6d28d9; --og-bg:#f5f3ff; --filler:#64748b; --filler-bg:#f8fafc;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font:14px/1.45 system-ui,sans-serif; background:var(--bg); color:var(--text); }}
    .wrap {{ max-width:1100px; margin:0 auto; padding:16px; }}
    h1 {{ margin:0 0 12px; font-size:18px; }}
    .bar {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:12px; }}
    select, input {{ padding:6px 8px; border:1px solid var(--border); border-radius:6px; background:#fff; }}
    #q-select {{ flex:1; min-width:240px; max-width:520px; }}
    .hint {{ color:var(--muted); font-size:12px; }}
    .card {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:12px 14px; margin-bottom:10px; }}
    .card h2 {{ margin:0 0 8px; font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }}
    .q {{ font-size:15px; font-weight:600; }}
    .meta {{ color:var(--muted); font-size:12px; margin-top:6px; }}
    .gold {{ background:var(--gold-bg); border-left:3px solid var(--gold); padding:6px 8px; margin:4px 0; font-size:13px; }}
    .gold-block {{ margin-bottom:14px; }}
    .gold-anchor {{ background:var(--gold-bg); border:1px solid #fcd34d; border-radius:6px; padding:8px 10px; margin-bottom:6px; font-size:13px; font-weight:600; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th, td {{ border-bottom:1px solid var(--border); padding:6px 8px; vertical-align:top; }}
    th {{ color:var(--muted); font-size:11px; text-align:left; }}
    .sim {{ font-family:ui-monospace,monospace; color:var(--accent); font-weight:600; white-space:nowrap; }}
    .cell-gold {{ background:var(--gold-bg); }}
    .cell-og {{ background:var(--og-bg); }}
    .cell-filler {{ background:var(--filler-bg); }}
    .tag {{ display:inline-block; font-size:10px; padding:1px 5px; border-radius:4px; margin-bottom:2px; }}
    .tag-sg {{ background:#fef3c7; color:var(--gold); }}
    .tag-og {{ background:var(--og-bg); color:var(--og); }}
    .tag-f {{ background:#eef2f7; color:var(--filler); }}
    .clip {{ display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }}
    .chain-step {{ padding:6px 8px; margin:4px 0; border-left:3px solid var(--accent); background:#f0f4ff; font-size:13px; }}
    .chain-edge {{ color:var(--muted); font-size:12px; margin:2px 0 6px 12px; }}
    .tl-item {{ display:flex; gap:8px; padding:5px 0; border-bottom:1px dashed var(--border); font-size:12px; }}
    .tl-ord {{ font-family:ui-monospace,monospace; color:var(--muted); min-width:28px; }}
    .tl-ent {{ color:var(--gold); font-weight:600; min-width:110px; }}
    .err {{ color:#b42318; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{html_lib.escape(title)}</h1>
    <div class="bar">
      <select id="f-phase"><option value="">phase 全部</option></select>
      <select id="f-ep"><option value="">episode 全部</option></select>
      <select id="q-select"><option value="">加载中…</option></select>
      <span class="hint" id="stats"></span>
    </div>
    <p class="hint">数据文件: {json_esc} · 相似度仅与灌库顺序更早的 memory 比较 · 黄=同链golden · 紫=链外golden · 灰=filler</p>
    <div id="detail"><p class="hint">加载数据…</p></div>
  </div>
  <script>
    const JSON_FILE = {json.dumps(json_filename)};
    const TOP_K = {meta.get("top_k", 20)};
    let rows = [], filtered = [];

    const esc = s => (s ?? "").toString().replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
    const clip = (s, n=96) => {{
      const t = (s ?? "").toString();
      return t.length > n ? t.slice(0, n) + "…" : t;
    }};
    const kindLabel = (k, ent) => {{
      if (k === "sc") return "同链golden" + (ent ? "·"+ent : "");
      if (k === "og") return "链外golden" + (ent ? "·"+ent : "");
      if (k === "gu") return "链外golden";
      return "filler";
    }};
    const kindClass = k => k === "sc" ? "cell-gold" : (k === "og" || k === "gu" ? "cell-og" : "cell-filler");
    const tagClass = k => k === "sc" ? "tag-sg" : (k === "f" ? "tag-f" : "tag-og");

    function renderPairs(pairs) {{
      return (pairs || []).map(p => {{
        const [rank, sim, other, kind, ent, oo] = p;
        const ordTag = oo !== undefined && oo !== null ? `<span class="hint">ord#${{oo}}</span> ` : '';
        return `<tr>
          <td class="sim">#${{rank}} ${{Number(sim).toFixed(4)}}</td>
          <td class="${{kindClass(kind)}}" title="${{esc(other)}}">
            <span class="tag ${{tagClass(kind)}}">${{esc(kindLabel(kind, ent))}}</span> ${{ordTag}}
            <div class="clip">${{esc(other)}}</div>
          </td>
        </tr>`;
      }}).join("");
    }}

    function renderLogic(logic) {{
      if (!logic || !logic.steps) return '';
      const rc = logic.root_change || {{}};
      const rootLine = logic.root ? `<div class="meta">root=${{esc(logic.root)}} · change: ${{esc(rc.before)}} → ${{esc(rc.after)}}</div>` : '';
      const edges = (logic.edges || []).map(e => `<div class="chain-edge">${{esc(e)}}</div>`).join('');
      const steps = logic.steps.map((s, i) => {{
        const dep = s.edge_from ? ` ← ${{esc(s.edge_from)}} (hop=${{s.edge_hop}}, ${{esc(s.pattern||'')}})` : '';
        return `<div class="chain-step"><b>#${{i+1}} ${{esc(s.entity)}}</b>${{dep}}<br/>value@phase: <b>${{esc(s.value ?? '—')}}</b></div>`;
      }}).join('');
      return `<div class="card"><h2>Cascade 逻辑链</h2>${{rootLine}}${{edges}}${{steps}}</div>`;
    }}

    function renderTimeline(tl) {{
      if (!tl || !tl.length) return '';
      const rows = tl.map(it => `
        <div class="tl-item">
          <span class="tl-ord">#${{it.order}}</span>
          <span class="tl-ent">${{esc(it.entity||'?')}}</span>
          <span title="${{esc(it.text)}}">sess${{it.session_index}}/chunk${{it.chunk_index}} · ${{esc(clip(it.text, 80))}}</span>
        </div>`).join('');
      return `<div class="card"><h2>Golden 时间顺序（灌库序）</h2>${{rows}}</div>`;
    }}

    function renderDetail(r) {{
      if (!r) return '<p class="hint">无题目</p>';
      const c = r.cnt || {{}};
      const tl = r.gold_tl || [];
      const ordMap = Object.fromEntries(tl.map(it => [it.text, it.order]));
      const blocks = (r.by_gold || []).map((pairs, gi) => {{
        const gtxt = (r.gold || [])[gi] || "";
        const ord = ordMap[gtxt];
        const ordLabel = ord !== undefined ? `order #${{ord}}` : '';
        return `<div class="gold-block">
          <div class="gold-anchor" title="${{esc(gtxt)}}">Golden #${{gi+1}} (${{ordLabel}}): ${{esc(clip(gtxt, 150))}}</div>
          <div class="hint" style="margin-bottom:4px">仅与 order &lt; ${{ord ?? '?'}} 的 memory 比较</div>
          <table><thead><tr><th>#</th><th>Prior Memory</th></tr></thead>
            <tbody>${{renderPairs(pairs) || '<tr><td colspan=2 class=hint>无更早 memory</td></tr>'}}</tbody></table>
        </div>`;
      }}).join("");
      return `
        <div class="card"><h2>Question</h2><div class="q">${{esc(r.q)}}</div>
          <div class="meta">${{esc(r.ph)}} · ${{esc(r.ep)}} · target=${{esc((r.ent||[]).join(","))}} · hop=${{r.hop??"—"}} · golden=${{c.golden??0}} · pool=${{c.pool??0}} · ans: ${{esc(r.ans)}}</div></div>
        ${{renderLogic(r.logic)}}
        ${{renderTimeline(tl)}}
        <div class="card"><h2>各 Golden vs 更早 Memory · Top-${{TOP_K}}</h2>${{blocks || '<span class=hint>无 golden</span>'}}</div>`;
    }}

    function rebuildSelect() {{
      const phase = document.getElementById("f-phase").value;
      const ep = document.getElementById("f-ep").value;
      filtered = rows.filter(r => (!phase || r.ph === phase) && (!ep || r.ep === ep));
      const sel = document.getElementById("q-select");
      const cur = sel.value;
      sel.innerHTML = filtered.map((r, i) => {{
        const first = r.by_gold && r.by_gold[0] && r.by_gold[0][0];
        const top = first ? Number(first[1]).toFixed(3) : "—";
        const ng = (r.gold || []).length;
        return `<option value="${{i}}">${{esc(r.ep)}} ${{esc(r.ph)}} · ${{esc(clip(r.q, 44))}} · g=${{ng}} top=${{top}}</option>`;
      }}).join("");
      document.getElementById("stats").textContent = `${{filtered.length}} / ${{rows.length}} 题`;
      if (filtered.length) {{
        const idx = [...sel.options].some(o => o.value === cur) ? cur : "0";
        sel.value = idx;
        document.getElementById("detail").innerHTML = renderDetail(filtered[Number(idx)]);
      }} else {{
        document.getElementById("detail").innerHTML = '<p class="hint">无匹配</p>';
      }}
    }}

    function initFilters() {{
      const phases = [...new Set(rows.map(r => r.ph))].sort();
      const eps = [...new Set(rows.map(r => r.ep))].sort();
      const fp = document.getElementById("f-phase");
      const fe = document.getElementById("f-ep");
      phases.forEach(v => {{ const o = document.createElement("option"); o.value=v; o.textContent=v; fp.appendChild(o); }});
      eps.forEach(v => {{ const o = document.createElement("option"); o.value=v; o.textContent=v; fe.appendChild(o); }});
      fp.onchange = fe.onchange = rebuildSelect;
      document.getElementById("q-select").onchange = e => {{
        document.getElementById("detail").innerHTML = renderDetail(filtered[Number(e.target.value)]);
      }};
      rebuildSelect();
    }}

    fetch(JSON_FILE).then(r => {{
      if (!r.ok) throw new Error(r.status + " " + r.statusText);
      return r.json();
    }}).then(data => {{
      rows = data.rows || [];
      initFilters();
    }}).catch(err => {{
      document.getElementById("detail").innerHTML = `<p class="err">加载 ${{esc(JSON_FILE)}} 失败: ${{esc(err.message)}}<br>请在同目录起静态服务: <code>python -m http.server -d ${{esc(location.pathname.replace(/[^/]+$/, ""))}}</code></p>`;
    }});
  </script>
</body>
</html>"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MEME Cas golden vs other memory similarity HTML")
    p.add_argument(
        "--dataset",
        type=Path,
        default=_REPO_ROOT / "data/raw_data/MEME/meme_filler32k.json",
    )
    p.add_argument(
        "--candidates-dir",
        type=Path,
        default=_REPO_ROOT / "MemDB/candidates/meme_filler32k_gemma4-26B_0519_as3",
    )
    p.add_argument("--embedding-model", default="qwen3-embedding-0.6b")
    p.add_argument("--embedding-base-url", default=None)
    p.add_argument("--embedding-api-key", default=None)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--phases", default="before,after")
    p.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "test/meme/output/meme_filler32k_cas_gold_similarity.html",
    )
    p.add_argument("--max-questions", type=int, default=0)
    p.add_argument(
        "--html-only",
        action="store_true",
        help="Skip embedding; rebuild HTML from existing .json next to --output",
    )
    return p.parse_args()


def main() -> None:
    load_env(_REPO_ROOT / ".env")
    args = parse_args()
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    json_path = args.output.with_suffix(".json")

    if args.html_only:
        if not json_path.is_file():
            raise SystemExit(f"--html-only 需要已有数据文件: {json_path}")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        meta = payload.get("meta") or {}
        n = len(payload.get("rows") or [])
    else:
        base_url = args.embedding_base_url or os.getenv("EMBEDDING_BASE_URL", "http://localhost:7110/v1/")
        api_key = args.embedding_api_key or os.getenv("EMBEDDING_API_KEY", "zjj")
        client = OpenAI(base_url=base_url, api_key=api_key)

        episodes = json.loads(args.dataset.read_text(encoding="utf-8"))
        episodes_by_id = {ep["episode_id"]: ep for ep in episodes}

        rows = build_report_rows(
            episodes_by_id,
            args.candidates_dir,
            client,
            args.embedding_model,
            args.batch_size,
            args.top_k,
            phases,
        )
        if args.max_questions > 0:
            rows = rows[: args.max_questions]

        meta = {
            "dataset": str(args.dataset.resolve()),
            "candidates_dir": str(args.candidates_dir.resolve()),
            "embedding_model": args.embedding_model,
            "similarity_scope": "within_episode_prior_only",
            "task_type": "Cas",
            "top_k": args.top_k,
            "n_questions": len(rows),
            "phases": phases,
        }
        compact = _compact_rows(rows)
        json_path.write_text(
            json.dumps({"meta": meta, "rows": compact}, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        n = len(compact)
        print(f"Wrote {json_path} ({n} questions)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_render_html(json_path.name, meta), encoding="utf-8")
    print(f"Wrote {args.output} (lightweight shell, data in {json_path.name})")


if __name__ == "__main__":
    main()
