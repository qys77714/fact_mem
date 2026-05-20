#!/usr/bin/env python3
"""Build a standalone HTML viewer for MEME gold_facts eval (pred.jsonl)."""

from __future__ import annotations

import argparse
import base64
import html as html_lib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_MEME_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MEME_DIR.parent.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_MEME_DIR) not in sys.path:
    sys.path.insert(0, str(_MEME_DIR))

from meme_gold_facts_eval import (  # noqa: E402
    GoldFactsMemoryBank,
    MemeQuestion,
    extract_gold_facts,
    load_episodes,
    serialize_retrieved,
)
from utils.env import load_env  # noqa: E402


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_eval_summary(eval_path: Optional[Path]) -> Dict[str, Any]:
    if eval_path is None or not eval_path.is_file():
        return {}
    with eval_path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list) and data:
        return data[-1]
    if isinstance(data, dict):
        return data
    return {}


def _b64_utf8(obj: Any) -> str:
    return base64.b64encode(json.dumps(obj, ensure_ascii=False).encode("utf-8")).decode("ascii")


def _enrich_memories(
    rows: List[Dict[str, Any]],
    episodes_by_id: Dict[str, Dict[str, Any]],
    memory_bank: Optional[GoldFactsMemoryBank],
    *,
    retrieve_topk: int,
    use_all_facts: bool,
) -> None:
    if memory_bank is None:
        return
    for ep in episodes_by_id.values():
        facts = extract_gold_facts(ep, max_session_index=None)
        memory_bank.build_episode(ep["episode_id"], facts)

    for row in rows:
        if row.get("retrieved_memories"):
            continue
        eid = str(row.get("episode_id", ""))
        q = MemeQuestion(
            episode_id=eid,
            domain=str(row.get("domain", "")),
            phase=str(row.get("phase", "after")),
            task_type=str(row.get("task_type", "")),
            question=str(row.get("question", "")),
            reference=str(row.get("answer", "")),
            question_time=str(row.get("question_time", "")),
            position_after_session=int(row.get("position_after_session", 0)),
            hop=row.get("hop"),
            entities=row.get("entities"),
        )
        retrieved = memory_bank.retrieve(
            eid,
            q.question,
            q.position_after_session,
            retrieve_topk,
            use_all_facts,
        )
        row["retrieved_memories"] = serialize_retrieved(retrieved)
        row["retrieved_count"] = len(retrieved)


def build_html_report(
    *,
    pred_path: Path,
    dataset_path: Path,
    output_path: Path,
    eval_path: Optional[Path] = None,
    retrieve_topk: int = 20,
    use_all_facts: bool = True,
    db_root: Optional[Path] = None,
    embedding_model: str = "qwen3-embedding-8b",
) -> Path:
    rows = _load_jsonl(pred_path)
    if not rows:
        raise FileNotFoundError(f"No rows in {pred_path}")

    episodes = load_episodes(dataset_path)
    episodes_by_id = {ep["episode_id"]: ep for ep in episodes}

    memory_bank: Optional[GoldFactsMemoryBank] = None
    needs_fetch = any(not r.get("retrieved_memories") for r in rows)
    if needs_fetch:
        from openai import OpenAI
        import os

        load_env(str(_PROJECT_ROOT / ".env"))
        embed_client = None
        if not use_all_facts:
            embed_base = os.getenv("EMBEDDING_BASE_URL", "http://localhost:7110/v1/")
            embed_key = os.getenv("EMBEDDING_API_KEY", "zjj")
            embed_client = OpenAI(api_key=embed_key, base_url=embed_base)
        cache_root = db_root or (pred_path.parent / "faiss_cache")
        memory_bank = GoldFactsMemoryBank(
            embed_client,
            embedding_model,
            Path(cache_root),
            use_all_facts=use_all_facts,
        )

    _enrich_memories(
        rows,
        episodes_by_id,
        memory_bank,
        retrieve_topk=retrieve_topk,
        use_all_facts=use_all_facts,
    )

    summary = _load_eval_summary(eval_path)
    meta = {
        "pred_path": str(pred_path.resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "n_questions": len(rows),
        "use_all_facts": use_all_facts,
        "retrieve_topk": retrieve_topk,
        "eval_summary": summary,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_html(rows, meta), encoding="utf-8")
    return output_path


def _render_html(rows: List[Dict[str, Any]], meta: Dict[str, Any]) -> str:
    payload = {"rows": rows, "meta": meta}
    b64 = _b64_utf8(payload)
    title = "MEME gold_facts QA"
    acc = meta.get("eval_summary", {}).get("overall_accuracy")
    acc_line = f" · overall acc {acc:.1%}" if isinstance(acc, (int, float)) else ""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html_lib.escape(title)}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;600&display=swap" rel="stylesheet" />
  <style>
    :root {{
      --bg: #0f1218;
      --surface: #171c26;
      --surface2: #1e2533;
      --border: #2a3348;
      --text: #e8ecf4;
      --muted: #8b95ab;
      --accent: #6ea8fe;
      --ok: #5fd68a;
      --bad: #f87171;
      --warn: #fbbf24;
      --mem: #a5b4fc;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      min-height: 100vh;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(15, 18, 24, 0.92);
      backdrop-filter: blur(8px);
      border-bottom: 1px solid var(--border);
      padding: 12px 20px;
    }}
    header h1 {{ margin: 0 0 8px; font-size: 1.15rem; font-weight: 600; }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      font-size: 0.85rem;
    }}
    .toolbar label {{ color: var(--muted); margin-right: 4px; }}
    select, input[type="search"] {{
      background: var(--surface2);
      border: 1px solid var(--border);
      color: var(--text);
      border-radius: 6px;
      padding: 6px 10px;
      font: inherit;
    }}
    input[type="search"] {{ min-width: 200px; flex: 1; max-width: 360px; }}
    .stats {{ color: var(--muted); font-size: 0.8rem; }}
    .layout {{
      display: grid;
      grid-template-columns: 280px 1fr;
      min-height: calc(100vh - 80px);
    }}
    @media (max-width: 900px) {{
      .layout {{ grid-template-columns: 1fr; }}
      aside {{ max-height: 220px; border-right: none; border-bottom: 1px solid var(--border); }}
    }}
    aside {{
      border-right: 1px solid var(--border);
      overflow-y: auto;
      background: var(--surface);
    }}
    #q-list {{ list-style: none; margin: 0; padding: 8px; }}
    #q-list li {{
      padding: 8px 10px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.8rem;
      border: 1px solid transparent;
      margin-bottom: 4px;
    }}
    #q-list li:hover {{ background: var(--surface2); }}
    #q-list li.active {{
      border-color: var(--accent);
      background: var(--surface2);
    }}
    #q-list .q-preview {{
      color: var(--muted);
      display: block;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      margin-top: 2px;
    }}
    main {{ padding: 20px; overflow-y: auto; }}
    .badge {{
      display: inline-block;
      font-size: 0.7rem;
      padding: 2px 8px;
      border-radius: 999px;
      margin-right: 6px;
      font-weight: 600;
      text-transform: uppercase;
    }}
    .badge-ok {{ background: rgba(95, 214, 138, 0.2); color: var(--ok); }}
    .badge-bad {{ background: rgba(248, 113, 113, 0.2); color: var(--bad); }}
    .badge-unk {{ background: rgba(139, 149, 171, 0.2); color: var(--muted); }}
    .badge-type {{ background: rgba(110, 168, 254, 0.15); color: var(--accent); }}
    .section {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 16px;
      margin-bottom: 14px;
    }}
    .section h2 {{
      margin: 0 0 10px;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      font-weight: 600;
    }}
    .question-text {{ font-size: 1.05rem; white-space: pre-wrap; }}
    .meta-line {{ font-size: 0.8rem; color: var(--muted); margin-top: 8px; }}
    .mem-list {{ margin: 0; padding: 0; list-style: none; }}
    .mem-item {{
      border-left: 3px solid var(--mem);
      padding: 8px 12px;
      margin-bottom: 8px;
      background: var(--surface2);
      border-radius: 0 8px 8px 0;
      font-size: 0.88rem;
    }}
    .mem-item .mem-meta {{
      font-family: "IBM Plex Mono", monospace;
      font-size: 0.72rem;
      color: var(--muted);
      margin-bottom: 4px;
    }}
    .mem-item .mem-text {{ white-space: pre-wrap; }}
    .answers {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    @media (max-width: 700px) {{ .answers {{ grid-template-columns: 1fr; }} }}
    .answer-box {{
      background: var(--surface2);
      border-radius: 8px;
      padding: 12px;
      white-space: pre-wrap;
      font-size: 0.92rem;
    }}
    .answer-box.ref {{ border-top: 3px solid var(--ok); }}
    .answer-box.pred {{ border-top: 3px solid var(--accent); }}
    .empty {{ color: var(--muted); font-style: italic; }}
    .hidden {{ display: none !important; }}
  </style>
</head>
<body>
  <header>
    <h1>{html_lib.escape(title)}{acc_line}</h1>
    <div class="toolbar">
      <span>
        <label for="f-task">Task</label>
        <select id="f-task"><option value="">全部</option></select>
      </span>
      <span>
        <label for="f-episode">Episode</label>
        <select id="f-episode"><option value="">全部</option></select>
      </span>
      <span>
        <label for="f-verdict">Judge</label>
        <select id="f-verdict">
          <option value="">全部</option>
          <option value="yes">正确</option>
          <option value="no">错误</option>
          <option value="unk">未评判</option>
        </select>
      </span>
      <input type="search" id="f-search" placeholder="搜索问题文本…" />
      <span class="stats" id="stats"></span>
    </div>
  </header>
  <div class="layout">
    <aside><ul id="q-list"></ul></aside>
    <main id="detail">
      <p class="empty">← 从左侧选择一题</p>
    </main>
  </div>
  <script id="payload" type="application/json+b64">{b64}</script>
  <script>
(function () {{
  const raw = document.getElementById("payload").textContent.trim();
  const data = JSON.parse(atob(raw));
  const rows = data.rows || [];
  const meta = data.meta || {{}};
  let activeIdx = -1;
  let filtered = rows.map((r, i) => ({{ ...r, _idx: i }}));

  const elTask = document.getElementById("f-task");
  const elEp = document.getElementById("f-episode");
  const elVerdict = document.getElementById("f-verdict");
  const elSearch = document.getElementById("f-search");
  const elList = document.getElementById("q-list");
  const elDetail = document.getElementById("detail");
  const elStats = document.getElementById("stats");

  function verdictOf(r) {{
    if (r.judge_api_failed) return "unk";
    if (r.is_correct === true) return "yes";
    if (r.is_correct === false) return "no";
    return "unk";
  }}

  function esc(s) {{
    const d = document.createElement("div");
    d.textContent = s == null ? "" : String(s);
    return d.innerHTML;
  }}

  const tasks = [...new Set(rows.map(r => r.task_type).filter(Boolean))].sort();
  const episodes = [...new Set(rows.map(r => r.episode_id).filter(Boolean))].sort();
  tasks.forEach(t => {{
    const o = document.createElement("option");
    o.value = t; o.textContent = t;
    elTask.appendChild(o);
  }});
  episodes.forEach(e => {{
    const o = document.createElement("option");
    o.value = e; o.textContent = e;
    elEp.appendChild(o);
  }});

  function applyFilters() {{
    const ft = elTask.value;
    const fe = elEp.value;
    const fv = elVerdict.value;
    const q = elSearch.value.trim().toLowerCase();
    filtered = rows
      .map((r, i) => ({{ ...r, _idx: i }}))
      .filter(r => {{
        if (ft && r.task_type !== ft) return false;
        if (fe && r.episode_id !== fe) return false;
        if (fv && verdictOf(r) !== fv) return false;
        if (q && !(r.question || "").toLowerCase().includes(q)) return false;
        return true;
      }});
    renderList();
    if (activeIdx >= 0) {{
      const still = filtered.findIndex(r => r._idx === activeIdx);
      if (still >= 0) renderDetail(filtered[still]);
      else {{
        activeIdx = -1;
        elDetail.innerHTML = '<p class="empty">无匹配题目</p>';
      }}
    }}
    const ok = filtered.filter(r => r.is_correct === true).length;
    const judged = filtered.filter(r => r.is_correct === true || r.is_correct === false).length;
    elStats.textContent = `显示 ${{filtered.length}} / ${{rows.length}}` +
      (judged ? ` · 正确 ${{ok}}/${{judged}}` : "");
  }}

  function badgeVerdict(r) {{
    const v = verdictOf(r);
    if (v === "yes") return '<span class="badge badge-ok">correct</span>';
    if (v === "no") return '<span class="badge badge-bad">wrong</span>';
    return '<span class="badge badge-unk">?</span>';
  }}

  function renderList() {{
    elList.innerHTML = "";
    filtered.forEach((r, i) => {{
      const li = document.createElement("li");
      if (r._idx === activeIdx) li.classList.add("active");
      li.innerHTML =
        badgeVerdict(r) +
        `<span class="badge badge-type">${{esc(r.task_type)}}</span>` +
        `<strong>${{esc(r.episode_id)}}</strong>` +
        `<span class="q-preview">${{esc(r.question)}}</span>`;
      li.onclick = () => {{
        activeIdx = r._idx;
        renderList();
        renderDetail(r);
      }};
      elList.appendChild(li);
    }});
  }}

  function renderMemories(mems) {{
    if (!mems || !mems.length) {{
      return '<p class="empty">（无召回记忆）</p>';
    }}
    return '<ul class="mem-list">' + mems.map((m, i) => {{
      const meta = [
        m.entity ? `entity=${{m.entity}}` : "",
        m.value != null ? `value=${{m.value}}` : "",
        m.session_index != null ? `sess=${{m.session_index}}` : "",
        m.score != null && m.score !== 1 ? `score=${{Number(m.score).toFixed(3)}}` : "",
        m.time || "",
      ].filter(Boolean).join(" · ");
      return `<li class="mem-item">
        <div class="mem-meta">#${{i + 1}} ${{esc(meta)}}</div>
        <div class="mem-text">${{esc(m.text)}}</div>
      </li>`;
    }}).join("") + "</ul>";
  }}

  function renderDetail(r) {{
    const mems = r.retrieved_memories || [];
    elDetail.innerHTML = `
      <div class="section">
        <h2>Question</h2>
        <div class="question-text">${{esc(r.question)}}</div>
        <div class="meta-line">
          ${{badgeVerdict(r)}}
          <span class="badge badge-type">${{esc(r.task_type)}}</span>
          <span class="badge badge-type">${{esc(r.phase)}}</span>
          episode <strong>${{esc(r.episode_id)}}</strong> ·
          ${{esc(r.domain)}} ·
          cutoff session ≤ ${{esc(r.position_after_session)}} ·
          recalled <strong>${{mems.length}}</strong> facts
          ${{r.hop != null ? " · hop " + esc(r.hop) : ""}}
          ${{r.question_time ? " · " + esc(r.question_time) : ""}}
        </div>
      </div>
      <div class="section">
        <h2>Retrieved memories (gold_facts)</h2>
        ${{renderMemories(mems)}}
      </div>
      <div class="section">
        <h2>Answers</h2>
        <div class="answers">
          <div>
            <h2 style="margin-top:0">Reference</h2>
            <div class="answer-box ref">${{esc(r.answer)}}</div>
          </div>
          <div>
            <h2 style="margin-top:0">Model</h2>
            <div class="answer-box pred">${{esc(r.model_answer || "(empty)")}}</div>
          </div>
        </div>
      </div>`;
  }}

  [elTask, elEp, elVerdict].forEach(el => el.addEventListener("change", applyFilters));
  elSearch.addEventListener("input", applyFilters);

  applyFilters();
  if (filtered.length) {{
    activeIdx = filtered[0]._idx;
    renderList();
    renderDetail(filtered[0]);
  }}

  if (meta.eval_summary && meta.eval_summary.overall_accuracy != null) {{
    console.log("eval_summary", meta.eval_summary);
  }}
}})();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build MEME gold_facts QA HTML viewer")
    p.add_argument(
        "--pred",
        type=Path,
        required=True,
        help="pred.jsonl from meme_gold_facts_eval.py",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=_PROJECT_ROOT / "data/raw_data/MEME/meme_nofiller.json",
    )
    p.add_argument(
        "--eval",
        type=Path,
        default=None,
        help="eval_judge.json (optional, for header accuracy)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output HTML (default: <pred-dir>/qa_viewer.html)",
    )
    p.add_argument("--retrieve-topk", type=int, default=20)
    p.add_argument("--use-all-facts", action="store_true")
    p.add_argument("--embedding-model", default="qwen3-embedding-8b")
    p.add_argument("--db-root", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_env(str(_PROJECT_ROOT / ".env"))
    pred_path = args.pred.resolve()
    out = args.out or (pred_path.parent / "qa_viewer.html")
    eval_path = args.eval
    if eval_path is None:
        candidate = pred_path.parent / "eval_judge.json"
        if candidate.is_file():
            eval_path = candidate

    path = build_html_report(
        pred_path=pred_path,
        dataset_path=args.dataset.resolve(),
        eval_path=eval_path.resolve() if eval_path else None,
        output_path=out.resolve(),
        retrieve_topk=args.retrieve_topk,
        use_all_facts=args.use_all_facts,
        db_root=args.db_root,
        embedding_model=args.embedding_model,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
