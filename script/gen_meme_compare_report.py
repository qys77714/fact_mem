#!/usr/bin/env python3
"""
生成 MEME relation_decision vs add_all 对比分析 HTML 报告。

特性:
- 整体准确率对比 + 按题型对比
- "A对B错" / "B对A错" 逐题详情（含 judge reason）
- relation_decision 的分类 trace 摘要（按 relation 分布）
- 对差异题目展示相关分类 prompt
- 交互式筛选：题型、对错分类、搜索
- 回答 prompt 模板说明

用法:
    python3 script/gen_meme_compare_report.py [experiment_dir]
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
from html import escape

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = PROJECT_ROOT / "experiment/meme_filler32k_cand0615_unified_gemma4-26B_gemma4-26B_tl512_0616_exp003_meme4p"

# ─── 常量 ───────────────────────────────────────────────────
COLORS = {
    "pass": "#27ae60", "fail": "#e74c3c",
    "rd": "#3498db", "aa": "#e67e22", "diff": "#9b59b6",
    "both_pass": "#27ae60", "rd_only_pass": "#3498db",
    "aa_only_pass": "#e67e22", "neither_pass": "#e74c3c",
}
QTYPE_LABELS = {
    "Cas": "Cascade (Cas)", "Abs": "Abstraction (Abs)", "Del": "Deletion (Del)",
    "ER": "Exact Recall (ER)", "Tr": "Tracking (Tr)", "Agg": "Aggregation (Agg)",
}
REL_COLORS = {"IND": "#95a5a6", "EQV": "#27ae60", "NSO": "#3498db", "OSN": "#e67e22", "CON": "#e74c3c"}

ANSWER_PROMPT_TEMPLATE = """You are a memory-augmented assistant. Use the retrieved memory units to provide accurate and context-aware answers to the user's questions.

[[ context_block ]]

### Question Details
- Current Date: [[ question_time ]]
- Question: [[ question ]]

Please give a short answer."""

CLASSIFY_PROMPT_NOTE = """分类器有两种后端:
- <b>classifier</b>: Qwen3-0.6B + Linear probe (5-class: IND/EQV/NSO/OSN/CON), max_length=192 tokens
- <b>llm</b>: LLM-based classification (fallback)

用户 prompt 格式: "old: {old_fact}\\nnew: {new_fact}"
System prompt: lme_relation_classification_system_en_v2.jinja (含每类 one-liner 示例)
分类后对非IND结果有 LLM verify 复核步骤。"""


# ─── 数据加载 ───────────────────────────────────────────────
def load_pred(path: Path) -> dict:
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            key = (d["history_name"], d["question_id"])
            records[key] = d
    return records


def load_trace_events(trace_dir: Path) -> dict:
    """
    加载 memory_trace/relation_decision/*.jsonl
    Returns: {(history_name, phase): [events]}
    """
    traces = {}
    rd_dir = trace_dir / "relation_decision"
    if not rd_dir.exists():
        return traces
    for fname in sorted(os.listdir(rd_dir)):
        if not fname.endswith(".jsonl"):
            continue
        stem = fname.replace(".jsonl", "")
        parts = stem.split("_", 2)
        if len(parts) < 3:
            continue
        num, phase = parts[1], parts[2]
        key = (f"sw_{num}", phase)
        events = []
        fpath = rd_dir / fname
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        traces[key] = events
    return traces


def trace_summary(events: list) -> dict:
    """Summarize classify/verify events into counts by relation type."""
    classify_counts = Counter()
    verify_pass = 0
    verify_fail = 0
    classify_samples = []  # keep up to 5 samples

    for evt in events:
        purpose = evt.get("purpose", "")
        if "classify_relation" not in purpose and "verify_relation" not in purpose:
            continue
        resp_raw = evt.get("response", {})
        if isinstance(resp_raw, str):
            try:
                resp = json.loads(resp_raw)
            except json.JSONDecodeError:
                resp = {}
        elif isinstance(resp_raw, dict):
            resp = resp_raw
        else:
            resp = {}

        if "verify" in purpose:
            correct = resp.get("correct")
            if correct is True or str(correct).lower() == "true":
                verify_pass += 1
            else:
                verify_fail += 1
        else:
            rel = resp.get("relation", "?")
            classify_counts[rel] += 1
            if len(classify_samples) < 5:
                req = evt.get("request", {})
                msgs = req.get("messages", [])
                user_content = ""
                for m in msgs:
                    if m.get("role") == "user":
                        user_content = m.get("content", "")
                        break
                meta = evt.get("metadata", {})
                classify_samples.append({
                    "relation": rel,
                    "old_new": user_content,
                    "backend": meta.get("backend", "?"),
                    "latency_ms": meta.get("latency_ms", 0),
                })

    return {
        "classify_counts": dict(classify_counts),
        "verify_pass": verify_pass,
        "verify_fail": verify_fail,
        "samples": classify_samples,
    }


def build_comparison(rd_records: dict, aa_records: dict) -> list:
    all_keys = set(rd_records.keys()) | set(aa_records.keys())
    rows = []
    for key in all_keys:
        rd = rd_records.get(key, {})
        aa = aa_records.get(key, {})
        rd_pass = rd.get("u_pass", False)
        aa_pass = aa.get("u_pass", False)
        if rd and aa:
            if rd_pass and aa_pass:       cat = "both_pass"
            elif rd_pass and not aa_pass: cat = "rd_only_pass"
            elif not rd_pass and aa_pass: cat = "aa_only_pass"
            else:                         cat = "neither_pass"
        elif rd:
            cat = "rd_only_pass" if rd_pass else "neither_pass"
        else:
            cat = "aa_only_pass" if aa_pass else "neither_pass"

        rows.append({
            "key": f"{key[0]}|{key[1]}",
            "history_name": key[0], "question_id": key[1],
            "question_type": rd.get("question_type") or aa.get("question_type") or "?",
            "phase": rd.get("phase") or aa.get("phase") or "?",
            "question": rd.get("question") or aa.get("question", ""),
            "answer": rd.get("answer") or aa.get("answer", ""),
            "rd_answer": rd.get("model_answer", ""),
            "aa_answer": aa.get("model_answer", ""),
            "rd_pass": rd_pass, "aa_pass": aa_pass,
            "rd_reason": rd.get("u_reason", ""),
            "aa_reason": aa.get("u_reason", ""),
            "rd_pass_type": rd.get("pass_type"),
            "aa_pass_type": aa.get("pass_type"),
            "category": cat,
            "entity_key": rd.get("entity_key") or aa.get("entity_key", ""),
            "entity_values": rd.get("entity_values") or aa.get("entity_values", {}),
            "question_time": rd.get("question_time") or aa.get("question_time", ""),
        })
    return rows


def build_summary(rows: list) -> dict:
    by_type = defaultdict(lambda: {"total": 0, "rd_correct": 0, "aa_correct": 0,
                                    "rd_only_pass": 0, "aa_only_pass": 0,
                                    "both_pass": 0, "neither_pass": 0})
    for r in rows:
        qt = r["question_type"]
        by_type[qt]["total"] += 1
        if r["rd_pass"]: by_type[qt]["rd_correct"] += 1
        if r["aa_pass"]: by_type[qt]["aa_correct"] += 1
        by_type[qt][r["category"]] += 1

    cat_counts = Counter(r["category"] for r in rows)
    rd_correct = sum(1 for r in rows if r["rd_pass"])
    aa_correct = sum(1 for r in rows if r["aa_pass"])
    total = len(rows)

    return {
        "total": total,
        "rd_correct": rd_correct, "aa_correct": aa_correct,
        "rd_accuracy": rd_correct / max(total, 1),
        "aa_accuracy": aa_correct / max(total, 1),
        "category_counts": dict(cat_counts),
        "by_type": dict(by_type),
    }


# ─── HTML 渲染 ──────────────────────────────────────────────
CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;background:#f5f6fa;color:#2c3e50;line-height:1.5}
.header{background:linear-gradient(135deg,#2c3e50 0%,#3498db 100%);color:white;padding:20px 28px}
.header h1{font-size:1.4em;margin-bottom:4px}
.header .sub{opacity:.85;font-size:.85em}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;padding:20px 28px}
.card{background:white;border-radius:8px;padding:14px;box-shadow:0 1px 3px rgba(0,0,0,.1);text-align:center}
.card .lbl{font-size:.75em;color:#7f8c8d;text-transform:uppercase;letter-spacing:.5px}
.card .val{font-size:1.8em;font-weight:700;margin:2px 0}
.card .sub{font-size:.7em;color:#95a5a6}
.card.rd{border-left:4px solid #3498db}
.card.aa{border-left:4px solid #e67e22}
.card.diff{border-left:4px solid #9b59b6}
.type-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;padding:0 28px 20px}
.type-card{background:white;border-radius:8px;padding:14px;box-shadow:0 1px 3px rgba(0,0,0,.1)}
.type-card h3{font-size:.95em;margin-bottom:8px}
.type-bar{display:flex;height:20px;border-radius:4px;overflow:hidden;margin:6px 0}
.type-bar div{display:flex;align-items:center;justify-content:center;font-size:.65em;font-weight:600;color:white}
.type-stats{display:grid;grid-template-columns:1fr 1fr;gap:2px 12px;font-size:.75em;margin-top:6px}
.type-stats .tl{color:#7f8c8d}.type-stats .tv{font-weight:600;text-align:right}
.bar{display:flex;gap:4px;flex-wrap:wrap;padding:12px 28px 0}
.chip{padding:5px 12px;border-radius:16px;font-size:.78em;font-weight:600;cursor:pointer;border:2px solid transparent;transition:.2s}
.chip:hover{transform:translateY(-1px);box-shadow:0 2px 6px rgba(0,0,0,.12)}
.chip.on{border-color:#2c3e50}
.tabs{display:flex;padding:0 28px;border-bottom:2px solid #ecf0f1;margin-top:12px}
.tab{padding:8px 16px;cursor:pointer;font-weight:600;font-size:.8em;border:none;background:none;color:#7f8c8d;transition:.2s}
.tab:hover{color:#2c3e50}
.tab.on{color:#3498db;border-bottom:2px solid #3498db;margin-bottom:-2px}
.ctrl{padding:12px 28px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.ctrl input{padding:6px 10px;border:1px solid #ddd;border-radius:6px;font-size:.85em;flex:1;max-width:360px}
.ctrl .info{font-size:.8em;color:#7f8c8d}
.cards{padding:0 28px 20px}
.dcard{background:white;border-radius:8px;padding:12px;margin:8px 0;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.dcard .qh{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px}
.dcard .qt{font-weight:600;font-size:.92em;flex:1}
.dcard .qm{font-size:.7em;color:#7f8c8d;white-space:nowrap;margin-left:12px}
.badge{display:inline-block;padding:2px 7px;border-radius:8px;font-size:.7em;font-weight:700}
.badge.pass{background:#d5f5e3;color:#27ae60}
.badge.fail{background:#fadbd8;color:#c0392b}
.gold-box{background:#f8f9fa;padding:6px 10px;border-radius:4px;margin:6px 0;font-size:.8em}
.compare{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:8px 0}
.abox{padding:10px;border-radius:6px;font-size:.8em}
.abox.rd{background:#eaf2f8;border:1px solid #3498db33}
.abox.aa{background:#fdf2e9;border:1px solid #e67e2233}
.abox h4{font-size:.75em;margin-bottom:4px}
.abox .reason{font-size:.7em;color:#7f8c8d;margin-top:3px;font-style:italic}
summary.trace-summary{cursor:pointer;font-weight:600;font-size:.8em;color:#3498db;padding:4px 0}
.trace-box{background:#f8f9fa;border-radius:4px;padding:8px;margin:4px 0;font-size:.75em}
.trace-dist{display:flex;gap:4px;flex-wrap:wrap;margin:4px 0}
.trace-dist span{display:inline-block;padding:1px 6px;border-radius:3px;font-weight:700;font-size:.75em}
.trace-item{font-family:monospace;font-size:.75em;padding:4px 0;border-bottom:1px solid #ecf0f1}
.trace-item .old{color:#7f8c8d}
.trace-item .new{color:#2c3e50}
.template-box{background:#2c3e50;color:#ecf0f1;padding:12px;border-radius:6px;font-family:monospace;font-size:.72em;white-space:pre-wrap;max-height:300px;overflow-y:auto;margin:6px 0}
.info-box{background:#eaf2f8;border:1px solid #3498db33;border-radius:6px;padding:10px;margin:6px 0;font-size:.8em}
summary.info-summary{cursor:pointer;font-weight:600;font-size:.82em;color:#2c3e50}
.pgn{display:flex;justify-content:center;gap:6px;padding:12px}
.pgn button{padding:6px 12px;border:1px solid #ddd;border-radius:6px;background:white;cursor:pointer;font-size:.8em}
.pgn button:hover{background:#f0f0f0}
.pgn button.on{background:#3498db;color:white;border-color:#3498db}
.hidden{display:none}
"""


def render(exp_name: str, summary: dict, rows: list, traces: dict) -> str:
    s = summary
    diff = s["rd_correct"] - s["aa_correct"]
    ds = "+" if diff > 0 else ""

    # -- 预计算每个 question 对应的 trace 摘要 --
    # trace key: (history_name, phase)
    trace_summaries = {}
    for (hn, phase), events in traces.items():
        trace_summaries[(hn, phase)] = trace_summary(events)

    # -- 生成每行数据 --
    rows_json = []
    for r in rows:
        hn, phase = r["history_name"], r["phase"]
        ts = trace_summaries.get((hn, phase), {})
        classify_counts = ts.get("classify_counts", {})
        verify_pass = ts.get("verify_pass", 0)
        verify_fail = ts.get("verify_fail", 0)
        samples = ts.get("samples", [])

        # 找与 entity_key 相关的 sample
        ek = r.get("entity_key", "").lower()
        related_samples = []
        for sp in samples:
            old_new = sp.get("old_new", "").lower()
            if ek and ek in old_new:
                related_samples.append(sp)
        if not related_samples:
            related_samples = samples[:3]  # fallback: first 3 samples

        rows_json.append({
            **r,
            "classify_counts": classify_counts,
            "verify_pass": verify_pass,
            "verify_fail": verify_fail,
            "samples": related_samples,
            "total_classify": sum(classify_counts.values()),
            "total_verify": verify_pass + verify_fail,
        })

    rows_json_str = json.dumps(rows_json, ensure_ascii=False)

    # ─── HTML ───
    cats = [
        ("all", "全部", s["total"], "#2c3e50"),
        ("both_pass", "✅ 都对", s["category_counts"].get("both_pass", 0), "#27ae60"),
        ("rd_only_pass", "🔵 RD独对", s["category_counts"].get("rd_only_pass", 0), "#3498db"),
        ("aa_only_pass", "🟠 AA独对", s["category_counts"].get("aa_only_pass", 0), "#e67e22"),
        ("neither_pass", "❌ 都错", s["category_counts"].get("neither_pass", 0), "#e74c3c"),
    ]

    cat_chips = ""
    for i, c in enumerate(cats):
        on_cls = "on" if i == 0 else ""
        cat_chips += '<span class="chip {}" data-cat="{}" style="background:{}15;color:{}" onclick="F(\'{}\')">{} ({})</span>'.format(
            on_cls, c[0], c[3], c[3], c[0], c[1], c[2]
        )

    qtabs = '<button class="tab on" data-qt="all" onclick="Q(\'all\')">全部</button>'
    for qt in ["Cas", "Abs", "Del", "ER", "Tr", "Agg"]:
        cnt = s["by_type"].get(qt, {}).get("total", 0)
        qtabs += f'<button class="tab" data-qt="{qt}" onclick="Q(\'{qt}\')">{QTYPE_LABELS.get(qt,qt)} ({cnt})</button>'

    # Type grid
    tgrid = ""
    bar_cats = ["both_pass", "rd_only_pass", "aa_only_pass", "neither_pass"]
    bar_labels = ["都对", "RD独对", "AA独对", "都错"]
    bar_colors = ["#27ae60", "#3498db", "#e67e22", "#e74c3c"]
    for qt in ["Cas", "Abs", "Del", "ER", "Tr", "Agg"]:
        bt = s["by_type"].get(qt, {})
        total = bt.get("total", 0)
        if total == 0: continue
        rd_acc = bt["rd_correct"] / total
        aa_acc = bt["aa_correct"] / total
        dif = bt["rd_correct"] - bt["aa_correct"]
        dsign = "+" if dif > 0 else ""

        bar_html = ""
        for i, cat in enumerate(bar_cats):
            cnt = bt.get(cat, 0)
            pct = cnt / total * 100
            if pct > 0:
                bar_html += f'<div style="flex:{pct};background:{bar_colors[i]}" title="{bar_labels[i]}: {cnt}">{cnt if pct>10 else ""}</div>'

        tgrid += f"""<div class="type-card">
<h3>{QTYPE_LABELS.get(qt,qt)} <span style="font-weight:400;color:#7f8c8d">({total}题)</span></h3>
<div class="type-bar">{bar_html}</div>
<div class="type-stats">
<span class="tl">RD</span><span class="tv" style="color:#3498db">{rd_acc:.1%}</span>
<span class="tl">AA</span><span class="tv" style="color:#e67e22">{aa_acc:.1%}</span>
<span class="tl">RD独对</span><span class="tv" style="color:#3498db">{bt.get('rd_only_pass',0)}</span>
<span class="tl">AA独对</span><span class="tv" style="color:#e67e22">{bt.get('aa_only_pass',0)}</span>
<span class="tl">差异</span><span class="tv" style="color:#9b59b6">{dsign}{dif}</span>
</div></div>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>MEME: RD vs AA — {escape(exp_name)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="header">
<h1>🔬 MEME: relation_decision vs add_all 对比分析</h1>
<div class="sub">{escape(exp_name)} | {s['total']} 题 | RD {s['rd_accuracy']:.1%} | AA {s['aa_accuracy']:.1%} | RD独对 {s['category_counts'].get('rd_only_pass',0)} | AA独对 {s['category_counts'].get('aa_only_pass',0)}</div>
</div>

<div class="grid2">
<div class="card rd"><div class="lbl">relation_decision</div><div class="val" style="color:#3498db">{s['rd_accuracy']:.1%}</div><div class="sub">{s['rd_correct']}/{s['total']}</div></div>
<div class="card aa"><div class="lbl">add_all</div><div class="val" style="color:#e67e22">{s['aa_accuracy']:.1%}</div><div class="sub">{s['aa_correct']}/{s['total']}</div></div>
<div class="card diff"><div class="lbl">差异</div><div class="val" style="color:#9b59b6">{ds}{diff}</div><div class="sub">RD独对 {s['category_counts'].get('rd_only_pass',0)} | AA独对 {s['category_counts'].get('aa_only_pass',0)}</div></div>
<div class="card"><div class="lbl">都对</div><div class="val" style="color:#27ae60">{s['category_counts'].get('both_pass',0)}</div></div>
<div class="card"><div class="lbl">都错</div><div class="val" style="color:#e74c3c">{s['category_counts'].get('neither_pass',0)}</div></div>
</div>

<h3 style="padding:0 28px 8px;font-size:.9em;color:#2c3e50">按题型对比</h3>
<div class="type-grid">{tgrid}</div>

<div class="bar">{cat_chips}</div>
<div class="tabs">{qtabs}</div>
<div class="ctrl">
<input id="search" type="text" placeholder="🔍 搜索问题/答案/历史..." onkeyup="S()">
<span class="info" id="page-info"></span>
</div>

<!-- Prompt 模板说明 -->
<details style="margin:8px 28px"><summary class="info-summary">📝 答题 Prompt 模板 & 分类器说明</summary>
<div class="info-box">
<h4>回答生成 Prompt（agent_prompt_en_open.jinja）</h4>
<div class="template-box">{escape(ANSWER_PROMPT_TEMPLATE)}</div>
<p style="font-size:.75em;color:#7f8c8d;margin-top:4px">
context_block 由 memory_system.format_retrieved_for_context() 生成，包含检索到的记忆文本+时间。<br>
截断由 trim_context() 控制，上限 memory_token_limit (配置中为 512 tokens)。
</p>
<h4 style="margin-top:12px">Relation Classifier</h4>
<p style="font-size:.78em">{CLASSIFY_PROMPT_NOTE}</p>
</div>
</details>

<div class="cards" id="cards"></div>
<div class="pgn" id="pgn"></div>

<script>
const DATA = {rows_json_str};
const QL = {json.dumps(QTYPE_LABELS)};
const PER_PAGE = 25;
let curCat='all', curQt='all', curPage=1, curSearch='';

function filter(){{
    return DATA.filter(d=>{{
        if(curCat!=='all'&&d.category!==curCat)return false;
        if(curQt!=='all'&&d.question_type!==curQt)return false;
        if(curSearch){{
            const s=curSearch.toLowerCase();
            return (d.question+d.answer+d.rd_answer+d.aa_answer+d.history_name).toLowerCase().includes(s);
        }}
        return true;
    }});
}}

function F(cat){{curCat=cat;curPage=1;document.querySelectorAll('.chip').forEach(e=>e.classList.toggle('on',e.dataset.cat===cat));R();}}
function Q(qt){{curQt=qt;curPage=1;document.querySelectorAll('.tab').forEach(e=>e.classList.toggle('on',e.dataset.qt===qt));R();}}
function S(){{curSearch=document.getElementById('search').value;curPage=1;R();}}

const PI={{true:'<span class="badge pass">PASS</span>',false:'<span class="badge fail">FAIL</span>'}};
const CL={{'both_pass':'✅ 都对','rd_only_pass':'🔵 RD独对','aa_only_pass':'🟠 AA独对','neither_pass':'❌ 都错'}};

function esc(s){{if(!s)return'';const d=document.createElement('div');d.textContent=s;return d.innerHTML;}}

function R(){{
    const fd=filter();
    const tp=Math.ceil(fd.length/PER_PAGE);
    const st=(curPage-1)*PER_PAGE;
    const pg=fd.slice(st,st+PER_PAGE);
    document.getElementById('page-info').textContent=`${{st+1}}-${{Math.min(st+PER_PAGE,fd.length)}} / ${{fd.length}} 题`;

    let ph='';
    if(tp>1){{
        ph+=`<button ${{curPage===1?'disabled':''}} onclick="curPage=1;R()">«</button>`;
        ph+=`<button ${{curPage===1?'disabled':''}} onclick="curPage--;R()">‹</button>`;
        for(let i=Math.max(1,curPage-2);i<=Math.min(tp,curPage+2);i++)
            ph+=`<button class="${{i===curPage?'on':''}}" onclick="curPage=${{i}};R()">${{i}}</button>`;
        ph+=`<button ${{curPage===tp?'disabled':''}} onclick="curPage++;R()">›</button>`;
        ph+=`<button ${{curPage===tp?'disabled':''}} onclick="curPage=${{tp}};R()">»</button>`;
    }}
    document.getElementById('pgn').innerHTML=ph;

    let h='';
    for(const d of pg){{
        const rdIcon=PI[d.rd_pass]||(d.rd_answer?PI[false]:'N/A');
        const aaIcon=PI[d.aa_pass]||(d.aa_answer?PI[false]:'N/A');

        // Trace distribution
        const cc=d.classify_counts||{{}};
        let distHtml='';
        const relOrder=['IND','EQV','NSO','OSN','CON'];
        const rc={{"IND":"#95a5a6","EQV":"#27ae60","NSO":"#3498db","OSN":"#e67e22","CON":"#e74c3c"}};
        for(const rel of relOrder){{
            if(cc[rel]) distHtml+=`<span style="background:${{rc[rel]}}22;color:${{rc[rel]}}">${{rel}}:${{cc[rel]}}</span>`;
        }}
        if(!distHtml) distHtml='<span style="color:#95a5a6">无分类</span>';

        // Samples
        let samplesHtml='';
        if(d.samples&&d.samples.length){{
            samplesHtml='<div style="margin-top:4px">';
            for(const sp of d.samples.slice(0,3)){{
                const on=sp.old_new||'';
                const parts=on.split('\\nnew: ');
                const oldT=parts[0]?parts[0].replace('old: ','').slice(0,150):'';
                const newT=parts[1]?parts[1].slice(0,150):'';
                samplesHtml+=`<div class="trace-item">[<b style="color:${{rc[sp.relation]||'#999'}}">${{esc(sp.relation)}}</b>] <span class="old">old: ${{esc(oldT)}}${{oldT.length>=150?'…':''}}</span> → <span class="new">new: ${{esc(newT)}}${{newT.length>=150?'…':''}}</span> <span style="color:#999">(${{esc(sp.backend)}}, ${{(sp.latency_ms/1000).toFixed(1)}}s)</span></div>`;
            }}
            samplesHtml+='</div>';
        }}

        // Entity info
        let ent='';
        if(d.entity_key) ent=`<div style="font-size:.7em;color:#7f8c8d;margin:2px 0">Entity: ${{esc(d.entity_key)}} = ${{esc(JSON.stringify(d.entity_values).slice(0,200))}}</div>`;

        // Trace section
        const vt=d.total_verify||0;
        const vp=d.verify_pass||0;
        const vf=d.verify_fail||0;

        h+=`<div class="dcard">
<div class="qh">
<div class="qt">
<span style="background:#ecf0f1;padding:1px 6px;border-radius:3px;font-size:.7em;margin-right:6px">${{QL[d.question_type]||d.question_type}}</span>
<span style="background:#e8f8f5;padding:1px 6px;border-radius:3px;font-size:.7em;margin-right:6px">${{esc(d.phase)}}</span>
${{esc(d.question)}}
</div>
<div class="qm">${{esc(d.history_name)}}<br>${{esc(d.question_id)}}<br>${{esc(d.question_time||'')}}</div>
</div>
${{ent}}
<div style="font-size:.75em;color:#7f8c8d;margin:2px 0">${{CL[d.category]||d.category}}</div>
<div class="gold-box"><strong style="color:#27ae60">Gold:</strong> ${{esc(d.answer)}}</div>
<div class="compare">
<div class="abox rd"><h4>🔵 relation_decision ${{rdIcon}}</h4><div>${{esc(d.rd_answer||'(无回答)')}}</div>${{d.rd_reason?`<div class="reason">Judge: ${{esc(d.rd_reason)}}</div>`:''}}${{d.rd_pass_type?`<div style="font-size:.7em;color:#7f8c8d">pass_type: ${{esc(d.rd_pass_type)}}</div>`:''}}</div>
<div class="abox aa"><h4>🟠 add_all ${{aaIcon}}</h4><div>${{esc(d.aa_answer||'(无回答)')}}</div>${{d.aa_reason?`<div class="reason">Judge: ${{esc(d.aa_reason)}}</div>`:''}}${{d.aa_pass_type?`<div style="font-size:.7em;color:#7f8c8d">pass_type: ${{esc(d.aa_pass_type)}}</div>`:''}}</div>
</div>
<details class="trace-summary"><summary>📋 分类 Trace: ${{d.total_classify}} classify, ${{vt}} verify (${{vp}} pass, ${{vf}} fail)</summary>
<div class="trace-box">
<div class="trace-dist">${{distHtml}}</div>
${{samplesHtml}}
</div></details>
</div>`;
    }}
    document.getElementById('cards').innerHTML=h||'<div style="text-align:center;padding:40px;color:#7f8c8d">没有匹配结果</div>';
}}

R();
</script>
</body></html>"""
    return html


def main():
    exp_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else EXP_DIR
    exp_name = exp_dir.name

    rd_path = exp_dir / "pred_relation_decision.jsonl"
    aa_path = exp_dir / "pred_add_all.jsonl"
    trace_dir = exp_dir / "memory_trace"

    if not rd_path.exists() or not aa_path.exists():
        print(f"ERROR: pred files not found in {exp_dir}")
        sys.exit(1)

    print(f"Loading {exp_name}...")
    rd_records = load_pred(rd_path)
    aa_records = load_pred(aa_path)
    print(f"  RD: {len(rd_records)}, AA: {len(aa_records)}")

    traces = load_trace_events(trace_dir) if trace_dir.exists() else {}
    print(f"  Trace files: {len(traces)}")

    rows = build_comparison(rd_records, aa_records)
    summary = build_summary(rows)
    print(f"  Questions: {len(rows)} | RD acc: {summary['rd_accuracy']:.1%} | AA acc: {summary['aa_accuracy']:.1%}")
    print(f"  RD only pass: {summary['category_counts'].get('rd_only_pass',0)} | AA only pass: {summary['category_counts'].get('aa_only_pass',0)}")

    html = render(exp_name, summary, rows, traces)

    output_path = exp_dir / "compare_report.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\n✅ Report: {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
