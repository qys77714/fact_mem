#!/usr/bin/env python3
"""
LME relation_decision vs add_all 对比分析 HTML 报告。

相比 MEME 报告，LME 有完整的 agent_trace（检索记忆 + 实际 prompt），
可以展示每题召回哪些 memory、prompt 长什么样、截断情况等。

用法:
    python3 script/gen_lme_compare_report.py [experiment_dir]
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
from html import escape

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = PROJECT_ROOT / "experiment/lme_s_cand0615_unified_gemma4-26B_gemma4-26B_tl512_exp002"

QTYPE_LABELS = {
    "single-session-user": "单轮-用户信息",
    "multi-session": "多轮对话",
    "single-session-preference": "单轮-偏好",
    "temporal-reasoning": "时间推理",
    "knowledge-update": "知识更新",
    "single-session-assistant": "单轮-助手信息",
}
CAT_LABELS = {
    "both_pass": "✅ 都对", "rd_only_pass": "🔵 RD独对",
    "aa_only_pass": "🟠 AA独对", "neither_pass": "❌ 都错",
}
CAT_COLORS = {
    "both_pass": "#27ae60", "rd_only_pass": "#3498db",
    "aa_only_pass": "#e67e22", "neither_pass": "#e74c3c",
}


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


def load_trace(trace_dir: Path) -> dict:
    """Load agent_trace. Keyed by (history_name, question_id)."""
    traces = {}
    if not trace_dir.exists():
        return traces
    for fname in os.listdir(trace_dir):
        if not fname.endswith(".jsonl"):
            continue
        fpath = trace_dir / fname
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (d.get("history_name", ""), d.get("question_id", ""))
                traces[key] = d
    return traces


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return len(text) // 4


def format_retrieved_memories(retrieved: list) -> str:
    """Format retrieved memories for display in HTML."""
    if not retrieved:
        return '<span style="color:#95a5a6">无检索结果</span>'

    parts = []
    for i, mem in enumerate(retrieved[:20]):  # cap at 20
        text = mem.get("text", "")
        score = mem.get("score", 0)
        time_str = mem.get("time", "")
        mem_id = mem.get("memory_id", "")[:12]

        parts.append(
            f'<div class="mem-item">'
            f'<span class="mem-idx">#{i+1}</span> '
            f'<span class="mem-score">{score:.3f}</span> '
            f'<span class="mem-text">{escape(text[:200])}{"…" if len(text)>200 else ""}</span>'
            f'<span class="mem-meta">id:{escape(mem_id)}'
            f'{" | " + escape(time_str) if time_str else ""}</span>'
            f'</div>'
        )

    if len(retrieved) > 20:
        parts.append(f'<div style="color:#7f8c8d;font-size:.75em">... 还有 {len(retrieved)-20} 条</div>')

    return "\n".join(parts)


def format_prompt_preview(prompt: str, max_len: int = 2000) -> tuple:
    """Return (preview_html, full_length). Truncated sections are highlighted."""
    full_len = len(prompt)
    tokens_est = estimate_tokens(prompt)

    if len(prompt) <= max_len:
        return (
            f'<div class="prompt-box">{escape(prompt)}</div>'
            f'<div class="prompt-info">总长度: {full_len} 字符 (≈{tokens_est} tokens)</div>',
            full_len, tokens_est
        )

    # Show truncated version with highlight
    truncated = prompt[:max_len]
    remaining = prompt[max_len:]

    return (
        f'<div class="prompt-box">'
        f'{escape(truncated)}'
        f'<span class="prompt-cut">⋯ 截断点 (还有 {len(remaining)} 字符) ⋯</span>'
        f'</div>'
        f'<div class="prompt-info">'
        f'总长度: {full_len} 字符 (≈{tokens_est} tokens) | '
        f'显示前 {max_len} 字符 | '
        f'截断比例: {len(remaining)/max(full_len,1)*100:.0f}%'
        f'</div>',
        full_len, tokens_est
    )


# ─── 数据处理 ───────────────────────────────────────────────
def build_rows(rd_pred: dict, aa_pred: dict, rd_trace: dict, aa_trace: dict) -> list:
    all_keys = set(rd_pred.keys()) | set(aa_pred.keys())
    rows = []
    for key in all_keys:
        rd_p = rd_pred.get(key, {})
        aa_p = aa_pred.get(key, {})
        rd_t = rd_trace.get(key, {})
        aa_t = aa_trace.get(key, {})

        rd_pass = rd_p.get("is_correct", False)
        aa_pass = aa_p.get("is_correct", False)

        if rd_p and aa_p:
            if rd_pass and aa_pass:       cat = "both_pass"
            elif rd_pass and not aa_pass: cat = "rd_only_pass"
            elif not rd_pass and aa_pass: cat = "aa_only_pass"
            else:                         cat = "neither_pass"
        elif rd_p:
            cat = "rd_only_pass" if rd_pass else "neither_pass"
        else:
            cat = "aa_only_pass" if aa_pass else "neither_pass"

        qtype = rd_p.get("question_type") or aa_p.get("question_type") or "?"

        # Format retrieved
        rd_retrieved = rd_t.get("retrieved", [])
        aa_retrieved = aa_t.get("retrieved", [])
        rd_retrieved_html = format_retrieved_memories(rd_retrieved)
        aa_retrieved_html = format_retrieved_memories(aa_retrieved)

        # Prompts
        rd_prompt = rd_t.get("prompt", "")
        aa_prompt = aa_t.get("prompt", "")

        rows.append({
            "key": f"{key[0]}|{key[1]}",
            "history_name": key[0],
            "question_id": key[1],
            "question_type": qtype,
            "question": rd_p.get("question") or aa_p.get("question", ""),
            "answer": rd_p.get("answer") or aa_p.get("answer", ""),
            "rd_answer": rd_p.get("model_answer", ""),
            "aa_answer": aa_p.get("model_answer", ""),
            "rd_pass": rd_pass,
            "aa_pass": aa_pass,
            "category": cat,
            "question_time": rd_p.get("question_time") or aa_p.get("question_time", ""),
            "rd_retrieved_count": len(rd_retrieved),
            "aa_retrieved_count": len(aa_retrieved),
            "rd_retrieved_html": rd_retrieved_html,
            "aa_retrieved_html": aa_retrieved_html,
            "rd_prompt": rd_prompt,
            "aa_prompt": aa_prompt,
            "rd_prompt_len": len(rd_prompt),
            "aa_prompt_len": len(aa_prompt),
            "rd_prompt_tokens": estimate_tokens(rd_prompt),
            "aa_prompt_tokens": estimate_tokens(aa_prompt),
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
CSS = r"""
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;background:#f5f6fa;color:#2c3e50;line-height:1.5}
.header{background:linear-gradient(135deg,#2c3e50 0%,#3498db 100%);color:white;padding:18px 24px}
.header h1{font-size:1.3em;margin-bottom:3px}
.header .sub{opacity:.85;font-size:.8em}
.g2{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;padding:16px 24px}
.card{background:white;border-radius:8px;padding:12px;box-shadow:0 1px 3px rgba(0,0,0,.1);text-align:center}
.card .lbl{font-size:.7em;color:#7f8c8d;text-transform:uppercase;letter-spacing:.5px}
.card .val{font-size:1.6em;font-weight:700;margin:1px 0}
.card .sub{font-size:.65em;color:#95a5a6}
.card.rd{border-left:4px solid #3498db}
.card.aa{border-left:4px solid #e67e22}
.card.df{border-left:4px solid #9b59b6}
.tg{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;padding:0 24px 16px}
.tc{background:white;border-radius:8px;padding:12px;box-shadow:0 1px 3px rgba(0,0,0,.1)}
.tc h3{font-size:.9em;margin-bottom:6px}
.tb{display:flex;height:18px;border-radius:3px;overflow:hidden;margin:4px 0}
.tb div{display:flex;align-items:center;justify-content:center;font-size:.6em;font-weight:600;color:white}
.ts{display:grid;grid-template-columns:1fr 1fr;gap:1px 10px;font-size:.7em;margin-top:4px}
.ts .tl{color:#7f8c8d}.ts .tv{font-weight:600;text-align:right}
.bar{display:flex;gap:4px;flex-wrap:wrap;padding:10px 24px 0}
.chip{padding:4px 10px;border-radius:14px;font-size:.72em;font-weight:600;cursor:pointer;border:2px solid transparent;transition:.2s}
.chip:hover{transform:translateY(-1px);box-shadow:0 2px 4px rgba(0,0,0,.1)}
.chip.on{border-color:#2c3e50}
.tabs{display:flex;padding:0 24px;border-bottom:2px solid #ecf0f1;margin-top:10px}
.tab{padding:8px 14px;cursor:pointer;font-weight:600;font-size:.75em;border:none;background:none;color:#7f8c8d;transition:.2s}
.tab:hover{color:#2c3e50}
.tab.on{color:#3498db;border-bottom:2px solid #3498db;margin-bottom:-2px}
.ctrl{padding:10px 24px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.ctrl input{padding:5px 8px;border:1px solid #ddd;border-radius:5px;font-size:.8em;flex:1;max-width:320px}
.ctrl .info{font-size:.75em;color:#7f8c8d}
.cards{padding:0 24px 16px}
.dc{background:white;border-radius:8px;padding:10px;margin:6px 0;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.dc .qh{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px}
.dc .qt{font-weight:600;font-size:.85em;flex:1}
.dc .qm{font-size:.65em;color:#7f8c8d;white-space:nowrap;margin-left:10px}
.badge{display:inline-block;padding:1px 6px;border-radius:6px;font-size:.65em;font-weight:700}
.badge.ok{background:#d5f5e3;color:#27ae60}
.badge.no{background:#fadbd8;color:#c0392b}
.gold-box{background:#f8f9fa;padding:5px 8px;border-radius:4px;margin:4px 0;font-size:.75em}
.compare{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:6px 0}
.abox{padding:8px;border-radius:5px;font-size:.75em}
.abox.rd{background:#eaf2f8;border:1px solid #3498db33}
.abox.aa{background:#fdf2e9;border:1px solid #e67e2233}
.abox h4{font-size:.7em;margin-bottom:3px}
details summary{cursor:pointer;font-weight:600;font-size:.75em;color:#3498db;padding:3px 0}
details summary:hover{color:#2980b9}
.mem-item{display:flex;align-items:baseline;gap:4px;padding:2px 0;font-size:.7em;border-bottom:1px solid #f0f0f0}
.mem-idx{color:#7f8c8d;font-weight:700;min-width:18px}
.mem-score{color:#3498db;font-weight:600;min-width:42px;font-size:.9em}
.mem-text{flex:1;color:#2c3e50}
.mem-meta{color:#95a5a6;font-size:.85em;white-space:nowrap;max-width:200px;overflow:hidden;text-overflow:ellipsis}
.mem-section{background:#f8f9fa;border-radius:4px;padding:6px;margin:4px 0;max-height:400px;overflow-y:auto}
.prompt-box{background:#2c3e50;color:#ecf0f1;padding:10px;border-radius:5px;font-family:monospace;font-size:.68em;white-space:pre-wrap;max-height:350px;overflow-y:auto;margin:4px 0}
.prompt-cut{display:block;background:#e74c3c;color:white;padding:4px 8px;text-align:center;font-weight:700;margin:4px 0;border-radius:3px}
.prompt-info{font-size:.65em;color:#7f8c8d;margin:2px 0}
.pgn{display:flex;justify-content:center;gap:4px;padding:10px}
.pgn button{padding:5px 10px;border:1px solid #ddd;border-radius:5px;background:white;cursor:pointer;font-size:.75em}
.pgn button:hover{background:#f0f0f0}
.pgn button.on{background:#3498db;color:white;border-color:#3498db}
.hidden{display:none}
.compare-mem{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.compare-mem h5{font-size:.72em;margin-bottom:4px;color:#2c3e50}
</style>
"""


def build_html(exp_name: str, summary: dict, rows: list) -> str:
    s = summary
    diff = s["rd_correct"] - s["aa_correct"]
    ds = "+" if diff > 0 else ""

    # Category chips
    cat_chips = ""
    cats = [
        ("all", "全部", s["total"], "#2c3e50"),
        ("both_pass", "✅ 都对", s["category_counts"].get("both_pass", 0), "#27ae60"),
        ("rd_only_pass", "🔵 RD独对", s["category_counts"].get("rd_only_pass", 0), "#3498db"),
        ("aa_only_pass", "🟠 AA独对", s["category_counts"].get("aa_only_pass", 0), "#e67e22"),
        ("neither_pass", "❌ 都错", s["category_counts"].get("neither_pass", 0), "#e74c3c"),
    ]
    for i, (cat, label, cnt, color) in enumerate(cats):
        on = "on" if i == 0 else ""
        cat_chips += '<span class="chip {}" data-cat="{}" style="background:{}15;color:{}" onclick="F(\'{}\')">{} ({})</span>'.format(
            on, cat, color, color, cat, label, cnt
        )

    # Type tabs
    qtabs = '<button class="tab on" data-qt="all" onclick="Q(\'all\')">全部</button>'
    for qt in sorted(QTYPE_LABELS.keys()):
        cnt = s["by_type"].get(qt, {}).get("total", 0)
        if cnt == 0:
            continue
        qtabs += '<button class="tab" data-qt="{}" onclick="Q(\'{}\')">{} ({})</button>'.format(
            qt, qt, QTYPE_LABELS.get(qt, qt), cnt
        )

    # Type grid
    tgrid = ""
    bar_cats = ["both_pass", "rd_only_pass", "aa_only_pass", "neither_pass"]
    bar_labels = ["都对", "RD独对", "AA独对", "都错"]
    bar_colors = ["#27ae60", "#3498db", "#e67e22", "#e74c3c"]
    for qt in sorted(QTYPE_LABELS.keys()):
        bt = s["by_type"].get(qt, {})
        total = bt.get("total", 0)
        if total == 0:
            continue
        rd_acc = bt["rd_correct"] / total if total else 0
        aa_acc = bt["aa_correct"] / total if total else 0
        dif = bt["rd_correct"] - bt["aa_correct"]
        dsign = "+" if dif > 0 else ""

        bar_html = ""
        for i, cat in enumerate(bar_cats):
            cnt = bt.get(cat, 0)
            pct = cnt / total * 100 if total else 0
            if pct > 0:
                bar_html += '<div style="flex:{};background:{}" title="{}: {}">{}</div>'.format(
                    pct, bar_colors[i], bar_labels[i], cnt, cnt if pct > 12 else ""
                )

        tgrid += """<div class="tc">
<h3>{} <span style="font-weight:400;color:#7f8c8d">({}题)</span></h3>
<div class="tb">{}</div>
<div class="ts">
<span class="tl">RD</span><span class="tv" style="color:#3498db">{:.1%}</span>
<span class="tl">AA</span><span class="tv" style="color:#e67e22">{:.1%}</span>
<span class="tl">RD独对</span><span class="tv" style="color:#3498db">{}</span>
<span class="tl">AA独对</span><span class="tv" style="color:#e67e22">{}</span>
<span class="tl">差异</span><span class="tv" style="color:#9b59b6">{}{}</span>
</div></div>""".format(
            QTYPE_LABELS.get(qt, qt), total, bar_html,
            rd_acc, aa_acc,
            bt.get('rd_only_pass', 0), bt.get('aa_only_pass', 0),
            dsign, dif
        )

    # Serialize rows
    rows_json = json.dumps(rows, ensure_ascii=False)

    html = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>LME: RD vs AA — """ + escape(exp_name) + """</title>
<style>""" + CSS + """</style>
</head>
<body>
<div class="header">
<h1>🔬 LME: relation_decision vs add_all 对比分析</h1>
<div class="sub">""" + escape(exp_name) + """ | """ + str(s['total']) + """ 题 | RD """ + f"{s['rd_accuracy']:.1%}" + """ | AA """ + f"{s['aa_accuracy']:.1%}" + """ | RD独对 """ + str(s['category_counts'].get('rd_only_pass', 0)) + """ | AA独对 """ + str(s['category_counts'].get('aa_only_pass', 0)) + """</div>
</div>

<div class="g2">
<div class="card rd"><div class="lbl">relation_decision</div><div class="val" style="color:#3498db">""" + f"{s['rd_accuracy']:.1%}" + """</div><div class="sub">""" + str(s['rd_correct']) + """/""" + str(s['total']) + """</div></div>
<div class="card aa"><div class="lbl">add_all</div><div class="val" style="color:#e67e22">""" + f"{s['aa_accuracy']:.1%}" + """</div><div class="sub">""" + str(s['aa_correct']) + """/""" + str(s['total']) + """</div></div>
<div class="card df"><div class="lbl">差异</div><div class="val" style="color:#9b59b6">""" + ds + str(diff) + """</div><div class="sub">RD独对 """ + str(s['category_counts'].get('rd_only_pass', 0)) + """ | AA独对 """ + str(s['category_counts'].get('aa_only_pass', 0)) + """</div></div>
<div class="card"><div class="lbl">都对</div><div class="val" style="color:#27ae60">""" + str(s['category_counts'].get('both_pass', 0)) + """</div></div>
<div class="card"><div class="lbl">都错</div><div class="val" style="color:#e74c3c">""" + str(s['category_counts'].get('neither_pass', 0)) + """</div></div>
</div>

<h3 style="padding:0 24px 6px;font-size:.85em;color:#2c3e50">按题型对比</h3>
<div class="tg">""" + tgrid + """</div>

<div class="bar">""" + cat_chips + """</div>
<div class="tabs">""" + qtabs + """</div>
<div class="ctrl">
<input id="search" type="text" placeholder="🔍 搜索问题/答案/历史名..." onkeyup="S()">
<span class="info" id="page-info"></span>
</div>

<div class="cards" id="cards"></div>
<div class="pgn" id="pgn"></div>

<script>
const DATA = """ + rows_json + """;
const QL = """ + json.dumps(QTYPE_LABELS) + """;
const PER_PAGE = 20;
let curCat='all', curQt='all', curPage=1, curSearch='';

function filter(){
    return DATA.filter(d=>{
        if(curCat!=='all'&&d.category!==curCat)return false;
        if(curQt!=='all'&&d.question_type!==curQt)return false;
        if(curSearch){
            const s=curSearch.toLowerCase();
            return (d.question+d.answer+d.rd_answer+d.aa_answer+d.history_name).toLowerCase().includes(s);
        }
        return true;
    });
}

function F(cat){curCat=cat;curPage=1;document.querySelectorAll('.chip').forEach(e=>e.classList.toggle('on',e.dataset.cat===cat));R();}
function Q(qt){curQt=qt;curPage=1;document.querySelectorAll('.tab').forEach(e=>e.classList.toggle('on',e.dataset.qt===qt));R();}
function S(){curSearch=document.getElementById('search').value;curPage=1;R();}

const PI={true:'<span class="badge ok">PASS</span>',false:'<span class="badge no">FAIL</span>'};
const CL=""" + json.dumps(CAT_LABELS) + """;

function esc(s){if(!s)return'';const d=document.createElement('div');d.textContent=s;return d.innerHTML;}

function R(){
    const fd=filter();
    const tp=Math.ceil(fd.length/PER_PAGE);
    const st=(curPage-1)*PER_PAGE;
    const pg=fd.slice(st,st+PER_PAGE);
    document.getElementById('page-info').textContent=(st+1)+'-'+Math.min(st+PER_PAGE,fd.length)+' / '+fd.length+' 题';

    let ph='';
    if(tp>1){
        ph+='<button '+(curPage===1?'disabled':'')+' onclick="curPage=1;R()">«</button>';
        ph+='<button '+(curPage===1?'disabled':'')+' onclick="curPage--;R()">‹</button>';
        for(let i=Math.max(1,curPage-2);i<=Math.min(tp,curPage+2);i++)
            ph+='<button class="'+(i===curPage?'on':'')+'" onclick="curPage='+i+';R()">'+i+'</button>';
        ph+='<button '+(curPage===tp?'disabled':'')+' onclick="curPage++;R()">›</button>';
        ph+='<button '+(curPage===tp?'disabled':'')+' onclick="curPage='+tp+';R()">»</button>';
    }
    document.getElementById('pgn').innerHTML=ph;

    let h='';
    for(const d of pg){
        const rdIcon=d.rd_pass!==undefined?PI[!!d.rd_pass]:'N/A';
        const aaIcon=d.aa_pass!==undefined?PI[!!d.aa_pass]:'N/A';

        h+='<div class="dc">'+
        '<div class="qh">'+
        '<div class="qt">'+
        '<span style="background:#ecf0f1;padding:1px 5px;border-radius:3px;font-size:.68em;margin-right:5px">'+(QL[d.question_type]||d.question_type)+'</span>'+
        esc(d.question)+
        '</div>'+
        '<div class="qm">'+esc(d.history_name)+'<br>'+esc((d.question_time||''))+'</div>'+
        '</div>'+
        '<div style="font-size:.7em;color:#7f8c8d;margin:2px 0">'+CL[d.category]+'</div>'+
        '<div class="gold-box"><strong style="color:#27ae60">Gold:</strong> '+esc(d.answer)+'</div>'+
        '<div class="compare">'+
        '<div class="abox rd"><h4>🔵 relation_decision '+rdIcon+'</h4><div>'+esc(d.rd_answer||'(无回答)')+'</div></div>'+
        '<div class="abox aa"><h4>🟠 add_all '+aaIcon+'</h4><div>'+esc(d.aa_answer||'(无回答)')+'</div></div>'+
        '</div>'+
        // Retrieved memories comparison
        '<details><summary>📋 召回 Memory 对比 (RD: '+d.rd_retrieved_count+'条, AA: '+d.aa_retrieved_count+'条)</summary>'+
        '<div class="compare-mem">'+
        '<div><h5>🔵 relation_decision 召回</h5><div class="mem-section">'+d.rd_retrieved_html+'</div></div>'+
        '<div><h5>🟠 add_all 召回</h5><div class="mem-section">'+d.aa_retrieved_html+'</div></div>'+
        '</div></details>'+
        // Prompt comparison (show truncated preview)
        '<details><summary>📝 答题 Prompt (RD: ≈'+d.rd_prompt_tokens+' tokens, AA: ≈'+d.aa_prompt_tokens+' tokens)</summary>'+
        '<div class="compare-mem">'+
        '<div><h5>🔵 relation_decision Prompt</h5>'+
        '<div class="prompt-box">'+esc(d.rd_prompt||'').substring(0,3000)+(d.rd_prompt&&d.rd_prompt.length>3000?'<span class="prompt-cut">⋯ 截断展示 (总长'+d.rd_prompt_len+'字符) ⋯</span>':'')+'</div>'+
        '<div class="prompt-info">总长度: '+d.rd_prompt_len+' 字符 (≈'+d.rd_prompt_tokens+' tokens)</div>'+
        '</div>'+
        '<div><h5>🟠 add_all Prompt</h5>'+
        '<div class="prompt-box">'+esc(d.aa_prompt||'').substring(0,3000)+(d.aa_prompt&&d.aa_prompt.length>3000?'<span class="prompt-cut">⋯ 截断展示 (总长'+d.aa_prompt_len+'字符) ⋯</span>':'')+'</div>'+
        '<div class="prompt-info">总长度: '+d.aa_prompt_len+' 字符 (≈'+d.aa_prompt_tokens+' tokens)</div>'+
        '</div>'+
        '</div></details>'+
        '</div>';
    }
    document.getElementById('cards').innerHTML=h||'<div style="text-align:center;padding:40px;color:#7f8c8d">没有匹配结果</div>';
}

R();
</script>
</body></html>"""
    return html


def main():
    exp_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else EXP_DIR
    exp_name = exp_dir.name

    rd_path = exp_dir / "pred_relation_decision.jsonl"
    aa_path = exp_dir / "pred_add_all.jsonl"
    rd_trace_dir = exp_dir / "agent_trace" / "relation_decision"
    aa_trace_dir = exp_dir / "agent_trace" / "add_all"

    if not rd_path.exists() or not aa_path.exists():
        print(f"ERROR: pred files not found")
        sys.exit(1)

    print(f"Loading {exp_name}...")
    rd_pred = load_pred(rd_path)
    aa_pred = load_pred(aa_path)
    print(f"  RD pred: {len(rd_pred)}, AA pred: {len(aa_pred)}")

    rd_trace = load_trace(rd_trace_dir)
    aa_trace = load_trace(aa_trace_dir)
    print(f"  RD trace: {len(rd_trace)}, AA trace: {len(aa_trace)}")

    rows = build_rows(rd_pred, aa_pred, rd_trace, aa_trace)
    summary = build_summary(rows)
    print(f"  Questions: {len(rows)}")
    print(f"  RD acc: {summary['rd_accuracy']:.1%}, AA acc: {summary['aa_accuracy']:.1%}")
    print(f"  RD only: {summary['category_counts'].get('rd_only_pass',0)}, AA only: {summary['category_counts'].get('aa_only_pass',0)}")

    html = build_html(exp_name, summary, rows)

    output_path = exp_dir / "compare_report.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\n✅ Report: {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
