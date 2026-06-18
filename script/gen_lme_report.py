#!/usr/bin/env python3
"""
Generate an HTML comparison report for LME benchmark results,
showing question, retrieved memories, and answers side-by-side
for relation_decision vs add_all.
"""

import json
import sys
from pathlib import Path

EXP_DIR = Path("experiment/lme_s_cand0615_unified_gemma4-26B_gemma4-26B_tl512_exp001")
TRACE_DIR = EXP_DIR / "agent_trace"

def load_pred(method: str):
    """Load pred file, return {question_id: row}."""
    path = EXP_DIR / f"pred_{method}.jsonl"
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[r["question_id"]] = r
    return rows

def load_trace(method: str):
    """Load trace files, return {question_id: [retrieved_memories]}.
    Only keep essential fields (text, score) to keep HTML size manageable.
    """
    trace_dir = TRACE_DIR / method
    traces = {}
    for fpath in trace_dir.glob("*.jsonl"):
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                qid = r["question_id"]
                # Strip to essential fields only
                slim = []
                for mem in r.get("retrieved", []):
                    slim.append({
                        "text": mem.get("text", ""),
                        "score": mem.get("score"),
                    })
                traces[qid] = slim
    return traces

def main():
    rd_pred = load_pred("relation_decision")
    aa_pred = load_pred("add_all")
    rd_trace = load_trace("relation_decision")
    aa_trace = load_trace("add_all")

    # Collect all question_ids present in both
    all_qids = sorted(set(rd_pred.keys()) & set(aa_pred.keys()),
                      key=lambda q: rd_pred[q].get("question_type", "") + rd_pred[q]["question"])

    # Build per-question data
    questions = []
    type_stats = {}
    for qid in all_qids:
        rd = rd_pred[qid]
        aa = aa_pred[qid]
        qtype = rd["question_type"]
        if qtype not in type_stats:
            type_stats[qtype] = {"total": 0, "rd_correct": 0, "aa_correct": 0}
        type_stats[qtype]["total"] += 1
        if rd["is_correct"]:
            type_stats[qtype]["rd_correct"] += 1
        if aa["is_correct"]:
            type_stats[qtype]["aa_correct"] += 1

        questions.append({
            "qid": qid,
            "history_name": rd["history_name"],
            "question": rd["question"],
            "answer": rd["answer"],
            "question_type": qtype,
            "question_time": rd.get("question_time", ""),
            "rd": {
                "model_answer": rd["model_answer"],
                "is_correct": rd["is_correct"],
                "retrieved": rd_trace.get(qid, []),
            },
            "aa": {
                "model_answer": aa["model_answer"],
                "is_correct": aa["is_correct"],
                "retrieved": aa_trace.get(qid, []),
            },
        })

    # Generate HTML
    print(render_html(questions, type_stats, rd_pred, aa_pred))

def render_html(questions, type_stats, rd_pred, aa_pred):
    rd_acc = sum(1 for r in rd_pred.values() if r["is_correct"]) / len(rd_pred)
    aa_acc = sum(1 for r in aa_pred.values() if r["is_correct"]) / len(aa_pred)

    type_options = "\n".join(
        f'<option value="{t}">{t} ({s["total"]}题 | RD:{s["rd_correct"]/s["total"]:.1%} AA:{s["aa_correct"]/s["total"]:.1%})</option>'
        for t, s in sorted(type_stats.items())
    )

    # Pre-render all question cards as JSON for JS
    questions_json = json.dumps(questions, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LME Benchmark: relation_decision vs add_all</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, 'Segoe UI', system-ui, sans-serif; background: #f5f5f5; color: #333; }}
.header {{ background: #1a1a2e; color: white; padding: 20px 30px; position: sticky; top: 0; z-index: 100; }}
.header h1 {{ font-size: 1.3em; margin-bottom: 8px; }}
.header .stats {{ display: flex; gap: 30px; flex-wrap: wrap; font-size: 0.9em; opacity: 0.9; }}
.header .stat {{ display: flex; align-items: center; gap: 6px; }}
.stat-val {{ font-weight: 700; font-size: 1.1em; }}
.badge-correct {{ background: #27ae60; color: white; padding: 1px 8px; border-radius: 10px; font-size: 0.8em; }}
.badge-wrong {{ background: #e74c3c; color: white; padding: 1px 8px; border-radius: 10px; font-size: 0.8em; }}
.controls {{ background: white; padding: 12px 30px; border-bottom: 1px solid #ddd; display: flex; gap: 15px; align-items: center; flex-wrap: wrap; position: sticky; top: 76px; z-index: 99; }}
.controls select, .controls input {{ padding: 6px 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 0.85em; }}
.controls label {{ font-size: 0.85em; font-weight: 600; color: #555; }}
#search {{ width: 250px; }}
#counter {{ font-size: 0.85em; color: #888; margin-left: auto; }}
.card {{ background: white; margin: 15px 30px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); overflow: hidden; }}
.card-header {{ padding: 12px 20px; border-bottom: 1px solid #eee; display: flex; align-items: center; gap: 12px; }}
.card-header .qtype {{ font-size: 0.75em; background: #e8f0fe; color: #1a73e8; padding: 2px 10px; border-radius: 10px; white-space: nowrap; }}
.card-header .qid {{ font-size: 0.75em; color: #999; }}
.question-text {{ padding: 15px 20px; font-size: 1.05em; line-height: 1.5; border-bottom: 1px solid #eee; }}
.question-text .q-label {{ font-size: 0.7em; color: #999; text-transform: uppercase; margin-bottom: 4px; }}
.answer-row {{ display: flex; gap: 1px; background: #eee; }}
.answer-col {{ flex: 1; padding: 12px 20px; background: white; }}
.answer-col .method {{ font-size: 0.75em; font-weight: 700; margin-bottom: 6px; color: #555; text-transform: uppercase; }}
.answer-col .model-answer {{ padding: 8px 12px; background: #f8f9fa; border-radius: 4px; border-left: 3px solid #3498db; margin-bottom: 8px; font-size: 0.9em; line-height: 1.4; }}
.answer-col.wrong .model-answer {{ border-left-color: #e74c3c; background: #fef5f5; }}
.answer-col .correct-answer {{ font-size: 0.85em; color: #27ae60; }}
.answer-col .correct-answer .label {{ font-weight: 600; color: #555; }}
.answer-col .verdict {{ font-size: 0.75em; margin-top: 4px; }}
.memories {{ padding: 8px 20px 15px; border-top: 1px solid #eee; }}
.memories summary {{ cursor: pointer; font-size: 0.85em; color: #666; padding: 6px 0; user-select: none; }}
.memories summary:hover {{ color: #333; }}
.mem-list {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; max-height: 300px; overflow-y: auto; }}
.mem-item {{ background: #f0f4ff; border: 1px solid #dde4f5; border-radius: 4px; padding: 5px 10px; font-size: 0.8em; max-width: 450px; line-height: 1.3; }}
.mem-item .mem-score {{ font-size: 0.7em; color: #888; }}
.mem-item.no-mem {{ background: #fff3cd; border-color: #ffeeba; }}
.empty {{ text-align: center; padding: 40px; color: #999; }}
</style>
</head>
<body>
<div class="header">
    <h1>LME Benchmark (lme_s): relation_decision vs add_all</h1>
    <div class="stats">
        <div class="stat">📋 总题数: <span class="stat-val">{len(questions)}</span></div>
        <div class="stat">🔷 relation_decision: <span class="stat-val">{rd_acc:.1%}</span>
            <span class="badge-correct">{sum(1 for r in rd_pred.values() if r["is_correct"])}</span>
            <span class="badge-wrong">{sum(1 for r in rd_pred.values() if not r["is_correct"])}</span>
        </div>
        <div class="stat">🔶 add_all: <span class="stat-val">{aa_acc:.1%}</span>
            <span class="badge-correct">{sum(1 for r in aa_pred.values() if r["is_correct"])}</span>
            <span class="badge-wrong">{sum(1 for r in aa_pred.values() if not r["is_correct"])}</span>
        </div>
    </div>
</div>

<div class="controls">
    <label>题型:</label>
    <select id="typeFilter">
        <option value="all">全部</option>
        {type_options}
    </select>
    <label>结果:</label>
    <select id="resultFilter">
        <option value="all">全部</option>
        <option value="rd_better">RD 正确 AA 错误</option>
        <option value="aa_better">AA 正确 RD 错误</option>
        <option value="both_correct">两者都正确</option>
        <option value="both_wrong">两者都错误</option>
    </select>
    <input type="text" id="search" placeholder="🔍 搜索关键词...">
    <span id="counter"></span>
</div>

<div id="cards"></div>

<script>
const DATA = {questions_json};

function renderCard(q) {{
    const rdRetrieved = (q.rd.retrieved || []).slice(0, 30);
    const aaRetrieved = (q.aa.retrieved || []).slice(0, 30);

    function memItems(items) {{
        if (!items.length) return '<div class="mem-item no-mem">无召回记忆</div>';
        return items.map(m => {{
            const score = m.score != null ? (typeof m.score === 'number' ? m.score.toFixed(3) : m.score) : '';
            return `<div class="mem-item">
                <span class="mem-score">${{score ? '[' + score + ']' : ''}}</span> ${{esc(m.text || '')}}
            </div>`;
        }}).join('');
    }}

    return `
    <div class="card" data-type="${{esc(q.question_type)}}" data-result="${{q.rd.is_correct ? 'rdT' : 'rdF'}}_${{q.aa.is_correct ? 'aaT' : 'aaF'}}">
        <div class="card-header">
            <span class="qtype">${{esc(q.question_type)}}</span>
            <span class="qid">${{esc(q.history_name)}} / ${{esc(q.qid)}}</span>
            <span style="font-size:0.75em;color:#aaa">${{esc(q.question_time || '')}}</span>
        </div>
        <div class="question-text">
            <div class="q-label">Question</div>
            ${{esc(q.question)}}
        </div>
        <div class="answer-row">
            <div class="answer-col ${{q.rd.is_correct ? '' : 'wrong'}}">
                <div class="method">🔷 relation_decision</div>
                <div class="model-answer">${{esc(q.rd.model_answer || '(no answer)')}}</div>
                <div class="correct-answer"><span class="label">正确答案:</span> ${{esc(q.answer)}}</div>
                <div class="verdict">${{q.rd.is_correct ? '✅ 正确' : '❌ 错误'}}</div>
            </div>
            <div class="answer-col ${{q.aa.is_correct ? '' : 'wrong'}}">
                <div class="method">🔶 add_all</div>
                <div class="model-answer">${{esc(q.aa.model_answer || '(no answer)')}}</div>
                <div class="correct-answer"><span class="label">正确答案:</span> ${{esc(q.answer)}}</div>
                <div class="verdict">${{q.aa.is_correct ? '✅ 正确' : '❌ 错误'}}</div>
            </div>
        </div>
        <div class="memories">
            <details>
                <summary>📝 召回记忆: RD(${{rdRetrieved.length}}条) | AA(${{aaRetrieved.length}}条)</summary>
                <div style="display:flex; gap:10px; margin-top:6px;">
                    <div style="flex:1">
                        <div style="font-size:0.8em;font-weight:600;color:#555;margin-bottom:4px;">🔷 relation_decision 召回:</div>
                        <div class="mem-list">${{memItems(rdRetrieved)}}</div>
                    </div>
                    <div style="flex:1">
                        <div style="font-size:0.8em;font-weight:600;color:#555;margin-bottom:4px;">🔶 add_all 召回:</div>
                        <div class="mem-list">${{memItems(aaRetrieved)}}</div>
                    </div>
                </div>
            </details>
        </div>
    </div>`;
}}

function esc(s) {{
    if (!s) return '';
    const el = document.createElement('span');
    el.textContent = s;
    return el.innerHTML;
}}

function filter() {{
    const typeFilter = document.getElementById('typeFilter').value;
    const resultFilter = document.getElementById('resultFilter').value;
    const search = document.getElementById('search').value.toLowerCase();
    const cards = document.querySelectorAll('.card');
    let count = 0;
    cards.forEach(card => {{
        const qtype = card.dataset.type;
        const result = card.dataset.result;
        const text = card.textContent.toLowerCase();
        let show = true;
        if (typeFilter !== 'all' && qtype !== typeFilter) show = false;
        if (resultFilter !== 'all') {{
            const [rd, aa] = result.split('_');
            const rdOk = rd === 'rdT', aaOk = aa === 'aaT';
            if (resultFilter === 'rd_better' && !(rdOk && !aaOk)) show = false;
            if (resultFilter === 'aa_better' && !(!rdOk && aaOk)) show = false;
            if (resultFilter === 'both_correct' && !(rdOk && aaOk)) show = false;
            if (resultFilter === 'both_wrong' && !(!rdOk && !aaOk)) show = false;
        }}
        if (search && !text.includes(search)) show = false;
        card.style.display = show ? '' : 'none';
        if (show) count++;
    }});
    document.getElementById('counter').textContent = `显示 ${{count}} 题`;
}}

function init() {{
    const container = document.getElementById('cards');
    container.innerHTML = DATA.map(renderCard).join('');
    filter();
    document.getElementById('typeFilter').addEventListener('change', filter);
    document.getElementById('resultFilter').addEventListener('change', filter);
    document.getElementById('search').addEventListener('input', filter);
}}

init();
</script>
</body>
</html>"""

if __name__ == "__main__":
    main()
