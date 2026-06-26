"""生成混淆数据集审查 HTML（自包含，无对话数据）"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))

DATA = json.load(open(os.path.join(REPO, "data/preprocessed/confusion_review_data.json")))
DATA_JS = json.dumps(DATA, ensure_ascii=False)

HTML = f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LME Confusion Dataset Review — 主集 {len(DATA["main"])} + partial {len(DATA["partial"])}</title>
<style>
:root {{
  --gold: #e6a817; --gold-bg: #fef9e7;
  --lowered: #d97706; --lowered-bg: #fff7ed;
  --dist: #3b82f6; --dist-bg: #eff6ff;
  --good: #10b981; --bad: #ef4444;
  --bg: #f8fafc; --card: #fff; --text: #1e293b; --muted: #64748b;
  --radius: 10px; --shadow: 0 1px 3px rgba(0,0,0,.08);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  background: var(--bg); color: var(--text); line-height: 1.5;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ max-width: 1100px; margin: 0 auto; padding: 24px 16px 80px; }}

/* ---- summary ---- */
.summary {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }}
.summary .stat {{ background: var(--card); border-radius: var(--radius); box-shadow: var(--shadow); padding: 14px 18px; min-width: 100px; }}
.stat .n {{ font-size: 28px; font-weight: 700; }}
.stat .l {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .5px; }}

/* ---- filters ---- */
.filters {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 16px; align-items: center; }}
.filters select, .filters input {{ padding: 8px 12px; border: 1px solid #e2e8f0; border-radius: 6px; font-size: 14px; background: var(--card); }}
.filters input {{ flex: 1; min-width: 200px; }}
.badge {{ display: inline-block; padding: 2px 10px; border-radius: 99px; font-size: 11px; font-weight: 600; letter-spacing: .3px; }}
.badge.golden {{ background: var(--gold-bg); color: #92400e; }}
.badge.lowered {{ background: var(--lowered-bg); color: #c2410c; }}
.badge.distractor {{ background: var(--dist-bg); color: #1d4ed8; }}

/* ---- card ---- */
.card {{ background: var(--card); border-radius: var(--radius); box-shadow: var(--shadow); margin-bottom: 12px; overflow: hidden; }}
.card-header {{ padding: 14px 18px; cursor: pointer; display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; }}
.card-header:hover {{ background: #f1f5f9; }}
.card-header .q {{ font-weight: 600; flex: 1; }}
.card-header .meta {{ font-size: 12px; color: var(--muted); white-space: nowrap; text-align: right; }}
.card-header .answer {{ font-size: 13px; color: var(--muted); margin-top: 2px; }}
.card-body {{ display: none; padding: 0 18px 16px; }}
.card.open .card-body {{ display: block; }}
.card-toggle {{ font-size: 18px; color: var(--muted); transition: transform .15s; }}
.card.open .card-toggle {{ transform: rotate(90deg); }}

/* ---- memory group ---- */
.mem-group {{ margin-bottom: 14px; }}
.mem-group h4 {{ font-size: 13px; font-weight: 600; margin-bottom: 6px; display: flex; align-items: center; gap: 6px; }}
.mem-item {{ display: flex; align-items: center; gap: 8px; padding: 5px 0; border-bottom: 1px solid #f1f5f9; font-size: 13px; }}
.mem-item .txt {{ flex: 1; }}
.mem-item .sim {{ font-size: 11px; font-weight: 600; color: var(--muted); min-width: 44px; text-align: right; }}
.bar-wrap {{ width: 120px; height: 8px; background: #f1f5f9; border-radius: 4px; overflow: hidden; flex-shrink: 0; }}
.bar {{ height: 100%; border-radius: 4px; transition: width .2s; }}

/* constraint line */
.threshold-line {{ border-top: 2px dashed var(--bad); margin: 8px 0 4px; position: relative; }}
.threshold-line::after {{ content: "← lowered_min=" attr(data-val); position: absolute; right: 0; top: -18px; font-size: 10px; color: var(--bad); white-space: nowrap; }}

/* constraint ok/ng indicator */
.constraint-tag {{ font-size: 11px; padding: 2px 8px; border-radius: 99px; font-weight: 600; }}
.constraint-tag.ok {{ background: #d1fae5; color: #065f46; }}
.constraint-tag.ng {{ background: #fee2e2; color: #991b1b; }}

/* collapsed preview */
.preview-dist {{ font-size: 12px; color: var(--muted); margin-top: 4px; display: flex; gap: 4px; flex-wrap: wrap; }}
.preview-dist span {{ background: #f1f5f9; padding: 1px 6px; border-radius: 3px; font-size: 11px; }}

.empty {{ color: var(--muted); font-style: italic; font-size: 13px; padding: 8px 0; }}

@media (max-width: 600px) {{
  .card-header {{ flex-direction: column; }}
  .bar-wrap {{ width: 60px; }}
}}
</style>
</head>
<body>

<h1 style="margin-bottom:4px">🧪 LME Confusion Dataset Review</h1>
<p style="color:var(--muted);margin-bottom:18px;font-size:14px">
  主集 <b id="n-main">{len(DATA["main"])}</b> 题（8/8 达标）+ partial <b id="n-partial">{len(DATA["partial"])}</b> 题
  &nbsp;|&nbsp; embedding: qwen3-embedding-0.6b
  &nbsp;|&nbsp; 对话数据已切除（体积优化）
</p>

<div class="summary" id="summary-stats"></div>

<div class="filters">
  <select id="filter-type"><option value="">全部类型</option></select>
  <select id="filter-set">
    <option value="all">全部 (主集+partial)</option>
    <option value="main">仅主集 (constraint_ok=true)</option>
    <option value="partial">仅 partial</option>
  </select>
  <input id="filter-search" placeholder="搜索 question / answer / memory 文本...">
  <span id="filter-count" style="font-size:13px;color:var(--muted)"></span>
</div>

<div id="cards"></div>

<script>
const DATA = {DATA_JS};
const ALL = [...DATA.main, ...DATA.partial];

// ---- summary stats ----
(()=>{{
  const m = DATA.main, p = DATA.partial;
  let types = {{}};
  m.forEach(r => {{ const t=r.question_type; types[t]=(types[t]||0)+1; }});
  let html = `<div class="stat"><div class="n">${{m.length}}</div><div class="l">主集 8/8达标</div></div>`;
  html += `<div class="stat"><div class="n">${{p.length}}</div><div class="l">partial</div></div>`;
  html += `<div class="stat"><div class="n">${{Object.keys(types).length}}</div><div class="l">question types</div></div>`;
  const loMin = Math.min(...m.map(r=>r.lowered_golden_min_sim));
  const loMax = Math.max(...m.map(r=>r.lowered_golden_min_sim));
  html += `<div class="stat"><div class="n">${{(loMin).toFixed(2)}}–${{(loMax).toFixed(2)}}</div><div class="l">lowered_golden_min range</div></div>`;
  for (const [t,c] of Object.entries(types).sort((a,b)=>b[1]-a[1]))
    html += `<div class="stat"><div class="n">${{c}}</div><div class="l">${{t}}</div></div>`;
  document.getElementById('summary-stats').innerHTML = html;
}})();

// ---- type filter ----
(()=>{{
  const types = [...new Set(ALL.map(r=>r.question_type))].sort();
  const sel = document.getElementById('filter-type');
  types.forEach(t => {{ const o=document.createElement('option'); o.value=t; o.textContent=t; sel.appendChild(o); }});
}})();

// ---- card builder ----
function simBar(sim, maxSim, color) {{
  const pct = Math.min(sim / Math.max(maxSim, 0.01) * 100, 100);
  return `<div class="bar-wrap"><div class="bar" style="width:${{pct}}%;background:${{color}}"></div></div>`;
}}

function renderMemItem(item, maxSim, color) {{
  return `<div class="mem-item">
    <span class="txt">${{esc(item.text)}}</span>
    ${{simBar(item.sim_q, maxSim, color)}}
    <span class="sim">${{item.sim_q.toFixed(4)}}</span>
  </div>`;
}}

function renderCard(rec, inMain) {{
  const goldenMax = Math.max(...rec.golden_memory.map(m=>m.sim_q), 0.01);
  const allSims = [...rec.golden_memory.map(m=>m.sim_q), ...rec.lowered_golden.map(m=>m.sim_q), ...rec.distractors.map(m=>m.sim_q)];
  const globalMax = Math.max(...allSims, 0.01);
  const loMin = rec.lowered_golden_min_sim;

  let goldenHtml = rec.golden_memory.map(m => renderMemItem(m, globalMax, 'var(--gold)')).join('');
  let loweredHtml = rec.lowered_golden.map((m,i) => {{
    const src = rec.golden_memory[m.source_idx];
    return `<div class="mem-item">
      <span class="txt">${{esc(m.text)}}</span>
      ${{simBar(m.sim_q, globalMax, 'var(--lowered)')}}
      <span class="sim">${{m.sim_q.toFixed(4)}} <span style="font-size:10px;color:var(--muted)">←golden#${{m.source_idx+1}} ${{src ? src.sim_q.toFixed(4) : '?'}}</span></span>
    </div>`;
  }}).join('');

  let distHtml = rec.distractors.map(m => {{
    const ok = m.sim_q > loMin;
    const color = ok ? 'var(--good)' : 'var(--bad)';
    return `<div class="mem-item">
      <span class="txt">${{esc(m.text)}}</span>
      ${{simBar(m.sim_q, globalMax, color)}}
      <span class="sim" style="color:${{ok?'var(--good)':'var(--bad)'}}">${{m.sim_q.toFixed(4)}}${{ok?' ✓':''}}</span>
    </div>`;
  }}).join('');

  // preview for collapsed state
  const distPreview = rec.distractors.map(m => {{
    const ok = m.sim_q > loMin;
    return `<span style="color:${{ok?'var(--good)':'var(--bad)'}}">${{m.sim_q.toFixed(3)}}${{ok?'✓':''}}</span>`;
  }}).join(' ');

  return `<div class="card" data-qid="${{esc(rec.question_id)}}" data-type="${{esc(rec.question_type)}}" data-set="${{inMain?'main':'partial'}}">
    <div class="card-header" onclick="this.parentElement.classList.toggle('open')">
      <div style="flex:1;min-width:0">
        <div class="q">${{esc(rec.question)}}</div>
        <div class="answer">✅ ${{esc(rec.answer)}}&nbsp;&nbsp;<span class="badge golden">${{rec.question_type}}</span></div>
        <div class="preview-dist">dist: ${{distPreview}}</div>
      </div>
      <div class="meta">
        <span class="constraint-tag ${{inMain?'ok':'ng'}}">${{inMain?'8/8 OK':'PARTIAL'}}</span>
        <div style="margin-top:4px;font-size:11px">min lowered: <b style="color:var(--bad)">${{loMin.toFixed(4)}}</b></div>
        <div style="font-size:10px;color:var(--muted)">${{rec.question_id}}</div>
        <div class="card-toggle">▶</div>
      </div>
    </div>
    <div class="card-body">
      <div class="mem-group">
        <h4><span class="badge golden">golden_memory</span> (${{rec.golden_memory.length}} 条)</h4>
        ${{goldenHtml || '<div class="empty">(无)</div>'}}
      </div>
      <div class="mem-group">
        <h4><span class="badge lowered">lowered_golden</span> (${{rec.lowered_golden.length}} 条, min_sim=${{loMin.toFixed(4)}})</h4>
        ${{loweredHtml || '<div class="empty">(无)</div>'}}
      </div>
      <div class="mem-group">
        <h4><span class="badge distractor">distractors</span> (${{rec.distractors.length}} 条, 约束: 每条 sim_q &gt; ${{loMin.toFixed(4)}})</h4>
        <div class="threshold-line" data-val="${{loMin.toFixed(4)}}"></div>
        ${{distHtml || '<div class="empty">(无)</div>'}}
        <div style="margin-top:4px;font-size:11px;color:var(--muted)">
          ${{rec.distractors.filter(d=>d.sim_q>loMin).length}}/${{rec.distractors.length}} 条满足 sim_q &gt; lowered_min
        </div>
      </div>
    </div>
  </div>`;
}}

function esc(s) {{
  if (typeof s !== 'string') return s;
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}

// ---- render all ----
function render() {{
  const typeFilter = document.getElementById('filter-type').value;
  const setFilter = document.getElementById('filter-set').value;
  const search = document.getElementById('filter-search').value.toLowerCase();

  let filtered = [];
  for (const rec of ALL) {{
    if (setFilter === 'main' && rec.constraint_ok !== true) continue;
    if (setFilter === 'partial' && rec.constraint_ok !== false) continue;
    if (typeFilter && rec.question_type !== typeFilter) continue;
    if (search) {{
      const haystack = JSON.stringify(rec).toLowerCase();
      if (!haystack.includes(search)) continue;
    }}
    const inMain = rec.constraint_ok === true;
    filtered.push({{rec, inMain}});
  }}
  document.getElementById('filter-count').textContent = `显示 ${{filtered.length}} / ${{ALL.length}} 题`;
  document.getElementById('cards').innerHTML = filtered.map(f => renderCard(f.rec, f.inMain)).join('');
}}
document.getElementById('filter-type').addEventListener('change', render);
document.getElementById('filter-set').addEventListener('change', render);
document.getElementById('filter-search').addEventListener('input', render);
render();
</script>
</body>
</html>'''

OUT = os.path.join(REPO, "docs/confusion_review.html")
with open(OUT, 'w', encoding='utf-8') as f:
    f.write(HTML)

size_mb = os.path.getsize(OUT) / (1024 * 1024)
print(f"✅ 已生成: {OUT} ({size_mb:.1f} MB)")
print(f"   用浏览器打开即可审查（自包含，无需服务器）")
