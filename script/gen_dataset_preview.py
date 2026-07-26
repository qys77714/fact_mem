"""生成 LME Hybrid Golden 数据集 HTML 预览."""

import json
from collections import Counter
from html import escape

DATA_PATH = "data/preprocessed/longmemeval_s_hybrid_golden.json"
OUT_PATH = "docs/lme_hybrid_golden_preview.html"


SKIP_FIELDS = {
    "haystack_sessions",           # 604 MB — 完整对话内容
    "_selected_filler_indices",    # ~0.1 MB — 内部索引
    "_selected_filler_keys",       # ~1.3 MB — filler 位置信息
    "_selected_filler_dates",      # ~0.4 MB — filler 日期
}


def load_data():
    with open(DATA_PATH) as f:
        data = json.load(f)
    # 只保留预览需要的轻量字段
    for item in data:
        for k in SKIP_FIELDS:
            item.pop(k, None)
    return data


def build_overview(data):
    """生成概览统计."""
    total = len(data)
    qt = Counter(d["question_type"] for d in data)
    ct = Counter(d["confusion_type"] for d in data)

    # distractor source 分类
    rel_types = Counter()
    non_rel = Counter()
    for item in data:
        for dist in item.get("distractors", []):
            src = dist.get("source", "unknown")
            if src.startswith("seed"):
                parts = src.split("_", 1)
                rel = parts[1] if len(parts) > 1 else "unknown"
                rel_types[rel] += 1
            else:
                non_rel[src] += 1

    gm_counts = Counter(len(d["golden_memory"]) for d in data)
    avg_gm = sum(len(d["golden_memory"]) for d in data) / total if total else 0

    return {
        "total": total,
        "question_types": qt,
        "confusion_types": ct,
        "rel_types": rel_types,
        "non_rel_sources": non_rel,
        "gm_counts": gm_counts,
        "avg_gm": avg_gm,
        "total_distractors": sum(rel_types.values()) + sum(non_rel.values()),
    }


def render_header(title):
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #f5f7fa;
  color: #1a1a2e;
  line-height: 1.6;
}}
.header {{
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
  color: #fff;
  padding: 32px 24px;
  text-align: center;
}}
.header h1 {{ font-size: 2rem; font-weight: 700; }}
.header .subtitle {{ color: #a0b4c8; margin-top: 6px; font-size: 0.9rem; }}
.container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
.section-title {{
  font-size: 1.25rem; font-weight: 700; margin: 28px 0 14px;
  padding-left: 12px; border-left: 4px solid #0f3460;
}}

/* 概览卡片 */
.overview-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 12px;
  margin-bottom: 20px;
}}
.ov-card {{
  background: #fff; border-radius: 10px; padding: 16px;
  text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,.06);
}}
.ov-card .num {{ font-size: 2rem; font-weight: 800; color: #0f3460; }}
.ov-card .label {{ font-size: 0.8rem; color: #888; margin-top: 2px; }}

/* 图表区 */
.charts-row {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
  gap: 16px;
  margin-bottom: 20px;
}}
.chart-box {{
  background: #fff; border-radius: 10px; padding: 16px;
  box-shadow: 0 1px 4px rgba(0,0,0,.06);
}}
.chart-box h3 {{ font-size: 0.95rem; margin-bottom: 10px; color: #333; }}
.chart-box canvas {{ max-height: 240px; }}
.chart-legend {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px; font-size: 0.8rem; }}
.chart-legend span {{ display: flex; align-items: center; gap: 4px; }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}

/* 搜索/过滤 */
.toolbar {{
  display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
  background: #fff; padding: 14px 16px; border-radius: 10px;
  margin-bottom: 14px; box-shadow: 0 1px 4px rgba(0,0,0,.06);
}}
.toolbar input, .toolbar select {{
  padding: 7px 12px; border: 1px solid #ddd; border-radius: 6px;
  font-size: 0.85rem; outline: none;
}}
.toolbar input:focus, .toolbar select:focus {{ border-color: #0f3460; }}
.toolbar input {{ flex: 1; min-width: 200px; }}

/* 表格 */
.table-wrap {{ overflow-x: auto; background: #fff; border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,.06); }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
thead th {{
  background: #f0f2f5; padding: 10px 12px; text-align: left;
  font-weight: 600; color: #555; border-bottom: 2px solid #ddd;
  cursor: pointer; user-select: none; white-space: nowrap;
}}
thead th:hover {{ background: #e4e8ee; }}
tbody td {{
  padding: 8px 12px; border-bottom: 1px solid #eee;
  max-width: 320px; overflow: hidden; text-overflow: ellipsis;
}}
tbody tr {{ cursor: pointer; transition: background .1s; }}
tbody tr:hover {{ background: #f7f9fc; }}
tbody tr.expanded {{ background: #eef2f8; }}

/* 展开详情 */
.detail-row {{ display: none; }}
.detail-row.show {{ display: table-row; }}
.detail-wrap {{
  padding: 16px 20px; background: #fafbfc;
  border-bottom: 2px solid #ddd;
}}
.detail-wrap h4 {{ font-size: 0.9rem; color: #0f3460; margin: 10px 0 6px; }}
.detail-wrap .mem-card {{
  background: #fff; border: 1px solid #e0e0e0; border-radius: 8px;
  padding: 10px 14px; margin: 6px 0; font-size: 0.8rem;
}}
.mem-card .meta {{ color: #999; font-size: 0.75rem; margin-top: 4px; }}
.mem-card.golden {{ border-left: 4px solid #27ae60; }}
.mem-card.distractor-eqv {{ border-left: 4px solid #f39c12; }}
.mem-card.distractor-osn {{ border-left: 4px solid #3498db; }}
.mem-card.distractor-nso {{ border-left: 4px solid #9b59b6; }}
.mem-card.distractor-con {{ border-left: 4px solid #e74c3c; }}
.mem-card.distractor-other {{ border-left: 4px solid #95a5a6; }}
.badge {{
  display: inline-block; padding: 2px 8px; border-radius: 12px;
  font-size: 0.72rem; font-weight: 600; color: #fff;
}}
.badge-eqv {{ background: #f39c12; }}
.badge-osn {{ background: #3498db; }}
.badge-nso {{ background: #9b59b6; }}
.badge-con {{ background: #e74c3c; }}
.badge-other {{ background: #95a5a6; }}
.badge-type-i {{ background: #2ecc71; }}
.badge-type-ii {{ background: #e67e22; }}
.badge-golden {{ background: #27ae60; }}
.badge-abstention {{ background: #c0392b; }}

.pagination {{ display: flex; justify-content: center; align-items: center; gap: 8px; padding: 16px; font-size: 0.82rem; }}
.pagination button {{
  padding: 6px 14px; border: 1px solid #ccc; border-radius: 6px;
  background: #fff; cursor: pointer; font-size: 0.8rem;
}}
.pagination button:disabled {{ opacity: .4; cursor: default; }}
.pagination button:hover:not(:disabled) {{ background: #f0f2f5; }}
.page-info {{ color: #666; }}
</style>
</head>
<body>

<div class="header">
  <h1>{title}</h1>
  <div class="subtitle">LongMemEval Hybrid Golden Dataset · 长对话记忆评测</div>
</div>
"""


COLORS = [
    "#0f3460", "#e76f51", "#2a9d8f", "#e9c46a", "#264653",
    "#f4a261", "#287271", "#8ecae6", "#219ebc", "#fb8500",
    "#6a4c93", "#1982c4", "#ff595e", "#8ac926",
]
COLORS_REL = {"eqv": "#f39c12", "osn": "#3498db", "nso": "#9b59b6", "con": "#e74c3c"}


def render_overview(stats):
    ov = stats
    total_gm = sum(ov["gm_counts"].values())
    total_dist = ov["total_distractors"]

    cards = [
        ("总题目数", ov["total"]),
        ("题型种类", len(ov["question_types"])),
        ("Golden Memory", f"{total_gm} 条  ({ov['avg_gm']:.1f}/题)"),
        ("Distractors", f"{total_dist} 条  ({total_dist / ov['total']:.1f}/题)"),
        ("可答题 (type_i)", ov["confusion_types"].get("type_i", 0)),
        ("知识更新 (type_ii)", ov["confusion_types"].get("type_ii", 0)),
    ]

    cards_html = "".join(
        f'<div class="ov-card"><div class="num">{v}</div><div class="label">{k}</div></div>'
        for k, v in cards
    )

    return f"""
<div class="container">
<div class="section-title">概览</div>
<div class="overview-grid">{cards_html}</div>
"""


def render_charts(stats, data):
    """生成 Chart.js 图表."""

    # 题型分布
    qt = stats["question_types"]
    qt_labels = [k for k, _ in qt.most_common()]
    qt_values = [v for _, v in qt.most_common()]

    # confusion_type
    ct = stats["confusion_types"]
    ct_labels = [k for k, _ in ct.most_common()]
    ct_values = [v for _, v in ct.most_common()]

    # Distractor 来源 - 关系类
    rt = stats["rel_types"]
    rt_labels = [k.upper() for k, _ in rt.most_common()]
    rt_values = [v for _, v in rt.most_common()]

    # 非关系来源
    nrs = stats["non_rel_sources"]
    nrs_labels = [k for k, _ in nrs.most_common()]
    nrs_values = [v for _, v in nrs.most_common()]

    # Golden memory 数分布
    gm = stats["gm_counts"]
    gm_labels = [str(k) for k in sorted(gm.keys())]
    gm_values = [gm[k] for k in sorted(gm.keys())]

    return f"""
<div class="charts-row">
  <div class="chart-box">
    <h3>题型分布 (question_type)</h3>
    <canvas id="chartQt"></canvas>
  </div>
  <div class="chart-box">
    <h3>Confusion 类型</h3>
    <canvas id="chartCt"></canvas>
  </div>
  <div class="chart-box">
    <h3>Distractor 关系类型</h3>
    <canvas id="chartRel"></canvas>
  </div>
  <div class="chart-box">
    <h3>Distractor 非关系来源</h3>
    <canvas id="chartNonRel"></canvas>
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
(function() {{
  const COLORS_QT = {json.dumps(COLORS[:len(qt_labels)])};
  const COLORS_REL = {json.dumps(COLORS_REL)};

  const commonOpts = {{
    responsive: true,
    maintainAspectRatio: true,
    plugins: {{ legend: {{ display: false }} }},
  }};

  function pie(canvasId, labels, values, bgColors) {{
    const ctx = document.getElementById(canvasId).getContext('2d');
    new Chart(ctx, {{
      type: 'doughnut',
      data: {{
        labels: labels,
        datasets: [{{ data: values, backgroundColor: bgColors }}]
      }},
      options: {{ ...commonOpts, cutout: '50%' }}
    }});
  }}

  function barH(canvasId, labels, values, color) {{
    const ctx = document.getElementById(canvasId).getContext('2d');
    new Chart(ctx, {{
      type: 'bar',
      data: {{
        labels: labels,
        datasets: [{{ data: values, backgroundColor: color }}]
      }},
      options: {{ ...commonOpts, indexAxis: 'y', scales: {{ x: {{ beginAtZero: true }} }} }}
    }});
  }}

  // 题型分布
  pie('chartQt', {json.dumps(qt_labels)}, {json.dumps(qt_values)}, COLORS_QT);

  // Confusion 类型
  pie('chartCt', {json.dumps(ct_labels)}, {json.dumps(ct_values)},
    ['#2ecc71', '#e67e22']);

  // 关系类型 - 柱状图
  barH('chartRel', {json.dumps(rt_labels)}, {json.dumps(rt_values)},
    {json.dumps([COLORS_REL.get(l, '#95a5a6') for l in rt_labels])});

  // 非关系来源
  pie('chartNonRel', {json.dumps(nrs_labels)}, {json.dumps(nrs_values)},
    COLORS_QT.slice().reverse());
}})();
</script>
"""


def relation_badge(src):
    """从 source 提取关系类型badge."""
    if src.startswith("seed"):
        parts = src.split("_", 1)
        rel = parts[1] if len(parts) > 1 else ""
        if rel in ("eqv", "osn", "nso", "con"):
            return f'<span class="badge badge-{rel}">{rel.upper()}</span>'
    return f'<span class="badge badge-other">{escape(src)}</span>'


def render_table(data):
    """生成搜索+表格."""

    qt_options = sorted(set(d["question_type"] for d in data))
    ct_options = sorted(set(d["confusion_type"] for d in data))

    return f"""
<div class="section-title">数据明细 (共 {len(data)} 题)</div>
<div class="toolbar">
  <input type="text" id="searchInput" placeholder="搜索 question / answer / question_id ..." oninput="filterTable()">
  <select id="filterQt" onchange="filterTable()">
    <option value="">全部题型</option>
    {"".join(f'<option value="{escape(q)}">{escape(q)}</option>' for q in qt_options)}
  </select>
  <select id="filterCt" onchange="filterTable()">
    <option value="">全部 Confusion</option>
    {"".join(f'<option value="{escape(c)}">{escape(c)}</option>' for c in ct_options)}
  </select>
  <span id="resultCount" style="color:#888;font-size:0.82rem;"></span>
</div>
<div class="table-wrap">
<table>
<thead>
<tr>
  <th onclick="sortTable(0)">#</th>
  <th>ID</th>
  <th>题型</th>
  <th>Confusion</th>
  <th>问题</th>
  <th>答案</th>
  <th>GM数</th>
  <th>Dist数</th>
  <th>D-Sources</th>
</tr>
</thead>
<tbody id="tableBody"></tbody>
</table>
</div>
<div class="pagination" id="pagination"></div>
"""


RENDER_FOOTER = """
<script>
const ALL_DATA = __DATA_PLACEHOLDER__;
const PER_PAGE = 25;
let currentPage = 1;
let filteredData = [...ALL_DATA];

function relationType(src) {
  if (src.startsWith('seed')) {
    const parts = src.split('_', 1);
    return parts[1] || src;
  }
  return src;
}

function distSummary(dists) {
  const c = {};
  dists.forEach(d => {
    const src = d.source || 'unknown';
    const rt = src.startsWith('seed') ? src.split('_',1)[1] : src;
    c[rt] = (c[rt] || 0) + 1;
  });
  return Object.entries(c).sort((a,b) => b[1]-a[1]).map(([k,v]) => {
    const label = k.length > 20 ? k.slice(0,20)+'…' : k;
    let cls = 'badge-other';
    if (k === 'eqv') cls = 'badge-eqv';
    else if (k === 'osn') cls = 'badge-osn';
    else if (k === 'nso') cls = 'badge-nso';
    else if (k === 'con') cls = 'badge-con';
    return `<span class="badge ${cls}">${label}:${v}</span>`;
  }).join(' ');
}

function buildRow(item, idx) {
  const realIdx = (currentPage - 1) * PER_PAGE + idx + 1;
  const dists = item.distractors || [];
  return `<tr onclick="toggleDetail(${idx})" id="row${idx}">
    <td>${realIdx}</td>
    <td style="font-family:monospace;font-size:0.75rem;">${esc(item.question_id || '')}</td>
    <td>${esc(item.question_type || '')}</td>
    <td><span class="badge badge-type-${(item.confusion_type||'').replace('_','-')}">${esc(item.confusion_type || '')}</span></td>
    <td>${esc((item.question||'').slice(0,80))}${(item.question||'').length > 80 ? '…' : ''}</td>
    <td style="font-weight:600;color:#0f3460;">${esc((item.answer||'').slice(0,60))}${(item.answer||'').length > 60 ? '…' : ''}</td>
    <td>${(item.golden_memory||[]).length}</td>
    <td>${dists.length}</td>
    <td>${distSummary(dists)}</td>
  </tr>
  <tr class="detail-row" id="detail${idx}"><td colspan="9"><div class="detail-wrap">
    <h4>⭐ Golden Memory</h4>
    ${(item.golden_memory||[]).map(gm => `
      <div class="mem-card golden">
        <div>${esc(gm.text || '')}</div>
        <div class="meta">sim_q: ${gm.sim_q ?? '—'} &nbsp;|&nbsp; date: ${esc(gm.date || '—')} &nbsp;|&nbsp; source: ${esc(gm.source || '—')}</div>
      </div>`).join('')}
    ${!item.golden_memory || item.golden_memory.length === 0 ? '<div style="color:#999;font-size:0.8rem;">(abstention)</div>' : ''}
    <h4>🎯 Distractors</h4>
    ${dists.map(d => {
      const src = d.source || '';
      let cls = 'distractor-other';
      if (src.includes('_eqv') || src === 'eqv') cls = 'distractor-eqv';
      else if (src.includes('_osn') || src === 'osn') cls = 'distractor-osn';
      else if (src.includes('_nso') || src === 'nso') cls = 'distractor-nso';
      else if (src.includes('_con') || src === 'con') cls = 'distractor-con';
      return `<div class="mem-card ${cls}">
        <div>${esc(d.text || '')}</div>
        <div class="meta">
          source: ${relationBadge(d.source || '')} &nbsp;|&nbsp;
          sim_q: ${typeof d.sim_q === 'number' ? d.sim_q.toFixed(5) : d.sim_q} &nbsp;|&nbsp;
          date: ${esc(d.date || '—')} &nbsp;|&nbsp;
          expected_wrong_answer: ${esc(String(d.expected_wrong_answer || '—'))}
        </div>
      </div>`;
    }).join('')}
    <h4>📋 完整信息</h4>
    <div style="font-size:0.78rem;color:#666;overflow-x:auto;">
      <b>Question (完整):</b> ${esc(item.question || '')}<br>
      <b>Answer:</b> ${esc(item.answer || '')}<br>
      <b>Answer Session IDs:</b> ${esc(JSON.stringify(item.answer_session_ids || []))}<br>
      <b>Question Date:</b> ${esc(item.question_date || '')}<br>
      <b>Golden Source:</b> ${esc(item.golden_source || '')}<br>
      <b>Embedding Model:</b> ${esc(item.embedding_model || '')}<br>
      <b>Haystack Sessions:</b> ${(item.haystack_session_ids || []).length} 条<br>
      <b>Filler Keys:</b> ${(item._selected_filler_keys || []).length} 条
    </div>
  </div></td></tr>`;
}

function esc(s) {
  if (typeof s !== 'string') return String(s);
  const el = document.createElement('span');
  el.textContent = s;
  return el.innerHTML;
}

function relationBadge(src) {
  if (src.startsWith('seed')) {
    const parts = src.split('_',1);
    const rel = parts[1] || '';
    if (['eqv','osn','nso','con'].includes(rel)) {
      return `<span class="badge badge-${rel}">${rel.toUpperCase()}</span>`;
    }
  }
  return `<span class="badge badge-other">${esc(src)}</span>`;
}

function toggleDetail(idx) {
  const detailRow = document.getElementById('detail' + idx);
  const row = document.getElementById('row' + idx);
  if (detailRow.classList.contains('show')) {
    detailRow.classList.remove('show');
    row.classList.remove('expanded');
  } else {
    // 关闭其他
    document.querySelectorAll('.detail-row.show').forEach(r => r.classList.remove('show'));
    document.querySelectorAll('tbody tr.expanded').forEach(r => r.classList.remove('expanded'));
    detailRow.classList.add('show');
    row.classList.add('expanded');
  }
}

function filterTable() {
  const search = (document.getElementById('searchInput').value || '').toLowerCase();
  const qt = document.getElementById('filterQt').value;
  const ct = document.getElementById('filterCt').value;

  filteredData = ALL_DATA.filter(item => {
    if (qt && item.question_type !== qt) return false;
    if (ct && item.confusion_type !== ct) return false;
    if (search) {
      const hay = [
        item.question_id, item.question, item.answer,
        ...(item.golden_memory||[]).map(g => g.text),
        ...(item.distractors||[]).map(d => d.text + d.source),
      ].join(' ').toLowerCase();
      if (!hay.includes(search)) return false;
    }
    return true;
  });
  currentPage = 1;
  renderPage();
  document.getElementById('resultCount').textContent = `显示 ${filteredData.length} / ${ALL_DATA.length} 题`;
}

function renderPage() {
  const totalPages = Math.ceil(filteredData.length / PER_PAGE) || 1;
  if (currentPage > totalPages) currentPage = totalPages;
  const start = (currentPage - 1) * PER_PAGE;
  const pageData = filteredData.slice(start, start + PER_PAGE);

  const tbody = document.getElementById('tableBody');
  tbody.innerHTML = pageData.map((item, i) => buildRow(item, i)).join('');

  // 分页
  const pag = document.getElementById('pagination');
  pag.innerHTML = `
    <button onclick="goPage(1)" ${currentPage <= 1 ? 'disabled' : ''}>««</button>
    <button onclick="goPage(currentPage-1)" ${currentPage <= 1 ? 'disabled' : ''}>«</button>
    <span class="page-info">第 ${currentPage} / ${totalPages} 页</span>
    <button onclick="goPage(currentPage+1)" ${currentPage >= totalPages ? 'disabled' : ''}>»</button>
    <button onclick="goPage(totalPages)" ${currentPage >= totalPages ? 'disabled' : ''}>»»</button>
  `;

  document.getElementById('resultCount').textContent =
    `显示 ${filteredData.length} / ${ALL_DATA.length} 题`;
}

function goPage(p) {
  const total = Math.ceil(filteredData.length / PER_PAGE) || 1;
  currentPage = Math.max(1, Math.min(p, total));
  renderPage();
  window.scrollTo({top: 0, behavior: 'smooth'});
}

let sortDir = {};
function sortTable(colIdx) {
  const KEY_MAP = ['', 'question_id', 'question_type', 'confusion_type', 'question', 'answer',
    (item) => (item.golden_memory||[]).length, (item) => (item.distractors||[]).length, null];
  const key = KEY_MAP[colIdx];
  if (key === null) return;
  const dir = sortDir[colIdx] = (sortDir[colIdx] || 0) === 1 ? -1 : 1;
  filteredData.sort((a, b) => {
    let va = typeof key === 'function' ? key(a) : (typeof a[key] === 'string' ? a[key] : String(a[key] || ''));
    let vb = typeof key === 'function' ? key(b) : (typeof b[key] === 'string' ? b[key] : String(b[key] || ''));
    if (typeof va === 'number' && typeof vb === 'number') return (va - vb) * dir;
    return String(va).localeCompare(String(vb)) * dir;
  });
  renderPage();
}

// init
filterTable();
</script>
</div>
</body>
</html>
"""


def main():
    print("Loading data...")
    data = load_data()
    stats = build_overview(data)

    print("Generating HTML...")
    data_json = json.dumps(data, ensure_ascii=False)

    html = (
        render_header("LME Hybrid Golden Dataset Preview")
        + render_overview(stats)
        + render_charts(stats, data)
        + render_table(data)
        + RENDER_FOOTER.replace("__DATA_PLACEHOLDER__", data_json)
    )

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Done → {OUT_PATH}")
    print(f"  文件大小: {len(html) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
