# 主实验：Unified Confusion Memory 鲁棒性评测 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 unified confusion 实验候选目录 → 运行 6 方法对比实验 → 汇总 N-accuracy 退化曲线

**Architecture:** 复用现有 Phase 3 实验流水线（build filler → build candidates → gen configs → run ingest/generate/evaluate → aggregate），适配统一数据集 `longmemeval_s_unified_confusion.json`，新增 N=6 档位，全部 6 方法对比（skip zep 初始），token_limit=256，RD 使用 LLM-only + v3 classification prompt

**Tech Stack:** Python (pipeline scripts) + gemma4-26B (ingest/answer) + qwen3-max (judge) + qwen3-embedding-0.6b (retrieval)

## 全局约束

- 所有脚本用 `uv run --no-sync python` 执行（裸 python 缺依赖）
- `PYTHONPATH=src`（直接调 pipeline 脚本时）
- 模型服务：gemma4-26B 在 7111，qwen3-embedding-0.6b 在 7110
- RD 只用 LLM-only 版本（不做 classifier 加速，不做 LLM 校验）
- RD classification system prompt: `lme_relation_classification_system_en_v3.jinja`
- memory_token_limit=256（非 top-k），retrieve_topk=50（足够大，由 token limit 截断）
- Judge: qwen3-max
- **先不跑 zep**（容易挂）
- Filler: 每题从非 evidence session 均匀采样 50 条
- N 档: 0, 2, 4, 6, 8（主实验 N=8）

---

## 文件结构

```
data/preprocessed/
  longmemeval_s_unified_confusion.json   # 已生成，470 题统一数据集
  lme_s_non_evidence_filler.json         # 待生成，filler 记忆（适配 unified 路径）

MemDB/candidates/
  lme_s_gemma4-26B_unified_filler_N0/    # 待构建
  lme_s_gemma4-26B_unified_filler_N2/
  lme_s_gemma4-26B_unified_filler_N4/
  lme_s_gemma4-26B_unified_filler_N6/
  lme_s_gemma4-26B_unified_filler_N8/

config/
  lme_unified_filler_N0.yaml             # 待生成
  lme_unified_filler_N2.yaml
  lme_unified_filler_N4.yaml
  lme_unified_filler_N6.yaml
  lme_unified_filler_N8.yaml

script/
  build_unified_filler.py                # 新建：从 unified 数据集提取 filler
  build_unified_candidates.py            # 新建：构建 unified 候选目录
  gen_unified_configs.py                 # 新建：生成 unified 实验 config
  aggregate_unified_results.py           # 新建：汇总 unified 实验结果

experiment/
  lme_s_candunified_filler_N{n}_gemma4-26B_gemma4-26B_tl256_unified_filler_N{n}/
    pred_{method}.jsonl                  # 各方法预测
    eval_judge.json                      # Judge 评分
```

---

### Task 1: 构建 Unified Filler 文件

**Files:**
- Create: `script/build_unified_filler.py`（基于 `build_confusion_filler.py` 改造）

**Interfaces:**
- 输入：`data/preprocessed/longmemeval_s_unified_confusion.json` + `MemDB/candidates/lme_s_gemma4-26B_0615_unified/`
- 输出：`data/preprocessed/lme_s_non_evidence_filler.json`

**与原始 `build_confusion_filler.py` 的区别：**
- 使用统一数据集路径（非 confusion v3）
- 适配 `gp_4_*` 文件名的 `gpt4_*` question_id 映射

- [ ] **Step 1: 创建 `script/build_unified_filler.py`**

```python
#!/usr/bin/env python3
"""
从 unified confusion 数据集中抽取非 evidence session 的 filler 记忆。

用法:
  uv run --no-sync python script/build_unified_filler.py

输入:
  - data/preprocessed/longmemeval_s_unified_confusion.json  (470 题，含 answer_session_ids)
  - MemDB/candidates/lme_s_gemma4-26B_0615_unified/        (candidate JSON 文件)

输出:
  - data/preprocessed/lme_s_non_evidence_filler.json
    {qid: {"filler_chunks": [{session_index, session_date, candidate_memories}, ...],
           "filler_memory_count": N}, ...}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_CAND_DIR = _REPO / "MemDB" / "candidates" / "lme_s_gemma4-26B_0615_unified"
_UNIFIED_PATH = _REPO / "data" / "preprocessed" / "longmemeval_s_unified_confusion.json"
_OUT_PATH = _REPO / "data" / "preprocessed" / "lme_s_non_evidence_filler.json"


def _candidate_path(qid: str) -> Path:
    """question_id → candidate JSON 文件路径。

    gpt4_* 前缀的 qid 映射到 gp_4_* 文件名。
    """
    if qid.startswith("gpt4_"):
        return _CAND_DIR / f"gp_4_{qid[5:]}.json"
    return _CAND_DIR / f"{qid}.json"


def main() -> None:
    with open(_UNIFIED_PATH) as f:
        unified_data = json.load(f)

    filler: dict[str, dict] = {}
    missing = 0
    total_filler_memories = 0

    for item in unified_data:
        qid = item["question_id"]
        answer_sids = set(item.get("answer_session_ids", []))
        hs_ids = item.get("haystack_session_ids", [])

        # 找出 evidence session_index
        evidence_si: set[int] = set()
        for i, sid in enumerate(hs_ids):
            if sid in answer_sids:
                evidence_si.add(i + 1)  # session_index = i+1

        cand_path = _candidate_path(qid)
        if not cand_path.exists():
            print(f"WARNING: 缺少 candidate 文件: {cand_path}", file=sys.stderr)
            missing += 1
            filler[qid] = {"filler_chunks": [], "filler_memory_count": 0}
            continue

        with open(cand_path) as f:
            cand = json.load(f)

        # 过滤: 只保留非 evidence session 的 chunks
        filler_chunks: list[dict] = []
        filler_count = 0
        for chunk in cand.get("chunks", []):
            si = chunk.get("session_index")
            if si in evidence_si:
                continue
            mems = chunk.get("candidate_memories", [])
            if not mems:
                continue
            filler_chunks.append({
                "session_index": si,
                "session_date": chunk.get("session_date", ""),
                "candidate_memories": mems,
            })
            filler_count += len(mems)

        filler[qid] = {
            "filler_chunks": filler_chunks,
            "filler_memory_count": filler_count,
        }
        total_filler_memories += filler_count

    with open(_OUT_PATH, "w") as f:
        json.dump(filler, f, ensure_ascii=False, indent=2)

    print(f"完成: {len(unified_data)} 题")
    print(f"  缺失 candidate: {missing}")
    print(f"  非 evidence filler 总记忆数: {total_filler_memories}")
    print(f"  平均每题 filler: {total_filler_memories / max(len(unified_data) - missing, 1):.1f}")
    print(f"  输出: {_OUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行脚本生成 filler**

```bash
cd /data/zjj/project_26/fact_mem && uv run --no-sync python script/build_unified_filler.py
```

预期输出：470 题全覆盖，平均每题 filler ~50-100 条（原始 candidate 的全量 filler），生成 `data/preprocessed/lme_s_non_evidence_filler.json`

- [ ] **Step 3: 验证 filler 文件**

```bash
python3 -c "
import json
with open('data/preprocessed/lme_s_non_evidence_filler.json') as f:
    d = json.load(f)
print(f'题目数: {len(d)}')
counts = [v['filler_memory_count'] for v in d.values()]
print(f'filler 数量: min={min(counts)}, max={max(counts)}, mean={sum(counts)/len(counts):.0f}')
zero = sum(1 for c in counts if c == 0)
print(f'零 filler: {zero} 题')
"
```

预期：零 filler 题数 ≤ 5

- [ ] **Step 4: Commit**

```bash
git add script/build_unified_filler.py data/preprocessed/lme_s_non_evidence_filler.json
git commit -m "feat: build unified filler from unified confusion dataset"
```

---

### Task 2: 构建 Unified 候选目录 (N=0,2,4,6,8)

**Files:**
- Create: `script/build_unified_candidates.py`（基于 `build_confusion_candidates.py` 改造）

**Interfaces:**
- 输入：`data/preprocessed/longmemeval_s_unified_confusion.json` + `data/preprocessed/lme_s_non_evidence_filler.json`
- 输出：5 个候选目录 `MemDB/candidates/lme_s_gemma4-26B_unified_filler_N{n}/`（n=0,2,4,6,8）
- 每个目录含 470 个 `{qid}.json` + `extract_progress.state`

**与原始 `build_confusion_candidates.py` 的区别：**
- 使用统一数据集（非 confusion v3）
- 新增 N=6 档位
- 适配 `gpt4_*` question_id 的文件名映射
- 黄金记忆选择逻辑不变（优先 lowered_golden）
- Type II distractor 自带 `expected_wrong_answer`，候选构建时保留该字段

- [ ] **Step 1: 创建 `script/build_unified_candidates.py`**

```python
#!/usr/bin/env python3
"""
基于 unified confusion 数据集 + non-evidence filler，构建 N∈{0,2,4,6,8} 条 distractor 的实验候选目录。

用法:
  uv run --no-sync python script/build_unified_candidates.py [--distractors 0,2,4,6,8]

输入:
  - data/preprocessed/longmemeval_s_unified_confusion.json
  - data/preprocessed/lme_s_non_evidence_filler.json

输出 (5 个目录):
  - MemDB/candidates/lme_s_gemma4-26B_unified_filler_N0/
  - MemDB/candidates/lme_s_gemma4-26B_unified_filler_N2/
  - MemDB/candidates/lme_s_gemma4-26B_unified_filler_N4/
  - MemDB/candidates/lme_s_gemma4-26B_unified_filler_N6/
  - MemDB/candidates/lme_s_gemma4-26B_unified_filler_N8/

每个目录含 470 个 {qid}.json + extract_progress.state
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_UNIFIED_PATH = _REPO / "data" / "preprocessed" / "longmemeval_s_unified_confusion.json"
_FILLER_PATH = _REPO / "data" / "preprocessed" / "lme_s_non_evidence_filler.json"
_CAND_BASE = _REPO / "MemDB" / "candidates"

_CAND_PREFIX = "lme_s_gemma4-26B"

EXTRACT_PROGRESS_VERSION = 5
EXTRACT_PROGRESS_KIND = "lme_candidate_extract_progress"


def _parse_date(date_str: str) -> datetime:
    """统一解析多种日期格式。"""
    s = (date_str or "").strip()
    if not s:
        return datetime(2000, 1, 1)
    m = re.match(r"(\d{4})/(\d{2})/(\d{2})\s*\([^)]+\)\s*(\d{2}):(\d{2})", s)
    if m:
        return datetime(int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5]))
    m = re.match(r"(\d{4})/(\d{2})/(\d{2})", s)
    if m:
        return datetime(int(m[1]), int(m[2]), int(m[3]))
    return datetime(2000, 1, 1)


def _select_golden(item: dict) -> list[dict]:
    """选择 golden memory: 优先 lowered_golden，回退到 golden_memory。"""
    status = item.get("lowering_status", "full_fallback")
    if status != "full_fallback":
        lowered = item.get("lowered_golden", [])
        if lowered:
            return [
                {"text": lg["text"], "date": lg.get("date", ""), "source": "lowered_golden",
                 "source_idx": lg.get("source_idx", 0)}
                for lg in lowered
            ]
    golden = item.get("golden_memory", [])
    return [
        {"text": gm["text"], "date": gm.get("date", ""), "source": "golden_memory"}
        for gm in golden
    ]


def _select_distractors(item: dict, n: int) -> list[dict]:
    """取前 N 条 distractor（已按 sim_q 降序排列）。保留 expected_wrong_answer 字段。"""
    dists = item.get("distractors", [])
    selected = dists[:n]
    result = []
    for d in selected:
        entry = {
            "text": d["text"],
            "date": d.get("date", ""),
            "source": d.get("source", "distractor"),
            "source_idx": d.get("source_idx", 0),
        }
        # Type II distractor 带有 expected_wrong_answer
        ewa = d.get("expected_wrong_answer")
        if ewa is not None:
            entry["expected_wrong_answer"] = ewa
        result.append(entry)
    return result


def _build_chunks(goldens, distractors, filler_chunks, max_filler: int = 0):
    """合并 golden + distractor + filler，按日期排序，重建 chunk_index 和 session_index。"""
    timed_items: list[tuple[datetime, str, dict]] = []

    for g in goldens:
        dt = _parse_date(g["date"])
        timed_items.append((dt, f"golden_{g.get('source_idx', hash(g['text']))}", {
            "type": "golden", "text": g["text"], "date": g["date"]
        }))

    for d in distractors:
        dt = _parse_date(d["date"])
        timed_items.append((dt, f"dist_{d.get('source', '')}_{d.get('source_idx', 0)}", {
            "type": "distractor", "text": d["text"], "date": d["date"]
        }))

    # Filler: 扁平化后按时序均匀采样
    filler_flat: list[tuple[datetime, str, str]] = []
    for fc in filler_chunks:
        dt = _parse_date(fc["session_date"])
        date_str = fc.get("session_date", "")
        for mem_text in fc.get("candidate_memories", []):
            filler_flat.append((dt, mem_text, date_str))
    filler_flat.sort(key=lambda x: x[0])

    if max_filler == 0:
        filler_flat = []
    elif max_filler > 0 and len(filler_flat) > max_filler:
        step = (len(filler_flat) - 1) / (max_filler - 1) if max_filler > 1 else 0
        indices = [round(i * step) for i in range(max_filler)]
        filler_flat = [filler_flat[idx] for idx in indices]

    for dt, mem_text, date_str in filler_flat:
        timed_items.append((dt, f"filler_{hash(mem_text)}", {
            "type": "filler", "text": mem_text, "date": date_str
        }))

    # 按日期排序；同日期 golden 在最后
    type_order = {"filler": 0, "distractor": 1, "golden": 2}
    timed_items.sort(key=lambda x: (x[0], type_order.get(x[2]["type"], 1)))

    chunks = []
    for chunk_idx, (dt, uid, entry) in enumerate(timed_items):
        chunks.append({
            "chunk_index": chunk_idx,
            "session_index": chunk_idx + 1,
            "turn_start": 0,
            "turn_end": 0,
            "turn_overlap": 0,
            "session_date": entry["date"],
            "candidate_memories": [entry["text"]],
            "parse_error": None,
        })

    return chunks, len(goldens), len(distractors), len(filler_flat)


def _candidate_filename(qid: str) -> str:
    """question_id → 候选 JSON 文件名。gpt4_* → gp_4_*"""
    if qid.startswith("gpt4_"):
        return f"gp_4_{qid[5:]}.json"
    return f"{qid}.json"


def build_one_config(
    conf_data: list[dict],
    filler_data: dict[str, dict],
    n_distractors: int,
    output_dir: Path,
    max_filler: int = 0,
) -> None:
    """为指定 distractor 数量构建候选目录。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    total_golden = total_dist = total_filler = 0

    for item in conf_data:
        qid = item["question_id"]

        goldens = _select_golden(item)
        distractors = _select_distractors(item, n_distractors)
        filler_info = filler_data.get(qid, {})
        filler_chunks = filler_info.get("filler_chunks", [])

        chunks, n_g, n_d, n_f = _build_chunks(
            goldens, distractors, filler_chunks, max_filler=max_filler
        )
        total_golden += n_g
        total_dist += n_d
        total_filler += n_f

        out = {
            "history_name": qid,
            "model": "gemma4-26B",
            "memory_granularity": 4,
            "turn_overlap": 0,
            "dialogue_format": "user_assistant",
            "chunks": chunks,
        }

        out_path = output_dir / _candidate_filename(qid)
        with open(out_path, "w") as f:
            json.dump(out, f, ensure_ascii=False)

        completed.append(qid)

    # 写 extract_progress.state
    progress = {
        "version": EXTRACT_PROGRESS_VERSION,
        "kind": EXTRACT_PROGRESS_KIND,
        "config": {
            "model": "gemma4-26B",
            "memory_granularity": "4",
            "turn_overlap": 0,
            "dialogue_format": "user_assistant",
            "prompt_template": "0_mem_extract_v2.jinja",
            "mem_extract_extra_templates": ["0_mem_extract_aspect_unified_en.jinja"],
            "mem_extract_aspects_only": True,
            "use_json_schema": True,
            "max_new_tokens": 2048,
            "note": f"unified confusion experiment, N={n_distractors} distractors",
        },
        "completed": completed,
    }
    with open(output_dir / "extract_progress.state", "w") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

    # 统计 Type I / Type II 分布
    type_i = sum(1 for item in conf_data if item.get("confusion_type") == "type_i")
    type_ii = sum(1 for item in conf_data if item.get("confusion_type") == "type_ii")
    print(f"  N={n_distractors}: {len(completed)} 题 (Type I={type_i}, Type II={type_ii}), "
          f"golden={total_golden}, dist={total_dist}, filler={total_filler}")


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 unified confusion 实验候选目录")
    parser.add_argument("--distractors", default="0,2,4,6,8",
                        help="逗号分隔的 distractor 数量 (default: 0,2,4,6,8)")
    parser.add_argument("--max-filler", type=int, default=50,
                        help="每题最多 filler 记忆数 (default: 50)")
    parser.add_argument("--suffix-prefix", default="unified_filler",
                        help="候选目录 suffix 前缀 (default: unified_filler)")
    args = parser.parse_args()

    ns = [int(x.strip()) for x in args.distractors.split(",")]

    with open(_UNIFIED_PATH) as f:
        conf_data = json.load(f)
    with open(_FILLER_PATH) as f:
        filler_data = json.load(f)

    print(f"Unified 数据: {len(conf_data)} 题 ({_UNIFIED_PATH})")
    print(f"Filler: {len(filler_data)} 题")
    print(f"Max filler/episode: {args.max_filler}")
    print(f"N 档: {ns}")

    output_base = _CAND_BASE
    prefix = args.suffix_prefix
    for n in ns:
        output_dir = output_base / f"{_CAND_PREFIX}_{prefix}_N{n}"
        print(f"\n构建 N={n} → {output_dir}")
        build_one_config(conf_data, filler_data, n, output_dir,
                         max_filler=args.max_filler)

    print("\n完成。")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行脚本构建候选目录**

```bash
cd /data/zjj/project_26/fact_mem && uv run --no-sync python script/build_unified_candidates.py
```

预期输出：5 个候选目录，每个 470 题。

- [ ] **Step 3: 验证候选目录**

```bash
python3 -c "
import json, os
for n in [0, 2, 4, 6, 8]:
    d = f'MemDB/candidates/lme_s_gemma4-26B_unified_filler_N{n}'
    files = [f for f in os.listdir(d) if f.endswith('.json')]
    state = os.path.join(d, 'extract_progress.state')
    print(f'N={n}: {len(files)} json files, state={os.path.exists(state)}')
    # 抽样检查一个文件
    if files:
        with open(os.path.join(d, files[0])) as f:
            item = json.load(f)
        chunks = item['chunks']
        goldens = [c for c in chunks if 'golden' in c['candidate_memories'][0][:20] or True]
        print(f'  sample: {len(chunks)} chunks, qid={item[\"history_name\"]}')
"
```

预期：N=0 少量 chunk（仅 golden+filler），N=8 最多 chunk。

- [ ] **Step 4: Commit**

```bash
git add script/build_unified_candidates.py
git commit -m "feat: build unified candidate directories for N=0,2,4,6,8"
```

---

### Task 3: 生成实验 Config YAML

**Files:**
- Create: `script/gen_unified_configs.py`

**Interfaces:**
- 输出：`config/lme_unified_filler_N{n}.yaml`（n=0,2,4,6,8）

**关键配置：**
- 5 方法启用：add_all, relation_decision, mem0, evermemos, amac（zep 关闭）
- RD: backend=llm, `lme_relation_classification_system_en_v3.jinja`
- memory_token_limit=256, retrieve_topk=50
- judge=qwen3-max

- [ ] **Step 1: 创建 `script/gen_unified_configs.py`**

```python
#!/usr/bin/env python3
"""
为 unified confusion memory 实验批量生成 config YAML (N=0,2,4,6,8)。

用法:
  uv run --no-sync python script/gen_unified_configs.py [--distractors 0,2,4,6,8]

输出:
  - config/lme_unified_filler_N0.yaml
  - config/lme_unified_filler_N2.yaml
  - config/lme_unified_filler_N4.yaml
  - config/lme_unified_filler_N6.yaml
  - config/lme_unified_filler_N8.yaml

每个 config 启用 add_all + relation_decision + mem0 + evermemos + amac (zep 关闭)。
"""

from __future__ import annotations

import argparse
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

YAML_TEMPLATE = """\
# fact_memory LME Unified Confusion Memory 对比实验配置 (N={n})
# 自动生成: script/gen_unified_configs.py
# 用法: PYTHONPATH=src uv run --no-sync python run_exp_lme.py --config config/lme_unified_filler_N{n}.yaml --stages ingest,generate,evaluate

experiment:
  benchmark: lme_s
  suffix: unified_filler_N{n}

models:
  extract: gemma4-26B
  manager: gemma4-26B
  answer: gemma4-26B
  judge: qwen3-max
  embedding: qwen3-embedding-0.6b

extract:
  candidate_suffix: unified_filler_N{n}   # 指向预制候选目录 MemDB/candidates/lme_s_gemma4-26B_unified_filler_N{n}/
  granularity: 4
  turn_overlap: 0
  language: en
  aspect_templates:
    - "0_mem_extract_aspect_unified_en.jinja"

methods:
  add_all:
    enabled: true

  relation_decision:
    enabled: true
    backend: llm                       # LLM-only，不经过 classifier 加速
    related_top_k: 3
    fusion_model: ""
    cascade_enabled: false
    deletion_enabled: false
    topic_aggregation_enabled: false
    condition_sim_threshold: 0.5
    pairwise_sim_threshold: 0.5

  mem0:
    enabled: true
    related_top_k: 3
    related_aggregate_max: 10

  evermemos:
    enabled: true
    similarity_threshold: 0.65
    max_time_gap_days: 7.0

  amac:
    enabled: true
    threshold: 0.55
    weights: "0.1,0.1,0.1,0.1,0.6"
    skip_utility: false
    recency_decay_per_step: 0.12
    novelty_max_existing: 64

  zep:
    enabled: false

generate:
  retrieve_topk: 50
  memory_token_limit: 256
  answer_stratified_sample: 0          # 0 = 全量 470 题
  answer_sample_seed: 43
  show_memory_time: true

  hybrid:
    enabled: true
    dense_weight: 0.8
    bm25_weight: 0.2
    pool_mult: 4

evaluate:
  use_cot: true
  judge_stratified_sample: 0
  judge_sample_seed: 43

parallel:
  extract_chunk_concurrency: 100
  ingest_relation_concurrency: 50
  ingest_episode_concurrency:
    relation_decision: 40
    mem0: 50
    add_all: 100
    zep: 50
    amac: 100
    evermemos: 5
    fusion_episodes: 100
    fusion_packages: 10
  generate_parallel_episodes: 50
  generate_answer_concurrency: 2
  evaluate_max_concurrency: 8

token_limits:
  extract_max_new_tokens: 2048
  ingest_relation_max_new_tokens: 256
  ingest_manager_max_new_tokens: 2048
  fusion_max_new_tokens: 512
  evaluate_max_new_tokens: 512

debug:
  evaluate_print_one_sample: false

prompts:
  relation_system_en: "lme_relation_classification_system_en_v3.jinja"
  relation_system_zh: "lme_relation_classification_system_zh_v2.jinja"
  relation_user: "lme_relation_classification_user.jinja"
  fusion_bundle_en: "fuse_memory_bundle_en_v3.jinja"
  fusion_bundle_zh: ""
  fusion_edge_labels_en: "fuse_memory_bundle_edge_labels_en_v2.jinja"
  fusion_edge_labels_zh: "fuse_memory_bundle_edge_labels_zh_v2.jinja"
  judge_oqa: "pipeline_eval_oqa.jinja"
  judge_mcq: "pipeline_eval_mcq.jinja"
  judge_system: "pipeline_eval_system.jinja"
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 unified confusion 实验 config")
    parser.add_argument("--distractors", default="0,2,4,6,8",
                        help="逗号分隔的 distractor 数量 (default: 0,2,4,6,8)")
    args = parser.parse_args()

    ns = [int(x.strip()) for x in args.distractors.split(",")]
    config_dir = _REPO / "config"

    for n in ns:
        yaml_content = YAML_TEMPLATE.format(n=n)
        yaml_content = yaml_content.strip() + "\n"
        out_path = config_dir / f"lme_unified_filler_N{n}.yaml"
        with open(out_path, "w") as f:
            f.write(yaml_content)
        print(f"生成: {out_path}")

    print(f"\n完成。{len(ns)} 个 config 已写入 {config_dir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行脚本生成 config**

```bash
cd /data/zjj/project_26/fact_mem && uv run --no-sync python script/gen_unified_configs.py
```

预期：生成 5 个 YAML 文件。

- [ ] **Step 3: 验证 config 关键字段**

```bash
python3 -c "
import yaml
for n in [0,2,4,6,8]:
    with open(f'config/lme_unified_filler_N{n}.yaml') as f:
        cfg = yaml.safe_load(f)
    methods = cfg['methods']
    enabled = [m for m in ['add_all','relation_decision','mem0','evermemos','amac','zep'] if methods[m]['enabled']]
    print(f'N={n}: enabled={enabled}, tl={cfg[\"generate\"][\"memory_token_limit\"]}, judge={cfg[\"models\"][\"judge\"]}, rd_prompt={cfg[\"prompts\"][\"relation_system_en\"]}')
"
```

预期：add_all, relation_decision, mem0, evermemos, amac 启用；zep 关闭；tl=256；judge=qwen3-max；rd_prompt=v3。

- [ ] **Step 4: Commit**

```bash
git add script/gen_unified_configs.py config/lme_unified_filler_N*.yaml
git commit -m "feat: generate unified experiment configs for N=0,2,4,6,8"
```

---

### Task 4: 运行 Ingest（所有 N 档，5 个方法）

**注意**：Ingest 是实验中最耗时的阶段。每个 N 档需要独立运行，但各 N 档之间可以并行。

- [ ] **Step 1: 确认模型服务正常**

```bash
curl -s http://localhost:7111/health | head -1
curl -s http://localhost:7110/health | head -1
```

- [ ] **Step 2: 运行 N=0 ingest（最轻量，验证流水线）**

```bash
cd /data/zjj/project_26/fact_mem && PYTHONPATH=src uv run --no-sync python run_exp_lme.py \
  --config config/lme_unified_filler_N0.yaml \
  --stages ingest
```

预期：5 个方法依次完成 ingest（add_all 最快，evermemos 最慢）。

- [ ] **Step 3: 验证 ingest 产出**

```bash
python3 -c "
import os
for method in ['add_all', 'relation_decision', 'mem0', 'evermemos', 'amac']:
    base = 'MemDB/ingest/lme_s_candunified_filler_N0_gemma4-26B_unified_filler_N0'
    d = os.path.join(base, method)
    if os.path.exists(d):
        ready_files = sum(1 for f in os.listdir(d) if 'memory_ready' in f)
        print(f'{method}: {ready_files} ready markers')
    else:
        print(f'{method}: directory missing')
"
```

- [ ] **Step 4: 如果 N=0 成功，运行 N=2,4,6,8 ingest（可并行）**

```bash
# 分多个终端或后台运行
for n in 2 4 6 8; do
  PYTHONPATH=src uv run --no-sync python run_exp_lme.py \
    --config config/lme_unified_filler_N${n}.yaml \
    --stages ingest &
done
wait
```

如果 GPU 资源有限（gemma4-26B 用 4 GPU），串行运行更安全：

```bash
for n in 2 4 6 8; do
  echo "=== Ingest N=${n} ==="
  PYTHONPATH=src uv run --no-sync python run_exp_lme.py \
    --config config/lme_unified_filler_N${n}.yaml \
    --stages ingest
done
```

- [ ] **Step 5: Commit（ingest 目录不提交，仅记录）**

---

### Task 5: 运行 Generate + Evaluate（所有 N 档）

- [ ] **Step 1: 运行 N=0 generate + evaluate（验证端到端流水线）**

```bash
cd /data/zjj/project_26/fact_mem && PYTHONPATH=src uv run --no-sync python run_exp_lme.py \
  --config config/lme_unified_filler_N0.yaml \
  --stages generate,evaluate
```

- [ ] **Step 2: 验证 N=0 结果**

```bash
python3 -c "
import json
from pathlib import Path
base = Path('experiment')
matches = list(base.glob('lme_s_candunified_filler_N0_*tl256*'))
if matches:
    eval_path = matches[0] / 'eval_judge.json'
    with open(eval_path) as f:
        records = json.load(f)
    for rec in records:
        stem = Path(rec['input_path']).stem.replace('pred_', '')
        print(f'{stem}: {rec[\"overall_accuracy\"]:.2%}')
else:
    print('未找到实验目录')
"
```

预期：N=0 时各方法 accuracy 接近（~82-94%，无 distractor 挤占）。

- [ ] **Step 3: 如果 N=0 成功，运行 N=2,4,6,8 generate + evaluate**

```bash
for n in 2 4 6 8; do
  echo "=== Generate+Evaluate N=${n} ==="
  PYTHONPATH=src uv run --no-sync python run_exp_lme.py \
    --config config/lme_unified_filler_N${n}.yaml \
    --stages generate,evaluate
done
```

---

### Task 6: 汇总结果

**Files:**
- Create: `script/aggregate_unified_results.py`（基于 `aggregate_confusion_results.py` 改造）

- [ ] **Step 1: 创建 `script/aggregate_unified_results.py`**

```python
#!/usr/bin/env python3
"""
汇总 unified confusion memory 对比实验结果，生成 N-accuracy 对比表。

用法:
  uv run --no-sync python script/aggregate_unified_results.py [--base experiment/] [--distractors 0,2,4,6,8]

查找路径:
  experiment/lme_s_candunified_filler_N{n}_gemma4-26B_gemma4-26B_tl256_unified_filler_N{n}/eval_judge.json

输出:
  - 终端表格: N vs method vs accuracy
  - (可选) CSV 导出
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def find_eval_files(base: Path, ns: list[int]) -> dict[int, Path]:
    """查找各 N 对应的 eval_judge.json。"""
    found: dict[int, Path] = {}
    for n in ns:
        pattern = f"lme_s_candunified_filler_N{n}_"
        matches = list(base.glob(f"{pattern}*"))
        if matches:
            eval_path = matches[0] / "eval_judge.json"
            if eval_path.exists():
                found[n] = eval_path
            else:
                print(f"WARNING: 目录存在但无 eval_judge.json: {matches[0]}")
        else:
            print(f"WARNING: N={n} 未找到实验目录 (pattern: {pattern}*)")
    return found


def load_accuracies(eval_path: Path) -> dict[str, float]:
    """从 eval_judge.json 提取各方法的 overall_accuracy。"""
    with open(eval_path) as f:
        records = json.load(f)

    acc: dict[str, float] = {}
    for rec in records:
        input_path = rec.get("input_path", "")
        stem = Path(input_path).stem
        if stem.startswith("pred_"):
            method = stem[5:]
        else:
            method = stem
        acc[method] = rec.get("overall_accuracy", 0.0)
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 unified confusion 实验结果")
    parser.add_argument("--base", default="experiment", help="实验输出根目录")
    parser.add_argument("--distractors", default="0,2,4,6,8",
                        help="逗号分隔的 distractor 数量")
    parser.add_argument("--csv", default="", help="可选 CSV 导出路径")
    args = parser.parse_args()

    ns = [int(x.strip()) for x in args.distractors.split(",")]
    base = Path(args.base)

    eval_files = find_eval_files(base, ns)
    if not eval_files:
        print("ERROR: 未找到任何 eval_judge.json 文件", file=sys.stderr)
        sys.exit(1)

    all_methods: set[str] = set()
    results: dict[int, dict[str, float]] = {}
    for n in sorted(eval_files):
        acc = load_accuracies(eval_files[n])
        results[n] = acc
        all_methods.update(acc.keys())

    # 排序: RD 优先，add_all 其次，其他字母序
    priority = {"relation_decision": 0, "add_all": 1}
    methods = sorted(all_methods, key=lambda m: (priority.get(m, 99), m))

    # 打印表格
    print()
    header = f"{'N':>4s}"
    for m in methods:
        header += f"  {m:>18s}"
    print(header)
    print("-" * len(header))

    for n in sorted(results):
        row = f"{n:4d}"
        for m in methods:
            acc = results[n].get(m)
            if acc is not None:
                row += f"  {acc:>17.2%}"
            else:
                row += f"  {'N/A':>17s}"
        print(row)

    # 打印 Δ 行 (RD vs 其他)
    print()
    rd_key = "relation_decision"
    if rd_key in methods:
        other_methods = [m for m in methods if m != rd_key]
        print(f"{'N':>4s}", end="")
        for m in other_methods:
            print(f"  {'RD−'+m:>18s}", end="")
        print()
        print("-" * (4 + 20 * len(other_methods)))
        for n in sorted(results):
            rd = results[n].get(rd_key, 0)
            print(f"{n:4d}", end="")
            for m in other_methods:
                delta = rd - results[n].get(m, 0)
                sign = "+" if delta >= 0 else ""
                print(f"  {sign}{delta:>16.2%}", end="")
            print()

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["N"] + methods)
            for n in sorted(results):
                writer.writerow([n] + [f"{results[n].get(m, 0):.4f}" for m in methods])
        print(f"\nCSV 已导出: {args.csv}")

    print()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行汇总脚本**

```bash
cd /data/zjj/project_26/fact_mem && uv run --no-sync python script/aggregate_unified_results.py --csv experiment/unified_results.csv
```

预期：N-accuracy 对比表，add_all 随 N 单调下降，RD 保持稳定，Δ 随 N 扩大。

- [ ] **Step 3: Commit**

```bash
git add script/aggregate_unified_results.py experiment/unified_results.csv
git commit -m "feat: aggregate unified confusion experiment results"
```

---

### Task 7: 分析实验 — N=16 退化曲线扩展点

**说明**：N=16 仅覆盖 398 道 Type I 题。需要额外的候选目录和 config。这是分析实验，可在主实验完成后进行。

**待做**：
- 构建 `lme_s_gemma4-26B_unified_filler_N16` 候选目录（脚本已支持 `--distractors` 参数，398 题需过滤掉 knowledge-update）
- 生成 `config/lme_unified_filler_N16.yaml`
- 运行 ingest → generate → evaluate
- 汇总 N=0,2,4,6,8,16 完整曲线

---

### Task 8: 分析实验 — Knowledge-update 子集分析

**说明**：从汇总脚本扩展，按 question_type 拆分 accuracy。

**思路**：修改 `aggregate_unified_results.py`，增加 `--by-type` 选项，读取 pred JSONL 中的 `question_type` 字段，按类型分组计算 accuracy。knowledge-update 72 题单独出一行。

---

## 执行顺序

```
Task 1 (filler) → Task 2 (candidates) → Task 3 (configs)
                                              ↓
Task 4 (ingest) ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
       ↓
Task 5 (generate + evaluate)
       ↓
Task 6 (aggregate)
       ↓
Task 7 (N=16 extension, optional)
Task 8 (knowledge-update analysis, optional)
```

Task 1-3 需顺序执行（产物依赖）。Task 4-5 每个 N 档独立，但建议先跑通 N=0 再批量跑。
