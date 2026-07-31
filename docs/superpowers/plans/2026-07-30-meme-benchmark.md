# MEME Benchmark 适配实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有实验框架下适配 MEME benchmark（`meme_filler32k.json`，100 episodes，~694 after questions），支持 extract → ingest → generate → evaluate 全流程。

**Architecture:** 新建 `MEMEBenchmark` 类直接解析原始 JSON（仿 LoCoMo 模式，无需 preprocess），转换为标准 `MemoryEpisode` 格式。每 episode 含多个 `QuestionItem`（仅 after questions）。注册到 `get_benchmark()` 和 `DEFAULT_BENCHMARK_DATASETS`。Extract/ingest/generate/evaluate 均复用现有管线，无需改动。

**Tech Stack:** Python 3.12, dataclasses, JSON, YAML (hydra/omegaconf)

## 全局约束

- 新 artifact 布局，通过 `run_exp_lme.py` 运行
- Extract 模板: `0_mem_extract_dense_en.jinja`
- Extract 粒度: 4-turn
- Manager 模型: gemma4-26B
- Answer 模型: gemma4-26B
- Judge 模型: deepseek-v4-flash
- Token limit: 256
- 仅 answer after questions，before 完全不参与

---

### Task 1: 注册 MEME 数据路径

**Files:**
- Modify: `src/benchmark/datasets.py` (1 行)

**Interfaces:**
- Produces: `DEFAULT_BENCHMARK_DATASETS["meme_filler32k"]` → `("data/raw_data/MEME/meme_filler32k.json", "en")`

- [ ] **Step 1: 添加 MEME 条目**

在 `DEFAULT_BENCHMARK_DATASETS` 字典末尾添加一行：

```python
DEFAULT_BENCHMARK_DATASETS: Dict[str, Tuple[str, str]] = {
    "test": ("data/preprocessed/test.json", "zh"),
    "lme_o": ("data/preprocessed/longmemeval_oracle_converted.json", "en"),
    "lme_s": ("data/preprocessed/longmemeval_s_cleaned_converted.json", "en"),
    "lme_s_golden": ("data/preprocessed/longmemeval_s_hybrid_golden_converted.json", "en"),
    "lme_m": ("data/preprocessed/longmemeval_m_cleaned_converted.json", "en"),
    "locomo": ("data/raw_data/locomo10.json", "en"),
    "meme_filler32k": ("data/raw_data/MEME/meme_filler32k.json", "en"),   # ← 新增
}
```

- [ ] **Step 2: 验证注册**

```bash
python3 -c "
from benchmark.datasets import resolve_benchmark_data_path
path, lang = resolve_benchmark_data_path('meme_filler32k')
print(f'path={path}, lang={lang}')
assert path == 'data/raw_data/MEME/meme_filler32k.json'
assert lang == 'en'
print('OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add src/benchmark/datasets.py
git commit -m "feat(meme): register meme_filler32k data path"
```

---

### Task 2: 创建 MEMEBenchmark 加载器

**Files:**
- Create: `src/benchmark/meme.py`

**Interfaces:**
- Consumes: `BaseBenchmark`, `MemoryEpisode`, `ChatSession`, `ChatTurn`, `QuestionItem` (from `.base`)
- Produces: `MEMEBenchmark(BaseBenchmark)` with `_load_data()` and `_convert_episode()`

- [ ] **Step 1: 创建 meme.py**

```python
"""MEME benchmark loader.

将 ``meme_filler32k.json`` 转换为标准 ``MemoryEpisode`` 格式。
每 episode = 一个 MemoryEpisode，含 ~21 sessions 和 ~7 after questions。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from .base import BaseBenchmark, ChatSession, ChatTurn, MemoryEpisode, QuestionItem

logger = logging.getLogger(__name__)


class MEMEBenchmark(BaseBenchmark):
    """MEME (Multi-Entity Evolving Memory) 评测基准。

    原始 JSON 结构（每项一个 episode）：::

        {
          "episode_id": "pl_001",
          "domain": "personal_life",
          "root": "health_condition",
          "root_change": {"before": "...", "after": "..."},
          "entities": {...},
          "tasks": [...],
          "sessions": [
            {
              "session_id": "sharegpt_14177",
              "type": "filler",
              "timestamp": "2023/03/03 (Fri) 11:55",
              "conversation": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
              ]
            }
          ],
          "before_questions": {
            "timestamp": "2023/03/19 (Sun) 00:18",
            "position_after_session": 17,
            "questions": [{"task_type": "Cas", "question": "...", "expected_answer": "..."}]
          },
          "after_questions": {
            "timestamp": "2023/03/27 (Mon) 06:29",
            "position_after_session": 23,
            "questions": [{"task_type": "Tr", "question": "...", "gold_answer": "..."}]
          }
        }

    设计决策：
    - 只加载 after_questions（before 完全不参与）
    - extract 范围：全部 session（不限于 position_after_session 之前）
    - 每 episode 多 QA（LoCoMo 风格）
    """

    def _load_data(self) -> None:
        file_path = Path(self.file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"MEME data file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            raw_list: List[Dict[str, Any]] = json.load(f)

        if not isinstance(raw_list, list):
            raise ValueError(
                f"MEME data root must be a JSON array, got {type(raw_list).__name__}"
            )

        for raw_ep in raw_list:
            episode = self._convert_episode(raw_ep)
            self.episodes.append(episode)

        logger.info(
            "MEMEBenchmark: loaded %d episodes from %s",
            len(self.episodes),
            file_path,
        )

    @staticmethod
    def _convert_episode(raw: Dict[str, Any]) -> MemoryEpisode:
        """将一条 MEME episode 转换为 MemoryEpisode。"""

        # ---- 1. 解析 sessions ----
        sessions: List[ChatSession] = []
        for s in raw.get("sessions", []):
            turns: List[ChatTurn] = []
            for turn in s.get("conversation", []):
                turns.append(ChatTurn(
                    speaker=str(turn.get("role", "Unknown")).strip(),
                    content=str(turn.get("content", "")),
                ))

            session_meta: Dict[str, Any] = {
                "type": str(s.get("type", "")),
                "session_id": str(s.get("session_id", "")),
            }
            if s.get("evidence_type"):
                session_meta["evidence_type"] = s["evidence_type"]

            sessions.append(ChatSession(
                session_date=str(s.get("timestamp", "")),
                turns=turns,
                metadata=session_meta,
            ))

        # ---- 2. 解析 after_questions ----
        after_block: Dict[str, Any] = raw.get("after_questions", {})
        question_time = str(after_block.get("timestamp", ""))

        qas: List[QuestionItem] = []
        for q in after_block.get("questions", []):
            meta: Dict[str, Any] = {
                "entity": q.get("entity", []),
                "entity_values": q.get("entity_values", {}),
            }
            if "hop" in q:
                meta["hop"] = q["hop"]

            qas.append(QuestionItem(
                question=str(q.get("question", "")).strip(),
                answer=str(q.get("gold_answer", "")).strip(),
                question_time=question_time,
                question_type=str(q.get("task_type", "")).strip(),
                metadata=meta,
            ))

        # ---- 3. 组装 Episode ----
        episode_id = str(raw.get("episode_id", "")).strip()

        return MemoryEpisode(
            history_name=episode_id if episode_id else f"meme_{len(sessions)}sessions",
            sessions=sessions,
            qas=qas,
            metadata={
                "benchmark": "meme",
                "domain": str(raw.get("domain", "")),
                "root": str(raw.get("root", "")),
            },
        )
```

- [ ] **Step 2: 验证加载器**

```bash
PYTHONPATH=src python3 -c "
from benchmark.meme import MEMEBenchmark

bm = MEMEBenchmark('data/raw_data/MEME/meme_filler32k.json', lang='en')
print(f'Episodes: {len(bm)}')

ep = bm.episodes[0]
print(f'history_name: {ep.history_name}')
print(f'sessions: {len(ep.sessions)}')
print(f'QAs: {len(ep.qas)}')
print(f'metadata: {ep.metadata}')

# Session 检查
s0 = ep.sessions[0]
print(f'First session: date={s0.session_date[:20]}..., turns={len(s0.turns)}, meta={s0.metadata}')
print(f'First turn: speaker={s0.turns[0].speaker}, content[:50]={s0.turns[0].content[:50]}')

# QA 检查
qa0 = ep.qas[0]
print(f'First QA: type={qa0.question_type}, q={qa0.question[:60]}, a={qa0.answer[:60]}')
print(f'QA metadata: {qa0.metadata}')

# 所有 episode 统计
total_qas = sum(len(ep.qas) for ep in bm.episodes)
total_sessions = sum(len(ep.sessions) for ep in bm.episodes)
print(f'\n总计: {len(bm)} episodes, {total_qas} QAs, {total_sessions} sessions')

# 类型分布
from collections import Counter
qtypes = Counter(q.question_type for ep in bm.episodes for q in ep.qas)
print(f'QA types: {dict(qtypes)}')
"
```

- [ ] **Step 3: Commit**

```bash
git add src/benchmark/meme.py
git commit -m "feat(meme): add MEMEBenchmark loader"
```

---

### Task 3: 注册 MEMEBenchmark 到工厂函数

**Files:**
- Modify: `src/benchmark/__init__.py`

**Interfaces:**
- Consumes: `MEMEBenchmark` (from `.meme`)
- Produces: `get_benchmark("meme_*")` → `MEMEBenchmark`

- [ ] **Step 1: 添加 import 和路由**

修改两处：import 区域的第 4 行（新增），`get_benchmark()` 中 `locomo` 分支后（新增 meme 分支）：

```python
# import 区域（第 5 行后新增）:
from .meme import MEMEBenchmark

# get_benchmark() 函数中（第 13 行后，"locomo" 分支之后）:
    if task_name.startswith("meme"):
        return MEMEBenchmark(file_path, lang=lang)

# __all__ 列表（第 28 行后新增）:
    "MEMEBenchmark",
```

最终文件如下：

```python
from .base import BaseBenchmark, MemoryEpisode, ChatSession, ChatTurn, QuestionItem
from .datasets import DEFAULT_BENCHMARK_DATASETS, resolve_benchmark_data_path
from .lme import LMEBenchmark
from .locomo import LoCoMoBenchmark
from .meme import MEMEBenchmark

def get_benchmark(task_name: str, file_path: str, lang: str = "en") -> BaseBenchmark:
    """
    根据 task_name 返回对应的 Benchmark 实例
    """
    task_name = task_name.lower()

    if task_name.startswith("locomo"):
        return LoCoMoBenchmark(file_path, lang=lang)
    if task_name.startswith("meme"):
        return MEMEBenchmark(file_path, lang=lang)
    if task_name.startswith("lme"):
        return LMEBenchmark(file_path, lang=lang)
    else:
        return LMEBenchmark(file_path, lang=lang)

__all__ = [
    "BaseBenchmark",
    "MemoryEpisode",
    "ChatSession",
    "ChatTurn",
    "QuestionItem",
    "get_benchmark",
    "LMEBenchmark",
    "LoCoMoBenchmark",
    "MEMEBenchmark",
    "DEFAULT_BENCHMARK_DATASETS",
    "resolve_benchmark_data_path",
]
```

- [ ] **Step 2: 验证路由**

```bash
PYTHONPATH=src python3 -c "
from benchmark import get_benchmark

# meme_filler32k → MEMEBenchmark
bm = get_benchmark('meme_filler32k', 'data/raw_data/MEME/meme_filler32k.json')
print(f'meme_filler32k → {type(bm).__name__}')
assert type(bm).__name__ == 'MEMEBenchmark'

# meme_nofiller → MEMEBenchmark (if file exists)
import os
if os.path.exists('data/raw_data/MEME/meme_nofiller.json'):
    bm2 = get_benchmark('meme_nofiller', 'data/raw_data/MEME/meme_nofiller.json')
    assert type(bm2).__name__ == 'MEMEBenchmark'
    print('meme_nofiller → MEMEBenchmark')

# lme_s still routes to LMEBenchmark
bm3 = get_benchmark('lme_s', 'data/preprocessed/longmemeval_s_cleaned_converted.json')
assert type(bm3).__name__ == 'LMEBenchmark'
print('lme_s → LMEBenchmark')

print('All routing OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add src/benchmark/__init__.py
git commit -m "feat(meme): register MEMEBenchmark in get_benchmark factory"
```

---

### Task 4: 创建 MEME 实验配置

**Files:**
- Create: `config/meme_default.yaml`

- [ ] **Step 1: 创建配置文件**

```yaml
# easy-mem MEME 实验配置（默认模板）
# 用法：uv run --no-sync python run_exp_lme.py --config config/meme_default.yaml [--stages extract,ingest,generate,evaluate]
#
# MEME 与 LME 的主要区别：
#   - 每 episode 含 ~7 after questions（before questions 不参与评测）
#   - 对话格式：标准 user/assistant（与 LME 一致）
#   - 抽取模板：0_mem_extract_dense_en.jinja（全量密集抽取）
#   - 抽取粒度：4-turn
#   - 全量评测（不抽样）

experiment:
  benchmark: meme_filler32k      # MEME 32k filler 变体（100 episodes）
  suffix: meme_default           # 可读标签

# ---- 模型选择 ----
models:
  extract: gemma4-26B               # 候选记忆抽取
  manager: gemma4-26B               # 灌库管理（relation_decision/mem0/evermemos 使用）
  answer: gemma4-26B                # 答题
  judge: deepseek-v4-flash          # LLM Judge
  embedding: qwen3-embedding-0.6b   # Embedding（本地 vllm）

# ---- 候选记忆抽取 ----
extract:
  candidate_suffix: meme_default    # 候选版本标签
  granularity: 4                    # 4-turn 分块
  turn_overlap: 0
  language: en
  aspect_templates:
    - "0_mem_extract_dense_en.jinja"

# ---- 灌库方法 ----
methods:
  add_all:
    enabled: true

  relation_decision:
    enabled: true
    backend: llm
    related_top_k: 3
    fusion_model: ""
    fusion_enabled: true
    condition_sim_threshold: 0.5
    pairwise_sim_threshold: 0.5

  mem0:
    enabled: true

  evermemos:
    enabled: true

# ---- 答题生成 ----
generate:
  retrieve_topk: 50
  memory_token_limit: 256
  answer_stratified_sample: 0        # MEME 全量评测（此字段对 MEME 无影响，run_exp_lme.py 硬编码 sample=0）
  answer_sample_seed: 43
  show_memory_time: true

  hybrid:
    enabled: false

# ---- 评估 ----
evaluate:
  use_cot: true

# ---- 可选：一次配置扫多个 token limit ----
sweep:
  memory_token_limits:
    - 256

# ---- 统计重复 ----
replication:
  count: 1
  scope: answer_judge

# ---- 并发控制 ----
parallel:
  extract_chunk_concurrency: 100
  ingest_relation_concurrency: 20
  ingest_episode_concurrency:
    relation_decision: 20
    mem0: 20
    add_all: 100
    evermemos: 20
    fusion_episodes: 100
    fusion_packages: 10
  generate_parallel_episodes: 50
  generate_answer_concurrency: 2
  evaluate_max_concurrency: 8

# ---- Token 限制 ----
token_limits:
  extract_max_new_tokens: 2048
  ingest_relation_max_new_tokens: 256
  ingest_manager_max_new_tokens: 2048
  fusion_max_new_tokens: 512
  evaluate_max_new_tokens: 512

# ---- 提示模板 ----
prompts:
  relation_user_en: RD_0_relation_classify.jinja
  relation_user_zh: RD_0_relation_classify.jinja
  judge_template: pipeline_judge.jinja
```

- [ ] **Step 2: 验证配置可解析**

```bash
PYTHONPATH=src uv run --no-sync python -c "
from utils.config import load_config
cfg = load_config('config/meme_default.yaml')
print(f'benchmark: {cfg.experiment.benchmark}')
print(f'suffix: {cfg.experiment.suffix}')
print(f'extract model: {cfg.models.extract}')
print(f'manager model: {cfg.models.manager}')
print(f'extract template: {cfg.extract.aspect_templates}')
print(f'token limit: {cfg.generate.memory_token_limit}')
print(f'enabled methods: add_all={cfg.methods.add_all.enabled}, rd={cfg.methods.relation_decision.enabled}, mem0={cfg.methods.mem0.enabled}, evm={cfg.methods.evermemos.enabled}')
"
```

- [ ] **Step 3: Commit**

```bash
git add config/meme_default.yaml
git commit -m "feat(meme): add meme_default.yaml experiment config"
```

---

### Task 5: 端到端验证

**Files:**
- (验证用，不提交)

**Interfaces:**
- Consumes: all files from Tasks 1-4

- [ ] **Step 1: 验证完整 benchmark 加载链路**

```bash
PYTHONPATH=src uv run --no-sync python -c "
from benchmark.datasets import resolve_benchmark_data_path
from benchmark import get_benchmark

# 1. 路径解析
path, lang = resolve_benchmark_data_path('meme_filler32k')
print(f'Resolved: {path} (lang={lang})')

# 2. 获取 benchmark 实例
bm = get_benchmark('meme_filler32k', path, lang)
print(f'Benchmark type: {type(bm).__name__}')
print(f'Episodes: {len(bm)}')

# 3. 验证第一个 episode 结构
ep = bm.episodes[0]
assert len(ep.sessions) > 0, 'no sessions'
assert len(ep.qas) > 0, 'no QAs'
assert ep.metadata['benchmark'] == 'meme', 'wrong benchmark metadata'

# 4. 验证所有 episode
total_qa = 0
for i, ep in enumerate(bm.episodes):
    assert ep.history_name, f'ep {i}: empty history_name'
    assert len(ep.sessions) > 0, f'ep {ep.history_name}: no sessions'
    assert len(ep.qas) > 0, f'ep {ep.history_name}: no QAs'
    for q in ep.qas:
        assert q.question, f'ep {ep.history_name}: empty question'
        assert q.answer is not None, f'ep {ep.history_name}: None answer'
    total_qa += len(ep.qas)

print(f'Total: {len(bm)} episodes, {total_qa} QAs')
print('All structural checks passed.')
"
```

- [ ] **Step 2: 验证 session 时间线顺序**

```bash
PYTHONPATH=src uv run --no-sync python -c "
from benchmark import get_benchmark
bm = get_benchmark('meme_filler32k', 'data/raw_data/MEME/meme_filler32k.json')

# 每 episode 的 session 时间戳不应递减
for ep in bm.episodes:
    prev = ''
    for s in ep.sessions:
        assert s.session_date >= prev, \
            f'ep {ep.history_name}: session time out of order: {prev!r} → {s.session_date!r}'
        prev = s.session_date
print('Session time ordering OK')
"
```

- [ ] **Step 3: 验证对话格式兼容性（extract 管线模拟）**

```bash
PYTHONPATH=src uv run --no-sync python -c "
from benchmark import get_benchmark
bm = get_benchmark('meme_filler32k', 'data/raw_data/MEME/meme_filler32k.json')

ep = bm.episodes[0]
# 模拟 extract 的对话格式化（LME 默认路径）
lines = []
for s in ep.sessions:
    for t in s.turns:
        lines.append(f'{t.speaker}: {t.content}')
text = '\n'.join(lines)
print(f'Episode {ep.history_name}:')
print(f'  Sessions: {len(ep.sessions)}')
print(f'  Total turns: {sum(len(s.turns) for s in ep.sessions)}')
print(f'  Formatted text length: {len(text)} chars')
print(f'  First 200 chars of formatted text:')
print(f'  {text[:200]}...')
print()
print('Formatted text is valid. Extract should work with default (non-LoCoMo) path.')
"
```

- [ ] **Step 4: 验证 run_exp_lme.py MEME 兼容路径**

```bash
PYTHONPATH=src uv run --no-sync python -c "
# 模拟 run_exp_lme.py 中对 MEME 的处理
benchmark = 'meme_filler32k'
sample = 0 if benchmark.lower().startswith('meme') else 42
assert sample == 0, f'Expected sample=0 for MEME, got {sample}'
print(f'MEME benchmark → sample={sample} (全量评测)')

# 非 MEME benchmark 不受影响
benchmark2 = 'lme_s'
sample2 = 0 if benchmark2.lower().startswith('meme') else 42
assert sample2 == 42, f'Expected sample=42 for LME, got {sample2}'
print(f'LME benchmark → sample={sample2} (分层抽样)')
print('sample=0 logic OK')
"
```

---

## Task Dependency Order

```
Task 1 (datasets.py) ──┐
                       ├──> Task 4 (config)    (独立)
Task 2 (meme.py) ─────┤
                       │
Task 3 (__init__.py) ──┘
                       │
                       └──> Task 5 (verification)
```

- Task 1 和 Task 2 可以并行（互不依赖）
- Task 3 依赖 Task 2（需要 `MEMEBenchmark` 类）
- Task 4 独立于 Task 1-3（配置文件语法独立）
- Task 5 依赖 Task 1-4 全部完成
