# MEME Benchmark 适配设计

**日期**: 2026-07-30
**状态**: 已确认

## 概述

在现有实验框架（新 artifact 布局）下适配 MEME（Multi-Entity and Evolving Memory Evaluation）benchmark。
使用 `meme_filler32k.json`（100 episodes, ~32k filler tokens），gemma4-26B 做记忆抽取和答题，deepseek-v4-flash 做 judge。

## 关键决策

| 决策 | 结论 |
|------|------|
| Question 范围 | 只答 after（~694 题），before 完全不参与 |
| Episode 建模 | 1 MEME raw item → 1 MemoryEpisode，含多道 QuestionItem（LoCoMo 风格） |
| Extract session 范围 | 全部 session（evidence + filler），不限 position_after_session |
| Extract 粒度 | 4-turn 分块 |
| Extract 模板 | `0_mem_extract_dense_en.jinja` |
| Ingest 方法 | add_all, relation_decision, mem0, evermemos 四者 |
| Manager 模型 | gemma4-26B |
| Answer 模型 | gemma4-26B |
| Token limit | 256 |
| Judge | deepseek-v4-flash + 通用 judge（`pipeline_judge.jinja`） |
| 框架 | 新 artifact 布局，通过 `run_exp_lme.py` 运行 |

## 数据

- 文件: `data/raw_data/MEME/meme_filler32k.json`
- 结构: 100 条 JSON 数组，每条约 ~21 sessions（5 evidence + ~16 filler），~7 after questions
- 对话格式: 标准 `user/assistant` turn（与 LME 一致）
- Task types: Tr（跟踪）、Cas（级联）、Abs（弃权）、Del（删除）、Agg（聚合）、ER（实体召回）

## 需要新建的文件

### `src/benchmark/meme.py` — MEMEBenchmark

- 直接解析原始 JSON（无需 preprocess，仿 LoCoMo 模式）
- `_load_data()` → 每 episode 创建 `MemoryEpisode`:
  - `history_name`: episode_id
  - `sessions`: 全部 session（不区分 evidence/filler），转换为 `ChatSession` 列表。对话格式 `user/assistant`（与 LME 相同），无需特殊格式化
  - `qas`: 仅取 `after_questions.questions[]`，映射 `gold_answer` → `answer`，`timestamp` → `question_time`，`task_type` → `question_type`
  - `metadata`: `{"benchmark": "meme", "domain": ..., "root": ...}`
- 注册 `DEFAULT_BENCHMARK_DATASETS` 条目:
  - `meme_filler32k` → `data/raw_data/MEME/meme_filler32k.json`

### `config/meme_default.yaml` — 实验配置

```yaml
experiment:
  name: meme_baseline
  suffix: meme_default
  benchmark: meme_filler32k

extract:
  candidate_extract_model: gemma4-26B
  candidate_suffix: meme_default
  granularity: 4
  template: 0_mem_extract_dense_en.jinja

models:
  extract: gemma4-26B
  manager: gemma4-26B
  answer: gemma4-26B
  judge: deepseek-v4-flash
  embedding: qwen3-embedding-0.6b

methods:
  add_all: {enabled: true}
  relation_decision: {enabled: true}
  mem0: {enabled: true}
  evermemos: {enabled: true}

generate:
  memory_token_limit: 256
  retrieve_topk: 50
  answer_stratified_sample: 0   # MEME 全量评测
  answer_batch_size: 10
  answer_concurrency: 5
```

## 需要改动的文件

### `src/benchmark/__init__.py`

`get_benchmark()` 添加:

```python
if task_name.startswith("meme"):
    return MEMEBenchmark(file_path, lang=lang)
```

### `src/benchmark/datasets.py`

```python
"meme_filler32k": ("data/raw_data/MEME/meme_filler32k.json", "en"),
```

## 不需要改动

- **extract_candidates.py**: MEME 对话格式 `user/assistant` 与 LME 一致，`0_mem_extract_dense_en.jinja` 可直接使用，无需特殊处理。4-turn 分块逻辑复用
- **run_exp_lme.py**: 已有 `sample=0` 兼容 MEME（line 340-342）
- **Agent/Judge**: 通用 answer (`pipeline_answer.jinja`) 和 judge (`pipeline_judge.jinja`) 可直接使用
- **mem_extract_schemas.py**: 复用 `MemExtractResponse`（字符串列表格式），无需新增 schema

## 实验执行

```bash
# 全阶段
uv run --no-sync python run_exp_lme.py --config config/meme_default.yaml --stages extract,ingest,generate,evaluate

# 或分步
uv run --no-sync python run_exp_lme.py --config config/meme_default.yaml --stages extract
uv run --no-sync python run_exp_lme.py --config config/meme_default.yaml --stages ingest
uv run --no-sync python run_exp_lme.py --config config/meme_default.yaml --stages generate,evaluate
```

## 注意事项

- MEME 没有 filler 等级机制，候选记忆直接按 episode 生成
- 多 QA episode 的 answer/judge 已有 LoCoMo 先例验证，无需特殊适配
- Ingest 目录与其他 benchmark 隔离（路径由 `history_name` 即 episode_id 决定）
- MEME 与 LME 共享 ingest 模板（RD 分类用 `RD_0_relation_classify.jinja`），无需专属模板
