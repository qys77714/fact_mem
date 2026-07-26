# 实验 Artifact 架构使用指南

本文介绍 **fact_mem / easy-mem** 在 LME 实验中的新产物布局。**新实验一律使用此架构**；`run_exp_lme.py` 默认启用，无需额外开关。

旧目录 `MemDB/`、`experiment/` 仅用于历史结果。若要按旧路径续跑，见文末「Legacy 布局」。

---

## 1. 快速开始

```bash
cd /path/to/fact_mem
source .venv/bin/activate

# 全流程（extract → ingest → generate → evaluate）
uv run --no-sync python run_exp_lme.py --config config/lme.yaml

# 只跑部分阶段
uv run --no-sync python run_exp_lme.py \
  --config config/lme.yaml \
  --stages ingest,generate,evaluate
```

运行结束后，终端会打印每个 **variant** 的 `run_root` 与各 method 的 ingest / answer / judge 路径。

---

## 2. 设计理念

| 概念 | 含义 |
|------|------|
| **内容寻址** | 目录名由「阶段相关配置 + 模板内容 + 数据集等」的指纹决定，而不是手工 suffix |
| **阶段复用** | candidates / ingest 放在全局缓存，多个 run 可共享同一份库 |
| **run 隔离** | answer / judge / trace 属于某次具体运行，互不覆盖 |
| **不可变快照** | 每次 run 写入 `manifest.json`，记录完整解析配置与 git commit |
| **variant** | 一次 YAML 可展开为多个 variant（token limit × repeat） |

一句话：**改 ingest 相关参数 → 新 ingest 目录；只改 token limit → 复用 ingest，只新跑 answer/judge。**

---

## 3. 目录结构

默认根目录为 `artifacts/`（可用 `--artifacts-root` 修改）：

```text
artifacts/
├── stages/                          # 全局阶段缓存（跨 run 复用）
│   ├── locks/                       # 跨进程文件锁（勿手动删）
│   ├── candidates/
│   │   └── <candidate_id>/
│   │       ├── stage_manifest.json  # 该阶段的 provenance
│   │       └── ...                  # 抽取产物（由 extract 脚本写入）
│   └── ingest/
│       └── <method>/
│           └── <ingest_id>/
│               ├── stage_manifest.json
│               ├── trace/             # 灌库 trace（按 method 隔离）
│               └── ...                # 向量库（episode 子目录等）
└── runs/
    └── <run_id>/                    # 单次逻辑实验的快照根
        ├── manifest.json            # 完整解析配置 + hash + git commit
        ├── run.yaml                 # 与 manifest 等价的 YAML 副本
        ├── stages.json              # 本 run 引用的各 stage_id
        ├── attempts/
        │   └── <attempt_id>/
        │       └── attempt.json     # 本次 CLI 参数、请求阶段
        ├── answer/
        │   └── <method>/
        │       └── <answer_id>/
        │           ├── pred.jsonl
        │           └── agent_trace/
        └── judge/
            └── <method>/
                └── <judge_id>/
                    ├── judged.jsonl   # 含 is_correct，不改写 pred
                    └── metrics.json   # 单次 Judge 汇总指标
```

### `run_id` 长什么样

形如：

```text
lme_s_exp001_relation_decision+mem0_tl256--a1b2c3d4
```

- 前半段 **slug**：benchmark、suffix、enabled methods、token limit 等可读片段
- 后半段 **8 位 hash**：完整解析配置的指纹

同一 YAML、同一 git 状态 → 同一 `run_id`；再次运行会复用已有 `manifest.json`（配置 hash 不一致则拒绝覆盖）。

---

## 4. 四个阶段分别写到哪里

| 阶段 | 写入位置 | 是否跨 run 复用 |
|------|----------|----------------|
| extract | `artifacts/stages/candidates/<candidate_id>/` | 是 |
| ingest | `artifacts/stages/ingest/<method>/<ingest_id>/` | 是 |
| generate | `artifacts/runs/<run_id>/answer/<method>/<answer_id>/` | 否 |
| evaluate | `artifacts/runs/<run_id>/judge/<method>/<judge_id>/` | 否 |

**Judge 不会改写 `pred.jsonl`**。评分结果在 `judged.jsonl`；汇总指标在 `metrics.json`。

---

## 5. 什么参数变化会影响哪个阶段

系统按「最小依赖」计算阶段指纹（详见 `src/utils/experiment_artifacts.py`）。

### Candidate（抽取）

受影响示例：

- `experiment.benchmark`
- `models.extract`
- `extract.candidate_suffix`、`granularity`、`language`、`aspect_templates`
- 抽取模板 **文件内容**（不仅是文件名）

**不受影响**：`generate.memory_token_limit`、judge 模型、ingest 阈值等。

> 改抽取策略时，务必同步修改 `extract.candidate_suffix`，避免 extract 脚本跳过已完成的 episode。

### Ingest（灌库，按 method 独立）

在 candidate 之上，还取决于：

- `models.manager`、`models.embedding`
- `methods.<method>.*`（如 RD 的阈值、fusion、active_relations）
- relation 相关 prompt 模板 **内容**

**不受影响**：`generate.memory_token_limit`、`retrieve_topk`、judge 配置。

不同 method 的 `ingest_id` 相互独立（例如 `relation_decision` 与 `mem0` 不会共用目录）。

### Answer（生成）

取决于：

- 上游 `ingest_id`
- `models.answer`、`models.embedding`
- `generate.*`（含 `memory_token_limit`、`retrieve_topk`、hybrid、抽样 seed）

### Judge（评测）

取决于：

- 上游 `answer_id`
- `models.judge`、`evaluate.*`、judge 模板 **内容**

---

## 6. 常见用法

### 6.1 只换 Memory Token Limit（复用 ingest）

场景：已有 `tl256` 的完整灌库，想再跑 `tl512` 答题。

**做法 A — 两个 YAML（推荐，语义清晰）**

`config/exp_tl256.yaml` 与 `config/exp_tl512.yaml` 仅 `generate.memory_token_limit` 不同，其余 ingest 相关字段保持一致：

```bash
# 第一次：全流程
uv run --no-sync python run_exp_lme.py --config config/exp_tl256.yaml

# 第二次：只跑下游
uv run --no-sync python run_exp_lme.py \
  --config config/exp_tl512.yaml \
  --stages generate,evaluate
```

**做法 B — 一个 YAML + sweep**

见第 7 节；一次命令扫 `[256, 512]`。

> 不要把 token limit 写进 `experiment.suffix` 来「区分实验」——新架构下 token limit 已进入 answer 阶段指纹，suffix 只作可读标签即可。

### 6.2 同时比较多个方法

在一个 config 里启用多个 method：

```yaml
methods:
  relation_decision:
    enabled: true
  mem0:
    enabled: true
  evermemos:
    enabled: true
  add_all:
    enabled: true
```

- **共享** 同一批 candidates
- **各自独立** ingest / answer / judge 目录
- 主入口按 method 依次 ingest、generate、evaluate，互不覆盖

```bash
uv run --no-sync python run_exp_lme.py --config config/multi_method.yaml
```

### 6.3 只重跑 Judge

若 `pred.jsonl` 已存在，只需 evaluate：

```bash
uv run --no-sync python run_exp_lme.py \
  --config config/my_exp.yaml \
  --stages evaluate
```

新 Judge 会写到新的 `judge/<judge_id>/`（若 judge 模型或模板变了），不会污染旧 `judged.jsonl`。

### 6.4 断点续跑

- **generate**：`pred.jsonl` 按 `(history_name, question_id)` 续写，已答题目跳过
- **ingest**：episode 级 marker + 配置指纹；配置未变则跳过已完成 episode
- **extract**：仍依赖 `extract.candidate_suffix` + progress state；改抽取策略必须换 suffix

并行跑 **同一 candidate_id / ingest_id** 时，框架会使用 `artifacts/stages/locks/` 下的文件锁，避免两个进程同时写同一阶段目录。

---

## 7. Sweep 与 Replication

在 YAML 中可选配置：

```yaml
generate:
  memory_token_limit: 256      # sweep 为空时使用此默认值
  answer_sample_seed: 43

sweep:
  memory_token_limits: [256, 512]

replication:
  count: 3
  scope: answer_judge          # answer_judge | full_pipeline
  seeds: [43, 44, 45]        # 省略则从 answer_sample_seed 递增：43, 44, 45
```

### Variant 展开规则

- 顺序：**token limit 外层 × repeat 内层**
- 每个 variant 有标签 `variant_id`，例如 `tl256-r00-s43`、`tl512-r02-s45`
- 每个 variant 对应独立的 `run_root`、answer、judge 目录

### `scope: answer_judge`（默认，推荐）

- 同一 **进程内** 的多个 variant：**共享** extract / ingest（每个 `candidate_dir`、`(method, ingest_dir)` 只跑一次）
- 每个 variant **独立** generate / evaluate
- 适合：扫 token limit、多 repeat 统计 mean/std，且不想重复灌库

### `scope: full_pipeline`

- 每个 variant 带独立 `stage_nonce`，candidate / ingest 目录互不相同
- extract / ingest **每个 repeat 都会重跑**
- 适合：衡量端到端随机性（成本高）

### 示例：2 个 token × 3 次 repeat

```yaml
sweep:
  memory_token_limits: [256, 512]
replication:
  count: 3
  scope: answer_judge
```

一次命令产生 **6 个 variant**（2×3），但 extract 1 次、每个 method 的 ingest 1 次，generate/evaluate 各 6 次。

```bash
uv run --no-sync python run_exp_lme.py --config config/sweep_rep.yaml
```

---

## 8. 聚合 Repeat 的 mean / std

每个 variant 的 Judge 输出独立 `metrics.json`。聚合时**显式列出**要合并的文件（工具不会扫描目录）：

```bash
uv run --no-sync python script/aggregate_experiment_metrics.py \
  --input \
    artifacts/runs/<run-a>/judge/relation_decision/<judge-id>/metrics.json \
    artifacts/runs/<run-b>/judge/relation_decision/<judge-id>/metrics.json \
    artifacts/runs/<run-c>/judge/relation_decision/<judge-id>/metrics.json \
  --output artifacts/summaries/rd-tl256-r3.json \
  --pretty
```

输出包含：

- 每次 repeat 的 `overall_accuracy`
- `mean`、`sample_std`（仅 1 个样本时 std 为 `null`）
- `judged_count` / `api_failure_count` 的 sum 与 mean

**约束**：所有输入必须同一 `benchmark`、同一 `judge_model`，否则拒绝聚合。

> 若 LLM 解码接近确定性，多次 repeat 结果可能完全相同，std 为 0 是正常的；manifest / `attempt.json` 仍会记录各次 seed。

---

## 9. 如何追溯一次实验

| 文件 | 作用 |
|------|------|
| `artifacts/runs/<run_id>/manifest.json` | 完整解析配置、source yaml 路径、git commit、创建时间 |
| `artifacts/runs/<run_id>/stages.json` | 本 run 使用的 candidate / 各 method 的 stage_id |
| `artifacts/runs/<run_id>/attempts/*/attempt.json` | 本次 CLI 的 `--stages` 与 argv |
| `artifacts/stages/.../stage_manifest.json` | 共享阶段的 producer run、上游 stage_id |
| `answer/.../agent_trace/` | 每题召回记忆、prompt（答题 trace） |
| `ingest/.../trace/` | 灌库 LLM 调用与写库操作 trace |

从论文表格反查某次结果：先找 `metrics.json` → 所在 `run_id` → 读 `manifest.json` + `stages.json` → 必要时打开 ingest trace。

---

## 10. CLI 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--config` | `config/lme.yaml` | 实验 YAML |
| `--stages` | 四阶段全开 | 逗号分隔：`extract,ingest,generate,evaluate` |
| `--artifacts-root` | `artifacts` | 新布局根目录（其下自动分 `stages/`、`runs/`） |
| `--legacy-layout` | 关闭 | 使用旧 `MemDB/` + `experiment/` 路径 |

---

## 11. 配置编写建议

1. **`experiment.suffix`**：人类可读标签即可，不必再靠它区分 token limit 或方法。
2. **`extract.candidate_suffix`**：抽取策略变更时必须更换，否则 extract 会跳过 episode。
3. **ingest 相关参数**（RD 阈值、embedding、manager、fusion 等）变更后，会自动得到新 `ingest_id`；无需手工删库。
4. **多方法比较**：共用一个 config，不要为每个 method 复制整份 YAML（除非 method 超参差异很大）。
5. **sweep + replication**：优先用 `answer_judge`，除非明确需要全链路重复。

### 最小可运行示例

```yaml
experiment:
  benchmark: lme_s
  suffix: hybrid_n8_main

models:
  extract: gemma4-26B
  manager: Qwen3-4B
  answer: gemma4-26B
  judge: qwen3-max
  embedding: qwen3-embedding-0.6b

extract:
  candidate_suffix: hybrid_filler_N8
  granularity: 4
  language: en
  aspect_templates:
    - "0_mem_extract_aspect_unified_en.jinja"

methods:
  relation_decision:
    enabled: true
  add_all:
    enabled: true

generate:
  memory_token_limit: 256
  retrieve_topk: 50
  answer_stratified_sample: 0

evaluate:
  use_cot: true
```

---

## 12. Legacy 布局（仅旧实验）

历史产物在：

```text
MemDB/candidates/...
MemDB/ingest/...
experiment/<long_name>/pred_*.jsonl
experiment/.../eval_judge.json
```

续跑或复现旧结果：

```bash
uv run --no-sync python run_exp_lme.py \
  --config config/old.yaml \
  --legacy-layout
```

注意：

- Legacy 下 Judge 仍可能 `--write_back` 改写 `pred_*.jsonl`
- 新旧结果的 metrics **不要混在同一聚合命令里**
- 旧目录不会被自动迁移；新实验请不要再写入 `MemDB/` / `experiment/`

---

## 13. 常见问题

**Q：换了 YAML 里几个 ingest 参数，为什么没重灌？**  
A：若 fingerprint 未变（或 episode marker 仍有效），会跳过。确认改动的字段是否属于 ingest 指纹；必要时检查 `stages.json` 里的 `ingest_id` 是否变化。

**Q：两个实验能共用 ingest 吗？**  
A：可以，只要 ingest 阶段指纹相同，就会落在同一 `artifacts/stages/ingest/<method>/<ingest_id>/`。

**Q：能否把 artifacts 放在别的磁盘？**  
A：可以：`--artifacts-root /data/exp_artifacts`。

**Q：run_id 变了但想复用旧 ingest？**  
A：只要 ingest 指纹相同，物理目录相同，与 `run_id` 无关；`run_id` 只标识「这一次 answer/judge 配置快照」。

**Q：MEME 实验也用这套吗？**  
A：当前新架构接入在 `run_exp_lme.py`；MEME 仍走 `run_exp_meme.py` 原有布局。

---

## 14. 相关代码

| 模块 | 路径 |
|------|------|
| 主入口 | `run_exp_lme.py` |
| 身份与目录布局 | `src/utils/experiment_artifacts.py` |
| 配置与 variant 展开 | `src/utils/config.py` |
| Judge 独立输出 | `src/pipeline_lme_evaluate.py` |
| metrics 聚合 | `src/utils/experiment_metrics.py`、`script/aggregate_experiment_metrics.py` |
