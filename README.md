# easy-mem

在长对话记忆评测基准上，对比多种记忆方法的实验框架。支持 LongMemEval (LME) / MEME 两个基准，比较 `relation_decision` / `amac` / `zep` / `mem0` / `add_all` / `evermemos` 六种灌库策略。

## 快速开始

### 1. 安装

```bash
cd /path/to/fact_memory
uv sync
source .venv/bin/activate
```

### 2. 配置密钥（`.env`）

在项目根目录创建 `.env`（由 [`src/utils/env.py`](src/utils/env.py) 自动加载）：

```bash
VLLM_API_KEY=your_key
VLLM_BASE_URL=http://localhost:8000/v1/
EMBEDDING_BASE_URL=http://localhost:7110/v1/
EMBEDDING_API_KEY=your_key
DASHSCOPE_API_KEY=your_key   # 云端 qwen3-max 等 judge 模型
```

模型别名（如 `gemma4-26B`、`qwen3-max`）定义于 [`src/utils/llm_api.py`](src/utils/llm_api.py)。

### 3. 启动模型服务

| 脚本 | 用途 |
|------|------|
| [`script/0_run_model.sh`](script/0_run_model.sh) | 对话模型服务（默认 gemma4-26B，端口 7111） |
| [`script/0_run_embedding.sh`](script/0_run_embedding.sh) | Embedding 服务（qwen3-embedding-0.6b，端口 7110） |

多模型并行时，手动指定不同端口和 GPU，在 `.env` 中配置 `PORT_{MODEL_NAME}` 变量（见 [`src/utils/llm_api.py`](src/utils/llm_api.py)）。

### 4. 运行实验

**新实验默认使用内容寻址的 artifact 布局**（见下文「实验产物」）。旧目录 `MemDB/`、`experiment/` 仅用于历史结果；如需沿用旧路径，显式加 `--legacy-layout`。

```bash
# LME (LongMemEval)
uv run --no-sync python run_exp_lme.py --config config/exp_N0_gemma4-26b_rd_addall.yaml
```

---

## 实验入口

### `run_exp_lme.py`

```bash
uv run --no-sync python run_exp_lme.py \
  [--config config/exp_N0_gemma4-26b_rd_addall.yaml] \
  [--stages extract,ingest,generate,evaluate] \
  [--artifacts-root artifacts]
```

阶段：
- **extract** — 候选记忆抽取
- **ingest** — 每个 enabled 方法写入向量库（`relation_decision` 在灌库时就地融合答题记忆）
- **generate** — 预建库检索 → Agent 答题 → `pred.jsonl`
- **evaluate** — LLM Judge → 独立 `judged.jsonl` + `metrics.json`（不改写 `pred.jsonl`）

```bash
# 只跑部分阶段（例如已有 ingest，只换 token limit 重跑答题）
uv run --no-sync python run_exp_lme.py \
  --config config/my_exp_tl512.yaml \
  --stages generate,evaluate

# 同时比较多个方法：在 config 中将需要的 methods 设为 enabled: true
uv run --no-sync python run_exp_lme.py --config config/my_exp.yaml

# 仅复现/续跑旧实验（旧 MemDB/experiment 路径）
uv run --no-sync python run_exp_lme.py --config config/old.yaml --legacy-layout
```

### 实验产物（新架构，默认）

每次运行会在 `artifacts/runs/<run_id>/` 写入不可变快照：`run.yaml`、`manifest.json`、`stages.json`、`attempts/<id>/attempt.json`。

**全局阶段缓存**（按内容指纹复用，跨 run 共享）：

```text
artifacts/stages/candidates/<candidate_id>/
artifacts/stages/ingest/<method>/<ingest_id>/
```

**单次 run 专属产物**（随 token limit、答题/Judge 配置变化）：

```text
artifacts/runs/<run_id>/
  answer/<method>/<answer_id>/pred.jsonl
  answer/<method>/<answer_id>/agent_trace/
  judge/<method>/<judge_id>/judged.jsonl
  judge/<method>/<judge_id>/metrics.json
```

典型用法：
- 只改 `generate.memory_token_limit`（如 256 → 512）→ 复用同一 `candidate_id` / `ingest_id`，只跑 `generate,evaluate`
- 同一 config 开多个 `methods.*.enabled: true` → 共享 candidates，各方法独立 ingest / answer / judge 目录
- 配置 `sweep` + `replication` → 主入口自动展开 token limit × repeat（见「配置文件」）

更完整的说明见 [`docs/experiment-artifacts.md`](docs/experiment-artifacts.md)。

---

## 配置文件

所有实验参数集中在一个 YAML，**无需改代码**。

| 文件 | 用途 |
|------|------|
| [`config/exp_N0_gemma4-26b_rd_addall.yaml`](config/exp_N0_gemma4-26b_rd_addall.yaml) | LME hybrid 主实验（filler=0, gemma4-26B） |
| `config/exp_N{0,2,4,6,8}_{model}_{method}.yaml` | 完整实验矩阵（N=filler数, model=答题模型, method=灌库方法） |

### 常用修改

```yaml
experiment:
  benchmark: lme_s        # 切换基准：lme_s（hybrid）/ lme_o（oracle）
  suffix: exp001          # 可读标签（写入 run_id slug；阶段复用由指纹决定，勿再靠手工 bump suffix）

models:
  answer: gemma4-26B      # 换答题模型

methods:                  # 开/关灌库方法（同时开多个 = 比较实验）
  relation_decision:
    enabled: true
  mem0:
    enabled: true
  add_all:
    enabled: true

generate:
  memory_token_limit: 256
  answer_stratified_sample: 500   # 分层抽样题数（0 = 全量）

# 可选：一次配置扫多个 token limit
sweep:
  memory_token_limits: [256, 512]

# 可选：统计重复（mean / std）
replication:
  count: 3
  scope: answer_judge     # answer_judge：复用 ingest；full_pipeline：每 repeat 重跑上游
  seeds: [43, 44, 45]   # 省略则从 generate.answer_sample_seed 递增
```

多 repeat 跑完后，用 [`script/aggregate_experiment_metrics.py`](script/aggregate_experiment_metrics.py) 聚合各次 `metrics.json` 的均值与样本标准差（详见 [`docs/experiment-artifacts.md`](docs/experiment-artifacts.md)）。

### 配置节说明

| 节 | 说明 |
|----|------|
| `experiment` | benchmark、可读 suffix |
| `models` | extract / manager / answer / judge / embedding 五个模型 |
| `extract` | 候选抽取：granularity、language、模板列表；改抽取策略须换 `candidate_suffix` |
| `methods` | 每个方法的 `enabled` 及方法特定超参 |
| `generate` | 检索 topk、memory token limit、hybrid、分层抽样 |
| `evaluate` | judge CoT、抽样 |
| `sweep` | 可选：多个 `memory_token_limits`，主入口自动展开 variant |
| `replication` | 可选：重复次数、scope、seeds |
| `parallel` | 各阶段并发数 |
| `token_limits` | 各阶段最大 token |
| `prompts` | Jinja 模板文件名覆盖（空 = 用代码内置默认） |

---

## Benchmark 数据集

| Benchmark | 说明 | 数据文件 |
|-----------|------|---------|
| `lme_o` (Oracle) | 使用原始 session 完整上下文作为记忆库（理论上界） | `longmemeval_oracle_converted.json` |
| `lme_s` (Single) | 单人对话 cleaned 数据 | `longmemeval_s_cleaned_converted.json` |
| `lme_s_golden` (Hybrid Golden) | Golden memory + BM25-dense 混合检索（**主实验**） | `longmemeval_s_hybrid_golden_converted.json` |
| `lme_m` (Multi) | 多人对话数据 | `longmemeval_m_cleaned_converted.json` |

## 实验状态

### 主实验矩阵（LME-S Hybrid）

实验矩阵：5 filler 等级 × 7 模型 × 3 灌库方法 × 2 token limit（256/512）

| 模型 | 状态 |
|------|------|
| `gemma4-26B` | ✅ 完成（N0/N2/N4/N6/N8，全部方法，tl256/512） |
| `gemma4-e4b` | ✅ 完成（N0/N2/N4/N6/N8，全部方法，tl256/512） |
| `Qwen3-4B` | ✅ 完成（N0/N2/N4/N6/N8，全部方法，tl256/512） |
| `Qwen3-8B` | ✅ 完成（N0/N2/N4/N6/N8，全部方法，tl256/512） |
| `Qwen3-32B` | 🔄 进行中（仅 N0 rd_addall 完成） |
| `Qwen3.5-4B` | 🔄 进行中（rd_addall N0-N6 完成，其余待跑） |
| `Qwen3.5-9B` | 🔄 进行中（仅 N0 rd_addall 完成） |

### Oracle 基线

| 模型 | 状态 |
|------|------|
| `gemma4-26B` | ✅ 完成（全部 4 方法，tl256） |

### 方法说明

| 方法 | 缩写 | 说明 |
|------|------|------|
| `relation_decision` + `add_all` | rd_addall | 关系分类灌库 + 全量基线（同一 config 内比较） |
| `evermemos` | evm | EverMemOS 增量语义聚类 |
| `mem0` | mem0 | Mem0 风格增量更新 |

### 数据获取

预处理数据集文件较大（200MB+），未包含在 Git 仓库中。数据集托管在 HuggingFace：

```bash
wget https://huggingface.co/datasets/Qys77/easy-mem-data/resolve/main/easy-mem-data.zip
unzip easy-mem-data.zip -d .
```

候选记忆（`easy-mem-candidates.zip`）由维护者通过网盘等方式提供。

---

## 灌库方法说明

| `--update-method` | 说明 |
|-------------------|------|
| `relation_decision` | 成对五类关系分类（IND/EQV/NSO/OSN/CON）→ 灌库时就地融合答题记忆（同库 `--answer-mode` 检索） |
| `amac` | A-MAC 五维准入评分（效用 U / 置信 C / 新颖 N / 时效 R / 类型先验 T）过滤后写入 |
| `zep` | Graphiti + Kuzu 知识图谱写入（Kuzu 偶发崩溃，流水线自动重启续传） |
| `mem0` | Mem0 风格：LLM 判断增 / 改 / 删 |
| `add_all` | 全量直接写入，不做判断 |
| `evermemos` | EverMemOS 增量语义聚类 + LLM 合并 |

---

## 流水线核心脚本

手工单步调用时 `PYTHONPATH` 需包含 `src`：

```bash
export PYTHONPATH=src
```

| 脚本 | 说明 |
|------|------|
| `src/pipeline/extract_candidates.py` | 候选记忆抽取 |
| `src/pipeline/ingest_candidates.py` | 候选写库（`--update-method` 选方法） |
| `src/pipeline_lme_generate.py` | 预建库检索 + Agent 答题 |
| `src/pipeline_lme_evaluate.py` | LLM Judge（LME 基准；支持独立 `--output` / `--metrics-output`） |
| `script/aggregate_experiment_metrics.py` | 聚合多次 repeat 的 `metrics.json`（mean / sample std） |

---

## 项目结构

```
run_exp_lme.py              # 主入口（默认 artifact 布局）
config/                     # 实验 YAML 配置（exp_N*_* 命名）
docs/
  experiment-artifacts.md   # 新架构：阶段复用、sweep、replication、聚合
artifacts/                  # 新实验产物（运行时生成）
  stages/                   # 全局内容寻址：candidates / ingest（可跨 run 复用）
  runs/                     # 每次 run：manifest、answer、judge、attempts
src/
  agent/                    # StandardAgent（检索 → 拼上下文 → 答题）
  benchmark/                # 基准数据加载（LME）
  memory/
    admission/              # A-MAC 五维准入评分
    candidate_ingest/       # relation_decision、add_all、amac、evermemos 写库逻辑
    fusion/                 # relation_decision 专用：关系包融合
    mem0/                   # Mem0 风格增量更新
    zep/                    # Graphiti + Kuzu 知识图谱适配
    storage/                # LocalFaiss 向量库封装
  pipeline/                 # extract_candidates、ingest_candidates
  pipeline_lme_generate.py  # 检索 + Agent 答题
  pipeline_lme_evaluate.py  # LLM Judge（独立 judged/metrics 输出）
  prompts/                  # Jinja 模板（抽取 / 关系分类 / 融合 / Judge / Agent）
  utils/
    experiment_artifacts.py # run_id、阶段指纹、ArtifactLayout
    experiment_metrics.py   # repeat metrics 聚合
    llm_api.py、config.py 等
script/
  0_run_model.sh            # 模型服务启动
  0_run_embedding.sh        # embedding 服务启动
  aggregate_experiment_metrics.py
  build_hybrid_golden_dataset.py
  build_lme_golden_memory_v2.py
tests/                      # pytest
```

---

## 测试

```bash
uv run --no-sync pytest
```
