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
| [`script/0_run_embedding_ppu.sh`](script/0_run_embedding_ppu.sh) | Embedding 服务（vLLM `--task embed`） |
| [`script/0_run_model_ppu_*.sh`](script/) | 对话模型服务（gemma4-26B、qwen3-4B/8B/32B） |
| [`script/0_run_reranker_ppu.sh`](script/0_run_reranker_ppu.sh) | Qwen3-Reranker 精排服务（开启 rerank 时需要） |

### 4. 运行实验

```bash
# LME (LongMemEval)
python run_exp_lme.py

# MEME 4-Phase 协议
python run_exp_meme.py
```

---

## 实验入口

### `run_exp_lme.py`（LME / LongMemEval）

```bash
python run_exp_lme.py [--config config/lme.yaml] [--stages extract,ingest,generate,evaluate]
```

阶段：
- **extract** — 候选记忆抽取（三方面 LLM 调用，事件/偏好/社交关系）
- **ingest** — 每个 enabled 方法写入向量库（方法特定策略；`relation_decision` 含后续关系包融合）
- **generate** — 预建库检索 → Agent 答题 → 输出 JSONL
- **evaluate** — LLM Judge 写回评分

```bash
# 只跑部分阶段（如候选已有，跳过 extract）
python run_exp_lme.py --stages ingest,generate,evaluate

# 同时比较多个方法：在 config 中将需要的 methods 设为 enabled: true
python run_exp_lme.py --config config/my_exp.yaml
```

### `run_exp_meme.py`（MEME 4-Phase 协议）

```bash
python run_exp_meme.py [--config config/meme.yaml] [--stages extract,run,evaluate]
```

与通用流水线的区别：
- **run** 阶段：ingest + answer 合并为单次 `pipeline_meme_4phase.py` 调用，按 per-episode 4-phase 执行（before ingest → before Q&A → after ingest → after Q&A）
- **evaluate** 阶段：使用 `pipeline_meme_evaluate.py`（task-specific prompts + trivial-pass 过滤）

---

## 配置文件

所有实验参数集中在一个 YAML，**无需改代码**。

| 文件 | 用途 |
|------|------|
| [`config/lme.yaml`](config/lme.yaml) | LME (LongMemEval) 实验 |
| [`config/meme.yaml`](config/meme.yaml) | MEME 4-Phase 实验 |

### 常用修改

```yaml
experiment:
  benchmark: lme_s        # 切换基准：lme_o / lme_s / lme_m / meme_filler32k / ...
  suffix: exp001          # 实验版本标签

models:
  answer: Qwen3-30B       # 换答题模型

methods:                  # 开/关灌库方法（同时开多个 = 比较实验）
  amac:
    enabled: true
  zep:
    enabled: true
  relation_decision:
    enabled: false        # 改为 true 则同时跑 relation_decision

generate:
  answer_stratified_sample: 500   # 分层抽样题数（0 = 全量）
```

### 配置节说明

| 节 | 说明 |
|----|------|
| `experiment` | benchmark、实验 suffix |
| `models` | extract / manager / answer / judge / embedding 五个模型 |
| `extract` | 候选抽取：granularity、language、三方面模板列表 |
| `methods` | 每个方法的 `enabled` 及方法特定超参 |
| `generate` | 检索 topk、memory token limit、hybrid、rerank、分层抽样 |
| `evaluate` | judge CoT、抽样 |
| `parallel` | 各阶段并发数 |
| `token_limits` | 各阶段最大 token |
| `prompts` | Jinja 模板文件名覆盖（空 = 用代码内置默认） |

---

## 数据

预处理数据路径由 [`src/benchmark/datasets.py`](src/benchmark/datasets.py) 中 `DEFAULT_BENCHMARK_DATASETS` 管理。

| `benchmark` 值 | 默认数据文件 | 语言 |
|----------------|-------------|------|
| `lme_o` | `data/preprocessed/longmemeval_oracle_converted.json` | en |
| `lme_s` | `data/preprocessed/longmemeval_s_cleaned_converted.json` | en |
| `lme_m` | `data/preprocessed/longmemeval_m_cleaned_converted.json` | en |
| `meme_nofiller` | `data/raw_data/MEME/meme_nofiller.json` | en |
| `meme_filler32k` | `data/raw_data/MEME/meme_filler32k.json` | en |
| `meme_filler128k` | `data/raw_data/MEME/meme_filler128k.json` | en |

---

## 灌库方法说明

| `--update-method` | 说明 |
|-------------------|------|
| `relation_decision` | 成对五类关系分类（IND/EQV/NSO/OSN/CON）→ 弱边聚合 → LLM 关系包融合（产出 `_fused` 库供检索） |
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
| `src/pipeline/fuse_lme_memory_bundles.py` | relation_decision 关系包融合（`run_exp_lme.py` 自动调用） |
| `src/pipeline_lme_generate.py` | 预建库检索 + Agent 答题 |
| `src/pipeline_lme_evaluate.py` | LLM Judge（LME 基准） |
| `src/pipeline_meme_4phase.py` | MEME 4-phase 灌库+答题 |
| `src/pipeline_meme_evaluate.py` | MEME Judge |

---

## 项目结构

```
run_exp_lme.py          # 主入口：LME / LongMemEval
run_exp_meme.py         # 主入口：MEME 4-Phase
config/
  lme.yaml              # LME 实验配置
  meme.yaml             # MEME 实验配置
src/
  agent/                # StandardAgent（检索 → 拼上下文 → 答题）
  benchmark/            # 基准数据加载（LME、MEME）
  memory/
    baselines/          # lme_prebuilt（检索阶段 memory system）
    admission/          # A-MAC 五维准入评分
    candidate_ingest/   # relation_decision、add_all、amac、evermemos 写库逻辑
    fusion/             # relation_decision 专用：关系包融合
    mem0/               # Mem0 风格增量更新
    zep/                # Graphiti + Kuzu 知识图谱适配
    storage/            # LocalFaiss 向量库封装
  pipeline/             # extract_candidates、ingest_candidates、fuse 子步骤
  pipeline_lme_generate.py
  pipeline_lme_evaluate.py
  pipeline_meme_4phase.py
  pipeline_meme_evaluate.py
  prompts/              # Jinja 模板（抽取 / 关系分类 / 融合 / Judge / Agent）
  utils/                # llm_api、embed_utils、reranker、eval_report 等
script/
  run_exp.sh            # 旧 Shell 入口（保留兼容）
  0_run_*.sh            # 模型服务启动示例
viewer/                 # 实验可视化 HTML 构建脚本（可选）
test/                   # pytest
experiment/             # 实验输出（运行时生成）
MemDB/                  # 向量库与候选缓存（运行时生成）
logs/                   # memory_trace / agent_trace（运行时生成）
```

---

## 测试

```bash
pytest
```
