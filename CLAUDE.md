# CLAUDE.md

本文件给在此仓库工作的 Claude 提供项目级约定。**回答一律用中文。**

> 📖 **协作者上手文档**：[`docs/collab-guide.md`](docs/collab-guide.md) — 从零搭建环境到跑通全实验的完整指南。

## 项目是什么

`easy-mem` / `fact_mem`：长对话记忆评测框架。在 **LongMemEval (LME)** 基准上，对比
`relation_decision` / `amac` / `zep` / `mem0` / `add_all` / `evermemos` 六种灌库策略。

当前论文主实验：**hybrid 数据集**（golden memory + BM25-dense 混合检索），
同时包含 **fusion 消融实验**（`fusion_enabled: false`，用分类器保留新旧中的一条，不用 LLM 融合）。

## 环境与运行

> ⚠️ **重要**：本仓库只包含代码和实验配置。**模型部署、API key、端口映射均由协作者自行配置**，不会随代码提供。`.env` 文件是连接代码与模型服务的唯一桥梁——你需要根据自己的模型部署情况填写 `.env`，并在 YAML config 中指定正确的模型别名。

### 依赖安装

- 包管理用 **uv**。跑任何脚本用 `uv run --no-sync python ...`（裸 `python` 缺 `openai`/`dotenv` 等依赖；不带 `--no-sync` 的 `uv run` 会重新 sync，可能改坏已验证好的环境）。
- 本机 glibc 2.31 → **vllm ≤ 0.19.1**；gemma4 需 transformers 5.x。

### 模型需求

实验需要 5 类模型，按用途分工：

| 用途 | 配置字段 | 推荐模型 | 部署方式 | 最低 GPU |
|------|---------|---------|---------|---------|
| 记忆抽取 | `models.extract` | gemma4-26B | 本地 vllm | 4×GPU (TP=4) |
| 灌库管理 | `models.manager` | gemma4-26B / gemma4-e4b | 本地 vllm | 4×GPU / 2×GPU |
| 答题 | `models.answer` | gemma4-26B | 本地 vllm | 4×GPU (TP=4) |
| Judge | `models.judge` | deepseek-v4-flash | 云端 API | 无需 |
| Embedding | `models.embedding` | qwen3-embedding-0.6b | 本地 vllm `--task embed` | 1×GPU |

**协作者模型选择建议**：
- **只用云端 API**：在 config 中把 `extract`/`manager`/`answer` 都改为 `qwen3-max` 或 `gpt-4o-mini`，无需 GPU。注意云端 API 有费用，extract 阶段 token 消耗较大。
- **混合部署**：embedding 必须本地（`qwen3-embedding-0.6b`，仅需 1 GPU），其他模型可云端。
- **全本地**：按上表部署多个 vllm 实例。`extract`/`manager`/`answer` 可共用同一模型实例（设为相同模型名）。
- 模型别名定义于 `src/utils/llm_api.py`，可通过改 YAML 的 `models.*` 字段换模型。

### .env 配置（连接代码与模型服务）

`.env` 文件的作用：告诉代码「每个模型别名对应哪个实际服务」。`src/utils/llm_api.py` 根据 `.env` 和 YAML config 中的 `models.*` 字段，将请求路由到正确的 API endpoint。

从模板复制并填写：`cp .env.example .env`

```bash
# === 本地 VLLM 模型（需 GPU 部署） ===
VLLM_API_KEY=zjj                            # 与启动脚本 --api-key 一致
VLLM_BASE_URL=http://localhost:8000/v1/     # 默认地址
# 多模型时每个模型指定端口（命名规则：PORT_{模型别名大写，-变_}）
PORT_GEMMA4_26B=7111
PORT_GEMMA4_E4B=7115

# === Embedding 服务（必须本地） ===
EMBEDDING_BASE_URL=http://localhost:7110/v1/
EMBEDDING_API_KEY=zjj

# === 云端 API（Judge 必须用 deepseek-v4-flash） ===
DEEPSEEK_API_KEY=your_key                   # deepseek-v4-flash judge（必须）
```

**工作流程**：
1. YAML config 中写 `models.answer: gemma4-26B`
2. `llm_api.py` 查 `.env` 中 `PORT_GEMMA4_26B` → 得到端口 7111
3. 请求发到 `http://localhost:7111/v1/`

如果没有配对应 `PORT_*` 变量，则 fallback 到 `VLLM_BASE_URL`。云端模型（qwen3-max、gpt-4o-mini 等）不需要 `PORT_*`，只需对应的 `*_API_KEY`。

密钥从根目录 `.env` 经 dotenv 自动加载（`env | grep` 看不到，需在 python 里 `load_dotenv()`）。

### 启动模型服务

```bash
# Embedding（必须先启动）
bash script/0_run_embedding.sh

# 对话模型（按需启动）
bash script/0_run_model.sh                  # gemma4-26B 示例
bash script/0_run_qwen3_8b.sh               # Qwen3-8B 示例
```

## 代码约定

- 提示模板在 `src/prompts/templates/*.jinja`，用 `from prompts import render_prompt` 渲染。
  loader 用 **`[[ ... ]]`** 作变量分隔符（不是默认 `{{ }}`），但 block 标签仍是默认 `{% ... %}`。
  `FileSystemLoader` 指向 `templates/`，子目录需带前缀访问。
- LLM 客户端：`load_api_chat_completion(model_name, async_=False)`。同步 `get_response_chat(messages, max_new_tokens, temperature, ...)`，
  重试耗尽或内容安全失败返回 `None`（上游需判空）。批量并发可用 `ThreadPoolExecutor` 包同步客户端。
- 实验参数集中在 YAML（`config/exp_N*_*.yaml`），**改抽取策略必须同步换 `candidate_suffix`**，否则复用旧 state 跳过 episode。
- 记忆抽取已统一为单 pass、user 中心模板 `0_mem_extract_aspect_unified_en.jinja`。

## 实验入口

> 🚫 **禁止使用 `--legacy-layout`**：所有新实验必须走新 artifact 布局（默认），产物在 `artifacts/`。
> `--legacy-layout`（`MemDB/` + `experiment/` 旧路径）仅用于兼容历史数据，**新实验一律不准用**。
> 所有 ingest/extract/generate/evaluate 都必须通过 `run_exp_lme.py`（新框架）运行，禁止使用直接脚本
> （如 `ingest_candidates.py`、`pipeline_lme_generate.py`）跑新实验，否则新框架无法追踪 stage 指纹。

```bash
# 主入口（新 artifact 布局，产物在 artifacts/）
uv run --no-sync python run_exp_lme.py [--config config/exp_N0_gemma4-26b_rd_addall.yaml] [--stages extract,ingest,generate,evaluate]

# 只跑部分阶段（例如只换 token limit 时复用已有 ingest）
uv run --no-sync python run_exp_lme.py --config config/tl512.yaml --stages generate,evaluate
```

四个阶段：
- **extract** — 候选记忆抽取（单 pass 统一模板）→ `artifacts/stages/candidates/<candidate_id>/`
- **ingest** — 每个 enabled 方法写入向量库；`relation_decision` 在灌库时就地融合（不再有独立 fuse 阶段）→ `artifacts/stages/ingest/<method>/<ingest_id>/`
- **generate** — 预建库检索 → Agent 答题 → `artifacts/runs/<run_id>/answer/<method>/<answer_id>/pred.jsonl`
- **evaluate** — LLM Judge → 独立 `judged.jsonl` + `metrics.json`（**不改写** pred）

配置中 `methods` 下同时开多个 `enabled: true` = 比较实验；共享 candidates，各 method 独立 ingest / answer / judge。

### Artifact 布局（新实验默认）

- **内容寻址**：目录名由阶段指纹决定（配置字段 + 模板内容 hash），不靠手工 bump suffix。
- **阶段复用**：只改 `generate.memory_token_limit` 等新 answer 参数时，复用同一 `candidate_id` / `ingest_id`，只重跑 generate/evaluate。
- **run 快照**：`artifacts/runs/<run_id>/` 含 `manifest.json`、`run.yaml`、`stages.json`、`attempts/*/attempt.json`。
- **sweep × replication**：YAML 中 `sweep.memory_token_limits` × `replication.count` 展开为多个 variant；`replication.scope: answer_judge`（默认）共享 extract/ingest，`full_pipeline` 每个 repeat 独立上游。
- **聚合**：多 repeat 的 `metrics.json` 用 `script/aggregate_experiment_metrics.py` 算 mean/sample_std。
- **完整用法**：[`docs/experiment-artifacts.md`](docs/experiment-artifacts.md)。

## Benchmark 与数据集

| Benchmark | 说明 | 数据文件 |
|-----------|------|---------|
| `lme_o` (Oracle) | 完整 session 上下文作为记忆库（理论上界） | `longmemeval_oracle_converted.json` |
| `lme_s` (Single) | 单人对话 cleaned 数据 | `longmemeval_s_cleaned_converted.json` |
| `lme_s_golden` (Hybrid Golden) | Golden memory + BM25-dense 混合检索（**主实验**） | `longmemeval_s_hybrid_golden_converted.json` |
| `lme_m` (Multi) | 多人对话数据 | `longmemeval_m_cleaned_converted.json` |

主实验使用 `benchmark: lme_s`（Hybrid Golden 数据，由 `script/build_hybrid_golden_dataset.py` 构建）。

> ⚠️ **Hybrid 数据集评测以 470 题为准**：hybrid golden 数据集共 500 条，其中 30 条为 abstention（`golden_memory=[]`），实际可答题 470 条。计算 accuracy 时应以 470 为分母（或确保 evaluate 脚本正确过滤 abstention），否则与 500 题全量计算的结果不可比。其他模型已有结果均基于 470 题。

## 实验状态

实验矩阵：**5 filler 等级（N0/N2/N4/N6/N8）× 7 模型 × 3 灌库方法 × 2 token limit（256/512）**。

### 已完成 ✅

| 模型 | 覆盖范围 |
|------|---------|
| `gemma4-26B` | 全部 filler × 全部方法 × tl256/512 |
| `gemma4-e4b` | 全部 filler × 全部方法 × tl256/512 |
| `Qwen3-4B` | 全部 filler × 全部方法 × tl256/512 |
| `Qwen3-8B` | 全部 filler × 全部方法 × tl256/512 |
| Oracle (`gemma4-26B`) | 全部 4 方法 × tl256 |

### 进行中 🔄

| 模型 | 已完成 | 待跑 |
|------|--------|------|
| `Qwen3-32B` | N0 rd_addall | 其余 filler × 方法 |
| `Qwen3.5-4B` | N0-N6 rd_addall | N8, evm, mem0 |
| `Qwen3.5-9B` | N0 rd_addall | 其余 filler × 方法 |

### 方法缩写

| Config 中 method 名 | 实际灌库方法 |
|---------------------|-------------|
| `rd_addall` | `relation_decision` + `add_all`（同一 config 内比较） |
| `evm` | `evermemos` |
| `mem0` | `mem0` |

## LoCoMo 实验

LoCoMo（Long Conversation Memory）是一个双人长对话记忆评测基准。与 LME 不同：
- 每段对话有 **2 位命名说话者**（真实姓名，不映射为 user/assistant）
- 每段对话 = 1 个 episode，含 19-32 个 session，105-260 道 QA 题
- 每个 conversation 对应一个独立的 memory store
- 记忆包含对话 ID 证据追踪（`evidence` 字段），不参与记忆管理决策
- 抽取按 session 整块处理（`granularity: all`），对话格式为 `[dia_id] Speaker: text`
- 图片 turn 使用 `blip_caption` 作为文本信息

### 数据

- 原始数据：`data/raw_data/locomo10.json`（10 conversations，1,986 QAs）
- 候选记忆（HuggingFace）：`wget https://huggingface.co/datasets/Qys77/easy-mem-data/resolve/main/easy-mem-candidates-locomo.zip` → 解压到项目根目录 → `artifacts/stages/candidates/7a91a190/`
- 候选记忆本地路径：`artifacts/stages/candidates/7a91a190/`（candidate_id = `7a91a190`，4,127 条，已抽取）

### 模板

| 阶段 | 模板 | 说明 |
|------|------|------|
| 抽取 | `Extract_Stage_LoCoMo.jinja` | 按 session 整块抽取，输出含 evidence |
| RD 分类 | `RD_0_LoCoMo_relation_classify.jinja` | 命名说话者、时序/事件规则 |
| 答题 | `pipeline_answer.jinja`（复用 LME） | 含 "the user" 措辞，对 LoCoMo 可接受 |
| Judge | `pipeline_judge.jinja`（复用 LME） | 与 benchmark 无关 |

### 代码改动

LoCoMo 相关代码：

| 文件 | 改动 |
|------|------|
| `src/benchmark/locomo.py` | LoCoMo 数据加载器 |
| `src/benchmark/__init__.py` | `get_benchmark("locomo")` 路由 |
| `src/benchmark/datasets.py` | 注册 `"locomo"` 数据路径 |
| `src/utils/mem_extract_schemas.py` | `LoCoMoMemExtractResponse` + `normalize_extract_memories()` |
| `src/pipeline/extract_candidates.py` | LoCoMo 模板选择、对话格式、render_kwargs 传递、evidence 合并 |

LoCoMo 自动检测：`episode.metadata.get("benchmark") == "locomo"` → 切换 LoCoMo 模式，LME 不受影响。

### 实验入口

```bash
# ========== 抽取候选记忆（已完成，candidate_id = 7a91a190，勿重跑）==========
# 产物在 artifacts/stages/candidates/7a91a190/，也可从 HuggingFace 下载

# ========== 灌库（add_all）==========
# 通过 run_exp_lme.py（推荐，新框架自动管理路径）：
#   uv run --no-sync python run_exp_lme.py --config config/locomo_default.yaml --stages ingest
# 或手动指定路径：
PYTHONPATH=src uv run --no-sync python src/pipeline/ingest_candidates.py \
  --benchmark locomo \
  --update-method add_all \
  --candidates-dir artifacts/stages/candidates/7a91a190 \
  --database-root MemDB/ingest/locomo/add_all \
  --candidate-extract-model gemma4-26B \
  --candidate-suffix locomo_default \
  --embedding-model qwen3-embedding-0.6b \
  --manager-model gemma4-26B \
  --language en \
  --add-all-episode-concurrency 10

# ========== 灌库（relation_decision）==========
PYTHONPATH=src uv run --no-sync python src/pipeline/ingest_candidates.py \
  --benchmark locomo \
  --update-method relation_decision \
  --candidates-dir artifacts/stages/candidates/7a91a190 \
  --database-root MemDB/ingest/locomo/relation_decision \
  --candidate-extract-model gemma4-26B \
  --candidate-suffix locomo_default \
  --embedding-model qwen3-embedding-0.6b \
  --manager-model gemma4-26B \
  --language en \
  --relation-episode-concurrency 10 \
  --relation-concurrency 20 \
  --relation-system-template-en RD_0_LoCoMo_relation_classify.jinja

# ========== 答题（推荐 parallel_episodes=10, answer-concurrency=5）==========
PYTHONPATH=src uv run --no-sync python src/pipeline_lme_generate.py \
  --benchmark locomo \
  --method prebuilt \
  --database_root MemDB/ingest/locomo/<method> \
  --answer_model gemma4-26B \
  --embedding_model qwen3-embedding-0.6b \
  --retrieve_topk 50 \
  --memory_token_limit 256 \
  --output MemDB/pred/locomo/<method>/pred_tl256.jsonl \
  --parallel_episodes 10 \
  --answer-concurrency 5 \
  --agent_trace_dir MemDB/pred/locomo/<method>/agent_trace_tl256

# ========== 评估 ==========
PYTHONPATH=src uv run --no-sync python src/pipeline_lme_evaluate.py \
  --benchmark locomo \
  --judge_model deepseek-v4-flash \
  --input MemDB/pred/locomo/<method>/pred_tl256.jsonl \
  --output MemDB/pred/locomo/<method>/judged_tl256.jsonl \
  --metrics-output MemDB/pred/locomo/<method>/metrics_tl256.json \
  --max_concurrency 8 --max_new_tokens 512 --use_cot
```

### LoCoMo 与 LME 的关键差异

| 维度 | LME | LoCoMo |
|------|-----|--------|
| 对话角色 | user/assistant | 真实姓名 |
| Episode 结构 | 1 题/episode | 多题/episode（105-260） |
| 对话格式 | `**user**: text` | `[D1:3] Caroline: text` |
| 抽取粒度 | N turn/chunk | session 整块 |
| 记忆命名空间 | per-episode | per-conversation（= per-episode） |
| Evidence | 无 | 每条记忆带 dialogue ID |
| Filler | N0-N8 五级 | 无 filler 机制 |

### LoCoMo 当前结果

> 已跑：extract=gemma4-26B, answer=gemma4-26B, judge=deepseek-v4-flash, candidate_id=7a91a190

**gemma4-26B（tl=1024）**

| Method | Accuracy | 备注 |
|--------|----------|------|
| add_all | 73.2% | evaluate 完成 |

**gemma4-e4b（manager=gemma4-e4b, tl=256，ingest 完成，generate/evaluate 待跑）**

| Method | ingest | generate | evaluate |
|--------|--------|----------|----------|
| add_all | —（复用 gemma4-26B candidates） | 🔲 | 🔲 |
| RD | ✅ 10/10 | 🔲 | 🔲 |
| mem0 | ✅ 10/10 | 🔲 | 🔲 |
| evermemos | ✅ 10/10 | 🔲 | 🔲 |

**Qwen3.5-4B（manager=Qwen3.5-4B, tl=256，ingest 完成，generate/evaluate 待跑）**

| Method | ingest | generate | evaluate |
|--------|--------|----------|----------|
| add_all | —（复用 gemma4-26B candidates） | 🔲 | 🔲 |
| RD | ✅ 10/10 | 🔲 | 🔲 |
| mem0 | ✅ 10/10 | 🔲 | 🔲 |
| evermemos | ✅ 10/10 | 🔲 | 🔲 |

### LoCoMo ingest 产物路径

| 模型 | 方法 | 路径 |
|------|------|------|
| gemma4-26B | add_all | `artifacts/stages/ingest/add_all_test/` |
| gemma4-e4b | RD | `artifacts/stages/ingest/locomo_gemma4-e4b/relation_decision/` |
| gemma4-e4b | mem0 | `artifacts/stages/ingest/locomo_gemma4-e4b/mem0/` |
| gemma4-e4b | evermemos | `artifacts/stages/ingest/locomo_gemma4-e4b/evermemos/` |
| Qwen3.5-4B | RD | `artifacts/stages/ingest/locomo_qwen3.5-4b/relation_decision/` |
| Qwen3.5-4B | mem0 | `artifacts/stages/ingest/locomo_qwen3.5-4b/mem0/` |
| Qwen3.5-4B | evermemos | `artifacts/stages/ingest/locomo_qwen3.5-4b/evermemos/` |

### LoCoMo 待跑：generate + evaluate

两个模型（gemma4-e4b, Qwen3.5-4B）× 4 方法（add_all, RD, mem0, evermemos），tl=256。

add_all 的 ingest 复用 gemma4-26B 那套（`artifacts/stages/ingest/add_all_test/`），只需换 answer model 重跑 generate+evaluate。

**注意**：
- LoCoMo answer 推荐 `--parallel_episodes 10 --answer-concurrency 5`
- Qwen3.5-4B 有 thinking 模式，代码已通过 `enable_thinking=false` 自动关闭
- Qwen3.5-4B RD 分类较慢，已降低并发（`--relation-episode-concurrency 5 --relation-concurrency 10`）
- gemma4-e4b 和 Qwen3.5-4B 不能同时跑（共享 GPU），需串行

## MEME 实验

MEME（Memory Evaluation for Multisession Entities）是一个多 session 实体记忆评测基准。
与 LME/LoCoMo 不同：
- 每 episode 含 ~7 道 after questions（before questions 不参与评测）
- 对话格式：标准 user/assistant（与 LME 一致）
- 抽取模板：`0_mem_extract_dense_en.jinja`（全量密集抽取），4-turn 粒度
- 全量评测，不抽样

### 数据

- 数据集：`data/raw_data/MEME/meme_filler32k.json`（100 episodes，包含于 `easy-mem-data.zip`）
- 候选记忆（HuggingFace）：`wget https://huggingface.co/datasets/Qys77/easy-mem-data/resolve/main/easy-mem-candidates-meme.zip` → 解压到项目根目录 → `artifacts/stages/candidates/ff157d29/`
- 候选记忆本地路径：`artifacts/stages/candidates/ff157d29/`（candidate_id = `ff157d29`，100 episodes，已抽取）
- candidate_suffix: `meme_default`

### 代码

| 文件 | 说明 |
|------|------|
| `src/benchmark/meme.py` | MEME 数据加载器 |
| `src/benchmark/__init__.py` | `get_benchmark("meme_filler32k")` 路由 |
| `src/benchmark/datasets.py` | 注册 `"meme_filler32k"` 数据路径 |

### 实验入口

```bash
# 灌库 + 答题 + 评估（推荐通过 run_exp_lme.py）
uv run --no-sync python run_exp_lme.py --config config/meme_e4b.yaml --stages ingest,generate,evaluate
```

### MEME 配置

| Config | Manager 模型 | Answer 模型 | 状态 |
|--------|-------------|-------------|------|
| `config/meme_default.yaml` | gemma4-26B | gemma4-26B | ✅ 已完成 |
| `config/meme_e4b.yaml` | gemma4-e4b | gemma4-26B | ✅ 已完成 |
| `config/meme_q35.yaml` | Qwen3.5-4B | gemma4-26B | ✅ 已完成 |
| `config/meme_qwen35_9b.yaml` | Qwen3.5-9B | gemma4-26B | 🔲 待跑 |
| （待创建） | gemma4-12b-it | gemma4-26B | 🔲 待创建 config |

> 所有 MEME 实验共享同一套候选记忆（`candidate_id = ff157d29`，extract=gemma4-26B）。
> 不同 manager 模型的 ingest 独立（指纹含 `models.manager`），answer 阶段共用 gemma4-26B。
> `gemma4-12b-it` 模型 alias 尚未在 `src/utils/llm_api.py` 中注册，需先添加
> `"gemma4-12b-it": "gemma-4-12B-it"`（或实际模型 ID）映射。

### MEME 实验矩阵

| Manager 模型 | add_all | relation_decision | mem0 | evermemos |
|-------------|---------|-------------------|------|-----------|
| gemma4-26B | ✅ | ✅ | ✅ | ✅ |
| gemma4-e4b | ✅ | ✅ | ✅ | ✅ |
| Qwen3.5-4B | ✅ | ✅ | ✅ | ✅ |
| Qwen3.5-9B | 🔲 | 🔲 | 🔲 | 🔲 |
| gemma4-12b-it | 🔲 | 🔲 | 🔲 | 🔲 |

> **注意**：`add_all` 与 manager 模型无关（只使用 embedding），各 manager 的 add_all 指纹不同仅因 config 中 `models.manager` 字段参与指纹计算。如需共享 add_all ingest，可统一用 `meme_default.yaml` 跑一次 add_all，其余 config 只开 RD/mem0/evermemos。

## 消融实验

两组消融实验，均在 **hybrid 数据集**（`benchmark: lme_s`）上运行，覆盖所有 filler 等级（N0/N2/N4/N6/N8），token limit 256。

### 消融 1：关系类型消融（Relation Type Ablation）

**目标**：验证不同关系类型对记忆质量的贡献。比较 Add-all → CON-only → CON+EQV → Full RD 四条线。

**模型**：`gemma4-e4b`、`Qwen3.5-4B`

**已跑（复用主实验）**：
- **Add-all**：主实验 `exp_N{0,2,4,6,8}_{model}_rd_addall.yaml` 中的 `add_all` 结果
- **Full RD**：主实验中同 config 的 `relation_decision` 结果

**待跑**：

| 变体 | 关键参数 | 建议配置命名 |
|------|---------|-------------|
| CON-only | `active_relations: ["CON"]`，`fusion_enabled: true`（默认） | `exp_N{0,2,4,6,8}_{model}_rd_con.yaml` |
| CON+EQV | `active_relations: ["CON", "EQV"]`，`fusion_enabled: true`（默认） | `exp_N{0,2,4,6,8}_{model}_rd_coneqv.yaml` |

**操作步骤**：
1. 复制已有主实验 config：`cp config/exp_N0_gemma4-e4b_rd_addall.yaml config/exp_N0_gemma4-e4b_rd_con.yaml`
2. 修改 `experiment.suffix` 为唯一新标签（如 `con_N0`）
3. `methods.add_all.enabled` 设为 `false`
4. `methods.relation_decision` 下添加/修改 `active_relations`
5. **改配置后必须全量清理**（见「改配置后必须全量清理」节）

### 消融 2：Fusion 消融

**目标**：验证 LLM 融合步骤的贡献。比较 Full RD ↔ RD without fusion（简单按关系类型保留，不调 LLM 融合）。

**模型**：`gemma4-e4b`、`Qwen3.5-4B`

**已跑（复用主实验）**：
- **Full RD**：主实验中的 `relation_decision` 结果（同上）

**待跑**：

| 变体 | 关键参数 | 建议配置命名 |
|------|---------|-------------|
| RD without fusion | `fusion_enabled: false` | `exp_N{0,2,4,6,8}_{model}_rd_nofusion.yaml` |

**操作步骤**：
1. 复制已有主实验 config：`cp config/exp_N0_gemma4-e4b_rd_addall.yaml config/exp_N0_gemma4-e4b_rd_nofusion.yaml`
2. 修改 `experiment.suffix` 为唯一新标签（如 `nofusion_N0`）
3. `methods.add_all.enabled` 设为 `false`
4. `methods.relation_decision.fusion_enabled` 设为 `false`

### 消融实验矩阵总览

| 消融 | 对比项 | 模型 | 已跑 | 待跑 |
|------|-------|------|------|------|
| 关系类型 | Add-all | gemma4-e4b, Qwen3.5-4B | ✅ 主实验 | — |
| 关系类型 | CON-only | gemma4-e4b, Qwen3.5-4B | — | 🔲 5 filler × 2 模型 |
| 关系类型 | CON+EQV | gemma4-e4b, Qwen3.5-4B | — | 🔲 5 filler × 2 模型 |
| 关系类型 | Full RD | gemma4-e4b, Qwen3.5-4B | ✅ 主实验 | — |
| Fusion | RD without fusion | gemma4-e4b, Qwen3.5-4B | — | 🔲 5 filler × 2 模型 |

## 配置文件

关键 config：
- `config/exp_N{0,2,4,6,8}_{model}_{method}.yaml` — hybrid 主实验配置（仅 `lme_default.yaml` 被 git 跟踪，其余为本地生成）
- `config/exp_oracle_gemma4-26b_all.yaml` — oracle 全方法比较
- `config/lme_default.yaml` — **默认模板（带完整注释）**，协作者复制此文件修改即可
- 可选 `sweep.memory_token_limits` + `replication.{count,scope,seeds}` — 一次 YAML 展开多 variant（见 artifact 文档）

### 如何编写配置文件

1. **复制默认模板**：`cp config/lme_default.yaml config/my_exp.yaml`
2. **必须修改的字段**：
   - `experiment.suffix` — 每次新实验换一个唯一标签
   - `extract.candidate_suffix` — 换了抽取策略（模板/granularity）时必须换，否则复用旧 state
   - `models.*` — 按实际部署的模型选别名（见上方 VLLM 端口映射表）
3. **灌库方法**：在 `methods` 下将需要的方法设为 `enabled: true`，不需要的设 `false`
4. **命名约定**：建议 `exp_N{filler数}_{模型}_{方法}.yaml`，如 `exp_N0_gemma4-26b_rd_addall.yaml`
5. **改配置后必须全量清理**（见「改配置后必须全量清理」节），否则新旧指纹不匹配导致检索为空
6. **常见实验变体**：
   - 换 filler 数量：改 `extract.candidate_suffix` 指向不同 filler 的候选目录
   - 换答题模型：改 `models.answer`
   - 换 token limit：改 `generate.memory_token_limit` 或使用 `sweep.memory_token_limits`
   - 消融实验：在 `methods.relation_decision` 下设 `active_relations: ["CON"]` 或 `fusion_enabled: false`
   - 多方法比较：同时开多个 `methods.*.enabled: true`

数据路径映射定义于 `src/benchmark/datasets.py` 的 `DEFAULT_BENCHMARK_DATASETS`。主实验使用 `benchmark: lme_s`；hybrid 特性由 filler/candidate 系统提供（golden memory + distractor 混合），不由 benchmark key 控制。

> **数据获取**：以下文件未包含在 Git 仓库中，协作者需从以下渠道获取：
> - 数据集（HuggingFace）：`wget https://huggingface.co/datasets/Qys77/easy-mem-data/resolve/main/easy-mem-data.zip` → 解压到项目根目录 → `data/`
> - 候选记忆（HuggingFace）：`wget https://huggingface.co/datasets/Qys77/easy-mem-data/resolve/main/easy-mem-candidates.zip` → 解压到项目根目录 → `artifacts/stages/candidates/`
> - MEME 候选记忆（HuggingFace）：`wget https://huggingface.co/datasets/Qys77/easy-mem-data/resolve/main/easy-mem-candidates-meme.zip` → 解压到项目根目录 → `artifacts/stages/candidates/ff157d29/`

### 必须获取（否则无法跑实验）

| 数据 | 目标路径 | 大小 | 说明 |
|------|---------|------|------|
| 数据集 JSON | `data/preprocessed/` + `data/raw_data/` | ~3GB | LME-S/O/M 的原始和预处理数据 |
| 候选记忆 | `artifacts/stages/candidates/` | ~41MB | 预抽取的候选记忆，按 filler 等级分目录 |
| MEME 候选记忆 | `artifacts/stages/candidates/ff157d29/` | ~3.1MB | MEME benchmark 预抽取候选记忆（100 episodes） |

**候选记忆目录 → candidate_suffix 映射**（放置后 config 中 `extract.candidate_suffix` 才能匹配）：

| 目录 Hash | candidate_suffix | 说明 | Episodes |
|-----------|-----------------|------|----------|
| `5b3b093d` | `hybrid_filler_N0` | 无 filler（仅 golden memory） | 470 |
| `40d0a4e8` | `hybrid_filler_N2` | 2 条 distractor / 题 | 470 |
| `dac380a4` | `hybrid_filler_N4` | 4 条 distractor / 题 | 470 |
| `111c614b` | `hybrid_filler_N6` | 6 条 distractor / 题 | 470 |
| `cda53dff` | `hybrid_filler_N8` | 8 条 distractor / 题 | 470 |
| `200a19dc` | `oracle_dense` | Oracle 全量 session 上下文 | 501 |
| `ff157d29` | `meme_default` | MEME 32k filler 候选记忆 | 100 |

### 可选获取（可自行生成但耗时/费钱）

| 数据 | 目标路径 | 大小 | 说明 |
|------|---------|------|------|
| Ingest 向量库 | `artifacts/stages/ingest/` | ~66GB | 灌库产物。可跳过，自行跑 `--stages ingest` 生成，但 relation_decision 需 LLM 关系分类（约 470×3 次调用/episode） |

## relation_decision 灌库策略

核心流程（`src/memory/candidate_ingest/relation_decision.py`）：
1. 召回 top-k 相关旧记忆
2. 逐对关系分类（五类：IND/EQV/OSN/NSO/CON）
3. 按关系类型写入/替换/融合，构建关系图

### 分类后端

实验 YAML 中 `relation_decision.backend` 仅支持 **`llm`**（使用 `models.manager` 指定的模型 + `RD_0_relation_classify.jinja`）。注意 manager 模型不一定是 answer 模型——例如 `exp_N0_gemma4-e4b_rd_addall.yaml` 中 manager=gemma4-e4b，answer=gemma4-26B。

### Fusion 消融

`relation_decision.fusion_enabled` 控制关系包融合策略：
- `true`（默认）：LLM 将同一关系包内的多条记忆融合为一条
- `false`：简单按关系类型保留新旧中的一条（不调 LLM），CLI 通过 `--no-fusion` 传递

其他关键参数：
- `active_relations`：限定生效的关系类型，用于消融（如 `["CON"]` 仅冲突管理）
- `condition_sim_threshold` / `pairwise_sim_threshold`：嵌入相似度阈值

## LME golden memory

为 LME 每题生成「能推出答案的最小原子记忆集」。

- 脚本：`script/build_lme_golden_memory_v2.py`
  ```bash
  uv run --no-sync python script/build_lme_golden_memory_v2.py [--limit N --stratified] [--out ...]
  ```
- 数据锚点：`data/raw_data/longmemeval_s_cleaned.json`（500 题）。证据靠 `answer_session_ids` 找 evidence session + turn 上的 `has_answer=True` 标记定位。
- 流程：取证据(纯代码) → gemma4-26B 蒸馏(gold answer 反向锚定) → 仅用 GM 回测 + gpt-4o-mini judge 闭环重试(≤2)。
  abstention(`_abs` 结尾)不调 LLM，`golden_memory=[]` + `abstention=true`。
- 产物：`data/preprocessed/longmemeval_s_golden.json`，500 条（30 abstention + 470 可答），字段
  `{question_id, question, answer, question_type, evidence_session_ids, abstention, golden_memory:[{"content":..., "date":...}, ...], judged_correct}`。
- 实测：470 可答题；平均 **1.90 条/题**（multi-session/preference ~2.5，单 session ~1）。
- 相关模板：`lme_golden_memory_distill_en.jinja` / `lme_golden_memory_answer_en.jinja` / 复用 `pipeline_eval_oqa.jinja`(judge)。

## 关键踩坑与 footgun

- **新实验必须用新框架**：禁止 `--legacy-layout`，禁止直接调用 `ingest_candidates.py` / `pipeline_lme_generate.py` 等脚本跑新实验。所有阶段必须通过 `run_exp_lme.py`（新 artifact 布局）运行，确保 stage 指纹追踪正确。
- **禁止随意删除实验数据**：不要删除 `artifacts/` 下的任何目录或文件，除非用户明确要求删除特定 ingest/run。即使要重跑某个 ingest，也必须先和用户确认后再删。误删实验数据不可恢复。
- **新 benchmark 数据集必须放 `raw_data/`**：`LMEBenchmark` 只对 `raw_data/` 下的原始格式（format A）自动转换为 preprocessed 格式（format B）。直接放 `preprocessed/` 下的原始数据不会被转换，导致 `QAs` 为空、0 题被处理。注册时用 `_converted` 后缀路径（如 `lme_s_golden → longmemeval_s_hybrid_golden_converted.json`）。
- **后台任务不要用 `tail -20`**：会截掉错误信息，导致任务静默失败而看不到原因。建议直接 `2>&1` 全量输出，完成后再 `tail` 查看末尾。

- **直接调 pipeline 脚本**（不用 `run_exp_lme`）需设 `PYTHONPATH=src`，否则 `ModuleNotFoundError: No module named 'benchmark'`。
- **改抽取策略必须同步换 `candidate_suffix`**，否则 extract 阶段复用旧 state 跳过 episode。
- **新布局下 trace 已按 stage 隔离**（ingest trace 在 `artifacts/stages/ingest/<method>/<ingest_id>/trace/`，answer trace 在 `answer/.../agent_trace/`）。**legacy 布局**仍可能混写旧 `experiment/` trace。
- **不要把 token limit 写进 `experiment.suffix` 来区分实验**；token limit 已进入 answer 阶段指纹，改 limit 会自动新 answer/judge 并复用 ingest。
- **ingest 与 memory token limit 无关**：ingest 阶段只负责向量库灌入，不受 `generate.memory_token_limit` 影响。同一 filler + manager 的 ingest 被所有 token limit 的 variant 共享（tl256 和 tl512 共用一个 ingest 目录）。
- **新旧 metrics 不要混聚合**：legacy 的 `eval_judge.json` 与新布局的 `metrics.json` 不是同一 schema。
- **relation_decision 灌库时就地融合**答题记忆 C（同库），不再有独立的事后 fuse 阶段。`run_exp_lme.py` 中 `stage_ingest` 对 `relation_decision` 不额外调用 fuse 脚本。
- **非 zep 的 ingest 不再默认 `--trust-apply-marker`**；配置指纹不一致时会拒绝静默复用错误库。zep 仅在崩溃重试时保留 marker 信任。

## 项目结构

```
run_exp_lme.py              # 主入口（默认新 artifact 布局）
config/                     # 实验 YAML 配置
artifacts/                  # 新实验产物根（stages/ 全局复用 + runs/ 快照）
docs/experiment-artifacts.md  # artifact 架构完整使用指南
src/
  agent/                    # StandardAgent（检索 → 拼上下文 → 答题）
  benchmark/                # 基准数据加载（LME、MEME）
  memory/
    admission/              # A-MAC 五维准入评分
    candidate_ingest/       # relation_decision、add_all、amac 等写库逻辑
    fusion/                 # relation_decision 专用：关系包 LLM 融合
    mem0/                   # Mem0 风格增量更新
    zep/                    # Graphiti + Kuzu 知识图谱适配
    storage/                # LocalFaiss 向量库封装
  pipeline/                 # extract_candidates、ingest_candidates 子步骤
  pipeline_lme_generate.py  # 检索 + Agent 答题
  pipeline_lme_evaluate.py  # LLM Judge（独立 judged/metrics 输出）
  prompts/                  # Jinja 模板
  utils/
    experiment_artifacts.py # run_id、阶段指纹、ArtifactLayout
    experiment_metrics.py   # 多 repeat metrics 聚合
    llm_api.py、config.py 等
script/
  0_run_model.sh            # 模型服务启动
  0_run_embedding.sh        # embedding 服务启动
  aggregate_experiment_metrics.py  # CLI：repeat metrics → mean/std
  build_hybrid_golden_dataset.py   # hybrid 数据集构建
  build_lme_golden_memory_v2.py    # golden memory 生成
  build_unified_candidates.py      # 候选记忆抽取
  build_unified_filler.py          # filler 候选构建
MemDB/、experiment/         # legacy 布局历史产物（--legacy-layout）
```

## 测试

```bash
uv run --no-sync pytest
```

## 实验运行经验

### 停止实验

**必须同时杀父进程和所有子进程**，否则 `ingest_candidates.py` 等子进程变孤儿继续跑（用 API 的会持续烧钱）。

```bash
# 正确停止
pkill -f "run_exp_lme|ingest_candidates|pipeline_lme_generate|pipeline_lme_evaluate"

# 验证干净
ps aux | grep -E "run_exp_lme|ingest_candidates|pipeline_lme" | grep -v grep
```

**坑**：`pkill -f "run_exp_lme.py"` 只杀调度器，`subprocess.run` 启动的子进程不受影响，变孤儿由 init 接管。

### 改配置后必须全量清理

改任何 YAML 配置（尤其是 `prompts` 相关），旧的 ingest/answer/judge 指纹可能与新配置不匹配。**必须全量清理再重跑**：

```bash
rm -rf artifacts/stages/ingest/* artifacts/stages/locks/* artifacts/runs/*
```

不清理会导致：
- ingest 写到目录 A，answer 去目录 B 检索 → 检索返回 0 → accuracy 异常低
- 两个 stage 的 `ingest_id` 指纹不一致（排查方法：最终输出看 `ingest=` 路径是否与实际有数据的目录一致）

### 启动实验前检查残留进程

```bash
# 确保没有实验进程在跑
ps aux | grep -E "run_exp_lme|ingest_candidates" | grep -v grep
# 有残留先全杀
pkill -f "run_exp_lme|ingest_candidates|pipeline_lme"
```

### 多模型并行注意事项

- 不同 vllm 实例（不同端口）可以并行，互不干扰
- 共享同一 API key（如 DashScope）的模型不能并行跑，会触发 429 限流
- vllm 模型并发建议：episode=20, relation=20（约 400 并发），过高会 OOM

### 模板管理

- RD 分类模板：当前使用 `RD_0_relation_classify.jinja`（内容来自 legacy `RD_0_relation_classify_old.jinja`）
- RD 融合模板：`RD_1_fuse_{CON,EQV,NSO,OSN}.jinja`
- 模板引用作为 `relation_user_en` 进入 ingest 指纹，改名需全量清理
- 旧版模板在 `src/prompts/templates/legacy/`

### 候选数据

- 候选数据在 `artifacts/stages/candidates/<id>/`，由 migration 从 `MemDB/candidates/` 迁移
- 每个 episode 一个 JSON，加上 `extract_progress.state`
- `gp_*` 前缀的文件也是合法候选（filler 数据），不能排除

### VLLM 端口映射

模型服务端口通过 `.env` 中 `PORT_{MODEL_NAME}` 变量动态配置（如 `PORT_GEMMA4_26B=7111`），由 `src/utils/llm_api.py` 自动读取。

当前 vllm 模型别名与对应实际模型：

| 模型别名 | 实际模型 ID |
|---------|---------|
| `gemma4-26B` | `gemma-4-26B-A4B-it` |
| `gemma4-e4b` | `gemma-4-E4B-it` |
| `gemma4-e2b` | `gemma-4-E2B-it` |
| `gemma4-31B` | `gemma-4-31B-it` |
| `Qwen3-8B` | `Qwen3-8B` |
| `Qwen3-4B` | `Qwen3-4B` |
| `Qwen3-30B` | `Qwen3-30B-A3B-Thinking-2507` |
| `Qwen3.5-27B` | `Qwen3.5-27B` |
| `Qwen3.5-4B` | `Qwen3.5-4B` |
| `Qwen3.5-9B` | `Qwen3.5-9B` |
| `qwen3-embedding-0.6b` | embedding 服务（默认端口 7110） |

本地启动脚本（`script/0_run_model.sh`、`script/0_run_embedding.sh`）为单 GPU/单端口示例；多模型部署需手动指定不同端口和 GPU。

API key 统一 `VLLM_API_KEY=zjj`（从 `.env` 加载）。

### 各方法对 LLM 的依赖

- **`add_all`**：只使用 embedding（`models.embedding`），不调用 LLM。与 `models.manager` 无关。
- **`relation_decision`**：依赖 `models.manager` 做 LLM 关系分类。换 manager 模型需重跑 RD ingest。
- **`mem0`**：使用 `models.manager` 做 LLM 增量判断（增/改/删）。换 manager 模型需重跑。
- **`evermemos`**：使用 `models.manager` 做 LLM 语义合并。换 manager 模型需重跑。

### 监控 ingest 进度

**关键**：不同灌库方法的 ingest 目录结构不同，看进度的方法也不同。

#### 找到 ingest 目录

先通过 run 的 `stages.json` 找到 `ingest_id`：

```bash
# 查看某个 run 各 method 的 stage ID
python3 -c "
import json
with open('artifacts/runs/<run_id>/stages.json') as f:
    s = json.load(f)
for m, info in s['methods'].items():
    print(f\"{m}: ingest={info['ingest']['stage_id']}\")
"
```

ingest 产物在 `artifacts/stages/ingest/<method>/<ingest_id>/`。

#### 各方法目录结构

| 方法 | 目录命名 | 一个目录代表 | 总数 (LME) | 进度查看命令 |
|------|---------|-------------|-----------|-------------|
| **relation_decision** | 混合（数字 `0-9` + hash） | 一个 episode | 470 | `ls <dir> \| grep -vcE "trace\|stage_manifest\|progress"` |
| **add_all** | hash（`001be529` 等） | 一个 episode | 470 | 同上 |
| **mem0** | hash（`001be529` 等） | 一个 episode | 470 | 同上 |
| **evermemos** | hash（`001be529` 等） | 一个 episode | 470 | 同上 |

**所有方法的统一进度查看**：`ls <ingest_dir> | grep -vcE "trace|stage_manifest|progress"`（排除 meta 文件，计数 = 已完成 episode 数）。
hash 命名来自 `artifacts/stages/candidates/<id>/` 下的 candidate JSON 文件名（每个 JSON 对应一个 episode）。

#### Trace 日志

每个 ingest 目录下都有 `trace/` 子目录，存放每个操作的结构化日志（`.jsonl`）。
可通过 trace 文件数量和最后修改时间判断活跃度：

```bash
ls -lt artifacts/stages/ingest/<method>/<ingest_id>/trace/ | head -5
```

#### 后台运行时看进度

如果通过 `run_in_background` 跑了 ingest，输出被管道截断（如 `| tail -20`），
**无法从 output 文件看到进度条**。应直接检查 ingest 目录的条目数或 trace 文件时间戳。见上方各方法的计数命令。
