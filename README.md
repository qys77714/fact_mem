# easy-mem

在长对话记忆评测基准上，对比 **LME 候选记忆抽取 → 灌库 → 混合检索答题** 等流程的实验框架。完整实验通常由 **[`script/run_exp.sh`](script/run_exp.sh)** 编排（候选抽取 → 灌库 → 生成 → LLM Judge；其中 **`relation_decision` 灌库含关系包融合子步骤**，非与灌库并列的通用阶段；脚本头部注明了可选 HTML 对照等步骤）。

手工单步跑时，核心仍是：**加载基准数据 → 写入/检索记忆 → 标准 Agent 答题 → 输出 JSONL**，并可选用 LLM Judge 写回评分。

## 环境要求

- **Python** ≥ 3.12  
- 依赖见 [`pyproject.toml`](pyproject.toml)（FAISS、OpenAI 兼容客户端、Transformers、Jinja2、**Graphiti + Kuzu**（Zep 灌库路径）、rouge-score、pytest 等）。对话/向量 **vLLM 服务在运行时按需单独部署**，不必写进 `pyproject.toml`。  
- 推荐使用 [uv](https://github.com/astral-sh/uv)：

```bash
cd /path/to/fact_memory   # 仓库根目录（含 pyproject.toml）
uv sync
source .venv/bin/activate   # 可选；等价于对单次命令使用 uv run ...
```

运行时请将 **`PYTHONPATH` 包含 `src`**（`run_exp.sh` 已 `export PYTHONPATH=src`；手工执行时请 `export PYTHONPATH=src` 或使用 `python -m` 等在项目内约定的方式）。

## 推荐入口：一键实验 `run_exp.sh`

主入口为 **[`script/run_exp.sh`](script/run_exp.sh)**（配置 **[`script/run_exp.config.yaml`](script/run_exp.config.yaml)**）。典型用法：

```bash
# 仓库根目录
./script/run_exp.sh
RUN_EXP_CONFIG=/path/to/custom.yaml ./script/run_exp.sh
./script/run_exp.sh --config script/run_exp.config.yaml   # 仍可附加参数，透传给生成阶段
```

脚本内可调变量（模型名、`benchmark`、`candidate_suffix`、检索与 Judge 等）见 `run_exp.sh` 注释；并行度、token 上限、Jinja 模板名等多在 YAML 中。

## 配置与密钥

支持通过项目根目录的 `.env` 加载环境变量（[`utils.env.load_env`](src/utils/env.py)）。

| 用途 | 变量 | 说明 |
|------|------|------|
| 本地 vLLM 对话模型 | `VLLM_API_KEY`（必填）、`VLLM_BASE_URL`（默认 `http://localhost:8000/v1/`） | 与 [`llm_api.load_api_chat_completion`](src/utils/llm_api.py) 中注册的 served 名称一致 |
| 向量服务（OpenAI 兼容） | `EMBEDDING_BASE_URL`、`EMBEDDING_API_KEY` | 生成流水线里 embedding 调用；也可用 CLI `--embedding_base_url` / `--embedding_api_key` 覆盖 |
| 通义千问等云端模型 | `DASHSCOPE_API_KEY` 等 | 见 `llm_api.py` 中各 provider 分支 |

具体模型别名（如 `gemma4-26B`、`qwen3-max`）以 [`src/utils/llm_api.py`](src/utils/llm_api.py) 为准。

## 配置文件 `run_exp.config.yaml`

[`script/run_exp.config.yaml`](script/run_exp.config.yaml) 管理所有并发度、token 上限、Jinja 模板文件名及路径模板。主要节：

| 节 | 说明 |
|----|------|
| `parallel` | 各步骤并发数（`extract_candidates_chunk_concurrency`、`ingest_*_episode_concurrency`、`generate_lme_prebuilt_*`、`evaluate_*` 等） |
| `token_limits` | 各步骤最大 token（`tok_extract_candidates_max_new_tokens`、`tok_ingest_candidates_*`、`tok_generate_lme_prebuilt_memory_token_limit` 等） |
| `prompts` | 全流水线 Jinja 模板文件名（位于 `src/prompts/templates/`）；未写的键与 `run_exp_load_config.py` 内建默认合并；空字符串表示不传该参数、由脚本/代码内置默认处理 |
| `paths` | 输出目录与文件路径模板（支持 `${benchmark}`、`${manager_model}` 等 shell 变量展开） |

## 数据

预处理/原始数据按 [`src/benchmark/datasets.py`](src/benchmark/datasets.py) 中 **`DEFAULT_BENCHMARK_DATASETS`** 解析。内置 `--benchmark` 与默认文件如下（也可用 `--benchmark_file` 指定任意兼容 JSON；语言可用 `--language zh|en` 覆盖默认值）。

| `--benchmark` | 默认数据文件 | 默认语言 |
|----------------|--------------|----------|
| `lme_o` | `data/preprocessed/longmemeval_oracle_converted.json` | en |
| `lme_s` | `data/preprocessed/longmemeval_s_cleaned_converted.json` | en |
| `lme_m` | `data/preprocessed/longmemeval_m_cleaned_converted.json` | en |
| `locomo` | `data/raw_data/locomo10.json` | en |
| `lmb_event` | `data/preprocessed/LifeMemBench_event.json` | zh |
| `emb_event` | `data/preprocessed/EgoMemBench_event_half.json` | en |
| `meme_nofiller` | `data/raw_data/MEME/meme_nofiller.json` | en |
| `meme_filler32k` | `data/raw_data/MEME/meme_filler32k.json` | en |
| `meme_filler128k` | `data/raw_data/MEME/meme_filler128k.json` | en |

## 完整流水线步骤

通用阶段为四步：**候选抽取 → 灌库 → 生成 → 评测**。关系包融合**不是**与灌库并列的通用阶段，而是 **`relation_decision` 方法在关系分类写库之后的内置后处理**（[`src/pipeline/fuse_lme_memory_bundles.py`](src/pipeline/fuse_lme_memory_bundles.py)）；`mem0` / `zep` / `add_all` / `amac` 等其它灌库策略不含此步骤。

### 1. 候选抽取（`src/pipeline/extract_candidates.py`）

从对话历史中用 LLM 抽取候选记忆片段，输出写入 `MemDB/candidates/`。支持三方面模板（事件/偏好/社交关系）与主模板两种抽取模式，由 `run_exp.config.yaml` 的 `prompts.mem_extract_aspects_only` 控制。

### 2. 灌库（`src/pipeline/ingest_candidates.py`）

将候选记忆写入向量库。`--update-method` 支持五种策略：

| `--update-method` | 说明 |
|-------------------|------|
| `relation_decision` | 成对五类关系分类（IND/EQV/NSO/OSN/CON）后写弱边，桶内聚合；随后对关系包做 LLM 融合，产出 `relation_decision_fused` 库供检索（融合脚本见 [`fuse_lme_memory_bundles.py`](src/pipeline/fuse_lme_memory_bundles.py)） |
| `add_all` | 全量写入，不做关系判断 |
| `mem0` | Mem0 风格：LLM 判断增/改/删 |
| `zep` | 基于 Graphiti + Kuzu 的知识图谱写入 |
| `amac` | A-MAC 风格：五维准入评分（效用 U / 置信 C / 新颖 N / 时效 R / 类型先验 T）过滤后写入 |

### 3. 生成（`src/pipeline_generate.py`）

以 `--method lme_prebuilt --prebuilt-memory` 从预建库检索，经标准 Agent 答题后输出 JSONL。支持混合检索（BM25 + 稠密）、Qwen3 Reranker、`--memory_token_limit` 等。

```bash
export PYTHONPATH=src
uv run python src/pipeline_generate.py --help
```

### 4. 评测（`src/pipeline_evaluate.py`）

对预测 JSONL 调用 LLM Judge，支持多个 `--input` 在同一进程内评测。

```bash
export PYTHONPATH=src
uv run python src/pipeline_evaluate.py \
  --input experiment/your_run.jsonl \
  --judge_model qwen3-max \
  --benchmark lme_s \
  --write_back
```

`--benchmark` 可省略时由脚本尽量从样本或路径推断。汇总结果默认写入与输入同目录的 **`eval_judge.json`**（标准 JSON 数组；可用 **`--append_result`** 指定路径；若路径以 `.jsonl` 结尾则仍为每行一条 JSON）。需要表格可加 `--csv`。

## 启动模型服务（示例脚本）

仓库内 shell 仅供参考，**路径、GPU、端口需按本机修改**：

| 脚本 | 用途 |
|------|------|
| [`script/0_run_embedding.sh`](script/0_run_embedding.sh) | vLLM `--task embed`（示例） |
| [`script/0_run_model.sh`](script/0_run_model.sh) | vLLM OpenAI 兼容对话服务（示例） |
| [`script/0_run_embedding_ppu.sh`](script/0_run_embedding_ppu.sh) | 另一套 embedding 启动示例 |
| [`script/0_run_model_ppu_*.sh`](script/) | 多模型规模对话服务示例（gemma4、qwen3-4b/8b/32b） |
| [`script/0_run_reranker_ppu.sh`](script/0_run_reranker_ppu.sh) | Qwen3 Reranker（`run_exp.sh` 中 `--rerank-qwen3-vllm` 时需先启动） |

## 项目结构（摘要）

```
src/
  agent/              # 标准答题 Agent
  benchmark/          # LME、LoCoMo、LifeMemBench、EgoMemBench、MEME 等基准与默认数据路径
  memory/
    baselines/        # lme_prebuilt（检索阶段 memory system）
    admission/        # A-MAC 五维准入评分（U/C/N/R/T）
    candidate_ingest/ # relation_decision、add_all、mem0、amac 写库逻辑
    fusion/           # relation_decision 专用：关系包融合（bundle prompt render + LME bundle fusion）
    mem0/             # Mem0 风格增量更新
    zep/              # Graphiti + Kuzu 知识图谱适配
    storage/          # LocalFaiss 向量库封装
    base.py           # BaseMemorySystem 抽象
    tracing.py        # MemoryTraceLogger
  pipeline/           # extract_candidates、ingest_candidates、fuse 等子步骤
  prompts/            # Jinja 模板（抽取 / 关系分类 / 融合 / Judge / Agent 等）
  utils/              # llm_api、embed_utils、qwen3_reranker_vllm、eval_report 等
  pipeline_generate.py
  pipeline_evaluate.py
viewer/               # 实验可视化 HTML 构建脚本（可选）
script/               # run_exp.sh、run_exp.config.yaml、模型启动示例、辅助脚本
test/                 # pytest
experiment/           # 实验输出（jsonl、eval_judge.json 等，按运行生成）
MemDB/                # 向量库与候选缓存（按运行生成）
logs/                 # memory_trace / agent_trace 日志（按运行生成）
```

## 测试

```bash
source .venv/bin/activate
pytest
```
