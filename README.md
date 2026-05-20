# easy-mem

在长对话记忆评测基准上，对比 **LME 候选记忆抽取 → 灌库 → 混合检索答题** 等流程的实验框架。完整实验通常由 **[`script/run_exp.sh`](script/run_exp.sh)** 编排（候选抽取 → 灌库 → 生成 → LLM Judge；脚本头部注明了可选 HTML 对照等步骤）。

手工单步跑时，核心仍是：**加载基准数据 → 写入/检索记忆 → 标准 Agent 答题 → 输出 JSONL**，并可选用 LLM Judge 写回评分。

## 环境要求

- **Python** ≥ 3.12  
- 依赖见 [`pyproject.toml`](pyproject.toml)（FAISS、OpenAI 兼容客户端、Transformers、Jinja2、**Graphiti + Kuzu**（Zep 灌库路径）、pytest 等）。对话/向量 **vLLM 服务在运行时按需单独部署**，不必写进 `pyproject.toml`。  
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

**变体：** 若灌库已完成，只需在已有 Zep 库上生成 + Judge，可用 **[`script/run_exp_zep.sh`](script/run_exp_zep.sh)**（支持 `RUN_EXP_ZEP_SKIP_GENERATE=1` 等，见脚本头注释）。

## 配置与密钥

支持通过项目根目录的 `.env` 加载环境变量（[`utils.env.load_env`](src/utils/env.py)）。

| 用途 | 变量 | 说明 |
|------|------|------|
| 本地 vLLM 对话模型 | `VLLM_API_KEY`（必填）、`VLLM_BASE_URL`（默认 `http://localhost:8000/v1/`） | 与 [`llm_api.load_api_chat_completion`](src/utils/llm_api.py) 中注册的 served 名称一致 |
| 向量服务（OpenAI 兼容） | `EMBEDDING_BASE_URL`、`EMBEDDING_API_KEY` | 生成流水线里 embedding 调用；也可用 CLI `--embedding_base_url` / `--embedding_api_key` 覆盖 |
| 通义千问等云端模型 | `DASHSCOPE_API_KEY` 等 | 见 `llm_api.py` 中各 provider 分支 |

具体模型别名（如 `gemma4-26B`、`qwen3-max`）以 [`src/utils/llm_api.py`](src/utils/llm_api.py) 为准。

## 数据

预处理/原始数据按 [`src/benchmark/datasets.py`](src/benchmark/datasets.py) 中 **`DEFAULT_BENCHMARK_DATASETS`** 解析。内置 `--benchmark` 与默认文件如下（也可用 `--benchmark_file` 指定任意兼容 JSON；语言可用 `--language zh|en` 覆盖默认值）。

| `--benchmark` | 默认数据文件 | 默认语言 |
|-----------------|--------------|----------|
| `test` | `data/preprocessed/test.json` | zh |
| `lme_o` | `data/preprocessed/longmemeval_oracle_converted.json` | en |
| `lme_s` | `data/preprocessed/longmemeval_s_cleaned_converted.json` | en |
| `lme_m` | `data/preprocessed/longmemeval_m_cleaned_converted.json` | en |
| `locomo` | `data/raw_data/locomo10.json` | en |
| `lmb_event` | `data/preprocessed/LifeMemBench_event.json` | zh |
| `emb_event` | `data/preprocessed/EgoMemBench_event_half.json` | en |

## 记忆方法与生成（`pipeline_generate.py`）

答题阶段当前通过 [`memory.get_memory_system`](src/memory/__init__.py) 仅注册 **`lme_prebuilt`**：在 **`ingest_candidates.py`** 等步骤预先写入向量与元数据后，由 **`pipeline_generate.py`** 以 **`--method lme_prebuilt --prebuilt-memory`**（及混合检索相关 flags）从 `--database_root` 读取。

向量库存储路径由 `--database_root` 等参数决定；`run_exp.sh` 中与灌库脚本约定目录布局。详见 `pipeline_generate.py --help`。

## 启动模型服务（示例脚本）

仓库内 shell 仅供参考，**路径、GPU、端口需按本机修改**：

| 脚本 | 用途 |
|------|------|
| [`script/0_run_embedding.sh`](script/0_run_embedding.sh) | vLLM `--task embed`（示例） |
| [`script/0_run_model.sh`](script/0_run_model.sh) | vLLM OpenAI 兼容对话服务（示例） |
| [`script/0_run_embedding_ppu.sh`](script/0_run_embedding_ppu.sh) | 另一套 embedding 启动示例 |
| [`script/0_run_model_ppu_*.sh`](script/) | 多模型规模对话服务示例 |
| [`script/0_run_reranker_ppu.sh`](script/0_run_reranker_ppu.sh) | Qwen3 Reranker（`run_exp.sh` 中 `--rerank-qwen3-vllm` 时需先启动） |

## 手工调用（不跑 `run_exp.sh` 时）

### 生成（`src/pipeline_generate.py`）

常用参数包括：`--benchmark`、`--output`、`--method lme_prebuilt`、`--prebuilt-memory`、`--database_root`、`--answer_model`、`--embedding_model`、混合检索与 `--retrieve_topk`、`--memory_token_limit` 等。

```bash
export PYTHONPATH=src
uv run python src/pipeline_generate.py --help
```

### 评测（`src/pipeline_evaluate.py`）

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

### 候选抽取与灌库（与 `run_exp.sh` 一致的子步骤）

- [`src/pipeline/extract_candidates.py`](src/pipeline/extract_candidates.py) — 候选记忆抽取  
- [`src/pipeline/ingest_candidates.py`](src/pipeline/ingest_candidates.py) — 多种 `--update-method`（如 `zep`；见 CLI `--help`）  
- [`src/pipeline/fuse_lme_memory_bundles.py`](src/pipeline/fuse_lme_memory_bundles.py) — 关系包融合（完整流水线里部分步骤可按需启用）

## 项目结构（摘要）

```
src/
  agent/              # 标准答题 Agent
  benchmark/          # LME、LoCoMo、事件类基准与默认数据路径
  memory/             # 记忆抽象与 lme_prebuilt 实现（baselines/ 等）
  pipeline/           # extract_candidates、ingest_candidates、fuse 等
  prompts/            # Jinja 模板（抽取 / Judge 等）
  utils/              # llm_api、env、评测汇总等
  pipeline_generate.py
  pipeline_evaluate.py
viewer/               # 实验可视化 HTML 构建脚本（可选）
script/               # run_exp.sh、模型启动示例、辅助脚本
test/                 # pytest
experiment/           # 实验输出（jsonl、eval_judge.json 等，按运行生成）
```

## 测试

```bash
source .venv/bin/activate
pytest
```
