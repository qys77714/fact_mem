# easy-mem

在长对话记忆评测基准上，统一对比多种记忆策略（RAG、全上下文、Mem0、AMem 等）的实验框架。流水线负责：**加载基准数据 → 写入/检索记忆 → 标准 Agent 答题 → 输出 JSONL**，并可选地用 LLM Judge 打分。

## 环境要求

- **Python** ≥ 3.12  
- 依赖见 [`pyproject.toml`](pyproject.toml)（FAISS、OpenAI 兼容客户端、Transformers、vLLM 等）  
- 推荐使用 [uv](https://github.com/astral-sh/uv) 安装：

```bash
cd easy_mem
uv sync
```

在项目根目录执行后续命令；生成/评测脚本通过 `python src/...` 运行，解释器会把 `src/` 加入模块搜索路径。

## 配置与密钥

支持通过项目根目录的 `.env` 加载环境变量（[`utils.env.load_env`](src/utils/env.py)）。

| 用途 | 变量 | 说明 |
|------|------|------|
| 本地 vLLM 对话模型 | `VLLM_API_KEY`（必填）、`VLLM_BASE_URL`（默认 `http://localhost:8000/v1/`） | 与 [`llm_api.load_api_chat_completion`](src/utils/llm_api.py) 中注册的 served 名称一致 |
| 向量服务（OpenAI 兼容） | `EMBEDDING_BASE_URL`、`EMBEDDING_API_KEY` | 生成流水线里 embedding 调用；也可用 CLI `--embedding_base_url` / `--embedding_api_key` 覆盖 |
| 通义千问等云端模型 | `DASHSCOPE_API_KEY` 等 | 见 `llm_api.py` 中各 provider 分支 |

具体模型别名（如 `Qwen3.5-27B`、`qwen3-max`）以 `src/utils/llm_api.py` 为准。

## 数据

预处理后的 JSON 需放在 `data/preprocessed/`。[`pipeline_generate.py`](src/pipeline_generate.py) 内置的 `benchmark` 名称与默认文件对应关系包括：

| `--benchmark` | 默认数据文件 | 语言 |
|---------------|----------------|------|
| `test` | `data/preprocessed/test.json` | zh |
| `lme_oracle` | `data/preprocessed/longmemeval_oracle_converted.json` | en |
| `lme_s` | `data/preprocessed/longmemeval_s_cleaned.json` | en |
| `locomo` | `data/preprocessed/locomo10_converted.json` | en |
| `lmb_event` | `data/preprocessed/LifeMemBench_event.json` | zh |
| `emb_event` | `data/preprocessed/EgoMemBench_event_half.json` | en |

也可用 `--benchmark_file` 指定任意兼容格式的 JSON 文件；语言可用 `--language zh|en` 覆盖。

## 记忆方法（`--method`）

[`memory.get_memory_system`](src/memory/__init__.py) 支持：

- **`rag`** — 按粒度切分后向量检索  
- **`full_context`** — 全历史上下文  
- **`only_query`** — 仅当前问题，无记忆  
- **`mem0` / `amem`** — 需 `--manager_model`，由 LLM 管理记忆并配合向量库  

向量库存储根目录可由 `--database_root` 指定；默认按 benchmark 与方法名自动组织。`--rebuild-memory` 可强制清空并重灌。

## 运行流程

### 1. 启动模型服务（示例）

仓库内脚本仅为参考，路径与 GPU 需按本机修改：

- **Embedding**：[`script/0_run_embedding.sh`](script/0_run_embedding.sh) — vLLM `--task embed`  
- **对话模型**：[`script/0_run_model.sh`](script/0_run_model.sh) — vLLM serve  

### 2. 生成预测（`pipeline_generate.py`）

示例见 [`script/1_generate.sh`](script/1_generate.sh)。核心参数：

- `--benchmark`、`--output`  
- `--method`、`--answer_model`  
- `mem0`/`amem`：`--manager_model`  
- `--embedding_model`、`--embedding_base_url`、`--embedding_api_key`  
- `--retrieve_topk`、`--memory_token_limit`、`--memory_granularity`（`all` 或正整数，表示每 N 轮一组）  
- `--parallel_episodes` 并行 episode 数；`--agent_trace_dir` 可传空字符串关闭 trace  

```bash
uv run python src/pipeline_generate.py --help
```

### 3. 评测（`pipeline_evaluate.py`）

对 JSONL 用 LLM Judge 判对错，示例见 [`script/2_evaluate.sh`](script/2_evaluate.sh)：

```bash
uv run python src/pipeline_evaluate.py \
  --input experiment/your_run.jsonl \
  --judge_model qwen3-max \
  --benchmark lme \
  --write_back
```

`--benchmark` 可省略，脚本会尽量从样本或文件名推断（`lme` / `lmb` / `emb` / `locomo`）。

## 项目结构（摘要）

```
src/
  agent/           # 标准答题 Agent
  benchmark/       # LME、LoCoMo、事件类基准加载
  memory/          # 记忆系统与本地 FAISS 存储
  prompts/         # Judge 等模板
  pipeline_generate.py
  pipeline_evaluate.py
script/            # 启动模型与批处理示例
test/              # pytest
```

