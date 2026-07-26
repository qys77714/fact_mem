# 协作者实验指南

本文档面向从零开始搭建实验环境的协作者。读完本文后，你应该能独立跑通 gemma4-12b × LME Hybrid 全实验。

## 你需要什么

### 硬件

| 资源 | 最低要求 |
|------|---------|
| GPU | 2×24GB（gemma4-12b-it）+ 4×GPU（gemma4-26B）+ 1×GPU（embedding） |
| 磁盘 | ~100GB（数据集 3GB + 候选记忆 41MB + ingest 产物 ~66GB + 日志） |
| 网络 | 能访问 DeepSeek API（Judge 用） |

如果没有 GPU，可以把 `extract`/`manager`/`answer` 模型改为云端 API（见下文「纯云端方案」）。

### 软件

- Python ≥ 3.12
- **uv** 包管理器
- vllm ≥ 0.19.1（如需本地部署模型）

### 从维护者获取

| 文件 | 获取方式 | 放置路径 |
|------|---------|---------|
| 数据集 | `wget https://huggingface.co/datasets/Qys77/easy-mem-data/resolve/main/easy-mem-data.zip` | 解压到项目根目录 → `data/` |
| 候选记忆 | `wget https://huggingface.co/datasets/Qys77/easy-mem-data/resolve/main/easy-mem-candidates.zip` | 解压到项目根目录 → `artifacts/stages/candidates/` |

## 快速开始

### 1. 克隆仓库 + 安装依赖

```bash
git clone git@github.com:qys77714/fact_mem.git
cd fact_mem
uv sync
```

### 2. 下载并解压数据

```bash
# 数据集（HuggingFace）
wget https://huggingface.co/datasets/Qys77/easy-mem-data/resolve/main/easy-mem-data.zip
unzip easy-mem-data.zip -d .

# 候选记忆
wget https://huggingface.co/datasets/Qys77/easy-mem-data/resolve/main/easy-mem-candidates.zip
unzip easy-mem-candidates.zip -d .
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填写以下内容：
```

`.env` 最小配置：

```bash
# 本地 VLLM
VLLM_API_KEY=zjj
PORT_GEMMA4_12B=7112          # gemma4-12b-it 端口
PORT_GEMMA4_26B=7111          # gemma4-26B 端口

# Embedding
EMBEDDING_BASE_URL=http://localhost:7110/v1/
EMBEDDING_API_KEY=zjj

# Judge（DeepSeek 云端 API）
DEEPSEEK_API_KEY=your_deepseek_key_here
```

### 4. 启动模型服务

按顺序启动三个模型：

```bash
# 1) Embedding（必须最先）
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/Qwen3-Embedding-0.6B \
    --task embed --port 7110 --api-key zjj &

# 2) gemma4-12b-it（ingest 管理用，2×GPU）
CUDA_VISIBLE_DEVICES=1,2 vllm serve /path/to/gemma-4-12B-it \
    --port 7112 --tensor-parallel-size 2 --api-key zjj &

# 3) gemma4-26B（answer 用，4×GPU）
CUDA_VISIBLE_DEVICES=3,4,5,6 vllm serve /path/to/gemma-4-26B-A4B-it \
    --port 7111 --tensor-parallel-size 4 --api-key zjj &
```

### 5. 一键启动实验

```bash
bash script/run_collab_exp.sh
```

## 实验流程详解

脚本 `run_collab_exp.sh` 按以下顺序执行：

```
生成配置（15 个 YAML）
  │
  ├─ [2/5] RD ingest ─── N=0 → N=2 → N=4 → N=6 → N=8 （串行，共用 gemma4-12b）
  │
  ├─ [3/5] Mem0 ingest ─ N=0,2,4,6,8 同时跑（并行，共用 gemma4-12b）
  │
  ├─ [4/5] EVM ingest ── N=0,2,4,6,8 同时跑（并行，共用 gemma4-12b）
  │
  └─ [5/5] Answer+Judge ─ 15 组同时跑（answer: gemma4-26B, judge: deepseek-v4-flash）
```

**为什么 RD 串行、Mem0/EVM 并行？**
- RD (relation_decision) 每对候选记忆都要调 LLM 做关系分类，调用量大，串行避免 vllm 过载。
- Mem0 和 EverMemOS 的 LLM 调用频率较低，5 个 N 级别并行通常不会压垮 vllm。
- 如果并行时 vllm OOM，降低 config 中 `parallel.ingest_episode_concurrency.{mem0, evermemos}` 的值。

### 实验矩阵

| 维度 | 值 |
|------|-----|
| Benchmark | LME Hybrid（`lme_s`） |
| Filler 等级 | N0, N2, N4, N6, N8（golden + N 条 distractor） |
| 灌库方法 | RD (relation_decision), Mem0, EverMemOS |
| 管理模型 | gemma4-12b-it |
| 答题模型 | gemma4-26B-A4B-it |
| Judge 模型 | deepseek-v4-flash（云端） |
| Token limit | 256 |

**共 3 方法 × 5 filler = 15 组实验。**

## 结果查看

实验完成后，结果在 `artifacts/runs/` 下。每组实验有独立的 `metrics.json`：

```bash
# 查看某组实验的指标
cat artifacts/runs/<run_id>/judge/<method>/<judge_id>/metrics.json
```

配置文件在 `config/collab/` 下，可以手动修改后重跑部分阶段：

```bash
# 只重跑 answer + judge（不重跑 ingest）
uv run --no-sync python run_exp_lme.py \
    --config config/collab/exp_N0_rd.yaml \
    --stages generate,evaluate
```

## 常见问题

### 纯云端方案（无 GPU）

如果没有 GPU，将所有模型改为云端 API。复制一份 config 修改 `models` 节：

```yaml
models:
  extract: deepseek-v4-flash    # 记忆抽取走 DeepSeek
  manager: deepseek-v4-flash    # 关系分类走 DeepSeek
  answer: deepseek-v4-flash     # 答题走 DeepSeek
  judge: deepseek-v4-flash      # Judge 走 DeepSeek
  embedding: qwen3-embedding-0.6b  # embedding 仍需本地（仅需 1 GPU 或改用云端 embedding）
```

### 模型挂载不上

1. 确认模型权重路径正确
2. 确认 `.env` 中 `PORT_*` 与 vllm `--port` 一致
3. 测试连通：`curl http://localhost:7112/v1/models`（返回模型列表即正常）

### ingest 阶段报指纹不匹配

说明 candidate 数据放的路径不对。检查 `artifacts/stages/candidates/` 下有 6 个 hash 目录，每个 470 个 JSON 文件。
