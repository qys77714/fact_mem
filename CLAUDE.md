# CLAUDE.md

本文件给在此仓库工作的 Claude 提供项目级约定。**回答一律用中文。**

## 项目是什么

`easy-mem` / `fact_mem`：长对话记忆评测框架。在 **LongMemEval (LME)** 基准上，对比
`relation_decision` / `amac` / `zep` / `mem0` / `add_all` / `evermemos` 六种灌库策略。

当前论文主实验：**hybrid 数据集**（golden memory + BM25-dense 混合检索），
同时包含 **fusion 消融实验**（`fusion_enabled: false`，用分类器保留新旧中的一条，不用 LLM 融合）。

## 环境与运行（关键约束）

- 包管理用 **uv**。跑任何脚本用 `uv run --no-sync python ...`（裸 `python` 缺 `openai`/`dotenv` 等依赖；不带 `--no-sync` 的 `uv run` 会重新 sync，可能改坏已验证好的环境）。
- 本机 glibc 2.31 → **vllm ≤ 0.19.1**；gemma4 需 transformers 5.x。
- 模型服务：gemma4-26B 在 **7111**(TP=4, GPU4-7)、qwen3-embedding-0.6b 在 **7110**。`.env` 里 `VLLM_API_KEY=zjj`，端口走 `PORT_GEMMA4_26B` 等变量。
- 模型别名定义于 `src/utils/llm_api.py`：本地 vllm(`gemma4-26B`、`Qwen3-4B`)、云端 `qwen3-max`/`gpt-4o-mini`(judge) 等。
- 密钥从根目录 `.env` 经 dotenv 自动加载（`env | grep` 看不到，需在 python 里 `load_dotenv()`）。

## 代码约定

- 提示模板在 `src/prompts/templates/*.jinja`，用 `from prompts import render_prompt` 渲染。
  loader 用 **`[[ ... ]]`** 作变量分隔符（不是默认 `{{ }}`），但 block 标签仍是默认 `{% ... %}`。
  `FileSystemLoader` 指向 `templates/`，子目录需带前缀访问。
- LLM 客户端：`load_api_chat_completion(model_name, async_=False)`。同步 `get_response_chat(messages, max_new_tokens, temperature, ...)`，
  重试耗尽或内容安全失败返回 `None`（上游需判空）。批量并发可用 `ThreadPoolExecutor` 包同步客户端。
- 实验参数集中在 YAML（`config/exp_N*_*.yaml`），**改抽取策略必须同步换 `candidate_suffix`**，否则复用旧 state 跳过 episode。
- 记忆抽取已统一为单 pass、user 中心模板 `0_mem_extract_aspect_unified_en.jinja`。

## 实验入口

```bash
# 主入口（默认新 artifact 布局，产物在 artifacts/）
uv run --no-sync python run_exp_lme.py [--config config/exp_N0_gemma4-26b_rd_addall.yaml] [--stages extract,ingest,generate,evaluate]

# 只跑部分阶段（例如只换 token limit 时复用已有 ingest）
uv run --no-sync python run_exp_lme.py --config config/tl512.yaml --stages generate,evaluate

# 旧路径续跑（MemDB/ + experiment/）
uv run --no-sync python run_exp_lme.py --config config/old.yaml --legacy-layout
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

## 当前数据集与配置

主实验数据集：`data/preprocessed/longmemeval_s_hybrid_golden.json`（由 `script/build_hybrid_golden_dataset.py` 构建）。

关键 config：
- `config/exp_N{0,2,4,6,8}_{model}_{method}.yaml` — hybrid 主实验配置（N=filler 数量，model=答题模型，method=灌库方法：`rd_addall` / `evm` / `mem0`）
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

数据路径映射定义于 `src/benchmark/datasets.py` 的 `DEFAULT_BENCHMARK_DATASETS`。`benchmark: lme_s` 默认读 `data/preprocessed/longmemeval_s_cleaned_converted.json`。

> **数据获取**：预处理数据集较大（200MB+），未包含在 Git 仓库中。协作者需从项目维护者处获取数据文件（网盘/其他渠道），放置于 `data/preprocessed/` 和 `data/raw_data/` 目录下。

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
| `Qwen3.5-9B` | `Qwen3.5-9B` |
| `qwen3-embedding-0.6b` | embedding 服务（默认端口 7110） |

本地启动脚本（`script/0_run_model.sh`、`script/0_run_embedding.sh`）为单 GPU/单端口示例；多模型部署需手动指定不同端口和 GPU。

API key 统一 `VLLM_API_KEY=zjj`（从 `.env` 加载）。

### add_all / mem0 / evermemos 与 manager 模型无关

`add_all`、`mem0`、`evermemos` 三种灌库方法**只使用 embedding**（`models.embedding`），不调用 LLM。
因此它们与 `models.manager` 无关——同一 filler 级别只需跑一次，不同 manager 模型可共享同一份 ingest。

只有 `relation_decision` 才依赖 `models.manager` 做 LLM 关系分类，换 manager 模型需要重跑 RD ingest。

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
