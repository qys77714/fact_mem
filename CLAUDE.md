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
- 实验参数集中在 YAML（`config/lme.yaml`），**改抽取策略必须同步换 `candidate_suffix`**，否则复用旧 state 跳过 episode。
- 记忆抽取已统一为单 pass、user 中心模板 `0_mem_extract_aspect_unified_en.jinja`。

## 实验入口

```bash
# 主入口
uv run --no-sync python run_exp_lme.py [--config config/lme_hybrid_filler_N0_tl256.yaml] [--stages extract,ingest,generate,evaluate]

# 只跑部分阶段
uv run --no-sync python run_exp_lme.py --stages ingest,generate,evaluate
```

四个阶段：
- **extract** — 候选记忆抽取（单 pass 统一模板）
- **ingest** — 每个 enabled 方法写入向量库；`relation_decision` 在灌库时就地融合（不再有独立 fuse 阶段）
- **generate** — 预建库检索 → Agent 答题 → 输出 JSONL
- **evaluate** — LLM Judge 写回评分

配置中 `methods` 下同时开多个 `enabled: true` = 比较实验，ingest 和 generate 依次执行。

## 当前数据集与配置

主实验数据集：`data/preprocessed/longmemeval_s_hybrid_golden.json`（由 `script/build_hybrid_golden_dataset.py` 构建）。

关键 config：
- `config/lme.yaml` — 基础模板（带注释说明各节含义）
- `config/lme_hybrid_filler_N{0,2,4,6,8}_tl256*.yaml` — hybrid 主实验配置
- `config/lme_golden_only.yaml` — golden only 基线
- `config/lme_lowered_golden_only.yaml` — lowered golden 基线

数据路径映射定义于 `src/benchmark/datasets.py` 的 `DEFAULT_BENCHMARK_DATASETS`。`benchmark: lme_s` 默认读 `data/preprocessed/longmemeval_s_cleaned_converted.json`。

## relation_decision 灌库策略

核心流程（`src/memory/candidate_ingest/relation_decision.py`）：
1. 召回 top-k 相关旧记忆
2. 逐对关系分类（五类：IND/EQV/OSN/NSO/CON）
3. 按关系类型写入/替换/融合，构建关系图

### 分类后端

`relation_decision.backend` 可选：
- `classifier`（默认）：用 `relation_classifier/` 中的冻结 Qwen3-0.6B + 线性探测头，test Macro F1 ≈ 0.88。**必须英文输入**，顺序 `(old, new)` 不可颠倒。
- `llm`：用 gemma4-26B 做关系分类，通过 `lme_relation_classification_system_en_v3.jinja` 模板。

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

- **直接调 pipeline 脚本**（不用 `run_exp_lme`）需设 `PYTHONPATH=src`，否则 `ModuleNotFoundError: No module named 'benchmark'`。
- **改抽取策略必须同步换 `candidate_suffix`**，否则 extract 阶段复用旧 state 跳过 episode。
- **trace 目录不带实验 suffix**，多次运行混写。如需分档统计，需改 trace 路径加 suffix 后重跑。
- **relation_classifier 输入必须英文**，且 `(old, new)` 顺序不可颠倒（颠倒会让 OSN/NSO 互换）。backbone 默认路径 `/mnt/data_oss/models/Qwen3-0.6B`，可通过 `RC_BACKBONE_PATH` 环境变量覆盖。
- **relation_decision 灌库时就地融合**答题记忆 C（同库），不再有独立的事后 fuse 阶段。`run_exp_lme.py` 中 `stage_ingest` 对 `relation_decision` 不额外调用 fuse 脚本。

## 项目结构

```
run_exp_lme.py              # 主入口
config/                     # 实验 YAML 配置
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
  pipeline_lme_evaluate.py  # LLM Judge
  prompts/                  # Jinja 模板
  utils/                    # llm_api、embed_utils、eval_report、config 等
relation_classifier/        # 五分类器推理包（Qwen3-0.6B + 线性探测头）
script/
  0_run_model.sh            # 模型服务启动
  0_run_embedding.sh        # embedding 服务启动
  build_hybrid_golden_dataset.py   # hybrid 数据集构建
  build_lme_golden_memory_v2.py    # golden memory 生成
  build_unified_candidates.py      # 候选记忆抽取
  build_unified_filler.py          # filler 候选构建
```

## 测试

```bash
uv run --no-sync pytest
```
