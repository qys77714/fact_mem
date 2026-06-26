# LME 混淆数据集（golden + lowered golden + distractor）设计

日期：2026/06/26
状态：设计已确认，待实现

## 1. 目标与动机

为论文主实验重造一个 LongMemEval (LME) 数据集，使其同时包含：
- 原始 LME 对话数据（不改）；
- 每题的 **golden memory**（干净 ground-truth）；
- **lowered golden**（与 golden 一一对应、仍能推出答案、但与 query 相似度更低）；
- 一组 **distractor（混淆 memory）**，每条与 query 的相似度都高于「该题最小 lowered golden 相似度」，但不泄露答案。

动机：现有 Phase3 `dup_items.json` 的混淆变体只在「种子」层面保证了 `sim_q > 最小 golden`，未对每条变体逐条过滤——全量 451 题中仅 **43.5%** 满足「8 条混淆项全部达标」，且 lowered/distractor 文本主语混用 "You"/"They"。本数据集逐条强制约束、统一主语，产出一个口径干净、可直接支撑主实验的数据集。

本次范围**仅交付数据集与生成脚本**，不接入实验跑批管线。

## 2. 数据集结构

单 JSON 文件存 `list`，每题一条记录。字段如下：

```jsonc
{
  // A. 原始 LME 字段（原封不动，provenance）
  "question_id":          "e47becba",
  "question_type":        "single-session-user",
  "question":             "What degree did I graduate with?",
  "question_date":        "2023/05/30 (Tue) 23:40",
  "answer":               "Business Administration",
  "answer_session_ids":   [...],
  "haystack_dates":       [...],
  "haystack_session_ids": [...],
  "haystack_sessions":    [...],

  // B. 三组记忆（新增）
  "golden_memory": [   // 干净 ground-truth，仅参考/校验，不灌库
    {"text": "The user graduated with a degree in Business Administration.", "sim_q": 0.83, "date": "2023/05/26 (...)"}
  ],
  "lowered_golden": [  // 数量 == golden；保答案；sim_q 尽量低；与 golden 按 source_idx 对应
    {"text": "The user wrapped up their studies in business management a while back.",
     "sim_q": 0.71, "source_idx": 0, "date": "2023/05/26 (...)"}
  ],
  "distractors": [     // 固定 8 条；同话题、非答案；每条 sim_q > lowered_golden_min_sim
    {"text": "The user is considering going back for a degree in Computer Science.", "sim_q": 0.78, "date": "..."},
    {"text": "...", "sim_q": 0.80, "date": "..."}
    // 共 8 条
  ],

  // C. 元信息（可审计/可复现）
  "embedding_model": "qwen3-embedding-0.6b",
  "lowered_golden_min_sim": 0.71,  // 约束基准：所有 distractor.sim_q 都 > 此值
  "constraint_ok": true            // 自检：lowered 全部保答案 且 8/8 distractor 达标
}
```

**字段约定**
- `sim_q`：该条记忆文本与 `question` 的归一化 embedding 余弦相似度（qwen3-embedding-0.6b）。
- `lowered_golden.source_idx`：指向对应的 `golden_memory` 下标，保证一一对应。
- `lowered_golden_min_sim = min(lowered_golden[*].sim_q)`，是 distractor 的约束基准线。
- `date`：用于实验灌库顺序（distractor 较早、lowered golden 用 golden 日期）；纯 Dup 不涉冲突，日期只影响展示/灌库顺序，不影响答案正确性。

**主语约束（硬性）**：`golden_memory` / `lowered_golden` / `distractors` 三组文本**全部用第三人称、主语统一为 "The user"**，对齐 golden_memory 既有风格。生成 prompt 必须显式要求，并在校验阶段抽查。

## 3. 灌库语义（仅记录约定，本次不实现）

实验时进入检索库的记忆集合 = `lowered_golden` + 前 N 条 `distractors`（N ∈ {0,1,2,4,8}）。`golden_memory` 不进库，仅作 ground-truth 参考与构造期校验源。这保证 distractor 的 sim_q 稳定高于被灌入的正确记忆（lowered golden），污染挑战成立。

## 4. 构造流程

**数据源 / 范围**
- 锚点：`data/preprocessed/longmemeval_s_golden.json` 的 **470 可答题**（排除 30 道 abstention，它们 `golden_memory=[]`）。
- `golden_memory` 直接取自该文件。
- 原始 LME 对话字段取自 `data/raw_data/longmemeval_s_cleaned.json`，按 `question_id` 对齐拷入。

**模型服务**
- 生成：`gemma4-26B`（别名→`gemma-4-26B-A4B-it`，端口 7111，`script/0_run_model.sh`）。
- embedding：`qwen3-embedding-0.6b`（端口 7110，`script/0_run_embedding.sh`）。
- 调用：`load_api_chat_completion("gemma4-26B")` 同步客户端；embedding 用 `utils.embed_utils.embed_texts` + OpenAI client（`EMBEDDING_BASE_URL`）。并发用 `ThreadPoolExecutor` 包同步客户端。

**Step 1 — lowered golden**（复用 Phase3 `lower_golden_casual` 逻辑）
- 对每条 golden 做多策略口语化/泛化改写；LLM 校验「答案等价」（仍能推出正确答案）。
- 保留**仍含答案且 sim_q 最低**的版本；数量与 golden 一一对应，记 `source_idx`。
- prompt 加主语约束（"The user ..."）。
- 计算 `lowered_golden_min_sim`。

**Step 2 — distractor（严格保证 8 条达标）**
- prompt 生成同话题、非答案的干扰句（语义靠近 query，但给出不同/错误取值；主语 "The user"）。
- **逐条**两道闸：
  1. `sim_q > lowered_golden_min_sim`（逐条检查，非仅种子）；
  2. LLM 校验「不泄露答案」（复用 `verify_seed_answer`：distractor 不含正确答案信息）。
- **生成-过滤循环**：累计达标到 8 条即停；每题最多 K 轮（默认 K=6），每轮可批量生成多候选以提速。
- 凑齐 8 条 → `constraint_ok=true`，进主数据集；凑不齐 → `constraint_ok=false`，进 `_partial.json`。

**Step 3 — 装配与落盘**
- 填 `date`：lowered golden 用 golden 原日期；distractor 用早于 golden 的日期。
- 写 `embedding_model`、`lowered_golden_min_sim`、`constraint_ok`。
- 拷入原始 LME 对话字段。
- 输出三个文件（见 §5）。

## 5. 产物

| 文件 | 内容 |
|---|---|
| `data/preprocessed/longmemeval_s_confusion.json` | 主数据集：`constraint_ok=true`（lowered 全保答案 + 8/8 distractor 达标）的题 |
| `data/preprocessed/longmemeval_s_confusion_partial.json` | 凑不齐 8 条或 lowered 校验失败的题（备查，不进主实验） |
| `data/preprocessed/confusion_build_stats.json` | 统计：入选题数、各 `question_type` 分布、distractor 平均尝试轮数、排除原因计数、sim_q 分布摘要 |

脚本：`script/build_confusion_dataset.py`（Step 1–3 一条龙，支持 `--limit N` 调试、断点续跑跳过已完成题）。

## 6. 验收标准

1. 主数据集每题：`len(lowered_golden) == len(golden_memory)`；`len(distractors) == 8`；每条 `distractor.sim_q > lowered_golden_min_sim`；`constraint_ok == true`。
2. 三组记忆文本主语均为第三人称 "The user"（抽样人工核验 + 简单正则统计代词比例）。
3. 每条 lowered golden 通过「答案等价」校验；每条 distractor 通过「不泄露答案」校验。
4. 原始 LME 对话字段与 `longmemeval_s_cleaned.json` 按 `question_id` 完全一致。
5. `confusion_build_stats.json` 如实报告最终入选题数（不做静默截断）。

## 7. 非目标（本次不做）

- 不接入实验：候选 chunk 拼装、config 生成、`run_exp_lme` 跑批、聚合曲线——均沿用现有 Phase3 管线，后续单独进行。
- 不含冲突型（CON）distractor：本数据集 distractor 仅为「同话题非答案」型。
- 不改动 `relation_classifier` / 抽取模板等无关代码。

## 8. 风险与说明

- **样本量未知**：能凑齐严格 8 条 distractor 的题数取决于生成成功率，乐观估 350–430 题，确切值跑完在 stats 中如实报告。
- **绑定 embedding 模型**：`sim_q` 与约束判定均依赖 qwen3-embedding-0.6b；换模型需重算，故 `embedding_model` 入库。
- **lowered golden 是可被 reviewer 质疑点**：主动降低正确记忆 sim_q。论文需将其定位为「模拟口语化/泛化表述使正确记忆显著性下降」的合理设定，并报告平均降幅。
