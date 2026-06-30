# Confusion Memory v3 生成方案

日期：2026/06/30

## 背景

`longmemeval_s_golden.json` 已更新（500 条，470 可答 + 30 abstention）。`golden_memory` 现在是 `[{"content": "...", "date": "..."}]` 格式的字典列表，每条包含记忆文本和对应的 session 日期。需要重新设计 lowering 和混淆记忆生成流程。

与 v2 的核心差异：
- lowering 从"整题过/不过"改为**逐条 lowering + 逐条 fallback**（不连坐）
- 闸门阈值统一为 `anchor_sim - 0.1`
- 种子生成改为多策略多 prompt
- lowering 无法降的题不再算失败，而是回退使用原始 golden

---

## 数据源

| 文件 | 用途 |
|------|------|
| `data/preprocessed/longmemeval_s_golden.json` | Golden memory（500 条：470 可答 + 30 abstention） |
| `data/raw_data/longmemeval_s_cleaned.json` | 原始 LME 数据，取 `question_date` 和 session 信息 |

### golden_memory 字段结构（新）

```json
{
  "question_id": "e47becba",
  "question": "What degree did I graduate with?",
  "answer": "Business Administration",
  "question_type": "single-session-user",
  "evidence_session_ids": ["answer_280352e9"],
  "abstention": false,
  "golden_memory": [
    {"content": "The user graduated with a degree in Business Administration.", "date": "2023/05/28"}
  ],
  "judged_correct": true
}
```

- `golden_memory`：字符串列表，每条为一句话记忆
- `abstention=true`：30 条，跳过不处理
- `judged_correct=true`：459 条，走完整 lowering + distractor 流程
- `judged_correct=false`：11 条，跳过 lowering，直接用 golden 做正确记忆，仅生成 distractor

---

## 整体流程

```
每题（abstention=false）:
  Step 0: 加载 golden memory + 原始 LME 数据，取 question_date，计算每条 golden 的 sim_q
  Step 1: 逐条 lowering golden（仅 judged_correct=true 的题）
          → 逐条 fallback，产出 mixed 正确记忆集合
  Step 2: 多策略种子生成（4 种 prompt，不按题型区分）
          → 过闸1(sim_q > anchor_sim - 0.1) + 闸2(IDK)
  Step 3: EQV/OSN/NSO 改写扩充 → 过闸 → 取前 8 distractor
  Step 4: 装配落盘
```

### judged_correct=false 的特殊处理

跳过 Step 1（lowering），`lowering_status = "full_fallback"`，`lowered_golden = []`，直接用 `golden_memory` 做正确记忆。Step 2-3 照常。

---

## Step 1 — Lowered Golden

### 粒度：逐条 lowering + 逐条 fallback

对每题每条 golden `g_i`：

```
1. 根据题型选择对应的 lowering prompt（内含保留约束）
2. 跑满 8 次尝试生成 lowered 候选
3. 计算各候选 sim_q，筛出在 [sim(g_i)-0.4, sim(g_i)-0.05] 区间内的
4. 区间内按 sim 从低到高排序 → 依次送替换验证
5. 替换验证:
   - 用 {g_0, ..., candidate_i, ..., g_n} 完整集合构建答题 prompt
   - gemma4-26B 答题 → gemma4-26B judge 比对 gold answer
   - 第一个通过验证的候选即接受为 lowered
   - 所有候选都失败 → 继续步骤 6
6. 全不通过或全不在区间 → 该条回退到原始 g_i
```

### 判定

- 逐条独立判定，互不影响（不连坐）
- `n_lowered = 接受的 l_i 数量`
- `n_lowered > 0` → `lowering_status = "partial"`，lowered_golden 含所有通过验证的 lowered 条
- `n_lowered = 0` → `lowering_status = "full_fallback"`，lowered_golden = []，直接用 golden_memory

### 题型独立的 lowering prompt

每种题型注入不同的保留约束：

| 题型 | 必须保留 | 可以改 |
|------|---------|--------|
| temporal-reasoning | 日期、时间、时序关系词（before/after/until） | 句式结构、修饰语、非时间性描述 |
| multi-session | 事件事实、数字、日期、会话间先后关系 | 措辞、句式、过渡描述 |
| knowledge-update | 最新/最终的事实值 | 旧值描述方式、句式 |
| single-session-user | 人名、地名、数字、实体名 | 句式、功能词、修饰副词 |
| single-session-assistant | 关键实体和事实值 | 句式、连接词 |
| single-session-preference | 偏好对象（喜欢/不喜欢的具体东西） | 情感程度修饰词、句式 |

温度 0.6–0.7（temporal 可略低），每条最多 8 次尝试。

### 替换验证

- **答题模板**：`agent_prompt_en_open.jinja` 或 `agent_prompt_en_mcq.jinja`（根据题型）
- **上下文模板**：`lme_memory_context_unit_en.jinja`
- **Judge 模板**：`pipeline_eval_oqa.jinja` 或 `pipeline_eval_mcq.jinja`，system 用 `pipeline_eval_system.jinja`
- **模型**：全部 gemma4-26B

---

## Step 2 — 种子生成

### 核心约束

- 不提供正确答案（那是 golden 的事）
- 不提供错误答案（不制造 misinformation）
- 复用 query 关键词提高相似度
- LLM 单独凭该记忆**无法**回答问题（IDK 闸门）

### 四种策略

| 策略 | 做法 | 示例（query: "What degree did I graduate with?" 关键词: degree, graduate） |
|------|------|------|
| `keyword-borrow` | 把 query 关键词嵌入一个无关的事实陈述 | "The user mentioned that having a degree is important for career advancement." |
| `topic-drift` | 从 query 主题滑向相邻但不包含答案的话题 | "The user discussed their graduation ceremony and the weather that day." |
| `generalize` | 把 query 的具体提问点泛化成一般性背景陈述 | "The user has been thinking about their educational background recently." |
| `context-surround` | 描述与 query 相关的上下文情景但不触及核心事实 | "The user talked about their university experience during the conversation." |

### 参数

- 每种策略独立 prompt，各自有 temperature 和 few-shot
- 不按题型区分
- 每轮每策略生成 **6 条候选**（与 v2 一致），4 策略并行 = 24 条/轮
- 目标：4 条种子，最多 8 轮
- 模型：gemma4-26B

### 闸门

- **闸1**：`sim_q > anchor_sim - 0.1`（`anchor_sim = min(所有正确记忆的 sim_q)`）
- **闸2**：IDK 测试（LLM 仅凭该记忆不可回答问题，必须输出 "I DON'T KNOW"）

---

## Step 3 — 改写扩充

### 策略（与 v2 一致）

| 策略 | 操作 | 约束 |
|------|------|------|
| EQV | 同义改写，换句式/语序 | 只动功能词（the/a/of）和普通动词，不动领域术语 |
| OSN | 加细节使其更具体，strict entailment | 加时间/频率/方式等修饰，不动关键词，不加答案信息 |
| NSO | 删次要修饰词使其更泛化，strict entailment | 只删形容词/副词/时间修饰，不动关键词 |

### 流程

- 改写产物同样过闸1 + 闸2，通过才保留
- 目标：种子 + 改写 ≥ 8 条，去重后按 sim_q 降序取前 8

---

## Step 4 — 输出格式

### 产物

| 文件 | 说明 |
|------|------|
| `longmemeval_s_confusion_v3.json` | 主数据集（constraint_ok=true） |
| `longmemeval_s_confusion_v3_partial.json` | 失败题（如 distractor 不足 8 条） |
| `confusion_v3_build_stats.json` | 构建统计 |

### 字段结构

```json
{
  "question_id": "e47becba",
  "question": "What degree did I graduate with?",
  "answer": "Business Administration",
  "question_type": "single-session-user",
  "question_date": "2023/05/28 (Thu) 20:00",
  "golden_memory": [
    {"text": "The user graduated with a degree in Business Administration.", "sim_q": 0.72, "date": "2023/05/28"}
  ],
  "lowering_status": "partial",
  "lowering_details": [
    {"golden_idx": 0, "success": true, "original_sim": 0.72, "lowered_sim": 0.63, "attempts": 3}
  ],
  "lowered_golden": [
    {"text": "...lowered version...", "sim_q": 0.63, "source_idx": 0, "date": "2023/05/28"}
  ],
  "anchor_sim": 0.63,
  "distractors": [
    {"text": "...", "sim_q": 0.68, "source": "keyword-borrow", "source_idx": 0, "date": "2023/05/27"}
  ],
  "embedding_model": "qwen3-embedding-0.6b",
  "constraint_ok": true
}
```

### 新增/变更字段

| 字段 | 说明 | 替代 v2 的什么 |
|------|------|---------------|
| `lowering_status` | `"partial"` \| `"full_fallback"` | 新字段 |
| `lowering_details` | 逐条 lowering 记录 `{golden_idx, success, original_sim, lowered_sim, attempts}` | 新字段 |
| `anchor_sim` | `min(所有正确记忆的 sim_q)`，闸门 = `anchor_sim - 0.1` | 替代 `lowered_golden_min_sim` |
| `golden_memory[].sim_q` | 构建时计算 | v2 已有 |
| `golden_memory[].date` | 预计算：通过 session 内容匹配获取，写回 golden JSON 文件 | v2 已有 |

---

## 日期处理

### Golden Memory date

预先通过 session 内容匹配为每条 golden memory 确定日期，写回 `longmemeval_s_golden.json` 中（每条 golden 新增 `date` 字段）。

### Distractor date

混淆记忆的 date 从**非 evidence session** 的日期中选取：

```
non_evidence_dates = haystack_dates - evidence_session_dates
distractor dates 散布在 non_evidence_dates 中（8 条 distractor 取 8 个不同 date）
```

这样混淆记忆看起来像来自真实但无关的会话，更自然。

---

## 关键参数

| 参数 | 值 |
|------|:---:|
| lowering 目标区间 | `[sim(g_i)-0.4, sim(g_i)-0.05]` |
| lowering 最多尝试 | 8 |
| lowering 温度 | 0.6–0.7 |
| 闸门阈值 | `anchor_sim - 0.1` |
| 每题目标 distractor | 8 |
| 每题目标种子 | 4 |
| 种子策略数 | 4 |
| 改写策略 | 3（EQV/OSN/NSO） |
| 生成模型 | gemma4-26B |
| embedding 模型 | qwen3-embedding-0.6b |

---

## 脚本

| 脚本 | 用途 |
|------|------|
| `script/build_confusion_v3.py` | 全量批量生成（支持 --resume, --regen-lowered） |
| （v2 脚本保留不动，方便回溯对比） | |

---

## 边界情况

| 情况 | 处理 |
|------|------|
| abstention=true（30 条） | 跳过 |
| judged_correct=false（11 条） | 跳过 lowering，直接用 golden + 生成 distractor |
| 所有 golden lowering 都失败 | lowering_status="full_fallback"，lowered_golden=[]，用 golden_memory |
| 种子不足 4 条 | 尽力而为，改写扩充补 |
| distractor 不足 8 条 | constraint_ok=false，进 partial |
