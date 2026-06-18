# Classifier 训练数据扩充设计

日期: 2026-06-18 | 状态: 待评审

## 1. 背景与目标

**问题**：Qwen3-0.6B + 线性探测头的五分类关系分类器（IND/EQV/OSN/NSO/CON）效果不好，训练数据存在明显缺陷：
- 仅 8,803 条非 IND 样本（`non_ind.jsonl`），IND 类别为 0 条
- 来源单一（LME/MEME benchmark），分布集中在 OSN (41%)
- EQV 仅 380 条 (4.3%)，分类器难以学到 EQV 的精细边界

**目标**：利用 PersonaMem-V2 数据集扩充训练数据，用 gemma4-26B 作为裁判打标，形成数据生产→判断→纠错→重训练的闭环。

**核心原则**：gemma4-26B 的标签为金标准（teacher），classifier 不一致时以 gemma4-26B 为准。

## 2. 整体流程

```
PersonaMem-V2 (57 personas, 5,140 self preferences)
        │
        ▼
┌─────────────────────────────────────────┐
│ Step 1: 原子记忆提取 & 主语改写           │
│   - 筛选 who="self"                     │
│   - 第三人称省主语 → "the user ..."      │
│   - 过滤 ask_to_forget                  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│ Step 2: (old, new) 配对构造              │
│   a. 更新链：prev_pref → preference      │  ~2,500 对 (天然非IND)
│   b. 同persona相似配对 (embedding筛选)    │  ~1,000+ 对 (挖掘非IND)
│   c. 同persona随机配对 (IND控制30%)       │  ~1,500 对 (IND)
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│ Step 3: 双裁判并行判断                     │
│   classifier (Qwen3-0.6B) → 五分类      │
│   gemma4-26B (分类prompt) → 五分类       │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│ Step 4: 比对 & 生成训练数据               │
│   不一致 → 以gemma4-26B标签为准           │
│   一致的非IND → 保留                      │
│   IND → 控制30%整体比例                   │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│ Step 5: 合并原有数据 → 重训练分类头        │
└─────────────────────────────────────────┘
```

## 3. 各步骤详细设计

### 3.1 Step 1: 原子记忆提取 & 主语改写

**输入**：`data/raw_data/PersonaMem-v2/data/raw_data/*.json`

**处理**：
1. 遍历所有 persona JSON，展开 `conversations` 下所有子类型（chat_message, personal_email 等）
2. 筛选 `who="self"` 的 preference（5,140 条，占比 95.2%）
3. 对 `who="others"` 的 preference（259 条，4.8%），用 LLM 改写为第一人称
4. **主语改写规则**（处理 PersonaMem 第三人称省主语）：
   - 对每条 preference 文本，用规则 + 轻量 LLM 改写为 `the user ...` 形式
   - 保留语义不变，时态不变
   - 示例：
     - `"Keeps up with Nollywood movie releases"` → `"the user keeps up with Nollywood movie releases"`
     - `"Does not keep up with Nollywood movie releases"` → `"the user does not keep up with Nollywood movie releases"`
     - `"My partner is James"` 已经是第一人称，保持不变
5. `pref_type="ask_to_forget"` 的条目保留，改写 `new` 文本为自然表达：
   - `"Do not remember 'X' in memory"` → `"the user wants to forget about X"` 或类似
   - 这类 pair 通常是 CON 关系（与原始训练数据中 "Please remove that from your memory" 一致）

**输出**：`data/intermediate/personamem_atomic_memories.jsonl`
```jsonl
{"persona_id": "0", "pref_id": "0_chat_message_3", "pref_type": "stereotypical_pref", "text": "the user keeps up with Nollywood movie releases", "updated": true, "prev_text": "the user does not keep up with Nollywood movie releases", ...}
```

### 3.2 Step 2: (old, new) 配对构造

#### 2a. 更新链配对（天然非 IND）

**规则**：对 `updated=true` 且 `prev_text` 非空的条目，构造 `(old=prev_text, new=text)`。

**预期数量**：~2,500 对（排除 ask_to_forget 后约 2,500 对）

**关系分布预期**：以 CON 和 OSN/NSO 为主（偏好反转/细化），少量 EQV

#### 2b. 同 persona 相似配对（挖掘非 IND）

**目的**：同一 persona 内可能有多条语义相近但非更新链的偏好，这些 pair 可能包含 EQV/OSN/NSO/CON，主动发现可弥补非 IND 数据不足。

**方法**：
1. 用 classifier 的 backbone（Qwen3-0.6B）对每条改写后的 preference 抽取 embedding（`_features` 方法）
2. 对同一 persona 内的 preference，计算余弦相似度矩阵
3. 筛选相似度 > 0.85 但非更新链的 pair
4. 由 gemma4-26B 在 Step 3 判断真实关系

**预期数量**：~1,000+ 对（取决于相似度阈值调优）

#### 2c. 同 persona 随机配对（IND 来源）

**规则**：
1. 同 persona 内随机配对（排除 2a 和 2b 已用过的）
2. 预期大部分为 IND
3. 数量控制在最终训练集 IND 占比 ~30%

**预期数量**：~1,500 对（动态调整）

**IND 配比逻辑**：最终训练集 IND = (2c 中被判 IND 的) + (2a/2b 中被判 IND 的)，目标占比 30%。实际采样数需要预判，后续可调。

### 3.3 Step 3: 双裁判并行判断

对 Step 2 构造的所有 pair，两个裁判并行判断。

#### classifier 判断

- 调用 `RelationClassifier.predict_batch(pairs)`
- 使用现有 `head_best.pt` 权重
- 输出：每条 pair 的预测标签 + 概率分布

#### gemma4-26B 判断

- 调用 gemma4-26B（通过现有 OpenAI-compatible client）
- 使用分类 prompt：`lme_relation_classification_system_en_v2.jinja` + `lme_relation_classification_user.jinja`
- 约束 JSON 输出格式，解析五分类标签
- 注意：需要检查 gemma4-26B 在 vLLM 上的部署情况（见 `vllm-glibc-gemma4-constraint` 记忆）

**输入格式**（与 classifier 训练时一致）：
```
[RELATION_DEF prefix]

old: the user keeps up with Nollywood movie releases
new: the user does not keep up with Nollywood movie releases
```

**输出格式**：`{"label": "CON", "reason": "..."}`（具体格式以现有分类 prompt 为准）

### 3.4 Step 4: 比对 & 训练数据生成

```
对每条 pair:
  if gemma_label == classifier_label:
    if gemma_label != "IND":
      → 加入训练集, agree=true
    else:
      → 加入候选池(IND), agree=true
  else:  # 不一致
    → 加入训练集, label=gemma_label, agree=false
```

**IND 比例控制**：
1. 先统计所有非 IND 样本数 N
2. IND 目标数 = N * 0.3 / 0.7（使得 IND 占总量 30%）
3. 从 IND 候选池中采样所需数量
4. 若候选池不足，补充同 persona 随机配对

### 3.5 Step 5: 合并 & 重训练

**合并策略**：
1. PersonaMem 训练数据 + 原有 `non_ind.jsonl`（保留其 label 不变）→ 完整训练集
2. 确保无重复 pair（基于 `(old, new)` 文本去重）
3. 打乱顺序

**重训练**：
- 使用与原始训练相同的超参数（`head_best.pt` 中 `cfg` 存储的配置）
- 学习率 0.001，weight_decay 0.01，batch_size 64，epochs 50，early_stop 8
- 输出新的 `head_best.pt`

## 4. 训练数据格式

```jsonl
{"old": "the user keeps up with Nollywood movie releases", "new": "the user does not keep up with Nollywood movie releases", "label": "CON", "source": "persona_0", "gemma_label": "CON", "classifier_label": "NSO", "pref_type": "stereotypical_pref", "agree": false}
{"old": "the user is interested in sustainable fashion and home decor", "new": "the user wants to support sustainable forestry practices as a consumer", "label": "NSO", "source": "original", "gemma_label": "", "classifier_label": "", "pref_type": "", "agree": true}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `old` | str | **必填**，旧记忆（训练输入），以 `the user` 为主语 |
| `new` | str | **必填**，新记忆（训练输入），以 `the user` 为主语 |
| `label` | str | **必填**，最终标签 IND/EQV/OSN/NSO/CON |
| `source` | str | 来源 `persona_{id}` / `original` |
| `gemma_label` | str | gemma4-26B 分类结果（新数据有，原有为空） |
| `classifier_label` | str | classifier 分类结果（新数据有，原有为空） |
| `pref_type` | str | PersonaMem 偏好类型，追溯用 |
| `agree` | bool | gemma4-26B 与 classifier 是否一致 |

训练时只用 `old`, `new`, `label` 三字段。

## 5. 数据量预期

| 来源 | 数量 | 说明 |
|------|------|------|
| 2a. 更新链配对 | ~2,500 | updated=true 的 prev_pref→pref |
| 2b. 同 persona 相似配对 | ~1,000+ | embedding 相似度筛选 |
| 2c. 同 persona 随机 IND | ~1,500 | 控制在总量 30% |
| 原有 non_ind.jsonl | 8,803 | 保留 |
| **总计** | **~13,800** | IND ~11% → 调整至 30% |

实际 IND 占比由 Step 4 的动态采样逻辑控制到 30%。

## 6. 文件结构

```
relation_classifier/
├── data_expansion/                    # 新增：数据扩充流水线
│   ├── extract_preferences.py         # Step 1: 提取 + 主语改写
│   ├── construct_pairs.py             # Step 2: 配对构造 (a/b/c)
│   ├── dual_judge.py                  # Step 3: 双裁判判断
│   ├── merge_training_data.py         # Step 4: 比对 + 生成训练数据
│   └── config.yaml                    # 扩充流水线配置
├── data/                              # 新增：中间数据
│   ├── personamem_atomic_memories.jsonl
│   ├── pairs_all.jsonl
│   ├── pairs_with_judgments.jsonl
│   └── training_data_expanded.jsonl   # 最终训练数据
├── classifier.py                      # 现有：推理（不变）
├── head_best.pt                       # 现有：将被新权重替换
└── config.yaml                        # 现有：推理配置（不变）
```

## 7. 风险与注意事项

1. **gemma4-26B 成本**：~4,000 对全量判断，每条约 200 tokens 输入 + 50 tokens 输出，总量约 1M tokens，可控
2. **gemma4-26B 自身准确率**：非完美，但对非 IND 的判断应显著优于小 classifier
3. **主语改写质量**：需要抽查改写后的文本是否保持语义、是否自然
4. **embedding 相似度阈值**：2b 步骤的阈值（0.85）需根据实际分布调优
5. **原有数据冲突**：如 PersonaMem 新数据与原有 `non_ind.jsonl` 有相同或矛盾的 (old, new) 对，保留原有标签

## 8. 验收标准

1. 训练数据中 IND 占比 30% ± 5%
2. 五类标签均有足够样本（最少类别 >= 200 条）
3. 主语改写抽查通过率 > 95%
4. gemma4-26B 和 classifier 定位出至少 20% 不一致样本
5. 重训练后 val_macro_f1 不低于当前 0.932（最好有提升）
