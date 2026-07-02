# 主实验设计：Confusion + Adversarial 记忆鲁棒性评测

**日期**：2026-07-01
**状态**：设计确认，待实现

---

## 1. 论文 Story 与贡献

### 核心叙事线

1. **现实问题**：长对话中存在两类混淆记忆——（Type I）与 query 相似但不含答案的噪声，仅占坑不提供信息；（Type II）提供已被更新的过期旧值，直接诱导错误。两类混淆均会挤占 top-k / token 预算，导致 LLM 无法获取正确信息。
2. **为什么是短事实记忆**：企业场景需要低成本检索，逐条查原始对话 token 消耗不现实；短事实记忆已有大量工作（Mem0 / Zep / Evermemos 等）证明其价值。
3. **现有方法的缺陷**：Mem0 等将记忆管理交给 LLM 做端到端决策（ADD / UPDATE / DELETE），需要极强的指令遵循能力，大模型也不一定做得好，小模型更难。
4. **我们的方法（第一性原理）**：记忆管理的本质是关系判断——两条事实之间的关系只有 5 种（EQV / OSN / NSO / CON / IND）。将记忆管理降维为关系分类问题，小模型即可胜任，极大降低企业部署成本。
5. **实验证明**：在系统构造的混淆记忆基准上，RD 在 N=8 下显著优于所有 baseline，优势随混淆数量 N 单调增长；小模型消融实验验证了降维设计的价值。

### 三点贡献

1. 提出记忆管理的"关系判断降维"视角，将端到端决策简化为成对关系分类
2. 构造系统的混淆记忆评测框架，覆盖 Type I（噪声型）和 Type II（冲突型）两种真实混淆
3. 实验证明 RD 在记忆噪声下的鲁棒性显著优于现有方法（add_all / mem0 / zep / amac / evermemos），且优势随混淆数量单调增长

---

## 2. 数据集构造

### 统一数据集：470 题

基于 LongMemEval 全量可答题（排除 30 道 abstention），每道题包含：

| 组成部分 | 说明 |
|---------|------|
| Golden Memory | 原始 golden + lowered 变体（来自 confusion v4 / adversarial v1） |
| Type I Confusion（噪声型） | 与 query 相似但不含答案，5 策略生成 + EQV/OSN/NSO 改写扩充，双闸门过滤 |
| Type II Confusion（冲突型） | 仅针对 knowledge-update 题型，提供已被更新的旧值（adversarial-old-value），主动诱导错误 |
| Filler | 从非 evidence session 中按 session 采样 ~50 条真实记忆 |

### 合并策略

- **398 道非 knowledge-update 题**：使用 Type I distractor，从 confusion v4 的 32 条中按 sim_q 降序取 8 条
- **72 道 knowledge-update 题**：用 Type II（adversarial v1）替换 Type I，每道 8 条 adversarial-old-value distractor
- **Filler**：统一从非 evidence session 按 session 随机采样，每道题约 50 条

### N 档设置

| N | 主实验 | 分析实验 | 覆盖题数 |
|---|--------|---------|---------|
| 0 | ✓ | ✓ | 470 |
| 2 | — | ✓ | 470 |
| 4 | — | ✓ | 470 |
| 6 | — | ✓ | 470 |
| **8** | **✓（主表）** | ✓ | 470 |
| 16 | — | ✓ | 398（仅 Type I 题） |

### 数据来源

- **统一数据集**：`data/preprocessed/longmemeval_s_unified_confusion.json`（470 题，由合并脚本生成）
  - Type I（噪声型）：398 题，distractor 来自 confusion v4，sim_q 降序取 8
  - Type II（冲突型）：72 题，distractor 来自 adversarial v1 替换
  - 合并脚本：`script/merge_confusion_adversarial.py`
- 记忆抽取候选：复用 `MemDB/candidates/lme_s_gemma4-26B_0615_unified`（500 文件，470/470 全覆盖，无需重新抽取）
- Golden memory：复用 `data/preprocessed/longmemeval_s_golden.json`

> **已确认**：候选目录 470 题全覆盖（368 普通 + 102 `gpt4_*` 前缀，均通过 `history_name` 字段匹配）。

---

## 3. 实验设置

### 方法与模型

| 组件 | 模型 | 说明 |
|------|------|------|
| 记忆抽取 | gemma4-26B | 复用现有候选，不重新抽取 |
| 灌库 | gemma4-26B | 所有方法共用同一套 candidate |
| 答题 | gemma4-26B | 统一答题模型 |
| Judge | qwen3-max | 独立打分，与灌库/答题解耦 |
| RD 关系判断 | gemma4-26B | LLM-only 版本（不用 classifier/校验） |

### Baselines（6 方法）

| 方法 | 灌库策略 | 关键 prompt |
|------|---------|------------|
| add_all | 无差别全存 | — |
| mem0 | 端到端 LLM 记忆管理 | `mem0_update_memory_default_en.jinja` |
| zep | 结构化记忆图 | 现有实现 |
| amac | 自适应记忆聚合 | 现有实现 |
| evermemos | 持续记忆更新 | `evermemos_consolidate_en.jinja` |
| **RD (ours)** | 逐对关系判断 → 折叠/融合 | 见 §4 |

### 公平性保证

- 所有方法使用**同一套 candidate memory** 作为灌库输入
- 灌库顺序严格遵守时间线（filler → distractor → golden，按 session date 排列）
- 答题时使用 **memory_token_limit=256**（非 top-k），检索 topk 设得足够大（如 999），实际由 token limit 截断
  - 理由：不同方法更新后记忆 token 数不同，top-k 会导致信息量不对等
- 同一模型（gemma4-26B）灌库和答题，消除模型差异

### 评测指标

- **Accuracy**：qwen3-max judge 判定 yes/no（独立于灌库/答题模型）
- 按 question_type 子集拆分：
  - knowledge-update（72 题）— Type II 重点分析
  - multi-session（121 题）
  - temporal-reasoning（127 题）
  - single-session-user（64 题）
  - single-session-assistant（56 题）
  - single-session-preference（30 题）

---

## 4. Prompt 配置

| 环节 | 方法 | Prompt 模板 |
|------|------|------------|
| **RD 分类 system** | RD | `lme_relation_classification_system_en_v3.jinja` |
| **RD 分类 user** | RD | `lme_relation_classification_user.jinja` |
| **RD 校验** | RD | **不使用**（LLM-only 版本不做校验） |
| **RD 融合 EQV** | RD | `lme_answer_fuse_eqv_en.jinja` |
| **RD 融合 OSN** | RD | `lme_answer_fuse_osn_en.jinja` |
| **RD 融合 NSO** | RD | `lme_answer_fuse_nso_en.jinja` |
| **RD 融合 CON** | RD | `lme_answer_fuse_con_en.jinja` |
| **mem0 更新** | mem0 | `mem0_update_memory_default_en.jinja` |
| **mem0 上下文** | mem0 | `agent_context_empty_en.jinja` + `mem0_context_unit_en.jinja` |
| **evermemos** | evermemos | `evermemos_consolidate_en.jinja` |
| **Judge** | 评测 | `pipeline_eval_oqa.jinja` |

---

## 5. 分析实验

### 5.1 N-退化曲线（N=0→2→4→6→8→16）

- 6 方法在 398 题上画 accuracy 随 N 的退化曲线
- 主表 N=8 已展示关键差异，曲线证明趋势单调性
- N=16 为极端挤占点（99.6% 题超 256 token），仅 Type I 题参与

### 5.2 Knowledge-update 子集分析（72 题）

- Type II 冲突型混淆的独立分析
- 对比各方法在"需要时序推理 + 旧值干扰"场景下的表现
- RD 的 CON 折叠机制在此子集上预期优势明显

### 5.3 机制分析：RD 折叠率与准确率

- 折叠率（非 primary 比例）随 N 的变化
- 各关系类型生效分布：EQV / OSN / NSO / CON
- 融合 C 的数量增长
- 论证折叠率 → 有效 token 节省 → accuracy 优势的因果链

### 5.4 小模型灌库消融

- 用更小模型（如 qwen3-8B）替换 gemma4-26B 做 RD 的关系判断
- 对比其他方法在小模型下的退化（mem0 等依赖强指令遵循的方法预期崩得更快）
- 验证"关系判断降维后小模型也可用"的核心论点

### 5.5 Filler 敏感度（可选）

- 对比加/不加 filler 的趋势一致性
- 若与 Phase 3 结论一致可简化为一段文字说明

---

## 6. 预期结果

### 主实验（N=8）

| 方法 | 预期 Accuracy (N=8) | 说明 |
|------|-------------------|------|
| add_all | ~50-60% | 无折叠，distractor 直接挤占 golden |
| mem0 | ~55-65% | LLM 端到端管理，部分去重 |
| zep | ~60-70% | 结构化可能减少冗余 |
| amac | ~55-65% | 自适应聚合，效果有限 |
| evermemos | ~60-70% | 持续更新，对冲突有一定处理 |
| **RD (ours)** | **~75-85%** | 折叠 distractor 为 evidence，不占独立槽位 |

### N-退化曲线

- add_all 预期单调陡降：92%→~50%（N=0→8）
- RD 预期平缓下降：94%→~78%（N=0→8）
- Δ 随 N 单调增长：~1%（N=0）→ ~25%（N=8）

---

## 7. 注意事项

1. **候选目录缺口**：`0615_unified` 仅覆盖 368/470 题，实现前需确认处理方案（交集 or 补抽）
2. **Filler 采样随机性**：需固定 seed 保证可复现
3. **灌库顺序**：Type II 的 adversarial-old-value 应灌在对应 golden 之前（更早的 session date），模拟"先知道旧值后知道新值"的时序
4. **Token limit 截断**：确认 memory_token_limit 在答题环节的截断策略（是否保留 golden 的 chunk 排序优势）
5. **多方法对比**：zep / amac 的现有实现需确认在当前代码中可直接运行
