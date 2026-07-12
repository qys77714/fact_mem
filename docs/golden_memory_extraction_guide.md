# Golden Memory 提取指南

本文档总结从 LME 数据集提取 golden memory 的完整流程和经验，可直接复用到新数据集。

## 一、什么是 Golden Memory

Golden memory 是从 evidence session 中蒸馏出的**最小原子事实集**。每条 fact 是一个以 "The user" 开头的陈述句，附带该事实对应的 session 日期。

```json
{
  "golden_memory": [
    {"content": "The user graduated with a degree in Business Administration.", "date": "2023/05/30"},
    {"content": "The user's previous last name was Johnson as of 2023/03/15.", "date": "2023/03/15"},
    {"content": "The user changed their last name to Winters as of 2023/05/28.", "date": "2023/05/28"}
  ]
}
```

## 二、前置条件

需要以下数据：

| 字段 | 说明 |
|------|------|
| `question` | 问题文本 |
| `answer` | 标准答案（gold answer）|
| `question_type` | 题型（knowledge-update / multi-session / temporal-reasoning / single-session-* / preference）|
| `question_date` | 提问日期，格式 `YYYY/MM/DD (Day) HH:MM` |
| `answer_session_ids` | evidence session 的 ID 列表 |
| `haystack_sessions` | 所有 session 的对话内容（list of turns） |
| `haystack_session_ids` | session ID 列表（与 haystack_sessions 对齐）|
| `haystack_dates` | session 日期列表（与 haystack_session_ids 对齐）|
| `has_answer` | 每个 turn 的布尔标记，标记是否包含答案相关信息 |

## 三、提取流程（三阶段）

### Phase 1：逐 session 提取

每个 evidence session 独立调一次 LLM，输入：

- 整个 session 的对话文本（`has_answer=True` 的 turn 用 `⚑` 前缀标记）
- 问题 + 标准答案 + 题型

输出：一条以 "The user" 开头的原子事实。

**模板**：`lme_golden_memory_distill_v2_en.jinja`

**关键约束**：
- 必须严格以 "The user" 开头（不能有 "As of..." 前缀）
- 相关事实被 assistant 说出时，改写为以用户为中心："The user was told about..."
- 允许输出多条（拆分裂开）或 null（本 session 无相关信息）

**并发**：用 `ThreadPoolExecutor`，建议 4-6 线程（取决于 vLLM 的 TP 配置）

**缓存**：提取结果存 JSON，避免重复调用（~900 次 LLM 调用/500 题）

### Phase 1.5：全局 Consolidation

将所有 session 的提取结果 + has_answer turns + session dates 一起给 LLM：

**输入**：question + question_type + question_date + gold answer + 全部 session 的 `(date, extracted_GM, has_answer_turns)`

**输出**：精炼后的 fact 列表

**模板**：`lme_golden_memory_consolidate_en.jinja`

**关键规则**：

| 规则 | 说明 |
|------|------|
| 去重合并 | 不同 session 提取了同一事实 → 保留一条最精确的 |
| 拆分裂开 | 一条 GM 包含多个独立可数事实 → 拆成多条 |
| 丢弃无关 | session 事实与答案无关 → 删除 |
| 日期转换 | "yesterday"/"last week" → 绝对日期（用 session_date 推算）|
| Knowledge-update ≥2 条 | 必须保留旧值和新值各一条，**不能合并** |
| 主语统一 | 全部以 "The user" 开头 |
| 永不返回空 | 非 abstention 题至少保留原始提取结果 |

### Phase 1.6：日期匹配

Consolidation 输出的 fact 是纯文本，需要用 LLM 将每条 fact 匹配回 source session 的日期：

**输入**：consolidation 后的 fact 列表 + 全部 session 的 `(date, has_answer_turns)`

**输出**：`[{"content": "...", "date": "YYYY/MM/DD"}, ...]`

**兜底**：正则从 fact 文本中提取日期（`\b\d{4}/\d{2}/\d{2}\b`、`in April 2023` 等）

### Phase 2：评估回答 + 兜底重试

**回答**：用 `agent_prompt_en_open.jinja` + `lme_memory_context_unit_en.jinja`（与 `run_exp_lme.py` 一致）

**Judge**：`pipeline_eval_system.jinja` + `pipeline_eval_oqa.jinja`

**兜底重试**：如果 consolidation 结果答错，做 holistic extraction——把全部 has_answer turns + question + gold answer 一次性给 LLM 重新提取。用失败答案作为 feedback。最多重试 5 次。

## 四、LLM 配置

| 用途 | 模型 | 说明 |
|------|------|------|
| 逐 session 提取 | gemma4-26B | 本地 vLLM，TP=4 |
| Consolidation | gemma4-26B | 同上 |
| 日期匹配 | gemma4-26B | 同上 |
| 回答 (answer) | gemma4-26B | 同上 |
| Judge | gemma4-26B | 同上 |

全部使用 `load_api_chat_completion("gemma4-26B")`，temperature=0.0。

## 五、Abstention 处理

`question_id` 以 `_abs` 结尾的题目为 abstention——证据不包含足够信息回答问题。这类题：

- `golden_memory = []`
- `abstention = true`
- 不调 LLM 提取

## 六、运行方式

```bash
# 首次运行（提取+consolidation+评估，提取结果缓存）
PYTHONPATH=src uv run --no-sync python script/build_lme_golden_memory_v2.py \
    --max-workers 6

# 后续调整 consolidation 参数（跳过提取，4分钟完成）
PYTHONPATH=src uv run --no-sync python script/build_lme_golden_memory_v2.py \
    --skip-extraction --max-workers 8

# 生成 HTML 审查报告
PYTHONPATH=src uv run --no-sync python script/reeval_golden_memory.py
```

## 七、踩过的坑

1. **Consolidation 误标冗余**：早期版本会标记 redundant，导致 Fitbit 等设备被当成冗余删除。解决方案：完全去掉 redundant 标记，每个 session 的事实都保留。

2. **"As of..." 日期前缀**：LLM 喜欢写 "As of 2023/05/30, the user..."，这违反 "The user" 开头规则。解决方案：模板明确禁止，日期存入独立字段。

3. **Knowledge-update 合并旧新值**：consolidation 倾向于把旧值和新值合并成一条。解决方案：模板强制要求 knowledge-update ≥2 条。

4. **回答模板日期缺失**：reeval 脚本里 `qtime = "unknown"` 写死了，导致所有 temporal 题无法计算时间差。解决方案：从原始数据读 `question_date`。

5. **模型算术失败**：列出 3 盆植物的 GM，模型不会数出 3。这是回答模型能力上限，不在 GM 层面解决。

6. **单 session 多 has_answer turn**：assistant 回声确认会产生连续标记 turn，提取时只取一条事实。

7. **并发过高导致 vLLM 超时**：TP=4 的 vLLM 建议 ≤6 并发线程，否则返回 None。

## 八、新数据集适配清单

1. 准备数据：确保有 `question / answer / question_type / question_date / answer_session_ids / haystack_sessions + ids + dates / has_answer`
2. 确认 abstention 标记方式（`_abs` 后缀或其他）
3. 修改 `load_original_data()` 适配数据路径和格式
4. 修改 `is_abstention()` 适配标记逻辑
5. 调整 `format_session_text()` 适配对话格式（speaker 字段名等）
6. 运行 `--max-workers 6` 首次提取
7. 根据准确率调整 consolidation 模板
8. 后续迭代用 `--skip-extraction` 加速
