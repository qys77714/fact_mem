# RD Fusion 消融实验设计

## 目标

验证 RD 的收益究竟来自"根据关系丢弃/归档冗余记忆"，还是来自"将相关事实融合成对回答友好的 Primary memory"。

## 实验设计

统一使用相同的 relation classifier、相同的 retrieved old memories、相同的五类关系和 Primary/Evidence 决策；**唯一改变 Primary memory 的生成方式**。

| 版本 | 关系后的处理 |
|------|-------------|
| Full RD | 对 EQV/OSN/CON 生成 fused Primary text；相关原始 memories 保留为 Evidence。CON 显式保留当前值、历史值及时间戳。 |
| RD w/o Fusion | 不生成 fused text。按关系规则选择一个原始 memory 作为 Primary，其余相关 memory 作为 Evidence。OSN→new, NSO→old, EQV→representative, CON→latest。 |

## 评测分组

- **非时序题** (289题): single-session-user, single-session-assistant, single-session-preference, multi-session
- **时序题** (211题): temporal-reasoning, knowledge-update

预期：非时序题两者接近 → relation-based consolidation 本身有效；时序题 Full RD 更好 → fusion 将"当前值+历史值+时间"显式写入可检索的 Primary。

## 实现方案

### 新增 `fusion_enabled` flag

| 层 | 文件 | 改动 |
|---|------|------|
| 配置模型 | `src/utils/config.py` | `RelationDecisionMethodConfig` 加 `fusion_enabled: bool = True` |
| 记忆系统 | `src/memory/candidate_ingest/memory_system.py` | `__init__` 接受 `fusion_enabled`；`_update_answer_memory` 分支 |
| CLI | `src/pipeline/ingest_candidates.py` | 加 `--no-fusion` flag |
| 实验脚本 | `run_exp_lme.py` | `stage_ingest()` 透传 |

### `_update_answer_memory()` 分支逻辑

- `fusion_enabled=True`: 调 `_fuse_answer_memory()` → LLM 融合 → answer memory C（现状）
- `fusion_enabled=False`: 按 relation 选原始文本 → answer memory C:
  - OSN: m_new（新事实已是 primary）
  - NSO: anchor old text（旧 primary 更具体）
  - EQV: anchor old text（representative）
  - CON: m_new（最新状态）

### 同步清理 topic aggregation

删除 `_aggregate_topic_profile`、`_find_topic_profile`、`topics.py`、`tag_candidate_topics.py` 及相关引用和配置项。
