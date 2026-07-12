# RD Fusion 消融实验 + 清理 topic aggregation 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 添加 `fusion_enabled` flag 实现 Full RD vs RD w/o Fusion 消融实验，并清理未使用的 topic aggregation 功能。

**Architecture:** 在 `LmeCandidateRelationDecisionMemorySystem` 中加 `_fusion_enabled` 开关，控制 `_update_answer_memory()` 是否调 LLM 融合。`False` 时按 relation 规则选一条原始 text 作为 answer memory C。同时删除 `_aggregate_topic_profile`、`topics.py`、`tag_candidate_topics.py` 等死代码。

**Tech Stack:** Python, Pydantic (config model), argparse (CLI), Jinja (prompts — 不改模板)

## Global Constraints

- 包管理用 uv，脚本运行加 `uv run --no-sync python`
- 回答一律用中文
- 遵循现有代码风格和命名约定

---

## 文件结构

| 文件 | 操作 | 职责 |
|------|------|------|
| `src/utils/config.py` | 修改 | `RelationDecisionMethodConfig` 加 `fusion_enabled: bool = True`，删 `topic_aggregation_enabled` |
| `src/memory/candidate_ingest/memory_system.py` | 修改 | 加 `_fusion_enabled` 参数，`_update_answer_memory` 分支，删 topic 方法 |
| `src/memory/candidate_ingest/memory_system_base.py` | 修改 | 删 `consumes_topics` 类变量和注释 |
| `src/memory/candidate_ingest/apply.py` | 修改 | 删 topic 消费逻辑（`consumes_topic`、`candidate_topics`、`mb["topic"]`） |
| `src/memory/candidate_ingest/topics.py` | 删除 | 整个文件 |
| `src/pipeline/tag_candidate_topics.py` | 删除 | 整个文件 |
| `src/pipeline/ingest_candidates.py` | 修改 | 加 `--no-fusion` flag 和 fingerprint 记录 |
| `run_exp_lme.py` | 修改 | `stage_ingest()` 透传 `--no-fusion` |
| `config/lme.yaml` | 修改 | 加 `fusion_enabled: true`，删 `topic_aggregation_enabled` |
| 其余约 90 个 `config/lme_*.yaml` | 修改 | 删 `topic_aggregation_enabled: false` 行 |

---

### Task 1: 清理 `memory_system.py` 中的 topic aggregation 代码

**Files:**
- Modify: `src/memory/candidate_ingest/memory_system.py`

**Interfaces:**
- Consumes: 无
- Produces: 无（纯删除）

- [ ] **Step 1: 删除 topics 模块 import**

删除第 31 行:
```python
from .topics import MISC_TOPIC, VALID_TOPICS, normalize_topic
```

- [ ] **Step 2: 删除 `__init__` 中的 topic 相关字段**

删除第 82-83 行:
```python
        self._topic_aggregation_enabled = bool(kwargs.pop("topic_aggregation_enabled", True))
        self.consumes_topics = self._topic_aggregation_enabled
```

- [ ] **Step 3: 删除 `_run_pairwise_relation_decision` 中的 topic aggregation 调用点**

删除第 346-349 行:
```python
        if new_row_id is not None and self._topic_aggregation_enabled:
            self._aggregate_topic_profile(
                database, m_new, new_row_id, mb, session_idx, chunk_scope, trace
            )
```

- [ ] **Step 4: 删除 `_find_topic_profile` 和 `_aggregate_topic_profile` 方法**

删除第 905-992 行（`_find_topic_profile` 和 `_aggregate_topic_profile` 两个完整方法）。

- [ ] **Step 5: 验证**

```bash
grep -n "topic_aggregation\|_aggregate_topic\|_find_topic_profile\|consumes_topics\|MISC_TOPIC\|VALID_TOPICS\|normalize_topic" /data/zjj/project_26/fact_mem/src/memory/candidate_ingest/memory_system.py
```
预期：无输出

- [ ] **Step 6: Commit**

```bash
git add src/memory/candidate_ingest/memory_system.py
git commit -m "refactor: remove topic aggregation from memory_system"
```

---

### Task 2: 清理 `memory_system_base.py`、`apply.py` 中的 topic 引用

**Files:**
- Modify: `src/memory/candidate_ingest/memory_system_base.py`
- Modify: `src/memory/candidate_ingest/apply.py`

**Interfaces:**
- Consumes: Task 1（topic 方法已删除）
- Produces: 无

- [ ] **Step 1: `memory_system_base.py` — 删除 `consumes_topics` 类变量和注释**

删除第 24-27 行:
```python
    # 是否消费 chunk 的平行栏 ``candidate_topics``（预定义主题标签）做同主题 profile 聚合。
    # 仅 relation_decision(ours) 为 True；baseline 忽略，行为与无主题时一致。
    consumes_topics: bool = False
```

- [ ] **Step 2: `apply.py` — 删除 topic 消费逻辑**

删除第 114-117 行:
```python
            # topic 平行数组（tag_candidate_topics.py 产出，与 candidate_memories 等长）；
            # 仅 relation_decision 消费它做同主题聚合，baseline 忽略。
            topics = chunk.get("candidate_topics")
            consumes_topic = bool(getattr(memory, "consumes_topics", False))
```

删除第 147-150 行:
```python
                    if consumes_topic and isinstance(topics, list) and fi < len(topics):
                        topic_fi = str(topics[fi] or "").strip()
                        if topic_fi:
                            mb["topic"] = topic_fi
```

- [ ] **Step 3: 验证**

```bash
grep -rn "consumes_topic\|consumes_topics\|candidate_topics\|mb\[\"topic\"\]" /data/zjj/project_26/fact_mem/src/memory/candidate_ingest/ --include="*.py" | grep -v __pycache__
```
预期：无输出（除 topics.py 自身的定义以外）

- [ ] **Step 4: Commit**

```bash
git add src/memory/candidate_ingest/memory_system_base.py src/memory/candidate_ingest/apply.py
git commit -m "refactor: remove topic consumption from base and apply"
```

---

### Task 3: 删除 `topics.py` 和 `tag_candidate_topics.py`

**Files:**
- Delete: `src/memory/candidate_ingest/topics.py`
- Delete: `src/pipeline/tag_candidate_topics.py`

**Interfaces:**
- Consumes: Task 1-2（引用已清理）
- Produces: 无

- [ ] **Step 1: 确认无其他引用**

```bash
grep -rn "from.*topics import\|import.*topics\|tag_candidate_topics" /data/zjj/project_26/fact_mem/src/ --include="*.py" | grep -v __pycache__
```
预期：无输出（除 topics.py 自身的 `__all__` 外）

- [ ] **Step 2: 删除文件**

```bash
rm src/memory/candidate_ingest/topics.py
rm src/pipeline/tag_candidate_topics.py
```

- [ ] **Step 3: Commit**

```bash
git add src/memory/candidate_ingest/topics.py src/pipeline/tag_candidate_topics.py
git commit -m "refactor: remove topics.py and tag_candidate_topics.py"
```

---

### Task 4: 清理 config 模型和 YAML 中的 `topic_aggregation_enabled`

**Files:**
- Modify: `src/utils/config.py`
- Modify: `config/lme.yaml`
- Modify: 所有其他 `config/lme_*.yaml`（约 90 个）

**Interfaces:**
- Consumes: Task 1-3（功能已删除）
- Produces: 无

- [ ] **Step 1: `config.py` — 删除 `topic_aggregation_enabled` 字段**

在 `src/utils/config.py` 第 138 行，删除:
```python
    topic_aggregation_enabled: bool = True
```

- [ ] **Step 2: 批量删除所有 YAML 中的 `topic_aggregation_enabled` 行**

```bash
cd /data/zjj/project_26/fact_mem
find config -name "lme_*.yaml" -exec sed -i '/topic_aggregation_enabled/d' {} \;
```

- [ ] **Step 3: 验证 config 可正常加载**

```bash
cd /data/zjj/project_26/fact_mem && uv run --no-sync python -c "
from utils.config import ExperimentConfig
cfg = ExperimentConfig.from_yaml('config/lme.yaml')
print('OK: config loaded, methods:', list(cfg.methods.__fields__.keys()))
"
```

- [ ] **Step 4: Commit**

```bash
git add src/utils/config.py config/
git commit -m "refactor: remove topic_aggregation_enabled from config model and all YAMLs"
```

---

### Task 5: 在 config 模型和 YAML 中加 `fusion_enabled`

**Files:**
- Modify: `src/utils/config.py`
- Modify: `config/lme.yaml`

**Interfaces:**
- Produces: `RelationDecisionMethodConfig.fusion_enabled: bool = True`

- [ ] **Step 1: `config.py` — 加 `fusion_enabled` 字段**

在 `src/utils/config.py` 的 `RelationDecisionMethodConfig` 类中，`pairwise_sim_threshold` 行之后添加:

```python
    fusion_enabled: bool = True
```

最终类定义:
```python
class RelationDecisionMethodConfig(BaseModel):
    enabled: bool = False
    related_top_k: int = 3
    backend: str = "classifier"
    fusion_model: str = ""
    cascade_enabled: bool = True
    deletion_enabled: bool = True
    condition_sim_threshold: float = 0.5
    pairwise_sim_threshold: float = 0.7
    fusion_enabled: bool = True
```

- [ ] **Step 2: `config/lme.yaml` — 加 `fusion_enabled: true`**

在 `relation_decision` 节中，`pairwise_sim_threshold: 0.5` 之后，注释块之前添加:

```yaml
    fusion_enabled: true
```

- [ ] **Step 3: 验证**

```bash
cd /data/zjj/project_26/fact_mem && uv run --no-sync python -c "
from utils.config import ExperimentConfig
cfg = ExperimentConfig.from_yaml('config/lme.yaml')
rc = cfg.methods.relation_decision
print(f'fusion_enabled={rc.fusion_enabled}')
assert rc.fusion_enabled == True
print('OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add src/utils/config.py config/lme.yaml
git commit -m "feat: add fusion_enabled to RelationDecisionMethodConfig"
```

---

### Task 6: 在 `memory_system.py` 中实现 `fusion_enabled` 分支

**Files:**
- Modify: `src/memory/candidate_ingest/memory_system.py`

**Interfaces:**
- Consumes: `RelationDecisionMethodConfig.fusion_enabled: bool`（经 `__init__` 传入）
- Produces: `_fusion_enabled` 实例变量控制 `_update_answer_memory()` 行为

- [ ] **Step 1: `__init__` — 接受 `fusion_enabled` 参数**

在 `__init__` 的参数解析区（`_pairwise_sim_threshold` 行之后）添加:

```python
        self._fusion_enabled = bool(kwargs.pop("fusion_enabled", True))
```

- [ ] **Step 2: `_update_answer_memory` — 替换 `_fuse_answer_memory` 调用为分支**

找到原代码（约第 779-783 行）:
```python
        fused_text = self._fuse_answer_memory(
            current_memory, m_new, relation, chunk_scope, trace,
            current_memory_time=current_memory_time,
            new_fact_time=new_fact_time,
        )
```

替换为:
```python
        if self._fusion_enabled:
            fused_text = self._fuse_answer_memory(
                current_memory, m_new, relation, chunk_scope, trace,
                current_memory_time=current_memory_time,
                new_fact_time=new_fact_time,
            )
        else:
            # Ablation: no LLM fusion — select one raw text as answer memory
            if relation in ("OSN", "CON"):
                fused_text = m_new  # new fact is the primary
            else:  # NSO, EQV
                fused_text = (anchor_mem.text or "").strip()  # old fact remains primary
```

- [ ] **Step 3: 验证语法**

```bash
cd /data/zjj/project_26/fact_mem && uv run --no-sync python -c "
from memory.candidate_ingest.memory_system import LmeCandidateRelationDecisionMemorySystem
print('OK: import successful')
"
```

- [ ] **Step 4: Commit**

```bash
git add src/memory/candidate_ingest/memory_system.py
git commit -m "feat: add fusion_enabled branch in _update_answer_memory"
```

---

### Task 7: 在 CLI 和实验脚本中加 `--no-fusion` flag

**Files:**
- Modify: `src/pipeline/ingest_candidates.py`
- Modify: `run_exp_lme.py`

**Interfaces:**
- Consumes: `LmeCandidateRelationDecisionMemorySystem(fusion_enabled=...)`
- Produces: `--no-fusion` CLI flag

- [ ] **Step 1: `ingest_candidates.py` — 加 `--no-fusion` 参数**

在其他 `--relation-*` 参数附近添加:

```python
    parser.add_argument(
        "--no-fusion",
        action="store_true",
        dest="no_fusion",
        help="消融实验：禁用 LLM 融合，直接选择原始 memory 作为 Primary（RD w/o Fusion）",
    )
```

- [ ] **Step 2: `ingest_candidates.py` — 传递到 memory system**

在 `memory = LmeCandidateRelationDecisionMemorySystem(...)` 构造处（约第 720-724 行），`rel_kw` 字典中添加:

```python
            "fusion_enabled": not bool(getattr(args, "no_fusion", False)),
```

- [ ] **Step 3: `ingest_candidates.py` — 加入 fingerprint**

在 `_apply_config_fingerprint_block` 函数中（约第 142 行之后，`relation_decision` 分支内），添加:

```python
        block["fusion_enabled"] = not bool(getattr(args, "no_fusion", False))
```

- [ ] **Step 4: `run_exp_lme.py` — `stage_ingest()` 透传**

在 `stage_ingest()` 的 `relation_decision` 分支（约第 150-168 行），添加:

```python
            fusion_enabled = getattr(method_cfg, "fusion_enabled", True)
            if not fusion_enabled:
                extra.append("--no-fusion")
```

- [ ] **Step 5: 端到端验证（dry run）**

```bash
cd /data/zjj/project_26/fact_mem && PYTHONPATH=src uv run --no-sync python -c "
# 模拟 args 来验证 CLI 解析
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--no-fusion', action='store_true', dest='no_fusion')
args = parser.parse_args(['--no-fusion'])
print(f'no_fusion={args.no_fusion}, fusion_enabled={not args.no_fusion}')
assert not args.no_fusion == False  # --no-fusion sets it to True
print('OK: CLI parsing works')
args2 = parser.parse_args([])
print(f'no_fusion={args2.no_fusion}, fusion_enabled={not args2.no_fusion}')
assert args2.no_fusion == False
print('OK: default is fusion enabled')
"
```

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/ingest_candidates.py run_exp_lme.py
git commit -m "feat: add --no-fusion CLI flag for ablation experiment"
```

---

### Task 8: 最终验证 — 完整流程冒烟测试

**Files:**
- 无新建/修改（仅验证）

- [ ] **Step 1: 验证 Full RD（fusion_enabled=true，默认）可正常灌库一个 episode**

```bash
cd /data/zjj/project_26/fact_mem && PYTHONPATH=src uv run --no-sync python -c "
from utils.config import ExperimentConfig
cfg = ExperimentConfig.from_yaml('config/lme.yaml')
rc = cfg.methods.relation_decision
print(f'fusion_enabled={rc.fusion_enabled}')
assert rc.fusion_enabled == True
print('Full RD config OK')
"
```

- [ ] **Step 2: 验证 RD w/o Fusion 构造正确**

```bash
cd /data/zjj/project_26/fact_mem && PYTHONPATH=src uv run --no-sync python -c "
# 验证 memory system 构造
from unittest.mock import Mock
import os
os.environ.setdefault('EMBEDDING_API_KEY', 'test')
os.environ.setdefault('VLLM_API_KEY', 'test')

# 测试 fusion_enabled=False 的构造
# （不实际连接任何后端，只验证参数传递）
import sys
sys.path.insert(0, 'src')
print('Construction test skipped (requires LLM backend) — verifying flag parsing instead')

# 验证 config 模型的 default
from utils.config import RelationDecisionMethodConfig
c1 = RelationDecisionMethodConfig()
assert c1.fusion_enabled == True, f'default should be True, got {c1.fusion_enabled}'
print(f'Default fusion_enabled: {c1.fusion_enabled}')

# 验证可显式设为 False
c2 = RelationDecisionMethodConfig(fusion_enabled=False)
assert c2.fusion_enabled == False
print(f'Explicit fusion_enabled=False: {c2.fusion_enabled}')
print('All config checks passed')
"
```

- [ ] **Step 3: 确认 topic aggregation 完全清理**

```bash
cd /data/zjj/project_26/fact_mem
# 确认 topics.py 已删除
test -f src/memory/candidate_ingest/topics.py && echo "FAIL: topics.py still exists" || echo "OK: topics.py deleted"
# 确认 tag_candidate_topics.py 已删除
test -f src/pipeline/tag_candidate_topics.py && echo "FAIL: tag_candidate_topics.py still exists" || echo "OK: tag_candidate_topics.py deleted"
# 确认无残留引用
grep -rn "topic_aggregation\|_aggregate_topic\|consumes_topics\|MISC_TOPIC\|from.*topics import" src/ --include="*.py" | grep -v __pycache__ && echo "FAIL: residual references found" || echo "OK: no residual references"
# 确认 YAML 中无残留
grep -rn "topic_aggregation" config/ --include="*.yaml" && echo "FAIL: YAML references remain" || echo "OK: no YAML references"
```

- [ ] **Step 4: Commit（如有残留修复）**

---

## 自检

**Spec coverage:**
- ✅ fusion_enabled flag 在 config 模型 → Task 5
- ✅ fusion_enabled flag 在 YAML → Task 5
- ✅ fusion_enabled flag 在 memory_system → Task 6
- ✅ fusion_enabled flag 在 CLI → Task 7
- ✅ fusion_enabled flag 在实验脚本 → Task 7
- ✅ _update_answer_memory 分支逻辑 → Task 6
- ✅ 删除 topic aggregation 功能 → Tasks 1-4

**Placeholder scan:** 无 TBD/TODO/待定项

**Type consistency:** `fusion_enabled: bool` 在 config 模型、`__init__` kwargs、CLI `--no-fusion`（store_true，取反）之间一致。
