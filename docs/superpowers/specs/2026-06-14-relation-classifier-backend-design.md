# relation_decision 关系判断改用 relation_classifier

日期：2026-06-14

## 背景与目标

`relation_decision` 灌库方法在"成对关系判断"阶段，对每条 `(m_old, m_new)`
记忆对调用 `manager_model`（LLM）输出五分类关系 `IND/EQV/OSN/NSO/CON`。
本设计把这一步替换为本地训练好的 `relation_classifier`
（冻结 Qwen3-0.6B + 线性探测头，test Macro F1 ≈ 0.936），
以降低对 LLM 端点的依赖、提升一致性与吞吐。

保留 `manager_model` 路径，通过开关切换，默认走分类器，便于对比实验。

## 范围

**替换范围**：仅 `LmeCandidateRelationDecisionMemorySystem._classify_relation`
这一个方法的成对关系判断逻辑。

**不在范围内**（仍用 `manager_model`）：
- cascade（`cas_update`）的两步级联判断
- deletion（用户删除请求）判断
- 事实抽取等其它 LLM 用途

下游不变：`_label_candidates` → `decide_lme_update_relation_decision`
→ `_execute_lme_plan` 完全不改。

## 关键决策

| 决策点 | 选择 |
|--------|------|
| 语言 | 分类器只支持英文；`language != "en"` 且 backend=classifier 时**构造期报错**（fail fast） |
| 切换 | 新增开关 `relation_backend`（`classifier`\|`llm`），默认 `classifier`，保留 LLM 可回退 |
| 配置/加载 | 不新增分类器 CLI 参数；backbone 路径/device 走 `RelationClassifier` 默认值 + `RC_BACKBONE_PATH` 环境变量；懒加载单例 |

## 组件设计

### 1. `relation_classifier_backend.py`（新增）

位置：`src/memory/candidate_ingest/relation_classifier_backend.py`

职责单一：隔离"线程安全 + 懒加载 + sys.path 接入"。

```python
class RelationClassifierBackend:
    def __init__(self, backbone_path=None, device=None):
        # 仅存配置，不加载 backbone
        ...
    def classify(self, old: str, new: str) -> str:
        # 懒加载单例 + 锁；返回五标签之一
        ...
```

行为：

- **懒加载**：第一次 `classify` 时才把 `relation_classifier/` 加入 `sys.path`、
  `import RelationClassifier`、构造实例（加载 ~1.5G backbone）。
  不用分类器的方法（mem0/add_all）不付出加载代价，import 期也不硬依赖 torch。
- **线程安全**：`RelationClassifier.predict` 文档明确线程不安全（共享 backbone 前向）；
  而 `_label_candidates` 用 `ThreadPoolExecutor`（`relation_concurrency`，默认 8）并发，
  episode 层也可能并发。用一把 `threading.Lock` 包住 `predict`。
  加载本身在锁内 double-checked，保证单进程只加载一次。
- **配置**：`backbone_path` / `device` 透传给 `RelationClassifier`；
  默认 `None`，由 `RelationClassifier` 默认值与 `RC_BACKBONE_PATH` 环境变量决定。

### 2. `_classify_relation` 分流（修改 `memory_system.py`）

`LmeCandidateRelationDecisionMemorySystem.__init__` 增加参数
`relation_backend: str = "classifier"`（`"classifier"` | `"llm"`）。

- backend=classifier 时构造 `self._rc_backend = RelationClassifierBackend(...)`（不触发加载）。
- **语言守卫**：`relation_backend == "classifier"` 且 `self.language != "en"` 时，
  构造期 `raise ValueError`。

`_classify_relation` 开头按 backend 分流：

- **`"classifier"`**：调 `self._rc_backend.classify(m_old_text, m_new)`，
  结果经现有 `_VALID_RELATIONS` 校验，非法回退 `"IND"`；
  写 trace（记录 backend、输入、label）。
- **`"llm"`**：走现有 `manager_model` 逻辑，原样保留。

### 3. 接线（修改 `ingest_candidates.py`）

- 新增 CLI 参数 `--relation-backend`，默认 `classifier`，`dest="relation_backend"`，
  choices `["classifier", "llm"]`。
- 透传进 `rel_kw`。
- `--relation-llm` 保持必填不变（cascade/deletion 仍需 `manager_model`）。

## 数据流

```
m_new ──> _dense_candidates ──> [(mem, ?)]
              │  每个 candidate
              ▼
       _classify_relation(mem.text, m_new)
              │  backend=classifier
              ▼
       RelationClassifierBackend.classify ──(lock)──> RelationClassifier.predict
              │  label ∈ {IND,EQV,OSN,NSO,CON}
              ▼
       decide_lme_update_relation_decision ──> _execute_lme_plan  (不变)
```

## 错误处理

- **分类器加载失败**（目录不存在 / torch 缺失）：首次 `classify` 抛出，
  沿用 `RelationClassifier` 的 `FileNotFoundError` 等明确报错。
  不静默回退到 LLM——用户明确选了分类器。
- **单次 predict 异常**：catch → 记 trace error → 回退 `"IND"`，
  与现有 LLM 失败回退一致，不中断整批灌库。

## 测试

- `RelationClassifierBackend` 单测（mock `RelationClassifier`）：
  懒加载只发生一次、锁存在、label 透传、单次异常回退 `IND`。
- `_classify_relation` 分流单测：
  backend=classifier 调 backend 不调 `llm_client`；backend=llm 反之。
- 语言守卫单测：`language="zh"` + classifier → 构造抛 `ValueError`。
- 真实 smoke（可选，需 GPU）：跑 `examples/quickstart.py` 同款 case 确认串通。
