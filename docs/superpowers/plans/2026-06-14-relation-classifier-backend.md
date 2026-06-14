# relation_classifier backend for relation_decision — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `relation_decision` 的成对关系判断默认改用本地 `relation_classifier`，保留 `manager_model` 作为可切换回退。

**Architecture:** 新增一个懒加载、加锁的 `RelationClassifierBackend` 封装 `relation_classifier.RelationClassifier`；`LmeCandidateRelationDecisionMemorySystem._classify_relation` 按 `relation_backend` 开关分流到分类器或现有 LLM 逻辑；CLI 暴露 `--relation-backend`，默认 `classifier`。

**Tech Stack:** Python 3.12, pytest, torch/transformers（分类器），现有 `memory.candidate_ingest` 模块。

---

## File Structure

- Create: `src/memory/candidate_ingest/relation_classifier_backend.py` — 懒加载 + 线程安全封装，对外 `classify(old, new) -> str`
- Modify: `src/memory/candidate_ingest/memory_system.py` — 构造参数 `relation_backend`、语言守卫、`_classify_relation` 分流
- Modify: `src/pipeline/ingest_candidates.py` — `--relation-backend` 参数 + 透传
- Create: `tests/conftest.py` — 把 `src/` 加入 `sys.path`
- Create: `tests/memory/candidate_ingest/test_relation_classifier_backend.py`
- Create: `tests/memory/candidate_ingest/test_classify_relation_dispatch.py`

`relation_classifier/` 仓库目录路径在 backend 内通过 repo 根定位（`<repo>/relation_classifier`）后 `sys.path.insert`。

---

## Task 1: 测试基础设施（conftest）

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: 写 conftest 把 src 加进 sys.path**

```python
# tests/conftest.py
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
```

- [ ] **Step 2: 验证可导入目标模块**

Run: `cd /data/zjj/project_26/fact_mem && python -c "import sys; sys.path.insert(0,'src'); import memory.candidate_ingest.memory_system"`
Expected: 无输出、退出码 0（确认包路径正确）。

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add conftest to put src on sys.path"
```

---

## Task 2: RelationClassifierBackend（懒加载 + 锁）

**Files:**
- Create: `src/memory/candidate_ingest/relation_classifier_backend.py`
- Test: `tests/memory/candidate_ingest/test_relation_classifier_backend.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/memory/candidate_ingest/test_relation_classifier_backend.py
import sys
import types
import pytest
from memory.candidate_ingest.relation_classifier_backend import RelationClassifierBackend


def _install_fake_classifier(monkeypatch, label="EQV", calls=None, raise_exc=None):
    """在 sys.modules 注入假的 classifier 模块，避免加载真实 backbone。"""
    instances = {"count": 0}

    class FakeRC:
        def __init__(self, *a, **k):
            instances["count"] += 1

        def predict(self, old, new, return_probs=True):
            if calls is not None:
                calls.append((old, new))
            if raise_exc is not None:
                raise raise_exc
            return {"label": label, "label_id": 1, "probs": {}}

    mod = types.ModuleType("classifier")
    mod.RelationClassifier = FakeRC
    monkeypatch.setitem(sys.modules, "classifier", mod)
    return instances


def test_classify_returns_label(monkeypatch):
    _install_fake_classifier(monkeypatch, label="CON")
    b = RelationClassifierBackend()
    assert b.classify("I live in Beijing.", "I moved to Shanghai.") == "CON"


def test_lazy_load_only_once(monkeypatch):
    inst = _install_fake_classifier(monkeypatch, label="IND")
    b = RelationClassifierBackend()
    assert inst["count"] == 0          # 构造不加载
    b.classify("a", "b")
    b.classify("c", "d")
    assert inst["count"] == 1          # 多次 classify 只加载一次


def test_predict_exception_falls_back_to_ind(monkeypatch):
    _install_fake_classifier(monkeypatch, raise_exc=RuntimeError("boom"))
    b = RelationClassifierBackend()
    assert b.classify("a", "b") == "IND"


def test_has_lock(monkeypatch):
    _install_fake_classifier(monkeypatch)
    b = RelationClassifierBackend()
    import threading
    assert isinstance(b._lock, type(threading.Lock()))
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /data/zjj/project_26/fact_mem && python -m pytest tests/memory/candidate_ingest/test_relation_classifier_backend.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'memory.candidate_ingest.relation_classifier_backend'`

- [ ] **Step 3: 写实现**

```python
# src/memory/candidate_ingest/relation_classifier_backend.py
"""relation_classifier 的薄封装：懒加载 + 线程安全，供 relation_decision 调用。

把 sys.path 接入、backbone 懒加载、并发加锁都隔离在这里；
对外只暴露 classify(old, new) -> 五分类标签字符串。
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_VALID = frozenset({"IND", "EQV", "NSO", "OSN", "CON"})

# <repo>/src/memory/candidate_ingest/relation_classifier_backend.py -> <repo>
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
_RC_DIR = os.path.join(_REPO_ROOT, "relation_classifier")


class RelationClassifierBackend:
    """持有懒加载的 RelationClassifier 单例 + 一把锁。

    RelationClassifier.predict 线程不安全（共享 backbone 前向），
    而调用方用 ThreadPoolExecutor 并发，故 classify 全程持锁。
    构造不加载 backbone；首次 classify 时才加载（双检）。
    """

    def __init__(self, backbone_path: Optional[str] = None,
                 device: Optional[str] = None) -> None:
        self._backbone_path = backbone_path
        self._device = device
        self._clf = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._clf is not None:
            return self._clf
        if _RC_DIR not in sys.path:
            sys.path.insert(0, _RC_DIR)
        from classifier import RelationClassifier  # noqa: E402
        self._clf = RelationClassifier(
            backbone_path=self._backbone_path,
            device=self._device,
        )
        return self._clf

    def classify(self, old: str, new: str) -> str:
        with self._lock:
            clf = self._ensure_loaded()
            try:
                out = clf.predict(old, new, return_probs=False)
            except Exception as exc:  # 单次推理失败回退 IND，不中断整批
                logger.warning("relation classifier predict failed: %s", exc)
                return "IND"
        label = str(out.get("label", "")).strip().upper()
        return label if label in _VALID else "IND"
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /data/zjj/project_26/fact_mem && python -m pytest tests/memory/candidate_ingest/test_relation_classifier_backend.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/memory/candidate_ingest/relation_classifier_backend.py tests/memory/candidate_ingest/test_relation_classifier_backend.py
git commit -m "feat: add RelationClassifierBackend (lazy-loaded, thread-safe)"
```

---

## Task 3: `_classify_relation` 分流 + 语言守卫

**Files:**
- Modify: `src/memory/candidate_ingest/memory_system.py`
- Test: `tests/memory/candidate_ingest/test_classify_relation_dispatch.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/memory/candidate_ingest/test_classify_relation_dispatch.py
import sys
import types
import pytest


def _make_system(monkeypatch, backend, language="en", classify_label="EQV"):
    """构造 LmeCandidateRelationDecisionMemorySystem，绕过真实 backbone / __init__。"""
    from memory.candidate_ingest import memory_system as ms

    # 假 backend：记录是否被调用
    class FakeBackend:
        def __init__(self, *a, **k):
            self.calls = []
        def classify(self, old, new):
            self.calls.append((old, new))
            return classify_label

    monkeypatch.setattr(ms, "RelationClassifierBackend", FakeBackend)

    sysobj = ms.LmeCandidateRelationDecisionMemorySystem.__new__(
        ms.LmeCandidateRelationDecisionMemorySystem
    )
    sysobj.language = language
    sysobj._relation_backend = backend
    sysobj._relation_system_en_template = None
    sysobj._relation_system_zh_template = None
    sysobj._relation_user_template = None
    sysobj._relation_max_new_tokens = 256
    sysobj._rc_backend = FakeBackend() if backend == "classifier" else None

    # 假 llm_client：记录是否被调用
    class FakeLLM:
        def __init__(self):
            self.calls = []
        def get_response_chat(self, *a, **k):
            self.calls.append((a, k))
            return {"relation": "CON"}
    sysobj.llm_client = FakeLLM()
    return sysobj, ms


class _Trace:
    def log_llm_interaction(self, **k): pass


def test_classifier_backend_used(monkeypatch):
    s, ms = _make_system(monkeypatch, "classifier", classify_label="OSN")
    lab = s._classify_relation("old", "new", "scope", _Trace())
    assert lab == "OSN"
    assert s._rc_backend.calls == [("old", "new")]
    assert s.llm_client.calls == []          # 不调 LLM


def test_llm_backend_used(monkeypatch):
    s, ms = _make_system(monkeypatch, "llm")
    lab = s._classify_relation("old", "new", "scope", _Trace())
    assert lab == "CON"
    assert s.llm_client.calls != []          # 调 LLM
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /data/zjj/project_26/fact_mem && python -m pytest tests/memory/candidate_ingest/test_classify_relation_dispatch.py -v`
Expected: FAIL（`_classify_relation` 尚未分流，classifier 用例失败：调用了 LLM / `_rc_backend` 未被使用）

- [ ] **Step 3: 改 import 与构造（memory_system.py 顶部）**

在 `memory_system.py` 现有 `from .relation_decision import ...` 之后加一行：

```python
from .relation_classifier_backend import RelationClassifierBackend
```

在 `LmeCandidateRelationDecisionMemorySystem.__init__` 中，`super().__init__(...)` 之前
（与其它 `kwargs.pop` 放一起，例如紧跟 `self._relation_user_template = ...` 那几行后）加入：

```python
        self._relation_backend = (kwargs.pop("relation_backend", "classifier") or "classifier")
```

在 `super().__init__(*args, **kwargs)` 与 `self.trace = ...` 之间加入语言守卫与 backend 构造：

```python
        if self._relation_backend == "classifier":
            if self.language != "en":
                raise ValueError(
                    "relation_backend='classifier' 只支持英文（language='en'），"
                    f"当前 language={self.language!r}。请用 relation_backend='llm' 或英文输入。"
                )
            self._rc_backend = RelationClassifierBackend()
        else:
            self._rc_backend = None
```

- [ ] **Step 4: 改 `_classify_relation` 分流（memory_system.py:79）**

把 `_classify_relation` 方法体最前面（构造 `system = ...` 之前）插入分类器分支：

```python
        if self._relation_backend == "classifier":
            label = self._rc_backend.classify(m_old_text, m_new)
            trace.log_llm_interaction(
                purpose="lme_candidate_relation_decision_classify_relation",
                messages=[{"role": "user", "content": f"old: {m_old_text}\nnew: {m_new}"}],
                response={"relation": label, "backend": "classifier"},
                scope_id=trace_scope_id,
                metadata={"backend": "classifier"},
            )
            return label if label in _VALID_RELATIONS else "IND"
```

其余（LLM 路径）保持不变。

- [ ] **Step 5: 运行测试确认通过**

Run: `cd /data/zjj/project_26/fact_mem && python -m pytest tests/memory/candidate_ingest/test_classify_relation_dispatch.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add src/memory/candidate_ingest/memory_system.py tests/memory/candidate_ingest/test_classify_relation_dispatch.py
git commit -m "feat: dispatch _classify_relation to classifier backend with language guard"
```

---

## Task 4: 语言守卫测试

**Files:**
- Test: `tests/memory/candidate_ingest/test_classify_relation_dispatch.py`（追加）

- [ ] **Step 1: 追加失败测试**

在该测试文件末尾追加：

```python
def test_language_guard_raises_for_non_english(monkeypatch):
    from memory.candidate_ingest import memory_system as ms

    class FakeBackend:
        def __init__(self, *a, **k): pass
    monkeypatch.setattr(ms, "RelationClassifierBackend", FakeBackend)

    with pytest.raises(ValueError):
        ms.LmeCandidateRelationDecisionMemorySystem(
            embed_model_name="x",
            llm_client=object(),
            language="zh",
            relation_backend="classifier",
        )
```

> 注：该用例走真实 `__init__`。它会调用 `super().__init__`（Mem0/基类）。
> 若基类构造对 `embed_client` 等有强依赖导致无法在守卫前到达，
> 改为直接断言守卫逻辑：把 `language` 检查提取为可单独调用的小函数
> `_check_relation_language(backend, language)` 并对其单测。
> 先按上面写；Step 2 失败信息若指向基类而非 ValueError，再走该回退。

- [ ] **Step 2: 运行**

Run: `cd /data/zjj/project_26/fact_mem && python -m pytest tests/memory/candidate_ingest/test_classify_relation_dispatch.py::test_language_guard_raises_for_non_english -v`
Expected: 守卫先于基类副作用触发则 PASS。若因基类构造报别的错，按 Step 1 注释把守卫抽成 `_check_relation_language` 并改测该函数，再跑至 PASS。

- [ ] **Step 3: Commit**

```bash
git add tests/memory/candidate_ingest/test_classify_relation_dispatch.py src/memory/candidate_ingest/memory_system.py
git commit -m "test: language guard rejects non-english for classifier backend"
```

---

## Task 5: CLI 接线 `--relation-backend`

**Files:**
- Modify: `src/pipeline/ingest_candidates.py`

- [ ] **Step 1: 加参数**

在 `--relation-llm` 的 `parser.add_argument(...)` 块（约 `ingest_candidates.py:385`）之后插入：

```python
    parser.add_argument(
        "--relation-backend",
        dest="relation_backend",
        choices=["classifier", "llm"],
        default="classifier",
        help="relation_decision 成对关系判断后端：classifier=本地 relation_classifier（默认，仅英文）；llm=manager_model",
    )
```

- [ ] **Step 2: 透传进 rel_kw**

在 `ingest_candidates.py` 约 696–701 行的 `rel_kw` 字典中加入一项：

```python
            "relation_backend": args.relation_backend,
```

- [ ] **Step 3: 验证 CLI 解析**

Run: `cd /data/zjj/project_26/fact_mem && python src/pipeline/ingest_candidates.py --help 2>&1 | grep -A2 relation-backend`
Expected: 输出包含 `--relation-backend {classifier,llm}` 帮助文本。

- [ ] **Step 4: Commit**

```bash
git add src/pipeline/ingest_candidates.py
git commit -m "feat: add --relation-backend CLI flag (default classifier)"
```

---

## Task 6: 全量测试 + 可选 smoke

**Files:** 无新增

- [ ] **Step 1: 跑全部新测试**

Run: `cd /data/zjj/project_26/fact_mem && python -m pytest tests/ -v`
Expected: 全部 passed。

- [ ] **Step 2（可选，需 GPU）: 真实 smoke**

Run:
```bash
cd /data/zjj/project_26/fact_mem && CUDA_VISIBLE_DEVICES=1 python -c "
import sys; sys.path.insert(0,'src')
from memory.candidate_ingest.relation_classifier_backend import RelationClassifierBackend
b = RelationClassifierBackend()
print(b.classify('I live in Beijing.', 'I moved to Shanghai.'))  # 期望 CON
"
```
Expected: 打印 `CON`（或合理的五分类标签），确认真实 backbone 串通。
若机器 backbone 路径非默认，需先 `export RC_BACKBONE_PATH=/data/zjj/models/Qwen3-0.6B`。

- [ ] **Step 3: 无新增提交**（本任务仅验证）

---

## Self-Review notes

- **Spec coverage**：语言守卫(Task 3/4)、开关默认 classifier(Task 3/5)、懒加载单例+锁(Task 2)、错误回退 IND(Task 2/3)、CLI 接线(Task 5)、下游不变(未触碰 `_label_candidates`/`decide_*`/`_execute_lme_plan`)，均有任务覆盖。
- **`manager_model` 仍必填**：未改 `--relation-llm` 校验，cascade/deletion 仍依赖它——与 spec 一致。
- **标签集合一致**：backend 与 `memory_system._VALID_RELATIONS` 均为 `{IND,EQV,NSO,OSN,CON}`。
