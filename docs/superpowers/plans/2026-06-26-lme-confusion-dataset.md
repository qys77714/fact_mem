# LME 混淆数据集 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成一个 LME 数据集，每题含原始对话 + golden_memory + lowered_golden + 8 条严格达标的 distractor，存为 JSON。

**Architecture:** 单脚本 `script/build_confusion_dataset.py`：纯逻辑 helper（约束判定/装配/主语校验）可单测；LLM/embedding 依赖部分（lowered 改写、distractor 生成-过滤循环）复用 `pollution_phase3_inject.py` 的 `lower_golden_casual` 与 `verify_seed_answer`，用 `ThreadPoolExecutor` 并发逐题处理，按 `constraint_ok` 分流到主集 / partial，并出 stats。

**Tech Stack:** Python，uv（`uv run --no-sync`），pytest，OpenAI 兼容客户端（`load_api_chat_completion`），qwen3-embedding（`embed_texts`），numpy。

## Global Constraints

- 运行一律 `uv run --no-sync python ...`（裸 python 缺依赖；不带 `--no-sync` 会重 sync）。
- 模型：生成用 `gemma4-26B`（端口 7111），embedding 用 `qwen3-embedding-0.6b`（端口 7110）。密钥经根目录 `.env` + `load_dotenv()`。
- `sys.path` 需含 repo 根与 `src/`（dual import），且含 `script/`（复用 `pollution_phase3_inject`）。
- 三组记忆文本主语统一第三人称 "The user"（硬约束）。
- 每条 distractor 的 `sim_q` 必须 > `min(lowered_golden[*].sim_q)`，逐条判定；且不泄露答案（`verify_seed_answer`）。
- 固定每题 8 条 distractor；lowered_golden 数量 == golden_memory 数量。
- 范围：470 可答题（排除 30 abstention，`golden_memory=[]`）；本次不接入实验跑批。
- sim_q = 归一化 embedding 余弦；`embedding_model` 入库。

---

## File Structure

- Create: `script/build_confusion_dataset.py` — 主脚本（纯 helper + 集成流程 + main）
- Create: `tests/test_confusion_dataset.py` — 纯 helper 单测 + 小集成测试
- Produce: `data/preprocessed/longmemeval_s_confusion.json` / `_partial.json` / `confusion_build_stats.json`
- Reuse (import, 不改): `script/pollution_phase3_inject.py` 的 `lower_golden_casual`、`verify_seed_answer`、`normalize`、`_parse_json_obj`

---

### Task 1: 脚本骨架与数据加载

**Files:**
- Create: `script/build_confusion_dataset.py`
- Test: `tests/test_confusion_dataset.py`

**Interfaces:**
- Produces: `load_sources() -> list[dict]`，每元素 `{"qid","golden_rec","lme_rec"}`，仅含可答题（`golden_memory` 非空），按 `question_id` 对齐 golden 与原始 LME。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_confusion_dataset.py
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REPO, "script"))
import build_confusion_dataset as B

def test_load_sources_answerable_only():
    rows = B.load_sources()
    assert len(rows) == 470                      # 500 - 30 abstention
    r = rows[0]
    assert set(r.keys()) == {"qid", "golden_rec", "lme_rec"}
    # 对齐：golden 与 lme 同一 question_id
    assert r["golden_rec"]["question_id"] == r["lme_rec"]["question_id"]
    # 可答题 golden 非空
    assert r["golden_rec"]["golden_memory"]
    # 原始 LME 字段在位
    assert "haystack_sessions" in r["lme_rec"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /data/zjj/project_26/fact_mem && uv run --no-sync python -m pytest tests/test_confusion_dataset.py::test_load_sources_answerable_only -v`
Expected: FAIL（`ModuleNotFoundError` 或 `AttributeError: load_sources`）

- [ ] **Step 3: 写脚本骨架与 load_sources**

```python
# script/build_confusion_dataset.py
"""为论文主实验生成 LME 混淆数据集：原始对话 + golden + lowered_golden + 8 distractor。
用法：uv run --no-sync python script/build_confusion_dataset.py [--limit N] [--workers 16]
"""
import os, sys, json, re, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
for p in (REPO, os.path.join(REPO, "src"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from utils.embed_utils import embed_texts
from utils.llm_api import load_api_chat_completion
from pollution_phase3_inject import lower_golden_casual, verify_seed_answer, normalize, _parse_json_obj

GOLDEN = os.path.join(REPO, "data/preprocessed/longmemeval_s_golden.json")
RAW_LME = os.path.join(REPO, "data/raw_data/longmemeval_s_cleaned.json")
OUT_MAIN = os.path.join(REPO, "data/preprocessed/longmemeval_s_confusion.json")
OUT_PARTIAL = os.path.join(REPO, "data/preprocessed/longmemeval_s_confusion_partial.json")
OUT_STATS = os.path.join(REPO, "data/preprocessed/confusion_build_stats.json")
EMB_MODEL = "qwen3-embedding-0.6b"
N_DISTRACTORS = 8
MAX_ROUNDS = 6


def load_sources():
    golden = {r["question_id"]: r for r in json.load(open(GOLDEN))}
    lme = {r["question_id"]: r for r in json.load(open(RAW_LME))}
    rows = []
    for qid, grec in golden.items():
        if grec.get("abstention") or not grec.get("golden_memory"):
            continue
        lrec = lme.get(qid)
        if lrec is None:
            continue
        rows.append({"qid": qid, "golden_rec": grec, "lme_rec": lrec})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    print(f"load_sources: {len(load_sources())} answerable questions")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /data/zjj/project_26/fact_mem && uv run --no-sync python -m pytest tests/test_confusion_dataset.py::test_load_sources_answerable_only -v`
Expected: PASS（若数字非 470，按实际 `confusion_build_stats` 口径核对 abstention 计数后修断言——先用 `uv run --no-sync python script/build_confusion_dataset.py` 打印实际可答题数确认）

- [ ] **Step 5: 提交**

```bash
cd /data/zjj/project_26/fact_mem
git add script/build_confusion_dataset.py tests/test_confusion_dataset.py
git commit -m "feat: confusion dataset script skeleton + load_sources"
```

---

### Task 2: 纯逻辑 helper（主语校验 / 约束判定 / 记录装配）

**Files:**
- Modify: `script/build_confusion_dataset.py`
- Test: `tests/test_confusion_dataset.py`

**Interfaces:**
- Produces:
  - `subject_is_user(text: str) -> bool`
  - `compute_constraint_ok(lowered: list[dict], distractors: list[dict], n_required: int = 8) -> bool`
  - `lowered_min_sim(lowered: list[dict]) -> float | None`
  - `assemble_record(lme_rec: dict, golden: list[dict], lowered: list[dict], distractors: list[dict], emb_model: str) -> dict`
  - 约定：golden/lowered/distractor 元素均为 `{"text": str, "sim_q": float, ...}`；lowered 另含 `source_idx`。

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_confusion_dataset.py
def test_subject_is_user():
    assert B.subject_is_user("The user takes yoga classes at Serenity Yoga.")
    assert not B.subject_is_user("You typically attend your yoga sessions downtown.")
    assert not B.subject_is_user("They wrapped up their studies a while back.")
    assert not B.subject_is_user("I graduated with a business degree.")
    assert not B.subject_is_user("Max is a Golden Retriever.")   # 无 user 主语

def test_compute_constraint_ok():
    lowered = [{"text": "a", "sim_q": 0.70}, {"text": "b", "sim_q": 0.75}]
    good = [{"text": f"d{i}", "sim_q": 0.71} for i in range(8)]
    assert B.compute_constraint_ok(lowered, good)               # 全 > 0.70
    bad = good[:7] + [{"text": "d7", "sim_q": 0.70}]            # 一条 == min，不达标
    assert not B.compute_constraint_ok(lowered, bad)
    assert not B.compute_constraint_ok(lowered, good[:7])       # 不足 8 条
    assert not B.compute_constraint_ok([], good)                # 无 lowered

def test_assemble_record():
    lme = {"question_id": "q1", "question": "Q?", "answer": "A",
           "question_type": "t", "question_date": "d",
           "answer_session_ids": [], "haystack_dates": [],
           "haystack_session_ids": [], "haystack_sessions": [[{"role": "user", "content": "x"}]]}
    golden = [{"text": "The user did A.", "sim_q": 0.83}]
    lowered = [{"text": "The user sort of did A.", "sim_q": 0.70, "source_idx": 0}]
    dist = [{"text": f"The user did X{i}.", "sim_q": 0.72} for i in range(8)]
    rec = B.assemble_record(lme, golden, lowered, dist, "qwen3-embedding-0.6b")
    assert rec["question_id"] == "q1"
    assert rec["haystack_sessions"] == lme["haystack_sessions"]   # 原始对话保留
    assert rec["golden_memory"] == golden
    assert rec["lowered_golden"] == lowered
    assert rec["distractors"] == dist
    assert rec["embedding_model"] == "qwen3-embedding-0.6b"
    assert rec["lowered_golden_min_sim"] == 0.70
    assert rec["constraint_ok"] is True
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /data/zjj/project_26/fact_mem && uv run --no-sync python -m pytest tests/test_confusion_dataset.py -k "subject or constraint or assemble" -v`
Expected: FAIL（`AttributeError`）

- [ ] **Step 3: 实现 helper**

```python
# 追加到 script/build_confusion_dataset.py（load_sources 之后）
_BANNED_LEADING = {"i", "i'm", "i've", "you", "your", "you're", "they", "we", "he", "she", "my", "me", "his", "her"}

def subject_is_user(text):
    t = (text or "").strip()
    if not t:
        return False
    first = re.split(r"[\s,]+", t.lower(), maxsplit=1)[0].strip(".,'\"")
    if first in _BANNED_LEADING:
        return False
    return "user" in t.lower()

def lowered_min_sim(lowered):
    if not lowered:
        return None
    return min(l["sim_q"] for l in lowered)

def compute_constraint_ok(lowered, distractors, n_required=N_DISTRACTORS):
    if not lowered or len(distractors) != n_required:
        return False
    lo = lowered_min_sim(lowered)
    return all(d["sim_q"] > lo for d in distractors)

_LME_FIELDS = ("question_id", "question_type", "question", "question_date", "answer",
               "answer_session_ids", "haystack_dates", "haystack_session_ids", "haystack_sessions")

def assemble_record(lme_rec, golden, lowered, distractors, emb_model):
    rec = {k: lme_rec.get(k) for k in _LME_FIELDS}
    rec["golden_memory"] = golden
    rec["lowered_golden"] = lowered
    rec["distractors"] = distractors
    rec["embedding_model"] = emb_model
    rec["lowered_golden_min_sim"] = lowered_min_sim(lowered)
    rec["constraint_ok"] = compute_constraint_ok(lowered, distractors)
    return rec
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /data/zjj/project_26/fact_mem && uv run --no-sync python -m pytest tests/test_confusion_dataset.py -k "subject or constraint or assemble" -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
cd /data/zjj/project_26/fact_mem
git add script/build_confusion_dataset.py tests/test_confusion_dataset.py
git commit -m "feat: pure helpers for confusion dataset (subject/constraint/assemble)"
```

---

### Task 3: embedding 包装与 sim 计算

**Files:**
- Modify: `script/build_confusion_dataset.py`
- Test: `tests/test_confusion_dataset.py`

**Interfaces:**
- Consumes: `OpenAI` embedding client（`EMBEDDING_BASE_URL`）。
- Produces:
  - `make_emb_client() -> OpenAI`
  - `embed_norm(emb_client, texts: list[str]) -> np.ndarray`（归一化，shape `(n, d)`）
  - `sim_to_q(vecs: np.ndarray, q_vec: np.ndarray) -> list[float]`

- [ ] **Step 1: 写集成测试（需 embedding 服务在线）**

```python
# 追加到 tests/test_confusion_dataset.py
import numpy as np

def test_embed_and_sim():
    emb = B.make_emb_client()
    q = B.embed_norm(emb, ["Where does the user do yoga?"])[0]
    vecs = B.embed_norm(emb, ["The user does yoga at Serenity Yoga.",
                              "The user enjoys cooking pasta on weekends."])
    sims = B.sim_to_q(vecs, q)
    assert len(sims) == 2
    assert all(-1.0 <= s <= 1.0 for s in sims)
    assert sims[0] > sims[1]              # 同话题更相似
    assert abs(np.linalg.norm(vecs[0]) - 1.0) < 1e-5   # 已归一化
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /data/zjj/project_26/fact_mem && uv run --no-sync python -m pytest tests/test_confusion_dataset.py::test_embed_and_sim -v`
Expected: FAIL（`AttributeError: make_emb_client`）

- [ ] **Step 3: 实现**

```python
# 追加到 script/build_confusion_dataset.py
def make_emb_client():
    return OpenAI(api_key=os.getenv("EMBEDDING_API_KEY", "EMPTY"),
                  base_url=os.getenv("EMBEDDING_BASE_URL", "http://localhost:7110/v1/"))

def embed_norm(emb_client, texts):
    return normalize(np.asarray(embed_texts(emb_client, texts, EMB_MODEL), dtype=float))

def sim_to_q(vecs, q_vec):
    return [float(v @ q_vec) for v in vecs]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /data/zjj/project_26/fact_mem && uv run --no-sync python -m pytest tests/test_confusion_dataset.py::test_embed_and_sim -v`
Expected: PASS（若服务未起：先 `bash script/0_run_embedding.sh` 后台启动）

- [ ] **Step 5: 提交**

```bash
cd /data/zjj/project_26/fact_mem
git add script/build_confusion_dataset.py tests/test_confusion_dataset.py
git commit -m "feat: embedding client + sim helpers for confusion dataset"
```

---

### Task 4: distractor 生成-过滤循环（核心）

**Files:**
- Modify: `script/build_confusion_dataset.py`
- Test: `tests/test_confusion_dataset.py`

**Interfaces:**
- Consumes: `gen_client`（`load_api_chat_completion("gemma4-26B")`）、`emb_client`、`q_vec`、`lowered_min`。
- Produces: `generate_distractors(gen_client, emb_client, question, correct_answer, q_vec, lowered_min, existing_vecs=None) -> list[dict]`，返回**最多** 8 条 `{"text","sim_q"}`；每条满足 `sim_q > lowered_min`、主语为 user、`verify_seed_answer` 通过、与已选项 cos < 0.97。

- [ ] **Step 1: 写集成测试（小样本，需两服务在线）**

```python
# 追加到 tests/test_confusion_dataset.py
def test_generate_distractors_constraint():
    gen = B.load_api_chat_completion("gemma4-26B")
    emb = B.make_emb_client()
    question = "Where does the user take yoga classes?"
    q = B.embed_norm(emb, [question])[0]
    lowered_min = 0.55   # 故意设低，便于在小预算内凑齐
    dists = B.generate_distractors(gen, emb, question, "Serenity Yoga", q, lowered_min)
    assert len(dists) <= 8
    for d in dists:
        assert d["sim_q"] > lowered_min
        assert B.subject_is_user(d["text"])
    texts = [d["text"] for d in dists]
    assert len(set(texts)) == len(texts)        # 无完全重复
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /data/zjj/project_26/fact_mem && uv run --no-sync python -m pytest tests/test_confusion_dataset.py::test_generate_distractors_constraint -v`
Expected: FAIL（`AttributeError: generate_distractors`）

- [ ] **Step 3: 实现 prompt + 循环**

```python
# 追加到 script/build_confusion_dataset.py
DISTRACTOR_PROMPT = (
    "A user asked the following question in a conversation. Create ONE distractor memory "
    "for adversarial robustness testing — a plausible factual statement ABOUT THE USER that:\n"
    "1. TOPIC: same topic domain as the question, so a retriever embeds it near the answer.\n"
    "2. WRONG VALUE: asserts a DIFFERENT, INCORRECT value than the true answer.\n"
    "3. DIFFERENT STRUCTURE: rephrase entirely, do not just swap one word.\n"
    "4. SUBJECT: the sentence MUST be third person with subject \"The user\" "
    "(never \"I\", \"you\", \"they\").\n"
    "5. NATURAL: sound like a real user memory.\n"
    "{avoid}\n"
    "Question: {question}\n\n"
    "Return ONLY the statement, one sentence starting with \"The user\"."
)
DEDUP_COS = 0.97

def generate_distractors(gen_client, emb_client, question, correct_answer,
                         q_vec, lowered_min, existing_vecs=None):
    picked = []
    picked_vecs = [] if existing_vecs is None else list(existing_vecs)
    for _round in range(MAX_ROUNDS):
        if len(picked) >= N_DISTRACTORS:
            break
        avoid = ""
        if picked:
            avoid = "Avoid repeating these existing memories:\n- " + "\n- ".join(
                d["text"] for d in picked) + "\n"
        resp = gen_client.get_response_chat(
            [{"role": "user", "content": DISTRACTOR_PROMPT.format(question=question, avoid=avoid)}],
            max_new_tokens=128, temperature=0.9,
        )
        text = (resp or "").strip().strip('"').strip("'")
        if not text or len(text) < 15 or not subject_is_user(text):
            continue
        vec = embed_norm(emb_client, [text])[0]
        sim_q = float(vec @ q_vec)
        if sim_q <= lowered_min:
            continue
        if any(float(vec @ pv) >= DEDUP_COS for pv in picked_vecs):
            continue
        ok, _info = verify_seed_answer(gen_client, question, text, correct_answer)
        if not ok:
            continue
        picked.append({"text": text, "sim_q": round(sim_q, 5)})
        picked_vecs.append(vec)
    return picked
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /data/zjj/project_26/fact_mem && uv run --no-sync python -m pytest tests/test_confusion_dataset.py::test_generate_distractors_constraint -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
cd /data/zjj/project_26/fact_mem
git add script/build_confusion_dataset.py tests/test_confusion_dataset.py
git commit -m "feat: distractor generate-filter loop (per-item sim + no-leak + user-subject)"
```

---

### Task 5: 单题编排（lowered + distractor → 记录）

**Files:**
- Modify: `script/build_confusion_dataset.py`
- Test: `tests/test_confusion_dataset.py`

**Interfaces:**
- Consumes: Task 2–4 全部 helper、`lower_golden_casual`（复用）。
- Produces: `process_question(row, gen_client, emb_client) -> dict`，返回 `assemble_record(...)` 的结果（含 `constraint_ok`）。
  - lowered：对每条 `golden_memory` 调 `lower_golden_casual` 得改写文本，连同重算的 `sim_q` 与 `source_idx` 组成 list。
  - distractor：以 `lowered_min_sim` 为基准调 `generate_distractors`。

- [ ] **Step 1: 写集成测试（--limit 行为，单题）**

```python
# 追加到 tests/test_confusion_dataset.py
def test_process_question_shapes():
    gen = B.load_api_chat_completion("gemma4-26B")
    emb = B.make_emb_client()
    row = B.load_sources()[0]
    rec = B.process_question(row, gen, emb)
    assert len(rec["lowered_golden"]) == len(rec["golden_memory"])
    assert all("source_idx" in l and "sim_q" in l for l in rec["lowered_golden"])
    # lowered sim_q 不高于对应 golden（尽量降）
    for l in rec["lowered_golden"]:
        assert l["sim_q"] <= rec["golden_memory"][l["source_idx"]]["sim_q"] + 1e-6
    assert isinstance(rec["constraint_ok"], bool)
    if rec["constraint_ok"]:
        assert len(rec["distractors"]) == 8
        lo = rec["lowered_golden_min_sim"]
        assert all(d["sim_q"] > lo for d in rec["distractors"])
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /data/zjj/project_26/fact_mem && uv run --no-sync python -m pytest tests/test_confusion_dataset.py::test_process_question_shapes -v`
Expected: FAIL（`AttributeError: process_question`）

- [ ] **Step 3: 实现**

```python
# 追加到 script/build_confusion_dataset.py
def _golden_with_sim(emb_client, goldens, q_vec):
    vecs = embed_norm(emb_client, goldens)
    sims = sim_to_q(vecs, q_vec)
    return [{"text": g, "sim_q": round(s, 5)} for g, s in zip(goldens, sims)]

def process_question(row, gen_client, emb_client):
    grec, lme_rec = row["golden_rec"], row["lme_rec"]
    question = grec["question"]
    correct = grec.get("answer", "")
    goldens = grec["golden_memory"]
    q_vec = embed_norm(emb_client, [question])[0]

    golden = _golden_with_sim(emb_client, goldens, q_vec)

    low = lower_golden_casual(gen_client, emb_client, question, goldens, correct, q_vec)
    lowered = []
    for idx, (text, vec) in enumerate(zip(low["lowered"], low["emb"])):
        lowered.append({"text": text, "sim_q": round(float(vec @ q_vec), 5), "source_idx": idx})

    lo = lowered_min_sim(lowered)
    dists = generate_distractors(gen_client, emb_client, question, correct, q_vec, lo)

    return assemble_record(lme_rec, golden, lowered, dists, EMB_MODEL)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /data/zjj/project_26/fact_mem && uv run --no-sync python -m pytest tests/test_confusion_dataset.py::test_process_question_shapes -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
cd /data/zjj/project_26/fact_mem
git add script/build_confusion_dataset.py tests/test_confusion_dataset.py
git commit -m "feat: per-question orchestration (lowered + distractors -> record)"
```

---

### Task 6: main 并发跑批、分流落盘、stats

**Files:**
- Modify: `script/build_confusion_dataset.py`
- Test: `tests/test_confusion_dataset.py`

**Interfaces:**
- Produces: `run_build(limit=0, workers=16) -> dict`（stats），写三个产物文件。main 块调用之。
  - 主集：`constraint_ok=true`；partial：其余；stats：题数/类型分布/排除原因。

- [ ] **Step 1: 写集成测试（--limit 3 端到端）**

```python
# 追加到 tests/test_confusion_dataset.py
def test_run_build_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "OUT_MAIN", str(tmp_path / "main.json"))
    monkeypatch.setattr(B, "OUT_PARTIAL", str(tmp_path / "partial.json"))
    monkeypatch.setattr(B, "OUT_STATS", str(tmp_path / "stats.json"))
    stats = B.run_build(limit=3, workers=3)
    main = json.load(open(B.OUT_MAIN))
    partial = json.load(open(B.OUT_PARTIAL))
    assert stats["processed"] == 3
    assert stats["main"] + stats["partial"] == 3
    assert len(main) == stats["main"] and len(partial) == stats["partial"]
    for rec in main:
        assert rec["constraint_ok"] is True
        assert len(rec["distractors"]) == 8
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /data/zjj/project_26/fact_mem && uv run --no-sync python -m pytest tests/test_confusion_dataset.py::test_run_build_limit -v`
Expected: FAIL（`AttributeError: run_build`）

- [ ] **Step 3: 实现 run_build + main**

```python
# 追加到 script/build_confusion_dataset.py（替换原 __main__ 块）
from collections import Counter

def run_build(limit=0, workers=16):
    rows = load_sources()
    if limit:
        rows = rows[:limit]
    gen = load_api_chat_completion("gemma4-26B")
    emb = make_emb_client()

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_question, row, gen, emb): row["qid"] for row in rows}
        for i, fut in enumerate(as_completed(futs), 1):
            qid = futs[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"[{i}/{len(rows)}] {qid} FAILED: {e}")
            if i % 20 == 0:
                print(f"  {i}/{len(rows)} done")

    main = [r for r in results if r["constraint_ok"]]
    partial = [r for r in results if not r["constraint_ok"]]
    json.dump(main, open(OUT_MAIN, "w"), ensure_ascii=False, indent=1)
    json.dump(partial, open(OUT_PARTIAL, "w"), ensure_ascii=False, indent=1)

    stats = {
        "processed": len(results),
        "main": len(main),
        "partial": len(partial),
        "main_by_type": dict(Counter(r["question_type"] for r in main)),
        "partial_by_type": dict(Counter(r["question_type"] for r in partial)),
        "partial_reason": {
            "lt8_distractors": sum(1 for r in partial if len(r["distractors"]) < 8),
            "no_lowered": sum(1 for r in partial if not r["lowered_golden"]),
        },
    }
    json.dump(stats, open(OUT_STATS, "w"), ensure_ascii=False, indent=1)
    print(f"main={len(main)} partial={len(partial)} -> {OUT_MAIN}")
    return stats

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    run_build(limit=args.limit, workers=args.workers)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /data/zjj/project_26/fact_mem && uv run --no-sync python -m pytest tests/test_confusion_dataset.py::test_run_build_limit -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
cd /data/zjj/project_26/fact_mem
git add script/build_confusion_dataset.py tests/test_confusion_dataset.py
git commit -m "feat: run_build concurrency + main/partial split + stats"
```

---

### Task 7: 全量跑批与验收

**Files:**
- Produce: `data/preprocessed/longmemeval_s_confusion.json` / `_partial.json` / `confusion_build_stats.json`

- [ ] **Step 1: 确认两服务在线**

Run: `curl -s http://localhost:7110/v1/models -H "Authorization: Bearer zjj" | head -c 80; echo; curl -s http://localhost:7111/v1/models -H "Authorization: Bearer zjj" | head -c 80`
Expected: 两者各返回 models JSON（未起则 `bash script/0_run_embedding.sh` / `bash script/0_run_model.sh` 后台启动并等待就绪）

- [ ] **Step 2: 全量跑（后台，耗时较长）**

Run: `cd /data/zjj/project_26/fact_mem && uv run --no-sync python script/build_confusion_dataset.py --workers 16`
Expected: 末行 `main=<N> partial=<M> -> .../longmemeval_s_confusion.json`

- [ ] **Step 3: 验收脚本（验证 §6 验收标准）**

```bash
cd /data/zjj/project_26/fact_mem && uv run --no-sync python -c "
import json
m=json.load(open('data/preprocessed/longmemeval_s_confusion.json'))
print('主集题数:',len(m))
bad=0
for r in m:
    lo=r['lowered_golden_min_sim']
    assert len(r['lowered_golden'])==len(r['golden_memory'])
    assert len(r['distractors'])==8
    assert all(d['sim_q']>lo for d in r['distractors'])
    assert r['constraint_ok'] is True
    # 主语抽查
    for grp in ('golden_memory','lowered_golden','distractors'):
        for x in r[grp]:
            if 'user' not in x['text'].lower(): bad+=1
print('主语缺 user 的条目:',bad)
print('验收通过' if bad==0 else '需检查主语')
"
```
Expected: `主语缺 user 的条目: 0` + `验收通过`；主集题数与 `confusion_build_stats.json` 一致

- [ ] **Step 4: 提交产物**

```bash
cd /data/zjj/project_26/fact_mem
git add data/preprocessed/longmemeval_s_confusion.json data/preprocessed/longmemeval_s_confusion_partial.json data/preprocessed/confusion_build_stats.json
git commit -m "data: build LME confusion dataset (golden+lowered+8 distractors)"
```

---

## Self-Review

**Spec coverage：**
- §2 schema → Task 2 `assemble_record`（字段齐全）+ Task 7 验收。
- §2 主语约束 → Task 2 `subject_is_user` + Task 4 prompt + Task 7 抽查。
- §3 灌库语义（lowered+N）→ 记录约定，本次不实现（§7 非目标），计划不含跑批，符合。
- §4 Step1 lowered → Task 5 `lower_golden_casual` 复用；Step2 distractor → Task 4；Step3 装配 → Task 5/6。
- §4 范围 470 可答题 → Task 1 `load_sources`。
- §5 三产物 → Task 6 `run_build`。
- §6 验收 1–5 → Task 7 验收脚本（含数量/约束/主语/stats 一致）。
- 注：§2 `date` 字段——计划当前未写入 distractor/lowered 的 `date`（Task 5 留了占位注释）。**修正**：日期对本次「仅交付数据集、不接入灌库」非必需（§3 标注日期仅用于未来灌库顺序），故 Task 中不实现 `date` 以避免 YAGNI；若后续接入实验再补。已与 §7「不接入实验」一致，无遗漏功能。

**Placeholder 扫描：** 已删除 Task 5 Step3 的无用占位行；其余步骤均含完整可执行代码，无 TBD/TODO。

**类型一致：** `subject_is_user/compute_constraint_ok/assemble_record/embed_norm/sim_to_q/generate_distractors/process_question/run_build` 命名在各 Task 间一致；元素结构 `{"text","sim_q",...}` 统一。
