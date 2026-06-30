#!/usr/bin/env python
"""
全量生成 confusion memory 数据集（v3 流程）。

流程:
  Step 0 — 加载 golden + 原始 LME，计算 sim_q，提取非 evidence session 日期
  Step 1 — 逐条 lowering golden（每题型独立 prompt + 替换验证 + 逐条 fallback）
  Step 2 — 多策略种子生成（4 prompt × 并行 batch）
  Step 3 — EQV/OSN/NSO 改写扩充
  Step 4 — 装配，落盘 JSON

用法:
  PYTHONPATH=src uv run --no-sync python script/build_confusion_v3.py \
    [--golden PATH] [--raw PATH] [--resume] [--max-workers N]
"""

import os, sys, json, time, re, argparse, textwrap, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import datetime

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO, "src"))

from utils.embed_utils import embed_texts
from utils.llm_api import load_api_chat_completion

# ---- 常量 ----
GEN_MODEL = "gemma4-26B"
EMB_MODEL = "qwen3-embedding-0.6b"
EMB_API_KEY = os.getenv("EMBEDDING_API_KEY", "EMPTY")
EMB_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:7110/v1/")

DEFAULT_GOLDEN = os.path.join(REPO, "data/preprocessed/longmemeval_s_golden.json")
DEFAULT_RAW = os.path.join(REPO, "data/raw_data/longmemeval_s_cleaned.json")
DEFAULT_OUT_DIR = os.path.join(REPO, "data/preprocessed")

TARGET_DISTRACTORS = 8
TARGET_SEEDS = 4
SEED_MAX_ROUNDS = 8
SEED_BATCH_SIZE = 6        # 每策略每轮 6 条
LOWER_MAX_ATTEMPTS = 8     # 每条 golden 最多 8 次 lowering
REWRITE_MAX_ATTEMPTS = 2
LOWER_SIM_LOWER = 0.4      # sim(g_i) - 0.4
LOWER_SIM_UPPER = 0.05     # sim(g_i) - 0.05
GATE_OFFSET = 0.1          # anchor_sim - 0.1

# ---- Lowering Prompts (per question type) ----
# 每条 prompt 是一个模板，用 {golden} 填充原始 golden 文本

LOWERING_PROMPTS = {
    "temporal-reasoning": textwrap.dedent("""\
Rewrite this statement VERY SLIGHTLY — change only the sentence STRUCTURE.
🔴 ALL dates, numbers, and time expressions must remain CHARACTER-BY-CHARACTER identical.
If you change ANY date, time, or number, the answer becomes WRONG and the task FAILS.

Rules:
1. 🔴 ALL dates, times, and numbers MUST be copied verbatim — not a single character changed
2. ALL temporal relationship words (before, after, until, since, between) MUST be preserved exactly
3. Only allowed changes: reorder clauses, change active↔passive, add casual framing words
4. Keep subject as "The user"
5. One sentence

Original: {golden}

Return ONLY the rewritten sentence."""),

    "multi-session": textwrap.dedent("""\
Rewrite this statement to be more casual and indirect, while preserving ALL factual information.
This fact is part of a MULTI-SESSION set — numbers, dates, and event sequences must stay EXACT.

Rules:
1. ALL numbers, numerical values, and count-related words MUST remain EXACTLY as written
2. ALL dates, times, proper nouns (names of people/places/things) MUST remain EXACTLY as written
3. Preserve the temporal/sequential relationship implied by the original
4. Rephrase as a casual conversational memory
5. Keep subject as "The user"
6. One sentence

Original: {golden}

Return ONLY the rewritten sentence."""),

    "knowledge-update": textwrap.dedent("""\
Rewrite this statement to be more casual and indirect. This fact is part of a KNOWLEDGE-UPDATE set.
It tracks a change over time — both the old value and the context around it must be preserved.

Rules:
1. ALL values (old and new), numbers, dates, and time markers MUST remain EXACTLY as written
2. Keep the temporal context (e.g., "as of DATE", "by DATE") verbatim
3. Make the phrasing more casual — less like a database log
4. Keep subject as "The user"
5. One sentence

Original: {golden}

Return ONLY the rewritten sentence."""),

    "single-session-user": textwrap.dedent("""\
Rewrite this statement to be more casual and indirect, while preserving ALL factual information.

Rules:
1. ALL names of people/places/things, numbers, dates, and proper nouns MUST remain EXACTLY as written
2. Common verbs and adjectives can be replaced with synonyms (e.g., "attended"→"went to")
3. Restructure the sentence — reorder clauses, change active↔passive
4. Add casual framing words ("happened to", "noted that", "ended up")
5. Keep subject as "The user"
6. One sentence

Original: {golden}

Return ONLY the rewritten sentence."""),

    "single-session-assistant": textwrap.dedent("""\
Rewrite this statement to be more casual and indirect. This fact comes from an ASSISTANT interaction.

Rules:
1. ALL key entities, facts, and values MUST remain EXACTLY as written
2. Only change: sentence connectors, function words (the/a/of), and common verbs
3. You may reorder clauses and change active↔passive
4. Keep subject as "The user" (convert from "you" if needed)
5. One sentence

Original: {golden}

Return ONLY the rewritten sentence."""),

    "single-session-preference": textwrap.dedent("""\
Rewrite this statement to be more casual and indirect. This fact expresses a PREFERENCE.

Rules:
1. The PREFERENCE OBJECT (what the user likes/dislikes) MUST remain EXACTLY as written — do not replace it
2. Intensity words (love, enjoy, like, prefer, hate) can be softened slightly but not changed to opposite meaning
3. Make phrasing more conversational and less like a direct statement of preference
4. Keep subject as "The user"
5. One sentence

Original: {golden}

Return ONLY the rewritten sentence."""),
}


def get_lowering_prompt(question_type):
    """根据题型返回对应的 lowering prompt 模板。"""
    return LOWERING_PROMPTS.get(question_type, LOWERING_PROMPTS["single-session-user"])


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=DEFAULT_GOLDEN)
    ap.add_argument("--raw", default=DEFAULT_RAW)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out-name", default="longmemeval_s_confusion_v3")
    return ap.parse_args()


# ============================================================
# 帮助函数
# ============================================================


def normalize(m: np.ndarray) -> np.ndarray:
    """L2 归一化（按行），零向量防止除零。"""
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def parse_json_obj(text):
    """从文本中提取第一个 JSON 对象。"""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def extract_json_array(text):
    """从文本中提取第一个 JSON 数组。"""
    if not text:
        return None
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def batch_embed(emb_client, texts):
    """批量 embedding → L2 归一化矩阵。"""
    if not texts:
        return np.array([])
    embs = embed_texts(emb_client, texts, EMB_MODEL)
    return normalize(np.array(embs))


# ============================================================
# 数据加载
# ============================================================


def load_data(golden_path=DEFAULT_GOLDEN, raw_path=DEFAULT_RAW):
    """加载 golden 和 raw 数据，返回待处理题目列表 + raw map。

    golden_memory 格式：[{"content": "...", "date": "..."}, ...]
    跳过 abstention 和无 golden_memory 的题。
    """
    golden_data = json.load(open(golden_path))
    raw_data = json.load(open(raw_path))
    raw_map = {r["question_id"]: r for r in raw_data}

    questions = []
    skipped_abstention = 0
    for r in golden_data:
        qid = r["question_id"]
        if r.get("abstention") or not r.get("golden_memory"):
            skipped_abstention += 1
            continue
        if qid not in raw_map:
            continue

        raw_rec = raw_map[qid]
        questions.append({
            "question_id": qid,
            "question": r["question"],
            "answer": r.get("answer", ""),
            "question_type": r.get("question_type", ""),
            "question_date": raw_rec.get("question_date", ""),
            "judged_correct": r.get("judged_correct", True),
            "golden_memory": r["golden_memory"],
            # 已经是 [{"content": "...", "date": "..."}, ...]
            "raw_rec": raw_rec,
        })

    n_jc_false = sum(1 for q in questions if not q["judged_correct"])
    print(f"[load] 可答题={len(questions)}, abstention跳过={skipped_abstention}, "
          f"judged_correct=false={n_jc_false}")
    return questions, raw_map


if __name__ == "__main__":
    args = parse_args()
    print(f"[confusion_v3] out_dir={args.out_dir}, max_workers={args.max_workers}, resume={args.resume}")
