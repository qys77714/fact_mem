#!/usr/bin/env python
"""
构建 LME 混淆数据集 (golden + lowered golden + distractors)。

用法：
  PYTHONPATH=src uv run --no-sync python script/build_confusion_dataset.py
    [--limit N]           仅处理前 N 题（调试）
    [--out-dir DIR]       输出目录（默认 data/preprocessed）
    [--resume]            断点续跑（跳过 valid 题）
    [--golden PATH]       golden 文件路径
    [--raw-lme PATH]      原始 LME 文件路径
"""
import os, sys, json, time, argparse, statistics as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

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
DEFAULT_RAW_LME = os.path.join(REPO, "data/raw_data/longmemeval_s_cleaned.json")
DEFAULT_OUT_DIR = os.path.join(REPO, "data/preprocessed")

OUT_MAIN = "longmemeval_s_confusion.json"
OUT_PARTIAL = "longmemeval_s_confusion_partial.json"
OUT_STATS = "confusion_build_stats.json"

# ---- helper ----

def normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n

def emb_similarity(emb_client, texts: list[str], query_emb: np.ndarray) -> np.ndarray:
    """返回 texts 与 query_emb 的余弦相似度 (n,)"""
    if not texts:
        return np.array([])
    embs = normalize(embed_texts(emb_client, texts, EMB_MODEL))
    return (embs @ query_emb).flatten()

def _parse_json_obj(text):
    """从 LLM 回复中提取 JSON 对象，失败返回 None"""
    import re
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# ---- 数据加载 ----

def load_sources(golden_path=DEFAULT_GOLDEN, raw_lme_path=DEFAULT_RAW_LME):
    """加载 golden memory 和原始 LME 数据，按 question_id 对齐。
    返回 (questions, raw_lme_map):
      questions: list[dict]  可答题 (golden_memory 非空, abstention 排除)
      raw_lme_map: dict[qid → 原始 LME 记录]
    """
    golden_data = json.load(open(golden_path))
    raw_lme_data = json.load(open(raw_lme_path))

    raw_lme_map = {r["question_id"]: r for r in raw_lme_data}

    questions = []
    skipped = 0
    for r in golden_data:
        qid = r["question_id"]
        if r.get("abstention") or not r.get("golden_memory"):
            skipped += 1
            continue
        if qid not in raw_lme_map:
            skipped += 1
            continue
        questions.append(r)

    print(f"[load] 可答题={len(questions)}, 排除(abstention/无golden/缺raw)={skipped}")
    return questions, raw_lme_map


# ---- CLI ----

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--golden", default=DEFAULT_GOLDEN)
    ap.add_argument("--raw-lme", default=DEFAULT_RAW_LME)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    questions, raw_lme_map = load_sources(args.golden, args.raw_lme)
    if args.limit:
        questions = questions[:args.limit]
        print(f"[main] 限制只处理前 {args.limit} 题")
    # — 占位: 后续任务在此扩展 —
    print("[main] 数据加载 OK, 后续任务衔接此脚本")
