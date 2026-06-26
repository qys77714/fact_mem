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
