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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=DEFAULT_GOLDEN)
    ap.add_argument("--raw", default=DEFAULT_RAW)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out-name", default="longmemeval_s_confusion_v3")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"[confusion_v3] out_dir={args.out_dir}, max_workers={args.max_workers}, resume={args.resume}")
