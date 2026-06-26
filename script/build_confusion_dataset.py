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


# ============================================================
# Step 1: lowered golden — 保证答案等价的前提下尽量降低 sim_q
# ============================================================

CASUAL_REWRITE_PROMPT = (
    "Rewrite the following factual statement about a user so that it CASUALLY and INDIRECTLY "
    "reveals the same factual information, rather than stating it directly. The fact should "
    "still be derivable by a reader, but the sentence should feel like a natural, incidental "
    "mention, not a direct answer to a question.\n\n"
    "Rules:\n"
    "1. The SAME factual information MUST still be present (numbers, names, dates unchanged)\n"
    "2. Rephrase as a casual memory, story fragment, or indirect mention\n"
    "3. Do NOT structure the sentence as a direct answer\n"
    "4. Keep the subject as \"The user\" (NOT \"I\" or first-person)\n"
    "5. Use conversational, natural language\n\n"
    "Original: {golden}\n\n"
    "Return ONLY the rewritten statement, one sentence."
)

SIMPLIFY_GOLDEN_PROMPT = (
    "Simplify the following factual statement about a user by removing non-essential details "
    "such as specific dates, times, locations, quantities, and modifiers. Keep ONLY the core "
    "factual information. The simplified version should be shorter, more generic, and NOT sound "
    "like a direct answer to any specific question.\n\n"
    "Rules:\n"
    "1. Keep the SAME core fact (e.g., who did what, what the key attribute/value is)\n"
    "2. Remove: specific dates (MM/DD/YYYY), times, locations unless they ARE the answer\n"
    "3. Remove: quantity modifiers, adjectives that don't change the core fact\n"
    "4. Keep the subject as \"The user\"\n"
    "5. Make it sound like a neutral database entry, not a conversational sentence\n\n"
    "Original: {golden}\n\n"
    "Return ONLY the simplified statement, one sentence."
)

NARRATIVE_EMBED_PROMPT = (
    "Rewrite the following factual statement about a user so the fact is mentioned only "
    "as a BACKGROUND DETAIL within a DIFFERENT anecdote or everyday observation. "
    "The main topic of the sentence should NOT be the factual claim itself — "
    "instead, bury it inside a broader narrative about a different subject.\n\n"
    "Rules:\n"
    "1. The SAME factual information MUST still be present (numbers, names, dates unchanged)\n"
    "2. The sentence's main topic must be something OTHER than what the fact states\n"
    "3. Frame it as reminiscing, a past habit, or a recent observation\n"
    "4. Keep the subject as \"The user\" (NOT \"I\" or first-person)\n"
    "5. Use conversational, natural language, one sentence\n\n"
    "Original: {golden}\n\n"
    "Return ONLY the rewritten statement, one sentence."
)

VOCAB_SHIFT_PROMPT = (
    "Rewrite the following factual statement about a user using COMPLETELY DIFFERENT vocabulary "
    "and sentence structure, while preserving ALL factual information (numbers, names, dates).\n\n"
    "Rules:\n"
    "1. Change EVERY noun, verb, and modifier to a synonym or near-synonym\n"
    "2. Restructure the sentence entirely (change from active to passive, change clause order, etc.)\n"
    "3. Avoid any direct quotes or formulaic phrasing\n"
    "4. **CRITICAL**: The sentence MUST begin with exactly \"The user\" followed by a verb. "
    "No leading clauses, no introductory phrases, no \"While...\", no \"Although...\", no \"Having...\". "
    "Start: 'The user ...'\n"
    "5. The rewritten version must still clearly convey the same fact\n"
    "6. Use natural, conversational language\n\n"
    "Original: {golden}\n\n"
    "Return ONLY the rewritten statement, one sentence that starts with 'The user'."
)

ANSWER_FROM_MEMORY_PROMPT = (
    "Based on the following memory about a user, answer the question.\n"
    "If the memory contains the answer (even indirectly), give it. "
    "If not, say 'NO ANSWER'.\n\n"
    "Memory: {memory}\nQuestion: {question}\n\n"
    "Return ONLY a JSON object: {{\"answer\": \"<your answer or NO ANSWER>\"}}"
)

SAME_ANSWER_PROMPT = (
    "Do these two answers convey the SAME factual information?\n"
    "Answer A: {answer_a}\nAnswer B: {answer_b}\n\n"
    'Return ONLY a JSON object: {{"same": true/false}}'
)

GOLDEN_LOWER_MAX_ATTEMPTS = 8


def _try_lower_one(gen_client, emb_client, g_text, question, correct_answer, q_emb,
                   orig_emb, orig_q, prompts_and_temps):
    """尝试多种改写策略，保留 sim_q 最低且答案等价的版本。
    返回 (best_text, best_emb, best_q)
    """
    best_text = g_text
    best_q = orig_q
    best_emb = orig_emb

    for prompt_tpl, temp in prompts_and_temps:
        for _ in range(GOLDEN_LOWER_MAX_ATTEMPTS):
            resp = gen_client.get_response_chat(
                [{"role": "user", "content": prompt_tpl.format(golden=g_text)}],
                max_new_tokens=256, temperature=temp,
            )
            rewritten = (resp or "").strip().strip('"').strip("'")
            if not rewritten or len(rewritten) < 12 or rewritten == g_text:
                continue
            # 强制 "The user" 开头
            if not rewritten.startswith("The user"):
                continue

            r_emb = normalize(embed_texts(emb_client, [rewritten], EMB_MODEL))[0]
            r_q = float(r_emb @ q_emb)
            if r_q >= best_q:
                continue

            # 验证答案等价性
            if not _verify_same_answer(gen_client, rewritten, question, correct_answer):
                continue

            best_q = r_q
            best_text = rewritten
            best_emb = r_emb

    return best_text, best_emb, best_q


def _verify_same_answer(gen_client, rewritten, question, correct_answer):
    """验证 rewritten 仍能推出正确答案（等价）。"""
    ans_resp = gen_client.get_response_chat(
        [{"role": "user", "content": ANSWER_FROM_MEMORY_PROMPT.format(
            memory=rewritten, question=question)}],
        max_new_tokens=128, temperature=0,
    )
    ans_obj = _parse_json_obj(ans_resp)
    if not ans_obj:
        return False
    ans = str(ans_obj.get("answer", "")).strip()
    if ans.upper() == "NO ANSWER" or not ans:
        return False

    same_resp = gen_client.get_response_chat(
        [{"role": "user", "content": SAME_ANSWER_PROMPT.format(
            answer_a=correct_answer, answer_b=ans)}],
        max_new_tokens=64, temperature=0,
    )
    same_obj = _parse_json_obj(same_resp)
    if same_obj and same_obj.get("same"):
        return True
    return False


def build_lowered_golden(gen_client, emb_client, rec, q_emb):
    """对一题的所有 golden 做多策略 lowering。
    部分 golden 改写失败的保留原文 —— 只要至少有一条成功则整体成功。
    返回 {"lowered_texts": [...], "lowered_embs": np.ndarray, "drops": [...], "lowered_min_sim": float}
    或 None 如果所有 golden 改写全部失败（全部原文和改写一样）。
    """
    goldens = rec["golden_memory"]
    if not goldens:
        return None
    question = rec["question"]
    correct_answer = rec.get("answer", "")

    strategies = [
        (CASUAL_REWRITE_PROMPT, 0.9),
        (SIMPLIFY_GOLDEN_PROMPT, 0.7),
        (NARRATIVE_EMBED_PROMPT, 1.0),
        (VOCAB_SHIFT_PROMPT, 0.8),
    ]

    lowered_texts = []
    lowered_embs = []
    drops = []
    n_lowered = 0

    for g_text in goldens:
        orig_emb = normalize(embed_texts(emb_client, [g_text], EMB_MODEL))[0]
        orig_q = float(orig_emb @ q_emb)

        best_text, best_emb, best_q = _try_lower_one(
            gen_client, emb_client, g_text, question, correct_answer, q_emb,
            orig_emb, orig_q, strategies)

        if best_text == g_text:
            # 改写失败，保留原文
            lowered_texts.append(g_text)
            lowered_embs.append(orig_emb)
            drops.append(0.0)
        else:
            n_lowered += 1
            drop = orig_q - best_q
            lowered_texts.append(best_text)
            lowered_embs.append(best_emb)
            drops.append(round(drop, 5))

    if n_lowered == 0:
        return None

    lowered_embs_arr = np.stack(lowered_embs)
    lowered_min_sim = float((lowered_embs_arr @ q_emb).min())

    return {
        "lowered_texts": lowered_texts,
        "lowered_embs": lowered_embs_arr,
        "drops": drops,
        "lowered_min_sim": lowered_min_sim,
    }


# ============================================================
# Step 2: distractors — 三道闸门保证每条达标
#   闸1: sim_q > lowered_min_sim
#   闸2: 不含正确答案 (verify_seed_answer)
#   闸3: 不触及答案属性 (不做任何正确/错误声称)
# ============================================================

DISTRACTOR_GEN_PROMPT = (
    "Generate ONE factual statement about a user that is highly semantically similar to the "
    "given question but does NOT answer it.\n\n"
    "Rules:\n"
    "1. REUSE the question's exact nouns, verbs, and key phrases to maximize cosine similarity.\n"
    "2. DO NOT reveal the answer or assert any value for the attribute the question asks about.\n"
    "3. Subject: \"The user\" (third person, NOT \"I\" or \"You\").\n\n"
    "Examples:\n"
    "  Q: What degree did I graduate with?\n"
    "  A: The user spent many late nights studying at the university library during their degree program.\n\n"
    "  Q: How long is my daily commute to work?\n"
    "  A: The user's daily commute to work involves driving through heavy traffic on the highway.\n\n"
    "  Q: Where did I redeem a $5 coupon on coffee creamer?\n"
    "  A: The user found a great deal on coffee creamer and used a $5 coupon at checkout.\n\n"
    "  Q: What is the name of the playlist I created on Spotify?\n"
    "  A: The user listens to their Spotify playlist every morning while getting ready.\n\n"
    "Question: {question}\n\n"
    "Return ONLY the statement, one sentence."
)

ANSWER_VERIFY_PROMPT = (
    "You are given a question and a factual statement. "
    "Your task: answer the question USING ONLY the information in the statement.\n"
    "If the statement contains enough information to answer the question (even if it gives "
    "a different answer than the truth), give that answer. If it doesn't contain relevant "
    "information, say 'NO ANSWER'.\n\n"
    "Question: {question}\nStatement: {seed}\n\n"
    'Return ONLY a JSON object: {{"answer": "<your answer or NO ANSWER>"}}'
)

ANSWER_LEAK_PROMPT = (
    "The correct answer to a question is:\n"
    "  Correct: {correct}\n\n"
    "A distractor memory gave this answer:\n"
    "  Distractor: {distractor}\n\n"
    "Does the distractor answer contain the SAME factual information as the correct answer? "
    "In other words, if a reader sees the distractor answer, would they learn the correct answer?\n\n"
    'Return ONLY a JSON object: {{"leaks": true/false}}'
)

NO_ANSWER_CLAIM_PROMPT = (
    "You are evaluating a distractor memory for a retrieval experiment.\n\n"
    "The correct answer to a question is:\n"
    "  Correct answer: {correct}\n\n"
    "A distractor statement about the user:\n"
    "  Statement: {text}\n\n"
    "Question: {question}\n\n"
    "Does the statement make ANY claim about the answer attribute — i.e. does it assert "
    "ANY value (correct OR incorrect) for the thing the question asks about?\n"
    "For example, if the question asks 'What degree did the user graduate with?', "
    "a statement that says 'The user graduated with a degree in CS' makes a claim "
    "(even if CS is wrong). A statement that says 'The user attended the university "
    "orientation in 2019' does NOT make a claim about what degree they got.\n\n"
    'Return ONLY a JSON object: {{"makes_claim": true/false}}'
)

DISTRACTOR_N = 8
DISTRACTOR_MAX_ROUNDS = 12
DISTRACTOR_SEEDS_PER_ROUND = 12  # 每轮生成 12 个候选


def _verify_no_answer_leak(gen_client, question, seed_text, correct_answer):
    """闸2: 种子作为唯一上下文→LLM答题→答案是否泄露正确答案。"""
    # Step 1: 用种子答题
    resp = gen_client.get_response_chat(
        [{"role": "user", "content": ANSWER_VERIFY_PROMPT.format(
            question=question, seed=seed_text)}],
        max_new_tokens=128, temperature=0,
    )
    obj = _parse_json_obj(resp)
    if not obj:
        return True, "parse_fail"

    ans = str(obj.get("answer", "")).strip()
    if ans.upper() == "NO ANSWER" or not ans:
        return True, "no_answer"

    # Step 2: 判断种子答案是否等于正确值
    resp2 = gen_client.get_response_chat(
        [{"role": "user", "content": ANSWER_LEAK_PROMPT.format(
            correct=correct_answer, distractor=ans)}],
        max_new_tokens=64, temperature=0,
    )
    obj2 = _parse_json_obj(resp2)
    if obj2 and obj2.get("leaks"):
        return False, "llm_leak"

    # fallback substring check
    correct_lower = correct_answer.lower().strip()
    ans_lower = ans.lower().strip()
    if len(correct_lower) > 3 and correct_lower in ans_lower:
        return False, "substring_leak"
    return True, "ok"


def _verify_no_answer_claim(gen_client, text, question, correct_answer):
    """闸3: distractor 不对答案属性做任何声称（不涉及正确/错误取值）。"""
    resp = gen_client.get_response_chat(
        [{"role": "user", "content": NO_ANSWER_CLAIM_PROMPT.format(
            correct=correct_answer, text=text, question=question)}],
        max_new_tokens=64, temperature=0,
    )
    obj = _parse_json_obj(resp)
    if not obj:
        return True, "parse_fail"  # 保守放行
    makes_claim = obj.get("makes_claim", False)
    return (not makes_claim), "makes_claim" if makes_claim else "ok"


def _distractor_passes(gen_client, emb_client, text, q_emb, lowered_min_sim,
                       question, correct_answer):
    """三道闸合一体：sim_q > min + 不泄答案 + 不触答案属性"""
    # 闸1: embedding similarity
    sim_q = float(
        normalize(embed_texts(emb_client, [text], EMB_MODEL))[0] @ q_emb
    )
    if sim_q <= lowered_min_sim:
        return False, f"sim_q={sim_q:.3f}<={lowered_min_sim:.3f}"

    # 闸2: 不含正确答案
    ok2, info2 = _verify_no_answer_leak(gen_client, question, text, correct_answer)
    if not ok2:
        return False, info2

    # 闸3: 不触及答案属性
    ok3, info3 = _verify_no_answer_claim(gen_client, text, question, correct_answer)
    if not ok3:
        return False, info3

    return True, {"sim_q": round(sim_q, 5), "gate2": info2, "gate3": info3}


def build_distractors(gen_client, emb_client, rec, q_emb, lowered_min_sim,
                      n=DISTRACTOR_N, max_rounds=DISTRACTOR_MAX_ROUNDS):
    """生成-过滤循环：攒够 n 条严格达标的 distractor 即停。
    返回 (distractors: list[dict] | None, stats: dict)
      distractor = {"text": str, "sim_q": float}
    """
    question = rec["question"]
    correct_answer = rec.get("answer", "")

    kept = []
    round_stats = {"rounds": 0, "total_candidates": 0, "reject_reasons": Counter()}

    for _round in range(max_rounds):
        round_stats["rounds"] = _round + 1
        # 生成一批候选
        candidates = []
        for _ in range(DISTRACTOR_SEEDS_PER_ROUND):
            resp = gen_client.get_response_chat(
                [{"role": "user", "content": DISTRACTOR_GEN_PROMPT.format(question=question)}],
                max_new_tokens=256, temperature=0.8,
            )
            text = (resp or "").strip().strip('"').strip("'")
            if text and len(text) >= 12 and text.startswith("The user"):
                candidates.append(text)

        round_stats["total_candidates"] += len(candidates)

        # 逐条过滤
        for text in candidates:
            if len(kept) >= n:
                break
            # 去重
            if text in [d["text"] for d in kept]:
                continue
            ok, info = _distractor_passes(
                gen_client, emb_client, text, q_emb, lowered_min_sim,
                question, correct_answer)
            if ok:
                kept.append({"text": text, "sim_q": info["sim_q"]})
            else:
                round_stats["reject_reasons"][info] += 1

        if len(kept) >= n:
            break

    round_stats["kept"] = len(kept)
    if len(kept) >= n:
        return kept[:n], round_stats
    else:
        return None, round_stats


# ============================================================
# Step 3: 单题编排 — 串联 lowering + distractor 生成 + 装配为完整记录
# ============================================================


def _assemble_record(rec, raw_lme_rec, lowered_result, distractors, emb_model_name, q_emb):
    """将各部分装配为完整 JSON 记录（schema §2）。
    q_emb: question embedding，用于计算 lowered_golden 的 sim_q。
    golden_memory.sim_q 在此为占位符 -1，后续由 process_one 填充正式值。
    """
    goldens = rec["golden_memory"]
    # golden_memory.sim_q 在此为占位符 -1，后续由 process_one 填充正式值
    record = {
        # A. 原始 LME 字段（拷入）
        "question_id": rec["question_id"],
        "question_type": rec["question_type"],
        "question": rec["question"],
        "question_date": raw_lme_rec.get("question_date", ""),
        "answer": rec.get("answer", ""),
        "answer_session_ids": raw_lme_rec.get("answer_session_ids", []),
        "haystack_dates": raw_lme_rec.get("haystack_dates", []),
        "haystack_session_ids": raw_lme_rec.get("haystack_session_ids", []),
        "haystack_sessions": raw_lme_rec.get("haystack_sessions", []),

        # B. 三组记忆
        "golden_memory": [
            {"text": t, "sim_q": -1} for t in goldens
        ],
        "lowered_golden": [
            {"text": t, "sim_q": round(float(e @ q_emb), 5),
             "source_idx": i, "date": raw_lme_rec.get("question_date", "")}
            for i, (t, e) in enumerate(zip(
                lowered_result["lowered_texts"],
                lowered_result["lowered_embs"]))
        ],
        "distractors": [
            {"text": d["text"], "sim_q": d["sim_q"],
             "date": raw_lme_rec.get("question_date", "")}
            for d in distractors
        ],

        # C. 元信息
        "embedding_model": emb_model_name,
        "lowered_golden_min_sim": round(lowered_result["lowered_min_sim"], 5),
        "constraint_ok": True,
    }
    return record


def process_one(gen_client, emb_client, rec, raw_lme_rec, golden_date=""):
    """单题全流程：lowered golden → distractors → 装配为完整记录。
    返回 (record: dict | None, stats_entry: dict)
      - record: 完整记录（constraint_ok=True）or None（失败）
      - stats_entry: 统计信息（无论成功失败）
    """
    qid = rec["question_id"]
    question = rec["question"]
    goldens = rec["golden_memory"]

    stats = {"qid": qid, "question_id": qid, "qtype": rec["question_type"], "n_golden": len(goldens)}

    # 预计算 question embedding
    q_emb = normalize(embed_texts(emb_client, [question], EMB_MODEL))[0]

    # Step 1: lowered golden
    lowered_result = build_lowered_golden(gen_client, emb_client, rec, q_emb)
    if lowered_result is None:
        stats["status"] = "lowered_fail"
        return None, stats

    stats["lowered_min_sim"] = round(lowered_result["lowered_min_sim"], 5)
    stats["lowered_drops"] = lowered_result["drops"]
    stats["lowered_n"] = len(lowered_result["lowered_texts"])
    stats["n_lowered_success"] = sum(1 for d in lowered_result["drops"] if d > 0)

    # Step 2: distractors (8 条严格达标)
    distractors, dist_stats = build_distractors(
        gen_client, emb_client, rec, q_emb, lowered_result["lowered_min_sim"])
    stats["distractor_stats"] = dist_stats

    if distractors is None:
        stats["status"] = "distractor_fail"
        return None, stats

    # Step 3: 装配
    record = _assemble_record(rec, raw_lme_rec, lowered_result, distractors, EMB_MODEL, q_emb)

    # 填充 golden_memory.sim_q（用已 compute 的 q_emb）
    golden_embs = normalize(embed_texts(emb_client, goldens, EMB_MODEL))
    for i, g_emb in enumerate(golden_embs):
        record["golden_memory"][i]["sim_q"] = round(float(g_emb @ q_emb), 5)

    # 填充 date 字段
    base_date = golden_date or raw_lme_rec.get("question_date", "")
    if base_date:
        parts = base_date.split(" ")
        year = int(parts[0].split("/")[0]) if "/" in parts[0] else 2023
        dist_date = f"{year-1}/01/01 (Tue) 00:00"
        for d in record["distractors"]:
            d["date"] = dist_date
        for g in record["golden_memory"]:
            g["date"] = base_date
        for l in record["lowered_golden"]:
            l["date"] = base_date

    stats["status"] = "ok"
    return record, stats


# ============================================================
# Main: 并发跑批 + stats + 落盘
# ============================================================

def _golden_date_of(rec, raw_lme_rec):
    """从 raw LME 数据中提取证据 session 第一个 chunk 的日期。"""
    # 简化：直接用 question_date
    return raw_lme_rec.get("question_date", "")


def run_build(questions, raw_lme_map, out_dir, max_workers=8, resume=False):
    """主跑批：对每道题调 process_one，并发出主集 / partial / stats。"""
    os.makedirs(out_dir, exist_ok=True)

    # 断点续跑
    main_path = os.path.join(out_dir, OUT_MAIN)
    partial_path = os.path.join(out_dir, OUT_PARTIAL)
    stats_path = os.path.join(out_dir, OUT_STATS)

    done_qids = set()
    if resume and os.path.exists(main_path):
        done = json.load(open(main_path))
        done_qids = {r["question_id"] for r in done}
        print(f"[resume] 已有 {len(done_qids)} 条主集记录跳过")
    # 注意：partial 条目不跳过——它们需要重试
    # （旧代码曾错误地加入了 partial 条目导致它们被跳过）

    gen_client = load_api_chat_completion(GEN_MODEL)
    emb_client = OpenAI(api_key=EMB_API_KEY, base_url=EMB_BASE_URL)

    records = []
    partials = []
    stats_entries = []

    to_process = [q for q in questions if q["question_id"] not in done_qids]
    print(f"[build] 需处理 {len(to_process)}/{len(questions)} 题, workers={max_workers}")

    t0 = time.time()

    def worker(rec):
        qid = rec["question_id"]
        raw = raw_lme_map.get(qid, {})
        gdate = _golden_date_of(rec, raw)
        try:
            record, stats = process_one(gen_client, emb_client, rec, raw, gdate)
        except Exception as e:
            return qid, None, {"qid": qid, "status": "exception", "error": str(e)}
        return qid, record, stats

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(worker, q): q for q in to_process}
        for i, fut in enumerate(as_completed(futures)):
            qid, record, stats = fut.result()
            if record is not None:
                records.append(record)
            else:
                partials.append(stats)  # 失败的存 stats 信息
            stats_entries.append(stats)
            if (i + 1) % 20 == 0 or i + 1 == len(to_process):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  [{i+1}/{len(to_process)}] ok={len(records)} fail={len(partials)} "
                      f"rate={rate:.1f}q/m elapsed={elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"[build] 完成: ok={len(records)} fail={len(partials)} 耗时={elapsed:.0f}s")

    # 合并断点续跑旧结果
    if resume:
        if os.path.exists(main_path):
            old_main = json.load(open(main_path))
            old_main_qids = {r["question_id"] for r in old_main}
            records = old_main + records
        if os.path.exists(partial_path):
            old_p = json.load(open(partial_path))
            # 排除已成功的旧 partial 条目（它们现在在 records 里）
            record_qids = {r["question_id"] for r in records}
            old_p_filtered = [p for p in old_p
                              if p.get("qid") not in record_qids
                              and p.get("question_id") not in record_qids]
            old_qids = {p.get("qid") or p.get("question_id") for p in old_p_filtered}
            partials = [p for p in partials if (p.get("qid") or p.get("question_id")) not in old_qids] + old_p_filtered

    # 落盘
    with open(main_path, "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[save] {len(records)} 条 → {main_path}")

    with open(partial_path, "w") as f:
        json.dump(partials, f, ensure_ascii=False, indent=2)
    print(f"[save] {len(partials)} 条失败 → {partial_path}")

    # stats
    s = _build_stats(records, partials, elapsed)
    with open(stats_path, "w") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)
    print(f"[save] stats → {stats_path}")
    print(f"[stats] {json.dumps(s, indent=2)}")


def _build_stats(records, partials, elapsed_sec):
    """生成构建统计。"""
    from collections import Counter
    qtypes = Counter(r["question_type"] for r in records)
    fail_status = Counter(p.get("status", "unknown") for p in partials)

    avg_lowered_n = 0
    avg_drop = 0
    avg_dist_sim = 0
    n_golden_dist = Counter()
    if records:
        n_golden_dist = Counter(len(r["golden_memory"]) for r in records)
        avg_lowered_n = sum(len(r["lowered_golden"]) for r in records) / len(records)
        drops_all = []
        for r in records:
            for l in r["lowered_golden"]:
                # 从 golden 和 lowered 的 sim_q 差推算 drop
                src = l["source_idx"]
                if src < len(r["golden_memory"]):
                    drops_all.append(r["golden_memory"][src]["sim_q"] - l["sim_q"])
        avg_drop = st.mean(drops_all) if drops_all else 0
        dist_sims = [d["sim_q"] for r in records for d in r["distractors"]]
        avg_dist_sim = st.mean(dist_sims) if dist_sims else 0

    return {
        "total_questions": len(records) + len(partials),
        "constraint_ok": len(records),
        "constraint_fail": len(partials),
        "success_rate": round(len(records) / max(len(records) + len(partials), 1), 4),
        "elapsed_sec": round(elapsed_sec, 0),
        "question_types": dict(qtypes),
        "n_golden_distribution": {str(k): v for k, v in sorted(n_golden_dist.items())},
        "avg_lowered_golden_per_question": round(avg_lowered_n, 2),
        "avg_lowered_drop": round(avg_drop, 5),
        "avg_distractor_sim_q": round(avg_dist_sim, 5),
        "fail_reasons": dict(fail_status),
    }


# ---- CLI 覆盖 ----

if __name__ == "__main__":
    args = parse_args()
    questions, raw_lme_map = load_sources(args.golden, args.raw_lme)
    if args.limit:
        questions = questions[:args.limit]
        print(f"[main] 限制只处理前 {args.limit} 题")
    run_build(questions, raw_lme_map, args.out_dir,
              max_workers=10, resume=args.resume)
