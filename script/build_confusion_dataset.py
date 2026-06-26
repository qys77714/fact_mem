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

GOLDEN_LOWER_MAX_ATTEMPTS = 5


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
    返回 {"lowered_texts": [...], "lowered_embs": np.ndarray, "drops": [...], "lowered_min_sim": float}
    或 None 如果某条 golden lowering 全部失败。
    """
    goldens = rec["golden_memory"]
    if not goldens:
        return None
    question = rec["question"]
    correct_answer = rec.get("answer", "")

    strategies = [
        (CASUAL_REWRITE_PROMPT, 0.9),
        (SIMPLIFY_GOLDEN_PROMPT, 0.7),
    ]

    lowered_texts = []
    lowered_embs = []
    drops = []

    for g_text in goldens:
        orig_emb = normalize(embed_texts(emb_client, [g_text], EMB_MODEL))[0]
        orig_q = float(orig_emb @ q_emb)

        best_text, best_emb, best_q = _try_lower_one(
            gen_client, emb_client, g_text, question, correct_answer, q_emb,
            orig_emb, orig_q, strategies)

        if best_text == g_text:
            return None

        drop = orig_q - best_q
        lowered_texts.append(best_text)
        lowered_embs.append(best_emb)
        drops.append(round(drop, 5))

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
    "A user asked the following question in a conversation. Generate ONE plausible factual "
    "statement about the user that satisfies ALL these rules:\n\n"
    "1. SAME TOPIC DOMAIN: The statement must be about the same general topic area as the "
    "   question, so it would appear semantically similar in a retrieval system.\n"
    "2. DO NOT ANSWER THE QUESTION: The statement must NOT reveal the correct answer, NOR "
    "   assert any alternative/wrong answer for the same attribute. It should describe a "
    "   RELATED but DIFFERENT aspect of the same topic — e.g. for 'What degree did I get?', "
    "   say something about university life (orientation, library, commute) rather than "
    "   naming any degree at all.\n"
    "3. NATURAL: Sound like a real user memory, not a test case.\n"
    "4. SUBJECT: Use \"The user\" as the subject (third person, NOT \"I\" or \"You\").\n\n"
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
DISTRACTOR_MAX_ROUNDS = 6
DISTRACTOR_SEEDS_PER_ROUND = 5  # 每轮生成 5 个候选


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
            if text and len(text) >= 12:
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


if __name__ == "__main__":
    args = parse_args()
    questions, raw_lme_map = load_sources(args.golden, args.raw_lme)
    if args.limit:
        questions = questions[:args.limit]
        print(f"[main] 限制只处理前 {args.limit} 题")
    # — 占位: 后续任务在此扩展 —
    print("[main] 数据加载 OK, 后续任务衔接此脚本")
