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


# ---- Seed Strategy Prompts ----

SEED_STRATEGY_PROMPTS = {
    "keyword-borrow": textwrap.dedent("""\
You are creating "confusion memories" for a retrieval experiment.
Given a question someone asked about a user, generate {n} DIFFERENT factual statements about the user.

Strategy: KEYWORD BORROW — take the key nouns/verbs from the question and embed them in an UNRELATED factual statement.

Rules:
1. REUSE the key content words from the question EXACTLY AS WRITTEN — do NOT replace with synonyms
2. The factual statement must be about a DIFFERENT aspect of the user's life, NOT about what the question asks
3. CRITICAL: MUST NOT provide any information that answers the question
4. Start with "The user" and write ONE sentence each
5. The {n} statements should be DIVERSE — different unrelated topics

Question: {question}

Return a JSON array of {n} strings: {{"statements": [...]}}"""),

    "topic-drift": textwrap.dedent("""\
You are creating "confusion memories" for a retrieval experiment.
Given a question someone asked about a user, generate {n} DIFFERENT factual statements.

Strategy: TOPIC DRIFT — start from the question's topic but drift to an ADJACENT but DIFFERENT sub-topic that does NOT contain the answer.

Rules:
1. REUSE key words from the question where natural, but the main topic should be adjacent, not overlapping
2. Talk about a related experience, context, or detail that does NOT reveal the answer
3. CRITICAL: MUST NOT provide any information that answers the question
4. Start with "The user" and write ONE sentence each
5. The {n} statements should drift to different adjacent topics

Question: {question}

Return a JSON array of {n} strings: {{"statements": [...]}}"""),

    "generalize": textwrap.dedent("""\
You are creating "confusion memories" for a retrieval experiment.
Given a question someone asked about a user, generate {n} DIFFERENT factual statements.

Strategy: GENERALIZE — take the question's core topic and make GENERAL background statements about it that do NOT answer the specific question.

Rules:
1. Use the general topic words from the question (e.g., "education" from "what degree")
2. Make broad, non-specific statements about the user's experience/thinking in that general area
3. CRITICAL: MUST NOT provide any SPECIFIC information that answers the question
4. Start with "The user" and write ONE sentence each
5. The {n} statements should cover different angles of the general topic

Question: {question}

Return a JSON array of {n} strings: {{"statements": [...]}}"""),

    "context-surround": textwrap.dedent("""\
You are creating "confusion memories" for a retrieval experiment.
Given a question someone asked about a user, generate {n} DIFFERENT factual statements.

Strategy: CONTEXT SURROUND — describe the SURROUNDING context, situation, or process that relates to the question topic but does NOT touch the core fact being asked.

Rules:
1. Use words from the question but describe the peripheral context — the setting, the process, the lead-up
2. CRITICAL: MUST NOT provide the SPECIFIC fact the question asks for
3. Start with "The user" and write ONE sentence each
4. The {n} statements should describe different contextual aspects

Question: {question}

Return a JSON array of {n} strings: {{"statements": [...]}}"""),
}

IDK_BATCH_PROMPT = textwrap.dedent("""\
You are an evaluator. You are given a question and several memory statements.

For EACH statement, answer the question using ONLY the information in that statement.

Rules (apply to EACH statement independently):
- If the statement contains information that DIRECTLY answers the question, give that answer concisely.
- If the statement does NOT contain enough information to answer, respond "I DON'T KNOW".
- Do NOT guess. Do NOT infer. Do NOT use your own knowledge.

Question: {question}

Statements:
{statements}

Return a JSON array, one entry per statement in the SAME ORDER:
{{"results": [{{"index": 0, "answer": "I DON'T KNOW"}}, {{"index": 1, "answer": "..."}}, ...]}}""")


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
# 种子生成 — 4 策略 prompt + 双闸门过滤
# ============================================================


def generate_seeds(gen_client, emb_client, question, q_emb, anchor_sim):
    """
    多策略种子生成。每种策略独立 prompt，每轮 batch 生成 SEED_BATCH_SIZE 条。
    闸1: sim_q > anchor_sim - GATE_OFFSET
    闸2: IDK 测试
    目标 TARGET_SEEDS 条种子，最多 SEED_MAX_ROUNDS 轮。
    返回 (kept_seeds, stats)。
    """
    gate_threshold = anchor_sim - GATE_OFFSET
    kept = []
    stats = {"rounds": 0, "llm_seed_calls": 0, "llm_idk_calls": 0, "emb_calls": 0}

    for rnd in range(SEED_MAX_ROUNDS):
        stats["rounds"] = rnd + 1
        round_candidates = []

        for strategy_name, prompt_tpl in SEED_STRATEGY_PROMPTS.items():
            resp = gen_client.get_response_chat(
                [{"role": "user", "content": prompt_tpl.format(n=SEED_BATCH_SIZE, question=question)}],
                max_new_tokens=1024, temperature=0.6,
            )
            stats["llm_seed_calls"] += 1
            obj = parse_json_obj(resp)
            candidates_raw = obj.get("statements", []) if obj else []
            if not candidates_raw:
                arr = extract_json_array(resp)
                if arr and isinstance(arr, list) and all(isinstance(x, str) for x in arr):
                    candidates_raw = arr
            candidates = [
                t.strip().strip('"').strip("'") for t in candidates_raw
                if t.strip() and len(t.strip()) >= 20 and t.strip().startswith("The user")
            ]
            for c in candidates:
                round_candidates.append((c, strategy_name))

        if not round_candidates:
            continue

        # 闸1: embedding similarity
        texts = [c[0] for c in round_candidates]
        embs = batch_embed(emb_client, texts)
        stats["emb_calls"] += 1
        sims = [float(e @ q_emb) for e in embs]

        # 闸2: 批量 IDK
        statements_str = "\n".join(f"[{i}] {t}" for i, t in enumerate(texts))
        idk_resp = gen_client.get_response_chat(
            [{"role": "user", "content": IDK_BATCH_PROMPT.format(question=question, statements=statements_str)}],
            max_new_tokens=512, temperature=0,
        )
        stats["llm_idk_calls"] += 1
        idk_array = extract_json_array(idk_resp)
        idk_map = {}
        if idk_array:
            for entry in idk_array:
                if isinstance(entry, dict) and "index" in entry:
                    idk_map[entry["index"]] = str(entry.get("answer", "")).strip().upper()

        for i, (text, source) in enumerate(round_candidates):
            if sims[i] <= gate_threshold:
                continue
            idk_ans = idk_map.get(i, "PARSE_FAIL")
            if idk_ans not in ("I DON'T KNOW", "I DONT KNOW", "I DON'T KNOW.", "I DO NOT KNOW"):
                continue
            if text.lower() in [k["text"].lower() for k in kept]:
                continue
            kept.append({"text": text, "sim_q": round(sims[i], 5), "source": source,
                         "source_idx": len(kept), "from_round": rnd + 1})

        if len(kept) >= TARGET_SEEDS:
            break

    return kept, stats


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


# ============================================================
# 替换验证引擎
# ============================================================


def build_answer_prompt(memories, question, question_date=""):
    """用 memories 集合构建答题 prompt，模拟 agent_prompt_en_open.jinja 的逻辑。"""
    # 构建 context_block：每条 memory 包装为 Retrieved Memory Unit
    context_parts = []
    for i, mem_text in enumerate(memories):
        unit = f"### Retrieved Memory Unit {i+1}\n<MemoryContent>\n{mem_text}\n</MemoryContent>"
        context_parts.append(unit)
    context_block = "\n\n".join(context_parts) if context_parts else "No relevant memory found."

    prompt = textwrap.dedent("""\
    You are a memory-augmented assistant. Use the retrieved memory units to provide accurate and context-aware answers to the user's questions.

    {context_block}

    ### Question Details
    {date_line}
    - Question: {question}

    Please give a short answer.""")

    date_line = f"- Current Date: {question_date}" if question_date else ""
    return prompt.format(context_block=context_block, date_line=date_line, question=question)


def build_judge_messages(question, gold_answer, candidate_answer):
    """构建 judge messages，模拟 pipeline_eval_oqa.jinja + pipeline_eval_system.jinja 的逻辑。"""
    system_msg = "You are a careful evaluation assistant."
    user_msg = textwrap.dedent("""\
    You are given a question, its ground-truth answer, and a model response. Judge if the model response is semantically correct. Be lenient for wording differences if the core meaning is correct.

    **Question**: {question}

    **Ground-truth answer**: {reference}

    **Model response**: {candidate}

    Answer yes or no only.""").format(question=question, reference=gold_answer, candidate=candidate_answer)

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def verify_replacement(gen_client, golden_memories, candidate_text, candidate_idx, question, gold_answer, question_date=""):
    """
    替换验证：用 candidate_text 替换 golden_memories[candidate_idx]，
    用替换后的完整集合让 LLM 答题，judge 比对 gold answer。
    返回 (passed: bool, llm_answer: str)。
    """
    test_memories = list(golden_memories)
    test_memories[candidate_idx] = candidate_text

    answer_prompt = build_answer_prompt(test_memories, question, question_date)
    resp = gen_client.get_response_chat(
        [{"role": "user", "content": answer_prompt}],
        max_new_tokens=128, temperature=0,
    )
    llm_answer = (resp or "").strip()

    judge_msgs = build_judge_messages(question, gold_answer, llm_answer)
    judge_resp = gen_client.get_response_chat(judge_msgs, max_new_tokens=32, temperature=0)
    passed = "yes" in (judge_resp or "").strip().lower()

    return passed, llm_answer


# ============================================================
# Lowering Engine — 逐条 lowering + 逐条 fallback
# ============================================================


def lower_one_golden(gen_client, emb_client, g_text, g_sim, g_idx,
                     all_golden_texts, question, gold_answer,
                     question_type, question_date, q_emb):
    """
    对单条 golden 执行 lowering（最多 8 次尝试，跑满）。
    目标 sim 区间: [g_sim - LOWER_SIM_LOWER, g_sim - LOWER_SIM_UPPER]
    区间内按 sim 从低到高做替换验证，第一个通过即接受。

    返回:
      {"golden_idx": g_idx, "success": bool, "original_sim": g_sim,
       "lowered_sim": float_or_None, "lowered_text": str_or_None,
       "attempts": int, "in_interval": int}
    """
    prompt_tpl = get_lowering_prompt(question_type)
    target_lower = g_sim - LOWER_SIM_LOWER
    target_upper = g_sim - LOWER_SIM_UPPER

    candidates = []  # (text, sim)
    for attempt in range(LOWER_MAX_ATTEMPTS):
        resp = gen_client.get_response_chat(
            [{"role": "user", "content": prompt_tpl.format(golden=g_text)}],
            max_new_tokens=256, temperature=0.65,
        )
        text = (resp or "").strip().strip('"').strip("'")
        if not text or len(text) < 12:
            continue
        if not text.startswith("The user"):
            continue
        if text.lower() == g_text.lower():
            continue

        e = normalize(embed_texts(emb_client, [text], EMB_MODEL))[0]
        sim = float(e @ q_emb)
        candidates.append((text, sim))

    # 筛出在目标区间内的
    in_interval = [(t, s) for t, s in candidates if target_lower <= s <= target_upper]
    # 按 sim 从低到高排序（最不相似优先验证）
    in_interval.sort(key=lambda x: x[1])

    result = {
        "golden_idx": g_idx,
        "success": False,
        "original_sim": round(g_sim, 5),
        "lowered_sim": None,
        "lowered_text": None,
        "attempts": len(candidates),
        "in_interval": len(in_interval),
    }

    for cand_text, cand_sim in in_interval:
        passed, llm_ans = verify_replacement(
            gen_client, all_golden_texts, cand_text, g_idx,
            question, gold_answer, question_date)
        if passed:
            result["success"] = True
            result["lowered_sim"] = round(cand_sim, 5)
            result["lowered_text"] = cand_text
            break

    return result


def run_lowering_for_question(gen_client, emb_client, golden_memories, question, gold_answer, question_type, question_date, q_emb):
    """
    对一题的所有 golden 逐条执行 lowering。
    返回:
      lowered_texts: list[str] — 混合集合（lowered成功的用lowered，否则用原始）
      golden_sims: list[float] — 原始 golden 的 sim_q
      correct_sims: list[float] — 混合集合每条记忆的 sim_q
      anchor_sim: float — min(correct_sims)
      lowering_details: list[dict]
      lowering_status: "partial" | "full_fallback"
    """
    golden_texts = [g["content"] for g in golden_memories]

    # 计算原始 sim_q
    golden_embs = batch_embed(emb_client, golden_texts)
    golden_sims = [float(e @ q_emb) for e in golden_embs]

    # 逐条 lowering
    lowering_details = []
    lowered_texts = list(golden_texts)  # 从原始开始

    for i, g_text in enumerate(golden_texts):
        detail = lower_one_golden(
            gen_client, emb_client, g_text, golden_sims[i], i,
            golden_texts, question, gold_answer,
            question_type, question_date, q_emb)
        lowering_details.append(detail)
        if detail["success"]:
            lowered_texts[i] = detail["lowered_text"]

    # 计算混合集合的 sim_q
    correct_embs = batch_embed(emb_client, lowered_texts)
    correct_sims = [float(e @ q_emb) for e in correct_embs]
    anchor_sim = min(correct_sims)

    n_lowered = sum(1 for d in lowering_details if d["success"])
    lowering_status = "partial" if n_lowered > 0 else "full_fallback"

    return {
        "lowered_texts": lowered_texts,
        "golden_sims": [round(s, 5) for s in golden_sims],
        "correct_sims": [round(s, 5) for s in correct_sims],
        "anchor_sim": round(anchor_sim, 5),
        "lowering_details": lowering_details,
        "lowering_status": lowering_status,
    }


if __name__ == "__main__":
    args = parse_args()
    print(f"[confusion_v3] out_dir={args.out_dir}, max_workers={args.max_workers}, resume={args.resume}")
