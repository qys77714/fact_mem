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
LOWER_SIM_LOWER = 0.8      # sim(g_i) - 0.8（足够宽，靠替换验证把关）
LOWER_SIM_UPPER = 0.05     # sim(g_i) - 0.05
GATE_OFFSET = 0.1          # anchor_sim - 0.1

# ---- Lowering Prompts (per question type) ----
# 新思路：保留事实信息，自由换词。不强制保留原词。

_BASE_LOWERING = textwrap.dedent("""\
Rewrite this memory statement in a NATURAL, CONVERSATIONAL way, as if someone casually mentioned it in passing.

CRITICAL RULE: The rewritten version MUST preserve ALL factual information — every number, name, date, and relationship. If someone reads the rewritten version, they must be able to derive the SAME FACTS as the original.

HOWEVER, you SHOULD change the WORDS and EXPRESSION:
- Replace specific nouns/verbs with different phrasings (e.g., "playlists" → "curated collections", "attended" → "went to", "graduated with" → "earned a degree in")
- Use longer, more natural sentence structures
- Make it sound like a casual recollection, not a database entry
- The EMBEDDING should be DIFFERENT from the original — aim for different vocabulary patterns

What MUST stay the same: the factual truth. What SHOULD change: how you say it.
Start your sentence with "The user".

Original: {golden}

Return ONLY the rewritten sentence, starting with "The user".""")

_TEMPORAL_LOWERING = textwrap.dedent("""\
Rewrite this TEMPORAL memory statement in a NATURAL, CONVERSATIONAL way, as if someone casually mentioned it in passing.

🔴 DATES ARE SACRED: ALL dates, times, and temporal expressions must be copied VERBATIM. Not a single character of any date or time may change. If you change "2023/05/20" to "May 2023", the answer becomes WRONG.

CRITICAL RULE: The rewritten version MUST preserve ALL factual information besides dates — every number, name, and relationship must be derivable.

HOWEVER, you SHOULD change the WORDS and EXPRESSION around the dates:
- Replace specific nouns/verbs with different phrasings
- Use longer, more natural sentence structures
- Make it sound like a casual recollection, not a database entry
- The EMBEDDING should be DIFFERENT from the original

Dates: verbatim. Everything else: rephrase freely while keeping facts intact.
Start your sentence with "The user".

Original: {golden}

Return ONLY the rewritten sentence, starting with "The user".""")

LOWERING_PROMPTS = {
    "temporal-reasoning": _TEMPORAL_LOWERING,
    "multi-session": _BASE_LOWERING,
    "knowledge-update": _BASE_LOWERING,
    "single-session-user": _BASE_LOWERING,
    "single-session-assistant": _BASE_LOWERING,
    "single-session-preference": _BASE_LOWERING,
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

    "keyword-dense": textwrap.dedent("""\
You are creating "confusion memories" for a retrieval experiment.
Given a question someone asked about a user, generate {n} DIFFERENT factual statements about the user.

Strategy: KEYWORD DENSE — extract ALL content words (nouns, verbs, proper names) from the question and use AS MANY of them as possible in each statement. The goal is MAXIMUM vocabulary overlap with the question.

Step 1: Identify the key content words in the question.
Step 2: Write statements that use ALL or MOST of these words EXACTLY AS WRITTEN.
Step 3: BUT ensure each statement is about a DIFFERENT ASPECT that does NOT answer the question.

Rules:
1. Extract and REUSE every content word from the question verbatim — nouns, verbs, proper names, numbers should all appear
2. The factual claim must be about something DIFFERENT from what the question asks
3. CRITICAL: MUST NOT provide the SPECIFIC information the question is asking for
4. Start with "The user" and write ONE sentence
5. The {n} statements should be diverse in their themes

Example:
Q: "How many largemouth bass did I catch on my fishing trip to Lake Michigan?"
Content words: largemouth, bass, catch/caught, fishing, trip, Lake, Michigan
GOOD: "The user spent weeks planning their fishing trip to Lake Michigan, researching the best spots to find largemouth bass."
  → uses ALL keywords, talks about PLANNING not CATCHING, doesn't say how many
GOOD: "The user recalled that largemouth bass were particularly active in Lake Michigan during their fishing trip."
  → uses ALL keywords, talks about fish ACTIVITY not catch count

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


# ---- Rewrite Prompts (EQV / OSN / NSO) ----

EQV_REWRITE_PROMPT = textwrap.dedent("""\
Rewrite the following statement to express the SAME meaning in a different way.
This is for a retrieval experiment where vocabulary overlap with a query matters.

Rules:
1. Keep ALL key nouns and content words EXACTLY AS IS. Do NOT replace domain-specific terms,
   proper nouns, or the main subject/object words with synonyms.
   Only change: function words (the, a, of), common verbs (is→represents, uses→utilizes),
   and sentence connectors. You may reorder clauses and change active↔passive.
2. The rewritten version MUST convey exactly the same facts.
3. One sentence, start with "The user".

Original: {seed}

Return ONLY the rewritten statement, starting with "The user".""")

OSN_REWRITE_PROMPT = textwrap.dedent("""\
Rewrite the following statement to be MORE SPECIFIC and CONCRETE (the new version strictly entails the original).

Rules:
1. Keep ALL key nouns and content words from the original EXACTLY AS IS. Do NOT replace them.
2. Add specific details to make the statement more vivid — elaborate on aspects already present
   (e.g., add time, frequency, manner, or a minor contextual detail).
3. The new version must logically entail the original (if new is true, original must be true).
4. CRITICAL: The added details must NOT introduce any information that could answer a question.
5. Start with "The user". One sentence only.

Original: {seed}

Return ONLY the rewritten statement, starting with "The user".""")

NSO_REWRITE_PROMPT = textwrap.dedent("""\
Rewrite the following statement to be slightly MORE GENERAL (the original entails the new version).

Rules:
1. Keep ALL key nouns and content words from the original EXACTLY AS IS. Do NOT replace them
   with broader category terms. Only remove or soften the least important modifiers (adjectives,
   adverbs, time/place qualifiers) while preserving the sentence's core meaning.
2. The original must entail the new version (if original is true, new must be true).
3. CRITICAL: Do NOT introduce any information that could answer a question.
4. Start with "The user". One sentence only.

Original: {seed}

Return ONLY the rewritten statement, starting with "The user".""")

REWRITE_STRATEGIES = [
    ("eqv", EQV_REWRITE_PROMPT, 0.7),
    ("osn", OSN_REWRITE_PROMPT, 0.85),
    ("nso", NSO_REWRITE_PROMPT, 0.7),
]


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
    ap.add_argument("--regen-lowered", action="store_true",
                    help="强制重新生成所有 lowered_golden（忽略断点续跑，重新处理所有题目）")
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


def get_non_evidence_dates(raw_rec):
    """从非 evidence session 中提取日期列表。"""
    ev_ids = set(raw_rec.get("answer_session_ids", []))
    haystack_ids = raw_rec.get("haystack_session_ids", [])
    haystack_dates = raw_rec.get("haystack_dates", [])
    non_ev = [d for i, d in enumerate(haystack_dates)
              if i < len(haystack_ids) and haystack_ids[i] not in ev_ids]
    return non_ev


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


def generate_seeds(gen_client, emb_client, question, q_emb, anchor_sim, skip_gates=False):
    """
    多策略种子生成。每种策略独立 prompt，每轮 batch 生成 SEED_BATCH_SIZE 条。
    闸1: sim_q > anchor_sim - GATE_OFFSET（skip_gates=True 时跳过）
    闸2: IDK 测试（skip_gates=True 时跳过）
    目标 TARGET_SEEDS 条种子，最多 SEED_MAX_ROUNDS 轮。
    返回 (kept_seeds, stats)。
    """
    gate_threshold = -999 if skip_gates else min(anchor_sim - GATE_OFFSET, 0.7)
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

        # 闸2: 批量 IDK（skip_gates 时跳过）
        if skip_gates:
            idk_map = {}  # 不检查，全通过
        else:
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
            if not skip_gates:
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
# 改写扩充 — EQV / OSN / NSO
# ============================================================


def rewrite_and_expand(gen_client, emb_client, seeds, question, q_emb, gate_threshold, skip_gates=False):
    """改写扩充 + 闸门过滤。返回 top 8 distractors 和 stats。
    skip_gates=True 时跳过所有闸门，直接取 sim 最高的 8 条。"""
    stats = {"llm_rewrite_calls": 0, "llm_idk_calls": 0, "emb_calls": 0}
    all_items = list(seeds)

    for si, seed in enumerate(seeds):
        rewrite_candidates = []
        for strat_name, prompt_tpl, temp in REWRITE_STRATEGIES:
            for attempt in range(REWRITE_MAX_ATTEMPTS):
                resp = gen_client.get_response_chat(
                    [{"role": "user", "content": prompt_tpl.format(seed=seed["text"])}],
                    max_new_tokens=200, temperature=temp,
                )
                stats["llm_rewrite_calls"] += 1
                text = (resp or "").strip().strip('"').strip("'")
                if not text or len(text) < 20 or not text.startswith("The user"):
                    continue
                if text.lower() == seed["text"].lower():
                    continue
                rewrite_candidates.append({"text": text, "source": f"seed{si}_{strat_name}",
                                           "source_idx": si, "strategy": strat_name})
                break

        if rewrite_candidates:
            texts = [c["text"] for c in rewrite_candidates]
            embs = batch_embed(emb_client, texts)
            stats["emb_calls"] += 1
            sims = [float(e @ q_emb) for e in embs]

            if skip_gates:
                idk_map = {}
            else:
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

            for i, cand in enumerate(rewrite_candidates):
                if sims[i] <= gate_threshold:
                    continue
                if not skip_gates:
                    ok_idk = idk_map.get(i, "PARSE_FAIL") in ("I DON'T KNOW", "I DONT KNOW", "I DON'T KNOW.", "I DO NOT KNOW")
                    if not ok_idk:
                        continue
                cand["sim_q"] = round(sims[i], 5)
                all_items.append(cand)

    # 去重
    seen = set()
    deduped = []
    for item in all_items:
        key = item["text"].lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    deduped.sort(key=lambda x: x.get("sim_q", 0), reverse=True)
    selected = deduped[:TARGET_DISTRACTORS]
    return selected, stats, {"total": len(all_items), "after_dedup": len(deduped), "selected": len(selected)}


# ============================================================
# 数据加载
# ============================================================


def load_data(golden_path=DEFAULT_GOLDEN, raw_path=DEFAULT_RAW):
    """加载 golden 和 raw 数据，返回待处理题目列表 + raw map。

    golden_memory 格式：[{"content": "...", "date": "..."}, ...]
    跳过 abstention 和无 golden_memory 的题。
    """
    with open(golden_path) as f:
        golden_data = json.load(f)
    with open(raw_path) as f:
        raw_data = json.load(f)
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


def process_one(gen_client, emb_client, q_input, raw_map):
    """单题全流程编排。返回 (record | None, stats dict)。"""
    t_start = time.time()
    qid = q_input["question_id"]
    question = q_input["question"]
    answer = q_input["answer"]
    question_type = q_input["question_type"]
    question_date = q_input["question_date"]
    judged_correct = q_input["judged_correct"]
    golden_memories = q_input["golden_memory"]
    raw_rec = raw_map.get(qid, {})

    stats = {"qid": qid, "question": question[:120], "answer": str(answer)[:80],
             "n_golden": len(golden_memories)}

    # 计算 query embedding
    q_emb = normalize(embed_texts(emb_client, [question], EMB_MODEL))[0]

    golden_texts = [g["content"] for g in golden_memories]
    golden_dates = [g.get("date", question_date) for g in golden_memories]

    # Step 0: 计算原始 golden sim_q
    golden_embs = batch_embed(emb_client, golden_texts)
    golden_sims = [round(float(e @ q_emb), 5) for e in golden_embs]

    # ---- Step 1: Lowering (仅 judged_correct=true) ----
    if judged_correct:
        lowering_result = run_lowering_for_question(
            gen_client, emb_client, golden_memories,
            question, answer, question_type, question_date, q_emb)
        lowered_texts = lowering_result["lowered_texts"]
        anchor_sim = lowering_result["anchor_sim"]
        lowering_status = lowering_result["lowering_status"]
        lowering_details = lowering_result["lowering_details"]
        stats["lowering_details"] = lowering_details
        stats["lowering_status"] = lowering_status
    else:
        # judged_correct=false: 跳过 lowering，直接用 golden
        lowered_texts = golden_texts
        golden_sims_only = [float(e @ q_emb) for e in golden_embs]
        anchor_sim = round(min(golden_sims_only), 5)
        lowering_status = "full_fallback"
        lowering_details = [
            {"golden_idx": i, "success": False, "original_sim": golden_sims[i],
             "lowered_sim": None, "lowered_text": None,
             "attempts": 0, "in_interval": 0, "skip_reason": "judged_correct=false"}
            for i in range(len(golden_texts))
        ]
        stats["lowering_details"] = lowering_details
        stats["lowering_status"] = lowering_status

    gate_threshold = min(anchor_sim - GATE_OFFSET, 0.7)  # 高 anchor 题放宽闸门
    stats["anchor_sim"] = anchor_sim
    stats["gate_threshold"] = round(gate_threshold, 5)

    # ---- Step 2: 种子生成 ----
    seeds, seed_stats = generate_seeds(gen_client, emb_client, question, q_emb, anchor_sim)
    stats["seed_stats"] = seed_stats

    no_gate = False
    if len(seeds) == 0:
        # 无闸门重试：跳过 sim/IDK 过滤，生成什么用什么
        seeds, seed_stats2 = generate_seeds(gen_client, emb_client, question, q_emb, anchor_sim, skip_gates=True)
        stats["seed_stats"] = {**seed_stats, "retry_no_gate": seed_stats2}
        stats["seed_retry_no_gate"] = True
        no_gate = True
        if len(seeds) == 0:
            stats["status"] = "seed_fail"
            stats["elapsed"] = round(time.time() - t_start, 1)
            return None, stats

    # ---- Step 3: 改写扩充 ----
    distractors, rw_stats, expand_info = rewrite_and_expand(
        gen_client, emb_client, seeds, question, q_emb, gate_threshold, skip_gates=no_gate)
    stats["expand_stats"] = expand_info
    n_dist = len(distractors)

    # ---- Step 4: 装配 ----
    # Golden memory 记录
    golden_with_sim = [
        {"text": golden_texts[i], "sim_q": golden_sims[i], "date": golden_dates[i]}
        for i in range(len(golden_texts))
    ]

    # Lowered golden（仅 accepted 的 lowered 条目）
    lowered_golden = []
    for detail in lowering_details:
        if detail["success"]:
            g_idx = detail["golden_idx"]
            lowered_golden.append({
                "text": detail["lowered_text"],
                "sim_q": detail["lowered_sim"],
                "source_idx": g_idx,
                "date": golden_dates[g_idx],
            })

    # Distractor dates: 散布在非 evidence session 的日期中
    non_ev_dates = get_non_evidence_dates(raw_rec)
    if non_ev_dates and len(non_ev_dates) >= len(distractors):
        dist_dates = random.sample(non_ev_dates, len(distractors))
    elif non_ev_dates:
        dist_dates = (non_ev_dates * ((len(distractors) // len(non_ev_dates)) + 1))[:len(distractors)]
    else:
        dist_dates = [question_date] * len(distractors)

    haystack_dates = raw_rec.get("haystack_dates", [])
    haystack_sids = raw_rec.get("haystack_session_ids", [])
    haystack_sessions = raw_rec.get("haystack_sessions", [])
    answer_sids = raw_rec.get("answer_session_ids", [])

    record = {
        "question_id": qid,
        "question_type": question_type,
        "question": question,
        "question_date": question_date,
        "answer": answer,
        "answer_session_ids": answer_sids,
        "haystack_dates": haystack_dates,
        "haystack_session_ids": haystack_sids,
        "haystack_sessions": haystack_sessions,
        "golden_memory": golden_with_sim,
        "lowering_status": lowering_status,
        "lowering_details": lowering_details,
        "lowered_golden": lowered_golden,
        "anchor_sim": round(anchor_sim, 5),
        "distractors": [
            {"text": d["text"], "sim_q": d["sim_q"],
             "source": d.get("source", "seed"), "source_idx": d.get("source_idx", -1),
             "date": dist_dates[i % len(dist_dates)]}
            for i, d in enumerate(distractors)
        ],
        "embedding_model": EMB_MODEL,
        "constraint_ok": n_dist >= TARGET_DISTRACTORS,
    }

    stats["status"] = "ok"
    stats["n_distractors"] = len(distractors)
    stats["elapsed"] = round(time.time() - t_start, 1)
    return record, stats


# ============================================================
# Main 入口 — 并发编排、断点续跑、统计
# ============================================================


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out_main = os.path.join(args.out_dir, f"{args.out_name}.json")
    out_partial = os.path.join(args.out_dir, f"{args.out_name}_partial.json")
    out_stats = os.path.join(args.out_dir, "confusion_v3_build_stats.json")

    questions, raw_map = load_data(args.golden, args.raw)

    # 断点续跑
    done_qids = set()
    if args.regen_lowered:
        print("[regen-lowered] 忽略断点续跑，重新处理所有题目（lowered_golden 将完全重建）")
    elif args.resume and os.path.exists(out_main):
        with open(out_main) as f:
            done = json.load(f)
        done_qids = {r["question_id"] for r in done}
        print(f"[resume] 已有 {len(done_qids)} 条跳过")

    gen_client = load_api_chat_completion(GEN_MODEL)
    emb_client = OpenAI(api_key=EMB_API_KEY, base_url=EMB_BASE_URL)

    records, partials = [], []
    to_process = [q for q in questions if q["question_id"] not in done_qids]
    n_total = len(to_process)
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"开始处理 {n_total} 题 (workers={args.max_workers})")
    print(f"{'='*60}\n")

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {}
        for q in to_process:
            fut = ex.submit(process_one, gen_client, emb_client, q, raw_map)
            futures[fut] = q

        for i, fut in enumerate(as_completed(futures)):
            q_input = futures[fut]
            qid = q_input["question_id"]
            try:
                record, stats = fut.result()
            except Exception as exc:
                print(f"  [{i+1}/{n_total}] EXCEPTION {qid}: {exc}")
                partials.append({
                    "qid": qid,
                    "question": q_input["question"][:120],
                    "status": "exception",
                    "error": str(exc),
                    "elapsed": round(time.time() - t0, 1),
                })
                continue

            if record:
                records.append(record)
            else:
                partials.append(stats)

            elapsed = time.time() - t0
            n_ok, n_fail = len(records), len(partials)
            status_icon = "OK" if record else "FAIL"
            detail = f'dist={len(record["distractors"])}' if record else f'status={stats.get("status", "?")}'
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0

            if (i + 1) % 10 == 0 or record is None:
                print(f"  [{i+1}/{n_total}] {status_icon} {qid} | {detail} | "
                      f"ok={n_ok} fail={n_fail} | {elapsed:.0f}s | {rate:.1f}q/m")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"完成: ok={len(records)} fail={len(partials)} 耗时={elapsed:.0f}s")
    print(f"{'='*60}")

    # 合并断点续跑
    if args.resume and os.path.exists(out_main):
        with open(out_main) as f:
            old = json.load(f)
        old_qids = {r["question_id"] for r in old}
        records = old + [r for r in records if r["question_id"] not in old_qids]

    # 落盘
    with open(out_main, "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[save] {len(records)} 条 -> {out_main}")

    with open(out_partial, "w") as f:
        json.dump(partials, f, ensure_ascii=False, indent=2)
    print(f"[save] {len(partials)} 条失败 -> {out_partial}")

    # 统计
    sims = [d["sim_q"] for r in records for d in r["distractors"]]
    import statistics as st
    fail_reasons = Counter(p.get("status", "?") for p in partials)
    n_partial_lowered = sum(1 for r in records if r["lowering_status"] == "partial")
    n_full_fallback = sum(1 for r in records if r["lowering_status"] == "full_fallback")

    s = {
        "time": datetime.now().isoformat(),
        "total_questions": len(questions),
        "constraint_ok": len(records),
        "constraint_fail": len(partials),
        "success_rate": round(len(records) / max(len(records) + len(partials), 1), 4),
        "elapsed_sec": round(elapsed, 0),
        "sim_q_mean": round(st.mean(sims), 4) if sims else 0,
        "sim_q_median": round(st.median(sims), 4) if sims else 0,
        "sim_q_min": round(min(sims), 4) if sims else 0,
        "sim_q_max": round(max(sims), 4) if sims else 0,
        "fail_reasons": dict(fail_reasons),
        "lowering_partial": n_partial_lowered,
        "lowering_full_fallback": n_full_fallback,
        "lowering_success_rate": round(
            n_partial_lowered / max(n_partial_lowered + n_full_fallback, 1), 4),
    }
    with open(out_stats, "w") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)
    print(f"[stats] -> {out_stats}")


if __name__ == "__main__":
    main()
