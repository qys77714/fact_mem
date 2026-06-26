"""为论文主实验生成 LME 混淆数据集：原始对话 + golden + lowered_golden + 8 distractor。
用法：uv run --no-sync python script/build_confusion_dataset.py [--limit N] [--workers 16]
"""
import os, sys, json, re, argparse
from collections import Counter
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


# ---- 纯逻辑 helper --------------------------------------------------
_BANNED_LEADING = {"i", "i'm", "i've", "you", "your", "you're", "they", "we", "he", "she", "my", "me", "his", "her"}

def subject_is_user(text):
    """如果 text 以"user"作主语返回 True（排除 I/you/they/he/she 等主语）。"""
    t = (text or "").strip()
    if not t:
        return False
    first = re.split(r"[\s,]+", t.lower(), maxsplit=1)[0].strip(".,'\"")
    if first in _BANNED_LEADING:
        return False
    return "user" in t.lower()

def lowered_min_sim(lowered):
    """lowered 列表中最小的 sim_q（空列表返回 None）。"""
    if not lowered:
        return None
    return min(l["sim_q"] for l in lowered)

def compute_constraint_ok(lowered, distractors, n_required=N_DISTRACTORS):
    """distractor 的 sim_q 是否全部 > lowered 最小 sim_q 且数量够。"""
    if not lowered or len(distractors) != n_required:
        return False
    lo = lowered_min_sim(lowered)
    return all(d["sim_q"] > lo for d in distractors)

_LME_FIELDS = ("question_id", "question_type", "question", "question_date", "answer",
               "answer_session_ids", "haystack_dates", "haystack_session_ids", "haystack_sessions")

def assemble_record(lme_rec, golden, lowered, distractors, emb_model):
    """装配一条完整的 confusion 记录。"""
    rec = {k: lme_rec.get(k) for k in _LME_FIELDS}
    rec["golden_memory"] = golden
    rec["lowered_golden"] = lowered
    rec["distractors"] = distractors
    rec["embedding_model"] = emb_model
    rec["lowered_golden_min_sim"] = lowered_min_sim(lowered)
    rec["constraint_ok"] = compute_constraint_ok(lowered, distractors)
    return rec


# ---- embedding wrapper & sim helpers ----------------------------------------
def make_emb_client():
    return OpenAI(api_key=os.getenv("EMBEDDING_API_KEY", "EMPTY"),
                  base_url=os.getenv("EMBEDDING_BASE_URL", "http://localhost:7110/v1/"))

def embed_norm(emb_client, texts):
    return normalize(np.asarray(embed_texts(emb_client, texts, EMB_MODEL), dtype=float))

def sim_to_q(vecs, q_vec):
    return [float(v @ q_vec) for v in vecs]


# ---- distractor 生成-过滤循环 --------------------------------------------------
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


# ---- per-question orchestration ------------------------------------------------
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
