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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    print(f"load_sources: {len(load_sources())} answerable questions")
