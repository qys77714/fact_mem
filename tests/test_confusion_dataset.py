# tests/test_confusion_dataset.py
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REPO, "script"))
import build_confusion_dataset as B

def test_load_sources_answerable_only():
    rows = B.load_sources()
    assert len(rows) == 469                      # 500 - 31 abstention
    r = rows[0]
    assert set(r.keys()) == {"qid", "golden_rec", "lme_rec"}
    # 对齐：golden 与 lme 同一 question_id
    assert r["golden_rec"]["question_id"] == r["lme_rec"]["question_id"]
    # 可答题 golden 非空
    assert r["golden_rec"]["golden_memory"]
    # 原始 LME 字段在位
    assert "haystack_sessions" in r["lme_rec"]

def test_subject_is_user():
    assert B.subject_is_user("The user takes yoga classes at Serenity Yoga.")
    assert not B.subject_is_user("You typically attend your yoga sessions downtown.")
    assert not B.subject_is_user("They wrapped up their studies a while back.")
    assert not B.subject_is_user("I graduated with a business degree.")
    assert not B.subject_is_user("Max is a Golden Retriever.")   # 无 user 主语

def test_compute_constraint_ok():
    lowered = [{"text": "a", "sim_q": 0.70}, {"text": "b", "sim_q": 0.75}]
    good = [{"text": f"d{i}", "sim_q": 0.71} for i in range(8)]
    assert B.compute_constraint_ok(lowered, good)               # 全 > 0.70
    bad = good[:7] + [{"text": "d7", "sim_q": 0.70}]            # 一条 == min，不达标
    assert not B.compute_constraint_ok(lowered, bad)
    assert not B.compute_constraint_ok(lowered, good[:7])       # 不足 8 条
    assert not B.compute_constraint_ok([], good)                # 无 lowered

def test_assemble_record():
    lme = {"question_id": "q1", "question": "Q?", "answer": "A",
           "question_type": "t", "question_date": "d",
           "answer_session_ids": [], "haystack_dates": [],
           "haystack_session_ids": [], "haystack_sessions": [[{"role": "user", "content": "x"}]]}
    golden = [{"text": "The user did A.", "sim_q": 0.83}]
    lowered = [{"text": "The user sort of did A.", "sim_q": 0.70, "source_idx": 0}]
    dist = [{"text": f"The user did X{i}.", "sim_q": 0.72} for i in range(8)]
    rec = B.assemble_record(lme, golden, lowered, dist, "qwen3-embedding-0.6b")
    assert rec["question_id"] == "q1"
    assert rec["haystack_sessions"] == lme["haystack_sessions"]   # 原始对话保留
    assert rec["golden_memory"] == golden
    assert rec["lowered_golden"] == lowered
    assert rec["distractors"] == dist
    assert rec["embedding_model"] == "qwen3-embedding-0.6b"
    assert rec["lowered_golden_min_sim"] == 0.70
    assert rec["constraint_ok"] is True


import numpy as np


def test_embed_and_sim():
    emb = B.make_emb_client()
    q = B.embed_norm(emb, ["Where does the user do yoga?"])[0]
    vecs = B.embed_norm(emb, ["The user does yoga at Serenity Yoga.",
                              "The user enjoys cooking pasta on weekends."])
    sims = B.sim_to_q(vecs, q)
    assert len(sims) == 2
    assert all(-1.0 <= s <= 1.0 for s in sims)
    assert sims[0] > sims[1]              # same topic more similar
    assert abs(np.linalg.norm(vecs[0]) - 1.0) < 1e-5   # already normalized
