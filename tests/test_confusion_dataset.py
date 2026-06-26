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
