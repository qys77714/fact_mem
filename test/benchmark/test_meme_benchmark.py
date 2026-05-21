import json
from pathlib import Path

from benchmark import get_benchmark
from benchmark.meme import MEMEBenchmark, _max_session_index_1based


def _write_meme_episode(path: Path) -> None:
    ep = {
        "episode_id": "pl_001",
        "domain": "personal_life",
        "root": "health_condition",
        "tasks": [
            {
                "type": "Cas",
                "question_template": "What medication am I taking?",
                "gold_answer": "Thrynexol",
            },
            {
                "type": "ER",
                "question_template": "Recite my motto.",
                "gold_answer": "verbatim text here",
            },
        ],
        "sessions": [
            {
                "type": "evidence",
                "timestamp": "2023/03/01 (Wed) 09:00",
                "conversation": [{"role": "user", "content": "hello"}],
                "gold_facts": [{"fact_id": 1, "entity": "x", "value": "y", "fact_text": "fact"}],
            },
            {
                "type": "filler",
                "timestamp": "2023/03/02 (Thu) 10:00",
                "conversation": [{"role": "user", "content": "noise"}],
            },
        ],
        "before_questions": {
            "timestamp": "2023/03/02 (Thu) 12:00",
            "position_after_session": 0,
            "questions": [
                {
                    "task_type": "Cas",
                    "entity": ["medication"],
                    "entity_values": {"medication": "Quelmithin"},
                    "question": "What medication am I taking?",
                    "expected_answer": "Quelmithin",
                    "hop": 1,
                }
            ],
        },
        "after_questions": {
            "timestamp": "2023/03/03 (Fri) 14:00",
            "position_after_session": 1,
            "questions": [
                {
                    "task_type": "Cas",
                    "entity": ["medication"],
                    "entity_values": {"medication": "Thrynexol"},
                    "question": "What medication am I taking?",
                    "hop": 1,
                },
                {
                    "task_type": "ER",
                    "entity": ["motto"],
                    "entity_values": {"motto": "verbatim text here"},
                    "question": "Recite my motto.",
                },
            ],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([ep], ensure_ascii=False), encoding="utf-8")


def test_max_session_index_1based():
    assert _max_session_index_1based(17) == 18
    assert _max_session_index_1based(0) == 1


def test_meme_loader_includes_before_and_after(tmp_path: Path):
    fp = tmp_path / "meme_test.json"
    _write_meme_episode(fp)
    bm = MEMEBenchmark(str(fp), lang="en")
    assert len(bm) == 1
    ep = bm.episodes[0]
    assert ep.history_name == "pl_001"
    assert len(ep.qas) == 3
    phases = [q.metadata["phase"] for q in ep.qas]
    assert phases == ["before", "after", "after"]
    before_q = ep.qas[0]
    assert before_q.metadata["question_id"] == "before:Cas:0"
    assert before_q.metadata["max_session_index"] == 1
    assert before_q.metadata["entity_key"] == "medication"
    assert before_q.answer == "Quelmithin"
    after_cas = ep.qas[1]
    assert after_cas.answer == "Thrynexol"
    assert after_cas.metadata["max_session_index"] == 2
    assert "phase_blocks" in ep.metadata


def test_get_benchmark_meme_prefix(tmp_path: Path):
    fp = tmp_path / "meme_nofiller.json"
    _write_meme_episode(fp)
    bm = get_benchmark("meme_nofiller", str(fp), lang="en")
    assert isinstance(bm, MEMEBenchmark)
