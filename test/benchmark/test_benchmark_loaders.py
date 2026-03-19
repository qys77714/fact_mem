import json
from pathlib import Path

from benchmark import get_benchmark
from benchmark.locomo import _map_locomo_category_to_question_type, LocomoBenchmark
from benchmark.lme import LMEBenchmark
from benchmark.event_bench import EventBenchmark


def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_locomo_category_mapping():
    assert _map_locomo_category_to_question_type(0) == "single-hop"
    assert _map_locomo_category_to_question_type("1") == "multi-hop"
    assert _map_locomo_category_to_question_type(5) == "adversarial"
    assert _map_locomo_category_to_question_type("x") is None


def test_locomo_loader_supports_dict_style(tmp_path: Path):
    fp = tmp_path / "locomo_preprocessed.json"
    data = [
        {
            "history_id": "sample_1",
            "chat_time": {"session_2": "2024/01/02 10:00", "session_1": "2024/01/01 10:00"},
            "chat_history": {
                "session_2": [{"speaker": "user", "content": "world"}],
                "session_1": [{"speaker": "assistant", "text": "hello", "metadata": {"x": 1}}],
            },
            "QAs": [
                {
                    "question": "q?",
                    "answer": 123,
                    "category": 2,
                    "golden_option": "B",
                }
            ],
        }
    ]
    _write_json(fp, data)

    bm = LocomoBenchmark(str(fp), lang="en")
    assert len(bm) == 1
    ep = bm.episodes[0]
    assert ep.history_name == "sample_1"
    assert [s.session_date for s in ep.sessions] == ["2024/01/01 10:00", "2024/01/02 10:00"]
    assert ep.sessions[0].turns[0].content == "hello"
    assert ep.qas[0].answer == "123"
    assert ep.qas[0].question_type == "temporal-reasoning"
    assert ep.qas[0].metadata["golden_option"] == "B"


def test_lme_loader_converts_raw_and_sorts_sessions(tmp_path: Path):
    raw_dir = tmp_path / "data" / "raw_data"
    raw_fp = raw_dir / "longmemeval_x.json"
    raw_data = [
        {
            "question_id": "qid_1",
            "question": "budget?",
            "answer": "3000",
            "question_date": "2024/01/03 10:00",
            "question_type": "single-hop",
            "haystack_dates": ["2024/01/02 (Tue) 10:00", "2024/01/01 (Mon) 10:00"],
            "haystack_sessions": [
                [{"role": "assistant", "content": "later", "has_answer": False}],
                [{"role": "user", "content": "earlier", "has_answer": True}],
            ],
        }
    ]
    _write_json(raw_fp, raw_data)

    bm = LMEBenchmark(str(raw_fp), lang="en")
    assert len(bm) == 1
    ep = bm.episodes[0]
    assert ep.history_name == "qid_1"
    assert ep.sessions[0].turns[0].content == "earlier"
    assert ep.sessions[1].turns[0].content == "later"
    assert ep.qas[0].question == "budget?"

    expected_preprocessed = tmp_path / "data" / "preprocessed" / "longmemeval_x_converted.json"
    assert expected_preprocessed.exists()


def test_event_loader_basic(tmp_path: Path):
    fp = tmp_path / "event.json"
    _write_json(
        fp,
        [
            {
                "history_name": "h1",
                "chat_time": ["2024-01-01"],
                "chat_history": [[{"speaker": "A", "content": "x"}]],
                "qa": [
                    {
                        "question": "qq",
                        "answer": "aa",
                        "question_time": "2024-01-02",
                        "golden_option": "C",
                    }
                ],
            }
        ],
    )

    bm = EventBenchmark(str(fp), lang="zh")
    assert len(bm) == 1
    ep = bm.episodes[0]
    assert ep.history_name == "h1"
    assert ep.sessions[0].turns[0].speaker == "A"
    assert ep.qas[0].metadata["golden_option"] == "C"


def test_get_benchmark_routes_to_expected_loader(tmp_path: Path):
    fp = tmp_path / "dummy.json"
    _write_json(fp, [])

    assert isinstance(get_benchmark("locomo", str(fp)), LocomoBenchmark)
    assert isinstance(get_benchmark("lme_oracle", str(fp)), LMEBenchmark)
    assert isinstance(get_benchmark("lmb_event", str(fp)), EventBenchmark)
    # fallback
    assert isinstance(get_benchmark("unknown_task", str(fp)), LMEBenchmark)
