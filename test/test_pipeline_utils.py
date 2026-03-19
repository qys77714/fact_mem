import pytest

import pipeline_evaluate as pe
import pipeline_generate as pg


def test_resolve_benchmark_with_explicit_file():
    cfg = pg.GenerateConfig(
        benchmark="custom",
        benchmark_file="/tmp/a.json",
        output="o.jsonl",
        method="rag",
        extractor_model=None,
        manager_model=None,
        answer_model="Qwen3-4B",
        embedding_model="embed",
        retrieve_topk=5,
        memory_token_limit=2048,
        memory_granularity="session",
        database_root=None,
        embedding_base_url="http://x",
        embedding_api_key="k",
        language="zh",
    )
    fp, lang = pg._resolve_benchmark(cfg)
    assert fp == "/tmp/a.json"
    assert lang == "zh"


def test_resolve_benchmark_unknown_raises():
    cfg = pg.GenerateConfig(
        benchmark="not_exists",
        benchmark_file=None,
        output="o.jsonl",
        method="rag",
        extractor_model=None,
        manager_model=None,
        answer_model="Qwen3-4B",
        embedding_model="embed",
        retrieve_topk=5,
        memory_token_limit=2048,
        memory_granularity="session",
        database_root=None,
        embedding_base_url="http://x",
        embedding_api_key="k",
        language=None,
    )
    with pytest.raises(ValueError):
        pg._resolve_benchmark(cfg)


def test_normalize_method_alias():
    assert pg._normalize_method("rag") == "rag_turn"
    assert pg._normalize_method("fullcontext") == "full_context"
    assert pg._normalize_method("amem") == "amem"


def test_build_record_with_optional_fields():
    from benchmark.base import QuestionItem

    q = QuestionItem(
        question="q",
        answer="a",
        question_time="t",
        options=["A", "B"],
        metadata={"question_id": "qid", "golden_option": "A"},
    )
    r = pg._build_record("lme", "h1", q, "pred")
    assert r["question_id"] == "qid"
    assert r["options"] == ["A", "B"]
    assert r["golden_option"] == "A"


def test_pipeline_evaluate_helpers():
    assert pe.infer_benchmark([], "x/locomo.jsonl", None) == "locomo"
    assert pe._parse_verdict("Final answer: yes") is True
    assert pe._parse_verdict("answer: no") is False
    assert pe._extract_response_text({"content": "yes"}) == "yes"

    prompt = pe._prompt_generic(
        {
            "question": "q",
            "answer": "a",
            "model_answer": "m",
            "options": ["A", "B"],
            "golden_option": "A",
        },
        use_cot=True,
        is_mcq=True,
    )
    assert "Final answer" in prompt
    assert "Options" in prompt
