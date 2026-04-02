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
        agent_trace_dir=None,
        parallel_episodes=1,
        rebuild_memory=False,
        mem0_dialogue_format="auto",
        manager_max_new_tokens=2048,
        mem0_extract_concurrency=8,
        relmem_relation_concurrency=8,
        answer_concurrency=2,
    )
    fp, lang = pg._resolve_benchmark(cfg)
    assert fp == "/tmp/a.json"
    assert lang == "zh"


def test_resolve_mem0_dialogue_format():
    base = dict(
        benchmark_file=None,
        output="o.jsonl",
        method="mem0",
        extractor_model=None,
        manager_model="m",
        answer_model="Qwen3-4B",
        embedding_model="embed",
        retrieve_topk=5,
        memory_token_limit=2048,
        memory_granularity="all",
        database_root=None,
        embedding_base_url="http://x",
        embedding_api_key="k",
        language=None,
        agent_trace_dir=None,
        parallel_episodes=1,
        rebuild_memory=False,
        manager_max_new_tokens=2048,
        mem0_extract_concurrency=8,
        relmem_relation_concurrency=8,
        answer_concurrency=2,
    )
    assert (
        pg._resolve_mem0_dialogue_format(
            pg.GenerateConfig(benchmark="locomo", mem0_dialogue_format="auto", **base)
        )
        == "named_speakers"
    )
    assert (
        pg._resolve_mem0_dialogue_format(
            pg.GenerateConfig(benchmark="lme_s", mem0_dialogue_format="auto", **base)
        )
        == "user_assistant"
    )
    assert (
        pg._resolve_mem0_dialogue_format(
            pg.GenerateConfig(
                benchmark="lme_s", mem0_dialogue_format="named_speakers", **base
            )
        )
        == "named_speakers"
    )


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
        agent_trace_dir=None,
        parallel_episodes=1,
        rebuild_memory=False,
        mem0_dialogue_format="auto",
        manager_max_new_tokens=2048,
        mem0_extract_concurrency=8,
        relmem_relation_concurrency=8,
        answer_concurrency=2,
    )
    with pytest.raises(ValueError):
        pg._resolve_benchmark(cfg)


def test_resolve_agent_trace_dir_uses_experiment_subdir():
    base = dict(
        benchmark_file=None,
        output="o.jsonl",
        method="mem0",
        extractor_model=None,
        manager_model="Qwen2.5-7B-Instruct",
        answer_model="Qwen3-4B",
        embedding_model="embed",
        retrieve_topk=5,
        memory_token_limit=2048,
        memory_granularity="4",
        database_root=None,
        embedding_base_url="http://x",
        embedding_api_key="k",
        language=None,
        agent_trace_dir="logs/answer_agent_trace",
        parallel_episodes=1,
        rebuild_memory=False,
        mem0_dialogue_format="auto",
        manager_max_new_tokens=2048,
        mem0_extract_concurrency=8,
        relmem_relation_concurrency=8,
        answer_concurrency=2,
    )
    cfg = pg.GenerateConfig(benchmark="locomo", **base)
    assert pg._resolve_agent_trace_dir(cfg) == f"logs/answer_agent_trace/{pg._build_experiment_name(cfg)}"
    cfg_none = pg.GenerateConfig(benchmark="locomo", **{**base, "agent_trace_dir": None})
    assert pg._resolve_agent_trace_dir(cfg_none) is None


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
    assert pe._judge_response_text(None) is None
    assert pe._judge_response_text("") is None
    assert pe._judge_response_text("  ") is None
    assert pe._judge_response_text({"content": None}) is None

    prompt = pe._build_judge_user_prompt(
        {
            "question": "q",
            "answer": "a",
            "model_answer": "m",
            "options": ["A", "B"],
            "golden_option": "A",
        },
        use_cot=True,
    )
    assert "Final answer" in prompt
    assert "Options" in prompt
