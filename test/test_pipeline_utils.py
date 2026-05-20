import pytest

import pipeline_evaluate as pe
import pipeline_generate as pg


def _cfg_base(**overrides):
    base = dict(
        benchmark_file=None,
        output="o.jsonl",
        extractor_model=None,
        manager_model=None,
        answer_model="Qwen3-4B",
        embedding_model="embed",
        retrieve_topk=5,
        memory_token_limit=2048,
        database_root=None,
        embedding_base_url="http://x",
        embedding_api_key="k",
        language=None,
        agent_trace_dir=None,
        parallel_episodes=1,
        rebuild_memory=False,
        dialogue_format="auto",
        manager_max_new_tokens=2048,
        fact_extract_concurrency=8,
        answer_concurrency=2,
        question_types=None,
        prebuilt_memory=False,
        agent_trace_label=None,
        hybrid_bm25_dense=False,
        hybrid_dense_weight=0.5,
        hybrid_bm25_weight=0.5,
        hybrid_pool_mult=4,
        hybrid_full_corpus_pool=False,
        unfused_rank_database_root=None,
        rerank_qwen3_vllm=False,
        rerank_qwen3_vllm_base_url=None,
        rerank_qwen3_vllm_api_key=None,
        rerank_qwen3_vllm_model="Qwen3-Reranker-0.6B",
        rerank_qwen3_vllm_timeout_s=120.0,
        rerank_top_k=None,
        answer_stratified_sample=0,
        answer_sample_seed=42,
        show_memory_time=True,
        require_lme_ingest_marker=False,
        ingest_marker_update_method="zep",
    )
    base.update(overrides)
    return base


def test_resolve_benchmark_with_explicit_file():
    cfg = pg.GenerateConfig(
        **_cfg_base(
            benchmark="custom",
            benchmark_file="/tmp/a.json",
            method="lme_prebuilt",
            memory_granularity="all",
            language="zh",
        ),
    )
    fp, lang = pg._resolve_benchmark(cfg)
    assert fp == "/tmp/a.json"
    assert lang == "zh"


def test_resolve_dialogue_format():
    base = _cfg_base(
        method="lme_prebuilt",
        memory_granularity="all",
    )
    assert (
        pg._resolve_dialogue_format(
            pg.GenerateConfig(benchmark="locomo", **base)
        )
        == "named_speakers"
    )
    assert (
        pg._resolve_dialogue_format(
            pg.GenerateConfig(benchmark="lme_s", **base)
        )
        == "user_assistant"
    )
    base_no_df = {k: v for k, v in base.items() if k != "dialogue_format"}
    assert (
        pg._resolve_dialogue_format(
            pg.GenerateConfig(
                benchmark="lme_s", dialogue_format="named_speakers", **base_no_df
            )
        )
        == "named_speakers"
    )


def test_resolve_benchmark_unknown_raises():
    cfg = pg.GenerateConfig(
        **_cfg_base(
            benchmark="not_exists",
            method="lme_prebuilt",
            memory_granularity="all",
        ),
    )
    with pytest.raises(ValueError):
        pg._resolve_benchmark(cfg)


def test_resolve_agent_trace_method_from_output_and_override():
    base_lme = dict(
        method="lme_prebuilt",
        memory_granularity="all",
    )
    cfg_pred = pg.GenerateConfig(
        benchmark="lme_o",
        **_cfg_base(output="eval/pred_add_all.jsonl", **base_lme),
    )
    assert pg._resolve_agent_trace_method(cfg_pred) == "add_all"
    cfg_explicit = pg.GenerateConfig(
        benchmark="lme_o",
        **_cfg_base(
            output="eval/pred_add_all.jsonl",
            agent_trace_label="relation_decision",
            **base_lme,
        ),
    )
    assert pg._resolve_agent_trace_method(cfg_explicit) == "relation_decision"
    cfg_method = pg.GenerateConfig(
        benchmark="lme_o",
        **_cfg_base(output="experiment/out.jsonl", **base_lme),
    )
    assert pg._resolve_agent_trace_method(cfg_method) == "lme_prebuilt"


def test_resolve_agent_trace_dir_is_agent_trace_dir_directly():
    base = _cfg_base(
        method="lme_prebuilt",
        memory_granularity="4",
        agent_trace_dir="logs/answer_agent_trace",
    )
    cfg = pg.GenerateConfig(benchmark="locomo", **base)
    assert pg._resolve_agent_trace_dir(cfg) == "logs/answer_agent_trace"
    cfg_none = pg.GenerateConfig(benchmark="locomo", **{**base, "agent_trace_dir": None})
    assert pg._resolve_agent_trace_dir(cfg_none) is None

    base_nested = _cfg_base(
        method="lme_prebuilt",
        memory_granularity="all",
        agent_trace_dir="logs/agent_trace/lme_s_cand0406_Qwen3-32B_0406/relation_decision",
    )
    cfg_nested = pg.GenerateConfig(benchmark="lme_s", **base_nested)
    assert (
        pg._resolve_agent_trace_dir(cfg_nested)
        == "logs/agent_trace/lme_s_cand0406_Qwen3-32B_0406/relation_decision"
    )


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
    redacted_no = (
        "<think>Reasoning here.</think>\n\nno"
    )
    redacted_yes = (
        "<think>Reasoning here.</think>\n\nyes"
    )
    assert pe._parse_verdict(redacted_no) is False
    assert pe._parse_verdict(redacted_yes) is True
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
        oqa_template="pipeline_eval_oqa_v2.jinja",
    )
    assert "Final answer" in prompt
    assert "Options" in prompt


def test_remove_episode_trace_jsonl_files_experiment_naming(tmp_path):
    from memory.tracing import remove_episode_trace_jsonl_files

    d = tmp_path / "trace"
    d.mkdir()
    (d / "ep1.jsonl").write_text("x\n", encoding="utf-8")
    (d / "ep1_abs.jsonl").write_text("y\n", encoding="utf-8")
    removed = remove_episode_trace_jsonl_files(
        log_dir=d,
        method="agent",
        history_name="ep1",
        use_experiment_naming=True,
    )
    assert len(removed) == 2
    assert not (d / "ep1.jsonl").exists()
