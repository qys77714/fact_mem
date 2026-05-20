"""Tests for ``ingest_candidates`` episode filtering by ``--question-types``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark import get_benchmark
from pipeline.ingest_candidates import history_names_for_question_type_filter


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _two_episode_lme_raw(tmp_path: Path) -> str:
    raw_dir = tmp_path / "data" / "raw_data"
    raw_fp = raw_dir / "longmemeval_filter.json"
    raw_data = [
        {
            "question_id": "ep_ku",
            "question": "ku?",
            "answer": "a1",
            "question_date": "2024/01/01 10:00",
            "question_type": "knowledge-update",
            "haystack_dates": ["2024/01/01 (Mon) 10:00"],
            "haystack_sessions": [[{"role": "user", "content": "hi", "has_answer": False}]],
        },
        {
            "question_id": "ep_sh",
            "question": "sh?",
            "answer": "a2",
            "question_date": "2024/01/02 10:00",
            "question_type": "single-hop",
            "haystack_dates": ["2024/01/02 (Tue) 10:00"],
            "haystack_sessions": [[{"role": "user", "content": "bye", "has_answer": False}]],
        },
    ]
    _write_json(raw_fp, raw_data)
    missing_pre = tmp_path / "data" / "preprocessed" / "longmemeval_filter_converted.json"
    assert not missing_pre.exists()
    get_benchmark("lme_o", file_path=str(missing_pre), lang="en")
    assert missing_pre.is_file()
    return str(missing_pre)


def test_history_names_for_question_type_filter_knowledge_update_only(tmp_path: Path) -> None:
    bm_file = _two_episode_lme_raw(tmp_path)
    names = history_names_for_question_type_filter(
        benchmark="lme_o",
        benchmark_file=bm_file,
        question_types_arg="knowledge-update",
    )
    assert names == frozenset({"ep_ku"})


def test_history_names_for_question_type_filter_none_means_no_filter(tmp_path: Path) -> None:
    bm_file = _two_episode_lme_raw(tmp_path)
    assert (
        history_names_for_question_type_filter(
            benchmark="lme_o",
            benchmark_file=bm_file,
            question_types_arg=None,
        )
        is None
    )
    assert (
        history_names_for_question_type_filter(
            benchmark="lme_o",
            benchmark_file=bm_file,
            question_types_arg="",
        )
        is None
    )


def test_history_names_for_question_type_filter_default_benchmark_file(tmp_path: Path) -> None:
    """Without benchmark_file, uses DEFAULT_BENCHMARK_DATASETS[benchmark] (may be missing in CI)."""
    try:
        names = history_names_for_question_type_filter(
            benchmark="lme_s",
            benchmark_file=None,
            question_types_arg="knowledge-update",
        )
    except (OSError, ValueError, FileNotFoundError):
        pytest.skip("default lme_s benchmark file not available")
    assert isinstance(names, frozenset)
    assert all(isinstance(x, str) for x in names)
