"""
用真实 embedding 与关系分类 LLM 跑 relation_decision 灌库路径（与 ``script/run_lme_ku_extract_apply_eval.sh``
+ ``src/pipeline/ingest_candidates.py`` 中 relation_decision 配置一致）。

候选目录：``MemDB/candidates/lme_o_Qwen3-8B_0406``，取文件名排序后的前 5 个 ``*.json`` episode。

需要环境变量（与灌库脚本相同）：
  - ``EMBEDDING_API_KEY``；可选 ``EMBEDDING_BASE_URL``
  - ``Qwen3-8B`` 走 VLLM：``VLLM_API_KEY``；可选 ``VLLM_BASE_URL``

未配置或缺少候选文件时跳过测试。建议：``uv run pytest test/memory/test_lme_relation_decision_candidates.py -v``。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from memory.candidate_ingest import (
    LmeCandidateRelationDecisionMemorySystem,
    apply_candidate_file,
    load_candidate_json,
)
from utils.env import load_env
from utils.llm_api import load_api_chat_completion


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LME_O_CANDIDATES_DIR = PROJECT_ROOT / "MemDB" / "candidates" / "lme_o_Qwen3-8B_0406"

# 与 run_lme_ku_extract_apply_eval.sh / ingest_candidates 默认值对齐
EMBEDDING_MODEL = "qwen3-embedding-8b"
MANAGER_MODEL = "Qwen3-8B"
RELATED_TOP_K = 5
RELATION_MAX_NEW_TOKENS = 256
MANAGER_MAX_NEW_TOKENS = 2048
RELATION_CONCURRENCY = 8


def _default_embedding_base_url() -> str:
    return os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")


def _integration_env_ready() -> bool:
    load_env(str(PROJECT_ROOT / ".env"))
    if not (os.getenv("EMBEDDING_API_KEY") or "").strip():
        return False
    if not (os.getenv("VLLM_API_KEY") or "").strip():
        return False
    return True


def _first_n_episode_json_paths(n: int) -> list[Path]:
    if not LME_O_CANDIDATES_DIR.is_dir():
        return []
    return sorted(LME_O_CANDIDATES_DIR.glob("*.json"))[:n]


@pytest.fixture(scope="module")
def relation_decision_memory(tmp_path_factory: pytest.TempPathFactory) -> LmeCandidateRelationDecisionMemorySystem:
    if not _integration_env_ready():
        pytest.skip(
            "需要 EMBEDDING_API_KEY 与 VLLM_API_KEY（Qwen3-8B），"
            "并可在项目根 .env 中配置 EMBEDDING_BASE_URL / VLLM_BASE_URL"
        )
    from openai import OpenAI

    db_root = tmp_path_factory.mktemp("lme_rd_real")
    embed_client = OpenAI(
        api_key=os.environ["EMBEDDING_API_KEY"],
        base_url=_default_embedding_base_url(),
    )
    llm_client = load_api_chat_completion(MANAGER_MODEL, async_=False)
    return LmeCandidateRelationDecisionMemorySystem(
        embed_model_name=EMBEDDING_MODEL,
        llm_client=llm_client,
        embed_client=embed_client,
        database_root=str(db_root),
        language="en",
        related_memory_top_k=RELATED_TOP_K,
        relation_concurrency=RELATION_CONCURRENCY,
        relation_max_new_tokens=RELATION_MAX_NEW_TOKENS,
        manager_max_new_tokens=MANAGER_MAX_NEW_TOKENS,
        trace_log_dir=None,
    )


def test_relation_decision_apply_first_five_lme_o_qwen3_8b_0406_real_api(
    relation_decision_memory: LmeCandidateRelationDecisionMemorySystem,
) -> None:
    if not LME_O_CANDIDATES_DIR.is_dir():
        pytest.skip(f"Candidates directory not found: {LME_O_CANDIDATES_DIR}")
    paths = _first_n_episode_json_paths(5)
    if not paths:
        pytest.skip(f"No *.json episodes under {LME_O_CANDIDATES_DIR}")

    for path in paths:
        payload = load_candidate_json(path)
        assert isinstance(payload.get("history_name"), str) and payload["history_name"].strip()
        chunks = payload.get("chunks") or []
        assert isinstance(chunks, list)

        stats = apply_candidate_file(relation_decision_memory, path)
        assert stats["history_name"] == payload["history_name"]
        assert stats["chunks"] == len(chunks)
        assert stats["facts_submitted"] >= 0
        assert 0 <= stats["memory_row_ops"] <= stats["facts_submitted"]
        if stats["facts_submitted"] > 0:
            assert stats["memory_row_ops"] >= 1
