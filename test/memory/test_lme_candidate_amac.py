"""Tests for A-MAC candidate ingest (admission + add_all-style writes)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from memory.candidate_ingest.apply import apply_candidate_episode_json
from memory.candidate_ingest.memory_system_amac import LmeCandidateAmacMemorySystem
from memory.admission.scorer import parse_amac_weights_arg


class _MockEmbedClient:
    class _Embeddings:
        def __init__(self, dim: int):
            self.dim = dim

        def create(self, input, model):
            class Item:
                def __init__(self, index, embedding):
                    self.index = index
                    self.embedding = embedding

            class Resp:
                def __init__(self, data):
                    self.data = data

            data = []
            for i, text in enumerate(input):
                vec = np.zeros(self.dim, dtype=np.float32)
                vec[i % self.dim] = 1.0
                if "overlap_token_xyz" in text.lower():
                    vec[0] = 5.0
                data.append(Item(i, vec.tolist()))
            return Resp(data)

    def __init__(self, dim: int = 8):
        self.embeddings = self._Embeddings(dim)


def test_parse_amac_weights_comma():
    w = parse_amac_weights_arg("0.1,0.1,0.1,0.1,0.6")
    assert w.shape == (5,)
    assert abs(float(np.sum(w)) - 1.0) < 1e-6


def test_amac_reject_no_primary_rows(tmp_path):
    llm = MagicMock()
    llm.get_response_chat.return_value = '{"utility": 0.0}'
    mem = LmeCandidateAmacMemorySystem(
        embed_model_name="mock",
        llm_client=llm,
        embed_client=_MockEmbedClient(dim=8),
        database_root=str(tmp_path),
        amac_weights="1,0,0,0,0",
        amac_threshold=0.9,
        amac_skip_utility=False,
        related_memory_top_k=1,
    )
    payload = {
        "history_name": "ep_amac_reject",
        "chunks": [
            {
                "chunk_index": 0,
                "session_index": 1,
                "session_date": "",
                "candidate_memories": ["some memory text overlap_token_xyz"],
            },
        ],
    }
    obs = {0: "line with overlap_token_xyz in context"}
    stats = apply_candidate_episode_json(mem, payload, observation_by_chunk_index=obs)
    db = mem._get_database("ep_amac_reject")
    assert stats["memory_row_ops"] == 0
    assert stats["amac_rejected"] == 1
    assert stats["amac_admitted"] == 0
    assert db.list_primary_texts_ordered() == []


def test_amac_admit_one_primary(tmp_path):
    llm = MagicMock()
    llm.get_response_chat.return_value = '{"utility": 1.0}'
    mem = LmeCandidateAmacMemorySystem(
        embed_model_name="mock",
        llm_client=llm,
        embed_client=_MockEmbedClient(dim=8),
        database_root=str(tmp_path),
        amac_weights="1,0,0,0,0",
        amac_threshold=0.5,
        amac_skip_utility=False,
        related_memory_top_k=1,
    )
    payload = {
        "history_name": "ep_amac_ok",
        "chunks": [
            {
                "chunk_index": 0,
                "session_index": 1,
                "session_date": "2024-01-15",
                "candidate_memories": ["user likes overlap_token_xyz coffee"],
            },
        ],
    }
    obs = {0: "user: I love overlap_token_xyz coffee drinks"}
    stats = apply_candidate_episode_json(mem, payload, observation_by_chunk_index=obs)
    db = mem._get_database("ep_amac_ok")
    assert stats["memory_row_ops"] == 1
    assert stats["amac_admitted"] == 1
    assert stats["amac_rejected"] == 0
    texts = db.list_primary_texts_ordered()
    assert len(texts) == 1
    assert "coffee" in texts[0]
