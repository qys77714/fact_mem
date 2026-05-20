import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from benchmark.base import ChatSession, ChatTurn
from memory import get_memory_system
from memory.baselines.lme_prebuilt import LmePrebuiltMemorySystem
from memory.candidate_ingest.apply_mem0 import apply_candidate_episode_mem0
from memory.mem0 import Mem0MemorySystem


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
                if "guitar" in text.lower():
                    vec[0] = 10.0
                data.append(Item(i, vec.tolist()))
            return Resp(data)

    def __init__(self, dim=8):
        self.embeddings = self._Embeddings(dim)


def test_get_memory_system_lme_prebuilt(tmp_path):
    embed = _MockEmbedClient(dim=4)
    mem = get_memory_system(
        method_name="lme_prebuilt",
        embed_model_name="mock",
        embed_client=embed,
        database_root=str(tmp_path),
    )
    assert isinstance(mem, LmePrebuiltMemorySystem)


def test_get_memory_system_unknown_raises():
    with pytest.raises(ValueError, match="Unknown memory method"):
        get_memory_system(
            method_name="rag_turn",
            embed_model_name="mock",
            embed_client=_MockEmbedClient(dim=4),
            database_root=".",
        )


class _DummyLLM:
    def get_response_chat(self, *args, **kwargs):
        raise AssertionError("LLM should not be called in transcript-only tests")


def test_mem0_chunk_transcript_user_assistant_maps_unknown_speakers(tmp_path):
    mem = Mem0MemorySystem(
        embed_model_name="mock",
        llm_client=_DummyLLM(),
        embed_client=_MockEmbedClient(dim=4),
        database_root=str(tmp_path),
        dialogue_format="user_assistant",
    )
    turns = [
        ChatTurn(speaker="Caroline", content="Hi"),
        ChatTurn(speaker="Melanie", content="Hello"),
    ]
    assert mem._build_chunk_transcript(turns) == "assistant: Hi\nassistant: Hello"


def test_mem0_chunk_transcript_named_speakers_keeps_labels(tmp_path):
    mem = Mem0MemorySystem(
        embed_model_name="mock",
        llm_client=_DummyLLM(),
        embed_client=_MockEmbedClient(dim=4),
        database_root=str(tmp_path),
        dialogue_format="named_speakers",
    )
    turns = [
        ChatTurn(speaker="Caroline", content="Hi"),
        ChatTurn(speaker="Melanie", content="Hello"),
    ]
    assert mem._build_chunk_transcript(turns) == "Caroline: Hi\nMelanie: Hello"


def test_mem0_fact_retrieval_prompt_switches_template():
    from memory.mem0.prompts import build_fact_retrieval_system_prompt

    multi = build_fact_retrieval_system_prompt(
        user_name="user", language="en", dialogue_format="named_speakers"
    )
    assert "multi-party" in multi.lower()

    single = build_fact_retrieval_system_prompt(
        user_name="user", language="en", dialogue_format="user_assistant"
    )
    assert "USER'S MESSAGES" in single


def test_apply_candidate_episode_mem0_smoke(tmp_path):
    """Mem0 update path runs with mocked LLM returning valid JSON operations."""
    llm = MagicMock()
    llm.get_response_chat.return_value = json.dumps(
        {"memory": [{"event": "ADD", "text": "User likes tea", "id": "0"}]}
    )
    mem = Mem0MemorySystem(
        embed_model_name="mock",
        llm_client=llm,
        embed_client=_MockEmbedClient(dim=8),
        database_root=str(tmp_path),
        language="en",
        manager_max_new_tokens=256,
        extract_concurrency=1,
    )
    payload = {
        "history_name": "ep1",
        "chunks": [
            {
                "chunk_index": 0,
                "session_index": 1,
                "session_date": "2024-01-01",
                "candidate_memories": ["User likes tea"],
            }
        ],
    }
    stats = apply_candidate_episode_mem0(mem, payload)
    assert stats["facts_submitted"] == 1
    assert stats["operation_batches"] == 1
    llm.get_response_chat.assert_called()

