import numpy as np
import pytest

from benchmark.base import ChatSession, ChatTurn
from memory import get_memory_system
from memory.baselines.only_query import OnlyQueryMemorySystem
from memory.baselines.rag import RagMemorySystem


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


def test_only_query_memory_always_empty():
    mem = OnlyQueryMemorySystem()
    mem.store_session("u1", 1, ChatSession(session_date="2024", turns=[]))
    assert mem.retrieve("u1", "anything", "2024", top_k=3) == []


def test_rag_memory_store_and_retrieve(tmp_path):
    mem = RagMemorySystem(
        granularity=1,
        embed_client=_MockEmbedClient(dim=8),
        embed_model_name="mock",
        database_root=str(tmp_path),
    )

    sess = ChatSession(
        session_date="2024-01-01",
        turns=[
            ChatTurn(speaker="user", content="I want to learn guitar"),
            ChatTurn(speaker="assistant", content="Great"),
        ],
    )
    mem.store_session("u1", 1, sess)

    results = mem.retrieve("u1", "guitar budget", "2024-01-02", top_k=1)
    assert len(results) == 1
    assert "guitar" in results[0].text.lower()


def test_rag_memory_granularity_validation():
    with pytest.raises(ValueError):
        RagMemorySystem(granularity=0)
    with pytest.raises(ValueError):
        RagMemorySystem(granularity="bad")


def test_get_memory_system_factory_basics(tmp_path):
    embed = _MockEmbedClient(dim=4)
    full_context = get_memory_system(
        method_name="full_context",
        embed_model_name="mock",
        embed_client=embed,
        database_root=str(tmp_path),
    )
    assert full_context.__class__.__name__ == "FullContextMemorySystem"

    only_query = get_memory_system(
        method_name="only_query",
        embed_model_name="mock",
        embed_client=embed,
        database_root=str(tmp_path),
    )
    assert isinstance(only_query, OnlyQueryMemorySystem)


def test_get_memory_system_rag_alias_behavior(tmp_path):
    rag_turn = get_memory_system(
        method_name="rag_turn",
        embed_model_name="mock",
        embed_client=_MockEmbedClient(dim=4),
        database_root=str(tmp_path),
    )
    assert isinstance(rag_turn, RagMemorySystem)
    assert rag_turn.granularity == 1

    rag_session = get_memory_system(
        method_name="rag_session",
        embed_model_name="mock",
        embed_client=_MockEmbedClient(dim=4),
        database_root=str(tmp_path),
    )
    assert isinstance(rag_session, RagMemorySystem)
    assert rag_session.granularity == "all"
