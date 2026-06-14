import sys
import types
import pytest


def _make_system(monkeypatch, backend, language="en", classify_label="EQV"):
    """构造 LmeCandidateRelationDecisionMemorySystem，绕过真实 backbone / __init__。"""
    from memory.candidate_ingest import memory_system as ms

    # 假 backend：记录是否被调用
    class FakeBackend:
        def __init__(self, *a, **k):
            self.calls = []
        def classify(self, old, new):
            self.calls.append((old, new))
            return classify_label

    monkeypatch.setattr(ms, "RelationClassifierBackend", FakeBackend)

    sysobj = ms.LmeCandidateRelationDecisionMemorySystem.__new__(
        ms.LmeCandidateRelationDecisionMemorySystem
    )
    sysobj.language = language
    sysobj._relation_backend = backend
    sysobj._relation_system_en_template = None
    sysobj._relation_system_zh_template = None
    sysobj._relation_user_template = None
    sysobj._relation_max_new_tokens = 256
    sysobj._rc_backend = FakeBackend() if backend == "classifier" else None

    # 假 llm_client：记录是否被调用
    class FakeLLM:
        def __init__(self):
            self.calls = []
        def get_response_chat(self, *a, **k):
            self.calls.append((a, k))
            return {"relation": "CON"}
    sysobj.llm_client = FakeLLM()
    return sysobj, ms


class _Trace:
    def log_llm_interaction(self, **k): pass


def test_classifier_backend_used(monkeypatch):
    s, ms = _make_system(monkeypatch, "classifier", classify_label="OSN")
    lab = s._classify_relation("old", "new", "scope", _Trace())
    assert lab == "OSN"
    assert s._rc_backend.calls == [("old", "new")]
    assert s.llm_client.calls == []          # 不调 LLM


def test_llm_backend_used(monkeypatch):
    s, ms = _make_system(monkeypatch, "llm")
    lab = s._classify_relation("old", "new", "scope", _Trace())
    assert lab == "CON"
    assert s.llm_client.calls != []          # 调 LLM


def test_language_guard_raises_for_non_english():
    from memory.candidate_ingest.memory_system import _check_relation_language
    with pytest.raises(ValueError):
        _check_relation_language("classifier", "zh")


def test_language_guard_allows_english():
    from memory.candidate_ingest.memory_system import _check_relation_language
    _check_relation_language("classifier", "en")  # no raise
    _check_relation_language("llm", "zh")          # no raise
