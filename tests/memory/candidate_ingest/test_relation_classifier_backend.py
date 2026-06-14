import sys
import types
import pytest
from memory.candidate_ingest.relation_classifier_backend import RelationClassifierBackend


def _install_fake_classifier(monkeypatch, label="EQV", calls=None, raise_exc=None):
    """在 sys.modules 注入假的 classifier 模块，避免加载真实 backbone。"""
    instances = {"count": 0}

    class FakeRC:
        def __init__(self, *a, **k):
            instances["count"] += 1

        def predict(self, old, new, return_probs=True):
            if calls is not None:
                calls.append((old, new))
            if raise_exc is not None:
                raise raise_exc
            return {"label": label, "label_id": 1, "probs": {}}

    mod = types.ModuleType("classifier")
    mod.RelationClassifier = FakeRC
    monkeypatch.setitem(sys.modules, "classifier", mod)
    return instances


def test_classify_returns_label(monkeypatch):
    _install_fake_classifier(monkeypatch, label="CON")
    b = RelationClassifierBackend()
    assert b.classify("I live in Beijing.", "I moved to Shanghai.") == "CON"


def test_lazy_load_only_once(monkeypatch):
    inst = _install_fake_classifier(monkeypatch, label="IND")
    b = RelationClassifierBackend()
    assert inst["count"] == 0          # 构造不加载
    b.classify("a", "b")
    b.classify("c", "d")
    assert inst["count"] == 1          # 多次 classify 只加载一次


def test_predict_exception_falls_back_to_ind(monkeypatch):
    _install_fake_classifier(monkeypatch, raise_exc=RuntimeError("boom"))
    b = RelationClassifierBackend()
    assert b.classify("a", "b") == "IND"


def test_has_lock(monkeypatch):
    _install_fake_classifier(monkeypatch)
    b = RelationClassifierBackend()
    import threading
    assert isinstance(b._lock, type(threading.Lock()))
