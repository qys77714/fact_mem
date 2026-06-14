"""relation_classifier 的薄封装：懒加载 + 线程安全，供 relation_decision 调用。

把 sys.path 接入、backbone 懒加载、并发加锁都隔离在这里；
对外只暴露 classify(old, new) -> 五分类标签字符串。
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_VALID = frozenset({"IND", "EQV", "NSO", "OSN", "CON"})

# <repo>/src/memory/candidate_ingest/relation_classifier_backend.py -> <repo>
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
_RC_DIR = os.path.join(_REPO_ROOT, "relation_classifier")


class RelationClassifierBackend:
    """持有懒加载的 RelationClassifier 单例 + 一把锁。

    RelationClassifier.predict 线程不安全（共享 backbone 前向），
    而调用方用 ThreadPoolExecutor 并发，故 classify 全程持锁。
    构造不加载 backbone；首次 classify 时才加载（双检）。
    """

    def __init__(self, backbone_path: Optional[str] = None,
                 device: Optional[str] = None) -> None:
        self._backbone_path = backbone_path
        self._device = device
        self._clf = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._clf is not None:
            return self._clf
        if _RC_DIR not in sys.path:
            sys.path.insert(0, _RC_DIR)
        from classifier import RelationClassifier  # noqa: E402
        self._clf = RelationClassifier(
            backbone_path=self._backbone_path,
            device=self._device,
        )
        return self._clf

    def classify(self, old: str, new: str) -> str:
        with self._lock:
            clf = self._ensure_loaded()
            try:
                out = clf.predict(old, new, return_probs=False)
            except Exception as exc:  # 单次推理失败回退 IND，不中断整批
                logger.warning("relation classifier predict failed: %s", exc)
                return "IND"
        label = str(out.get("label", "")).strip().upper()
        return label if label in _VALID else "IND"
