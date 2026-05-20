"""Tests for unfused hybrid/dense ranking mapped to fused rows (dedupe)."""

import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from memory.baselines.lme_prebuilt import LmePrebuiltMemorySystem
from memory.storage.local_faiss import LocalFaissDatabase


class _ConstEmbedClient:
    """OpenAI-compatible client: every text embeds to the same vector."""

    def __init__(self, vec: list[float]) -> None:
        self._vec = [float(x) for x in vec]

    class _Emb:
        def __init__(self, parent: "_ConstEmbedClient") -> None:
            self._p = parent

        def create(self, input, model):
            class Item:
                def __init__(self, index: int, embedding: list[float]) -> None:
                    self.index = index
                    self.embedding = embedding

            class Resp:
                def __init__(self, data) -> None:
                    self.data = data

            v = self._p._vec
            return Resp([Item(i, v) for i in range(len(input))])

    @property
    def embeddings(self):
        return self._Emb(self)


def test_unfused_rank_maps_to_fused_dedupes_preserve_order(tmp_path):
    """Two pre-fusion hits in same bundle → one fused row; second bundle follows."""
    unfused_root = str(tmp_path / "u")
    fused_root = str(tmp_path / "f")
    ns = "ep1"
    db_u = LocalFaissDatabase(namespace=ns, database_root=unfused_root)
    z1 = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    p1 = db_u.add("a", "s", "t", {"memory_role": "primary"}, embedding=z1)
    z2 = np.array([[0.99, 0.01, 0.0, 0.0]], dtype=np.float32)
    e1 = db_u.add(
        "b",
        "s",
        "t",
        {"memory_role": "evidence", "parent_primary": p1},
        embedding=z2,
    )
    z3 = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    p2 = db_u.add("c", "s", "t", {"memory_role": "primary"}, embedding=z3)

    db_f = LocalFaissDatabase(namespace=ns, database_root=fused_root)
    zf1 = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    f1 = db_f.add(
        "fused1",
        "sf1",
        "t",
        {
            "memory_role": "primary",
            "lme_fused_bundle": True,
            "lme_fused_member_ids": [p1, e1],
        },
        embedding=zf1,
    )
    zf2 = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    f2 = db_f.add(
        "fused2",
        "sf2",
        "t",
        {"memory_role": "primary", "lme_fused_bundle": True, "lme_fused_member_ids": [p2]},
        embedding=zf2,
    )

    mem = LmePrebuiltMemorySystem(
        embed_model_name="m",
        embed_client=_ConstEmbedClient([1.0, 0.0, 0.0, 0.0]),
        database_root=fused_root,
        unfused_rank_database_root=unfused_root,
        use_hybrid_retrieval=False,
        hybrid_pool_mult=4,
    )
    out = mem.retrieve(ns, "query", "t", top_k=5)
    assert len(out) == 2
    assert out[0].memory_id == f1
    assert out[1].memory_id == f2
    assert out[0].text == "fused1"
    assert out[1].text == "fused2"
    assert out[0].score >= out[1].score


def test_build_pre_fusion_member_maps_from_fusion_module(tmp_path):
    from memory.fusion.lme_bundle_fusion import build_pre_fusion_member_to_fused_maps

    fused_root = str(tmp_path / "f")
    ns = "h"
    db = LocalFaissDatabase(namespace=ns, database_root=fused_root)
    emb = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    m1 = db.add("x", "s", "t", {"memory_role": "primary"}, embedding=emb)
    m2 = db.add("y", "s", "t", {"memory_role": "primary"}, embedding=emb)
    fid = db.add(
        "fused",
        "sf",
        "t",
        {
            "memory_role": "primary",
            "lme_fused_bundle": True,
            "lme_fused_member_ids": [m1, m2],
        },
        embedding=emb,
    )
    member_to_fused, fused_by_id = build_pre_fusion_member_to_fused_maps(db)
    assert member_to_fused[m1] == fid
    assert member_to_fused[m2] == fid
    assert fused_by_id[fid].text == "fused"
