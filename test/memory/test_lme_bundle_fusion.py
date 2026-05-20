"""Unit tests for LME post-ingest bundle listing (no LLM)."""

import shutil
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from memory.fusion.bundle_prompt_render import render_fusion_user_prompt
from memory.fusion.lme_bundle_fusion import (
    fuse_local_faiss_database,
    is_local_faiss_database_fused,
    list_depth_one_leaf_star_packages,
    list_disjoint_depth_one_partition_packages,
    list_fusion_packages,
    list_multimember_depth_one_partition_wave,
    list_whole_tree_fusion_packages,
)
from memory.base import RetrievedMemory
from memory.storage.local_faiss import LocalFaissDatabase


def test_list_fusion_packages_primary_and_evidence():
    tmp = project_root / "test" / "test_db_temp_fuse"
    if tmp.exists():
        shutil.rmtree(tmp)
    db = LocalFaissDatabase(namespace="h1", database_root=str(tmp))
    emb = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    pid = db.add(
        text="root fact",
        source_index="s1",
        time="2024/01/01",
        metadata={"memory_role": "primary"},
        embedding=emb,
    )
    db.add(
        text="equiv fact",
        source_index="s2",
        time="2024/01/02",
        metadata={
            "memory_role": "evidence",
            "parent_primary": pid,
            "lme_edge": "EQUIV",
        },
        embedding=emb,
    )
    pkgs = list_fusion_packages(db)
    assert len(pkgs) == 1
    assert len(pkgs[0]) == 2
    assert pkgs[0][0].text == "root fact"
    assert (pkgs[0][1].metadata or {}).get("memory_role") == "evidence"
    shutil.rmtree(tmp)


def test_empty_database_is_not_considered_fused():
    """Resume must not skip episodes with an empty/corrupt store (no rows)."""
    tmp = project_root / "test" / "test_db_temp_fuse_empty"
    if tmp.exists():
        shutil.rmtree(tmp)
    db = LocalFaissDatabase(namespace="h_empty", database_root=str(tmp))
    assert not is_local_faiss_database_fused(db)
    shutil.rmtree(tmp)


def test_list_fusion_packages_two_isolated_primaries():
    tmp = project_root / "test" / "test_db_temp_fuse2"
    if tmp.exists():
        shutil.rmtree(tmp)
    db = LocalFaissDatabase(namespace="h2", database_root=str(tmp))
    emb = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    db.add("a", "s", "t", {"memory_role": "primary"}, embedding=emb)
    db.add("b", "s", "t", {"memory_role": "primary"}, embedding=emb)
    pkgs = list_fusion_packages(db)
    assert len(pkgs) == 2
    assert all(len(p) == 1 for p in pkgs)
    assert list_depth_one_leaf_star_packages(db) == []
    shutil.rmtree(tmp)


def test_list_depth_one_same_as_full_when_primary_plus_leaf_evidence():
    tmp = project_root / "test" / "test_db_temp_fuse_depth1"
    if tmp.exists():
        shutil.rmtree(tmp)
    db = LocalFaissDatabase(namespace="hd1", database_root=str(tmp))
    emb = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    pid = db.add(
        text="root",
        source_index="s1",
        time="2024/01/01",
        metadata={"memory_role": "primary"},
        embedding=emb,
    )
    db.add(
        text="leaf ev",
        source_index="s2",
        time="2024/01/02",
        metadata={"memory_role": "evidence", "parent_primary": pid, "lme_edge": "EQUIV"},
        embedding=emb,
    )
    d1 = list_depth_one_leaf_star_packages(db)
    assert len(d1) == 1 and len(d1[0]) == 2
    assert d1[0][0].memory_id == pid
    shutil.rmtree(tmp)


def test_partition_depth_one_chain_singleton_r_then_ab():
    tmp = project_root / "test" / "test_db_temp_fuse_chain"
    if tmp.exists():
        shutil.rmtree(tmp)
    db = LocalFaissDatabase(namespace="hchain", database_root=str(tmp))
    emb = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    rid = db.add("R", "s0", "t0", {"memory_role": "primary"}, embedding=emb)
    aid = db.add(
        "A",
        "s1",
        "t1",
        metadata={
            "memory_role": "evidence",
            "parent_primary": rid,
            "lme_edge": "ATTACH",
            "evidence_depth": 1,
        },
        embedding=emb,
    )
    bid = db.add(
        "B",
        "s2",
        "t2",
        metadata={
            "memory_role": "evidence",
            "parent_primary": aid,
            "lme_edge": "ATTACH",
            "evidence_depth": 2,
        },
        embedding=emb,
    )
    blocks = list_disjoint_depth_one_partition_packages(db)
    lens = sorted(len(b) for b in blocks)
    assert lens == [1, 2]
    multi = list_multimember_depth_one_partition_wave(db)
    assert len(multi) == 1
    assert {m.memory_id for m in multi[0]} == {aid, bid}
    assert multi[0][0].memory_id == aid
    shutil.rmtree(tmp)


def test_list_whole_tree_fusion_packages_chain_includes_root():
    """整树分包：R→A→B 为单包 [R,A,B]，与深度≤1 分块不同。"""
    tmp = project_root / "test" / "test_db_temp_fuse_whole_chain"
    if tmp.exists():
        shutil.rmtree(tmp)
    db = LocalFaissDatabase(namespace="hwhole", database_root=str(tmp))
    emb = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    rid = db.add("R", "s0", "t0", {"memory_role": "primary"}, embedding=emb)
    aid = db.add(
        "A",
        "s1",
        "t1",
        {
            "memory_role": "evidence",
            "parent_primary": rid,
            "lme_edge": "ATTACH",
            "evidence_depth": 1,
        },
        embedding=emb,
    )
    bid = db.add(
        "B",
        "s2",
        "t2",
        {
            "memory_role": "evidence",
            "parent_primary": aid,
            "lme_edge": "ATTACH",
            "evidence_depth": 2,
        },
        embedding=emb,
    )
    whole = list_whole_tree_fusion_packages(db)
    assert len(whole) == 1
    assert [m.memory_id for m in whole[0]] == [rid, aid, bid]
    shutil.rmtree(tmp)


def test_partition_star_r_with_leaves_and_branch_wave():
    """R 下叶 B,C 与分支 A→D,E：一轮 multimember 为 {R,B,C} 与 {A,D,E}，不交。"""
    tmp = project_root / "test" / "test_db_temp_fuse_fork"
    if tmp.exists():
        shutil.rmtree(tmp)
    db = LocalFaissDatabase(namespace="hfork", database_root=str(tmp))
    emb = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    rid = db.add("R", "s0", "t0", {"memory_role": "primary"}, embedding=emb)
    bid = db.add(
        "B",
        "sb",
        "tb",
        {"memory_role": "evidence", "parent_primary": rid, "lme_edge": "ATTACH", "evidence_depth": 1},
        embedding=emb,
    )
    cid = db.add(
        "C",
        "sc",
        "tc",
        {"memory_role": "evidence", "parent_primary": rid, "lme_edge": "ATTACH", "evidence_depth": 1},
        embedding=emb,
    )
    aid = db.add(
        "A",
        "sa",
        "ta",
        {"memory_role": "evidence", "parent_primary": rid, "lme_edge": "ATTACH", "evidence_depth": 1},
        embedding=emb,
    )
    did = db.add(
        "D",
        "sd",
        "td",
        {"memory_role": "evidence", "parent_primary": aid, "lme_edge": "ATTACH", "evidence_depth": 2},
        embedding=emb,
    )
    eid = db.add(
        "E",
        "se",
        "te",
        {"memory_role": "evidence", "parent_primary": aid, "lme_edge": "ATTACH", "evidence_depth": 2},
        embedding=emb,
    )
    multi = list_multimember_depth_one_partition_wave(db)
    assert len(multi) == 2
    by_center = {p[0].memory_id: {m.memory_id for m in p} for p in multi}
    assert by_center[rid] == {rid, bid, cid}
    assert by_center[aid] == {aid, did, eid}
    shutil.rmtree(tmp)


class _MockChat:
    def get_response_chat(self, messages, max_new_tokens=0, temperature=0, verbose=False):
        return "fused"


def test_fuse_local_faiss_whole_tree_chain():
    """整树：R→A→B 同一 prompt 融成一条 primary；库内仅保留该融合行。"""
    tmp = project_root / "test" / "test_db_temp_fuse_iter"
    if tmp.exists():
        shutil.rmtree(tmp)
    db = LocalFaissDatabase(namespace="hit", database_root=str(tmp))
    emb = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    def embed_fn(texts):
        return np.tile(emb, (len(texts), 1)).astype(np.float32)

    rid = db.add("R", "s0", "t0", {"memory_role": "primary"}, embedding=emb.copy())
    aid = db.add(
        "A",
        "s1",
        "t1",
        {
            "memory_role": "evidence",
            "parent_primary": rid,
            "lme_edge": "ATTACH",
            "evidence_depth": 1,
        },
        embedding=emb.copy(),
    )
    db.add(
        "B",
        "s2",
        "t2",
        {
            "memory_role": "evidence",
            "parent_primary": aid,
            "lme_edge": "ATTACH",
            "evidence_depth": 2,
        },
        embedding=emb.copy(),
    )
    st = fuse_local_faiss_database(
        db,
        embed_fn,
        _MockChat(),
        language="en",
        fuse_max_new_tokens=32,
        package_concurrency=2,
    )
    assert st.get("skipped") is False
    assert st["rounds"] == 1
    assert st["packages"] == 1
    assert st["fusion_strategy"] == "whole_tree_single_wave"
    assert is_local_faiss_database_fused(db)
    rows = db.list_all_memories(sort_by_time=False)
    assert len(rows) == 1
    assert (rows[0].metadata or {}).get("memory_role") == "primary"
    assert all((m.metadata or {}).get("lme_fused_bundle") for m in rows)
    shutil.rmtree(tmp)


def test_fuse_local_faiss_whole_tree_star_one_package():
    """R 下叶 B,C 与分支 A→D,E：整棵树一包，得到一条融合记忆。"""
    tmp = project_root / "test" / "test_db_temp_fuse_star2"
    if tmp.exists():
        shutil.rmtree(tmp)
    db = LocalFaissDatabase(namespace="hstar2", database_root=str(tmp))
    emb = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    def embed_fn(texts):
        return np.tile(emb, (len(texts), 1)).astype(np.float32)

    rid = db.add("R", "s0", "t0", {"memory_role": "primary"}, embedding=emb.copy())
    for txt, sid in (("B", "sb"), ("C", "sc")):
        db.add(
            txt,
            sid,
            "tb",
            {"memory_role": "evidence", "parent_primary": rid, "lme_edge": "ATTACH", "evidence_depth": 1},
            embedding=emb.copy(),
        )
    aid = db.add(
        "A",
        "sa",
        "ta",
        {"memory_role": "evidence", "parent_primary": rid, "lme_edge": "ATTACH", "evidence_depth": 1},
        embedding=emb.copy(),
    )
    for txt, sid in (("D", "sd"), ("E", "se")):
        db.add(
            txt,
            sid,
            "td",
            {"memory_role": "evidence", "parent_primary": aid, "lme_edge": "ATTACH", "evidence_depth": 2},
            embedding=emb.copy(),
        )
    st = fuse_local_faiss_database(db, embed_fn, _MockChat(), language="en", fuse_max_new_tokens=32)
    assert st["skipped"] is False
    assert st["rounds"] == 1
    assert st["packages"] == 1
    rows = db.list_all_memories(sort_by_time=False)
    assert len(rows) == 1
    assert all((m.metadata or {}).get("lme_fused_bundle") for m in rows)
    assert is_local_faiss_database_fused(db)
    shutil.rmtree(tmp)


def test_render_fusion_user_prompt_zh_multi_member():
    a = RetrievedMemory(
        memory_id="m1",
        text="主事实内容",
        source_index="s0",
        time="2024/01/01",
        score=0.0,
        metadata={"memory_role": "primary"},
    )
    b = RetrievedMemory(
        memory_id="m2",
        text="等价补充",
        source_index="s1",
        time="2024/01/02",
        score=0.0,
        metadata={"memory_role": "evidence", "lme_edge": "EQUIV", "parent_primary": "m1"},
    )
    out = render_fusion_user_prompt([a, b], language="zh")
    assert "主事实内容" in out and "等价补充" in out
    assert "树形节点列表" in out or "记忆树" in out
    assert "包内树" in out and "2→1" in out
    assert "父节点（第 1 条）" in out


def test_render_fusion_user_prompt_zh_update_uses_beigengxin_label():
    root = RetrievedMemory(
        memory_id="r1",
        text="当前住在柏林。",
        source_index="s0",
        time="2024-06-01",
        score=0.0,
        metadata={"memory_role": "primary"},
    )
    old = RetrievedMemory(
        memory_id="e1",
        text="此前登记住址为慕尼黑。",
        source_index="s1",
        time="2024-01-01",
        score=0.0,
        metadata={
            "memory_role": "evidence",
            "lme_edge": "UPDATE",
            "parent_primary": "r1",
            "evidence_depth": 1,
        },
    )
    out = render_fusion_user_prompt([root, old], language="zh")
    assert "【被更新】" in out
    assert "树形节点列表" in out or "记忆树" in out


def test_render_fusion_user_prompt_nested_parent_zh():
    root = RetrievedMemory(
        memory_id="p1",
        text="根",
        source_index="s0",
        time="t0",
        score=0.0,
        metadata={"memory_role": "primary"},
    )
    mid = RetrievedMemory(
        memory_id="e1",
        text="中层",
        source_index="s1",
        time="t1",
        score=0.0,
        metadata={
            "memory_role": "evidence",
            "lme_edge": "ATTACH",
            "parent_primary": "p1",
            "evidence_depth": 1,
        },
    )
    leaf = RetrievedMemory(
        memory_id="e2",
        text="叶子",
        source_index="s2",
        time="t2",
        score=0.0,
        metadata={
            "memory_role": "evidence",
            "lme_edge": "ATTACH",
            "parent_primary": "e1",
            "evidence_depth": 2,
        },
    )
    out = render_fusion_user_prompt([root, mid, leaf], language="zh")
    assert "3→2" in out
    assert "【附属】对父节点（第 2 条）的补充、细化或从属信息" in out
    assert "从属" in out  # 模板措辞可能为「从属于该父分支」或「…从属信息」
