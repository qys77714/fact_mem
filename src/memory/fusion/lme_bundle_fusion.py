"""
将同一 episode 向量库中的「关系树」打成包；多成员包经 LLM 合成一段融合文本，删旧行后只保留融合行，供答题阶段 dense 检索。

融合策略（默认）：**单轮整树**——对每个**无 parent_primary 的根 primary**，将其与
``collect_evidence_descendants`` 得到的**整条 evidence 子树（任意深度）**放在**同一包**内，
**一次** ``render_fusion_user_prompt`` / LLM 调用融合为一条记忆；episode 内多棵不相交的根树则各调用一次。
**不会**把本轮产生的多条融合行再并成一包。

单结点根（仅一条 primary、无 evidence）在收尾阶段单独打 ``lme_fused_bundle``（无 LLM）。

深度 ≤1 分块列举 API 保留为 ``list_disjoint_depth_one_partition_packages`` / ``list_multimember_depth_one_partition_wave``（对照、测试）。
``list_fusion_packages``：每条 primary（含非根）各列一包；默认融合路径请用 ``list_whole_tree_fusion_packages``。
"""

from __future__ import annotations

import logging
import re
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Sequence, Set, Tuple

import numpy as np

from memory.base import RetrievedMemory
from memory.storage.local_faiss import LocalFaissDatabase

from memory.candidate_ingest.cas_update import is_cascade_root

from .bundle_prompt_render import _is_primary_meta, render_fusion_user_prompt

logger = logging.getLogger(__name__)

# ``fuse_local_faiss_database`` 写入 stats / marker 的策略名（单轮整树，不递归合并融合行）
LME_FUSION_STRATEGY_WHOLE_TREE = "whole_tree_single_wave"


def list_fusion_packages(db: LocalFaissDatabase) -> List[List[RetrievedMemory]]:
    """每条 primary 与其所有 evidence 后代（BFS）组成一包；顺序为 [root] + descendants。"""
    all_mems = db.list_all_memories(sort_by_time=False)
    roots = [m for m in all_mems if _is_primary_meta(m.metadata)]
    packages: List[List[RetrievedMemory]] = []
    for root in roots:
        desc = db.collect_evidence_descendants(root.memory_id)
        packages.append([root] + desc)
    return packages


def _parent_primary_str(meta: Dict[str, Any] | None) -> str | None:
    if not meta:
        return None
    pp = meta.get("parent_primary")
    if not isinstance(pp, str):
        return None
    s = pp.strip()
    return s or None


def _center_outputs_primary_row(center: RetrievedMemory) -> bool:
    """True iff fused row should be a top-level primary (no parent_primary on fused row)."""
    meta = center.metadata or {}
    if not _is_primary_meta(meta):
        return False
    return _parent_primary_str(meta) is None


def _children_map(db: LocalFaissDatabase) -> Tuple[Dict[str, RetrievedMemory], Dict[str, List[str]]]:
    all_mems = db.list_all_memories(sort_by_time=False)
    by_id: Dict[str, RetrievedMemory] = {m.memory_id: m for m in all_mems}
    children: Dict[str, List[str]] = {}
    for m in all_mems:
        pp = _parent_primary_str(m.metadata)
        if pp is not None and pp in by_id:
            children.setdefault(pp, []).append(m.memory_id)
    for pid in children:
        children[pid].sort()
    return by_id, children


def _root_primary_ids(all_mems: List[RetrievedMemory]) -> List[str]:
    out: List[str] = []
    for m in all_mems:
        if _is_primary_meta(m.metadata) and _parent_primary_str(m.metadata) is None:
            out.append(m.memory_id)
    return sorted(out)


def _bfs_depth_from_roots(children: Dict[str, List[str]], root_ids: Sequence[str]) -> Dict[str, int]:
    """每个结点到其所在连通分量根 primary 的边数（根为 0）。"""
    depth: Dict[str, int] = {}
    for rid in root_ids:
        dq: deque[Tuple[str, int]] = deque([(rid, 0)])
        seen: Set[str] = {rid}
        while dq:
            mid, d = dq.popleft()
            depth[mid] = d
            for c in children.get(mid, ()):
                if c not in seen:
                    seen.add(c)
                    dq.append((c, d + 1))
    return depth


def list_disjoint_depth_one_partition_packages(db: LocalFaissDatabase) -> List[List[RetrievedMemory]]:
    """
    对当前库中「自顶向下」一轮划分：深度 ≤ 1 的不相交块。

    - 若中心有叶孩子：一块 ``[中心] + 叶孩子``（均为该中心直接子结点中的叶）。
    - 若中心仅有非叶孩子：中心单独成块 ``[中心]``，再递归各非叶子树。
    - 若中心无孩子：``[中心]``。

    块与块之间结点不交；顺序为 DFS 先序。仅用于列举；融合时只取 ``len >= 2`` 的块。
    """
    all_mems = db.list_all_memories(sort_by_time=False)
    by_id, children = _children_map(db)
    roots = _root_primary_ids(all_mems)
    packages: List[List[RetrievedMemory]] = []

    def is_leaf(mid: str) -> bool:
        return len(children.get(mid, ())) == 0

    def walk(center_id: str) -> None:
        if center_id not in by_id:
            return
        ch = children.get(center_id, [])
        leaf_ch = [c for c in ch if is_leaf(c)]
        nonleaf_ch = [c for c in ch if not is_leaf(c)]
        if leaf_ch:
            members = [by_id[center_id]] + [by_id[c] for c in leaf_ch if c in by_id]
            packages.append(members)
        elif ch:
            packages.append([by_id[center_id]])
        else:
            packages.append([by_id[center_id]])
        for c in nonleaf_ch:
            walk(c)

    for rid in roots:
        walk(rid)
    return packages


def list_multimember_depth_one_partition_wave(db: LocalFaissDatabase) -> List[List[RetrievedMemory]]:
    """当前一轮中需 LLM 融合的包：分块后仅 ``len >= 2``；按中心深度升序（祖先先于后代）便于写库时 parent 已存在。"""
    blocks = list_disjoint_depth_one_partition_packages(db)
    multi = [b for b in blocks if len(b) > 1]
    if not multi:
        return []
    _, children = _children_map(db)
    roots = _root_primary_ids(db.list_all_memories(sort_by_time=False))
    depth = _bfs_depth_from_roots(children, roots)

    def sort_key(pack: List[RetrievedMemory]) -> Tuple[int, str]:
        cid = pack[0].memory_id
        return (depth.get(cid, 0), cid)

    return sorted(multi, key=sort_key)


def list_depth_one_leaf_star_packages(db: LocalFaissDatabase) -> List[List[RetrievedMemory]]:
    """兼容旧名：等价于 ``list_multimember_depth_one_partition_wave``。"""
    return list_multimember_depth_one_partition_wave(db)


def list_whole_tree_fusion_packages(db: LocalFaissDatabase) -> List[List[RetrievedMemory]]:
    """
    每个无 ``parent_primary`` 的根 primary 与其**全部** evidence 后代（任意深度，BFS）组成一包；
    顺序为 ``[root] + collect_evidence_descendants(root)``。

    仅返回成员数 ``>= 2`` 的包；单结点根留给 ``fuse_local_faiss_database`` 收尾阶段无 LLM 打标。
    多棵根树按根 ``memory_id`` 升序排列，便于确定性与并行写库（树与树之间无父子依赖）。
    """
    all_mems = db.list_all_memories(sort_by_time=False)
    by_id: Dict[str, RetrievedMemory] = {m.memory_id: m for m in all_mems}
    out: List[List[RetrievedMemory]] = []
    for rid in _root_primary_ids(all_mems):
        if rid not in by_id:
            continue
        root = by_id[rid]
        if is_cascade_root(root.metadata):
            continue
        desc = db.collect_evidence_descendants(rid)
        pack = [root] + desc
        if len(pack) >= 2:
            out.append(pack)
    return out


def _strip_fenced_text(raw: Any) -> str:
    if raw is None:
        return ""
    t = raw if isinstance(raw, str) else str(raw)
    t = t.strip()
    m = re.match(r"^```(?:\w*)?\s*([\s\S]*?)```\s*$", t)
    if m:
        return m.group(1).strip()
    return t


def _fuse_bundle_with_llm(
    members: Sequence[RetrievedMemory],
    *,
    llm_client: Any,
    language: str,
    max_new_tokens: int,
    bundle_template_en: str | None = None,
    bundle_template_zh: str | None = None,
    edge_labels_en: str | None = None,
    edge_labels_zh: str | None = None,
) -> str:
    user_content = render_fusion_user_prompt(
        members,
        language=language,
        bundle_template_en=bundle_template_en,
        bundle_template_zh=bundle_template_zh,
        edge_labels_en=edge_labels_en,
        edge_labels_zh=edge_labels_zh,
    )
    messages = [{"role": "user", "content": user_content}]
    try:
        raw = llm_client.get_response_chat(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=0,
            verbose=False,
        )
    except Exception as exc:
        logger.warning("fuse bundle LLM failed, falling back to concatenation: %s", exc)
        return "\n".join(m.text.strip() for m in members if m.text.strip())

    text = _strip_fenced_text(raw)
    return text if text else "\n".join(m.text.strip() for m in members if m.text.strip())


def _pick_time(members: Sequence[RetrievedMemory]) -> str:
    times = [str(m.time or "").strip() for m in members if m.time]
    return max(times) if times else ""


def _fuse_one_package(
    members: List[RetrievedMemory],
    *,
    llm_client: Any,
    language: str,
    fuse_max_new_tokens: int,
    bundle_template_en: str | None = None,
    bundle_template_zh: str | None = None,
    edge_labels_en: str | None = None,
    edge_labels_zh: str | None = None,
) -> Tuple[List[str], Tuple[str, str, str, Dict[str, Any]]]:
    """Return (memory_ids to delete, row for db.add: text, source_index, time, metadata)."""
    mids = [m.memory_id for m in members]
    center = members[0]
    if len(members) == 1:
        fused_text = (center.text or "").strip()
    else:
        fused_text = _fuse_bundle_with_llm(
            members,
            llm_client=llm_client,
            language=language,
            max_new_tokens=fuse_max_new_tokens,
            bundle_template_en=bundle_template_en,
            bundle_template_zh=bundle_template_zh,
            edge_labels_en=edge_labels_en,
            edge_labels_zh=edge_labels_zh,
        )
    fused_text = (fused_text or "").strip()
    if not fused_text:
        fused_text = "\n".join((m.text or "").strip() for m in members if (m.text or "").strip())

    meta0 = dict(center.metadata or {})
    base_meta = {k: v for k, v in meta0.items() if k not in ("parent_primary", "lme_edge", "evidence_depth")}
    if _center_outputs_primary_row(center):
        fused_meta: Dict[str, Any] = {
            **base_meta,
            "memory_role": "primary",
            "lme_fused_bundle": True,
            "lme_fused_member_ids": mids,
            "lme_fused_member_count": len(members),
        }
    else:
        pp = _parent_primary_str(meta0)
        if not pp:
            logger.warning(
                "fuse center %r has no parent_primary but is not a root primary; emitting primary row",
                center.memory_id,
            )
            fused_meta = {
                **base_meta,
                "memory_role": "primary",
                "lme_fused_bundle": True,
                "lme_fused_member_ids": mids,
                "lme_fused_member_count": len(members),
            }
        else:
            edge = meta0.get("lme_edge")
            depth = meta0.get("evidence_depth")
            if not isinstance(depth, int) or depth < 1:
                depth = 1
            fused_meta = {
                **base_meta,
                "memory_role": "evidence",
                "parent_primary": pp,
                "lme_fused_bundle": True,
                "lme_fused_member_ids": mids,
                "lme_fused_member_count": len(members),
            }
            if edge is not None:
                fused_meta["lme_edge"] = edge
            fused_meta["evidence_depth"] = depth

    time_s = _pick_time(members)
    src_idx = f"fused_{uuid.uuid4().hex[:16]}"
    return mids, (fused_text, src_idx, time_s, fused_meta)


def _rewire_parent_primary_after_fusion(
    db: LocalFaissDatabase,
    mid_to_fused: Dict[str, str],
    skip_ids: Set[str],
) -> int:
    """Rewrite parent_primary for survivors pointing into deleted ids. Returns update count."""
    if not mid_to_fused:
        return 0
    n = 0
    for mem in db.list_all_memories(sort_by_time=False):
        if mem.memory_id in skip_ids:
            continue
        meta = mem.metadata or {}
        pp = meta.get("parent_primary")
        if not isinstance(pp, str):
            continue
        if pp not in mid_to_fused:
            continue
        db.update_memory(mem.memory_id, metadata_updates={"parent_primary": mid_to_fused[pp]})
        n += 1
    return n


def _database_already_fused(db: LocalFaissDatabase) -> bool:
    """True iff 库内每一行都已带 ``lme_fused_bundle``（允许根 primary 下仍挂融合子 evidence）。"""
    all_mems = db.list_all_memories(sort_by_time=False)
    # Empty store is not "successfully fused" — e.g. interrupted copy/crash leaves no rows;
    # treating it as fused caused fuse_lme_memory_bundles resume to skip those episodes.
    if not all_mems:
        return False
    for m in all_mems:
        if not (m.metadata or {}).get("lme_fused_bundle"):
            return False
    return True


def is_local_faiss_database_fused(db: LocalFaissDatabase) -> bool:
    """True if this namespace has already been through bundle fusion (every row marked ``lme_fused_bundle``)."""
    return _database_already_fused(db)


def build_pre_fusion_member_to_fused_maps(
    db: LocalFaissDatabase,
) -> Tuple[Dict[str, str], Dict[str, RetrievedMemory]]:
    """
    For a fused namespace: map each pre-fusion ``memory_id`` (from ``lme_fused_member_ids``) to the
    fused row's ``memory_id``, and collect a fused ``RetrievedMemory`` per fused row (score = 0).
    """
    member_to_fused: Dict[str, str] = {}
    fused_by_id: Dict[str, RetrievedMemory] = {}
    for mem in db.list_all_memories(sort_by_time=False):
        meta = mem.metadata or {}
        if not meta.get("lme_fused_bundle"):
            continue
        mids = meta.get("lme_fused_member_ids")
        if not mids or not isinstance(mids, (list, tuple)):
            continue
        fid = mem.memory_id
        fused_by_id[fid] = RetrievedMemory(
            memory_id=mem.memory_id,
            text=mem.text,
            source_index=mem.source_index,
            time=mem.time,
            score=0.0,
            metadata=dict(meta),
        )
        for mid in mids:
            s = str(mid)
            prev = member_to_fused.get(s)
            if prev is not None and prev != fid:
                logger.warning(
                    "lme_fused_member_ids: duplicate member id %r (fused %s vs %s); using latter",
                    s,
                    prev,
                    fid,
                )
            member_to_fused[s] = fid
    return member_to_fused, fused_by_id


def fuse_local_faiss_database(
    db: LocalFaissDatabase,
    embed_fn: Callable[[List[str]], np.ndarray],
    llm_client: Any,
    *,
    language: str = "en",
    fuse_max_new_tokens: int = 512,
    package_concurrency: int = 1,
    bundle_template_en: str | None = None,
    bundle_template_zh: str | None = None,
    edge_labels_en: str | None = None,
    edge_labels_zh: str | None = None,
) -> Dict[str, Any]:
    """
    将当前 namespace 内每棵根树（根 primary + 全部 evidence 子树）**单轮**各融合一次：
    每棵树一次 LLM、一条融合行；**不再**对本轮产生的多条融合行做第二轮合并。

    ``package_concurrency``：同一轮内并行调用 LLM（多棵树时）；删库、写库、重连 parent_primary 仍串行。
    """
    if _database_already_fused(db):
        return {
            "skipped": True,
            "reason": "already_fused",
            "packages": 0,
            "added": 0,
            "rounds": 0,
            "fusion_strategy": LME_FUSION_STRATEGY_WHOLE_TREE,
        }

    all_mems = db.list_all_memories(sort_by_time=False)
    if not all_mems:
        return {
            "skipped": True,
            "reason": "empty",
            "packages": 0,
            "added": 0,
            "rounds": 0,
            "fusion_strategy": LME_FUSION_STRATEGY_WHOLE_TREE,
        }

    rounds = 0
    total_packages = 0
    total_deleted = 0
    total_added = 0

    def _work(members: List[RetrievedMemory]) -> Tuple[List[str], Tuple[str, str, str, Dict[str, Any]]]:
        return _fuse_one_package(
            members,
            llm_client=llm_client,
            language=language,
            fuse_max_new_tokens=fuse_max_new_tokens,
            bundle_template_en=bundle_template_en,
            bundle_template_zh=bundle_template_zh,
            edge_labels_en=edge_labels_en,
            edge_labels_zh=edge_labels_zh,
        )

    packages = list_whole_tree_fusion_packages(db)
    if packages:
        rounds = 1
        total_packages = len(packages)
        workers = min(max(1, int(package_concurrency)), len(packages))
        if workers <= 1:
            fused_out = [_work(m) for m in packages]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                fused_out = list(pool.map(_work, packages))

        delete_ids: List[str] = []
        to_add: List[tuple[str, str, str, Dict[str, Any]]] = []
        for mids, row in fused_out:
            delete_ids.extend(mids)
            to_add.append(row)

        skip_delete = set(delete_ids)
        texts = [t[0] for t in to_add]
        emb = embed_fn(texts)
        if emb.size == 0:
            return {
                "skipped": False,
                "packages": total_packages,
                "added": total_added,
                "deleted": total_deleted,
                "rounds": rounds,
                "fusion_strategy": LME_FUSION_STRATEGY_WHOLE_TREE,
                "error": "embedding_empty_mid_fuse",
            }

        mid_to_fused: Dict[str, str] = {}
        for i, (mids, (text, src_idx, time_s, meta)) in enumerate(fused_out):
            row_vec = emb[i] if emb.ndim == 2 else emb
            fid = db.add(
                text=text,
                source_index=src_idx,
                time=time_s or "unknown_time",
                metadata=meta,
                embedding=np.asarray(row_vec, dtype=np.float32),
            )
            for mid in mids:
                mid_to_fused[mid] = fid

        _rewire_parent_primary_after_fusion(db, mid_to_fused, skip_delete)

        for mid in delete_ids:
            db.delete(mid)
        total_deleted += len(delete_ids)
        total_added += len(to_add)

    # 仍为 primary、未打融合标记：
    # - 无 evidence 子树；或
    # - 子 evidence 已全部为融合行（单轮后仅剩「根 + 若干融合子块」）
    # 仅对该根做单条 ``lme_fused_bundle`` 标记（无 LLM），并先重连子结点 ``parent_primary`` 再删旧行。
    for mem in list(db.list_all_memories(sort_by_time=False)):
        if not _is_primary_meta(mem.metadata):
            continue
        if is_cascade_root(mem.metadata):
            continue
        if (mem.metadata or {}).get("lme_fused_bundle"):
            continue
        evs = db.collect_evidence_descendants(mem.memory_id)
        if evs and not all((e.metadata or {}).get("lme_fused_bundle") for e in evs):
            continue
        mids, row = _fuse_one_package(
            [mem],
            llm_client=llm_client,
            language=language,
            fuse_max_new_tokens=fuse_max_new_tokens,
            bundle_template_en=bundle_template_en,
            bundle_template_zh=bundle_template_zh,
            edge_labels_en=edge_labels_en,
            edge_labels_zh=edge_labels_zh,
        )
        text, src_idx, time_s, meta = row
        emb_one = embed_fn([text])
        if emb_one.size == 0:
            continue
        vec = emb_one[0] if emb_one.ndim == 2 else emb_one
        fid = db.add(
            text=text,
            source_index=src_idx,
            time=time_s or "unknown_time",
            metadata=meta,
            embedding=np.asarray(vec, dtype=np.float32),
        )
        mid_to_fused = {mid: fid for mid in mids}
        _rewire_parent_primary_after_fusion(db, mid_to_fused, set(mids))
        for mid in mids:
            db.delete(mid)
        total_deleted += len(mids)
        total_added += 1

    return {
        "skipped": False,
        "packages": total_packages,
        "added": total_added,
        "deleted": total_deleted,
        "rounds": rounds,
        "fusion_strategy": LME_FUSION_STRATEGY_WHOLE_TREE,
    }
