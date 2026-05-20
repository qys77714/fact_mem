#!/usr/bin/env python3
"""
灌库完成后：在**副本目录**中对各 episode 做关系包 LLM 融合（源目录不修改）。

- ``--database-root``：灌库产物（未融合），与 ingest 的 ``--database-root`` 一致；**不会被本脚本改写**。
- ``--fused-output-root``：融合输出根目录（默认 ``<父目录>/<源目录名>_fused``）。每个 episode 先从源目录拷贝子目录到此处，再在本目录内执行删旧写新。
- 断点续传：拷贝灌库产物后会**删除**目标 episode 下的 ``.memory_ready.json``（去掉灌库侧的完成标记）；融合**成功结束后**再写入该文件。若目标 episode 下已有 ``.memory_ready.json``，视为本脚本已完成该集融合并跳过。``--force-refuse`` 时忽略该文件并重新拷贝再融合。

可用 ``--episode-concurrency`` 并行处理多个 episode；``--package-concurrency`` 在单 episode 内并行融合关系包（向量库删写仍串行）。
融合为**单轮整树**：每个根 primary 与其整条 evidence 子树同一 prompt 融为一条，多根则多包并行（``--package-concurrency``）；**不会**把本轮已融合的多条再合并。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List

from tqdm import tqdm

from memory.fusion.lme_bundle_fusion import fuse_local_faiss_database
from memory.storage.local_faiss import LocalFaissDatabase
from utils.embed_utils import embed_texts
from utils.env import load_env
from utils.llm_api import load_api_chat_completion

LME_FUSION_MEMORY_READY_VERSION = 3
LME_FUSION_MARKER_KIND = "lme_bundle_fusion"


def _episode_fusion_marker_path(episode_dir: Path) -> Path:
    return episode_dir / ".memory_ready.json"


def _write_fusion_marker_atomic(episode_dir: Path, payload: dict[str, Any]) -> None:
    path = _episode_fusion_marker_path(episode_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _remove_ready_marker_after_copy(episode_dir: Path) -> None:
    """灌库侧 ``.memory_ready.json`` 会随 copytree 带入；删掉后才表示「本集融合尚未完成」。"""
    p = _episode_fusion_marker_path(episode_dir)
    try:
        p.unlink(missing_ok=True)
    except OSError:
        pass


def _is_episode_fusion_done(episode_dir: Path) -> bool:
    """True iff 本脚本已在该 episode 融合成功后写入 ``.memory_ready.json``（仅看文件是否存在）。"""
    return _episode_fusion_marker_path(episode_dir).is_file()


def _list_history_names(database_root: Path) -> List[str]:
    if not database_root.is_dir():
        return []
    out: List[str] = []
    for p in sorted(database_root.iterdir()):
        if p.is_dir() and (p / "ids.json").exists():
            out.append(p.name)
    return out


def main() -> int:
    load_env()
    parser = argparse.ArgumentParser(description="LME 向量库：关系包融合（灌库后运行，写入副本目录）")
    parser.add_argument(
        "--database-root",
        type=Path,
        required=True,
        help="未融合的向量库根目录（与 ingest 的 --database-root 一致；只读拷贝，不修改）",
    )
    parser.add_argument(
        "--fused-output-root",
        type=Path,
        default=None,
        help="融合结果根目录。默认：<database-root 的父目录>/<database-root 目录名>_fused",
    )
    parser.add_argument(
        "--force-refuse",
        action="store_true",
        help="忽略 fused episode 下已存在的融合完成标记，从源重新拷贝并再融合（覆盖输出侧该 episode）",
    )
    parser.add_argument(
        "--fusion-model",
        default=os.getenv("FUSION_MODEL", ""),
        help="融合用 LLM（OpenAI 兼容）；可与 --manager-model 二选一",
    )
    parser.add_argument(
        "--manager-model",
        default="",
        help="若未传 --fusion-model，则使用该模型名",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        help="融合后重新 embedding 的模型名",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="融合提示语：zh 或 en",
    )
    parser.add_argument(
        "--fusion-bundle-template-en",
        default="",
        metavar="NAME.jinja",
        help="覆盖英文融合 user 模板文件名（默认 lme_fuse_memory_bundle_en_v3.jinja）；置于 src/prompts/templates/",
    )
    parser.add_argument(
        "--fusion-bundle-template-zh",
        default="",
        metavar="NAME.jinja",
        help="覆盖中文融合 user 模板文件名（默认 lme_fuse_memory_bundle_zh_v3.jinja）",
    )
    parser.add_argument(
        "--fusion-edge-labels-template-en",
        default="",
        metavar="NAME.jinja",
        help="英文：包内行前缀边标签模板（默认 lme_fuse_memory_bundle_edge_labels_en_v2.jinja）",
    )
    parser.add_argument(
        "--fusion-edge-labels-template-zh",
        default="",
        metavar="NAME.jinja",
        help="中文：包内行前缀边标签模板（默认 lme_fuse_memory_bundle_edge_labels_zh_v2.jinja）",
    )
    parser.add_argument(
        "--fuse-max-new-tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--episode-concurrency",
        type=int,
        default=1,
        help="跨 episode 并行融合的线程数（每个 episode 独立子目录，默认 1 顺序）",
    )
    parser.add_argument(
        "--package-concurrency",
        type=int,
        default=1,
        help="单 episode 内并行融合关系包的线程数（仅并行 LLM，删写库仍串行；默认 1）",
    )
    parser.add_argument(
        "--history",
        default=None,
        help="只处理该 history_name（子目录名）；默认处理源根下全部 episode",
    )
    args = parser.parse_args()

    model = (args.fusion_model or args.manager_model or "").strip()
    if not model:
        print("ERROR: set --fusion-model or FUSION_MODEL or --manager-model", file=sys.stderr)
        return 1

    api_key = os.getenv("EMBEDDING_API_KEY")
    if not api_key:
        print("ERROR: EMBEDDING_API_KEY is not set.", file=sys.stderr)
        return 1

    from openai import OpenAI

    embed_base = os.getenv("EMBEDDING_BASE_URL", "http://localhost:7110/v1")
    embed_client = OpenAI(api_key=api_key, base_url=embed_base)
    llm_client = load_api_chat_completion(model, async_=False)

    source_root = args.database_root.resolve()
    fused_root = (args.fused_output_root or (source_root.parent / f"{source_root.name}_fused")).resolve()
    if source_root == fused_root:
        print("ERROR: --database-root and --fused-output-root must differ (source vs fused copy).", file=sys.stderr)
        return 1

    def _embed_batch(texts: List[str]):
        return embed_texts(embed_client, texts, args.embedding_model)

    names = [args.history] if args.history else _list_history_names(source_root)
    names = [n for n in names if n]
    if not names:
        print(f"No episode dirs under {source_root}", file=sys.stderr)
        return 1

    fused_root.mkdir(parents=True, exist_ok=True)
    print(f"Source (unfused): {source_root}", flush=True)
    print(f"Fused output root: {fused_root}", flush=True)

    def _fuse_episode(hn: str) -> tuple[str, dict[str, Any]]:
        src_ep = source_root / hn
        dst_ep = fused_root / hn
        if not src_ep.is_dir():
            return hn, {"skipped": True, "reason": "missing_source"}

        if not args.force_refuse and dst_ep.is_dir() and _is_episode_fusion_done(dst_ep):
            return hn, {"skipped": True, "reason": "memory_ready"}

        if dst_ep.exists():
            shutil.rmtree(dst_ep)
        shutil.copytree(src_ep, dst_ep)
        _remove_ready_marker_after_copy(dst_ep)

        db = LocalFaissDatabase(namespace=hn, database_root=str(fused_root))
        tpl_en = (args.fusion_bundle_template_en or "").strip() or None
        tpl_zh = (args.fusion_bundle_template_zh or "").strip() or None
        el_en = (args.fusion_edge_labels_template_en or "").strip() or None
        el_zh = (args.fusion_edge_labels_template_zh or "").strip() or None
        st = fuse_local_faiss_database(
            db,
            _embed_batch,
            llm_client,
            language=args.language,
            fuse_max_new_tokens=args.fuse_max_new_tokens,
            package_concurrency=args.package_concurrency,
            bundle_template_en=tpl_en,
            bundle_template_zh=tpl_zh,
            edge_labels_en=el_en,
            edge_labels_zh=el_zh,
        )
        st["source_episode"] = str(src_ep)
        st["fused_episode"] = str(dst_ep)
        stats_payload = {k: v for k, v in st.items() if k not in ("source_episode", "fused_episode")}
        stats_payload.setdefault("fusion_strategy", "whole_tree_single_wave")
        _write_fusion_marker_atomic(
            dst_ep,
            {
                "version": LME_FUSION_MEMORY_READY_VERSION,
                "kind": LME_FUSION_MARKER_KIND,
                "history_name": hn,
                "fusion_model": model,
                "embedding_model": args.embedding_model,
                "language": args.language,
                "fuse_max_new_tokens": args.fuse_max_new_tokens,
                "stats": stats_payload,
            },
        )
        return hn, st

    ep_workers = min(max(1, int(args.episode_concurrency)), len(names))
    total_stats: dict[str, Any] = {}
    if ep_workers <= 1:
        for hn in tqdm(names, desc="fuse", unit="ep"):
            h, st = _fuse_episode(hn)
            total_stats[h] = st
    else:
        with ThreadPoolExecutor(max_workers=ep_workers) as pool:
            results = list(
                tqdm(
                    pool.map(_fuse_episode, names),
                    total=len(names),
                    desc="fuse",
                    unit="ep",
                )
            )
        total_stats = {h: st for h, st in results}

    skipped = sum(1 for s in total_stats.values() if s.get("skipped"))
    fused = sum(1 for s in total_stats.values() if not s.get("skipped"))
    print(
        f"Done: {len(names)} episode(s), {skipped} skipped, {fused} fused. "
        f"source={source_root} fused_root={fused_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
