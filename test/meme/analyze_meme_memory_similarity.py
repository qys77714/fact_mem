#!/usr/bin/env python3
"""
Analyze embedding cosine similarity for MEME candidate memories.

IMPORTANT: similarity is computed ONLY within the same episode (one memory bank).
No cross-episode pairs are ever formed.

Per episode:
  - gold-gold: pairwise among evidence_gold_facts memories
  - gold-filler: each gold memory vs each filler-extracted memory

Uses the same embedding API as the MEME pipeline (qwen3-embedding via vLLM).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from openai import OpenAI
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.embed_utils import embed_texts  # noqa: E402
from utils.env import load_env  # noqa: E402


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix between rows of a (n,d) and b (m,d)."""
    if a.size == 0 or b.size == 0:
        return np.empty((a.shape[0], b.shape[0]), dtype=np.float32)
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    a_norm = np.linalg.norm(a64, axis=1, keepdims=True) + 1e-12
    b_norm = np.linalg.norm(b64, axis=1, keepdims=True) + 1e-12
    return ((a64 / a_norm) @ (b64 / b_norm).T).astype(np.float32)


def _macro_avg(per_episode: List[Dict[str, Any]], key: str, stat: str = "mean") -> float | None:
    """Average a per-episode statistic across episodes (equal weight per episode)."""
    vals = [
        float(ep[key][stat])
        for ep in per_episode
        if key in ep and isinstance(ep[key], dict) and stat in ep[key]
    ]
    return float(np.mean(vals)) if vals else None


def _summary_stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def _load_episode_memories(path: Path) -> Tuple[str, List[str], List[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    episode_id = str(data.get("history_name") or path.stem)
    gold: List[str] = []
    filler: List[str] = []
    for chunk in data.get("chunks", []):
        mems = [str(m).strip() for m in chunk.get("candidate_memories", []) if str(m).strip()]
        if not mems:
            continue
        if chunk.get("source") == "evidence_gold_facts":
            gold.extend(mems)
        else:
            filler.extend(mems)
    return episode_id, gold, filler


def _embed_batched(
    client: OpenAI,
    texts: List[str],
    model: str,
    batch_size: int,
) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    chunks: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        chunks.append(embed_texts(client, batch, model))
    return np.vstack(chunks).astype(np.float32)


def _top_pairs(
    sim: np.ndarray,
    texts_a: List[str],
    texts_b: List[str],
    *,
    top_k: int,
    skip_diag: bool = False,
) -> List[Dict[str, Any]]:
    pairs: List[Tuple[float, int, int]] = []
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            if skip_diag and i == j:
                continue
            pairs.append((float(sim[i, j]), i, j))
    pairs.sort(key=lambda x: x[0], reverse=True)
    out: List[Dict[str, Any]] = []
    for score, i, j in pairs[:top_k]:
        out.append({
            "similarity": score,
            "text_a": texts_a[i],
            "text_b": texts_b[j],
        })
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MEME gold vs filler memory embedding similarity")
    p.add_argument(
        "--candidates-dir",
        type=Path,
        default=_REPO_ROOT / "MemDB/candidates/meme_filler32k_gemma4-26B_0519_as3",
    )
    p.add_argument("--embedding-model", default="qwen3-embedding-0.6b")
    p.add_argument("--embedding-base-url", default=None)
    p.add_argument("--embedding-api-key", default=None)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--top-k", type=int, default=20, help="Top similar pairs to save")
    p.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "test/meme/output/meme_filler32k_gold_filler_similarity.json",
    )
    p.add_argument("--max-episodes", type=int, default=0, help="Debug limit (0=all)")
    return p.parse_args()


def main() -> None:
    load_env(_REPO_ROOT / ".env")
    args = parse_args()

    base_url = args.embedding_base_url or os.getenv("EMBEDDING_BASE_URL", "http://localhost:7110/v1/")
    api_key = args.embedding_api_key or os.getenv("EMBEDDING_API_KEY", "zjj")
    client = OpenAI(base_url=base_url, api_key=api_key)

    cand_dir = args.candidates_dir
    if not cand_dir.is_dir():
        raise SystemExit(f"candidates dir not found: {cand_dir}")

    files = sorted(p for p in cand_dir.glob("*.json") if p.name != "extract_progress.state")
    if args.max_episodes > 0:
        files = files[: args.max_episodes]

    all_gold_gold: List[float] = []
    all_gold_filler: List[float] = []
    per_episode: List[Dict[str, Any]] = []
    top_gold_gold: List[Dict[str, Any]] = []
    top_gold_filler: List[Dict[str, Any]] = []

    total_gold = total_filler = 0

    for path in tqdm(files, desc="episodes"):
        episode_id, gold, filler = _load_episode_memories(path)
        total_gold += len(gold)
        total_filler += len(filler)

        ep_result: Dict[str, Any] = {
            "episode_id": episode_id,
            "gold_count": len(gold),
            "filler_count": len(filler),
        }

        if len(gold) < 2 and not filler:
            ep_result["skipped"] = "no_pairs"
            per_episode.append(ep_result)
            continue

        gold_emb = _embed_batched(client, gold, args.embedding_model, args.batch_size)
        filler_emb = _embed_batched(client, filler, args.embedding_model, args.batch_size)

        if len(gold) >= 2:
            gg_sim = _cosine_sim_matrix(gold_emb, gold_emb)
            gg_vals = [
                float(gg_sim[i, j])
                for i in range(gg_sim.shape[0])
                for j in range(i + 1, gg_sim.shape[1])
            ]
            all_gold_gold.extend(gg_vals)
            ep_result["gold_gold"] = _summary_stats(gg_vals)
            for pair in _top_pairs(gg_sim, gold, gold, top_k=min(3, args.top_k), skip_diag=True):
                top_gold_gold.append({"episode_id": episode_id, **pair})

        if gold and filler:
            gf_sim = _cosine_sim_matrix(gold_emb, filler_emb)
            gf_vals = [float(x) for x in gf_sim.ravel()]
            all_gold_filler.extend(gf_vals)
            ep_result["gold_filler"] = _summary_stats(gf_vals)
            for pair in _top_pairs(gf_sim, gold, filler, top_k=min(3, args.top_k)):
                top_gold_filler.append({"episode_id": episode_id, **pair})

        per_episode.append(ep_result)

    top_gold_gold.sort(key=lambda x: x["similarity"], reverse=True)
    top_gold_filler.sort(key=lambda x: x["similarity"], reverse=True)

    report = {
        "similarity_scope": "within_episode_only",
        "candidates_dir": str(cand_dir.resolve()),
        "embedding_model": args.embedding_model,
        "embedding_base_url": base_url,
        "episodes": len(files),
        "memory_counts": {
            "gold": total_gold,
            "filler": total_filler,
            "total": total_gold + total_filler,
        },
        # micro: pool all within-episode pairs (episode size weighted)
        "gold_gold": _summary_stats(all_gold_gold),
        "gold_filler": _summary_stats(all_gold_filler),
        # macro: equal weight per episode
        "gold_gold_macro_avg": {
            "mean": _macro_avg(per_episode, "gold_gold", "mean"),
            "median": _macro_avg(per_episode, "gold_gold", "median"),
            "p95": _macro_avg(per_episode, "gold_gold", "p95"),
        },
        "gold_filler_macro_avg": {
            "mean": _macro_avg(per_episode, "gold_filler", "mean"),
            "median": _macro_avg(per_episode, "gold_filler", "median"),
            "p95": _macro_avg(per_episode, "gold_filler", "p95"),
        },
        "top_gold_gold_pairs": top_gold_gold[: args.top_k],
        "top_gold_filler_pairs": top_gold_filler[: args.top_k],
        "per_episode": per_episode,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "output": str(args.output),
        "similarity_scope": report["similarity_scope"],
        "episodes": report["episodes"],
        "memory_counts": report["memory_counts"],
        "gold_gold": report["gold_gold"],
        "gold_filler": report["gold_filler"],
        "gold_gold_macro_avg": report["gold_gold_macro_avg"],
        "gold_filler_macro_avg": report["gold_filler_macro_avg"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
