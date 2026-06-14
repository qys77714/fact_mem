#!/usr/bin/env python3
"""
Offline split compound evidence gold facts in candidate JSON folders.

For ``source == evidence_gold_facts`` chunks, each ``candidate_memories`` string
is split via ``split_golden_memory``:
  - ``candidate_memories`` keeps primary text only (e.g. medication name)
  - ``cas_update_rules`` holds the condition tail (parallel list, null when none)

Filler chunks are copied unchanged.

Example:
  PYTHONPATH=src python3 -m pipeline.split_candidates_cas_update \\
    --input-dir MemDB/candidates/meme_filler32k_gemma4-26B_0519_as3
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from memory.candidate_ingest.cas_update import build_evidence_gold_chunk_fields

DEFAULT_SUFFIX = "_cas_split"


def _split_gold_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    mems = chunk.get("candidate_memories") or []
    if not isinstance(mems, list):
        mems = []
    mems_s = [raw if isinstance(raw, str) else str(raw) for raw in mems]
    out = dict(chunk)

    existing_rules = chunk.get("cas_update_rules")
    if isinstance(existing_rules, list) and len(existing_rules) == len(mems_s):
        out["candidate_memories"] = mems_s
        out["cas_update_rules"] = [
            (str(r).strip() if r else None) for r in existing_rules
        ]
    else:
        triggers = chunk.get("cascade_trigger_texts")
        if isinstance(triggers, list) and len(triggers) == len(mems_s):
            originals = [
                str(t).strip() if t else m for t, m in zip(triggers, mems_s)
            ]
            out.update(build_evidence_gold_chunk_fields(originals))
        else:
            out.update(build_evidence_gold_chunk_fields(mems_s))

    for key in (
        "gold_candidate_kinds",
        "cascade_trigger_texts",
        "cas_split",
        "cas_split_count",
    ):
        out.pop(key, None)
    return out


def transform_episode_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    chunks = payload.get("chunks") or []
    if not isinstance(chunks, list):
        raise ValueError("candidate json: 'chunks' must be a list")

    new_chunks: List[Dict[str, Any]] = []
    rules_non_null_count = 0
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        if str(chunk.get("source") or "") == "evidence_gold_facts":
            nc = _split_gold_chunk(chunk)
            rules = nc.get("cas_update_rules") or []
            if isinstance(rules, list):
                rules_non_null_count += sum(1 for r in rules if r)
            new_chunks.append(nc)
        else:
            new_chunks.append(dict(chunk))

    out = dict(payload)
    out["chunks"] = new_chunks
    out["cas_split_meta"] = {
        "source_kind": "offline_cas_update_split",
        "rules_non_null_count": rules_non_null_count,
    }
    return out


def process_candidates_dir(
    input_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> Dict[str, Any]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input dir not found: {input_dir}")

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"output dir already exists: {output_dir} (pass --overwrite)"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "episodes": 0,
        "rules_non_null_count": 0,
        "output_dir": str(output_dir),
        "input_dir": str(input_dir),
    }

    json_files = sorted(input_dir.glob("*.json"))
    for path in json_files:
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            continue
        transformed = transform_episode_payload(payload)
        stats["episodes"] += 1
        stats["rules_non_null_count"] += int(
            (transformed.get("cas_split_meta") or {}).get("rules_non_null_count") or 0
        )
        out_path = output_dir / path.name
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(transformed, f, ensure_ascii=False, indent=2)

    progress_src = input_dir / "extract_progress.state"
    if progress_src.is_file():
        shutil.copy2(progress_src, output_dir / "extract_progress.state")

    manifest = {
        "transform": "split_candidates_cas_update",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        **stats,
    }
    with (output_dir / "cas_split_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return stats


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Source candidates folder (per-episode JSON)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Destination folder (default: input name + {DEFAULT_SUFFIX})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output directory contents if it exists",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    if not input_dir.is_absolute():
        input_dir = repo_root / input_dir

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = input_dir.parent / f"{input_dir.name}{DEFAULT_SUFFIX}"
    elif not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    stats = process_candidates_dir(input_dir, output_dir, overwrite=args.overwrite)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
