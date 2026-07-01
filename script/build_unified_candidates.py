#!/usr/bin/env python3
"""
基于 unified confusion 数据集 + non-evidence filler，构建 N∈{0,2,4,6,8} 条 distractor 的实验候选目录。

用法:
  uv run --no-sync python script/build_unified_candidates.py [--distractors 0,2,4,6,8]

输入:
  - data/preprocessed/longmemeval_s_unified_confusion.json
  - data/preprocessed/lme_s_non_evidence_filler.json

输出 (5 个目录):
  - MemDB/candidates/lme_s_gemma4-26B_unified_filler_N0/
  - MemDB/candidates/lme_s_gemma4-26B_unified_filler_N2/
  - MemDB/candidates/lme_s_gemma4-26B_unified_filler_N4/
  - MemDB/candidates/lme_s_gemma4-26B_unified_filler_N6/
  - MemDB/candidates/lme_s_gemma4-26B_unified_filler_N8/

每个目录含 470 个 {qid}.json + extract_progress.state
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_UNIFIED_PATH = _REPO / "data" / "preprocessed" / "longmemeval_s_unified_confusion.json"
_FILLER_PATH = _REPO / "data" / "preprocessed" / "lme_s_non_evidence_filler.json"
_CAND_BASE = _REPO / "MemDB" / "candidates"

_CAND_PREFIX = "lme_s_gemma4-26B"

EXTRACT_PROGRESS_VERSION = 5
EXTRACT_PROGRESS_KIND = "lme_candidate_extract_progress"


def _parse_date(date_str: str) -> datetime:
    """统一解析多种日期格式。"""
    s = (date_str or "").strip()
    if not s:
        return datetime(2000, 1, 1)
    m = re.match(r"(\d{4})/(\d{2})/(\d{2})\s*\([^)]+\)\s*(\d{2}):(\d{2})", s)
    if m:
        return datetime(int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5]))
    m = re.match(r"(\d{4})/(\d{2})/(\d{2})", s)
    if m:
        return datetime(int(m[1]), int(m[2]), int(m[3]))
    return datetime(2000, 1, 1)


def _select_golden(item: dict) -> list[dict]:
    """选择 golden memory: 优先 lowered_golden，回退到 golden_memory。"""
    status = item.get("lowering_status", "full_fallback")
    if status != "full_fallback":
        lowered = item.get("lowered_golden", [])
        if lowered:
            return [
                {"text": lg["text"], "date": lg.get("date", ""), "source": "lowered_golden",
                 "source_idx": lg.get("source_idx", 0)}
                for lg in lowered
            ]
    golden = item.get("golden_memory", [])
    return [
        {"text": gm["text"], "date": gm.get("date", ""), "source": "golden_memory"}
        for gm in golden
    ]


def _select_distractors(item: dict, n: int) -> list[dict]:
    """取前 N 条 distractor（已按 sim_q 降序排列）。保留 expected_wrong_answer 字段。"""
    dists = item.get("distractors", [])
    selected = dists[:n]
    result = []
    for d in selected:
        entry = {
            "text": d["text"],
            "date": d.get("date", ""),
            "source": d.get("source", "distractor"),
            "source_idx": d.get("source_idx", 0),
        }
        # Type II distractor 带有 expected_wrong_answer
        ewa = d.get("expected_wrong_answer")
        if ewa is not None:
            entry["expected_wrong_answer"] = ewa
        result.append(entry)
    return result


def _build_chunks(goldens, distractors, filler_chunks, max_filler: int = 0):
    """合并 golden + distractor + filler，按日期排序，重建 chunk_index 和 session_index。"""
    timed_items: list[tuple[datetime, str, dict]] = []

    for g in goldens:
        dt = _parse_date(g["date"])
        timed_items.append((dt, f"golden_{g.get('source_idx', hash(g['text']))}", {
            "type": "golden", "text": g["text"], "date": g["date"]
        }))

    for d in distractors:
        dt = _parse_date(d["date"])
        entry: dict[str, object] = {
            "type": "distractor", "text": d["text"], "date": d["date"]
        }
        ewa = d.get("expected_wrong_answer")
        if ewa is not None:
            entry["expected_wrong_answer"] = ewa
        timed_items.append((dt, f"dist_{d.get('source', '')}_{d.get('source_idx', 0)}", entry))

    # Filler: 扁平化后按时序均匀采样
    filler_flat: list[tuple[datetime, str, str]] = []
    for fc in filler_chunks:
        dt = _parse_date(fc["session_date"])
        date_str = fc.get("session_date", "")
        for mem_text in fc.get("candidate_memories", []):
            filler_flat.append((dt, mem_text, date_str))
    filler_flat.sort(key=lambda x: x[0])

    if max_filler == 0:
        filler_flat = []
    elif max_filler > 0 and len(filler_flat) > max_filler:
        step = (len(filler_flat) - 1) / (max_filler - 1) if max_filler > 1 else 0
        indices = [round(i * step) for i in range(max_filler)]
        filler_flat = [filler_flat[idx] for idx in indices]

    for dt, mem_text, date_str in filler_flat:
        timed_items.append((dt, f"filler_{hash(mem_text)}", {
            "type": "filler", "text": mem_text, "date": date_str
        }))

    # 按日期排序；同日期 golden 在最后
    type_order = {"filler": 0, "distractor": 1, "golden": 2}
    timed_items.sort(key=lambda x: (x[0], type_order.get(x[2]["type"], 1)))

    chunks = []
    for chunk_idx, (dt, uid, entry) in enumerate(timed_items):
        chunk: dict[str, object] = {
            "chunk_index": chunk_idx,
            "session_index": chunk_idx + 1,
            "turn_start": 0,
            "turn_end": 0,
            "turn_overlap": 0,
            "session_date": entry["date"],
            "candidate_memories": [entry["text"]],
            "parse_error": None,
        }
        ewa = entry.get("expected_wrong_answer")
        if ewa is not None:
            chunk["expected_wrong_answer"] = ewa
        chunks.append(chunk)

    return chunks, len(goldens), len(distractors), len(filler_flat)


def _candidate_filename(qid: str) -> str:
    """question_id → 候选 JSON 文件名。gpt4_* → gp_4_*"""
    if qid.startswith("gpt4_"):
        return f"gp_4_{qid[5:]}.json"
    return f"{qid}.json"


def build_one_config(
    conf_data: list[dict],
    filler_data: dict[str, dict],
    n_distractors: int,
    output_dir: Path,
    max_filler: int = 0,
) -> None:
    """为指定 distractor 数量构建候选目录。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    total_golden = total_dist = total_filler = 0

    for item in conf_data:
        qid = item["question_id"]

        goldens = _select_golden(item)
        distractors = _select_distractors(item, n_distractors)
        filler_info = filler_data.get(qid, {})
        filler_chunks = filler_info.get("filler_chunks", [])

        chunks, n_g, n_d, n_f = _build_chunks(
            goldens, distractors, filler_chunks, max_filler=max_filler
        )
        total_golden += n_g
        total_dist += n_d
        total_filler += n_f

        out = {
            "history_name": qid,
            "model": "gemma4-26B",
            "memory_granularity": 4,
            "turn_overlap": 0,
            "dialogue_format": "user_assistant",
            "chunks": chunks,
        }

        out_path = output_dir / _candidate_filename(qid)
        with open(out_path, "w") as f:
            json.dump(out, f, ensure_ascii=False)

        completed.append(qid)

    # 写 extract_progress.state
    progress = {
        "version": EXTRACT_PROGRESS_VERSION,
        "kind": EXTRACT_PROGRESS_KIND,
        "config": {
            "model": "gemma4-26B",
            "memory_granularity": "4",
            "turn_overlap": 0,
            "dialogue_format": "user_assistant",
            "prompt_template": "0_mem_extract_v2.jinja",
            "mem_extract_extra_templates": ["0_mem_extract_aspect_unified_en.jinja"],
            "mem_extract_aspects_only": True,
            "use_json_schema": True,
            "max_new_tokens": 2048,
            "note": f"unified confusion experiment, N={n_distractors} distractors",
        },
        "completed": completed,
    }
    with open(output_dir / "extract_progress.state", "w") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

    # 统计 Type I / Type II 分布
    type_i = sum(1 for item in conf_data if item.get("confusion_type") == "type_i")
    type_ii = sum(1 for item in conf_data if item.get("confusion_type") == "type_ii")
    print(f"  N={n_distractors}: {len(completed)} 题 (Type I={type_i}, Type II={type_ii}), "
          f"golden={total_golden}, dist={total_dist}, filler={total_filler}")


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 unified confusion 实验候选目录")
    parser.add_argument("--distractors", default="0,2,4,6,8",
                        help="逗号分隔的 distractor 数量 (default: 0,2,4,6,8)")
    parser.add_argument("--max-filler", type=int, default=50,
                        help="每题最多 filler 记忆数 (default: 50)")
    parser.add_argument("--suffix-prefix", default="unified_filler",
                        help="候选目录 suffix 前缀 (default: unified_filler)")
    args = parser.parse_args()

    ns = [int(x.strip()) for x in args.distractors.split(",")]

    with open(_UNIFIED_PATH) as f:
        conf_data = json.load(f)
    with open(_FILLER_PATH) as f:
        filler_data = json.load(f)

    print(f"Unified 数据: {len(conf_data)} 题 ({_UNIFIED_PATH})")
    print(f"Filler: {len(filler_data)} 题")
    print(f"Max filler/episode: {args.max_filler}")
    print(f"N 档: {ns}")

    output_base = _CAND_BASE
    prefix = args.suffix_prefix
    for n in ns:
        output_dir = output_base / f"{_CAND_PREFIX}_{prefix}_N{n}"
        print(f"\n构建 N={n} → {output_dir}")
        build_one_config(conf_data, filler_data, n, output_dir,
                         max_filler=args.max_filler)

    print("\n完成。")


if __name__ == "__main__":
    main()
