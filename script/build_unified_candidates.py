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
_HYBRID_GOLDEN_PATH = _REPO / "data" / "preprocessed" / "longmemeval_s_hybrid_golden.json"
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
    """Hybrid golden 数据集已包含最终 golden_memory，直接返回。"""
    golden = item.get("golden_memory", [])
    return [
        {"text": gm["text"], "date": gm.get("date", ""), "source": item.get("golden_source", "hybrid")}
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


def _build_chunks(goldens, distractors, filler_chunks, selected_filler_indices=None, max_filler: int = 0):
    """保留 filler 原始多 fact chunk 结构，golden/distractor 按日期合并进 filler chunk。

    - Distractor: 精确匹配 session_date → 合并到该 chunk
    - Golden: 精确匹配 session_date → 合并到该 chunk
    - 只保留 selected_filler_indices 指定的 filler chunks（其余丢弃）
    - 最终按 session_date 排序，重建 chunk_index
    """
    import copy

    # 过滤：按索引保留选中的 filler chunks
    selected_indices = set(selected_filler_indices or [])
    if selected_indices:
        filler_chunks = [fc for i, fc in enumerate(filler_chunks) if i in selected_indices]

    # 深拷贝 filler chunks，避免修改原数据
    working_chunks = []
    for fc in filler_chunks:
        c = {
            "session_date": fc.get("session_date", ""),
            "candidate_memories": list(fc.get("candidate_memories", [])),
            "turn_start": fc.get("turn_start", 0),
            "turn_end": fc.get("turn_end", 0),
            "turn_overlap": fc.get("turn_overlap", 0),
        }
        # 保留 expected_wrong_answer
        if "expected_wrong_answer" in fc:
            c["expected_wrong_answer"] = fc["expected_wrong_answer"]
        working_chunks.append(c)

    # 建立索引：精确日期 → chunk 索引
    by_exact_date: dict[str, list[int]] = {}
    for idx, c in enumerate(working_chunks):
        date = c["session_date"]
        by_exact_date.setdefault(date, []).append(idx)

    matched_chunk_indices: set[int] = set()

    # 1. Distractor: 精确匹配 session_date → 合并到匹配 chunk
    for d in distractors:
        d_date = d.get("date", "")
        matches = by_exact_date.get(d_date, [])
        if matches:
            idx = matches[0]  # 取第一个匹配的 chunk
            working_chunks[idx]["candidate_memories"].append(d["text"])
            matched_chunk_indices.add(idx)
        else:
            # 无匹配 filler chunk → 创建新 chunk
            new_idx = len(working_chunks)
            working_chunks.append({
                "session_date": d_date,
                "candidate_memories": [d["text"]],
                "turn_start": 0, "turn_end": 0, "turn_overlap": 0,
            })
            by_exact_date.setdefault(d_date, []).append(new_idx)
            day = d_date[:10]
            by_day.setdefault(day, []).append(new_idx)
            matched_chunk_indices.add(new_idx)

    # 2. Golden: 精确匹配 session_date → 合并到匹配 chunk
    for g in goldens:
        g_date = g.get("date", "")
        matches = by_exact_date.get(g_date, [])
        if matches:
            idx = matches[0]
            working_chunks[idx]["candidate_memories"].append(g["text"])
            matched_chunk_indices.add(idx)
        else:
            new_idx = len(working_chunks)
            working_chunks.append({
                "session_date": g_date,
                "candidate_memories": [g["text"]],
                "turn_start": 0, "turn_end": 0, "turn_overlap": 0,
            })
            by_exact_date.setdefault(g_date, []).append(new_idx)
            matched_chunk_indices.add(new_idx)

    # 3. 保留选中的 30 个 filler chunks（通过 golden/distractor 日期匹配）+ 其余丢弃

    # 3. 所有选中的 filler chunk 都保留（已在过滤阶段筛选），按日期排序
    # 分离已匹配和未匹配（均已被 golden/distractor 日期选中，未匹配的只是没合并到 fact）
    matched = []
    unmatched = []
    for idx, c in enumerate(working_chunks):
        if idx in matched_chunk_indices:
            matched.append(c)
        else:
            unmatched.append(c)

    matched.sort(key=lambda c: _parse_date(c["session_date"]))
    unmatched.sort(key=lambda c: _parse_date(c["session_date"]))

    # 4. 合并、排序、重建 chunk_index
    all_chunks = matched + unmatched
    all_chunks.sort(key=lambda c: _parse_date(c["session_date"]))

    chunks = []
    for i, c in enumerate(all_chunks):
        chunk: dict[str, object] = {
            "chunk_index": i,
            "session_index": i + 1,
            "turn_start": c.get("turn_start", 0),
            "turn_end": c.get("turn_end", 0),
            "turn_overlap": c.get("turn_overlap", 0),
            "session_date": c["session_date"],
            "candidate_memories": c["candidate_memories"],
            "parse_error": None,
        }
        if "expected_wrong_answer" in c:
            chunk["expected_wrong_answer"] = c["expected_wrong_answer"]
        chunks.append(chunk)

    return chunks, len(goldens), len(distractors), 0


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

        selected_indices = item.get("_selected_filler_indices", None)
        chunks, n_g, n_d, n_f = _build_chunks(
            goldens, distractors, filler_chunks,
            selected_filler_indices=selected_indices,
            max_filler=max_filler,
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
            "prompt_template": "0_mem_extract_aspect_unified_en.jinja",
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

    with open(_HYBRID_GOLDEN_PATH) as f:
        conf_data = json.load(f)
    with open(_FILLER_PATH) as f:
        filler_data = json.load(f)

    print(f"Hybrid Golden 数据: {len(conf_data)} 题 ({_HYBRID_GOLDEN_PATH})")
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
