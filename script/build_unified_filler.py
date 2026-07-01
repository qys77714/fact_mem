#!/usr/bin/env python3
"""
从 unified confusion 数据集中抽取非 evidence session 的 filler 记忆。

用法:
  uv run --no-sync python script/build_unified_filler.py

输入:
  - data/preprocessed/longmemeval_s_unified_confusion.json  (470 题，含 answer_session_ids)
  - MemDB/candidates/lme_s_gemma4-26B_0615_unified/        (candidate JSON 文件)

输出:
  - data/preprocessed/lme_s_non_evidence_filler.json
    {qid: {"filler_chunks": [{session_index, session_date, candidate_memories}, ...],
           "filler_memory_count": N}, ...}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_CAND_DIR = _REPO / "MemDB" / "candidates" / "lme_s_gemma4-26B_0615_unified"
_UNIFIED_PATH = _REPO / "data" / "preprocessed" / "longmemeval_s_unified_confusion.json"
_OUT_PATH = _REPO / "data" / "preprocessed" / "lme_s_non_evidence_filler.json"


def _candidate_path(qid: str) -> Path:
    """question_id → candidate JSON 文件路径。

    gpt4_* 前缀的 qid 映射到 gp_4_* 文件名。
    """
    if qid.startswith("gpt4_"):
        return _CAND_DIR / f"gp_4_{qid[5:]}.json"
    return _CAND_DIR / f"{qid}.json"


def main() -> None:
    with open(_UNIFIED_PATH) as f:
        unified_data = json.load(f)

    filler: dict[str, dict] = {}
    missing = 0
    total_filler_memories = 0

    for item in unified_data:
        qid = item["question_id"]
        answer_sids = set(item.get("answer_session_ids", []))
        hs_ids = item.get("haystack_session_ids", [])

        # 找出 evidence session_index
        evidence_si: set[int] = set()
        for i, sid in enumerate(hs_ids):
            if sid in answer_sids:
                evidence_si.add(i + 1)  # session_index = i+1

        cand_path = _candidate_path(qid)
        if not cand_path.exists():
            print(f"WARNING: 缺少 candidate 文件: {cand_path}", file=sys.stderr)
            missing += 1
            filler[qid] = {"filler_chunks": [], "filler_memory_count": 0}
            continue

        with open(cand_path) as f:
            cand = json.load(f)

        # 过滤: 只保留非 evidence session 的 chunks
        filler_chunks: list[dict] = []
        filler_count = 0
        for chunk in cand.get("chunks", []):
            si = chunk.get("session_index")
            if si in evidence_si:
                continue
            mems = chunk.get("candidate_memories", [])
            if not mems:
                continue
            filler_chunks.append({
                "session_index": si,
                "session_date": chunk.get("session_date", ""),
                "candidate_memories": mems,
            })
            filler_count += len(mems)

        filler[qid] = {
            "filler_chunks": filler_chunks,
            "filler_memory_count": filler_count,
        }
        total_filler_memories += filler_count

    with open(_OUT_PATH, "w") as f:
        json.dump(filler, f, ensure_ascii=False, indent=2)

    print(f"完成: {len(unified_data)} 题")
    print(f"  缺失 candidate: {missing}")
    print(f"  非 evidence filler 总记忆数: {total_filler_memories}")
    print(f"  平均每题 filler: {total_filler_memories / max(len(unified_data) - missing, 1):.1f}")
    print(f"  输出: {_OUT_PATH}")


if __name__ == "__main__":
    main()
