#!/usr/bin/env python3
"""
基于 lowered golden 答题测试结果，生成 hybrid golden 数据集：
- 答对的 partial 题 → 使用 lowered_golden
- 答错的 partial 题 → 回退为 golden_memory (original)
- full_fallback 题 → 使用 golden_memory (original)

输出: data/preprocessed/longmemeval_s_hybrid_golden.json
"""

import json
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_UNIFIED_PATH = _REPO / "data" / "preprocessed" / "longmemeval_s_unified_confusion.json"
_OUTPUT_PATH = _REPO / "data" / "preprocessed" / "longmemeval_s_hybrid_golden.json"

# 使用 unified_filler N0, gemma4-26B answer, qwen3-max judge, 1024 token 的实验结果
_PRED_PATH = _REPO / "experiment" / "lme_s_candunified_filler_N0_Qwen3-4B_gemma4-26B_tl1024_unified_filler_N0_qwen3-4b" / "pred_add_all.jsonl"


def main():
    # Load unified confusion data
    with open(_UNIFIED_PATH) as f:
        conf = json.load(f)

    # Load pred results to determine which lowered golden works
    with open(_PRED_PATH) as f:
        preds = [json.loads(line) for line in f if line.strip()]

    # Build correctness lookup by history_name
    correct_map = {}
    for p in preds:
        hn = p.get("history_name", "")
        correct_map[hn] = p.get("is_correct", False)

    conf_map = {q["question_id"]: q for q in conf}

    output = []
    stats = {"keep_lowered": 0, "revert_original": 0, "full_fallback": 0}

    for item in conf:
        qid = item["question_id"]
        status = item.get("lowering_status", "full_fallback")
        is_correct = correct_map.get(qid, False)

        new_item = {k: v for k, v in item.items()}

        if status == "partial" and is_correct:
            # 用 lowered golden 能答对 → 保留 lowered
            new_item["golden_memory"] = [
                {"text": lg["text"], "sim_q": lg.get("sim_q", 0),
                 "date": lg.get("date", ""), "source": "lowered_golden"}
                for lg in item.get("lowered_golden", [])
            ]
            new_item["golden_source"] = "lowered"
            stats["keep_lowered"] += 1
        else:
            # 回退为 original golden
            new_item["golden_memory"] = [
                {"text": gm["text"], "sim_q": gm.get("sim_q", 0),
                 "date": gm.get("date", ""), "source": "golden_memory"}
                for gm in item.get("golden_memory", [])
            ]
            new_item["golden_source"] = "original"
            if status == "partial":
                stats["revert_original"] += 1
            else:
                stats["full_fallback"] += 1

        # 清理 intermediate fields
        new_item.pop("lowered_golden", None)
        new_item.pop("lowering_status", None)
        new_item.pop("lowering_details", None)
        new_item.pop("anchor_sim", None)

        output.append(new_item)

    # 统计
    print(f"Total questions: {len(output)}")
    print(f"  Keep lowered_golden: {stats['keep_lowered']}")
    print(f"  Revert to original (partial wrong): {stats['revert_original']}")
    print(f"  Full_fallback → original: {stats['full_fallback']}")
    print(f"  Total original: {stats['revert_original'] + stats['full_fallback']}")

    # 新 golden 的 sim_q 分布
    import numpy as np
    all_sims = []
    for item in output:
        for g in item["golden_memory"]:
            all_sims.append(g["sim_q"])
    print(f"\nNew golden memory sim_q stats:")
    print(f"  Mean: {np.mean(all_sims):.4f}")
    print(f"  Median: {np.median(all_sims):.4f}")
    print(f"  Min: {np.min(all_sims):.4f}")
    print(f"  Max: {np.max(all_sims):.4f}")

    # Compare with original
    orig_sims = []
    lowered_sims = []
    for item in conf:
        for g in item["golden_memory"]:
            orig_sims.append(g.get("sim_q", 0))
        for g in item.get("lowered_golden", []):
            lowered_sims.append(g.get("sim_q", 0))

    print(f"\nOriginal golden sim_q: mean={np.mean(orig_sims):.4f}, median={np.median(orig_sims):.4f}")
    print(f"Pure lowered sim_q: mean={np.mean(lowered_sims):.4f}, median={np.median(lowered_sims):.4f}")
    print(f"Hybrid golden sim_q: mean={np.mean(all_sims):.4f}, median={np.median(all_sims):.4f}")

    # Distractor sim_q vs golden sim_q gap
    lower_golden = [g["sim_q"] for item in output for g in item["golden_memory"] if item["golden_source"] == "lowered"]
    orig_golden = [g["sim_q"] for item in output for g in item["golden_memory"] if item["golden_source"] == "original"]
    print(f"\nHybrid lowered subset sim_q: mean={np.mean(lower_golden):.4f} ({len(lower_golden)} items)")
    print(f"Hybrid original subset sim_q: mean={np.mean(orig_golden):.4f} ({len(orig_golden)} items)")

    # Save
    with open(_OUTPUT_PATH, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
