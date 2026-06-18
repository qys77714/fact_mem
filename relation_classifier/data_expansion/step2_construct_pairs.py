"""Step 2: 从原子记忆构造 (old, new) 配对 — 更新链 / 相似配对 / 随机 IND。"""

import json
import os
import random
import sys
import yaml
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple


def load_memories(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_update_chain_pairs(memories: List[dict]) -> List[dict]:
    """2a: 更新链配对 prev_text → text（天然非 IND）。"""
    pairs = []
    for m in memories:
        if m["updated"] and m["prev_text"]:
            pairs.append({
                "old": m["prev_text"],
                "new": m["text"],
                "pair_type": "update_chain",
                "persona_id": m["persona_id"],
                "old_pref_id": m["pref_id"] + "_prev",
                "new_pref_id": m["pref_id"],
                "source_detail": f"pref_type={m['pref_type']}",
                "pref_type_original": m.get("pref_type", ""),
            })
    return pairs


def build_similar_pairs(
    memories: List[dict],
    similarity_threshold: float = 0.85,
    max_per_persona: int = 50,
) -> List[dict]:
    """2b: 同 persona 内 embedding 相似但非更新链的配对，挖掘潜在非 IND。

    使用 classifier backbone 抽取 embedding，计算余弦相似度。
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from classifier import RelationClassifier

    # 按 persona 分组
    by_persona = defaultdict(list)
    for m in memories:
        by_persona[m["persona_id"]].append(m)

    # 建立所有更新链 pair 的 (old_text, new_text) 集合，避免重复
    update_pairs_text_set = set()
    for m in memories:
        if m["updated"] and m["prev_text"]:
            update_pairs_text_set.add((m["prev_text"], m["text"]))

    # 加载 classifier 用于抽取 embedding
    clf = RelationClassifier()
    pairs = []

    for persona_id, mems in by_persona.items():
        if len(mems) < 2:
            continue

        texts = [m["text"] for m in mems]
        features = clf._features(texts)  # [N, 1024]
        features_np = features.cpu().numpy().astype(np.float32)

        # 归一化
        norms = np.linalg.norm(features_np, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        features_np = features_np / norms

        # 余弦相似度矩阵
        sim = np.dot(features_np, features_np.T)

        # 收集高相似度对 (排除对角线、排除更新链)
        candidates = []
        n = len(mems)
        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] < similarity_threshold:
                    continue
                if (mems[i]["text"], mems[j]["text"]) in update_pairs_text_set or \
                   (mems[j]["text"], mems[i]["text"]) in update_pairs_text_set:
                    continue
                candidates.append((sim[i, j], i, j))

        # 按相似度降序，取 top max_per_persona
        candidates.sort(key=lambda x: x[0], reverse=True)
        for sim_score, i, j in candidates[:max_per_persona]:
            # old/new 顺序：按 pref_id 保证一致性
            if mems[i]["pref_id"] < mems[j]["pref_id"]:
                old_m, new_m = mems[i], mems[j]
            else:
                old_m, new_m = mems[j], mems[i]
            pairs.append({
                "old": old_m["text"],
                "new": new_m["text"],
                "pair_type": "similar",
                "persona_id": persona_id,
                "old_pref_id": old_m["pref_id"],
                "new_pref_id": new_m["pref_id"],
                "source_detail": f"cosine_sim={sim_score:.4f}",
                "pref_type_original": "",
            })

    return pairs


def build_random_ind_pairs(
    memories: List[dict],
    existing_pairs: List[dict],
    target_count: int,
) -> List[dict]:
    """2c: 同 persona 内随机配对，作为 IND 候选。"""
    existing_text_set = set()
    for p in existing_pairs:
        existing_text_set.add((p["old"], p["new"]))
        existing_text_set.add((p["new"], p["old"]))

    by_persona = defaultdict(list)
    for m in memories:
        by_persona[m["persona_id"]].append(m)

    random.seed(42)
    candidates = []

    # 每个 persona 内随机生成候选对
    for persona_id, mems in by_persona.items():
        if len(mems) < 2:
            continue
        ids = list(range(len(mems)))
        random.shuffle(ids)
        for i_idx in range(len(ids)):
            for j_idx in range(i_idx + 1, len(ids)):
                i, j = ids[i_idx], ids[j_idx]
                if (mems[i]["text"], mems[j]["text"]) in existing_text_set:
                    continue
                candidates.append((persona_id, i, j, mems))
                existing_text_set.add((mems[i]["text"], mems[j]["text"]))
                existing_text_set.add((mems[j]["text"], mems[i]["text"]))

    # 随机采样 target_count 条
    random.shuffle(candidates)
    selected = candidates[:target_count]

    pairs = []
    for persona_id, i, j, mems in selected:
        if mems[i]["pref_id"] < mems[j]["pref_id"]:
            old_m, new_m = mems[i], mems[j]
        else:
            old_m, new_m = mems[j], mems[i]
        pairs.append({
            "old": old_m["text"],
            "new": new_m["text"],
            "pair_type": "random_ind",
            "persona_id": persona_id,
            "old_pref_id": old_m["pref_id"],
            "new_pref_id": new_m["pref_id"],
            "source_detail": "random_sampling",
            "pref_type_original": "",
        })

    return pairs


def build_all_pairs(memories_path: str, output_path: str, config: dict) -> Dict[str, int]:
    """构造全部配对并写入输出文件。

    Returns:
        {"update_chain": N, "similar": N, "random_ind": N}
    """
    memories = load_memories(memories_path)
    print(f"加载 {len(memories)} 条原子记忆")

    # 2a: 更新链配对
    update_pairs = build_update_chain_pairs(memories)
    print(f"2a 更新链配对: {len(update_pairs)} 对")

    # 2b: 相似配对
    sim_threshold = config.get("similarity_threshold", 0.85)
    max_per_persona = config.get("max_similar_pairs_per_persona", 50)
    similar_pairs = build_similar_pairs(memories, sim_threshold, max_per_persona)
    print(f"2b 相似配对: {len(similar_pairs)} 对")

    # 2c: 随机 IND 配对（初步估计数量，Step 4 会精调）
    all_nonrandom = update_pairs + similar_pairs
    # 预估 IND 数量：设非 IND 总量 N，IND = N * 0.3 / 0.7
    est_non_ind = len(all_nonrandom)  # 预估全为非 IND
    est_ind = int(est_non_ind * config.get("ind_ratio_target", 0.30) / 0.70)
    random_pairs = build_random_ind_pairs(memories, all_nonrandom, est_ind)
    print(f"2c 随机 IND 配对: {len(random_pairs)} 对")

    # 合并写入
    all_pairs = update_pairs + similar_pairs + random_pairs
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    counts = {
        "update_chain": len(update_pairs),
        "similar": len(similar_pairs),
        "random_ind": len(random_pairs),
        "total": len(all_pairs),
    }
    print(f"全部配对: {counts} → {output_path}")
    return counts


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Step 2: Construct (old, new) pairs")
    ap.add_argument("--memories", required=True, help="Step 1 输出的原子记忆 JSONL")
    ap.add_argument("--output", required=True, help="输出 pairs JSONL 路径")
    ap.add_argument("--config", default=None, help="YAML 配置文件")
    args = ap.parse_args()

    cfg = {}
    if args.config:
        cfg = yaml.safe_load(open(args.config))
    build_all_pairs(args.memories, args.output, cfg)


if __name__ == "__main__":
    main()
