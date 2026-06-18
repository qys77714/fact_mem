"""Step 4: 比对双裁判结果，以 gemma4-26B 为准生成训练数据，控制 IND 比例。"""

import json
import os
import random
import yaml
from typing import List, Dict


def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def generate_training_data(
    judged_path: str,
    original_path: str,
    output_path: str,
    config: dict,
):
    """生成最终训练数据。

    策略:
    1. 新数据: gemma_label 与 classifier_label 不一致 → 以 gemma 为准，agree=false
    2. 新数据: 一致且非 IND → 保留，agree=true
    3. 新数据: 一致且 IND → 进入 IND 候选池
    4. 合并原有数据
    5. IND 比例控制到 30%
    """
    judged = load_jsonl(judged_path)
    ind_ratio_target = config.get("ind_ratio_target", 0.30)

    non_ind_samples = []  # 非 IND（确定加入训练集）
    ind_candidates = []   # IND 候选池
    errors = []           # gemma 解析失败的

    for p in judged:
        gemma_label = p.get("gemma_label", "")

        if gemma_label == "PARSE_ERROR":
            errors.append(p)
            continue

        # 以 gemma4-26B 标签为准
        agree = (gemma_label == p.get("classifier_label", ""))

        sample = {
            "old": p["old"],
            "new": p["new"],
            "label": gemma_label,
            "source": f"persona_{p.get('persona_id', 'unknown')}",
            "gemma_label": gemma_label,
            "classifier_label": p.get("classifier_label", ""),
            "pref_type": p.get("source_detail", ""),
            "agree": agree,
        }

        if gemma_label == "IND":
            ind_candidates.append(sample)
        else:
            non_ind_samples.append(sample)

    # 加载原有训练数据
    original = []
    if os.path.exists(original_path):
        original = load_jsonl(original_path)
        # 原有数据统一补充字段
        for o in original:
            o.setdefault("source", "original")
            o.setdefault("gemma_label", "")
            o.setdefault("classifier_label", "")
            o.setdefault("pref_type", "")
            o.setdefault("agree", True)
    else:
        print(f"警告: 原有训练数据 {original_path} 不存在，跳过合并")

    # IND 比例控制：目标 IND 占总量 30%
    total_non_ind = len(non_ind_samples) + len([o for o in original if o.get("label") != "IND"])
    ind_needed = int(total_non_ind * ind_ratio_target / (1 - ind_ratio_target))

    random.seed(42)
    if len(ind_candidates) > ind_needed:
        ind_selected = random.sample(ind_candidates, ind_needed)
        print(f"IND 候选池 {len(ind_candidates)} → 采样 {ind_needed} 条（控制 {ind_ratio_target:.0%}）")
    else:
        ind_selected = ind_candidates
        print(f"IND 候选池不足: 需要 {ind_needed}, 实际 {len(ind_candidates)}，全部保留")

    # 合并全部数据
    all_samples = non_ind_samples + ind_selected + original

    # 基于 (old, new) 去重（保留首次出现）
    seen = set()
    deduped = []
    for s in all_samples:
        key = (s["old"].strip().lower(), s["new"].strip().lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)

    # 打乱
    random.shuffle(deduped)

    # 统计
    from collections import Counter
    label_counts = Counter(s["label"] for s in deduped)
    ind_pct = label_counts.get("IND", 0) / len(deduped) if deduped else 0

    print(f"训练数据统计:")
    print(f"  新数据-非IND: {len(non_ind_samples)}")
    print(f"  新数据-IND: {len(ind_selected)}")
    print(f"  原有数据: {len(original)}")
    print(f"  去重后合计: {len(deduped)}")
    print(f"  gemma 解析失败: {len(errors)}")
    print(f"  标签分布: {dict(label_counts)}")
    print(f"  IND 占比: {ind_pct:.2%}")
    print(f"  不一致样本 (agree=false): {sum(1 for s in deduped if s.get('agree') is False)}")

    # 写入
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in deduped:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"训练数据写入 → {output_path}")
    return deduped


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Step 4: Generate training data")
    ap.add_argument("--judged", required=True, help="Step 3 输出的 judged pairs JSONL")
    ap.add_argument("--original", default="non_ind.jsonl", help="原有训练数据")
    ap.add_argument("--output", required=True, help="输出训练数据 JSONL 路径")
    ap.add_argument("--config", default=None, help="YAML 配置文件")
    args = ap.parse_args()

    cfg = {}
    if args.config:
        cfg = yaml.safe_load(open(args.config))
    generate_training_data(args.judged, args.original, args.output, cfg)


if __name__ == "__main__":
    main()
