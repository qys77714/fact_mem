#!/usr/bin/env python3
"""
评估 relation classification 准确性。

Stage:
  annotate  — 用 deepseek-v4-pro 标注 golden label（data/test_golden.jsonl）
  classify  — 用指定模型分类（--model gemma4-26B / Qwen3-4B）
  evaluate  — 计算 Precision/Recall/F1、混淆矩阵、Cohen's κ

用法:
  uv run --no-sync python script/eval_relation_accuracy.py --stage annotate
  uv run --no-sync python script/eval_relation_accuracy.py --stage classify --model gemma4-26B
  uv run --no-sync python script/eval_relation_accuracy.py --stage evaluate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from prompts import render_prompt
from utils.llm_api import load_api_chat_completion

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
TEST_FILE = REPO_ROOT / "data" / "test.jsonl"
GOLDEN_FILE = REPO_ROOT / "data" / "test_golden.jsonl"
PRED_DIR = REPO_ROOT / "data"

LABELS = ["IND", "EQV", "OSN", "NSO", "CON"]
VALID_LABELS = set(LABELS)

SYSTEM_TEMPLATE = "RD_0_relation_classify.jinja"

# 标注用模型（DeepSeek API，目前最强）
ANNOTATOR_MODEL = "deepseek-v3"

# batch / retry
BATCH_SIZE = 10
MAX_RETRIES = 3
RETRY_DELAY = 2.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load_test_pairs(path: Path) -> List[Dict[str, str]]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pairs.append({"old": obj["old"].strip(), "new": obj["new"].strip()})
    return pairs


def build_messages(old: str, new: str, system_template: str = SYSTEM_TEMPLATE) -> List[Dict[str, str]]:
    user_prompt = render_prompt(system_template, m_old=old, m_new=new)
    return [
        {"role": "user", "content": user_prompt},
    ]


def parse_relation(raw: Optional[str]) -> str:
    """从 LLM 原始输出中提取 relation 标签。"""
    if not raw:
        return "IND"
    raw = raw.strip()
    # 尝试直接 JSON 解析
    try:
        obj = json.loads(raw)
        label = str(obj.get("relation", "")).strip().upper()
        if label in VALID_LABELS:
            return label
    except (json.JSONDecodeError, TypeError):
        pass
    # 尝试从文本中提取
    for label in LABELS:
        if label in raw:
            return label
    # 回退：找 "relation": "XXX"
    import re
    m = re.search(r'"relation"\s*:\s*"([A-Z]+)"', raw)
    if m and m.group(1) in VALID_LABELS:
        return m.group(1)
    return "IND"


def classify_batch(
    client,
    pairs: List[Dict[str, str]],
    batch_size: int = BATCH_SIZE,
    max_retries: int = MAX_RETRIES,
    label: str = "",
    use_response_format: bool = True,
    system_template: str = SYSTEM_TEMPLATE,
) -> List[str]:
    """批量分类，返回标签列表。"""
    results: List[Optional[str]] = [None] * len(pairs)

    for batch_start in range(0, len(pairs), batch_size):
        batch_end = min(batch_start + batch_size, len(pairs))
        batch_pairs = pairs[batch_start:batch_end]

        for attempt in range(max_retries):
            try:
                for i, pair in enumerate(batch_pairs):
                    idx = batch_start + i
                    msgs = build_messages(pair["old"], pair["new"], system_template=system_template)
                    kwargs = dict(max_new_tokens=128, temperature=0.0, verbose=False)
                    if use_response_format:
                        kwargs["response_format"] = {"type": "json_object"}
                    raw = client.get_response_chat(msgs, **kwargs)
                    results[idx] = parse_relation(raw)

                # 批次成功
                done = batch_end - batch_start
                if label:
                    print(f"  [{label}] {batch_end}/{len(pairs)} 完成")
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    wait = RETRY_DELAY * (attempt + 1)
                    print(f"  [{label}] batch {batch_start}-{batch_end} 失败 (attempt {attempt+1}): {e}，{wait}s 后重试...")
                    time.sleep(wait)
                else:
                    print(f"  [{label}] batch {batch_start}-{batch_end} 最终失败: {e}，回退 IND")
                    for i in range(batch_start, batch_end):
                        if results[i] is None:
                            results[i] = "IND"

        # 批次间短暂休息，避免 rate limit
        time.sleep(0.5)

    return [r if r else "IND" for r in results]


# ---------------------------------------------------------------------------
# stage: annotate
# ---------------------------------------------------------------------------
def stage_annotate(annotator_model: str = ANNOTATOR_MODEL, system_template: str = SYSTEM_TEMPLATE):
    print(f"加载测试集: {TEST_FILE}")
    pairs = load_test_pairs(TEST_FILE)
    print(f"共 {len(pairs)} 对")

    # 根据模板版本命名 golden 文件
    if system_template != SYSTEM_TEMPLATE:
        tag = system_template.replace("RD_0_relation_classify", "").replace(".jinja", "").strip("_") or "default"
        golden_file = REPO_ROOT / "data" / f"test_golden_{tag}.jsonl"
    else:
        golden_file = GOLDEN_FILE

    # 检查是否已有中间结果
    if golden_file.exists():
        existing = load_golden(golden_file)
        if len(existing) == len(pairs):
            print(f"Golden 文件已存在且完整 ({len(existing)} 条)，跳过标注。")
            print(f"如需重新标注，请删除 {golden_file}")
            return
        print(f"Golden 文件不完整 ({len(existing)}/{len(pairs)})，从头重新标注。")

    print(f"标注模型: {annotator_model}")
    print(f"System template: {system_template}")
    print(f"User template: {USER_TEMPLATE}")
    print()

    client = load_api_chat_completion(annotator_model)
    # v3 支持 response_format，v4-pro 不支持
    _use_rf = "v4-pro" not in annotator_model
    labels = classify_batch(client, pairs, label="annotate", use_response_format=_use_rf,
                            system_template=system_template)

    # 保存
    with open(golden_file, "w", encoding="utf-8") as f:
        for pair, label in zip(pairs, labels):
            obj = {"old": pair["old"], "new": pair["new"], "golden_relation": label}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # 统计分布
    dist = Counter(labels)
    print(f"\nGolden label 分布 ({len(pairs)} 条):")
    for lbl in LABELS:
        print(f"  {lbl}: {dist.get(lbl, 0)} ({dist.get(lbl, 0)/len(pairs)*100:.1f}%)")
    print(f"\n已保存: {golden_file}")


def load_golden(path: Path) -> List[Dict]:
    results = []
    if not path.exists():
        return results
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


# ---------------------------------------------------------------------------
# stage: classify
# ---------------------------------------------------------------------------
def stage_classify(model: str, system_template: str = SYSTEM_TEMPLATE):
    if not GOLDEN_FILE.exists():
        print(f"错误: Golden 文件不存在 ({GOLDEN_FILE})，请先运行 --stage annotate")
        sys.exit(1)

    # 根据模板版本命名预测文件
    if system_template != SYSTEM_TEMPLATE:
        tag = system_template.replace("RD_0_relation_classify", "").replace(".jinja", "").strip("_") or "default"
        pred_file = PRED_DIR / f"test_{model.replace('-', '_').replace('.', '_')}_pred_{tag}.jsonl"
    else:
        pred_file = PRED_DIR / f"test_{model.replace('-', '_').replace('.', '_')}_pred.jsonl"

    pairs = load_test_pairs(TEST_FILE)
    print(f"加载测试集: {TEST_FILE} ({len(pairs)} 对)")
    print(f"分类模型: {model}")
    print(f"System template: {system_template}")
    print(f"输出文件: {pred_file}")
    print()

    client = load_api_chat_completion(model)
    labels = classify_batch(client, pairs, label=model, system_template=system_template)

    with open(pred_file, "w", encoding="utf-8") as f:
        for pair, label in zip(pairs, labels):
            obj = {"old": pair["old"], "new": pair["new"], "pred_relation": label}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    dist = Counter(labels)
    print(f"\n{model} 预测分布 ({len(pairs)} 条):")
    for lbl in LABELS:
        print(f"  {lbl}: {dist.get(lbl, 0)} ({dist.get(lbl, 0)/len(pairs)*100:.1f}%)")
    print(f"\n已保存: {pred_file}")


# ---------------------------------------------------------------------------
# stage: evaluate
# ---------------------------------------------------------------------------
def stage_evaluate(golden_file: Optional[Path] = None):
    if golden_file is None:
        golden_file = GOLDEN_FILE
    if not golden_file.exists():
        print(f"错误: Golden 文件不存在 ({golden_file})，请先运行 --stage annotate")
        sys.exit(1)

    golden_data = load_golden(golden_file)
    golden_labels = [d["golden_relation"] for d in golden_data]
    print(f"Golden: {golden_file} ({len(golden_labels)} 条)")

    # 找所有预测文件（兼容 v3 的 test_*_pred.jsonl 和 v4 的 test_*_pred_v4.jsonl）
    pred_files = sorted(PRED_DIR.glob("test_*_pred*.jsonl"))
    # 过滤掉 golden 文件
    pred_files = [f for f in pred_files if "golden" not in f.name]

    if not pred_files:
        print("错误: 未找到预测文件 (*_pred.jsonl)，请先运行 --stage classify")
        sys.exit(1)

    for pf in pred_files:
        model_name = pf.stem.replace("test_", "").replace("_pred", "")
        print(f"\n{'='*70}")
        print(f"模型: {model_name}")
        print(f"{'='*70}")

        pred_data = []
        with open(pf, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pred_data.append(json.loads(line))
        pred_labels = [d["pred_relation"] for d in pred_data]

        if len(pred_labels) != len(golden_labels):
            print(f"  警告: 预测数量 ({len(pred_labels)}) != Golden ({len(golden_labels)})，跳过")
            continue

        # ---- 整体 Accuracy ----
        correct = sum(1 for g, p in zip(golden_labels, pred_labels) if g == p)
        acc = correct / len(golden_labels)
        print(f"\n  整体 Accuracy: {correct}/{len(golden_labels)} = {acc:.4f} ({acc*100:.2f}%)")

        # ---- Per-class Precision / Recall / F1 ----
        print(f"\n  {'Class':<6} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}")
        print(f"  {'-'*48}")

        # 计算每类指标
        class_metrics = {}
        for lbl in LABELS:
            tp = sum(1 for g, p in zip(golden_labels, pred_labels) if g == lbl and p == lbl)
            fp = sum(1 for g, p in zip(golden_labels, pred_labels) if g != lbl and p == lbl)
            fn = sum(1 for g, p in zip(golden_labels, pred_labels) if g == lbl and p != lbl)
            support = sum(1 for g in golden_labels if g == lbl)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            class_metrics[lbl] = {"precision": prec, "recall": rec, "f1": f1, "support": support}
            print(f"  {lbl:<6} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {support:>8}")

        # Macro / Weighted avg
        macro_p = sum(class_metrics[l]["precision"] for l in LABELS) / len(LABELS)
        macro_r = sum(class_metrics[l]["recall"] for l in LABELS) / len(LABELS)
        macro_f1 = sum(class_metrics[l]["f1"] for l in LABELS) / len(LABELS)

        total = len(golden_labels)
        weighted_p = sum(class_metrics[l]["precision"] * class_metrics[l]["support"] for l in LABELS) / total
        weighted_r = sum(class_metrics[l]["recall"] * class_metrics[l]["support"] for l in LABELS) / total
        weighted_f1 = sum(class_metrics[l]["f1"] * class_metrics[l]["support"] for l in LABELS) / total

        print(f"  {'-'*48}")
        print(f"  {'macro':<6} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f} {total:>8}")
        print(f"  {'weight':<6} {weighted_p:>10.4f} {weighted_r:>10.4f} {weighted_f1:>10.4f} {total:>8}")

        # ---- 混淆矩阵 ----
        print(f"\n  混淆矩阵 (行=Golden, 列=Pred):")
        header = "        " + "".join(f"{l:>7}" for l in LABELS)
        print(header)
        for gl in LABELS:
            row = f"  {gl:<6}"
            for pl in LABELS:
                cnt = sum(1 for g, p in zip(golden_labels, pred_labels) if g == gl and p == pl)
                row += f"{cnt:>7}"
            print(row)

        # ---- Cohen's κ ----
        kappa = _cohen_kappa(golden_labels, pred_labels)
        print(f"\n  Cohen's κ: {kappa:.4f}")

        # ---- 各类别错误分析 ----
        print(f"\n  各类别误判详情:")
        for lbl in LABELS:
            errors = []
            for g, p in zip(golden_labels, pred_labels):
                if g == lbl and p != lbl:
                    errors.append(p)
            if errors:
                err_dist = Counter(errors)
                err_str = ", ".join(f"{k}:{v}" for k, v in err_dist.most_common())
                total_cls = sum(1 for g in golden_labels if g == lbl)
                print(f"    {lbl} (n={total_cls}): {len(errors)} 误判 → {err_str}")
            else:
                total_cls = sum(1 for g in golden_labels if g == lbl)
                print(f"    {lbl} (n={total_cls}): 0 误判")

        # ---- 跨模型一致性（如果有多个预测文件） ----
        if len(pred_files) > 1:
            print(f"\n  Backend 间一致性:")
            for pf2 in pred_files:
                if pf2 == pf:
                    continue
                m2 = pf2.stem.replace("test_", "").replace("_pred", "")
                pred2_labels = []
                with open(pf2, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            pred2_labels.append(json.loads(line)["pred_relation"])
                if len(pred2_labels) != len(pred_labels):
                    continue
                agree = sum(1 for a, b in zip(pred_labels, pred2_labels) if a == b)
                agree_pct = agree / len(pred_labels)
                k2 = _cohen_kappa(pred_labels, pred2_labels)
                print(f"    {model_name} vs {m2}: agreement={agree}/{len(pred_labels)} ({agree_pct:.4f}), κ={k2:.4f}")

                # 一致性矩阵
                print(f"\n    {model_name} vs {m2} 一致性矩阵:")
                header2 = "        " + "".join(f"{l:>7}" for l in LABELS)
                print(header2)
                for l1 in LABELS:
                    row = f"    {l1:<4}"
                    for l2 in LABELS:
                        cnt = sum(1 for a, b in zip(pred_labels, pred2_labels) if a == l1 and b == l2)
                        row += f"{cnt:>7}"
                    print(row)


def _cohen_kappa(y1: List[str], y2: List[str]) -> float:
    """计算 Cohen's κ。"""
    n = len(y1)
    if n == 0:
        return 0.0

    # 观测一致率
    po = sum(1 for a, b in zip(y1, y2) if a == b) / n

    # 期望一致率
    all_labels = sorted(set(y1) | set(y2))
    pe = 0.0
    for lbl in all_labels:
        p1 = y1.count(lbl) / n
        p2 = y2.count(lbl) / n
        pe += p1 * p2

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Relation Classification 准确性评估")
    parser.add_argument("--stage", required=True, choices=["annotate", "classify", "evaluate"],
                        help="annotate=标注golden; classify=模型分类; evaluate=计算指标")
    parser.add_argument("--model", default="gemma4-26B",
                        help="分类模型名 (classify stage, 默认 gemma4-26B)")
    parser.add_argument("--annotator", default=ANNOTATOR_MODEL,
                        help=f"标注模型名 (annotate stage, 默认 {ANNOTATOR_MODEL})")
    parser.add_argument("--template", default=SYSTEM_TEMPLATE,
                        help=f"System prompt 模板 (默认 {SYSTEM_TEMPLATE})")
    parser.add_argument("--golden", default=None,
                        help="Golden 文件路径 (evaluate stage, 默认 data/test_golden.jsonl)")
    args = parser.parse_args()

    if args.stage == "annotate":
        stage_annotate(args.annotator, system_template=args.template)
    elif args.stage == "classify":
        stage_classify(args.model, system_template=args.template)
    elif args.stage == "evaluate":
        golden_file = Path(args.golden) if args.golden else None
        stage_evaluate(golden_file=golden_file)


if __name__ == "__main__":
    main()
