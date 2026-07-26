"""
Evaluate gemma4-26B CoT on data/test_relation.jsonl for relation classification.
Computes per-class and macro/micro precision, recall, F1.
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prompts import render_prompt
from utils.llm_api import load_api_chat_completion

MODEL_NAME = "gemma4-26B"
TEMPLATE_NAME = "RD_0_relation_classify_cot.jinja"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "test_relation.jsonl"
VALID_LABELS = {"IND", "EQV", "OSN", "NSO", "CON"}

samples = []
with open(DATA_PATH) as f:
    for line in f:
        samples.append(json.loads(line))

print(f"Loaded {len(samples)} samples")

client = load_api_chat_completion(MODEL_NAME)

predictions = []
errors = 0
empty_responses = 0
t0 = time.time()

for i, s in enumerate(samples):
    prompt = render_prompt(TEMPLATE_NAME, m_old=s["old"], m_new=s["new"])
    messages = [{"role": "user", "content": prompt}]

    response = client.get_response_chat(
        messages=messages,
        max_new_tokens=256,
        temperature=0.0,
    )

    pred = None
    if response is None:
        empty_responses += 1
    else:
        # Try direct JSON parse
        try:
            parsed = json.loads(response.strip())
            pred = parsed.get("relation", None)
        except json.JSONDecodeError:
            # Try regex
            match = re.search(r'"relation"\s*:\s*"([A-Z]+)"', response)
            if match:
                pred = match.group(1)

    if pred not in VALID_LABELS:
        if response:
            for label in VALID_LABELS:
                if label in response:
                    pred = label
                    break

    if pred not in VALID_LABELS:
        pred = None
        errors += 1

    predictions.append(pred)

    if (i + 1) % 50 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        print(f"  {i+1}/{len(samples)} ({rate:.1f} samples/s), "
              f"errors: {errors}, empty: {empty_responses}")

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s ({len(samples)/elapsed:.1f} samples/s)")

classes = sorted(VALID_LABELS)
per_class = {}
for cls in classes:
    tp = sum(1 for s, p in zip(samples, predictions) if s["relation"] == cls and p == cls)
    fp = sum(1 for s, p in zip(samples, predictions) if s["relation"] != cls and p == cls)
    fn = sum(1 for s, p in zip(samples, predictions) if s["relation"] == cls and p != cls)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    per_class[cls] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}

macro_p = sum(v["precision"] for v in per_class.values()) / len(classes)
macro_r = sum(v["recall"] for v in per_class.values()) / len(classes)
macro_f1 = sum(v["f1"] for v in per_class.values()) / len(classes)

micro_tp = sum(v["tp"] for v in per_class.values())
micro_fp = sum(v["fp"] for v in per_class.values())
micro_fn = sum(v["fn"] for v in per_class.values())
micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

correct = sum(1 for s, p in zip(samples, predictions) if s["relation"] == p)
accuracy = correct / len(samples)

print("\n" + "=" * 70)
print("PER-CLASS RESULTS (CoT)")
print("=" * 70)
print(f"{'Class':<8} {'TP':>5} {'FP':>5} {'FN':>5} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 60)
for cls in classes:
    v = per_class[cls]
    print(f"{cls:<8} {v['tp']:>5} {v['fp']:>5} {v['fn']:>5} {v['precision']:>10.4f} {v['recall']:>10.4f} {v['f1']:>10.4f}")

print("\n" + "=" * 70)
print("SUMMARY (CoT)")
print("=" * 70)
print(f"Accuracy:        {accuracy:.4f} ({correct}/{len(samples)})")
print(f"Macro Precision: {macro_p:.4f}")
print(f"Macro Recall:    {macro_r:.4f}")
print(f"Macro F1:        {macro_f1:.4f}")
print(f"Micro Precision: {micro_p:.4f}")
print(f"Micro Recall:    {micro_r:.4f}")
print(f"Micro F1:        {micro_f1:.4f}")
print(f"Empty responses: {empty_responses}")
print(f"Parse errors:    {errors}")

print("\n" + "=" * 70)
print("CONFUSION MATRIX (rows=true, cols=pred)")
print("=" * 70)
confusion = Counter()
for s, p in zip(samples, predictions):
    confusion[(s["relation"], p)] += 1

header = "      " + " ".join(f"{c:>5}" for c in classes)
print(header)
for true_cls in classes:
    row = f"{true_cls:<5} " + " ".join(f"{confusion[(true_cls, pred_cls)]:>5}" for pred_cls in classes)
    print(row)

# Compare with non-CoT baseline
print("\n" + "=" * 70)
print("COMPARISON: CoT vs Non-CoT")
print("=" * 70)
# Try to load baseline predictions if available
baseline_path = Path(__file__).resolve().parent.parent / "data" / "test_relation_predictions.jsonl"
if baseline_path.exists():
    baseline_preds = []
    with open(baseline_path) as f:
        for line in f:
            baseline_preds.append(json.loads(line)["pred"])

    baseline_correct = sum(1 for s, p in zip(samples, baseline_preds) if s["relation"] == p)
    baseline_acc = baseline_correct / len(samples)

    # Per-class baseline
    baseline_per_class = {}
    for cls in classes:
        tp = sum(1 for s, p in zip(samples, baseline_preds) if s["relation"] == cls and p == cls)
        fp = sum(1 for s, p in zip(samples, baseline_preds) if s["relation"] != cls and p == cls)
        fn = sum(1 for s, p in zip(samples, baseline_preds) if s["relation"] == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        baseline_per_class[cls] = {"f1": f1}

    baseline_macro_f1 = sum(v["f1"] for v in baseline_per_class.values()) / len(classes)

    print(f"{'Class':<8} {'Non-CoT F1':>12} {'CoT F1':>10} {'Δ':>8}")
    print("-" * 45)
    for cls in classes:
        delta = per_class[cls]["f1"] - baseline_per_class[cls]["f1"]
        print(f"{cls:<8} {baseline_per_class[cls]['f1']:>12.4f} {per_class[cls]['f1']:>10.4f} {delta:>+8.4f}")

    print("-" * 45)
    delta_macro = macro_f1 - baseline_macro_f1
    delta_acc = accuracy - baseline_acc
    print(f"{'Macro F1':<8} {baseline_macro_f1:>12.4f} {macro_f1:>10.4f} {delta_macro:>+8.4f}")
    print(f"{'Accuracy':<8} {baseline_acc:>12.4f} {accuracy:>10.4f} {delta_acc:>+8.4f}")

# Save
out_path = Path(__file__).resolve().parent.parent / "data" / "test_relation_predictions_cot.jsonl"
with open(out_path, "w") as f:
    for s, p in zip(samples, predictions):
        f.write(json.dumps({"old": s["old"], "new": s["new"],
                            "true": s["relation"], "pred": p}) + "\n")
print(f"\nPredictions saved to {out_path}")
