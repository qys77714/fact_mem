"""
Evaluate gemma4-26B on data/test_relation.jsonl for relation classification.
Computes per-class and macro/micro precision, recall, F1.
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prompts import render_prompt
from utils.llm_api import load_api_chat_completion

# --- Config ---
MODEL_NAME = "gemma4-26B"
TEMPLATE_NAME = "RD_0_relation_classify.jinja"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "test_relation.jsonl"
VALID_LABELS = {"IND", "EQV", "OSN", "NSO", "CON"}

# --- Load data ---
samples = []
with open(DATA_PATH) as f:
    for line in f:
        samples.append(json.loads(line))

print(f"Loaded {len(samples)} samples")

# --- Load model ---
client = load_api_chat_completion(MODEL_NAME)

# --- Classify each sample ---
predictions = []
errors = 0
empty_responses = 0
t0 = time.time()

for i, s in enumerate(samples):
    prompt = render_prompt(TEMPLATE_NAME, m_old=s["old"], m_new=s["new"])
    messages = [{"role": "user", "content": prompt}]

    response = client.get_response_chat(
        messages=messages,
        max_new_tokens=64,
        temperature=0.0,
    )

    # Parse JSON from response
    pred = None
    if response is None:
        empty_responses += 1
    else:
        # Try to extract JSON object
        try:
            # Direct parse
            parsed = json.loads(response.strip())
            pred = parsed.get("relation", None)
        except json.JSONDecodeError:
            # Try regex to find {"relation": "XXX"}
            match = re.search(r'\{"relation"\s*:\s*"([A-Z]+)"\}', response)
            if match:
                pred = match.group(1)

    if pred not in VALID_LABELS:
        # Fallback: try to find any valid label in response
        for label in VALID_LABELS:
            if label in response if response else "":
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

# --- Compute metrics ---
# Per-class
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

# Macro average
macro_p = sum(v["precision"] for v in per_class.values()) / len(classes)
macro_r = sum(v["recall"] for v in per_class.values()) / len(classes)
macro_f1 = sum(v["f1"] for v in per_class.values()) / len(classes)

# Micro average (overall accuracy since balanced)
micro_tp = sum(v["tp"] for v in per_class.values())
micro_fp = sum(v["fp"] for v in per_class.values())
micro_fn = sum(v["fn"] for v in per_class.values())
micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

# Accuracy
correct = sum(1 for s, p in zip(samples, predictions) if s["relation"] == p)
accuracy = correct / len(samples)

# --- Print results ---
print("\n" + "=" * 70)
print("PER-CLASS RESULTS")
print("=" * 70)
print(f"{'Class':<8} {'TP':>5} {'FP':>5} {'FN':>5} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 60)
for cls in classes:
    v = per_class[cls]
    print(f"{cls:<8} {v['tp']:>5} {v['fp']:>5} {v['fn']:>5} {v['precision']:>10.4f} {v['recall']:>10.4f} {v['f1']:>10.4f}")

print("\n" + "=" * 70)
print("SUMMARY")
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

# --- Confusion matrix ---
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

# --- Error analysis ---
print("\n" + "=" * 70)
print("ERROR EXAMPLES (first 3 per confusion type, pred!=true)")
print("=" * 70)
error_by_type = Counter()
for s, p in zip(samples, predictions):
    if p is not None and s["relation"] != p:
        error_by_type[(s["relation"], p)] += 1

for (true_cls, pred_cls), count in error_by_type.most_common(10):
    print(f"\n  {true_cls} → {pred_cls} ({count} cases):")
    shown = 0
    for s, p in zip(samples, predictions):
        if s["relation"] == true_cls and p == pred_cls:
            print(f"    old: {s['old'][:80]}")
            print(f"    new: {s['new'][:80]}")
            shown += 1
            if shown >= 3:
                break

# --- Save predictions ---
out_path = Path(__file__).resolve().parent.parent / "data" / "test_relation_predictions.jsonl"
with open(out_path, "w") as f:
    for s, p in zip(samples, predictions):
        f.write(json.dumps({"old": s["old"], "new": s["new"],
                            "true": s["relation"], "pred": p}) + "\n")
print(f"\nPredictions saved to {out_path}")
