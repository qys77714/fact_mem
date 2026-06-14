"""relation_classifier CLI：批量给 jsonl 打关系标签。

输入 jsonl 每行至少含 {"old": ..., "new": ...}（其它字段原样保留）。
输出每行追加 "label" 与 "probs"。

用法:
    python cli.py --input mem.jsonl --output preds.jsonl
    python cli.py --input mem.jsonl --output preds.jsonl --backbone /path/to/Qwen3-0.6B
    python cli.py --input mem.jsonl                       # 不写 output 则打印到 stdout
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classifier import RelationClassifier


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "old" not in obj or "new" not in obj:
                raise ValueError(f"第 {ln} 行缺少 old/new 字段: {line[:80]}")
            rows.append(obj)
    return rows


def main():
    ap = argparse.ArgumentParser(description="关系五分类批量打标")
    ap.add_argument("--input", "-i", required=True, help="输入 jsonl（含 old/new）")
    ap.add_argument("--output", "-o", default=None, help="输出 jsonl，不填则打印 stdout")
    ap.add_argument("--backbone", "-b", default=None,
                    help="Qwen3-0.6B 本地目录，默认 /mnt/data_oss/models/Qwen3-0.6B")
    ap.add_argument("--no-probs", action="store_true", help="不输出 probs")
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    clf = RelationClassifier(backbone_path=args.backbone)
    preds = clf.predict_batch([(r["old"], r["new"]) for r in rows],
                              return_probs=not args.no_probs)

    out_f = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    try:
        for r, p in zip(rows, preds):
            r["label"] = p["label"]
            if not args.no_probs:
                r["probs"] = p["probs"]
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
    finally:
        if args.output:
            out_f.close()

    if args.output:
        print(f"已写出 {len(preds)} 条 -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
