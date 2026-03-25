#!/usr/bin/env python3
"""
Fill empty ``answer`` in locomo experiment jsonl from raw LoCoMo JSON.

Adversarial items only have ``adversarial_answer`` in raw data; older pipelines
wrote empty ``answer`` into jsonl. This script keys by (history_name, question).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]


def _gold_answer_from_qa(qa: dict) -> str:
    """Raw LoCoMo: ``answer`` or adversarial-only ``adversarial_answer``."""
    a = qa.get("answer", None)
    if a is not None and str(a).strip() != "":
        return str(a)
    adv = qa.get("adversarial_answer", None)
    if adv is not None and str(adv).strip() != "":
        return str(adv)
    return ""


def _build_lookup(raw_path: Path) -> Dict[Tuple[str, str], str]:
    with raw_path.open(encoding="utf-8") as f:
        data = json.load(f)
    lookup: Dict[Tuple[str, str], str] = {}
    for item in data:
        sid = str(item.get("sample_id", item.get("history_id", "")))
        for qa in item.get("qa", []) or []:
            q = (qa.get("question") or "").strip()
            if not q:
                continue
            gold = _gold_answer_from_qa(qa)
            if gold:
                lookup[(sid, q)] = gold
    return lookup


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw",
        type=Path,
        default=ROOT / "data/raw_data/locomo10.json",
        help="Raw LoCoMo JSON (default: data/raw_data/locomo10.json)",
    )
    p.add_argument("jsonl", type=Path, help="Experiment .jsonl to fix")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: overwrite the input jsonl)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print counts only")
    args = p.parse_args()

    jsonl_path = args.jsonl.resolve()
    raw_path = args.raw.resolve()
    if not raw_path.is_file():
        sys.exit(f"raw file not found: {raw_path}")
    if not jsonl_path.is_file():
        sys.exit(f"jsonl not found: {jsonl_path}")

    out_path = args.output
    if out_path is None:
        out_path = jsonl_path

    lookup = _build_lookup(raw_path)
    lines_out: list[str] = []
    filled = 0
    still_empty = 0
    skipped_nonempty = 0
    not_adversarial = 0

    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qt = rec.get("question_type")
            ans = rec.get("answer")
            if qt != "adversarial":
                not_adversarial += 1
                lines_out.append(json.dumps(rec, ensure_ascii=False))
                continue
            if ans is not None and str(ans).strip() != "":
                skipped_nonempty += 1
                lines_out.append(json.dumps(rec, ensure_ascii=False))
                continue
            h = str(rec.get("history_name", ""))
            q = str(rec.get("question", "")).strip()
            gold = lookup.get((h, q), "")
            if gold:
                rec["answer"] = gold
                filled += 1
            else:
                still_empty += 1
            lines_out.append(json.dumps(rec, ensure_ascii=False))

    print(
        f"adversarial filled={filled} still_empty={still_empty} "
        f"skipped_nonempty={skipped_nonempty} other_rows={not_adversarial}"
    )

    if args.dry_run:
        return

    text = "\n".join(lines_out) + ("\n" if lines_out else "")
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
