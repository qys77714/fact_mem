"""统计 knowledge-update 题型中 confusion+golden 集合内两两关系的五类比例。

对 data/preprocessed/longmemeval_s_hybrid_golden.json 中 question_type ==
"knowledge-update" 的每道题，将 golden_memory 与 distractors 合并为一个集合，
集合内两两配对（按 date 早的作 OLD FACT、晚的作 NEW FACT，与灌库时间顺序一致），
用 relation_decision LLM 后端同款 prompt
（RD_0_relation_classify.jinja，单条 user message，
temperature=0，结构化输出）调 gemma4-26B 分类，输出五类比例。

运行：PYTHONPATH=src uv run --no-sync python script/analyze_ku_confusion_relation.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from memory.candidate_ingest.prompts import (  # noqa: E402
    build_relation_classification_prompt,
)
from memory.candidate_ingest.schemas import (  # noqa: E402
    LME_RELATION_CLASSIFICATION_RESPONSE_FORMAT,
)
from utils.llm_api import load_api_chat_completion  # noqa: E402

VALID = {"IND", "EQV", "NSO", "OSN", "CON"}


def parse_relation(raw: str | None) -> str:
    if not raw:
        return "PARSE_FAIL"
    try:
        rel = str(json.loads(raw).get("relation", "")).strip().upper()
        return rel if rel in VALID else "PARSE_FAIL"
    except Exception:
        m = re.search(r"\b(IND|EQV|NSO|OSN|CON)\b", raw.upper())
        return m.group(1) if m else "PARSE_FAIL"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/preprocessed/longmemeval_s_hybrid_golden.json")
    ap.add_argument("--model", default="gemma4-26B")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="logs/analysis/ku_confusion_relation_gemma4.json")
    args = ap.parse_args()

    data = json.loads(Path(args.data).read_text())
    ku = [d for d in data if d.get("question_type") == "knowledge-update"]

    # 每题：golden + distractor 合并为一个集合，集合内两两配对；
    # date 早者作 OLD、晚者作 NEW（同 date 时保持列表原序）。
    pairs = []
    for item in ku:
        pool = [
            {"kind": "golden", "idx": gi, "text": g["text"], "date": g.get("date", "")}
            for gi, g in enumerate(item["golden_memory"])
        ] + [
            {"kind": "distractor", "idx": di, "text": d["text"], "date": d.get("date", "")}
            for di, d in enumerate(item["distractors"])
        ]
        for a, b in itertools.combinations(pool, 2):
            old, new = (a, b) if a["date"] <= b["date"] else (b, a)
            pair_type = "-".join(sorted([a["kind"], b["kind"]]))
            pairs.append(
                {
                    "question_id": item["question_id"],
                    "pair_type": pair_type,  # golden-golden / distractor-golden / distractor-distractor
                    "old": f"{old['kind']}[{old['idx']}]",
                    "new": f"{new['kind']}[{new['idx']}]",
                    "m_old": old["text"],
                    "m_new": new["text"],
                }
            )
    print(f"questions={len(ku)}  pairs={len(pairs)}")

    client = load_api_chat_completion(args.model, async_=False)

    def classify(p: dict) -> dict:
        user_prompt = build_relation_classification_prompt(
            p["m_old"], p["m_new"], language="en"
        )
        raw = client.get_response_chat(
            [{"role": "user", "content": user_prompt}],
            max_new_tokens=64,
            temperature=0,
            response_format=LME_RELATION_CLASSIFICATION_RESPONSE_FORMAT,
            verbose=False,
        )
        return {**p, "relation": parse_relation(raw), "raw": raw}

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(classify, p) for p in pairs]
        for i, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if i % 200 == 0:
                print(f"{i}/{len(pairs)} done")

    def dist(rows: list[dict]) -> dict:
        c = Counter(r["relation"] for r in rows)
        n = len(rows) or 1
        return {k: {"count": v, "ratio": round(v / n, 4)} for k, v in c.most_common()}

    summary = {
        "model": args.model,
        "n_questions": len(ku),
        "n_pairs": len(results),
        "overall": dist(results),
        "by_pair_type": {
            pt: dist([r for r in results if r["pair_type"] == pt])
            for pt in sorted({r["pair_type"] for r in results})
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=1))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
