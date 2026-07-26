"""去掉 knowledge-update 题 distractor 文本中的时间表达（LLM 改写，两步走）。

第一步（默认，review）：只生成改写提案，不动数据集
    产物：logs/analysis/ku_distractor_notime_proposal.json
          logs/analysis/ku_distractor_notime_review.html  ← 人工 review 修改前后
第二步（--apply）：经人工确认后，把提案写回数据集（原文件先备份 .bak）

运行：
    PYTHONPATH=src uv run --no-sync python script/strip_distractor_time.py           # 生成提案+review
    PYTHONPATH=src uv run --no-sync python script/strip_distractor_time.py --apply   # 确认后写回
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from utils.llm_api import load_api_chat_completion  # noqa: E402

STRIP_TIME_PROMPT = """You will rewrite a memory sentence to remove temporal anchors. The sentence is a candidate memory for answering the QUESTION below.

REMOVE every expression that anchors WHEN the fact was stated, held, or happened. Delete the whole phrase, do not rephrase it. Examples (all must be fully removed):
- "as of 2023/01/15", "as of late 2022", "as of January 2023"
- "back in early 2023", "back in 2022/11/30", "in early 2023", "in late 2022", "in mid-2023", "in September 2022", "in 2022", "around May 2023"
- "at the start of the year", "by the end of 2022", "by the spring of 2023", "during the summer of 2023", "throughout February 2023", "before 2023/01/15", "on 2022/12/01", "recently"

The ONLY exception — KEEP a temporal expression when the QUESTION itself asks for that kind of time information and it is the value being asked about:
- QUESTION asks which day / what time / how often → keep weekdays, clock times, frequencies ("every Monday", "on Tuesday evenings", "twice a week")
- QUESTION asks about a specific time window → keep that window phrase ("in the last 3 months")
Calendar dates and years are NEVER the exception unless the QUESTION explicitly asks for a date.

Keep every other word, fact, and value unchanged (including phrases like "The user noted that"). Fix grammar minimally so the sentence stays fluent.
Return ONLY the rewritten sentence, nothing else. If nothing should be removed, return the sentence unchanged.

QUESTION: {question}

Sentence: {text}

Rewritten sentence:"""


def esc(s: object) -> str:
    return html_mod.escape(str(s or ""))


def render_review_html(proposal: dict) -> str:
    parts = [
        """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>Review: KU distractor 去时间改写提案</title>
<style>
 body{font-family:-apple-system,"Segoe UI",Roboto,Arial,"PingFang SC","Microsoft YaHei",sans-serif;margin:24px;background:#f7f7f9;color:#222;}
 h1{font-size:20px;}
 .meta{color:#666;font-size:13px;margin-bottom:16px;}
 details.q{background:#fff;border:1px solid #ddd;border-radius:8px;margin-bottom:12px;padding:8px 14px;}
 details.q>summary{cursor:pointer;font-weight:600;line-height:1.5;font-size:14px;}
 table{border-collapse:collapse;width:100%;margin:8px 0;font-size:13px;}
 th,td{border:1px solid #e2e2e2;padding:6px 8px;text-align:left;vertical-align:top;line-height:1.5;}
 th{background:#f0f0f3;font-size:12px;}
 td.idx{width:36px;color:#888;}
 td.orig{width:44%;} td.new{width:44%;background:#f2f9f2;}
 .same{background:#fff6e5 !important;}
 .tag{display:inline-block;background:#d9534f;color:#fff;border-radius:4px;padding:0 6px;font-size:11px;margin-left:6px;}
</style></head><body>
<h1>Review：knowledge-update distractor 去时间改写提案（未写回数据集）</h1>"""
    ]
    n_total = sum(len(q["distractors"]) for q in proposal["questions"])
    n_changed = sum(1 for q in proposal["questions"] for d in q["distractors"] if d["stripped"] != d["original"])
    n_same = n_total - n_changed
    parts.append(
        f'<div class="meta">模型：{esc(proposal["model"])}　共 {len(proposal["questions"])} 题、'
        f"{n_total} 条 distractor；改写 {n_changed} 条，未变化 {n_same} 条（未变化行标黄，需人工留意）。"
        "golden_memory 不动。确认无误后运行 <code>--apply</code> 写回。</div>"
    )
    for q in proposal["questions"]:
        parts.append(
            f'<details class="q" open><summary>{esc(q["question_id"])} — {esc(q["question"])}</summary>'
            "<table><tr><th>#</th><th>原文</th><th>改写后（去时间）</th></tr>"
        )
        for i, d in enumerate(q["distractors"]):
            same = d["stripped"] == d["original"]
            cls = ' class="same"' if same else ""
            tag = '<span class="tag">未变化</span>' if same else ""
            parts.append(
                f'<tr{cls}><td class="idx">{i}</td><td class="orig">{esc(d["original"])}</td>'
                f'<td class="new">{esc(d["stripped"])}{tag}</td></tr>'
            )
        parts.append("</table></details>")
    parts.append("</body></html>")
    return "".join(parts)


def do_review(args: argparse.Namespace) -> None:
    data = json.loads(Path(args.data).read_text())
    ku = [d for d in data if d.get("question_type") == "knowledge-update"]

    client = load_api_chat_completion(args.model, async_=False)

    def strip(question: str, text: str) -> str:
        raw = client.get_response_chat(
            [{"role": "user", "content": STRIP_TIME_PROMPT.format(question=question, text=text)}],
            max_new_tokens=256,
            temperature=0,
            verbose=False,
        )
        out = (raw or "").strip().strip('"')
        return out or text  # LLM 失败则保留原文

    jobs = [
        (qi, di, item["question"], d["text"])
        for qi, item in enumerate(ku)
        for di, d in enumerate(item["distractors"])
    ]
    stripped: dict[tuple[int, int], str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(strip, q, t): (qi, di) for qi, di, q, t in jobs}
        for n, fut in enumerate(as_completed(futs), 1):
            stripped[futs[fut]] = fut.result()
            if n % 100 == 0:
                print(f"{n}/{len(jobs)} done")

    proposal = {
        "model": args.model,
        "data": args.data,
        "questions": [
            {
                "question_id": item["question_id"],
                "question": item["question"],
                "distractors": [
                    {"original": d["text"], "stripped": stripped[(qi, di)]}
                    for di, d in enumerate(item["distractors"])
                ],
            }
            for qi, item in enumerate(ku)
        ],
    }
    out = Path(args.proposal)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(proposal, ensure_ascii=False, indent=1))
    html_path = out.with_name("ku_distractor_notime_review.html")
    html_path.write_text(render_review_html(proposal))
    print(f"proposal -> {out}")
    print(f"review   -> {html_path}")


def do_apply(args: argparse.Namespace) -> None:
    proposal = json.loads(Path(args.proposal).read_text())
    by_qid = {q["question_id"]: q for q in proposal["questions"]}

    data_path = Path(args.data)
    data = json.loads(data_path.read_text())
    n = 0
    for item in data:
        q = by_qid.get(item.get("question_id"))
        if not q or item.get("question_type") != "knowledge-update":
            continue
        assert len(item["distractors"]) == len(q["distractors"]), item["question_id"]
        for d, p in zip(item["distractors"], q["distractors"]):
            assert d["text"] == p["original"], f"文本不匹配: {item['question_id']}"
            if p["stripped"] != d["text"]:
                d["text_before_time_strip"] = d["text"]
                d["text"] = p["stripped"]
                n += 1
    bak = data_path.with_suffix(data_path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(data_path, bak)
        print(f"backup -> {bak}")
    data_path.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(f"applied {n} rewrites -> {data_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/preprocessed/longmemeval_s_hybrid_golden.json")
    ap.add_argument("--model", default="gemma4-26B")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--proposal", default="logs/analysis/ku_distractor_notime_proposal.json")
    ap.add_argument("--apply", action="store_true", help="把提案写回数据集（先备份）")
    args = ap.parse_args()
    if args.apply:
        do_apply(args)
    else:
        do_review(args)


if __name__ == "__main__":
    main()
