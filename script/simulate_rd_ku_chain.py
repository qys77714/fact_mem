"""模拟 RD 顺序灌库链，观察 fusion 前后文本（knowledge-update 题型）。

每题：distractors 按 date 升序排列，golden 放最后（new-value golden 排最末）。
顺序灌入：第 N 条与第 N-1 条**原子原文**做关系分类（同 relation_decision LLM 后端
prompt：v3 英文 system + user 模板，temperature=0）。

关系非 IND 时按真实 RD 的答题记忆 C 就地融合逻辑滚动融合：
  current_memory = 上一条原子已挂的 C 文本（若有），否则上一条原子原文
  C' = LLM_fuse(current_memory, m_new, relation)   # per-relation 模板（CON/OSN/NSO/EQV）
原子原文永不改写，后续分类仍用原子；C 只滚动记录。

产物：JSON 明细 + HTML 可视化报告（每题一个区块，展示链式关系判断与 fusion 前后文本）。

运行：PYTHONPATH=src uv run --no-sync python script/simulate_rd_ku_chain.py
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from memory.candidate_ingest.prompts import (  # noqa: E402
    build_lme_answer_fuse_prompt,
    build_relation_classification_prompt,
)
from memory.candidate_ingest.schemas import (  # noqa: E402
    LME_RELATION_CLASSIFICATION_RESPONSE_FORMAT,
)
from prompts import render_prompt  # noqa: E402
from utils.llm_api import load_api_chat_completion  # noqa: E402

VALID = {"IND", "EQV", "NSO", "OSN", "CON"}


def parse_relation(raw: str | None) -> str:
    if not raw:
        return "PARSE_FAIL"
    try:
        rel = str(json.loads(raw).get("relation", "")).strip().upper()
        return rel if rel in VALID else "PARSE_FAIL"
    except Exception:
        m = re.search(r"\b(IND|EQV|NSO|OSN|CON)\b", (raw or "").upper())
        return m.group(1) if m else "PARSE_FAIL"


def build_sequence(item: dict) -> list[dict]:
    """distractor 按 date 升序在前；golden 放最后，new-value golden 排最末。"""
    dis = sorted(
        (
            {"kind": "distractor", "idx": i, "text": d["text"], "date": d.get("date", "")}
            for i, d in enumerate(item["distractors"])
        ),
        key=lambda x: x["date"],
    )
    nv_idx = item.get("new_value_golden_idx")
    goldens = [
        {"kind": "golden", "idx": i, "text": g["text"], "date": g.get("date", "")}
        for i, g in enumerate(item["golden_memory"])
    ]
    goldens.sort(key=lambda g: g["idx"] == nv_idx)  # new-value 最末
    return dis + goldens


REL_COLORS = {
    "CON": "#d9534f",
    "OSN": "#f0ad4e",
    "NSO": "#5bc0de",
    "EQV": "#5cb85c",
    "IND": "#999999",
    "PARSE_FAIL": "#000000",
}


def esc(s: object) -> str:
    return html_mod.escape(str(s or ""))


def render_html(summary: dict, results: list[dict]) -> str:
    parts = [
        """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>RD 顺序灌库链模拟 — knowledge-update fusion 前后文本</title>
<style>
 body{font-family:-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,"PingFang SC","Microsoft YaHei",sans-serif;margin:24px;background:#f7f7f9;color:#222;}
 h1{font-size:22px;} h2{font-size:16px;margin:0 0 6px;}
 .summary{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px 16px;margin-bottom:20px;}
 .summary td{padding:2px 14px 2px 0;}
 details.q{background:#fff;border:1px solid #ddd;border-radius:8px;margin-bottom:14px;padding:10px 16px;}
 details.q>summary{cursor:pointer;font-weight:600;line-height:1.5;}
 .meta{color:#666;font-size:13px;margin:4px 0 10px;}
 .step{border-left:4px solid #ccc;margin:10px 0;padding:8px 12px;background:#fafafa;border-radius:0 6px 6px 0;}
 .badge{display:inline-block;color:#fff;border-radius:4px;padding:1px 8px;font-size:12px;font-weight:700;margin-right:8px;}
 .lbl{display:inline-block;min-width:120px;color:#888;font-size:12px;vertical-align:top;}
 .txt{display:inline-block;max-width:85%;font-size:13px;line-height:1.5;}
 .fuse{margin-top:6px;border-top:1px dashed #ddd;padding-top:6px;}
 .fuse .out{background:#eef7ee;border:1px solid #cde5cd;border-radius:4px;padding:6px 8px;display:inline-block;max-width:85%;}
 .final{margin-top:10px;background:#eef3fb;border:1px solid #ccd9ef;border-radius:6px;padding:8px 12px;font-size:13px;}
 details.p{margin:4px 0 0;font-size:12px;}
 details.p>summary{cursor:pointer;color:#7a6ea0;}
 details.p pre{white-space:pre-wrap;background:#f4f2fa;border:1px solid #e0dcee;border-radius:4px;padding:8px;font-size:12px;line-height:1.45;max-height:420px;overflow:auto;}
 code{background:#eee;border-radius:3px;padding:0 4px;font-size:12px;}
</style></head><body>
<h1>RD 顺序灌库链模拟 — knowledge-update 的 fusion 前后文本</h1>
<div class="summary"><table>"""
    ]
    rel_dist = summary.get("relation_dist", {})
    n_steps = summary.get("n_steps", 0) or 1
    rel_s = "　".join(
        f'<span class="badge" style="background:{REL_COLORS.get(k, "#333")}">{k}</span>{v} ({v / n_steps:.1%})'
        for k, v in rel_dist.items()
    )
    parts.append(
        f"<tr><td>模型</td><td>{esc(summary['model'])}</td></tr>"
        f"<tr><td>题数</td><td>{summary['n_questions']}</td></tr>"
        f"<tr><td>链上分类步数</td><td>{summary['n_steps']}</td></tr>"
        f"<tr><td>关系分布</td><td>{rel_s}</td></tr>"
        f"<tr><td>触发 fusion 次数</td><td>{summary['n_fusions']}</td></tr>"
        f"<tr><td>平均每题最终答题记忆 C 条数</td><td>{summary['avg_final_answer_memories']}</td></tr>"
        "</table><div class='meta'>顺序：distractor 按 date 升序在前，golden 放最后（new-value golden 排最末）。"
        "分类始终用原子原文（贴近真实 RD）；非 IND 时滚动融合答题记忆 C = fuse(当前C, 新事实)。</div>"
    )
    if summary.get("classify_system_prompt"):
        parts.append(
            '<details class="p"><summary>关系分类 system prompt（v3，所有步骤共用）</summary>'
            f'<pre>{esc(summary["classify_system_prompt"])}</pre></details>'
        )
    parts.append("</div>")
    for r in results:
        n_fuse = sum(1 for s in r["steps"] if s.get("fuse_output"))
        parts.append(
            f'<details class="q"><summary>[{esc(r.get("question_type", ""))}] {esc(r["question_id"])} — {esc(r["question"])} '
            f'<code>answer: {esc(r["answer"])}</code>（fusion×{n_fuse}）</summary>'
            f'<div class="meta">灌库顺序：{esc(" → ".join(r["sequence"]))}</div>'
        )
        for s in r["steps"]:
            color = REL_COLORS.get(s["relation"], "#333")
            parts.append(
                f'<div class="step" style="border-left-color:{color}">'
                f'<span class="badge" style="background:{color}">{esc(s["relation"])}</span>'
                f'step {s["step"]}: <code>{esc(s["old"])}</code> (OLD) vs <code>{esc(s["new"])}</code> (NEW)<br>'
                f'<span class="lbl">OLD 原子</span><span class="txt">{esc(s["m_old"])}</span><br>'
                f'<span class="lbl">NEW 原子</span><span class="txt">{esc(s["m_new"])}</span>'
            )
            if s.get("classify_user_prompt"):
                parts.append(
                    '<details class="p"><summary>分类 user prompt</summary>'
                    f'<pre>{esc(s["classify_user_prompt"])}</pre></details>'
                )
            if "fuse_input_current_memory" in s:
                parts.append(
                    '<div class="fuse">'
                    f'<span class="lbl">fuse 输入（当前C）</span><span class="txt">{esc(s["fuse_input_current_memory"])}</span><br>'
                    f'<span class="lbl">fuse 输出（新C）</span><span class="txt out">{esc(s.get("fuse_output") or "（融合失败/空）")}</span>'
                )
                if s.get("fuse_prompt"):
                    parts.append(
                        '<details class="p"><summary>fuse 完整 prompt</summary>'
                        f'<pre>{esc(s["fuse_prompt"])}</pre></details>'
                    )
                parts.append("</div>")
            parts.append("</div>")
        finals = r["final_answer_memories"]
        if finals:
            items = "".join(f"<li>{esc(t)}</li>" for t in finals)
            parts.append(f'<div class="final"><b>最终答题记忆 C（{len(finals)} 条）：</b><ul>{items}</ul></div>')
        else:
            parts.append('<div class="final">全链无 fusion（全部 IND），无答题记忆 C。</div>')
        parts.append("</details>")
    parts.append("</body></html>")
    return "".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/preprocessed/longmemeval_s_hybrid_golden.json")
    ap.add_argument("--model", default="gemma4-26B")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0=全部）")
    ap.add_argument("--types", default="knowledge-update", help="逗号分隔的 question_type 列表")
    ap.add_argument("--sample-per-type", type=int, default=0, help="每题型随机抽 N 题（seed=42，0=全部）")
    ap.add_argument(
        "--con-fuse-template",
        default="",
        help="CON 融合改用指定模板（如 lme_answer_fuse_con_en_compact.jinja）；其他关系仍用默认",
    )
    ap.add_argument("--out", default="logs/analysis/ku_rd_chain_gemma4.json")
    args = ap.parse_args()

    data = json.loads(Path(args.data).read_text())
    types = [t.strip() for t in args.types.split(",") if t.strip()]
    ku = []
    rng = random.Random(42)
    for t in types:
        pool = [d for d in data if d.get("question_type") == t]
        if args.sample_per_type and len(pool) > args.sample_per_type:
            pool = rng.sample(pool, args.sample_per_type)
        ku.extend(pool)
    if args.limit:
        ku = ku[: args.limit]

    client = load_api_chat_completion(args.model, async_=False)

    def classify(m_old: str, m_new: str) -> tuple[str, str | None, str]:
        user_prompt = build_relation_classification_prompt(m_old, m_new, language="en")
        raw = client.get_response_chat(
            [{"role": "user", "content": user_prompt}],
            max_new_tokens=64,
            temperature=0,
            response_format=LME_RELATION_CLASSIFICATION_RESPONSE_FORMAT,
            verbose=False,
        )
        return parse_relation(raw), raw, user_prompt

    def fuse(current_memory: str, m_new: str, relation: str, cur_time: str, new_time: str) -> tuple[str | None, str]:
        if relation == "CON" and args.con_fuse_template:
            prompt = render_prompt(
                args.con_fuse_template,
                current_memory=current_memory,
                new_fact=m_new,
                relation=relation,
                current_memory_time=cur_time,
                new_fact_time=new_time,
            )
        else:
            prompt = build_lme_answer_fuse_prompt(
                current_memory,
                m_new,
                relation,
                language="en",
                current_memory_time=cur_time,
                new_fact_time=new_time,
            )
        raw = client.get_response_chat(
            [{"role": "user", "content": prompt}],
            max_new_tokens=512,
            temperature=0,
            verbose=False,
        )
        return (raw or "").strip() or None, prompt

    def simulate(item: dict) -> dict:
        seq = build_sequence(item)
        # answer_id 语义：原子 → 其所属 C；C 滚动更新
        c_of: dict[int, int] = {}  # 原子序号 -> C 编号
        c_texts: dict[int, str] = {}  # C 编号 -> 当前文本
        c_times: dict[int, str] = {}
        next_c = 0
        steps = []
        for i in range(1, len(seq)):
            old, new = seq[i - 1], seq[i]
            relation, _raw, cls_user_prompt = classify(old["text"], new["text"])
            step = {
                "step": i,
                "old": f"{old['kind']}[{old['idx']}]",
                "new": f"{new['kind']}[{new['idx']}]",
                "m_old": old["text"],
                "m_new": new["text"],
                "relation": relation,
                "classify_user_prompt": cls_user_prompt,
            }
            if relation in ("CON", "OSN", "NSO", "EQV"):
                # anchor = old 原子；其已有 C 则续融，否则以 old 原文起步
                if i - 1 in c_of:
                    cid = c_of[i - 1]
                    current_memory, cur_time = c_texts[cid], c_times[cid]
                else:
                    cid = next_c
                    next_c += 1
                    current_memory, cur_time = old["text"], old["date"]
                fused, fuse_prompt = fuse(current_memory, new["text"], relation, cur_time, new["date"])
                step["fuse_input_current_memory"] = current_memory
                step["fuse_prompt"] = fuse_prompt
                step["fuse_output"] = fused
                if fused:
                    c_texts[cid] = fused
                    c_times[cid] = new["date"]
                    c_of[i - 1] = cid
                    c_of[i] = cid
            steps.append(step)
        return {
            "question_id": item["question_id"],
            "question_type": item["question_type"],
            "question": item["question"],
            "answer": item["answer"],
            "sequence": [f"{s['kind']}[{s['idx']}] @ {s['date']}" for s in seq],
            "steps": steps,
            "final_answer_memories": [c_texts[c] for c in sorted(c_texts)],
        }

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(simulate, item): item["question_id"] for item in ku}
        for n, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            print(f"[{n}/{len(ku)}] {futs[fut]} done")
    results.sort(key=lambda r: r["question_id"])

    rel_counter = Counter(s["relation"] for r in results for s in r["steps"])
    fuse_count = sum(1 for r in results for s in r["steps"] if s.get("fuse_output"))
    summary = {
        "model": args.model,
        "n_questions": len(results),
        "n_steps": sum(len(r["steps"]) for r in results),
        "relation_dist": dict(rel_counter.most_common()),
        "relation_dist_by_type": {
            t: dict(Counter(s["relation"] for r in results if r["question_type"] == t for s in r["steps"]).most_common())
            for t in sorted({r["question_type"] for r in results})
        },
        "n_fusions": fuse_count,
        "avg_final_answer_memories": round(
            sum(len(r["final_answer_memories"]) for r in results) / max(len(results), 1), 2
        ),
        "classify_system_prompt": system,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=1))
    print(f"saved -> {out}")

    html_path = out.with_suffix(".html")
    html_path.write_text(render_html(summary, results))
    print(f"saved -> {html_path}")


if __name__ == "__main__":
    main()
