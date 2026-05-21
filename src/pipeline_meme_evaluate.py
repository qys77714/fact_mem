"""
MEME benchmark evaluation — task-specific judge + trivial-pass filter.

Aligned with MEME-public ``code/eval/judge.py`` scoring protocol:
  - Per-task judge prompts (ER/Agg/Tr/Cas/Abs/Del)
  - before-phase questions included in score (except ER-before)
  - Cas/Abs/Del trivial-pass classification via before×after entity match
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from eval.meme_judge import (
    MemeLLMJudge,
    aggregate_meme_metrics,
    classify_trivial_pass,
    task_base,
)
from utils.eval_report import append_eval_json, utc_timestamp_iso
from utils.llm_api import load_api_chat_completion
from utils.question_filter import (
    answer_row_key,
    filter_jsonl_rows_by_question_type,
    parse_question_types_arg,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MEME task-specific judge with trivial-pass filter.")
    p.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="pred JSONL file(s) from pipeline_generate (MEME protocol fields required)",
    )
    p.add_argument("--judge_model", required=True, help="Judge LLM (default MEME paper: gpt-4o)")
    p.add_argument(
        "--benchmark",
        default=None,
        help="Optional benchmark label written to eval report",
    )
    p.add_argument("--max_concurrency", type=int, default=20)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument(
        "--append_result",
        default=None,
        help="Eval summary JSON path; default <input_dir>/eval_meme_judge.json",
    )
    p.add_argument(
        "--write_back",
        action="store_true",
        help="Write u_pass / pass_type / u_reason back into input JSONL",
    )
    p.add_argument(
        "--question-types",
        default=None,
        help="Optional comma-separated question_type filter",
    )
    p.add_argument(
        "--files-concurrency",
        type=int,
        default=None,
        help="Max parallel input files (default: all)",
    )
    return p.parse_args()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _gold_for_row(row: Dict[str, Any]) -> Any:
    ev = row.get("entity_values") or {}
    tb = task_base(str(row.get("question_type", "")))
    if tb == "Agg" and isinstance(ev, dict) and ev:
        return ev
    if tb == "Tr" and isinstance(ev, dict) and ev:
        ent = row.get("entity_key") or (next(iter(ev.keys())) if ev else "")
        val = ev.get(ent)
        if isinstance(val, list):
            return val
        if isinstance(val, str) and "," in val:
            return [x.strip() for x in val.split(",")]
        return val
    return row.get("answer", "")


async def judge_rows(
    rows: List[Dict[str, Any]],
    judge: MemeLLMJudge,
    max_concurrency: int,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    if not rows:
        return []

    sem = asyncio.Semaphore(max(1, max_concurrency))

    async def _one(row: Dict[str, Any]) -> Dict[str, Any]:
        async with sem:
            out = dict(row)
            phase = str(row.get("phase", "after"))
            tb = task_base(str(row.get("question_type", "")))
            question = str(row.get("question", ""))
            agent_answer = str(row.get("model_answer", row.get("hypothesis", "")))
            gold = _gold_for_row(row)
            try:
                if tb == "Agg":
                    ev = gold if isinstance(gold, dict) else (row.get("entity_values") or {})
                    result = await judge.u_check_multi(
                        question=question,
                        entity_values=ev,
                        agent_answer=agent_answer,
                        max_new_tokens=max_new_tokens,
                    )
                else:
                    result = await judge.u_check(
                        question=question,
                        gold_value=gold,
                        agent_answer=agent_answer,
                        task_type=str(row.get("question_type", "")),
                        phase=phase,
                        max_new_tokens=max_new_tokens,
                    )
                out["judge_api_failed"] = False
                out["u_pass"] = result.u_pass
                out["u_reason"] = result.u_reason
            except Exception as exc:
                out["judge_api_failed"] = True
                out["u_pass"] = None
                out["u_reason"] = f"judge error: {exc}"
            return out

    judged = await asyncio.gather(*(_one(r) for r in rows))

    # Trivial-pass: build before map per (history_name, entity_key) for Cas/Abs/Del
    before_pass: Dict[Tuple[str, str], bool] = {}
    for row in judged:
        if str(row.get("phase", "")) != "before":
            continue
        tb = task_base(str(row.get("question_type", "")))
        if tb == "ER":
            continue
        ent = str(row.get("entity_key", ""))
        if not ent:
            continue
        before_pass[(str(row["history_name"]), ent)] = bool(row.get("u_pass", False))

    for row in judged:
        if str(row.get("phase", "")) != "after":
            row["pass_type"] = None
            continue
        tb = task_base(str(row.get("question_type", "")))
        ent = str(row.get("entity_key", ""))
        pt = classify_trivial_pass(
            task_type=tb,
            entity_key=ent,
            before_pass_by_entity={
                k[1]: v for k, v in before_pass.items() if k[0] == str(row["history_name"])
            },
            after_u_pass=bool(row.get("u_pass", False)),
        )
        row["pass_type"] = pt

    return judged


async def evaluate_one_input(
    input_path: str,
    args: argparse.Namespace,
) -> None:
    rows_all = load_jsonl(input_path)
    q_types = parse_question_types_arg(args.question_types)
    rows = filter_jsonl_rows_by_question_type(rows_all, q_types)

    client = load_api_chat_completion(args.judge_model, async_=True)
    judge = MemeLLMJudge(client=client, model=client.model_name)

    judged = await judge_rows(
        rows,
        judge=judge,
        max_concurrency=args.max_concurrency,
        max_new_tokens=args.max_new_tokens,
    )

    metrics = aggregate_meme_metrics(judged)
    benchmark = args.benchmark or (rows[0].get("benchmark") if rows else "meme")
    metrics["benchmark"] = benchmark
    metrics["eval_type"] = "meme_judge"
    metrics["judge_model"] = args.judge_model

    if args.write_back:
        judged_by_key = {answer_row_key(r): r for r in judged}
        out_rows = []
        for row in rows_all:
            key = answer_row_key(row)
            if key in judged_by_key:
                merged = dict(row)
                merged.update(
                    {
                        k: judged_by_key[key].get(k)
                        for k in (
                            "judge_api_failed",
                            "u_pass",
                            "u_reason",
                            "pass_type",
                        )
                        if k in judged_by_key[key]
                    }
                )
                out_rows.append(merged)
            else:
                out_rows.append(row)
        with Path(input_path).open("w", encoding="utf-8") as f:
            for row in out_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    ts = utc_timestamp_iso()
    record = {
        "timestamp": ts,
        "eval_type": "meme_judge",
        "input_path": str(Path(input_path).resolve()),
        "judge_model": args.judge_model,
        "max_concurrency": args.max_concurrency,
        "max_new_tokens": args.max_new_tokens,
        "n": len(rows),
        "benchmark": benchmark,
        "question_types_filter": sorted(q_types) if q_types else None,
        **metrics,
    }

    append_path = (
        Path(args.append_result)
        if args.append_result
        else Path(input_path).resolve().parent / "eval_meme_judge.json"
    )
    append_eval_json(append_path, record)

    print(
        f"[meme_judge] {Path(input_path).name}: "
        f"score={metrics['meme_score']:.3f} "
        f"(after {metrics['after_pass']}/{metrics['after_total']}, "
        f"raw_after={metrics.get('after_pass_raw', 0)}/{metrics['after_total']}) "
        f"judge_totals={metrics.get('meme_score_judge_totals', 0):.3f}",
        flush=True,
    )
    for tb in ("Cas", "Abs", "Del"):
        ta = metrics.get("trivial_analysis", {}).get(tb, {})
        if ta.get("total", 0):
            print(
                f"  {tb} after: real={ta.get('real_pass', 0)}/{ta['total']} "
                f"(trivial={ta.get('trivial_pass', 0)} "
                f"kbf={ta.get('knew_but_failed', 0)} nk={ta.get('never_knew', 0)})",
                flush=True,
            )


async def async_main(args: argparse.Namespace) -> None:
    inputs = list(args.input)
    fc = args.files_concurrency or len(inputs)
    fc = max(1, min(fc, len(inputs)))
    sem = asyncio.Semaphore(fc)

    async def run_one(path: str) -> None:
        async with sem:
            await evaluate_one_input(path, args)

    await asyncio.gather(*(run_one(p) for p in inputs))


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
