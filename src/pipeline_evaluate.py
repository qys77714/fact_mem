import argparse
import asyncio
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from prompts import render_prompt
from utils.eval_report import append_csv_row, append_jsonl, utc_timestamp_iso
from utils.llm_api import load_api_chat_completion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="读取 jsonl 并使用 LLM Judge 评估。")
    parser.add_argument("--input", required=True, help="待评估的 jsonl 文件")
    parser.add_argument("--judge_model", required=True, help="Judge 模型名")
    parser.add_argument(
        "--benchmark",
        default=None,
        help="可选：仅写入结果 metrics 的 benchmark 字段（lme/lmb/emb/locomo），不影响 Judge prompt；不传则从数据或路径推断",
    )
    parser.add_argument("--use_cot", action="store_true", help="是否让 Judge 输出简短推理")
    parser.add_argument("--max_concurrency", type=int, default=20)
    parser.add_argument(
        "--append_result",
        default="experiment/eval_judge.jsonl",
        help="评测结果汇总 JSONL（每行一条完整记录）",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="可选：同时向该 CSV 追加一行扁平指标（便于表格对比）",
    )
    parser.add_argument(
        "--write_back",
        action="store_true",
        help="是否写回输入 jsonl：is_correct（API 失败时为 null）与 judge_api_failed",
    )
    return parser.parse_args()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def infer_benchmark(samples: List[Dict[str, Any]], input_path: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit.lower()

    if samples and isinstance(samples[0].get("benchmark"), str):
        bmk = samples[0]["benchmark"].lower()
        if bmk.startswith("lme"):
            return "lme"
        if bmk.startswith("lmb"):
            return "lmb"
        if bmk.startswith("emb"):
            return "emb"
        if bmk.startswith("locomo"):
            return "locomo"

    lower = input_path.lower()
    if "locomo" in lower:
        return "locomo"
    if "lmb" in lower:
        return "lmb"
    if "emb" in lower:
        return "emb"
    return "lme"


class JudgeOutcome(TypedDict):
    """单条 judge 结果：API 无返回/空内容与模型给出 verdict 区分开。"""

    api_failed: bool
    is_correct: Optional[bool]


def _judge_response_text(resp: Any) -> Optional[str]:
    """有可用正文则返回 strip 后的 str；API 失败、None、空串、非 str/dict 结构视为无正文。"""
    if resp is None:
        return None
    if isinstance(resp, str):
        t = resp.strip()
        return t if t else None
    if isinstance(resp, dict):
        for key in ("response", "text", "content"):
            v = resp.get(key)
            if isinstance(v, str):
                t = v.strip()
                if t:
                    return t
        return None
    return None


def _extract_response_text(resp: Any) -> str:
    """兼容测试与旧逻辑：无正文时返回空串。"""
    t = _judge_response_text(resp)
    return t if t is not None else ""


def _parse_verdict(text: str) -> bool:
    normalized = text.strip().lower()
    m = re.search(r"(final answer|answer)\s*[:\-]\s*(yes|no)", normalized)
    if m:
        return m.group(2) == "yes"
    return normalized.startswith("yes")


def _build_judge_user_prompt(item: Dict[str, Any], use_cot: bool) -> str:
    question = item.get("question", "")
    reference = item.get("answer", "")
    candidate = item.get("model_answer", item.get("hypothesis", ""))
    is_mcq = bool(item.get("options"))

    if is_mcq:
        options = item.get("options", []) or []
        options_block = "\n".join(options) if options else "(no options)"
        golden_option = item.get("golden_option", "")
        return render_prompt(
            "pipeline_eval_mcq.jinja",
            question=question,
            options_block=options_block,
            golden_option=golden_option,
            reference=reference,
            candidate=candidate,
            use_cot=use_cot,
        )
    return render_prompt(
        "pipeline_eval_oqa.jinja",
        question=question,
        reference=reference,
        candidate=candidate,
        use_cot=use_cot,
    )


async def evaluate(
    samples: List[Dict[str, Any]],
    judge_model: str,
    use_cot: bool,
    max_concurrency: int,
) -> Tuple[Dict[str, Any], List[JudgeOutcome]]:
    if not samples:
        return {
            "overall_accuracy": 0.0,
            "per_type": {},
            "n_samples": 0,
            "api_failure_count": 0,
            "judged_count": 0,
        }, []

    client = load_api_chat_completion(judge_model, async_=True)

    messages_list: List[List[Dict[str, str]]] = []
    meta: List[str] = []

    for item in samples:
        prompt = _build_judge_user_prompt(item, use_cot=use_cot)
        messages_list.append(
            [
                {"role": "system", "content": render_prompt("pipeline_eval_system.jinja")},
                {"role": "user", "content": prompt},
            ]
        )
        meta.append(str(item.get("question_type", "unknown")))

    responses = await client.get_response_chat(
        messages_list,
        max_new_tokens=2048,
        temperature=0.0,
        max_concurrency=max_concurrency,
        use_tqdm=True,
        verbose=True,
    )

    per_type: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "judged": 0, "api_failed": 0}
    )
    outcomes: List[JudgeOutcome] = []

    for resp, q_type in zip(responses, meta):
        verdict = _judge_response_text(resp)
        if verdict is None:
            outcomes.append(JudgeOutcome(api_failed=True, is_correct=None))
            per_type[q_type]["api_failed"] += 1
            continue
        ok = _parse_verdict(verdict)
        outcomes.append(JudgeOutcome(api_failed=False, is_correct=ok))
        per_type[q_type]["judged"] += 1
        if ok:
            per_type[q_type]["correct"] += 1

    api_failure_count = sum(v["api_failed"] for v in per_type.values())
    judged_count = sum(v["judged"] for v in per_type.values())
    total_correct = sum(v["correct"] for v in per_type.values())

    metrics = {
        "n_samples": len(samples),
        "api_failure_count": api_failure_count,
        "judged_count": judged_count,
        "overall_accuracy": (total_correct / judged_count) if judged_count else 0.0,
        "per_type": {
            q_type: {
                "accuracy": (v["correct"] / v["judged"]) if v["judged"] else 0.0,
                "correct": v["correct"],
                "judged": v["judged"],
                "api_failed": v["api_failed"],
            }
            for q_type, v in per_type.items()
        },
    }
    return metrics, outcomes


def main() -> None:
    args = parse_args()
    samples = load_jsonl(args.input)
    benchmark = infer_benchmark(samples, args.input, args.benchmark)

    metrics, outcomes = asyncio.run(
        evaluate(
            samples=samples,
            judge_model=args.judge_model,
            use_cot=args.use_cot,
            max_concurrency=args.max_concurrency,
        )
    )
    metrics["benchmark"] = benchmark

    if args.write_back:
        if len(outcomes) != len(samples):
            raise ValueError("Judge results size mismatch with sample size.")
        with Path(args.input).open("w", encoding="utf-8") as f:
            for row, o in zip(samples, outcomes):
                row["judge_api_failed"] = o["api_failed"]
                row["is_correct"] = o["is_correct"]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    resolved_input = str(Path(args.input).resolve())
    n = len(samples)
    ts = utc_timestamp_iso()
    record = {
        "timestamp": ts,
        "eval_type": "judge",
        "input_path": resolved_input,
        "judge_model": args.judge_model,
        "use_cot": args.use_cot,
        "max_concurrency": args.max_concurrency,
        "n": n,
        "benchmark": benchmark,
        "overall_accuracy": metrics["overall_accuracy"],
        "api_failure_count": metrics["api_failure_count"],
        "judged_count": metrics["judged_count"],
        "per_type": metrics["per_type"],
    }
    append_jsonl(Path(args.append_result), record)

    if args.csv:
        append_csv_row(
            Path(args.csv),
            {
                "timestamp": ts,
                "eval_type": "judge",
                "input_path": resolved_input,
                "benchmark": benchmark,
                "n": n,
                "judge_model": args.judge_model,
                "use_cot": args.use_cot,
                "max_concurrency": args.max_concurrency,
                "overall_accuracy": metrics["overall_accuracy"],
                "api_failure_count": metrics["api_failure_count"],
                "judged_count": metrics["judged_count"],
                "mean_f1": "",
                "mean_exact_match": "",
                "token_mode": "",
                "per_type_json": metrics["per_type"],
            },
        )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
