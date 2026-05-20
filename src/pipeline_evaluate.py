import argparse
import asyncio
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from prompts import render_prompt
from utils.eval_report import append_csv_row, append_eval_json, append_jsonl, utc_timestamp_iso
from utils.llm_api import load_api_chat_completion
from utils.thinking_text import split_embedded_thinking
from utils.question_filter import (
    answer_row_key,
    filter_jsonl_rows_by_question_type,
    parse_question_types_arg,
    stratified_sample_by_question_type,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="读取 jsonl 并使用 LLM Judge 评估。")
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="待评估的 jsonl 文件；可传多个，在同一进程内并行评测（便于 Ctrl+C 一次结束）。",
    )
    parser.add_argument("--judge_model", required=True, help="Judge 模型名")
    parser.add_argument(
        "--benchmark",
        default=None,
        help="可选：仅写入结果 metrics 的 benchmark 字段（lme/lmb/emb/locomo），不影响 Judge prompt；不传则从数据或路径推断",
    )
    parser.add_argument("--use_cot", action="store_true", help="是否让 Judge 输出简短推理")
    parser.add_argument(
        "--judge-oqa-template",
        default="pipeline_eval_oqa.jinja",
        metavar="NAME.jinja",
        help="开放问答 Judge 的 user 模板（置于 src/prompts/templates/）",
    )
    parser.add_argument(
        "--judge-mcq-template",
        default="pipeline_eval_mcq.jinja",
        metavar="NAME.jinja",
        help="选择题 Judge 的 user 模板（置于 src/prompts/templates/）",
    )
    parser.add_argument(
        "--judge-system-template",
        default="pipeline_eval_system.jinja",
        metavar="NAME.jinja",
        help="Judge 的 system 模板（置于 src/prompts/templates/）",
    )
    parser.add_argument(
        "--judge-qwen-thinking",
        action="store_true",
        help="Judge 走 vLLM/Qwen3 时，为 ``chat_template_kwargs.enable_thinking`` 开启思考。",
    )
    parser.add_argument("--max_concurrency", type=int, default=20)
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Judge 调用的 max_new_tokens 上限。",
    )
    parser.add_argument(
        "--append_result",
        default=None,
        metavar="PATH",
        help="评测结果汇总文件；默认写入与 --input 同目录的 eval_judge.json（标准 JSON 数组）。"
        "若路径以 .jsonl 结尾则仍为每行一条 JSON。",
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
    parser.add_argument(
        "--question-types",
        default=None,
        metavar="TYPES",
        help="可选：只评估 question_type 在该列表中的行（逗号分隔，与 jsonl 的 question_type 一致）",
    )
    parser.add_argument(
        "--print-one-sample",
        action="store_true",
        help="在终端打印 1 条 Judge 样例（prompt + reasoning_content + content）。",
    )
    parser.add_argument(
        "--stratified-sample-n",
        type=int,
        default=0,
        metavar="N",
        help=(
            "可选：仅对 N 条做 Judge，按 question_type 在**当前输入文件**（经 --question-types 过滤后）"
            "中的比例分层抽样。0 表示不抽样。用于已有完整 pred.jsonl 时节省 Judge 调用。"
        ),
    )
    parser.add_argument(
        "--stratified-sample-seed",
        type=int,
        default=42,
        help="--stratified-sample-n 的随机种子",
    )
    parser.add_argument(
        "--files-concurrency",
        type=int,
        default=None,
        metavar="N",
        help="多文件时同时评测的文件个数上限；默认与 --input 数量相同（全部并行）。",
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
    raw: Optional[str] = None
    if isinstance(resp, str):
        raw = resp.strip()
    elif isinstance(resp, dict):
        for key in ("response", "text", "content"):
            v = resp.get(key)
            if isinstance(v, str):
                t = v.strip()
                if t:
                    raw = t
                    break
    else:
        return None
    if not raw:
        return None
    _, tail = split_embedded_thinking(raw)
    out = tail.strip()
    return out if out else None


def _extract_response_text(resp: Any) -> str:
    """兼容测试与旧逻辑：无正文时返回空串。"""
    t = _judge_response_text(resp)
    return t if t is not None else ""


def _parse_verdict(text: str) -> bool:
    """True iff the judge's final verdict is *yes* (student answer correct)."""
    _, tail = split_embedded_thinking(text.strip())
    core = tail.strip()
    normalized = core.lower()
    m = re.search(r"(final answer|answer)\s*[:\-]\s*(yes|no)\b", normalized)
    if m:
        return m.group(2) == "yes"
    lines = [ln.strip() for ln in normalized.splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        m_last = re.match(r"^(yes|no)([\.\!\s]*)$", last)
        if m_last:
            return m_last.group(1) == "yes"
    return normalized.startswith("yes")


def _build_judge_user_prompt(
    item: Dict[str, Any],
    use_cot: bool,
    *,
    oqa_template: str,
    mcq_template: str,
) -> str:
    question = item.get("question", "")
    reference = item.get("answer", "")
    candidate = item.get("model_answer", item.get("hypothesis", ""))
    question_time = item.get("question_time", "") or ""
    is_mcq = bool(item.get("options"))

    if is_mcq:
        options = item.get("options", []) or []
        options_block = "\n".join(options) if options else "(no options)"
        golden_option = item.get("golden_option", "")
        return render_prompt(
            mcq_template,
            question=question,
            options_block=options_block,
            golden_option=golden_option,
            reference=reference,
            candidate=candidate,
            use_cot=use_cot,
        )
    return render_prompt(
        oqa_template,
        question=question,
        question_time=question_time,
        reference=reference,
        candidate=candidate,
        use_cot=use_cot,
    )


async def evaluate(
    samples: List[Dict[str, Any]],
    judge_model: str,
    use_cot: bool,
    max_concurrency: int,
    max_new_tokens: int,
    judge_qwen_thinking: bool,
    print_one_sample: bool,
    judge_oqa_template: str,
    judge_mcq_template: str,
    judge_system_template: str,
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
        prompt = _build_judge_user_prompt(
            item,
            use_cot=use_cot,
            oqa_template=judge_oqa_template,
            mcq_template=judge_mcq_template,
        )
        messages_list.append(
            [
                {"role": "system", "content": render_prompt(judge_system_template)},
                {"role": "user", "content": prompt},
            ]
        )
        meta.append(str(item.get("question_type", "unknown")))

    if print_one_sample and messages_list:
        sample_result = await client.get_response_chat(
            [messages_list[0]],
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            max_concurrency=1,
            use_tqdm=False,
            verbose=False,
            qwen_enable_thinking=judge_qwen_thinking,
            return_raw_message=True,
        )
        sample_resp = sample_result[0] if sample_result else None
        user_prompt = messages_list[0][1].get("content", "") if len(messages_list[0]) > 1 else ""
        print("\n=== [Judge Sample] prompt ===", flush=True)
        print(user_prompt, flush=True)
        print("=== [Judge Sample] response.reasoning_content ===", flush=True)
        reasoning_text = ""
        content_text = ""
        if isinstance(sample_resp, dict):
            reasoning_text = sample_resp.get("reasoning_content", "") or ""
            content_text = sample_resp.get("content", "") or ""
        elif isinstance(sample_resp, str):
            content_text = sample_resp
        print(reasoning_text, flush=True)
        print("=== [Judge Sample] response.content ===", flush=True)
        print(content_text, flush=True)
        print("=== [Judge Sample End] ===\n", flush=True)

    responses = await client.get_response_chat(
        messages_list,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        max_concurrency=max_concurrency,
        use_tqdm=True,
        verbose=True,
        qwen_enable_thinking=judge_qwen_thinking,
        return_raw_message=print_one_sample,
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


async def evaluate_one_input(
    input_path: str,
    args: argparse.Namespace,
    *,
    print_one_sample: bool,
) -> None:
    samples_all = load_jsonl(input_path)
    q_types = parse_question_types_arg(args.question_types)
    samples = filter_jsonl_rows_by_question_type(samples_all, q_types)
    if int(getattr(args, "stratified_sample_n", 0) or 0) > 0:
        keyed = [(answer_row_key(r), r.get("question_type")) for r in samples]
        keep = stratified_sample_by_question_type(
            keyed,
            int(args.stratified_sample_n),
            int(getattr(args, "stratified_sample_seed", 42)),
        )
        samples = [r for r in samples if answer_row_key(r) in keep]
    benchmark = infer_benchmark(samples, input_path, args.benchmark)

    metrics, outcomes = await evaluate(
        samples=samples,
        judge_model=args.judge_model,
        use_cot=args.use_cot,
        max_concurrency=args.max_concurrency,
        max_new_tokens=args.max_new_tokens,
        judge_qwen_thinking=args.judge_qwen_thinking,
        print_one_sample=print_one_sample,
        judge_oqa_template=args.judge_oqa_template,
        judge_mcq_template=args.judge_mcq_template,
        judge_system_template=args.judge_system_template,
    )
    metrics["benchmark"] = benchmark

    if args.write_back:
        if len(outcomes) != len(samples):
            raise ValueError("Judge results size mismatch with sample size.")
        merge_keys = q_types is not None or int(getattr(args, "stratified_sample_n", 0) or 0) > 0
        if not merge_keys:
            rows_out = list(zip(samples, outcomes))
        else:
            by_key = {answer_row_key(r): o for r, o in zip(samples, outcomes)}
            rows_out = []
            for row in samples_all:
                key = answer_row_key(row)
                o = by_key.get(key)
                rows_out.append((row, o))
        with Path(input_path).open("w", encoding="utf-8") as f:
            for row, o in rows_out:
                if o is not None:
                    row["judge_api_failed"] = o["api_failed"]
                    row["is_correct"] = o["is_correct"]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    resolved_input = str(Path(input_path).resolve())
    n = len(samples)
    ts = utc_timestamp_iso()
    record = {
        "timestamp": ts,
        "eval_type": "judge",
        "input_path": resolved_input,
        "judge_model": args.judge_model,
        "use_cot": args.use_cot,
        "judge_oqa_template": args.judge_oqa_template,
        "judge_mcq_template": args.judge_mcq_template,
        "judge_system_template": args.judge_system_template,
        "judge_qwen_thinking": args.judge_qwen_thinking,
        "max_concurrency": args.max_concurrency,
        "max_new_tokens": args.max_new_tokens,
        "n": n,
        "benchmark": benchmark,
        "question_types_filter": sorted(q_types) if q_types else None,
        "stratified_sample_n": int(getattr(args, "stratified_sample_n", 0) or 0),
        "stratified_sample_seed": int(getattr(args, "stratified_sample_seed", 42)),
        "overall_accuracy": metrics["overall_accuracy"],
        "api_failure_count": metrics["api_failure_count"],
        "judged_count": metrics["judged_count"],
        "per_type": metrics["per_type"],
    }
    append_path = (
        Path(args.append_result)
        if args.append_result
        else Path(input_path).resolve().parent / "eval_judge.json"
    )
    if append_path.suffix.lower() == ".jsonl":
        append_jsonl(append_path, record)
    else:
        append_eval_json(append_path, record)

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
                "judge_oqa_template": args.judge_oqa_template,
                "judge_mcq_template": args.judge_mcq_template,
                "judge_system_template": args.judge_system_template,
                "judge_qwen_thinking": args.judge_qwen_thinking,
                "max_concurrency": args.max_concurrency,
                "max_new_tokens": args.max_new_tokens,
                "overall_accuracy": metrics["overall_accuracy"],
                "api_failure_count": metrics["api_failure_count"],
                "judged_count": metrics["judged_count"],
                "mean_f1": "",
                "mean_exact_match": "",
                "token_mode": "",
                "per_type_json": metrics["per_type"],
            },
        )


async def async_main(args: argparse.Namespace) -> None:
    inputs = list(args.input)
    fc = args.files_concurrency
    if fc is None:
        fc = len(inputs)
    fc = max(1, min(fc, len(inputs)))
    sem = asyncio.Semaphore(fc)

    async def run_one(idx: int, path: str) -> None:
        async with sem:
            await evaluate_one_input(
                path,
                args,
                print_one_sample=bool(args.print_one_sample and idx == 0),
            )

    await asyncio.gather(*(run_one(i, p) for i, p in enumerate(inputs)))


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
