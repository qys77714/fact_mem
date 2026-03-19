import argparse
import asyncio
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from prompts import render_prompt
from utils.llm_api import load_api_chat_completion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="读取 jsonl 并使用 LLM Judge 评估。")
    parser.add_argument("--input", required=True, help="待评估的 jsonl 文件")
    parser.add_argument("--judge_model", required=True, help="Judge 模型名")
    parser.add_argument("--benchmark", default=None, help="可选：lme/lmb/emb/locomo，不传则自动推断")
    parser.add_argument("--use_cot", action="store_true", help="是否让 Judge 输出简短推理")
    parser.add_argument("--max_concurrency", type=int, default=20)
    parser.add_argument("--append_result", default="experiment/eval_results.txt", help="评测结果汇总文件")
    parser.add_argument("--write_back", action="store_true", help="是否把 is_correct 写回输入 jsonl")
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


def _extract_response_text(resp: Any) -> str:
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        for key in ("response", "text", "content"):
            if key in resp and isinstance(resp[key], str):
                return resp[key]
    try:
        return str(resp)
    except Exception:
        return ""


def _parse_verdict(text: str) -> bool:
    normalized = text.strip().lower()
    m = re.search(r"(final answer|answer)\s*[:\-]\s*(yes|no)", normalized)
    if m:
        return m.group(2) == "yes"
    return normalized.startswith("yes")


def _prompt_lme(item: Dict[str, Any]) -> str:
    q_type = item.get("question_type", "unknown")
    question = item.get("question", "")
    reference = item.get("answer", "")
    candidate = item.get("model_answer", item.get("hypothesis", ""))

    template_map = {
        "knowledge-update": "pipeline_eval_lme_knowledge_update.jinja",
        "temporal-reasoning": "pipeline_eval_lme_temporal_reasoning.jinja",
        "single-session-preference": "pipeline_eval_lme_single_session_preference.jinja",
    }
    template = template_map.get(q_type, "pipeline_eval_lme_default.jinja")
    return render_prompt(
        template,
        question=question,
        reference=reference,
        candidate=candidate,
    )


def _prompt_generic(item: Dict[str, Any], use_cot: bool, is_mcq: bool) -> str:
    question = item.get("question", "")
    reference = item.get("answer", "")
    candidate = item.get("model_answer", item.get("hypothesis", ""))

    if is_mcq:
        options = item.get("options", []) or []
        options_block = "\n".join(options) if options else "(no options)"
        golden_option = item.get("golden_option", "")
        return render_prompt(
            "pipeline_eval_generic_mcq.jinja",
            question=question,
            options_block=options_block,
            golden_option=golden_option,
            reference=reference,
            candidate=candidate,
            use_cot=use_cot,
        )
    else:
        return render_prompt(
            "pipeline_eval_generic_non_mcq.jinja",
            question=question,
            reference=reference,
            candidate=candidate,
            use_cot=use_cot,
        )


async def evaluate(
    samples: List[Dict[str, Any]],
    benchmark: str,
    judge_model: str,
    use_cot: bool,
    max_concurrency: int,
) -> Tuple[Dict[str, Any], List[bool]]:
    if not samples:
        return {"overall_accuracy": 0.0, "per_type": {}}, []

    client = load_api_chat_completion(judge_model, async_=True)

    is_mcq = any(sample.get("options") for sample in samples)
    messages_list: List[List[Dict[str, str]]] = []
    meta: List[str] = []

    for item in samples:
        if benchmark == "lme":
            prompt = _prompt_lme(item)
        else:
            prompt = _prompt_generic(item, use_cot=use_cot, is_mcq=is_mcq)

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

    per_type: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    flags: List[bool] = []

    for resp, q_type in zip(responses, meta):
        verdict = _extract_response_text(resp)
        ok = _parse_verdict(verdict)
        flags.append(ok)
        per_type[q_type]["total"] += 1
        if ok:
            per_type[q_type]["correct"] += 1

    total_correct = sum(v["correct"] for v in per_type.values())
    total_count = sum(v["total"] for v in per_type.values())

    metrics = {
        "overall_accuracy": (total_correct / total_count) if total_count else 0.0,
        "per_type": {
            q_type: {
                "accuracy": (v["correct"] / v["total"]) if v["total"] else 0.0,
                "correct": v["correct"],
                "total": v["total"],
            }
            for q_type, v in per_type.items()
        },
    }
    return metrics, flags


def main() -> None:
    args = parse_args()
    samples = load_jsonl(args.input)
    benchmark = infer_benchmark(samples, args.input, args.benchmark)

    metrics, flags = asyncio.run(
        evaluate(
            samples=samples,
            benchmark=benchmark,
            judge_model=args.judge_model,
            use_cot=args.use_cot,
            max_concurrency=args.max_concurrency,
        )
    )

    if args.write_back:
        if len(flags) != len(samples):
            raise ValueError("Judge results size mismatch with sample size.")
        with Path(args.input).open("w", encoding="utf-8") as f:
            for row, ok in zip(samples, flags):
                row["is_correct"] = ok
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    result_file = Path(args.append_result)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with result_file.open("a", encoding="utf-8") as f:
        f.write(f"{args.input}\n")
        f.write(json.dumps(metrics, ensure_ascii=False, indent=2))
        f.write("\n\n")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
