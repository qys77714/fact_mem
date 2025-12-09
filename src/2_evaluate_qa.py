import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List
import re
from generate.load_api_llm import load_api_chat_completion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate QA results.")
    parser.add_argument("--input_path", required=True, help="输入 JSONL 文件路径")
    parser.add_argument(
        "--evaluate_task",
        required=True,
        choices=("lme", "lmb", "emb", "locomo"),
        help="评估任务类型",
    )
    parser.add_argument("--judge_model", required=True, help="评估所用模型名称")
    parser.add_argument("--use_cot", action="store_true", help="是否在评估时使用链式思维提示")
    return parser.parse_args()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"未找到文件：{path}")
    records: List[Dict[str, Any]] = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def evaluate_lme(*_args, **_kwargs):
    pass


def evaluate_locomo(*_args, **_kwargs):
    pass


def evaluate_lmb(
        samples: List[Dict[str, Any]], 
        judge_model: str, 
        use_cot: bool = False,
        is_mcq: bool = False,
) -> Dict[str, Any]:
    if not samples:
        return {"overall_accuracy": 0.0, "per_type": {}}, []

    async def _run() -> Dict[str, Any]:
        client = load_api_chat_completion(judge_model, async_=True)
        messages_list: List[List[Dict[str, str]]] = []
        meta: List[Dict[str, Any]] = []

        for item in samples:
            question = item.get("question", "")
            reference = item.get("answer", "")
            candidate = item.get("model_answer")
            question_type = item.get("question_type", "unknown")
            options = item.get("options", [])
            if is_mcq:
                user_prompt = _build_eval_prompt_mcq(
                    question=question,
                    options=options,
                    golden_option=item.get("golden_option"),
                    reference=reference,
                    candidate=candidate,
                    use_cot=use_cot,
                )
            else:
                user_prompt = _build_eval_prompt(
                    question=question,
                    reference=reference,
                    candidate=candidate,
                    use_cot=use_cot,
                )
            messages_list.append(
                [
                    {"role": "system", "content": "You are a careful evaluation assistant."},
                    {"role": "user", "content": user_prompt},
                ]
            )
            meta.append({"question_type": question_type})

        responses = await client.get_response_chat(
            messages_list,
            max_new_tokens=2048,
            temperature=0.0,
            max_concurrency=20,
            use_tqdm=True,
            verbose=True,
        )

        per_type: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
        is_correct_flags: List[bool] = []

        for resp, info in zip(responses, meta):
            verdict = _extract_response_text(resp)
            is_correct = _parse_verdict(verdict)
            is_correct_flags.append(is_correct)
            q_type = info["question_type"]
            per_type[q_type]["total"] += 1
            if is_correct:
                per_type[q_type]["correct"] += 1

        total_correct = sum(stat["correct"] for stat in per_type.values())
        total_count = sum(stat["total"] for stat in per_type.values())
        overall_accuracy = total_correct / total_count if total_count else 0.0

        per_type_accuracy = {
            q_type: {
                "accuracy": stat["correct"] / stat["total"] if stat["total"] else 0.0,
                "total": stat["total"],
                "correct": stat["correct"],
            }
            for q_type, stat in per_type.items()
        }
        return {"overall_accuracy": overall_accuracy, "per_type": per_type_accuracy}, is_correct_flags

    return asyncio.run(_run())

def _build_eval_prompt(question: str, reference: str, candidate: str, use_cot: bool) -> str:
    prompt = (
        "You are given a question, its ground-truth answer, and a model response. "
        "Determine if the model response is semantically equivalent or meaningfully similar to the ground-truth answer. "
        "Consider the following as acceptable variations:\n"
        "- Different wording but same core meaning\n"
        "- Partial answers that contain the key information\n"
        "- Answers with additional relevant context\n"
        "- Answers that rephrase the same idea\n"
        "- Minor factual details may differ if the main point is correct\n\n"
        "Be lenient in your judgment - if the response captures the essence of the correct answer, consider it correct.\n\n"
        f"Question: {question}\n"
        f"Ground-truth answer: {reference}\n"
        f"Model response: {candidate}\n\n"
    )
    if use_cot:
        prompt += (
            "Think step by step about whether the model response matches the ground-truth answer. "
            "Provide a brief reasoning and then end with 'Final answer: yes' or 'Final answer: no'."
        )
    else:
        prompt += "Answer yes or no only."
    return prompt

def _build_eval_prompt_mcq(
    question: str,
    options: List[Any],
    golden_option: Any,
    reference: str,
    candidate: str,
    use_cot: bool,
) -> str:
    options_block = "\n".join(options) if options else "（无可用选项）"
    ground_truth = f"{golden_option}. {reference}"

    prompt = (
        "You are given a multiple-choice question with options, the correct option, "
        "and a model response. Determine if the model response selects the correct option.\n\n"
        f"Question: {question}\n"
        f"Options:\n{options_block}\n"
        f"Ground-truth answer: {ground_truth}\n"
        f"Model response: {candidate}\n\n"
    )
    if use_cot:
        prompt += (
            "Reason briefly about whether the model picked the correct choice, then end with "
            "'Final answer: yes' or 'Final answer: no'."
        )
    else:
        prompt += "Answer yes or no only."
    return prompt

def _parse_verdict(text: str) -> bool:
    normalized = text.strip().lower()
    match = re.search(r"(final answer|answer)\s*[:\-]\s*(yes|no)", normalized)
    if match:
        return match.group(2) == "yes"
    else:
        print(f"无法解析的评估响应：{text}")
        return normalized.startswith("yes")


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


def main() -> None:
    args = parse_args()
    samples = load_jsonl(args.input_path)
    is_mcq_eval = "_mcq" in args.input_path

    per_sample_flags: List[bool] = []
    if args.evaluate_task in ("lmb", "emb"):
        metrics, per_sample_flags = evaluate_lmb(samples, args.judge_model, args.use_cot, is_mcq_eval)
    elif args.evaluate_task == "lme":
        metrics = evaluate_lme(samples, args.judge_model)
    else:
        metrics = evaluate_locomo(samples, args.judge_model)

    if per_sample_flags:
        if len(per_sample_flags) != len(samples):
            raise ValueError("判定结果数量与样本数量不一致。")
        for sample, flag in zip(samples, per_sample_flags):
            sample["is_correct"] = flag
        with open(args.input_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with open("experiment/eval_results.txt", "a", encoding="utf-8") as f:
        f.write(f"{args.input_path}\n")
        f.write(json.dumps(metrics, ensure_ascii=False, indent=2))
        f.write("\n\n")


if __name__ == "__main__":
    main()