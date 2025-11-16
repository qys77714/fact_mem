import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from generate.load_api_llm import load_api_chat_completion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate QA results.")
    parser.add_argument("--input_path", required=True, help="输入 JSONL 文件路径")
    parser.add_argument(
        "--evaluate_task",
        required=True,
        choices=("lme", "lmb", "locomo"),
        help="评估任务类型",
    )
    parser.add_argument("--judge_model", required=True, help="评估所用模型名称")
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


def evaluate_lmb(samples: List[Dict[str, Any]], judge_model: str) -> Dict[str, Any]:
    if not samples:
        return {"overall_accuracy": 0.0, "per_type": {}}

    async def _run() -> Dict[str, Any]:
        client = load_api_chat_completion(judge_model, async_=True)
        messages_list: List[List[Dict[str, str]]] = []
        meta: List[Dict[str, Any]] = []

        for item in samples:
            question = item.get("question", "")
            reference = item.get("answer", "")
            candidate = item.get("model_answer")
            question_type = item.get("question_type", "unknown")
            user_prompt = (
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
                "Answer yes or no only."
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
            max_new_tokens=8,
            temperature=0.0,
            max_concurrency=5,
            use_tqdm=True,
            verbose=False,
        )

        per_type: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
        for resp, info in zip(responses, meta):
            verdict = _extract_response_text(resp)
            normalized = verdict.strip().lower()
            is_correct = normalized.startswith("yes")
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
        return {"overall_accuracy": overall_accuracy, "per_type": per_type_accuracy}

    return asyncio.run(_run())


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

    if args.evaluate_task == "lmb":
        metrics = evaluate_lmb(samples, args.judge_model)
    elif args.evaluate_task == "lme":
        metrics = evaluate_lme(samples, args.judge_model)
    else:
        metrics = evaluate_locomo(samples, args.judge_model)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()