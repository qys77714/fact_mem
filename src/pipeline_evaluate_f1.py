"""读取 jsonl，用 token-level F1 / EM 评估（无需 LLM Judge）。"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.eval_report import append_csv_row, append_jsonl, utc_timestamp_iso
from utils.qa_metrics import TokenMode, compute_f1_em


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="读取 jsonl 并计算 F1 / EM（token 级）。")
    parser.add_argument("--input", required=True, help="待评估的 jsonl 文件")
    parser.add_argument(
        "--benchmark",
        default=None,
        help="可选：仅写入 metrics 的 benchmark 字段；不传则从数据或路径推断",
    )
    parser.add_argument(
        "--token_mode",
        choices=("whitespace", "char"),
        default="whitespace",
        help="分词方式：whitespace=SQuAD 风格空白分词；char=去空格后按字符（适合部分中文场景）",
    )
    parser.add_argument(
        "--append_result",
        default="experiment/eval_f1.jsonl",
        help="结果汇总 JSONL（每行一条完整记录）",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="可选：同时向该 CSV 追加一行扁平指标（便于表格对比）",
    )
    parser.add_argument("--write_back", action="store_true", help="是否把 f1、exact_match 写回 jsonl")
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


def evaluate_f1(
    samples: List[Dict[str, Any]],
    token_mode: TokenMode,
) -> Tuple[Dict[str, Any], List[Tuple[float, bool]]]:
    if not samples:
        return {
            "mean_f1": 0.0,
            "mean_exact_match": 0.0,
            "per_type": {},
            "token_mode": token_mode,
        }, []

    per_type: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"f1_sum": 0.0, "em_sum": 0.0, "total": 0}
    )
    scores: List[Tuple[float, bool]] = []

    for item in samples:
        gold = str(item.get("answer", ""))
        pred = str(item.get("model_answer", item.get("hypothesis", "")))
        f1, em = compute_f1_em(pred, gold, token_mode)
        scores.append((f1, em))
        q_type = str(item.get("question_type", "unknown"))
        per_type[q_type]["f1_sum"] += f1
        per_type[q_type]["em_sum"] += 1.0 if em else 0.0
        per_type[q_type]["total"] += 1

    n = len(samples)
    mean_f1 = sum(s[0] for s in scores) / n
    mean_em = sum(1.0 if s[1] else 0.0 for s in scores) / n

    metrics = {
        "mean_f1": mean_f1,
        "mean_exact_match": mean_em,
        "token_mode": token_mode,
        "n": n,
        "per_type": {
            qt: {
                "mean_f1": (v["f1_sum"] / v["total"]) if v["total"] else 0.0,
                "mean_exact_match": (v["em_sum"] / v["total"]) if v["total"] else 0.0,
                "total": int(v["total"]),
            }
            for qt, v in per_type.items()
        },
    }
    return metrics, scores


def main() -> None:
    args = parse_args()
    samples = load_jsonl(args.input)
    benchmark = infer_benchmark(samples, args.input, args.benchmark)
    token_mode: TokenMode = args.token_mode

    metrics, scores = evaluate_f1(samples, token_mode)
    metrics["benchmark"] = benchmark

    if args.write_back:
        if len(scores) != len(samples):
            raise ValueError("Scores size mismatch with sample size.")
        with Path(args.input).open("w", encoding="utf-8") as f:
            for row, (f1, em) in zip(samples, scores):
                row["f1"] = f1
                row["exact_match"] = em
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    resolved_input = str(Path(args.input).resolve())
    ts = utc_timestamp_iso()
    record = {
        "timestamp": ts,
        "eval_type": "f1",
        "input_path": resolved_input,
        "token_mode": token_mode,
        "n": metrics["n"],
        "benchmark": benchmark,
        "mean_f1": metrics["mean_f1"],
        "mean_exact_match": metrics["mean_exact_match"],
        "per_type": metrics["per_type"],
    }
    append_jsonl(Path(args.append_result), record)

    if args.csv:
        append_csv_row(
            Path(args.csv),
            {
                "timestamp": ts,
                "eval_type": "f1",
                "input_path": resolved_input,
                "benchmark": benchmark,
                "n": metrics["n"],
                "judge_model": "",
                "use_cot": "",
                "max_concurrency": "",
                "overall_accuracy": "",
                "api_failure_count": "",
                "judged_count": "",
                "mean_f1": metrics["mean_f1"],
                "mean_exact_match": metrics["mean_exact_match"],
                "token_mode": token_mode,
                "per_type_json": metrics["per_type"],
            },
        )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
