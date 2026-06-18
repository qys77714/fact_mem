"""Step 5: 用 verify prompt 对训练数据做二次质检。

对每条训练样本，用 gemma4-26B verify prompt 判断标签是否正确。
输出增加 verify_correct 字段，可选过滤掉 verify_correct=false 的样本。
"""
import json
import os
import sys
import time
import yaml
import asyncio
from typing import List

HERE = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_repo_root, "src"))


def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_verify_messages(old: str, new: str, label: str, language: str = "en") -> List[dict]:
    """构建 verify prompt messages。按 label 选专属模板，IND 回退通用模板。"""
    from prompts import render_prompt

    lang = (language or "en").strip().lower()
    suffix = "zh" if lang.startswith("zh") else "en"
    rel = (label or "").strip().upper()

    # 按标签选专属 verify system prompt（与 prompts.py 的 lme_relation_verify_system_prompt_for_language 一致）
    per_label = {
        "EQV": f"lme_relation_verify_system_eqv_{suffix}.jinja",
        "NSO": f"lme_relation_verify_system_nso_{suffix}.jinja",
        "OSN": f"lme_relation_verify_system_osn_{suffix}.jinja",
        "CON": f"lme_relation_verify_system_con_{suffix}.jinja",
    }
    template = per_label.get(rel, f"lme_relation_verify_system_{suffix}.jinja")
    system_prompt = render_prompt(template)

    user_prompt = render_prompt(
        "lme_relation_verify_user.jinja",
        m_old=old, m_new=new, relation=rel,
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def verify_training_data(
    training_path: str,
    output_path: str,
    model_name: str = "gemma4-26B",
    max_concurrency: int = 10,
    filter_rejected: bool = False,
):
    """对训练数据逐条用 gemma4-26B verify prompt 复核标签。

    Args:
        training_path: 训练数据 JSONL
        output_path: 输出 JSONL（增加 verify_correct 字段）
        model_name: gemma 模型名
        max_concurrency: 异步并发数
        filter_rejected: True=仅保留 verify_correct=true 的样本
    """
    samples = load_jsonl(training_path)
    print(f"加载 {len(samples)} 条训练样本")

    from utils.llm_api import load_api_chat_completion

    client = load_api_chat_completion(model_name, async_=True)

    # 构建 messages（每条样本按自己的 label 选专属 verify 模板）
    messages_list = []
    for s in samples:
        msgs = build_verify_messages(s["old"], s["new"], s["label"])
        messages_list.append(msgs)

    print(f"发送 {len(messages_list)} 条 verify 请求到 {model_name}...")
    t0 = time.time()

    responses = asyncio.run(
        client.get_response_chat(
            messages_list,
            max_new_tokens=64,
            temperature=0.0,
            max_concurrency=max_concurrency,
            use_tqdm=True,
        )
    )

    elapsed = time.time() - t0
    print(f"verify 完成: {len(responses)} 条, 耗时 {elapsed:.1f}s")

    # 解析结果
    verified = 0
    rejected = 0
    parse_error = 0
    for s, resp in zip(samples, responses):
        correct = None
        if resp:
            try:
                resp_clean = resp.strip()
                if resp_clean.startswith("```"):
                    resp_clean = resp_clean.split("\n", 1)[-1]
                    resp_clean = resp_clean.rsplit("```", 1)[0]
                obj = json.loads(resp_clean)
                correct = obj.get("correct", None)
            except (json.JSONDecodeError, AttributeError):
                import re as _re
                m = _re.search(r'"correct"\s*:\s*(true|false)', resp, _re.IGNORECASE)
                if m:
                    correct = m.group(1).lower() == "true"

        if correct is True:
            verified += 1
            s["verify_correct"] = True
        elif correct is False:
            rejected += 1
            s["verify_correct"] = False
        else:
            parse_error += 1
            s["verify_correct"] = None

    print(f"verify 结果: 通过 {verified}, 拒绝 {rejected}, 解析失败 {parse_error}")

    # 可选过滤
    output_samples = samples
    if filter_rejected:
        output_samples = [s for s in samples if s.get("verify_correct") is True]
        print(f"过滤后保留: {len(output_samples)} 条")

    # 统计各标签通过率
    from collections import Counter
    label_pass = Counter()
    label_total = Counter()
    for s in samples:
        label_total[s["label"]] += 1
        if s.get("verify_correct") is True:
            label_pass[s["label"]] += 1
    print("各标签通过率:")
    for lbl in ["IND", "EQV", "OSN", "NSO", "CON"]:
        t = label_total.get(lbl, 0)
        p = label_pass.get(lbl, 0)
        rate = f"{p/t:.1%}" if t > 0 else "N/A"
        print(f"  {lbl}: {p}/{t} ({rate})")

    # 写入
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in output_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"输出 → {output_path} ({len(output_samples)} 条)")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Step 5: Verify training data labels with gemma4-26B")
    ap.add_argument("--training", required=True, help="训练数据 JSONL")
    ap.add_argument("--output", required=True, help="输出 JSONL")
    ap.add_argument("--model", default="gemma4-26B", help="LLM 模型名")
    ap.add_argument("--concurrency", type=int, default=10, help="异步并发数")
    ap.add_argument("--filter", action="store_true", help="仅保留 verify_correct=true 的样本")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 条（调试）")
    args = ap.parse_args()

    if args.limit > 0:
        samples = load_jsonl(args.training)[:args.limit]
        tmp_path = args.training + ".tmp_limit"
        with open(tmp_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        args.training = tmp_path

    verify_training_data(
        args.training, args.output,
        model_name=args.model,
        max_concurrency=args.concurrency,
        filter_rejected=args.filter,
    )

    if args.limit > 0:
        os.remove(tmp_path)


if __name__ == "__main__":
    main()
