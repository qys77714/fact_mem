"""Step 6: 用改进的 v3 分类 prompt 重新判断被 verify 拒绝的样本，再验证，合并入纯净集。"""
import json
import os
import sys
import time
import asyncio
from typing import List
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_repo_root, "src"))


def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def classify_with_v3(pairs: List[dict], model_name="gemma4-26B", max_concurrency=20):
    """用 v3 分类 prompt 重新判断。"""
    from utils.llm_api import load_api_chat_completion
    from prompts import render_prompt

    system_prompt = render_prompt("lme_relation_classification_system_en_v3.jinja")
    messages_list = []
    for p in pairs:
        user_prompt = render_prompt(
            "lme_relation_classification_user.jinja",
            m_old=p["old"], m_new=p["new"],
        )
        messages_list.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

    client = load_api_chat_completion(model_name, async_=True)
    print(f"v3 分类: {len(messages_list)} 条请求 → {model_name}...")
    t0 = time.time()
    responses = asyncio.run(
        client.get_response_chat(messages_list, max_new_tokens=128, temperature=0.0,
                                 max_concurrency=max_concurrency, use_tqdm=True)
    )
    print(f"完成: {time.time()-t0:.1f}s")

    valid_labels = {"IND", "EQV", "OSN", "NSO", "CON"}
    for p, resp in zip(pairs, responses):
        label = None
        if resp:
            try:
                resp_clean = resp.strip()
                if resp_clean.startswith("```"):
                    resp_clean = resp_clean.split("\n", 1)[-1].rsplit("```", 1)[0]
                obj = json.loads(resp_clean)
                if isinstance(obj, dict):
                    label = obj.get("relation", "").strip().upper()
            except (json.JSONDecodeError, AttributeError):
                import re as _re
                for lbl in valid_labels:
                    if _re.search(r'\b' + _re.escape(lbl) + r'\b', resp):
                        label = lbl
                        break
        p["v3_label"] = label if label in valid_labels else "PARSE_ERROR"

    return pairs


def verify_pairs(pairs: List[dict], model_name="gemma4-26B", max_concurrency=20):
    """用 verify prompt 验证 v3 标签。"""
    from utils.llm_api import load_api_chat_completion
    from prompts import render_prompt

    suffix = "en"
    per_label = {
        "EQV": f"lme_relation_verify_system_eqv_{suffix}.jinja",
        "NSO": f"lme_relation_verify_system_nso_{suffix}.jinja",
        "OSN": f"lme_relation_verify_system_osn_{suffix}.jinja",
        "CON": f"lme_relation_verify_system_con_{suffix}.jinja",
    }

    # 按 label 分组，同一 label 共享 system prompt
    by_label = {}
    for p in pairs:
        lbl = p.get("v3_label", "IND")
        by_label.setdefault(lbl, []).append(p)

    client = load_api_chat_completion(model_name, async_=True)

    for lbl, group in by_label.items():
        template = per_label.get(lbl, f"lme_relation_verify_system_{suffix}.jinja")
        system_prompt = render_prompt(template)
        messages_list = []
        for p in group:
            user_prompt = render_prompt(
                "lme_relation_verify_user.jinja",
                m_old=p["old"], m_new=p["new"], relation=lbl,
            )
            messages_list.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])

        print(f"  verify [{lbl}]: {len(messages_list)} 条...")
        t0 = time.time()
        responses = asyncio.run(
            client.get_response_chat(messages_list, max_new_tokens=64, temperature=0.0,
                                     max_concurrency=max_concurrency, use_tqdm=True)
        )
        print(f"    完成: {time.time()-t0:.1f}s")

        for p, resp in zip(group, responses):
            correct = None
            if resp:
                try:
                    resp_clean = resp.strip()
                    if resp_clean.startswith("```"):
                        resp_clean = resp_clean.split("\n", 1)[-1].rsplit("```", 1)[0]
                    obj = json.loads(resp_clean)
                    correct = obj.get("correct", None)
                except (json.JSONDecodeError, AttributeError):
                    import re as _re
                    m = _re.search(r'"correct"\s*:\s*(true|false)', resp, _re.IGNORECASE)
                    if m:
                        correct = m.group(1).lower() == "true"
            p["rejudge_verify"] = correct

    return pairs


def rejudge_and_merge(
    verified_path: str,
    clean_path: str,
    output_path: str,
    model_name="gemma4-26B",
    max_concurrency=20,
):
    """主流程：加载被拒样本 → v3 重分类 → verify → 合并入纯净集。"""
    verified = load_jsonl(verified_path)
    rejected = [s for s in verified if s.get("verify_correct") is False]
    print(f"被拒样本: {len(rejected)} 条")

    # v3 重分类
    rejected = classify_with_v3(rejected, model_name, max_concurrency)

    # 检查 v3 标签变更
    label_changes = Counter()
    for s in rejected:
        old_lbl = s.get("label", "")
        new_lbl = s.get("v3_label", "")
        if old_lbl != new_lbl:
            label_changes[(old_lbl, new_lbl)] += 1
    print(f"v3 标签变更: {sum(label_changes.values())} 条")
    for (old, new), n in label_changes.most_common(20):
        print(f"  {old} → {new}: {n}")

    # verify v3 标签
    rejected = verify_pairs(rejected, model_name, max_concurrency)

    # 筛选通过 verify 的
    passed = [s for s in rejected if s.get("rejudge_verify") is True]
    failed = [s for s in rejected if s.get("rejudge_verify") is False]
    print(f"\n重判结果: 通过 {len(passed)}, 未通过 {len(failed)}")

    # 通过标签统计
    pass_labels = Counter(s.get("v3_label") for s in passed)
    print(f"通过标签分布: {dict(pass_labels)}")

    # 合并：纯净集 + 重判通过的
    clean = load_jsonl(clean_path)
    # 重判通过的样本：用 v3_label 作为最终 label
    for s in passed:
        s["label"] = s["v3_label"]
        s["rejudge_passed"] = True

    merged = clean + passed

    # 去重
    seen = set()
    deduped = []
    for s in merged:
        key = (s["old"].strip().lower(), s["new"].strip().lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)

    import random
    random.seed(42)
    random.shuffle(deduped)

    label_counts = Counter(s["label"] for s in deduped)
    total = len(deduped)
    print(f"\n最终训练集: {total} 条")
    print(f"新增 (重判通过): {len(passed)} 条")
    for lbl in ["IND", "EQV", "OSN", "NSO", "CON"]:
        n = label_counts.get(lbl, 0)
        print(f"  {lbl}: {n:>5} ({n/total:.1%})")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in deduped:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"输出 → {output_path}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Step 6: Re-judge rejected samples with v3 prompt")
    ap.add_argument("--verified", required=True, help="含 verify_correct 字段的训练数据")
    ap.add_argument("--clean", required=True, help="当前纯净训练集（verify_correct=true）")
    ap.add_argument("--output", required=True, help="输出合并后的训练数据")
    ap.add_argument("--model", default="gemma4-26B")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 条被拒样本（调试）")
    args = ap.parse_args()

    if args.limit > 0:
        verified = load_jsonl(args.verified)
        rejected = [s for s in verified if s.get("verify_correct") is False][:args.limit]
        tmp_path = args.verified + ".tmp_rejected"
        with open(tmp_path, "w") as f:
            for s in rejected:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        # 需要也复制 clean 的路径逻辑——直接传 args 处理
        args.verified = tmp_path
        # 对于 limit 模式，从原 verified 取 clean（verify_correct=true）
        clean_in_verified = [s for s in verified if s.get("verify_correct") is True]
        tmp_clean = args.verified + ".tmp_clean"
        with open(tmp_clean, "w") as f:
            for s in clean_in_verified:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        args.clean = tmp_clean

    rejudge_and_merge(args.verified, args.clean, args.output,
                      model_name=args.model, max_concurrency=args.concurrency)

    if args.limit > 0:
        for p in [args.verified, args.clean]:
            if os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    main()
