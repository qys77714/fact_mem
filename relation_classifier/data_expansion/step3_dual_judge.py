"""Step 3: classifier (Qwen3-0.6B) 与 gemma4-26B 并行判断五分类关系。"""

import json
import os
import sys
import time
from typing import List


def load_pairs(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def classifier_judge(pairs: List[dict]) -> List[dict]:
    """用 Qwen3-0.6B classifier 对全部 pair 做五分类预测。"""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from classifier import RelationClassifier

    clf = RelationClassifier()
    old_new = [(p["old"], p["new"]) for p in pairs]
    results = clf.predict_batch(old_new, return_probs=True)

    for p, r in zip(pairs, results):
        p["classifier_label"] = r["label"]
        p["classifier_probs"] = r["probs"]

    return pairs


def build_gemma_messages(old: str, new: str, language: str = "en") -> List[dict]:
    """构建 gemma4-26B 分类请求的 messages。"""
    from src.memory.candidate_ingest.prompts import (
        lme_relation_system_prompt_for_language,
        build_lme_relation_classification_user_prompt,
    )
    system_prompt = lme_relation_system_prompt_for_language(language)
    user_prompt = build_lme_relation_classification_user_prompt(old, new)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def gemma_judge(pairs: List[dict], model_name: str = "gemma4-26B",
                max_concurrency: int = 10) -> List[dict]:
    """用 gemma4-26B 对全部 pair 做五分类。"""
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _src = os.path.join(_script_dir, "..", "..", "src")
    sys.path.insert(0, os.path.abspath(_src))
    from utils.llm_api import load_api_chat_completion
    from src.memory.candidate_ingest.prompts import (
        lme_relation_system_prompt_for_language,
        build_lme_relation_classification_user_prompt,
    )

    system_prompt = lme_relation_system_prompt_for_language("en")
    messages_list = []
    for p in pairs:
        user_prompt = build_lme_relation_classification_user_prompt(p["old"], p["new"])
        messages_list.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

    print(f"发送 {len(messages_list)} 条分类请求到 {model_name}...")
    t0 = time.time()

    # 使用异步客户端批量请求
    import asyncio
    async_client = load_api_chat_completion(model_name, async_=True)
    responses = asyncio.run(
        async_client.get_response_chat(
            messages_list,
            max_new_tokens=128,
            temperature=0.0,
            max_concurrency=max_concurrency,
            use_tqdm=True,
        )
    )

    elapsed = time.time() - t0
    print(f"gemma4-26B 判断完成: {len(responses)} 条, 耗时 {elapsed:.1f}s")

    valid_labels = {"IND", "EQV", "OSN", "NSO", "CON"}
    success = 0
    failed = 0
    for i, (p, resp) in enumerate(zip(pairs, responses)):
        label = None
        if resp:
            try:
                # 尝试解析 JSON
                # 处理可能的 markdown code block 包裹
                resp_clean = resp.strip()
                if resp_clean.startswith("```"):
                    resp_clean = resp_clean.split("\n", 1)[-1]
                    resp_clean = resp_clean.rsplit("```", 1)[0]
                obj = json.loads(resp_clean)
                if not isinstance(obj, dict):
                    raise ValueError(f"Expected dict, got {type(obj).__name__}")
                label = obj.get("relation", "").strip().upper()
            except (json.JSONDecodeError, AttributeError):
                # 尝试从文本中提取标签
                import re as _re
                for lbl in valid_labels:
                    if _re.search(r'\b' + _re.escape(lbl) + r'\b', resp):
                        label = lbl
                        break

        if label in valid_labels:
            p["gemma_label"] = label
            success += 1
        else:
            p["gemma_label"] = "PARSE_ERROR"
            p["gemma_raw_response"] = resp[:200] if resp else "None"
            failed += 1

    print(f"gemma4-26B 标签解析: 成功 {success}, 失败 {failed}")
    return pairs


def judge_all_pairs(pairs_path: str, output_path: str, config: dict):
    """完整 Step 3: classifier + gemma4-26B 双裁判。"""
    pairs = load_pairs(pairs_path)
    print(f"加载 {len(pairs)} 对")

    # 1. classifier 判断（快速，先跑）
    print("--- classifier 判断 ---")
    pairs = classifier_judge(pairs)

    # 2. gemma4-26B 判断
    print("--- gemma4-26B 判断 ---")
    model = config.get("gemma_model", "gemma4-26B")
    max_concurrency = config.get("gemma_max_concurrency", 10)
    pairs = gemma_judge(pairs, model_name=model, max_concurrency=max_concurrency)

    # 写入输出
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"双裁判完成 → {output_path}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Step 3: Dual judge with classifier + gemma4-26B")
    ap.add_argument("--pairs", required=True, help="Step 2 输出的 pairs JSONL")
    ap.add_argument("--output", required=True, help="输出 JSONL 路径")
    ap.add_argument("--config", default=None, help="YAML 配置文件")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 对（调试用）")
    args = ap.parse_args()

    cfg = {}
    if args.config:
        import yaml
        cfg = yaml.safe_load(open(args.config))

    # 支持 limit 模式用于快速测试
    if args.limit > 0:
        pairs = load_pairs(args.pairs)[:args.limit]
        tmp_path = args.pairs + ".tmp_limit"
        with open(tmp_path, "w") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        args.pairs = tmp_path

    judge_all_pairs(args.pairs, args.output, cfg)

    if args.limit > 0:
        os.remove(tmp_path)


if __name__ == "__main__":
    main()
