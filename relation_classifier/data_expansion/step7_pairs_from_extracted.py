"""Step 7: 从抽取的原子记忆构造配对 → gemma4-26B 分类 → verify，目标各类 3k 条。

策略:
- 相似配对 (高 cosine, 同 persona): 富含 EQV/OSN/NSO/CON
- 随机配对 (同 persona): 富含 IND
- 分类 + verify 后按标签收集，达到各类目标数量
"""
import json
import os
import sys
import time
import asyncio
import random
import numpy as np
from collections import defaultdict, Counter
from typing import List

HERE = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_repo_root, "src"))
sys.path.insert(0, os.path.join(_repo_root, "relation_classifier"))


def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_candidate_pairs(memories, sim_lo=0.55, sim_hi=0.97, max_sim_per_persona=200,
                          random_per_persona=60):
    """构造候选配对: 相似对(中高相似度) + 随机对。"""
    from classifier import RelationClassifier

    by_persona = defaultdict(list)
    for m in memories:
        by_persona[m["persona_id"]].append(m)

    clf = RelationClassifier()
    similar_pairs = []
    random_pairs = []

    personas = list(by_persona.keys())
    for pi, (pid, mems) in enumerate(by_persona.items()):
        if len(mems) < 2:
            continue
        texts = [m["text"] for m in mems]
        feats = clf._features(texts).cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        feats = feats / norms

        sim = feats @ feats.T
        n = len(mems)

        # 相似对: sim_lo < cosine < sim_hi (排除近重复)
        cands = []
        for i in range(n):
            for j in range(i + 1, n):
                s = float(sim[i, j])
                if sim_lo < s < sim_hi:
                    cands.append((s, i, j))
        cands.sort(reverse=True)
        for s, i, j in cands[:max_sim_per_persona]:
            similar_pairs.append({
                "old": mems[i]["text"], "new": mems[j]["text"],
                "pair_type": "similar", "persona_id": pid,
                "source_detail": f"cosine={s:.3f}",
            })

        # 随机对
        seen = set()
        attempts = 0
        cnt = 0
        while cnt < random_per_persona and attempts < random_per_persona * 5:
            attempts += 1
            i, j = random.sample(range(n), 2)
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            random_pairs.append({
                "old": mems[i]["text"], "new": mems[j]["text"],
                "pair_type": "random", "persona_id": pid,
                "source_detail": "random",
            })
            cnt += 1

    print(f"候选: 相似 {len(similar_pairs)}, 随机 {len(random_pairs)}")
    return similar_pairs, random_pairs


def classify_pairs(pairs, model_name="gemma4-26B", max_concurrency=80):
    """用 v3 prompt 分类。"""
    from utils.llm_api import load_api_chat_completion
    from prompts import render_prompt

    system_prompt = render_prompt("lme_relation_classification_system_en_v3.jinja")
    messages_list = []
    for p in pairs:
        user_prompt = render_prompt("lme_relation_classification_user.jinja",
                                    m_old=p["old"], m_new=p["new"])
        messages_list.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

    print(f"分类: {len(messages_list)} 对 → {model_name}...")
    t0 = time.time()
    client = load_api_chat_completion(model_name, async_=True)
    responses = asyncio.run(client.get_response_chat(
        messages_list, max_new_tokens=128, temperature=0.0,
        max_concurrency=max_concurrency, use_tqdm=True))
    print(f"完成: {(time.time()-t0)/60:.1f}min")

    import re as _re
    valid = {"IND", "EQV", "OSN", "NSO", "CON"}
    for p, resp in zip(pairs, responses):
        label = None
        if resp:
            try:
                rc = resp.strip()
                if rc.startswith("```"):
                    rc = rc.split("\n", 1)[-1].rsplit("```", 1)[0]
                obj = json.loads(rc)
                if isinstance(obj, dict):
                    label = obj.get("relation", "").strip().upper()
            except (json.JSONDecodeError, AttributeError):
                for lbl in valid:
                    if _re.search(r'\b' + lbl + r'\b', resp):
                        label = lbl
                        break
        p["label"] = label if label in valid else "PARSE_ERROR"
    return pairs


def verify_pairs(pairs, model_name="gemma4-26B", max_concurrency=80):
    """按 label 用对应 verify prompt 验证。IND 不验证(直接信任)。"""
    from utils.llm_api import load_api_chat_completion
    from prompts import render_prompt

    per_label = {
        "EQV": "lme_relation_verify_system_eqv_en.jinja",
        "NSO": "lme_relation_verify_system_nso_en.jinja",
        "OSN": "lme_relation_verify_system_osn_en.jinja",
        "CON": "lme_relation_verify_system_con_en.jinja",
    }
    to_verify = [p for p in pairs if p["label"] in per_label]
    print(f"验证: {len(to_verify)} 对 (非IND/EQV...)")

    client = load_api_chat_completion(model_name, async_=True)
    by_label = defaultdict(list)
    for p in to_verify:
        by_label[p["label"]].append(p)

    for lbl, group in by_label.items():
        system_prompt = render_prompt(per_label[lbl])
        messages_list = []
        for p in group:
            user_prompt = render_prompt("lme_relation_verify_user.jinja",
                                        m_old=p["old"], m_new=p["new"], relation=lbl)
            messages_list.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])
        print(f"  verify [{lbl}]: {len(messages_list)}...")
        responses = asyncio.run(client.get_response_chat(
            messages_list, max_new_tokens=64, temperature=0.0,
            max_concurrency=max_concurrency, use_tqdm=True))
        for p, resp in zip(group, responses):
            correct = None
            if resp:
                try:
                    rc = resp.strip()
                    if rc.startswith("```"):
                        rc = rc.split("\n", 1)[-1].rsplit("```", 1)[0]
                    correct = json.loads(rc).get("correct", None)
                except (json.JSONDecodeError, AttributeError):
                    import re as _re
                    m = _re.search(r'"correct"\s*:\s*(true|false)', resp, _re.IGNORECASE)
                    if m:
                        correct = m.group(1).lower() == "true"
            p["verify_correct"] = correct

    # IND 默认通过
    for p in pairs:
        if p["label"] == "IND":
            p["verify_correct"] = True
    return pairs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--memories", default="relation_classifier/data_expansion/data/extracted_memories.jsonl")
    ap.add_argument("--output", default="relation_classifier/data_expansion/data/extracted_training_data.jsonl")
    ap.add_argument("--model", default="gemma4-26B")
    ap.add_argument("--concurrency", type=int, default=80)
    ap.add_argument("--target-per-class", type=int, default=3000)
    args = ap.parse_args()

    random.seed(42)
    mems_path = args.memories if os.path.isabs(args.memories) else os.path.join(_repo_root, args.memories)
    out_path = args.output if os.path.isabs(args.output) else os.path.join(_repo_root, args.output)

    memories = load_jsonl(mems_path)
    print(f"加载 {len(memories)} 条记忆")

    similar, rand = build_candidate_pairs(memories)
    all_pairs = similar + rand

    all_pairs = classify_pairs(all_pairs, args.model, args.concurrency)
    print(f"分类分布: {dict(Counter(p['label'] for p in all_pairs))}")

    all_pairs = verify_pairs(all_pairs, args.model, args.concurrency)

    # 收集通过验证的, 按类去重并限量
    seen = set()
    by_label = defaultdict(list)
    for p in all_pairs:
        if p.get("verify_correct") is not True:
            continue
        key = (p["old"].strip().lower(), p["new"].strip().lower())
        if key in seen:
            continue
        seen.add(key)
        by_label[p["label"]].append(p)

    print(f"\n验证通过分布: {dict((k, len(v)) for k, v in by_label.items())}")

    final = []
    for lbl in ["IND", "EQV", "OSN", "NSO", "CON"]:
        group = by_label.get(lbl, [])
        random.shuffle(group)
        take = group[:args.target_per_class]
        final.extend(take)
        print(f"  {lbl}: {len(take)} / 目标 {args.target_per_class}")

    random.shuffle(final)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in final:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\n最终: {len(final)} 条 → {out_path}")


if __name__ == "__main__":
    main()
