# Classifier 训练数据扩充 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从 PersonaMem-V2 提取原子记忆、构造配对、双裁判打标、生成扩展训练数据，最终重训练分类头。

**Architecture:** 5 个独立步骤脚本 + 1 个编排脚本，全部放在 `relation_classifier/data_expansion/` 下。每步读写 JSONL 中间文件，步间低耦合。LLM 调用复用现有 `load_api_chat_completion`，分类器调用复用 `RelationClassifier`。

**Tech Stack:** Python 3.10+, torch, transformers, openai (existing), PyYAML, numpy, scikit-learn (cosine_similarity)

## Global Constraints

- gemma4-26B 通过 vLLM 调用，使用现有 `load_api_chat_completion("gemma4-26B")`，依赖 `VLLM_BASE_URL` 环境变量
- gemma4-26B 分类使用 prompt `lme_relation_classification_system_en_v2.jinja` + `lme_relation_classification_user.jinja`
- gemma4-26B 输出格式 `{"relation": "IND"|"EQV"|"NSO"|"OSN"|"CON"}`
- classifier 使用 `RelationClassifier`，backbone 默认 `/mnt/data_oss/models/Qwen3-0.6B`
- 所有记忆文本以 `the user` 为主语（第三人称），与现有训练数据格式一致
- IND 最终占比控制在 30%
- 重训练超参数与原始训练一致（从 `head_best.pt` 的 `cfg` 读取）

---

### Task 1: 配置 + Step 1 — 原子记忆提取与主语改写

**Files:**
- Create: `relation_classifier/data_expansion/__init__.py`
- Create: `relation_classifier/data_expansion/config.yaml`
- Create: `relation_classifier/data_expansion/step1_extract_preferences.py`
- Create: `relation_classifier/data_expansion/data/.gitkeep`

**Interfaces:**
- Produces: `extract_personamem_preferences(persona_dir, output_path)` → writes JSONL
- Output format: `{"persona_id", "pref_id", "pref_type", "text", "updated", "prev_text", "who", "topic_preference"}`

- [ ] **Step 1: Create directory structure and config**

```bash
mkdir -p relation_classifier/data_expansion/data
touch relation_classifier/data_expansion/data/.gitkeep
```

```yaml
# relation_classifier/data_expansion/config.yaml
# PersonaMem 数据路径
personamem_dir: "data/raw_data/PersonaMem-v2/data/raw_data"

# 中间输出路径
data_dir: "relation_classifier/data_expansion/data"
atomic_memories_path: "relation_classifier/data_expansion/data/personamem_atomic_memories.jsonl"
pairs_all_path: "relation_classifier/data_expansion/data/pairs_all.jsonl"
pairs_judged_path: "relation_classifier/data_expansion/data/pairs_with_judgments.jsonl"
training_data_path: "relation_classifier/data_expansion/data/training_data_expanded.jsonl"

# 配对构造参数
similarity_threshold: 0.85       # 同 persona 内相似配对的最低余弦相似度
ind_ratio_target: 0.30           # IND 目标占比
max_similar_pairs_per_persona: 50  # 每个 persona 相似配对数上限

# gemma4-26B 裁判
gemma_model: "gemma4-26B"
gemma_max_concurrency: 10
gemma_max_new_tokens: 128

# classifier 配置（继承 relation_classifier/config.yaml）
classifier_backbone: "/mnt/data_oss/models/Qwen3-0.6B"
classifier_config: "relation_classifier/config.yaml"
classifier_checkpoint: "relation_classifier/head_best.pt"

# 原有训练数据
original_training_data: "non_ind.jsonl"
```

- [ ] **Step 2: Write Step 1 extraction script**

```python
# relation_classifier/data_expansion/step1_extract_preferences.py
"""Step 1: 从 PersonaMem-V2 提取原子记忆 + 主语改写为 'the user' 形式。"""

import json
import glob
import re
import os
import sys


def rewrite_to_first_person(text: str) -> str:
    """将 PersonaMem 偏好文本改写为以 'the user' 为主语的原子记忆。

    处理以下模式：
    1. "My X ..." → "the user's X ..."
    2. "I ..." → "the user ..."
    3. Bare predicate → prepend "the user "
    4. "Do not remember 'X' in memory" → "the user wants to forget about X"
    """
    text = text.strip()

    # Case 1: "My X ..."
    m = re.match(r'^[Mm]y\s+(\S)(.*)', text)
    if m:
        rest = m.group(1).lower() + m.group(2)
        return f"the user's {rest}"

    # Case 2: "I ..." (但排除 "In " "If " 等)
    m = re.match(r'^[Ii]\s+(\S)(.*)', text)
    if m:
        rest = m.group(1).lower() + m.group(2)
        return f"the user {rest}"

    # Case 4: ask_to_forget pattern
    m = re.match(r"^[Dd]o not remember\s+['\"](.+)['\"]\s+in memory", text)
    if m:
        return f"the user wants to forget about {m.group(1)}"

    # Case 3: Bare predicate — no clear subject, prepend "the user"
    # Check if already starts with "the user" or "The user"
    if re.match(r'^[Tt]he user\b', text):
        return text

    # Otherwise prepend "the user " with lowercase first char
    first_char = text[0].lower() if text[0].isupper() else text[0]
    return f"the user {first_char}{text[1:]}"


def extract_personamem_preferences(persona_dir: str, output_path: str) -> list[dict]:
    """遍历 PersonaMem JSON 文件，提取 who='self' 的 preference。

    Args:
        persona_dir: PersonaMem raw_data 目录路径
        output_path: 输出 JSONL 路径

    Returns:
        提取的 preference 列表
    """
    results = []
    json_files = sorted(glob.glob(os.path.join(persona_dir, "*.json")))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found under {persona_dir}")

    for fp in json_files:
        data = json.load(open(fp, encoding="utf-8"))
        for persona_id, pdata in data.items():
            for conv_type, items in pdata.get("conversations", {}).items():
                if not isinstance(items, list):
                    continue
                for i, item in enumerate(items):
                    if not isinstance(item, dict):
                        continue
                    if "preference" not in item:
                        continue

                    who = item.get("who", "unknown")
                    if who != "self":
                        continue  # 跳过 others（259 条，后续 LLM 改写）

                    pref_text = item["preference"]
                    rewritten = rewrite_to_first_person(pref_text)

                    prev_text = item.get("prev_pref", "")
                    prev_rewritten = ""
                    if prev_text:
                        prev_rewritten = rewrite_to_first_person(prev_text)

                    result = {
                        "persona_id": persona_id,
                        "pref_id": f"{persona_id}_{conv_type}_{i}",
                        "pref_type": item.get("pref_type", ""),
                        "topic_preference": item.get("topic_preference", ""),
                        "text": rewritten,
                        "original_text": pref_text,
                        "updated": bool(item.get("updated", False)),
                        "prev_text": prev_rewritten,
                        "prev_original": prev_text,
                        "who": who,
                        "conversation_scenario": item.get("conversation_scenario", ""),
                    }
                    results.append(result)

    # 写入输出
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"提取完成: {len(results)} 条 self preference → {output_path}")
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Step 1: Extract & rewrite PersonaMem preferences")
    ap.add_argument("--persona-dir", required=True, help="PersonaMem raw_data 目录")
    ap.add_argument("--output", required=True, help="输出 JSONL 路径")
    args = ap.parse_args()
    extract_personamem_preferences(args.persona_dir, args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke test — run on PersonaMem data**

```bash
cd /data/zjj/project_26/fact_mem && python relation_classifier/data_expansion/step1_extract_preferences.py \
  --persona-dir data/raw_data/PersonaMem-v2/data/raw_data \
  --output relation_classifier/data_expansion/data/personamem_atomic_memories.jsonl
```

Expected: prints ~5,140 extracted preferences, file created.

- [ ] **Step 4: Verify output quality — spot check**

```bash
# 检查改写质量：查看前 20 条
head -20 relation_classifier/data_expansion/data/personamem_atomic_memories.jsonl | python3 -m json.tool --no-ensure-ascii 2>/dev/null || head -20 relation_classifier/data_expansion/data/personamem_atomic_memories.jsonl

# 检查 updated=true 的条目数
python3 -c "
import json
data = [json.loads(l) for l in open('relation_classifier/data_expansion/data/personamem_atomic_memories.jsonl')]
updated = [d for d in data if d['updated']]
print(f'Total: {len(data)}, Updated: {len(updated)}')
# 抽查改写质量
for d in updated[:5]:
    print(f\"  prev: {d['prev_text']}\")
    print(f\"  curr: {d['text']}\")
    print(f\"  type: {d['pref_type']}\")
    print()
"
```

Expected: ~5,140 total, ~2,800 updated. Rewritten texts start with "the user".

- [ ] **Step 5: Commit**

```bash
git add relation_classifier/data_expansion/__init__.py \
        relation_classifier/data_expansion/config.yaml \
        relation_classifier/data_expansion/step1_extract_preferences.py \
        relation_classifier/data_expansion/data/.gitkeep
git commit -m "feat: Step 1 - PersonaMem preference extraction & first-person rewrite"
```

---

### Task 2: Step 2 — (old, new) 配对构造

**Files:**
- Create: `relation_classifier/data_expansion/step2_construct_pairs.py`

**Interfaces:**
- Consumes: `personamem_atomic_memories.jsonl` (from Task 1)
- Produces: `build_all_pairs(memories_path, output_path, config)` → writes `pairs_all.jsonl`
- Output format: `{"old", "new", "pair_type", "persona_id", "old_pref_id", "new_pref_id", "source_detail"}`
- `pair_type`: `"update_chain"` | `"similar"` | `"random_ind"`

- [ ] **Step 1: Write Step 2 script**

```python
# relation_classifier/data_expansion/step2_construct_pairs.py
"""Step 2: 从原子记忆构造 (old, new) 配对 — 更新链 / 相似配对 / 随机 IND。"""

import json
import os
import random
import sys
import yaml
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple


def load_memories(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_update_chain_pairs(memories: List[dict]) -> List[dict]:
    """2a: 更新链配对 prev_text → text（天然非 IND）。"""
    pairs = []
    for m in memories:
        if m["updated"] and m["prev_text"]:
            pairs.append({
                "old": m["prev_text"],
                "new": m["text"],
                "pair_type": "update_chain",
                "persona_id": m["persona_id"],
                "old_pref_id": m["pref_id"] + "_prev",
                "new_pref_id": m["pref_id"],
                "source_detail": f"pref_type={m['pref_type']}",
            })
    return pairs


def build_similar_pairs(
    memories: List[dict],
    similarity_threshold: float = 0.85,
    max_per_persona: int = 50,
) -> List[dict]:
    """2b: 同 persona 内 embedding 相似但非更新链的配对，挖掘潜在非 IND。

    使用 classifier backbone 抽取 embedding，计算余弦相似度。
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from classifier import RelationClassifier

    # 按 persona 分组
    by_persona = defaultdict(list)
    for m in memories:
        by_persona[m["persona_id"]].append(m)

    # 建立所有更新链 pair 的 (old_pref_id, new_pref_id) 集合，避免重复
    update_pairs_set = set()
    for m in memories:
        if m["updated"] and m["prev_text"]:
            update_pairs_set.add((m["pref_id"] + "_prev", m["pref_id"]))

    # 加载 classifier 用于抽取 embedding
    clf = RelationClassifier()
    pairs = []

    for persona_id, mems in by_persona.items():
        if len(mems) < 2:
            continue

        texts = [m["text"] for m in mems]
        features = clf._features(texts)  # [N, 1024]
        features_np = features.cpu().numpy().astype(np.float32)

        # 归一化
        norms = np.linalg.norm(features_np, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        features_np = features_np / norms

        # 余弦相似度矩阵
        sim = np.dot(features_np, features_np.T)

        # 收集高相似度对 (排除对角线、排除更新链)
        candidates = []
        n = len(mems)
        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] < similarity_threshold:
                    continue
                pair_key_i = (mems[i]["pref_id"], mems[j]["pref_id"])
                pair_key_j = (mems[j]["pref_id"], mems[i]["pref_id"])
                if pair_key_i in update_pairs_set or pair_key_j in update_pairs_set:
                    continue
                candidates.append((sim[i, j], i, j))

        # 按相似度降序，取 top max_per_persona
        candidates.sort(key=lambda x: x[0], reverse=True)
        for sim_score, i, j in candidates[:max_per_persona]:
            # old/new 顺序：按 pref_id 保证一致性
            if mems[i]["pref_id"] < mems[j]["pref_id"]:
                old_m, new_m = mems[i], mems[j]
            else:
                old_m, new_m = mems[j], mems[i]
            pairs.append({
                "old": old_m["text"],
                "new": new_m["text"],
                "pair_type": "similar",
                "persona_id": persona_id,
                "old_pref_id": old_m["pref_id"],
                "new_pref_id": new_m["pref_id"],
                "source_detail": f"cosine_sim={sim_score:.4f}",
            })

    return pairs


def build_random_ind_pairs(
    memories: List[dict],
    existing_pairs: List[dict],
    target_count: int,
) -> List[dict]:
    """2c: 同 persona 内随机配对，作为 IND 候选。"""
    existing_set = set()
    for p in existing_pairs:
        existing_set.add((p["old_pref_id"], p["new_pref_id"]))
        existing_set.add((p["new_pref_id"], p["old_pref_id"]))

    by_persona = defaultdict(list)
    for m in memories:
        by_persona[m["persona_id"]].append(m)

    random.seed(42)
    candidates = []

    # 每个 persona 内随机生成候选对
    for persona_id, mems in by_persona.items():
        if len(mems) < 2:
            continue
        ids = list(range(len(mems)))
        random.shuffle(ids)
        for i_idx in range(min(len(ids), len(ids))):
            for j_idx in range(i_idx + 1, len(ids)):
                i, j = ids[i_idx], ids[j_idx]
                pair_key = (mems[i]["pref_id"], mems[j]["pref_id"])
                if pair_key in existing_set:
                    continue
                candidates.append((persona_id, i, j, mems))
                existing_set.add(pair_key)
                existing_set.add((pair_key[1], pair_key[0]))

    # 随机采样 target_count 条
    random.shuffle(candidates)
    selected = candidates[:target_count]

    pairs = []
    for persona_id, i, j, mems in selected:
        if mems[i]["pref_id"] < mems[j]["pref_id"]:
            old_m, new_m = mems[i], mems[j]
        else:
            old_m, new_m = mems[j], mems[i]
        pairs.append({
            "old": old_m["text"],
            "new": new_m["text"],
            "pair_type": "random_ind",
            "persona_id": persona_id,
            "old_pref_id": old_m["pref_id"],
            "new_pref_id": new_m["pref_id"],
            "source_detail": "random_sampling",
        })

    return pairs


def build_all_pairs(memories_path: str, output_path: str, config: dict) -> Dict[str, int]:
    """构造全部配对并写入输出文件。

    Returns:
        {"update_chain": N, "similar": N, "random_ind": N}
    """
    memories = load_memories(memories_path)
    print(f"加载 {len(memories)} 条原子记忆")

    # 2a: 更新链配对
    update_pairs = build_update_chain_pairs(memories)
    print(f"2a 更新链配对: {len(update_pairs)} 对")

    # 2b: 相似配对
    sim_threshold = config.get("similarity_threshold", 0.85)
    max_per_persona = config.get("max_similar_pairs_per_persona", 50)
    similar_pairs = build_similar_pairs(memories, sim_threshold, max_per_persona)
    print(f"2b 相似配对: {len(similar_pairs)} 对")

    # 2c: 随机 IND 配对（初步估计数量，Step 4 会精调）
    all_nonrandom = update_pairs + similar_pairs
    # 预估 IND 数量：设非 IND 总量 N，IND = N * 0.3 / 0.7
    est_non_ind = len(all_nonrandom)  # 预估全为非 IND
    est_ind = int(est_non_ind * config.get("ind_ratio_target", 0.30) / 0.70)
    random_pairs = build_random_ind_pairs(memories, all_nonrandom, est_ind)
    print(f"2c 随机 IND 配对: {len(random_pairs)} 对")

    # 合并写入
    all_pairs = update_pairs + similar_pairs + random_pairs
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    counts = {
        "update_chain": len(update_pairs),
        "similar": len(similar_pairs),
        "random_ind": len(random_pairs),
        "total": len(all_pairs),
    }
    print(f"全部配对: {counts} → {output_path}")
    return counts


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Step 2: Construct (old, new) pairs")
    ap.add_argument("--memories", required=True, help="Step 1 输出的原子记忆 JSONL")
    ap.add_argument("--output", required=True, help="输出 pairs JSONL 路径")
    ap.add_argument("--config", default=None, help="YAML 配置文件")
    args = ap.parse_args()

    cfg = {}
    if args.config:
        cfg = yaml.safe_load(open(args.config))
    build_all_pairs(args.memories, args.output, cfg)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test — run pair construction**

```bash
cd /data/zjj/project_26/fact_mem && python relation_classifier/data_expansion/step2_construct_pairs.py \
  --memories relation_classifier/data_expansion/data/personamem_atomic_memories.jsonl \
  --output relation_classifier/data_expansion/data/pairs_all.jsonl \
  --config relation_classifier/data_expansion/config.yaml
```

Expected: prints pair counts, ~4,000-5,000 total pairs.

- [ ] **Step 3: Verify pair quality**

```bash
# 查看各类别配对数
python3 -c "
import json
from collections import Counter
pairs = [json.loads(l) for l in open('relation_classifier/data_expansion/data/pairs_all.jsonl')]
c = Counter(p['pair_type'] for p in pairs)
print('Pair type distribution:', dict(c))
print('Total:', len(pairs))
# 抽查几条
for p in pairs[:3]:
    print(f\"  [{p['pair_type']}] old: {p['old'][:80]}\")
    print(f\"  [{p['pair_type']}] new: {p['new'][:80]}\")
    print()
"
```

- [ ] **Step 4: Commit**

```bash
git add relation_classifier/data_expansion/step2_construct_pairs.py
git commit -m "feat: Step 2 - (old, new) pair construction (update/similar/random)"
```

---

### Task 3: Step 3 — 双裁判并行判断

**Files:**
- Create: `relation_classifier/data_expansion/step3_dual_judge.py`

**Interfaces:**
- Consumes: `pairs_all.jsonl` (from Task 2), `RelationClassifier`, `load_api_chat_completion`
- Produces: `judge_all_pairs(pairs_path, output_path, config)` → writes `pairs_with_judgments.jsonl`
- Output format: adds `classifier_label`, `classifier_probs`, `gemma_label` to each pair

- [ ] **Step 1: Write Step 3 script**

```python
# relation_classifier/data_expansion/step3_dual_judge.py
"""Step 3: classifier (Qwen3-0.6B) 与 gemma4-26B 并行判断五分类关系。"""

import json
import os
import sys
import time
import yaml
from typing import List, Dict


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


def build_gemma_messages(old: str, new: str) -> List[dict]:
    """构建 gemma4-26B 分类请求的 messages。"""
    from prompts import render_prompt

    system_prompt = render_prompt("lme_relation_classification_system_en_v2.jinja")
    user_prompt = render_prompt("lme_relation_classification_user.jinja",
                                m_old=old, m_new=new)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def gemma_judge(pairs: List[dict], model_name: str = "gemma4-26B",
                max_concurrency: int = 10) -> List[dict]:
    """用 gemma4-26B 对全部 pair 做五分类。"""
    from utils.llm_api import load_api_chat_completion

    client = load_api_chat_completion(model_name)
    messages_list = [build_gemma_messages(p["old"], p["new"]) for p in pairs]

    print(f"发送 {len(messages_list)} 条分类请求到 {model_name}...")
    t0 = time.time()

    # 使用异步客户端批量请求
    if hasattr(client, 'get_response_chat') and 'messages_list' in str(type(client)):
        # 同步客户端，逐条调用
        responses = []
        for i, msgs in enumerate(messages_list):
            if i % 50 == 0:
                print(f"  gemma4-26B 进度: {i}/{len(messages_list)}")
            resp = client.get_response_chat(msgs, max_new_tokens=128, temperature=0.0)
            responses.append(resp)
    else:
        # 异步客户端
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
                label = obj.get("relation", "").strip().upper()
            except (json.JSONDecodeError, AttributeError):
                # 尝试从文本中提取标签
                for lbl in valid_labels:
                    if lbl in resp:
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


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test — run on 5 pairs to verify**

```bash
cd /data/zjj/project_26/fact_mem && python relation_classifier/data_expansion/step3_dual_judge.py \
  --pairs relation_classifier/data_expansion/data/pairs_all.jsonl \
  --output relation_classifier/data_expansion/data/pairs_with_judgments.jsonl \
  --config relation_classifier/data_expansion/config.yaml \
  --limit 5
```

Expected: prints progress, output file has `classifier_label`, `gemma_label` fields.

- [ ] **Step 3: Verify dual judge output**

```bash
python3 -c "
import json
from collections import Counter
pairs = [json.loads(l) for l in open('relation_classifier/data_expansion/data/pairs_with_judgments.jsonl')]
print(f'Total judged: {len(pairs)}')
if len(pairs) > 0:
    print('Fields:', list(pairs[0].keys()))
    clf_labels = Counter(p.get('classifier_label') for p in pairs)
    gemma_labels = Counter(p.get('gemma_label') for p in pairs)
    print('Classifier labels:', dict(clf_labels))
    print('Gemma labels:', dict(gemma_labels))
"
```

- [ ] **Step 4: Commit**

```bash
git add relation_classifier/data_expansion/step3_dual_judge.py
git commit -m "feat: Step 3 - dual judge with classifier + gemma4-26B"
```

---

### Task 4: Step 4 — 比对 & 生成最终训练数据

**Files:**
- Create: `relation_classifier/data_expansion/step4_generate_training_data.py`

**Interfaces:**
- Consumes: `pairs_with_judgments.jsonl` (from Task 3), `non_ind.jsonl` (original)
- Produces: `generate_training_data(judged_path, original_path, output_path, config)` → writes `training_data_expanded.jsonl`
- Output format: `{"old", "new", "label", "source", "gemma_label", "classifier_label", "pref_type", "agree"}`
- IND ratio controlled to 30%

- [ ] **Step 1: Write Step 4 script**

```python
# relation_classifier/data_expansion/step4_generate_training_data.py
"""Step 4: 比对双裁判结果，以 gemma4-26B 为准生成训练数据，控制 IND 比例。"""

import json
import os
import random
import yaml
from typing import List, Dict


def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def generate_training_data(
    judged_path: str,
    original_path: str,
    output_path: str,
    config: dict,
):
    """生成最终训练数据。

    策略:
    1. 新数据: gemma_label 与 classifier_label 不一致 → 以 gemma 为准，agree=false
    2. 新数据: 一致且非 IND → 保留，agree=true
    3. 新数据: 一致且 IND → 进入 IND 候选池
    4. 合并原有数据
    5. IND 比例控制到 30%
    """
    judged = load_jsonl(judged_path)
    ind_ratio_target = config.get("ind_ratio_target", 0.30)

    non_ind_samples = []  # 非 IND（确定加入训练集）
    ind_candidates = []   # IND 候选池
    errors = []           # gemma 解析失败的

    for p in judged:
        gemma_label = p.get("gemma_label", "")

        if gemma_label == "PARSE_ERROR":
            errors.append(p)
            continue

        # 以 gemma4-26B 标签为准
        agree = (gemma_label == p.get("classifier_label", ""))

        sample = {
            "old": p["old"],
            "new": p["new"],
            "label": gemma_label,
            "source": f"persona_{p.get('persona_id', 'unknown')}",
            "gemma_label": gemma_label,
            "classifier_label": p.get("classifier_label", ""),
            "pref_type": p.get("source_detail", ""),
            "agree": agree,
        }

        if gemma_label == "IND":
            ind_candidates.append(sample)
        else:
            non_ind_samples.append(sample)

    # 加载原有训练数据
    original = []
    if os.path.exists(original_path):
        original = load_jsonl(original_path)
        # 原有数据统一补充字段
        for o in original:
            o.setdefault("source", "original")
            o.setdefault("gemma_label", "")
            o.setdefault("classifier_label", "")
            o.setdefault("pref_type", "")
            o.setdefault("agree", True)
    else:
        print(f"警告: 原有训练数据 {original_path} 不存在，跳过合并")

    # IND 比例控制：目标 IND 占总量 30%
    total_non_ind = len(non_ind_samples) + len([o for o in original if o.get("label") != "IND"])
    ind_needed = int(total_non_ind * ind_ratio_target / (1 - ind_ratio_target))

    random.seed(42)
    if len(ind_candidates) > ind_needed:
        ind_selected = random.sample(ind_candidates, ind_needed)
        print(f"IND 候选池 {len(ind_candidates)} → 采样 {ind_needed} 条（控制 {ind_ratio_target:.0%}）")
    else:
        ind_selected = ind_candidates
        print(f"IND 候选池不足: 需要 {ind_needed}, 实际 {len(ind_candidates)}，全部保留")

    # 合并全部数据
    all_samples = non_ind_samples + ind_selected + original

    # 基于 (old, new) 去重（保留首次出现）
    seen = set()
    deduped = []
    for s in all_samples:
        key = (s["old"].strip().lower(), s["new"].strip().lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)

    # 打乱
    random.shuffle(deduped)

    # 统计
    from collections import Counter
    label_counts = Counter(s["label"] for s in deduped)
    ind_pct = label_counts.get("IND", 0) / len(deduped) if deduped else 0

    print(f"训练数据统计:")
    print(f"  新数据-非IND: {len(non_ind_samples)}")
    print(f"  新数据-IND: {len(ind_selected)}")
    print(f"  原有数据: {len(original)}")
    print(f"  去重后合计: {len(deduped)}")
    print(f"  gemma 解析失败: {len(errors)}")
    print(f"  标签分布: {dict(label_counts)}")
    print(f"  IND 占比: {ind_pct:.2%}")
    print(f"  不一致样本 (agree=false): {sum(1 for s in deduped if s.get('agree') is False)}")

    # 写入
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in deduped:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"训练数据写入 → {output_path}")
    return deduped


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Step 4: Generate training data")
    ap.add_argument("--judged", required=True, help="Step 3 输出的 judged pairs JSONL")
    ap.add_argument("--original", default="non_ind.jsonl", help="原有训练数据")
    ap.add_argument("--output", required=True, help="输出训练数据 JSONL 路径")
    ap.add_argument("--config", default=None, help="YAML 配置文件")
    args = ap.parse_args()

    cfg = {}
    if args.config:
        cfg = yaml.safe_load(open(args.config))
    generate_training_data(args.judged, args.original, args.output, cfg)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test with mock judged data**

```bash
cd /data/zjj/project_26/fact_mem

# 先用 --limit 跑出少量 judged 数据做测试
python relation_classifier/data_expansion/step3_dual_judge.py \
  --pairs relation_classifier/data_expansion/data/pairs_all.jsonl \
  --output relation_classifier/data_expansion/data/pairs_judged_test.jsonl \
  --config relation_classifier/data_expansion/config.yaml \
  --limit 20

# 运行 Step 4
python relation_classifier/data_expansion/step4_generate_training_data.py \
  --judged relation_classifier/data_expansion/data/pairs_judged_test.jsonl \
  --original non_ind.jsonl \
  --output relation_classifier/data_expansion/data/training_data_test.jsonl \
  --config relation_classifier/data_expansion/config.yaml
```

Expected: prints statistics with label distribution, file created.

- [ ] **Step 3: Verify output format**

```bash
python3 -c "
import json
from collections import Counter
data = [json.loads(l) for l in open('relation_classifier/data_expansion/data/training_data_test.jsonl')]
print(f'Total: {len(data)}')
print('Fields:', list(data[0].keys()))
c = Counter(d['label'] for d in data)
print('Label distribution:', dict(c))
# 检查必须字段
for d in data[:3]:
    for f in ['old', 'new', 'label', 'source', 'gemma_label', 'classifier_label', 'agree']:
        assert f in d, f'Missing field: {f}'
print('All required fields present ✓')
"
```

- [ ] **Step 4: Commit**

```bash
git add relation_classifier/data_expansion/step4_generate_training_data.py
git commit -m "feat: Step 4 - training data generation with IND ratio control"
```

---

### Task 5: 端到端编排脚本

**Files:**
- Create: `relation_classifier/data_expansion/run_pipeline.py`

**Interfaces:**
- Consumes: all step modules, `config.yaml`
- Produces: runs full pipeline end-to-end
- CLI: `python run_pipeline.py [--steps 1,2,3,4] [--config config.yaml]`

- [ ] **Step 1: Write pipeline orchestrator**

```python
# relation_classifier/data_expansion/run_pipeline.py
"""端到端训练数据扩充流水线编排。"""

import os
import sys
import argparse
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG = os.path.join(HERE, "config.yaml")


def load_config(config_path):
    cfg = {}
    if config_path and os.path.exists(config_path):
        cfg = yaml.safe_load(open(config_path))
    return cfg


def resolve_path(cfg, key, default_relative):
    """解析配置中的路径：支持相对路径（相对 data_expansion 目录）。"""
    val = cfg.get(key, default_relative)
    if val and not os.path.isabs(val):
        # 相对路径相对于 repo 根目录
        repo_root = os.path.abspath(os.path.join(HERE, "..", ".."))
        val = os.path.join(repo_root, val)
    return val


def main():
    ap = argparse.ArgumentParser(description="训练数据扩充流水线")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="YAML 配置文件")
    ap.add_argument("--steps", default="1,2,3,4", help="要运行的步骤，逗号分隔")
    ap.add_argument("--limit", type=int, default=0, help="Step 3 仅处理前 N 对（调试用）")
    ap.add_argument("--dry-run", action="store_true", help="仅打印配置，不实际运行")
    args = ap.parse_args()

    cfg = load_config(args.config)
    steps = [int(s.strip()) for s in args.steps.split(",")]

    # 解析路径
    persona_dir = cfg.get("personamem_dir", "data/raw_data/PersonaMem-v2/data/raw_data")
    if not os.path.isabs(persona_dir):
        repo_root = os.path.abspath(os.path.join(HERE, "..", ".."))
        persona_dir = os.path.join(repo_root, persona_dir)

    data_dir = cfg.get("data_dir", os.path.join(HERE, "data"))
    mem_path = cfg.get("atomic_memories_path", os.path.join(data_dir, "personamem_atomic_memories.jsonl"))
    pairs_path = cfg.get("pairs_all_path", os.path.join(data_dir, "pairs_all.jsonl"))
    judged_path = cfg.get("pairs_judged_path", os.path.join(data_dir, "pairs_with_judgments.jsonl"))
    train_path = cfg.get("training_data_path", os.path.join(data_dir, "training_data_expanded.jsonl"))
    original_path = cfg.get("original_training_data", "non_ind.jsonl")
    if not os.path.isabs(original_path):
        original_path = os.path.join(repo_root, original_path)

    if args.dry_run:
        print("=== Dry Run ===")
        print(f"Config: {args.config}")
        print(f"Steps: {steps}")
        print(f"PersonaMem dir: {persona_dir}")
        print(f"Atomic memories: {mem_path}")
        print(f"Pairs: {pairs_path}")
        print(f"Judged: {judged_path}")
        print(f"Training data: {train_path}")
        print(f"Original data: {original_path}")
        return

    sys.path.insert(0, HERE)

    # Step 1
    if 1 in steps:
        print("=" * 60)
        print("Step 1: 原子记忆提取 & 主语改写")
        print("=" * 60)
        from step1_extract_preferences import extract_personamem_preferences
        extract_personamem_preferences(persona_dir, mem_path)

    # Step 2
    if 2 in steps:
        print("=" * 60)
        print("Step 2: (old, new) 配对构造")
        print("=" * 60)
        from step2_construct_pairs import build_all_pairs
        build_all_pairs(mem_path, pairs_path, cfg)

    # Step 3
    if 3 in steps:
        print("=" * 60)
        print("Step 3: 双裁判判断")
        print("=" * 60)
        from step3_dual_judge import judge_all_pairs

        if args.limit > 0:
            from step3_dual_judge import load_pairs
            pairs = load_pairs(pairs_path)[:args.limit]
            tmp_path = pairs_path + ".tmp_limit"
            import json
            with open(tmp_path, "w") as f:
                for p in pairs:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            pairs_path = tmp_path

        judge_all_pairs(pairs_path, judged_path, cfg)

    # Step 4
    if 4 in steps:
        print("=" * 60)
        print("Step 4: 生成训练数据")
        print("=" * 60)
        from step4_generate_training_data import generate_training_data
        generate_training_data(judged_path, original_path, train_path, cfg)

    print("=" * 60)
    print("流水线完成!")
    print(f"最终训练数据: {train_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test dry run**

```bash
cd /data/zjj/project_26/fact_mem && python relation_classifier/data_expansion/run_pipeline.py --dry-run
```

Expected: prints all paths, no errors.

- [ ] **Step 3: Commit**

```bash
git add relation_classifier/data_expansion/run_pipeline.py
git commit -m "feat: end-to-end pipeline orchestrator"
```

---

### Task 6: 集成测试 — 小规模端到端运行

- [ ] **Step 1: Run full pipeline with --limit 10**

```bash
cd /data/zjj/project_26/fact_mem && python relation_classifier/data_expansion/run_pipeline.py --limit 10
```

Expected: all 4 steps complete without error.

- [ ] **Step 2: Verify final training data is valid for training**

```bash
python3 -c "
import json
from collections import Counter

data = [json.loads(l) for l in open('relation_classifier/data_expansion/data/training_data_expanded.jsonl')]
print(f'Total samples: {len(data)}')

# 检查标签有效性
valid_labels = {'IND', 'EQV', 'OSN', 'NSO', 'CON'}
labels = [d['label'] for d in data]
invalid = [l for l in labels if l not in valid_labels]
if invalid:
    print(f'INVALID labels: {invalid}')
else:
    print('All labels valid ✓')

# 检查字段完整性
required = ['old', 'new', 'label']
for i, d in enumerate(data):
    for f in required:
        if f not in d or not d[f]:
            print(f'Missing/empty {f} at line {i}')
            break
    else:
        continue
    break
else:
    print('All required fields present ✓')

# 标签分布
c = Counter(labels)
print(f'Label distribution: {dict(c)}')
"
```

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "test: integration test dry-run with --limit 10"
```
