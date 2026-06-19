"""从 PersonaMem chat_history 中抽取原子化事实，用于扩充训练集。

用 gemma4-26B + unified extraction prompt 从对话中抽取 user-centered 原子记忆，
然后同 persona 内配对、分类、验证，生成训练数据。
"""
import json
import os
import sys
import time
import asyncio
import glob
import random
from collections import Counter
from typing import List

HERE = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_repo_root, "src"))


def chunk_messages(messages: List[dict], chunk_size: int = 12) -> List[str]:
    """将 chat_history 消息分块，转为 extraction prompt 期望的 user/assistant 文本格式。"""
    chunks = []
    # 跳过 system 消息
    dial = [(m["role"], m["content"]) for m in messages if m["role"] != "system"]

    for i in range(0, len(dial), chunk_size):
        batch = dial[i:i + chunk_size]
        lines = []
        for role, content in batch:
            if role == "user":
                lines.append(f"user: {content}")
            elif role == "assistant":
                lines.append(f"assistant: {content}")
        if lines:
            chunks.append("\n\n".join(lines))
    return chunks


ASPECT_TEMPLATES = {
    "events": "legacy/0_mem_extract_aspect_events_en.jinja",
    "preferences": "legacy/0_mem_extract_aspect_preferences_en.jinja",
    "social": "legacy/0_mem_extract_aspect_social_en.jinja",
}


def extract_memories_from_chunks(
    chunks: List[str],
    persona_id: str,
    aspect: str,
    model_name: str = "gemma4-26B",
    max_concurrency: int = 30,
) -> List[dict]:
    """用 gemma4-26B 从一个 aspect 的对话块中抽取原子记忆。"""
    from utils.llm_api import load_api_chat_completion
    from prompts import render_prompt

    template = ASPECT_TEMPLATES.get(aspect, ASPECT_TEMPLATES["events"])
    system_prompt = render_prompt(template)
    system_prompt += (
        "\n\n## Critical: User-Only Extraction\n"
        "ONLY extract facts about **the user**. The user is the person speaking as `user:` "
        "in the transcript (rewritten as `the user`).\n"
        "Do NOT extract facts about third parties (math problem characters, named speakers, "
        "hypothetical people, anyone who is not the user).\n"
        "Every memory must have `the user` as its subject.\n"
        "When in doubt, OMIT. It is correct for a chunk to yield {\"memories\": []}.\n"
    )
    messages_list = []
    for chunk in chunks:
        user_prompt = chunk  # unified template expects observation in the user message
        messages_list.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

    print(f"  抽取: {len(messages_list)} 块 → {model_name}...")
    t0 = time.time()
    client = load_api_chat_completion(model_name, async_=True)
    responses = asyncio.run(
        client.get_response_chat(
            messages_list, max_new_tokens=512, temperature=0.0,
            max_concurrency=max_concurrency, use_tqdm=True,
        )
    )
    print(f"  完成: {time.time()-t0:.1f}s")

    all_memories = []
    parse_ok = 0
    parse_fail = 0
    for i, resp in enumerate(responses):
        mems = []
        if resp:
            try:
                resp_clean = resp.strip()
                if resp_clean.startswith("```"):
                    resp_clean = resp_clean.split("\n", 1)[-1].rsplit("```", 1)[0]
                obj = json.loads(resp_clean)
                mems = obj.get("memories", [])
                if isinstance(mems, list):
                    parse_ok += 1
                else:
                    parse_fail += 1
                    continue
            except (json.JSONDecodeError, AttributeError):
                parse_fail += 1
                continue

        for mem in mems:
            if isinstance(mem, str) and mem.strip():
                text = mem.strip()
                # 只保留以 the user 开头的记忆
                if text.lower().startswith("the user"):
                    all_memories.append({
                        "persona_id": persona_id,
                        "chunk_idx": i,
                        "aspect": aspect,
                        "text": text,
                    })

    print(f"  抽取到 {len(all_memories)} 条记忆 (parse OK: {parse_ok}, fail: {parse_fail})")
    return all_memories


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Extract atomic memories from PersonaMem chat_history")
    ap.add_argument("--chat-dir", default="data/raw_data/PersonaMem-v2/data/chat_history_128k")
    ap.add_argument("--output", default="relation_classifier/data_expansion/data/extracted_memories.jsonl")
    ap.add_argument("--model", default="gemma4-26B")
    ap.add_argument("--concurrency", type=int, default=30)
    ap.add_argument("--limit-personas", type=int, default=0, help="仅处理前 N 个 persona")
    ap.add_argument("--chunk-size", type=int, default=12)
    ap.add_argument("--aspects", default="events,preferences,social", help="逗号分隔的 aspect")
    args = ap.parse_args()

    aspects = [a.strip() for a in args.aspects.split(",")]

    # 解析路径
    personamem_root = os.path.join(_repo_root, "data/raw_data/PersonaMem-v2")
    chat_dir = args.chat_dir
    if not os.path.isabs(chat_dir):
        chat_dir = os.path.join(_repo_root, chat_dir)
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(_repo_root, output_path)

    chat_files = sorted(glob.glob(os.path.join(chat_dir, "*.json")))
    if args.limit_personas > 0:
        chat_files = chat_files[:args.limit_personas]

    # 按 persona 分组（一个 persona 可能有多个 chat_history 文件）
    persona_files = {}
    for fp in chat_files:
        data = json.load(open(fp, encoding="utf-8"))
        pid = str(data.get("metadata", {}).get("persona_id", os.path.basename(fp)))
        persona_files.setdefault(pid, []).append(fp)

    persona_list = list(persona_files.items())
    if args.limit_personas > 0:
        persona_list = persona_list[:args.limit_personas]

    print(f"处理 {len(persona_list)} 个 persona...")

    all_memories = []

    # 批量准备全部 extraction 请求，跨 persona 并行
    from utils.llm_api import load_api_chat_completion
    from prompts import render_prompt

    all_requests = []  # (persona_id, aspect, chunk_idx, message)
    for pid, files in persona_list:
        all_chunks = []
        for fp in files:
            data = json.load(open(fp, encoding="utf-8"))
            messages = data.get("chat_history", [])
            all_chunks.extend(chunk_messages(messages, args.chunk_size))

        for aspect in aspects:
            template = ASPECT_TEMPLATES.get(aspect, ASPECT_TEMPLATES["events"])
            system_prompt = render_prompt(template)
            system_prompt += (
                "\n\n## Critical: User-Only Extraction\n"
                "ONLY extract facts about **the user**. The user is the person speaking as `user:` "
                "in the transcript (rewritten as `the user`).\n"
                "Do NOT extract facts about third parties (math problem characters, named speakers, "
                "hypothetical people, anyone who is not the user).\n"
                "Every memory must have `the user` as its subject.\n"
                "When in doubt, OMIT. It is correct to yield {\"memories\": []}.\n"
            )
            for ci, chunk in enumerate(all_chunks):
                all_requests.append({
                    "persona_id": pid,
                    "aspect": aspect,
                    "chunk_idx": ci,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": chunk},
                    ],
                })

    print(f"总请求: {len(all_requests)} 条 ({len(persona_list)} personas × {len(aspects)} aspects)")
    t0 = time.time()

    client = load_api_chat_completion(args.model, async_=True)
    responses = asyncio.run(
        client.get_response_chat(
            [r["messages"] for r in all_requests],
            max_new_tokens=512, temperature=0.0,
            max_concurrency=args.concurrency,
            use_tqdm=True,
        )
    )

    elapsed = time.time() - t0
    print(f"抽取完成: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    for req, resp in zip(all_requests, responses):
        if not resp:
            continue
        mems = []
        try:
            resp_clean = resp.strip()
            if resp_clean.startswith("```"):
                resp_clean = resp_clean.split("\n", 1)[-1].rsplit("```", 1)[0]
            obj = json.loads(resp_clean)
            mems = obj.get("memories", [])
        except (json.JSONDecodeError, AttributeError):
            pass

        for mem in mems:
            if isinstance(mem, str) and mem.strip():
                text = mem.strip()
                if text.lower().startswith("the user"):
                    all_memories.append({
                        "persona_id": req["persona_id"],
                        "aspect": req["aspect"],
                        "text": text,
                    })

    # 去重 (相同 text)
    seen = set()
    deduped = []
    for m in all_memories:
        key = m["text"].strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(m)

    print(f"\n总计: {len(all_memories)} → 去重后 {len(deduped)} 条")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for m in deduped:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"输出 → {output_path}")


if __name__ == "__main__":
    main()
