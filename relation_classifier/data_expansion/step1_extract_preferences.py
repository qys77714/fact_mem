"""Step 1: 从 PersonaMem-V2 提取原子记忆 + 主语改写为 'the user' 形式。"""

import json
import glob
import re
import os
import sys
from typing import List


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


def extract_personamem_preferences(persona_dir: str, output_path: str) -> List[dict]:
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
