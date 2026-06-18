"""Topic-tagging pass: 给已抽好的 candidate JSON 的每条 atomic fact 赋一个预定义主题。

抽取(extract_candidates.py)已产出 ``candidate_memories``（每 chunk 若干 atomic 字符串）。
本 pass 不重跑抽取，只对**每个 chunk 内的若干 fact 批量调用一次 LLM**，从 ``topics.TOPIC_TAXONOMY``
的固定枚举里各选一个 slug，写成与 ``candidate_memories`` 等长的平行数组 ``candidate_topics``
（与 ``cas_update_rules`` 同样的透传模式，下游 apply.py 按下标对齐取用）。

预定义枚举为通用生活/工作维度，不含 benchmark 专有 slot 名，避免 leakage 质疑。
打标失败 / 越界 / 缺项一律落 ``misc``（不参与灌库期聚合），保证稳健。

输入输出就地原文件改写：读 ``--candidates-dir`` 下每个 episode JSON，加 ``candidate_topics``
字段后写回。``misc`` 兜底意味着即使全程失败，灌库行为也退化为「不聚合」，与现状等价。

用法::

    uv run --no-sync python -m pipeline.tag_candidate_topics \
        --candidates-dir MemDB/candidates/meme_filler32k_gemma4-26B_0615_unified \
        --model gemma4-26B [--limit N] [--episode-concurrency 8] [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from memory.candidate_ingest.cas_update import candidate_memory_display_text
from memory.candidate_ingest.topics import (
    MISC_TOPIC,
    normalize_topic,
    topic_menu_lines,
)
from prompts import render_prompt
from utils.env import load_env
from utils.llm_api import load_api_chat_completion

TAG_TEMPLATE = "topic_tag_en.jinja"


def _parse_labels(text: Optional[str], n: int) -> List[str]:
    """LLM 返回 → 长度 n 的 slug 列表；任何缺失/越界/解析失败处填 ``misc``。"""
    out = [MISC_TOPIC] * n
    if not text or not str(text).strip():
        return out
    t = str(text).strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", t)
    if fence:
        t = fence.group(1).strip()
    obj: Any = None
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        i, j = t.find("{"), t.rfind("}")
        if i != -1 and j > i:
            try:
                obj = json.loads(t[i : j + 1])
            except json.JSONDecodeError:
                obj = None
    if not isinstance(obj, dict):
        return out
    labels = obj.get("labels", obj)  # 容忍模型直接吐 {"0": "..."} 不带 labels 包裹
    if not isinstance(labels, dict):
        return out
    for k, v in labels.items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < n:
            out[idx] = normalize_topic(v)
    return out


def _tag_one_chunk(client: Any, facts: List[str], max_new_tokens: int) -> List[str]:
    if not facts:
        return []
    facts_block = "\n".join(f"{i}. {f}" for i, f in enumerate(facts))
    prompt = render_prompt(
        TAG_TEMPLATE,
        topic_menu="\n".join(topic_menu_lines()),
        facts_block=facts_block,
    )
    raw = client.get_response_chat(
        [{"role": "user", "content": prompt}],
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        verbose=False,
    )
    text = raw[0] if isinstance(raw, (list, tuple)) and raw else raw
    return _parse_labels(text, len(facts))


def tag_episode_payload(
    client: Any,
    payload: Dict[str, Any],
    max_new_tokens: int,
    executor: Optional[ThreadPoolExecutor] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """就地给 payload 的每个 chunk 加 ``candidate_topics`` 平行数组。"""
    chunks = payload.get("chunks") or []
    if not isinstance(chunks, list):
        return payload

    # 收集需要打标的 (chunk_idx_in_list, fact_texts)
    work: List[tuple[int, List[str]]] = []
    for li, ch in enumerate(chunks):
        if not isinstance(ch, dict):
            continue
        if ch.get("candidate_topics") is not None and not overwrite:
            continue
        mems = ch.get("candidate_memories") or []
        texts = [candidate_memory_display_text(m) for m in mems] if isinstance(mems, list) else []
        if not texts:
            ch["candidate_topics"] = []
            continue
        work.append((li, texts))

    if not work:
        return payload

    if executor is not None:
        futs = {li: executor.submit(_tag_one_chunk, client, texts, max_new_tokens) for li, texts in work}
        for li, fut in futs.items():
            chunks[li]["candidate_topics"] = fut.result()
    else:
        for li, texts in work:
            chunks[li]["candidate_topics"] = _tag_one_chunk(client, texts, max_new_tokens)

    return payload


def _iter_episode_files(candidates_dir: Path) -> List[Path]:
    return sorted(p for p in candidates_dir.glob("*.json") if p.name != "extract_progress.state")


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(
        description="给已抽好的 candidate JSON 每条 fact 打预定义主题标签（平行数组 candidate_topics）。"
    )
    parser.add_argument("--candidates-dir", required=True, help="extract_candidates 的输出根目录")
    parser.add_argument("--model", required=True, help="打标用 LLM（与 llm_api 模型名规则一致）")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 个 episode（0=全部）")
    parser.add_argument(
        "--chunk-concurrency", type=int, default=40,
        help="单 episode 内 chunk 打标并发（episode 之间串行）",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="已有 candidate_topics 也重新打标（默认跳过已标记的 chunk）",
    )
    args = parser.parse_args()

    cdir = Path(args.candidates_dir)
    if not cdir.is_dir():
        parser.error(f"--candidates-dir not found: {cdir}")

    files = _iter_episode_files(cdir)
    if args.limit and args.limit > 0:
        files = files[: args.limit]
    if not files:
        print(f"No episode JSON under {cdir}", flush=True)
        return

    client = load_api_chat_completion(args.model, async_=False)
    print(f"[tag_candidate_topics] {len(files)} episode(s) under {cdir} model={args.model}", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, int(args.chunk_concurrency))) as ex:
        for path in tqdm(files, desc="tag topics", unit="episode", dynamic_ncols=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                print(f"  skip {path.name}: {e}", flush=True)
                continue
            if not isinstance(payload, dict):
                continue
            tagged = tag_episode_payload(
                client, payload, args.max_new_tokens, executor=ex, overwrite=args.overwrite
            )
            body = json.dumps(tagged, ensure_ascii=False, indent=2)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(body, encoding="utf-8")
            tmp.replace(path)


if __name__ == "__main__":
    main()
