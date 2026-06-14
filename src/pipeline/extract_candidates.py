"""
使用 Jinja 模板从 benchmark 数据抽取 candidate memory（原子块 + 每块若干候选句）。
默认 ``0_mem_extract_v2.jinja``（中文 ``0_mem_extract_v2_zh.jinja``，由 ``--language`` 决定，见 ``_resolve_mem_extract_prompt_template``）。
可用 ``--mem-extract-template`` 指定文件名覆盖上述规则（置于 ``src/prompts/templates/``）。
两种互斥模式：（1）默认仅使用主模板 ``--mem-extract-template``（或 benchmark+language 解析结果），不传方面模板即可；
（2）``--mem-extract-aspects-only`` 且 ``--mem-extract-extra-template``（1～3 次）：每块**仅**并行调用这些方面模板，按顺序合并 ``candidate_memories`` 并精确去重，主模板不参与 LLM。

默认输出目录：``MemDB/candidates/{benchmark}/{extract_model_tag}/{suffix}/``（可用 ``--output`` 覆盖）。

MEME benchmark 的 evidence session：``gold_facts`` 不经 LLM，默认拆分为 ``candidate_memories``（primary）
+ 并行 ``cas_update_rules``（无规则时为 null）。

续跑：在输出根目录维护单一 ``extract_progress.state``（JSON），记录已完成 ``history_name``；
判定是否跳过**仅**看 ``completed`` 是否包含该 episode，不比对 ``config`` 指纹（``config`` 仅作留痕写入）。
未在 ``completed`` 中的 episode 一律重新抽取（不因磁盘上已有同名 JSON 而跳过）。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from tqdm import tqdm

from benchmark import get_benchmark
from benchmark.base import MemoryEpisode
from benchmark.datasets import DEFAULT_BENCHMARK_DATASETS, resolve_benchmark_data_path
from memory.candidate_ingest.cas_update import build_evidence_gold_chunk_fields
from pipeline.paths import default_candidates_dir, safe_model_tag
from prompts import render_prompt
from utils.env import load_env
from utils.llm_api import load_api_chat_completion
from utils.mem_extract_schemas import MEM_EXTRACT_RESPONSE_FORMAT, MemExtractResponse
from utils.question_filter import parse_question_types_arg


def _resolve_mem_extract_prompt_template(benchmark: str, language: str = "en") -> str:
    """按 benchmark 与提示语语言选择模板。"""
    lang = (language or "en").strip().lower()
    use_zh = lang == "zh" or lang.startswith("zh-")
    return "0_mem_extract_v2_zh.jinja" if use_zh else "0_mem_extract_v2.jinja"


EXTRACT_PROGRESS_VERSION = 5
EXTRACT_PROGRESS_KIND = "lme_candidate_extract_progress"
PROGRESS_FILENAME = "extract_progress.state"

_UNSAFE_NAME_CHARS = frozenset(r'\/:*?"<>|\n\r\t')


def _normalize_memory_granularity(value: str) -> Union[str, int]:
    """与 ``pipeline_lme_generate`` 的 ``--memory-granularity`` 一致：``all`` 或正整数。"""
    v = str(value).strip().lower()
    if v == "all":
        return "all"
    if v.isdigit() and int(v) > 0:
        return int(v)
    raise ValueError("--memory-granularity must be 'all' or a positive integer (e.g., 1/2/3/4).")


def _normalize_turn_overlap(value: str, memory_granularity: Union[str, int]) -> int:
    """相邻 turn 窗口之间重叠的 turn 数；``memory-granularity=all`` 时强制为 0。"""
    if memory_granularity == "all":
        return 0
    s = str(value).strip()
    if not s.isdigit():
        raise ValueError("--turn-overlap must be a non-negative integer.")
    overlap = int(s)
    chunk_size = int(memory_granularity)
    if overlap >= chunk_size:
        raise ValueError(
            f"--turn-overlap ({overlap}) must be < --memory-granularity ({chunk_size}) "
            "so each step advances by at least one turn."
        )
    return overlap


def _safe_history_basename(history_name: str, max_len: int = 200) -> str:
    """文件名安全、与 history_name 一一对应（供路径使用）。"""
    s = str(history_name).strip()
    if not s:
        s = "episode"
    s = "".join("_" if c in _UNSAFE_NAME_CHARS else c for c in s)
    s = s.strip(" .") or "episode"
    if len(s) > max_len:
        s = s[:max_len]
    return s


def _iter_turn_chunks(
    turns: list,
    granularity: Union[str, int],
    overlap_turns: int = 0,
) -> List[Tuple[Optional[int], Optional[int], list]]:
    """在单个 session 内按 turn 切块。``overlap_turns>0`` 时为滑动窗口（步长 ``granularity - overlap_turns``）。"""
    if granularity == "all":
        return [(None, None, turns)]

    chunk_size = int(granularity)
    stride = chunk_size - int(overlap_turns)
    if stride <= 0:
        raise ValueError("internal: stride must be positive (check overlap vs granularity).")

    chunks: List[Tuple[Optional[int], Optional[int], list]] = []
    i = 0
    while i < len(turns):
        chunk_turns = turns[i : i + chunk_size]
        if chunk_turns:
            chunks.append((i, i + len(chunk_turns) - 1, chunk_turns))
        i += stride
    return chunks


def _turns_to_chunk_transcript(turns: list, dialogue_format: str) -> str:
    """Turn 列表转写为单段文本（user_assistant / named_speakers，与灌库切块一致）。"""
    lines: List[str] = []
    df = (dialogue_format or "user_assistant").strip().lower()
    for turn in turns:
        content = (turn.content or "").strip()
        if not content:
            continue
        if df == "named_speakers":
            speaker = (turn.speaker or "Unknown").strip()
            lines.append(f"{speaker}: {content}")
            continue
        role = (turn.speaker or "").lower()
        if role not in ("user", "assistant"):
            role = "user" if role in ("human", "人") else "assistant"
        lines.append(f"**{role}**: {content}")
    return "\n\n".join(lines)


def episode_to_observation_chunks(
    episode: MemoryEpisode,
    granularity: Union[str, int],
    dialogue_format: str = "user_assistant",
    *,
    overlap_turns: int = 0,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    与 candidate ingest 灌库时一致：按 session 遍历，每个 session 内再按 ``granularity`` 切块；
    每一块对应一次抽取用的 observation 文本。

    ``overlap_turns``：相邻窗口共享的 turn 数（滑动窗口）；0 表示互不重叠（与原行为一致）。

    返回 ``(observation_text, meta)`` 列表，``meta`` 含全局 ``chunk_index``（按 session→turn 块顺序递增）。
    """
    chunks_out: List[Tuple[str, Dict[str, Any]]] = []
    chunk_index = 0
    for sess_idx, session in enumerate(episode.sessions, start=1):
        st = (session.session_date or "").strip()
        for turn_start, turn_end, chunk_turns in _iter_turn_chunks(
            session.turns, granularity, overlap_turns=overlap_turns
        ):
            parts = [f"=== Session {sess_idx}"]
            if st:
                parts[0] += f" ({st})"
            if turn_start is not None:
                parts.append(f"turns {turn_start}-{turn_end}")
            header = " ".join(parts) + " ==="
            transcript = _turns_to_chunk_transcript(chunk_turns, dialogue_format)
            text = f"{header}\n{transcript}"
            meta: Dict[str, Any] = {
                "chunk_index": chunk_index,
                "session_index": sess_idx,
                "turn_start": turn_start,
                "turn_end": turn_end,
                "turn_overlap": int(overlap_turns),
                "session_date": st,
            }
            chunks_out.append((text, meta))
            chunk_index += 1
    return chunks_out


def _first_pending_observation_sample(
    pending: List[MemoryEpisode],
    memory_granularity: Union[str, int],
    dialogue_format: str,
    overlap_turns: int = 0,
) -> Optional[str]:
    """首个待抽取 episode 的第一块 observation（与将送入 LLM 的 user message 中 [[ observation ]] 一致）。"""
    for ep in pending:
        chunks = episode_to_observation_chunks(
            ep, memory_granularity, dialogue_format, overlap_turns=overlap_turns
        )
        if chunks:
            return chunks[0][0]
    return None


def _parse_json_from_llm(text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not text or not str(text).strip():
        return None
    t = str(text).strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", t)
    if fence:
        t = fence.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    i = t.find("{")
    j = t.rfind("}")
    if i != -1 and j > i:
        try:
            return json.loads(t[i : j + 1])
        except json.JSONDecodeError:
            return None
    return None


def _episode_out_path(out_root: Path, episode: MemoryEpisode) -> Path:
    base = _safe_history_basename(episode.history_name)
    return out_root / f"{base}.json"


def _extract_config_dict(
    args: argparse.Namespace,
    memory_granularity: Union[str, int],
    prompt_template: str,
    turn_overlap: int,
    mem_extract_extra_templates: List[str],
) -> Dict[str, Any]:
    gran_s = "all" if memory_granularity == "all" else str(int(memory_granularity))
    cfg: Dict[str, Any] = {
        "model": args.model,
        "memory_granularity": gran_s,
        "turn_overlap": int(turn_overlap),
        "dialogue_format": args.dialogue_format,
        "prompt_template": prompt_template,
        "mem_extract_extra_templates": list(mem_extract_extra_templates),
        "mem_extract_aspects_only": bool(getattr(args, "mem_extract_aspects_only", False)),
        "use_json_schema": not args.no_json_schema,
        "max_new_tokens": args.max_new_tokens,
    }
    return cfg


def _progress_state_path(out_root: Path) -> Path:
    return out_root / PROGRESS_FILENAME


def _read_json_dict(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _load_completed_set(out_root: Path) -> set[str]:
    """读取 ``extract_progress.state`` 中的 ``completed`` 集合。

    判定是否跳过**仅**看 ``completed`` 列表是否包含该 ``history_name``；
    不再比对 ``config`` 指纹（``model`` / ``max_new_tokens`` / ``prompt_template`` 等任一字段变化都不再触发全量重抽）。仅保留 ``kind`` 白名单校验，
    避免误读到其它类型的 state 文件。
    """
    data = _read_json_dict(_progress_state_path(out_root))
    if not data:
        return set()
    if str(data.get("kind", "")) != EXTRACT_PROGRESS_KIND:
        return set()
    comp = data.get("completed")
    if not isinstance(comp, list):
        return set()
    return {str(x) for x in comp}


def _write_extract_progress_atomic(
    out_root: Path,
    args: argparse.Namespace,
    memory_granularity: Union[str, int],
    completed: set[str],
    prompt_template: str,
    turn_overlap: int,
) -> None:
    path = _progress_state_path(out_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": EXTRACT_PROGRESS_VERSION,
        "kind": EXTRACT_PROGRESS_KIND,
        "config": _extract_config_dict(
            args,
            memory_granularity,
            prompt_template,
            turn_overlap,
            getattr(args, "mem_extract_extra_templates", []) or [],
        ),
        "completed": sorted(completed),
    }
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)


def _merge_ordered_candidate_memories(memory_lists: List[List[Any]]) -> List[str]:
    """按列表顺序拼接多条抽取结果，对 strip 后的字符串精确去重（保留首次出现顺序）。"""
    seen: set[str] = set()
    out: List[str] = []
    for mems in memory_lists:
        if not isinstance(mems, list):
            continue
        for m in mems:
            raw = m if isinstance(m, str) else str(m)
            s = raw.strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
    return out


def _merge_chunk_pass_rows(meta: Dict[str, Any], pass_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """将同一 observation 的多次 LLM 抽取合并为单行 chunk 记录。"""
    mem_lists = [r.get("candidate_memories") or [] for r in pass_rows]
    merged = _merge_ordered_candidate_memories(mem_lists)
    errs: List[str] = []
    for i, r in enumerate(pass_rows):
        e = r.get("parse_error")
        if e:
            errs.append(f"pass{i}:{e}")
    parse_err: Optional[str] = "; ".join(errs) if errs else None
    row: Dict[str, Any] = {
        "chunk_index": meta["chunk_index"],
        "session_index": meta["session_index"],
        "turn_start": meta["turn_start"],
        "turn_end": meta["turn_end"],
        "turn_overlap": meta.get("turn_overlap", 0),
        "session_date": meta.get("session_date", ""),
        "candidate_memories": merged,
        "parse_error": parse_err,
    }
    return row


def _cleanup_stale_tmp(out_json: Path) -> None:
    tmp = out_json.with_suffix(out_json.suffix + ".tmp")
    if tmp.is_file():
        try:
            tmp.unlink()
        except OSError:
            pass


def _process_single_chunk(
    client: Any,
    observation: str,
    meta: Dict[str, Any],
    prompt_template: str,
    max_new_tokens: int,
    use_json_schema: bool,
    pbar: Optional[Any] = None,
) -> Dict[str, Any]:
    user_prompt = render_prompt(prompt_template, observation=observation)
    messages = [{"role": "user", "content": user_prompt}]
    chat_kw: Dict[str, Any] = {}
    if use_json_schema:
        chat_kw["response_format"] = MEM_EXTRACT_RESPONSE_FORMAT
    raw = client.get_response_chat(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        verbose=False,
        **chat_kw,
    )
    if pbar is not None:
        pbar.update(1)
    memories: List[Any] = []
    err: Optional[str] = None
    if raw is None:
        err = "llm_api_failed"
    else:
        parsed = _parse_json_from_llm(raw)
        if parsed is None:
            err = "json_parse_failed_or_empty"
        else:
            try:
                validated = MemExtractResponse.model_validate(parsed)
                memories = list(validated.memories)
            except Exception as e:
                err = f"pydantic_validate_failed:{e!s}"
    row: Dict[str, Any] = {
        "chunk_index": meta["chunk_index"],
        "session_index": meta["session_index"],
        "turn_start": meta["turn_start"],
        "turn_end": meta["turn_end"],
        "turn_overlap": meta.get("turn_overlap", 0),
        "session_date": meta.get("session_date", ""),
        "candidate_memories": memories,
        "parse_error": err,
    }
    return row


def _count_meme_llm_chunks(
    episode: MemoryEpisode,
    memory_granularity: Union[str, int],
    overlap_turns: int,
    dialogue_format: str,
) -> int:
    """Count LLM chunks for a MEME episode: only filler sessions need LLM extraction."""
    count = 0
    for session in episode.sessions:
        if session.metadata.get("type") == "evidence":
            continue
        count += len(_iter_turn_chunks(session.turns, memory_granularity, overlap_turns))
    return count


def _run_meme_episode_extract(
    client: Any,
    episode: MemoryEpisode,
    model: str,
    max_new_tokens: int,
    memory_granularity: Union[str, int],
    dialogue_format: str,
    prompt_template: str,
    templates_order: List[str],
    use_json_schema: bool = True,
    pbar: Optional[Any] = None,
    executor: Optional[ThreadPoolExecutor] = None,
    overlap_turns: int = 0,
) -> Dict[str, Any]:
    """
    MEME-specific extraction:
    - evidence sessions → gold_facts split to primary_text + cas_update_condition (no LLM)
    - filler sessions   → LLM extraction via _process_single_chunk (same as lme_s)

    Maintains a unified sequential chunk_index across all session types.
    """
    gran_s = "all" if memory_granularity == "all" else str(int(memory_granularity))
    hn = str(episode.history_name)

    header: Dict[str, Any] = {
        "history_name": hn,
        "model": model,
        "memory_granularity": gran_s,
        "turn_overlap": int(overlap_turns),
        "dialogue_format": dialogue_format,
    }

    chunk_index = 0
    chunk_rows: List[Dict[str, Any]] = []
    # Collect filler work for (possibly parallel) LLM calls: (obs, meta)
    filler_items: List[Tuple[str, Dict[str, Any]]] = []

    for sess_idx, session in enumerate(episode.sessions, start=1):
        st = (session.session_date or "").strip()
        sess_type = session.metadata.get("type", "filler")

        if sess_type == "evidence":
            # Direct: gold_facts → primary + cas_update_rules (no LLM)
            gold_facts: List[Dict[str, Any]] = session.metadata.get("gold_facts") or []
            fact_texts = [
                str(f.get("fact_text") or "").strip()
                for f in gold_facts
                if isinstance(f, dict) and f.get("fact_text")
            ]
            gold_fields = build_evidence_gold_chunk_fields(fact_texts)
            chunk_rows.append({
                "chunk_index": chunk_index,
                "session_index": sess_idx,
                "turn_start": None,
                "turn_end": None,
                "turn_overlap": 0,
                "session_date": st,
                "parse_error": None,
                "source": "evidence_gold_facts",
                **gold_fields,
            })
            chunk_index += 1
        else:
            # Filler: normal turn-level chunking → deferred LLM extraction
            for turn_start, turn_end, chunk_turns in _iter_turn_chunks(
                session.turns, memory_granularity, overlap_turns
            ):
                parts = [f"=== Session {sess_idx}"]
                if st:
                    parts[0] += f" ({st})"
                if turn_start is not None:
                    parts.append(f"turns {turn_start}-{turn_end}")
                header_str = " ".join(parts) + " ==="
                transcript = _turns_to_chunk_transcript(chunk_turns, dialogue_format)
                observation = f"{header_str}\n{transcript}"
                meta: Dict[str, Any] = {
                    "chunk_index": chunk_index,
                    "session_index": sess_idx,
                    "turn_start": turn_start,
                    "turn_end": turn_end,
                    "turn_overlap": int(overlap_turns),
                    "session_date": st,
                }
                filler_items.append((observation, meta))
                chunk_index += 1

    # --- LLM extraction for filler chunks ---
    filler_rows: Dict[int, Dict[str, Any]] = {}

    def _chunk_passes_seq(observation: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        pass_rows: List[Dict[str, Any]] = []
        for ti, tpl in enumerate(templates_order):
            pass_rows.append(
                _process_single_chunk(
                    client, observation, meta, tpl,
                    max_new_tokens, use_json_schema,
                    pbar,
                )
            )
        return _merge_chunk_pass_rows(meta, pass_rows)

    if filler_items:
        if executor is not None:
            indexed: List[Tuple[int, int, Any]] = []
            for observation, meta in filler_items:
                ci = int(meta["chunk_index"])
                for ti, tpl in enumerate(templates_order):
                    indexed.append((
                        ci, ti,
                        executor.submit(
                            _process_single_chunk,
                            client, observation, meta, tpl,
                            max_new_tokens, use_json_schema,
                            pbar,
                        ),
                    ))
            by_ci: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
            for ci, ti, fut in indexed:
                by_ci[ci][ti] = fut.result()
            for observation, meta in filler_items:
                ci = int(meta["chunk_index"])
                pairs = sorted(by_ci[ci].items(), key=lambda x: x[0])
                pass_rows_m = [pr for _, pr in pairs]
                filler_rows[ci] = _merge_chunk_pass_rows(meta, pass_rows_m)
        else:
            for observation, meta in filler_items:
                ci = int(meta["chunk_index"])
                filler_rows[ci] = _chunk_passes_seq(observation, meta)

    # Merge evidence + filler rows in chunk_index order
    for row in chunk_rows:
        ci = row["chunk_index"]
        if ci in filler_rows:
            # Shouldn't happen, but prefer filler_rows (LLM result) if collision
            chunk_rows[chunk_rows.index(row)] = filler_rows.pop(ci)
    # Append any filler rows not already in chunk_rows (all filler chunks)
    for ci in sorted(filler_rows):
        chunk_rows.append(filler_rows[ci])
    chunk_rows.sort(key=lambda r: r["chunk_index"])

    header["chunks"] = chunk_rows
    return header


def _run_episode_extract(
    client: Any,
    episode: MemoryEpisode,
    model: str,
    max_new_tokens: int,
    memory_granularity: Union[str, int],
    dialogue_format: str,
    prompt_template: str,
    mem_extract_extra_templates: Optional[List[str]] = None,
    mem_extract_aspects_only: bool = False,
    use_json_schema: bool = True,
    pbar: Optional[Any] = None,
    executor: Optional[ThreadPoolExecutor] = None,
    overlap_turns: int = 0,
    benchmark: str = "",
) -> Dict[str, Any]:
    gran_s = "all" if memory_granularity == "all" else str(int(memory_granularity))
    hn = str(episode.history_name)
    extras = [str(x).strip() for x in (mem_extract_extra_templates or []) if str(x).strip()]
    if mem_extract_aspects_only:
        templates_order = list(extras)
    else:
        templates_order = [prompt_template]

    # MEME benchmark: dispatch to hybrid extractor (evidence→gold_facts, filler→LLM)
    if benchmark.lower().startswith("meme"):
        return _run_meme_episode_extract(
            client=client,
            episode=episode,
            model=model,
            max_new_tokens=max_new_tokens,
            memory_granularity=memory_granularity,
            dialogue_format=dialogue_format,
            prompt_template=prompt_template,
            templates_order=templates_order,
            use_json_schema=use_json_schema,
            pbar=pbar,
            executor=executor,
            overlap_turns=overlap_turns,
        )

    header: Dict[str, Any] = {
        "history_name": hn,
        "model": model,
        "memory_granularity": gran_s,
        "turn_overlap": int(overlap_turns),
        "dialogue_format": dialogue_format,
    }

    obs_chunks = episode_to_observation_chunks(
        episode, memory_granularity, dialogue_format, overlap_turns=overlap_turns
    )
    if not obs_chunks:
        header["chunks"] = []
        header["episode_parse_error"] = "empty_episode_no_observation_chunks"
        return header

    chunk_rows: List[Dict[str, Any]] = []

    def _chunk_passes_sequential(observation: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        pass_rows: List[Dict[str, Any]] = []
        for ti, tpl in enumerate(templates_order):
            pass_rows.append(
                _process_single_chunk(
                    client,
                    observation,
                    meta,
                    tpl,
                    max_new_tokens,
                    use_json_schema,
                    pbar,
                )
            )
        return _merge_chunk_pass_rows(meta, pass_rows)

    if executor is not None:
        indexed: List[Tuple[int, int, Any]] = []
        for observation, meta in obs_chunks:
            ci = int(meta["chunk_index"])
            for ti, tpl in enumerate(templates_order):
                indexed.append(
                    (
                        ci,
                        ti,
                        executor.submit(
                            _process_single_chunk,
                            client,
                            observation,
                            meta,
                            tpl,
                            max_new_tokens,
                            use_json_schema,
                            pbar,
                        ),
                    )
                )
        by_ci: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        for ci, ti, fut in indexed:
            by_ci[ci][ti] = fut.result()
        for observation, meta in obs_chunks:
            ci = int(meta["chunk_index"])
            pairs = sorted(by_ci[ci].items(), key=lambda x: x[0])
            pass_rows = [pr for _, pr in pairs]
            chunk_rows.append(_merge_chunk_pass_rows(meta, pass_rows))
    else:
        for observation, meta in obs_chunks:
            chunk_rows.append(_chunk_passes_sequential(observation, meta))

    header["chunks"] = chunk_rows
    return header


def _write_episode_file(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    load_env()
    ds_keys = ", ".join(sorted(DEFAULT_BENCHMARK_DATASETS.keys()))
    parser = argparse.ArgumentParser(
        description="记忆抽取：默认仅主模板；或 --mem-extract-aspects-only + --mem-extract-extra-template（≤3）仅用方面模板。"
    )
    parser.add_argument(
        "--benchmark",
        default="lme_o",
        help=f"内置数据键: {ds_keys}；亦用于默认输出路径",
    )
    parser.add_argument("--benchmark-file", default=None, help="覆盖：直接指定预处理后的 JSON 路径")
    parser.add_argument(
        "--output",
        default=None,
        help="输出根目录（每 episode 一个 json）。默认：MemDB/candidates/{benchmark}/{extract_model}/{suffix}/",
    )
    parser.add_argument(
        "--suffix",
        default="default",
        help="默认 MemDB 布局下子目录名（如 ku、gran4）",
    )
    parser.add_argument(
        "--model",
        "--extract-model",
        dest="model",
        required=True,
        help="抽取用 LLM（与 pipeline_lme_generate 模型名规则一致）",
    )
    parser.add_argument(
        "--question-types",
        default="",
        help="",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument(
        "--memory-granularity",
        default="4",
        help="与 pipeline_lme_generate 相同：'all' 或每 N 个 turn 一块（在单个 session 内切块）",
    )
    parser.add_argument(
        "--turn-overlap",
        default="0",
        help="滑动窗口：相邻块共享的 turn 数（0=互不重叠）。须小于 memory-granularity；all 时忽略。",
    )
    parser.add_argument(
        "--dialogue-format",
        default="user_assistant",
        choices=["user_assistant", "named_speakers"],
        help="与灌库时的 chunk 转写一致（user_assistant / named_speakers）",
    )
    parser.add_argument(
        "--chunk-concurrency",
        type=int,
        default=40,
        help="单个 episode 内抽取 chunk 的并发量（episode 之间串行处理）",
    )
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 条（0 表示不限制）")
    parser.add_argument(
        "--no-json-schema",
        action="store_true",
        help="不向 API 传 response_format（仍用 Pydantic 校验返回 JSON）",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="记忆抽取提示模板语言：en / zh（zh 时用 0_mem_extract_v2_zh.jinja，否则 0_mem_extract_v2.jinja）",
    )
    parser.add_argument(
        "--mem-extract-template",
        default="",
        metavar="NAME.jinja",
        help="覆盖记忆抽取模板文件名（默认按 benchmark+language 选择；与 src/prompts/templates/ 下文件名一致）",
    )
    parser.add_argument(
        "--mem-extract-extra-template",
        action="append",
        default=None,
        dest="mem_extract_extra_templates",
        metavar="NAME.jinja",
        help="方面抽取模板，可重复至多 3 次；须与 --mem-extract-aspects-only 联用，按顺序合并去重",
    )
    parser.add_argument(
        "--mem-extract-aspects-only",
        action="store_true",
        help="仅运行方面模板（须至少一次 --mem-extract-extra-template）；不传则仅运行主模板（忽略方面模板）",
    )
    args = parser.parse_args()
    memory_granularity = _normalize_memory_granularity(args.memory_granularity)
    turn_overlap = _normalize_turn_overlap(args.turn_overlap, memory_granularity)
    tpl_override = (args.mem_extract_template or "").strip()
    prompt_template = (
        tpl_override if tpl_override else _resolve_mem_extract_prompt_template(args.benchmark, args.language)
    )
    extras_raw = getattr(args, "mem_extract_extra_templates", None) or []
    mem_extract_extra_templates = [str(x).strip() for x in extras_raw if str(x).strip()]
    if len(mem_extract_extra_templates) > 3:
        parser.error("--mem-extract-extra-template 至多使用 3 次")
    if args.mem_extract_aspects_only:
        if not mem_extract_extra_templates:
            parser.error("--mem-extract-aspects-only 需要至少一次 --mem-extract-extra-template")
    elif mem_extract_extra_templates:
        print(
            "[extract_candidates] 警告：已传入方面模板但未使用 --mem-extract-aspects-only，"
            "将仅使用主模板，方面模板已忽略。",
            flush=True,
        )
    args.mem_extract_extra_templates = mem_extract_extra_templates
    _templates_dir = Path(__file__).resolve().parent.parent / "prompts" / "templates"
    _template_abs = _templates_dir / prompt_template
    mode_line = (
        f"aspects_only templates ({len(mem_extract_extra_templates)}): {mem_extract_extra_templates!r}"
        if args.mem_extract_aspects_only
        else f"base_only → {prompt_template}"
    )
    print(
        f"[extract_candidates] benchmark={args.benchmark!r}  language={args.language!r}  "
        f"mode: {mode_line}\n"
        f"  primary template (metadata): {prompt_template}\n"
        f"  primary file: {_template_abs}",
        flush=True,
    )

    data_path, lang = resolve_benchmark_data_path(args.benchmark, args.benchmark_file)
    q_types = parse_question_types_arg(args.question_types)

    benchmark = get_benchmark(args.benchmark, file_path=data_path, lang=lang)

    tasks: List[MemoryEpisode] = []
    for episode in benchmark:
        if q_types is None:
            tasks.append(episode)
        elif any(q.question_type in q_types for q in episode.qas):
            tasks.append(episode)

    if args.limit and args.limit > 0:
        tasks = tasks[: args.limit]

    if args.output:
        out_root = Path(args.output)
    else:
        out_root = default_candidates_dir(
            benchmark=args.benchmark,
            extract_model=args.model,
            suffix=args.suffix,
        )

    completed = _load_completed_set(out_root)
    pending: List[MemoryEpisode] = []
    skipped = 0
    for ep in tasks:
        outp = _episode_out_path(out_root, ep)
        _cleanup_stale_tmp(outp)
        hn = str(ep.history_name)
        if hn in completed:
            skipped += 1
            continue
        pending.append(ep)

    n_tasks = len(tasks)
    if skipped == n_tasks and not pending:
        print(
            f"Skip extract: all {n_tasks} episode(s) up-to-date per {_progress_state_path(out_root)}.",
            flush=True,
        )
        return
    if skipped:
        print(
            f"Partial skip extract: {skipped} in progress file, "
            f"{len(pending)} episode(s) to run.",
            flush=True,
        )

    n_extract_passes = (
        len(mem_extract_extra_templates) if args.mem_extract_aspects_only else 1
    )
    _is_meme = args.benchmark.lower().startswith("meme")
    total_llm_chunks = 0
    for ep in pending:
        if _is_meme:
            # evidence sessions produce no LLM calls; only filler sessions count
            total_llm_chunks += n_extract_passes * _count_meme_llm_chunks(
                ep, memory_granularity, turn_overlap, args.dialogue_format
            )
        else:
            total_llm_chunks += n_extract_passes * len(
                episode_to_observation_chunks(
                    ep, memory_granularity, args.dialogue_format, overlap_turns=turn_overlap
                )
            )

    client = load_api_chat_completion(args.model, async_=False)
    n_pending = len(pending)

    sample_obs = _first_pending_observation_sample(
        pending, memory_granularity, args.dialogue_format, overlap_turns=turn_overlap
    )
    if sample_obs is not None:
        print(
            "[extract_candidates] prompt_template (Jinja file name): "
            f"{prompt_template}\n"
            "[extract_candidates] sample observation (printed once; first chunk text embedded in LLM user prompt):\n"
            f"{sample_obs}\n",
            flush=True,
        )

    _benchmark_name = args.benchmark  # string name, passed into work closure

    def work(
        ep: MemoryEpisode, chunk_pbar: Optional[Any], executor: Optional[ThreadPoolExecutor]
    ) -> Tuple[Path, Dict[str, Any]]:
        payload = _run_episode_extract(
            client,
            ep,
            args.model,
            args.max_new_tokens,
            memory_granularity,
            args.dialogue_format,
            prompt_template,
            mem_extract_extra_templates=mem_extract_extra_templates,
            mem_extract_aspects_only=args.mem_extract_aspects_only,
            use_json_schema=not args.no_json_schema,
            pbar=chunk_pbar,
            executor=executor,
            overlap_turns=turn_overlap,
            benchmark=_benchmark_name,
        )
        return _episode_out_path(out_root, ep), payload

    chunk_bar_fmt = (
        "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} chunks "
        "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
    )
    task_bar_fmt = (
        "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} tasks "
        "[{elapsed}<{remaining}, {rate_fmt}]"
    )

    with ThreadPoolExecutor(max_workers=max(1, int(args.chunk_concurrency))) as ex:
        if total_llm_chunks > 0:
            with tqdm(
                total=total_llm_chunks,
                desc="LLM extract",
                unit="chunk",
                dynamic_ncols=True,
                bar_format=chunk_bar_fmt,
                smoothing=0.05,
                miniters=1,
                mininterval=0.1,
            ) as chunk_pbar:
                chunk_pbar.set_postfix_str(f"tasks 0/{n_pending} model={safe_model_tag(args.model)}")
                done_tasks = 0
                for ep in pending:
                    path, payload = work(ep, chunk_pbar, ex)
                    _write_episode_file(path, payload)
                    completed.add(str(ep.history_name))
                    _write_extract_progress_atomic(
                        out_root,
                        args,
                        memory_granularity,
                        completed,
                        prompt_template,
                        turn_overlap,
                    )
                    done_tasks += 1
                    chunk_pbar.set_postfix_str(f"tasks {done_tasks}/{n_pending}")
        else:
            for ep in tqdm(
                pending,
                total=n_pending,
                desc="LLM extract (no obs chunks)",
                unit="task",
                dynamic_ncols=True,
                bar_format=task_bar_fmt,
            ):
                path, payload = work(ep, None, ex)
                _write_episode_file(path, payload)
                completed.add(str(ep.history_name))
                _write_extract_progress_atomic(
                    out_root,
                    args,
                    memory_granularity,
                    completed,
                    prompt_template,
                    turn_overlap,
                )


if __name__ == "__main__":
    main()
