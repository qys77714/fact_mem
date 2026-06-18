"""Apply LME candidate JSON → relation_decision（桶内聚合 + EQUIV/ATTACH/UPDATE 弱边，无物理 DELETE）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cas_update import (
    candidate_memory_display_text,
    merge_cas_rule_into_text,
    parse_candidate_memory,
)
from .memory_system_amac import LmeCandidateAmacMemorySystem
from .memory_system_base import LmeCandidateMemorySystemBase


def _sorted_chunks(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = payload.get("chunks") or []
    if not isinstance(raw, list):
        raise ValueError("candidate json: 'chunks' must be a list")
    chunks = [c for c in raw if isinstance(c, dict)]
    return sorted(chunks, key=lambda c: int(c.get("chunk_index", 0)))


def sorted_candidate_chunks(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Public alias for loaders that apply candidate JSON (e.g. mem0 ingest)."""
    return _sorted_chunks(payload)


def apply_candidate_episode_json(
    memory: LmeCandidateMemorySystemBase,
    payload: Dict[str, Any],
    *,
    source_path: Optional[Path] = None,
    observation_by_chunk_index: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """
    按 chunk_index 顺序、块内 candidate_memories 顺序逐条更新；写库语义见
    ``relation_decision.decide_lme_update_relation_decision`` 与 ``LmeCandidateRelationDecisionMemorySystem``；
    ``update_method=amac`` 时可通过 ``observation_by_chunk_index`` 传入与抽取一致的 chunk observation 文本。
    """
    history_name = str(payload.get("history_name") or "").strip()
    if not history_name:
        raise ValueError("candidate json: missing history_name")

    database = memory._get_database(history_name)
    trace = memory.trace.get_logger_for(history_name)
    sorted_c = _sorted_chunks(payload)

    ep_scope = trace.create_scope(
        "lme_candidate_ingest_episode",
        metadata={
            "history_name": history_name,
            "source_file": str(source_path) if source_path else None,
            "num_chunks": len(sorted_c),
        },
    )

    stats: Dict[str, Any] = {
        "history_name": history_name,
        "chunks": len(sorted_c),
        "facts_submitted": 0,
        "memory_row_ops": 0,
        "amac_rejected": 0,
        "amac_admitted": 0,
    }

    try:
        metadata_common = {
            "method": "lme_candidate_apply",
            "history_name": history_name,
            "source": "lme_candidate_extract",
        }

        obs_by_ci = observation_by_chunk_index or {}
        chunk_total = len(sorted_c)

        for chunk_ordinal, chunk in enumerate(sorted_c):
            ci = chunk.get("chunk_index")
            session_date = str(chunk.get("session_date") or "").strip()
            try:
                ci_int = int(ci) if ci is not None and str(ci).strip() != "" else -1
            except (TypeError, ValueError):
                ci_int = -1
            obs_text = obs_by_ci.get(ci_int, "") if ci_int >= 0 else ""
            ch_scope = trace.create_scope(
                "lme_candidate_ingest_chunk",
                parent_scope_id=ep_scope,
                metadata={
                    "chunk_index": ci,
                    "session_index": chunk.get("session_index"),
                    "turn_start": chunk.get("turn_start"),
                    "turn_end": chunk.get("turn_end"),
                },
            )
            metadata_base = {
                **metadata_common,
                "date": session_date,
                "chunk_source": str(chunk.get("source") or "filler"),
                "lme_chunk_index": ci,
                "lme_session_index": chunk.get("session_index"),
                "lme_turn_start": chunk.get("turn_start"),
                "lme_turn_end": chunk.get("turn_end"),
                "amac_observation": obs_text,
                "amac_chunk_ordinal": chunk_ordinal,
                "amac_chunk_count": chunk_total,
            }
            session_idx = int(chunk.get("session_index") or 1)
            mems = chunk.get("candidate_memories") or []
            if not isinstance(mems, list):
                mems = []
            cas_rules = chunk.get("cas_update_rules")
            # topic 平行数组（tag_candidate_topics.py 产出，与 candidate_memories 等长）；
            # 仅 relation_decision 消费它做同主题聚合，baseline 忽略。
            topics = chunk.get("candidate_topics")
            consumes_topic = bool(getattr(memory, "consumes_topics", False))
            # ours(relation_decision)消费平行栏条件做级联；baseline 不消费，
            # 此时把条件 merge 回文本，保证各方法输入信息对等(见 memory_system_base.consumes_cas_rules)。
            consumes_cas = bool(getattr(memory, "consumes_cas_rules", False))

            op_sub = 0
            try:
                for fi, m_new in enumerate(mems):
                    rule_fi = (
                        cas_rules[fi]
                        if isinstance(cas_rules, list) and fi < len(cas_rules)
                        else None
                    )
                    if consumes_cas:
                        parsed = parse_candidate_memory(m_new, cas_update_rule=rule_fi)
                        s = parsed.text.strip()
                        cas_rule_used = parsed.cas_update_rule
                    else:
                        # baseline：条件折回文本，平行栏不单独传(否则信息泄露)
                        s = merge_cas_rule_into_text(
                            candidate_memory_display_text(m_new), rule_fi
                        )
                        cas_rule_used = None
                    if not s:
                        continue
                    stats["facts_submitted"] += 1
                    mb = dict(metadata_base)
                    mb["lme_fact_index_in_chunk"] = fi
                    if cas_rule_used:
                        mb["gold_cas_update_condition"] = cas_rule_used
                    if consumes_topic and isinstance(topics, list) and fi < len(topics):
                        topic_fi = str(topics[fi] or "").strip()
                        if topic_fi:
                            mb["topic"] = topic_fi
                    r = memory._process_one_new_fact(
                        database,
                        s,
                        mb,
                        session_idx,
                        ch_scope,
                        trace,
                    )
                    op_sub += r
                    stats["memory_row_ops"] += r
                    if isinstance(memory, LmeCandidateAmacMemorySystem):
                        if r:
                            stats["amac_admitted"] += 1
                        else:
                            stats["amac_rejected"] += 1
                # Skip dedup when no writes happened in this chunk (e.g. evermemos defers
                # all DB writes to finalize_episode, so op_sub is always 0 here).
                removed = database.deduplicate_identical_text() if op_sub > 0 else 0
                trace.close_scope(
                    ch_scope,
                    status="ok",
                    metadata={"memory_ops": op_sub, "dedupe_removed": removed},
                )
            except Exception:
                trace.close_scope(ch_scope, status="error")
                raise
    finally:
        trace.close_scope(ep_scope, status="ok", metadata=stats)

    return stats


def load_candidate_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: root must be object")
    return data


def apply_candidate_file(
    memory: LmeCandidateMemorySystemBase,
    path: Path,
    *,
    observation_by_chunk_index: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    payload = load_candidate_json(path)
    return apply_candidate_episode_json(
        memory,
        payload,
        source_path=path,
        observation_by_chunk_index=observation_by_chunk_index,
    )
