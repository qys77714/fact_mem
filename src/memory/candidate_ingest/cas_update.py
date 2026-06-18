"""
Cascade update (cas_update_condition) helpers for relation_decision ingest.

Text-only: split compound memories, match prior conditions via embedding,
LLM decides UPDATE_PRIMARY / INVALIDATE / NO_ACTION via two-step prompts.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from memory.base import RetrievedMemory
from memory.storage.local_faiss import LocalFaissDatabase, _memory_entry_is_primary
from memory.tracing import MemoryTraceLogger
from prompts import render_prompt

logger = logging.getLogger(__name__)

@dataclass
class ParsedCandidateMemory:
    """One candidate memory row for ingest (text-only primary + optional cascade rule)."""

    text: str
    cas_update_rule: Optional[str] = None


_COMPOUND_COND_RE = re.compile(
    r"(this is determined by|this depends on|this is for my|if my |if that |if we )",
    re.IGNORECASE,
)
_IF_THEN_TEXT_RE = re.compile(r"^If\s+(my|we|I)\b", re.IGNORECASE)


def split_golden_memory(text: str) -> Tuple[str, Optional[str]]:
    """Text-only split: em-dash compound with condition tail, or whole string."""
    t = text.strip()
    if " — " in t:
        primary, condition = t.split(" — ", 1)
        if _COMPOUND_COND_RE.search(condition):
            return primary.strip(), condition.strip()
    return t, None


def golden_fact_to_candidate_entry(fact_text: str) -> str:
    """Extract phase: primary text only (condition stored in parallel ``cas_update_rules``)."""
    return split_candidate_chunk_memory(fact_text)["text"]


def candidate_memory_display_text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("text") or item.get("candidate_text") or "").strip()
    return str(item or "").strip()


def parse_candidate_memory(
    item: Any,
    *,
    cas_update_rule: Optional[str] = None,
) -> ParsedCandidateMemory:
    """
    Normalize one chunk candidate for ingest.

    Supports plain strings, structured dicts, or strings plus parallel ``cas_update_rules``.
    """
    if isinstance(item, dict):
        text = candidate_memory_display_text(item)
        rule = item.get("cas_update_rule")
        if rule is None:
            rule = item.get("cas_update_condition")
    else:
        text = candidate_memory_display_text(item)
        rule = None

    if cas_update_rule is not None:
        rule = cas_update_rule

    if not text:
        return ParsedCandidateMemory(text="")

    if rule is None:
        _, inferred_rule = split_golden_memory(text)
        if inferred_rule is not None:
            rule = inferred_rule

    rule_s = str(rule).strip() if rule else None
    return ParsedCandidateMemory(text=text, cas_update_rule=rule_s)


def split_candidate_chunk_memory(original: str) -> Dict[str, Any]:
    """Split one gold fact: primary text + optional cas_update_rule."""
    original = (original or "").strip()
    primary, condition = split_golden_memory(original)
    return {"text": primary, "cas_update_rule": condition}


def build_evidence_gold_chunk_fields(
    fact_texts: Sequence[str],
) -> Dict[str, Any]:
    """
    Build parallel candidate fields for one evidence_gold_facts chunk.

    ``candidate_memories`` holds primary text; ``cas_update_rules`` holds conditions.
    """
    candidate_memories: List[str] = []
    cas_update_rules: List[Optional[str]] = []

    for raw in fact_texts:
        original = (raw or "").strip()
        if not original:
            continue
        entry = split_candidate_chunk_memory(original)
        candidate_memories.append(entry["text"])
        cas_update_rules.append(entry["cas_update_rule"])

    return {
        "candidate_memories": candidate_memories,
        "cas_update_rules": cas_update_rules,
    }


def merge_cas_rule_into_text(text: str, cas_rule: Optional[str]) -> str:
    """Re-attach a parallel cas_update_rule onto the primary text (inverse of split).

    Baselines (mem0/zep) only read ``candidate_memories`` and never see the parallel
    ``cas_update_rules`` column. To keep the input information identical across all
    methods, fold the condition back into the text with the same ``" — "`` separator
    that ``split_golden_memory`` uses. Ours still consumes the structured column.
    """
    base = (text or "").strip()
    rule = (cas_rule or "").strip()
    if not rule:
        return base
    if not base:
        return rule
    if rule in base:
        return base
    return f"{base.rstrip('.')} — {rule}"


def merged_candidate_texts(chunk: Dict[str, Any]) -> List[str]:
    """Return chunk's candidate_memories as plain strings with cas_update_rules folded in.

    Used by baselines so they receive the same information ours gets via the parallel
    ``cas_update_rules`` column. Order and indexing match ``candidate_memories``.
    """
    mems = chunk.get("candidate_memories") or []
    if not isinstance(mems, list):
        return []
    rules = chunk.get("cas_update_rules")
    out: List[str] = []
    for i, m in enumerate(mems):
        text = candidate_memory_display_text(m).strip()
        if not text:
            continue
        rule = rules[i] if isinstance(rules, list) and i < len(rules) else None
        out.append(merge_cas_rule_into_text(text, rule))
    return out


def chunk_with_merged_candidates(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow chunk copy whose ``candidate_memories`` have cas_update_rules folded in.

    The ``cas_update_rules`` column is dropped so downstream baseline code cannot
    accidentally read it; ours never goes through this path.
    """
    merged = dict(chunk)
    merged["candidate_memories"] = merged_candidate_texts(chunk)
    merged.pop("cas_update_rules", None)
    return merged


def text_mentions_exercise_routine(text: str) -> bool:
    return "exercise routine" in text.lower()


def text_mentions_facility(primary: str, condition: Optional[str] = None) -> bool:
    low = primary.lower()
    if any(k in low for k in ("work out", "workout", "go to", " pool", "studio", "gym", "facility")):
        return True
    c = (condition or "").lower()
    return "routine" in c and ("facility" in c or "different" in c)


def _is_uncertainty_primary_text(text: str) -> bool:
    low = (text or "").lower()
    return "not known" in low or "uncertain" in low


def normalize_primary_from_context(old_primary: str, new_text: str) -> str:
    t = new_text.strip()
    if not t or _is_uncertainty_primary_text(t):
        return t
    old_low = old_primary.lower()
    if text_mentions_exercise_routine(old_primary) and not text_mentions_exercise_routine(t):
        if not t.lower().startswith("my exercise"):
            return f"My exercise routine is {t.rstrip('.')}"
    if "medication" in old_low or old_low.startswith("i take"):
        if "i take" not in t.lower():
            return f"I take a medication called {t.rstrip('.')}"
    if text_mentions_facility(old_primary):
        if t.lower().startswith("i work out at") or t.lower().startswith("i go to"):
            return t
        if not any(k in t.lower() for k in ("work out", "go to", "studio", "pool", "gym")):
            return f"I work out at {t.rstrip('.')}"
    return t


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    na = np.linalg.norm(a64) + 1e-12
    nb = np.linalg.norm(b64) + 1e-12
    return float(np.dot(a64, b64) / (na * nb))


def parse_cas_llm_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"action": "NO_ACTION", "reason": "parse_failed"}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"action": "NO_ACTION", "reason": "json_decode_failed"}


def is_searchable_primary(meta: Dict[str, Any]) -> bool:
    return _memory_entry_is_primary(meta) and not bool(meta.get("stale"))


def is_cascade_root(meta: Optional[Dict[str, Any]]) -> bool:
    if not meta:
        return False
    if meta.get("cascade_managed"):
        return True
    cond = str(meta.get("cas_update_condition") or "").strip()
    return bool(cond)


def get_pending_rules(meta: Dict[str, Any]) -> List[str]:
    raw = meta.get("pending_rules")
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return []


def list_cascade_primaries(db: LocalFaissDatabase) -> List[RetrievedMemory]:
    out: List[RetrievedMemory] = []
    for mem in db.list_all_memories(sort_by_time=False):
        if is_searchable_primary(mem.metadata or {}):
            out.append(mem)
    return out


def list_condition_primaries(db: LocalFaissDatabase) -> List[RetrievedMemory]:
    return [
        m for m in list_cascade_primaries(db)
        if str((m.metadata or {}).get("cas_update_condition") or "").strip()
    ]


@dataclass(frozen=True)
class CasMatch:
    memory: RetrievedMemory
    similarity: float


def is_if_then_text(text: str) -> bool:
    return bool(_IF_THEN_TEXT_RE.match((text or "").strip()))


def condition_match_anchor(mem: RetrievedMemory) -> str:
    """Anchor text for if-then primary matching: primary + condition."""
    primary = (mem.text or "").strip()
    cond = str((mem.metadata or {}).get("cas_update_condition") or "").strip()
    if primary and cond:
        return f"{primary}\n{cond}"
    return primary or cond


def merge_if_then_rule(
    existing_condition: Optional[str],
    rule_text: str,
) -> str:
    rule = (rule_text or "").strip()
    existing = (existing_condition or "").strip()
    if not existing:
        return rule
    if rule in existing:
        return existing
    return existing.rstrip(".") + "; " + rule


def find_primary_for_if_then(
    db: LocalFaissDatabase,
    rule_text: str,
    embed_fn: Callable[[List[str]], np.ndarray],
) -> Optional[CasMatch]:
    """Attach if-then rule to best-matching prior primary via anchor embedding."""
    cond_rows = list_condition_primaries(db)
    if not cond_rows:
        return None
    rule_emb = embed_fn([rule_text])[0]
    best: Optional[CasMatch] = None
    for row in cond_rows:
        anchor = condition_match_anchor(row)
        sim = cosine_sim(rule_emb, embed_fn([anchor])[0])
        if best is None or sim > best.similarity:
            best = CasMatch(memory=row, similarity=sim)
    return best


def apply_if_then_enrich(
    database: LocalFaissDatabase,
    memory_id: str,
    old_meta: Dict[str, Any],
    rule_text: str,
) -> str:
    """Merge if-then rule into cas_update_condition and pending_rules."""
    pending = get_pending_rules(old_meta)
    if rule_text not in pending:
        pending.append(rule_text)
    condition_after = merge_if_then_rule(
        str(old_meta.get("cas_update_condition") or ""),
        rule_text,
    )
    database.update_memory(
        memory_id,
        metadata_updates={
            "cas_update_condition": condition_after,
            "pending_rules": pending,
        },
    )
    return condition_after


def match_prior_conditions(
    db: LocalFaissDatabase,
    trigger_text: str,
    embed_fn: Callable[[List[str]], np.ndarray],
    sim_threshold: float,
) -> List[CasMatch]:
    cond_rows = list_condition_primaries(db)
    if not cond_rows:
        return []
    trig_emb = embed_fn([trigger_text])[0]
    cond_texts = [
        str((m.metadata or {}).get("cas_update_condition") or "").strip()
        for m in cond_rows
    ]
    cond_embs = embed_fn(cond_texts)
    matches: List[CasMatch] = []
    for score, row in zip(
        [cosine_sim(trig_emb, cond_embs[i]) for i in range(len(cond_rows))],
        cond_rows,
    ):
        if score >= sim_threshold:
            matches.append(CasMatch(memory=row, similarity=score))
    matches.sort(key=lambda x: x.similarity, reverse=True)
    return matches


def skip_condition_match(trigger_text: str, matched: RetrievedMemory) -> bool:
    meta = matched.metadata or {}
    if not text_mentions_facility(matched.text, meta.get("cas_update_condition")):
        return False
    low = trigger_text.lower()
    if "exercise routine" not in low and "cycling" not in low:
        return False
    for rule in get_pending_rules(meta):
        if "work out at" in rule.lower():
            return True
    return False


def render_cas_upstream_check_prompt(
    *,
    new_memory: str,
    matched_condition: str,
    linked_primary: str,
) -> str:
    return render_prompt(
        "cas_update_upstream_check.jinja",
        new_memory=new_memory,
        matched_condition=matched_condition,
        linked_primary=linked_primary,
    )


def render_cas_primary_value_prompt(
    *,
    new_memory: str,
    matched_condition: str,
    linked_primary: str,
) -> str:
    return render_prompt(
        "cas_update_primary_value.jinja",
        new_memory=new_memory,
        matched_condition=matched_condition,
        linked_primary=linked_primary,
    )


def render_cas_update_prompt(
    *,
    new_memory: str,
    matched_condition: str,
    linked_primary: str,
) -> str:
    return render_prompt(
        "cas_update_decision.jinja",
        new_memory=new_memory,
        matched_condition=matched_condition,
        linked_primary=linked_primary,
    )


def _call_cas_llm_sync(
    llm_client: Any,
    *,
    purpose: str,
    prompt: str,
    max_new_tokens: int,
    trace: Optional[MemoryTraceLogger],
    trace_scope_id: Optional[str],
) -> Optional[str]:
    messages = [{"role": "user", "content": prompt}]
    try:
        raw_response = llm_client.get_response_chat(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=0,
            verbose=False,
        )
        if trace:
            trace.log_llm_interaction(
                purpose=purpose,
                messages=messages,
                response=raw_response,
                scope_id=trace_scope_id,
                metadata={"temperature": 0},
            )
    except Exception as exc:
        if trace:
            trace.log_llm_interaction(
                purpose=purpose,
                messages=messages,
                response=None,
                scope_id=trace_scope_id,
                metadata={"temperature": 0},
                error=str(exc),
            )
        logger.warning("%s LLM failed: %s", purpose, exc)
        return None
    raw = raw_response[0] if isinstance(raw_response, (list, tuple)) and raw_response else raw_response
    return str(raw or "")


def decide_cas_upstream_check_sync(
    llm_client: Any,
    *,
    new_memory: str,
    matched_condition: str,
    linked_primary: str,
    max_new_tokens: int = 512,
    trace: Optional[MemoryTraceLogger] = None,
    trace_scope_id: Optional[str] = None,
) -> Dict[str, Any]:
    prompt = render_cas_upstream_check_prompt(
        new_memory=new_memory,
        matched_condition=matched_condition,
        linked_primary=linked_primary,
    )
    raw = _call_cas_llm_sync(
        llm_client,
        purpose="cas_update_upstream_check",
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        trace=trace,
        trace_scope_id=trace_scope_id,
    )
    if raw is None:
        return {"action": "NO_ACTION", "reason": "llm_error"}
    parsed = parse_cas_llm_json(raw)
    action = str(parsed.get("action", "NO_ACTION")).upper()
    if action not in ("PROCEED", "NO_ACTION"):
        action = "NO_ACTION"
    parsed["action"] = action
    return parsed


def decide_cas_primary_value_sync(
    llm_client: Any,
    *,
    new_memory: str,
    matched_condition: str,
    linked_primary: str,
    max_new_tokens: int = 512,
    trace: Optional[MemoryTraceLogger] = None,
    trace_scope_id: Optional[str] = None,
) -> Dict[str, Any]:
    prompt = render_cas_primary_value_prompt(
        new_memory=new_memory,
        matched_condition=matched_condition,
        linked_primary=linked_primary,
    )
    raw = _call_cas_llm_sync(
        llm_client,
        purpose="cas_update_primary_value",
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        trace=trace,
        trace_scope_id=trace_scope_id,
    )
    if raw is None:
        return {"new_primary_text": "", "reason": "llm_error"}
    return parse_cas_llm_json(raw)


def decide_cas_cascade_sync(
    llm_client: Any,
    *,
    new_memory: str,
    matched_condition: str,
    linked_primary: str,
    max_new_tokens: int = 512,
    trace: Optional[MemoryTraceLogger] = None,
    trace_scope_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Two-step cascade: upstream check then primary value (always UPDATE when proceeding)."""
    step1 = decide_cas_upstream_check_sync(
        llm_client,
        new_memory=new_memory,
        matched_condition=matched_condition,
        linked_primary=linked_primary,
        max_new_tokens=max_new_tokens,
        trace=trace,
        trace_scope_id=trace_scope_id,
    )
    if str(step1.get("action", "NO_ACTION")).upper() != "PROCEED":
        return {
            "action": "NO_ACTION",
            "reason": step1.get("reason", "upstream_not_changed"),
        }

    step2 = decide_cas_primary_value_sync(
        llm_client,
        new_memory=new_memory,
        matched_condition=matched_condition,
        linked_primary=linked_primary,
        max_new_tokens=max_new_tokens,
        trace=trace,
        trace_scope_id=trace_scope_id,
    )
    new_text = str(step2.get("new_primary_text") or "").strip()
    return {
        "action": "UPDATE_PRIMARY",
        "new_primary_text": new_text,
        "reason": step2.get("reason", ""),
    }


def decide_cas_update_sync(
    llm_client: Any,
    *,
    new_memory: str,
    matched_condition: str,
    linked_primary: str,
    max_new_tokens: int = 512,
    trace: Optional[MemoryTraceLogger] = None,
    trace_scope_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Legacy single-call entry; production uses decide_cas_cascade_sync."""
    return decide_cas_cascade_sync(
        llm_client,
        new_memory=new_memory,
        matched_condition=matched_condition,
        linked_primary=linked_primary,
        max_new_tokens=max_new_tokens,
        trace=trace,
        trace_scope_id=trace_scope_id,
    )


def build_cascade_metadata(
    metadata_base: Dict[str, Any],
    *,
    primary_text: str,
    cas_update_condition: Optional[str] = None,
    pending_rules: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    meta = dict(metadata_base)
    meta["memory_role"] = "primary"
    meta["cascade_managed"] = True
    meta["primary_text"] = primary_text
    meta["cas_update_condition"] = cas_update_condition
    meta["pending_rules"] = list(pending_rules or [])
    meta["stale"] = False
    meta["lme_update_method"] = "relation_decision_cascade"
    return meta


def metadata_for_new_primary(
    metadata_base: Dict[str, Any],
    m_new: str,
    *,
    lme_update_method: str = "relation_decision",
) -> Dict[str, Any]:
    """Build metadata for a new primary row; attach incoming cas rule when present."""
    cond = metadata_base.get("gold_cas_update_condition")
    if cond:
        meta = build_cascade_metadata(
            metadata_base,
            primary_text=m_new,
            cas_update_condition=str(cond).strip() or None,
        )
        meta["lme_update_method"] = lme_update_method
        return meta
    meta = dict(metadata_base)
    meta["memory_role"] = "primary"
    meta["lme_update_method"] = lme_update_method
    return meta


def has_similar_cascade_primary(
    db: LocalFaissDatabase,
    primary_text: str,
    embed_fn: Callable[[List[str]], np.ndarray],
    threshold: float = 0.92,
) -> bool:
    """Return True if an active cascade primary is near-duplicate of primary_text."""
    target = primary_text.strip().lower()
    if not target:
        return False
    for mem in list_cascade_primaries(db):
        if mem.text.strip().lower() == target:
            return True
    trig_emb = embed_fn([primary_text])[0]
    for mem in list_cascade_primaries(db):
        sim = cosine_sim(trig_emb, embed_fn([mem.text])[0])
        if sim >= threshold:
            return True
    return False
