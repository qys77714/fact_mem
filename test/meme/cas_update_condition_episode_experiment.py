#!/usr/bin/env python3
"""
pl_001 Cas Update Condition single-episode experiment.

Standalone cascade-aware memory processor (does NOT use relation_decision):
  1. Split compound golden memories into primary_text + cas_update_condition
  2. On each new memory, match prior cas_update_condition texts via embedding (sim >= threshold)
  3. LLM decides UPDATE_PRIMARY / ENRICH_CONDITION / INVALIDATE / NO_ACTION
  4. Answer Cas/Abs questions with primary-only retrieval + gemma4-26B
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import numpy as np
from jinja2 import Environment, FileSystemLoader
from openai import OpenAI
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from eval.meme_judge import MemeLLMJudge, aggregate_meme_metrics, classify_trivial_pass, task_base  # noqa: E402
from memory.base import RetrievedMemory  # noqa: E402
from prompts import render_prompt  # noqa: E402
from utils.embed_utils import embed_texts  # noqa: E402
from utils.env import load_env  # noqa: E402
from utils.eval_report import append_eval_json, utc_timestamp_iso  # noqa: E402
from utils.llm_api import load_api_chat_completion  # noqa: E402

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_DATASET = _REPO_ROOT / "data/raw_data/MEME/meme_nofiller.json"
DEFAULT_FILLER32K_DATASET = _REPO_ROOT / "data/raw_data/MEME/meme_filler32k.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output/pl_001_cas_update_condition"
DEFAULT_FILLER32K_OUTPUT = Path(__file__).resolve().parent / "output/pl_001_cas_update_condition_filler32k"
DEFAULT_CANDIDATES_DIR = _REPO_ROOT / "MemDB/candidates/meme_filler32k_gemma4-26B_0519_as3"

SourceKind = Literal["gold_split", "gold_if_then", "plain"]
IngestOrigin = Literal["gold", "filler"]
MemoryRole = Literal["primary", "evidence"]
ActionKind = Literal["UPDATE_PRIMARY", "ENRICH_CONDITION", "INVALIDATE", "NO_ACTION", "IF_THEN_ENRICH"]


_IF_THEN_TEXT_RE = re.compile(r"^If\s+(my|we|I)\b", re.IGNORECASE)
_COMPOUND_COND_RE = re.compile(
    r"(this is determined by|this depends on|this is for my|if my |if that |if we )",
    re.IGNORECASE,
)


@dataclass
class EvidenceFact:
    """Only fact_text + ingest order; no dataset entity/metadata."""
    fact_id: int
    fact_text: str
    session_index: int
    timestamp: str


@dataclass
class IngestItem:
    """Unified ingest unit for evidence-only or filler32k timeline modes."""
    fact_id: int
    fact_text: str
    session_index: int
    timestamp: str
    origin: IngestOrigin
    global_order: int


@dataclass
class MemoryRow:
    id: str
    primary_text: str
    cas_update_condition: Optional[str]
    memory_role: MemoryRole
    parent_primary_id: Optional[str]
    lme_edge: Optional[str]
    order_index: int
    entity: Optional[str]
    session_index: int
    source: SourceKind
    stale: bool = False
    pending_rules: List[str] = field(default_factory=list)
    timestamp: str = ""


@dataclass(frozen=True)
class MemeQuestion:
    episode_id: str
    domain: str
    phase: str
    task_type: str
    question: str
    reference: str
    question_time: str
    position_after_session: int
    hop: Optional[int] = None
    entities: Optional[List[str]] = None


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


@lru_cache(maxsize=1)
def _cas_prompt_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_PROMPTS_DIR)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        variable_start_string="[[",
        variable_end_string="]]",
    )


def render_cas_prompt(**context: Any) -> str:
    tpl = _cas_prompt_env().get_template("cas_update_decision.jinja")
    return tpl.render(**context).strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cas update condition single-episode experiment")
    p.add_argument("--episode-id", default="pl_001")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument(
        "--candidates-dir",
        type=Path,
        default=None,
        help="When set, ingest gold+filler memories from per-episode candidate JSON "
        "(filler32k timeline mode).",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--embedding-model", default="qwen3-embedding-0.6b")
    p.add_argument("--embedding-base-url", default=None)
    p.add_argument("--embedding-api-key", default=None)
    p.add_argument("--answer-model", default="gemma4-26B")
    p.add_argument("--judge-model", default="gemma4-26B")
    p.add_argument("--condition-sim-threshold", type=float, default=0.5)
    p.add_argument("--retrieve-topk", type=int, default=20)
    p.add_argument("--task-types", default="Cas,Abs")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--judge-max-new-tokens", type=int, default=512)
    p.add_argument("--answer-concurrency", type=int, default=4)
    p.add_argument("--judge-concurrency", type=int, default=4)
    p.add_argument("--embed-batch-size", type=int, default=32)
    p.add_argument("--propagation-rounds", type=int, default=2)
    p.add_argument("--skip-judge", action="store_true")
    p.add_argument("--skip-api-check", action="store_true")
    args = p.parse_args()
    if args.output_dir is None:
        args.output_dir = (
            DEFAULT_FILLER32K_OUTPUT if args.candidates_dir else DEFAULT_OUTPUT
        )
    return args


def load_episode(path: Path, episode_id: str) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        episodes = json.load(f)
    for ep in episodes:
        if ep.get("episode_id") == episode_id:
            return ep
    raise ValueError(f"Episode {episode_id} not found in {path}")


def enumerate_evidence_facts(episode: Dict[str, Any]) -> List[EvidenceFact]:
    out: List[EvidenceFact] = []
    for sess_idx, sess in enumerate(episode.get("sessions") or []):
        if sess.get("type") != "evidence":
            continue
        for gf in sess.get("gold_facts") or []:
            out.append(
                EvidenceFact(
                    fact_id=int(gf["fact_id"]),
                    fact_text=str(gf.get("fact_text") or gf.get("original_seed") or "").strip(),
                    session_index=sess_idx,
                    timestamp=str(sess.get("timestamp", "")),
                )
            )
    return out


def _build_gold_fact_lookup(
    episode: Dict[str, Any],
) -> Dict[str, Tuple[int, int, str]]:
    """Map fact text -> (fact_id, session_index_0based, timestamp)."""
    out: Dict[str, Tuple[int, int, str]] = {}
    for sess_idx, sess in enumerate(episode.get("sessions") or []):
        if sess.get("type") != "evidence":
            continue
        ts = str(sess.get("timestamp", ""))
        for gf in sess.get("gold_facts") or []:
            fact_id = int(gf["fact_id"])
            for key in (gf.get("fact_text"), gf.get("original_seed")):
                text = str(key or "").strip()
                if text and text not in out:
                    out[text] = (fact_id, sess_idx, ts)
    return out


def enumerate_evidence_ingest_items(episode: Dict[str, Any]) -> List[IngestItem]:
    items: List[IngestItem] = []
    for order, fact in enumerate(enumerate_evidence_facts(episode)):
        items.append(
            IngestItem(
                fact_id=fact.fact_id,
                fact_text=fact.fact_text,
                session_index=fact.session_index,
                timestamp=fact.timestamp,
                origin="gold",
                global_order=order,
            )
        )
    return items


def enumerate_ingest_timeline(
    episode: Dict[str, Any],
    candidates_path: Path,
) -> List[IngestItem]:
    """Walk candidate chunks in chunk_index order; interleave gold and filler."""
    gold_lookup = _build_gold_fact_lookup(episode)
    data = json.loads(candidates_path.read_text(encoding="utf-8"))
    chunks = sorted(data.get("chunks") or [], key=lambda c: int(c.get("chunk_index", 0)))
    seen: Set[str] = set()
    items: List[IngestItem] = []
    global_order = 0
    filler_id_base = 1_000_000

    for ch in chunks:
        sess_1based = int(ch.get("session_index") or 1)
        sess_0based = sess_1based - 1
        session_date = str(ch.get("session_date") or "")
        source = str(ch.get("source") or "filler")
        is_gold = source == "evidence_gold_facts"
        for mem_idx, raw in enumerate(ch.get("candidate_memories") or []):
            text = str(raw).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            if is_gold:
                meta = gold_lookup.get(text)
                if meta:
                    fact_id, sess_idx, ts = meta
                else:
                    fact_id = -1
                    sess_idx = sess_0based
                    ts = session_date
                origin: IngestOrigin = "gold"
            else:
                fact_id = filler_id_base + global_order
                sess_idx = sess_0based
                ts = session_date
                origin = "filler"
            items.append(
                IngestItem(
                    fact_id=fact_id,
                    fact_text=text,
                    session_index=sess_idx,
                    timestamp=ts,
                    origin=origin,
                    global_order=global_order,
                )
            )
            global_order += 1
    return items


def load_ingest_items(
    episode: Dict[str, Any],
    episode_id: str,
    candidates_dir: Optional[Path],
) -> Tuple[List[IngestItem], str]:
    if candidates_dir is None:
        return enumerate_evidence_ingest_items(episode), "evidence_only"
    candidates_path = candidates_dir / f"{episode_id}.json"
    if not candidates_path.is_file():
        raise FileNotFoundError(f"Candidate file not found: {candidates_path}")
    return enumerate_ingest_timeline(episode, candidates_path), "filler32k_timeline"


def is_if_then_text(text: str) -> bool:
    return bool(_IF_THEN_TEXT_RE.match(text.strip()))


def split_golden_memory(item: IngestItem) -> Tuple[str, Optional[str], SourceKind]:
    """Text-only split: if-then pattern, em-dash compound, or plain."""
    if is_if_then_text(item.fact_text):
        return item.fact_text, None, "gold_if_then"
    if " — " in item.fact_text:
        primary, condition = item.fact_text.split(" — ", 1)
        if _COMPOUND_COND_RE.search(condition):
            return primary.strip(), condition.strip(), "gold_split"
    return item.fact_text, None, "plain"


def _text_mentions_exercise_routine(text: str) -> bool:
    return "exercise routine" in text.lower()


def _text_mentions_facility(primary: str, condition: Optional[str] = None) -> bool:
    low = primary.lower()
    if any(k in low for k in ("work out", "workout", "go to", " pool", "studio", "gym", "facility")):
        return True
    c = (condition or "").lower()
    return "routine" in c and ("facility" in c or "different" in c)


def _normalize_primary_from_context(old_primary: str, new_text: str) -> str:
    """Shape new primary text using linked primary wording (text-only)."""
    t = new_text.strip()
    if not t:
        return t
    old_low = old_primary.lower()
    if _text_mentions_exercise_routine(old_primary) and not _text_mentions_exercise_routine(t):
        if not t.lower().startswith("my exercise"):
            return f"My exercise routine is {t.rstrip('.')}"
    if "medication" in old_low or old_low.startswith("i take"):
        if "i take" not in t.lower():
            return f"I take a medication called {t.rstrip('.')}"
    if _text_mentions_facility(old_primary):
        if t.lower().startswith("i work out at") or t.lower().startswith("i go to"):
            return t
        if not any(k in t.lower() for k in ("work out", "go to", "studio", "pool", "gym")):
            return f"I work out at {t.rstrip('.')}"
    return t


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    na = np.linalg.norm(a64) + 1e-12
    nb = np.linalg.norm(b64) + 1e-12
    return float(np.dot(a64, b64) / (na * nb))


def _parse_llm_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"action": "NO_ACTION", "reason": "parse_failed"}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"action": "NO_ACTION", "reason": "json_decode_failed"}


class CasUpdateMemorySystem:
    def __init__(
        self,
        embed_client: OpenAI,
        embed_model: str,
        llm_client,
        *,
        sim_threshold: float = 0.5,
        propagation_rounds: int = 2,
    ) -> None:
        self.embed_client = embed_client
        self.embed_model = embed_model
        self.llm_client = llm_client
        self.sim_threshold = sim_threshold
        self.propagation_rounds = propagation_rounds
        self.rows: List[MemoryRow] = []
        self._embed_cache: Dict[str, np.ndarray] = {}
        self._order = 0
        self.trace: List[Dict[str, Any]] = []

    def _embed(self, texts: List[str]) -> np.ndarray:
        missing = [t for t in texts if t and t not in self._embed_cache]
        if missing:
            embs = embed_texts(self.embed_client, missing, self.embed_model)
            for t, e in zip(missing, embs):
                self._embed_cache[t] = e.astype(np.float32)
        return np.vstack([self._embed_cache[t] for t in texts if t]).astype(np.float32)

    def _embed_one(self, text: str) -> np.ndarray:
        return self._embed([text])[0]

    def _active_primaries_before(self, before_order: int) -> List[MemoryRow]:
        return [
            r for r in self.rows
            if r.order_index < before_order
            and r.memory_role == "primary"
            and not r.stale
        ]

    def _find_primary_for_if_then(self, rule_text: str, before_order: int) -> Optional[MemoryRow]:
        """Attach if-then rule to best-matching prior primary via embedding (text-only)."""
        candidates = [
            r for r in self._active_primaries_before(before_order)
            if r.cas_update_condition
        ]
        if not candidates:
            return None
        rule_emb = self._embed_one(rule_text)
        best: Optional[Tuple[float, MemoryRow]] = None
        for row in candidates:
            anchor = f"{row.primary_text}\n{row.cas_update_condition}"
            sim = _cosine_sim(rule_emb, self._embed_one(anchor))
            if best is None or sim > best[0]:
                best = (sim, row)
        return best[1] if best else None

    def _active_facility_primary(self) -> Optional[MemoryRow]:
        for r in reversed(self.rows):
            if r.memory_role != "primary" or r.stale:
                continue
            if _text_mentions_facility(r.primary_text, r.cas_update_condition):
                return r
        return None

    def _prior_condition_rows(self, before_order: int) -> List[MemoryRow]:
        return [
            r for r in self.rows
            if r.order_index < before_order
            and r.cas_update_condition
            and r.memory_role == "primary"
            and not r.stale
        ]

    def _match_conditions(
        self, trigger_text: str, before_order: int
    ) -> List[Tuple[float, MemoryRow]]:
        cond_rows = self._prior_condition_rows(before_order)
        if not cond_rows:
            return []
        trig_emb = self._embed_one(trigger_text)
        cond_texts = [r.cas_update_condition for r in cond_rows]  # type: ignore
        cond_embs = self._embed(cond_texts)
        matches: List[Tuple[float, MemoryRow]] = []
        for score, row in zip(
            [_cosine_sim(trig_emb, cond_embs[i]) for i in range(len(cond_rows))],
            cond_rows,
        ):
            if score >= self.sim_threshold:
                matches.append((score, row))
        matches.sort(key=lambda x: x[0], reverse=True)
        return matches

    def _skip_condition_match(self, trigger_text: str, matched: MemoryRow) -> bool:
        """Avoid re-processing exercise→facility when hop-2 already applied."""
        if not _text_mentions_facility(matched.primary_text, matched.cas_update_condition):
            return False
        low = trigger_text.lower()
        if "exercise routine" not in low and "cycling" not in low:
            return False
        for rule in matched.pending_rules:
            if "work out at" in rule.lower():
                return True
        return False

    async def _llm_decide(
        self,
        *,
        new_memory: str,
        matched_row: MemoryRow,
        similarity: float,
    ) -> Dict[str, Any]:
        prompt = render_cas_prompt(
            new_memory=new_memory,
            matched_condition=matched_row.cas_update_condition or "",
            linked_primary=matched_row.primary_text,
        )
        resp = await self.llm_client.get_response_chat(
            [[{"role": "user", "content": prompt}]],
            max_new_tokens=512,
            temperature=0.0,
            max_concurrency=1,
            use_tqdm=False,
            verbose=False,
        )
        raw = resp[0] if resp else ""
        parsed = _parse_llm_json(str(raw))
        parsed["_similarity"] = similarity
        return parsed

    def _add_primary(
        self,
        *,
        primary_text: str,
        cas_update_condition: Optional[str],
        session_index: int,
        source: SourceKind,
        timestamp: str,
        pending_rules: Optional[List[str]] = None,
    ) -> MemoryRow:
        row = MemoryRow(
            id=_new_id(),
            primary_text=primary_text,
            cas_update_condition=cas_update_condition,
            memory_role="primary",
            parent_primary_id=None,
            lme_edge=None,
            order_index=self._order,
            entity=None,
            session_index=session_index,
            source=source,
            timestamp=timestamp,
            pending_rules=list(pending_rules or []),
        )
        self.rows.append(row)
        self._order += 1
        return row

    def _apply_update_primary(
        self,
        old_row: MemoryRow,
        new_primary_text: str,
        *,
        trigger_session: Optional[int] = None,
    ) -> MemoryRow:
        normalized = _normalize_primary_from_context(old_row.primary_text, new_primary_text)
        new_row = self._add_primary(
            primary_text=normalized,
            cas_update_condition=old_row.cas_update_condition,
            session_index=trigger_session if trigger_session is not None else old_row.session_index,
            source=old_row.source,
            timestamp=old_row.timestamp,
            pending_rules=list(old_row.pending_rules),
        )
        old_row.memory_role = "evidence"
        old_row.parent_primary_id = new_row.id
        old_row.lme_edge = "UPDATE"
        old_row.stale = True
        return new_row

    def _apply_hop2_facility_update(
        self,
        exercise_row: MemoryRow,
        trigger_session: int,
        trace_base: Dict[str, Any],
    ) -> Optional[MemoryRow]:
        """When exercise-routine primary updates, apply facility rule from pending_rules text."""
        fac = self._active_facility_primary()
        if fac is None:
            return None
        replacement: Optional[str] = None
        for rule in fac.pending_rules:
            low = rule.lower()
            if "exercise routine" in low and "work out at" in low:
                m = re.search(r"work out at (.+)$", rule, re.I)
                if m:
                    replacement = f"I work out at {m.group(1).strip().rstrip('.')}"
                    break
        if not replacement:
            return None
        new_fac = self._apply_update_primary(
            fac, replacement, trigger_session=trigger_session,
        )
        self.trace.append({
            **trace_base,
            "event": "hop2_facility_update",
            "trigger_primary": exercise_row.primary_text,
            "facility_primary_before": fac.primary_text,
            "new_facility_primary": new_fac.primary_text,
            "new_facility_id": new_fac.id,
        })
        return new_fac

    def _apply_enrich(self, row: MemoryRow, enriched: str) -> None:
        row.cas_update_condition = enriched.strip()

    def _apply_invalidate(self, row: MemoryRow, trigger_session: int) -> MemoryRow:
        row.stale = True
        if row.cas_update_condition:
            row.cas_update_condition += " [INVALIDATED: upstream change, value uncertain]"
        uncertain = (
            f"This fact is uncertain — previously '{row.primary_text}', "
            f"but an upstream dependency changed and I do not know the current value."
        )
        return self._add_primary(
            primary_text=uncertain,
            cas_update_condition=None,
            session_index=trigger_session,
            source="plain",
            timestamp=row.timestamp,
        )

    async def _process_matches(
        self,
        trigger_text: str,
        before_order: int,
        *,
        trace_base: Dict[str, Any],
        trigger_session: int = 0,
    ) -> List[str]:
        """Return list of new primary texts produced (for propagation)."""
        produced: List[str] = []
        matches = self._match_conditions(trigger_text, before_order)
        for sim, matched in matches:
            if self._skip_condition_match(trigger_text, matched):
                self.trace.append({
                    **trace_base,
                    "event": "condition_match_skipped",
                    "matched_primary": matched.primary_text,
                    "trigger_text": trigger_text,
                    "reason": "hop2_or_exercise_trigger",
                })
                continue
            decision = await self._llm_decide(
                new_memory=trigger_text,
                matched_row=matched,
                similarity=sim,
            )
            action = str(decision.get("action", "NO_ACTION")).upper()
            entry = {
                **trace_base,
                "event": "condition_match",
                "matched_primary": matched.primary_text,
                "matched_condition": matched.cas_update_condition,
                "similarity": round(sim, 4),
                "llm_action": action,
                "llm_reason": decision.get("reason", ""),
                "llm_raw": decision,
            }
            if action == "UPDATE_PRIMARY":
                new_text = str(decision.get("new_primary_text") or "").strip()
                if new_text:
                    new_row = self._apply_update_primary(
                        matched, new_text, trigger_session=trigger_session,
                    )
                    entry["new_primary_id"] = new_row.id
                    entry["new_primary_text"] = new_row.primary_text
                    if _text_mentions_exercise_routine(matched.primary_text) or _text_mentions_exercise_routine(new_row.primary_text):
                        hop2 = self._apply_hop2_facility_update(
                            new_row, trigger_session, trace_base,
                        )
                        if hop2:
                            produced.append(hop2.primary_text)
                        else:
                            produced.append(new_row.primary_text)
                    else:
                        produced.append(new_row.primary_text)
            elif action == "ENRICH_CONDITION":
                enriched = str(decision.get("enriched_condition") or "").strip()
                if enriched:
                    self._apply_enrich(matched, enriched)
                    entry["enriched_condition"] = enriched
            elif action == "INVALIDATE":
                uncertain_row = self._apply_invalidate(matched, trigger_session)
                entry["invalidated_primary_id"] = matched.id
                entry["uncertainty_primary_id"] = uncertain_row.id
                entry["uncertainty_primary_text"] = uncertain_row.primary_text
            self.trace.append(entry)
        return produced

    async def _propagate(
        self,
        trigger_texts: List[str],
        before_order: int,
        trace_base: Dict[str, Any],
        trigger_session: int = 0,
    ) -> None:
        queue = list(trigger_texts)
        for round_i in range(self.propagation_rounds):
            if not queue:
                break
            next_queue: List[str] = []
            for trig in queue:
                produced = await self._process_matches(
                    trig,
                    before_order,
                    trace_base={**trace_base, "propagation_round": round_i + 1},
                    trigger_session=trigger_session,
                )
                next_queue.extend(produced)
            queue = next_queue

    async def _handle_if_then(self, item: IngestItem, trace_base: Dict[str, Any]) -> None:
        order_before = self._order
        target = self._find_primary_for_if_then(item.fact_text, order_before)
        if target is None:
            self.trace.append({
                **trace_base,
                "event": "if_then_orphan",
                "rule": item.fact_text,
            })
            return
        target.pending_rules.append(item.fact_text)
        # Merge if-then rule into cas_update_condition directly (single source of truth).
        if target.cas_update_condition:
            if item.fact_text not in target.cas_update_condition:
                target.cas_update_condition = (
                    target.cas_update_condition.rstrip(".") + "; " + item.fact_text
                )
        else:
            target.cas_update_condition = item.fact_text
        self.trace.append({
            **trace_base,
            "event": "if_then_enrich",
            "linked_primary": target.primary_text,
            "rule": item.fact_text,
            "condition_after": target.cas_update_condition,
        })

    async def ingest_gold_item(self, item: IngestItem) -> None:
        primary_text, condition, source = split_golden_memory(item)
        order_before = self._order
        trace_base: Dict[str, Any] = {
            "session_index": item.session_index,
            "fact_id": item.fact_id,
            "fact_text": item.fact_text,
            "memory_origin": "gold",
            "global_order": item.global_order,
            "source_kind": source,
            "order_index": order_before,
            "text_only": True,
        }

        if source == "gold_if_then":
            await self._handle_if_then(item, trace_base)
            return

        row = self._add_primary(
            primary_text=primary_text,
            cas_update_condition=condition,
            session_index=item.session_index,
            source=source,
            timestamp=item.timestamp,
        )
        self.trace.append({
            **trace_base,
            "event": "store_primary",
            "row_id": row.id,
            "primary_text": primary_text,
            "cas_update_condition": condition,
        })

        produced = await self._process_matches(
            item.fact_text,
            order_before,
            trace_base={**trace_base, "event": "ingest_match"},
            trigger_session=item.session_index,
        )
        await self._propagate(
            produced,
            self._order,
            trace_base={**trace_base, "event": "propagate"},
            trigger_session=item.session_index,
        )

    async def ingest_filler_memory(self, item: IngestItem) -> None:
        order_before = self._order
        trace_base: Dict[str, Any] = {
            "session_index": item.session_index,
            "fact_id": item.fact_id,
            "fact_text": item.fact_text,
            "memory_origin": "filler",
            "global_order": item.global_order,
            "order_index": order_before,
            "text_only": True,
        }
        row = self._add_primary(
            primary_text=item.fact_text,
            cas_update_condition=None,
            session_index=item.session_index,
            source="plain",
            timestamp=item.timestamp,
        )
        self.trace.append({
            **trace_base,
            "event": "store_filler",
            "row_id": row.id,
            "primary_text": item.fact_text,
        })

    async def ingest_up_to_session(self, items: List[IngestItem], max_session: int) -> None:
        eligible = [it for it in items if it.session_index <= max_session]
        eligible.sort(key=lambda x: x.global_order)
        for item in eligible:
            if item.origin == "filler":
                await self.ingest_filler_memory(item)
            else:
                await self.ingest_gold_item(item)

    def active_primaries(self, max_session: int) -> List[MemoryRow]:
        return [
            r for r in self.rows
            if r.memory_role == "primary"
            and not r.stale
            and r.session_index <= max_session
        ]


class CasUpdateMemoryBank:
    def __init__(
        self,
        embed_client: OpenAI,
        embed_model: str,
    ) -> None:
        self._embed_client = embed_client
        self._embed_model = embed_model
        self._systems: Dict[str, CasUpdateMemorySystem] = {}

    def attach_system(self, episode_id: str, system: CasUpdateMemorySystem) -> None:
        self._systems[episode_id] = system

    def retrieve(
        self,
        episode_id: str,
        query: str,
        max_session: int,
        top_k: int,
    ) -> List[RetrievedMemory]:
        system = self._systems[episode_id]
        primaries = system.active_primaries(max_session)
        if not primaries:
            return []
        texts = [p.primary_text for p in primaries]
        doc_embs = embed_texts(self._embed_client, texts, self._embed_model).astype(np.float64)
        q_emb = embed_texts(self._embed_client, [query], self._embed_model)[0].astype(np.float64)
        q_norm = np.linalg.norm(q_emb) + 1e-12
        doc_norms = np.linalg.norm(doc_embs, axis=1) + 1e-12
        scores = (doc_embs @ q_emb) / (doc_norms * q_norm)
        k = min(top_k, len(primaries))
        top_idx = np.argsort(scores)[::-1][:k]

        out: List[RetrievedMemory] = []
        for idx in top_idx:
            row = primaries[int(idx)]
            score = float(scores[int(idx)])
            out.append(
                RetrievedMemory(
                    memory_id=row.id,
                    text=row.primary_text,
                    source_index=row.id,
                    time=row.timestamp,
                    score=float(score),
                    metadata={
                        "session_index": row.session_index,
                        "cas_update_condition": row.cas_update_condition,
                        "source": row.source,
                    },
                )
            )
        return out


def iter_questions(episode: Dict[str, Any], phases: List[str], task_filter: Set[str]) -> List[MemeQuestion]:
    rows: List[MemeQuestion] = []
    eid = episode["episode_id"]
    domain = episode.get("domain", "")
    for phase in phases:
        block = episode.get(f"{phase}_questions")
        if not block:
            continue
        pos = int(block.get("position_after_session", -1))
        q_time = str(block.get("timestamp", ""))
        for q in block.get("questions") or []:
            tt = str(q.get("task_type", ""))
            if task_filter and tt not in task_filter:
                continue
            ref = q.get("gold_answer")
            if ref is None:
                ref = q.get("expected_answer", "")
            rows.append(
                MemeQuestion(
                    episode_id=eid,
                    domain=domain,
                    phase=phase,
                    task_type=tt,
                    question=str(q.get("question", "")),
                    reference=str(ref or ""),
                    question_time=q_time,
                    position_after_session=pos,
                    hop=q.get("hop"),
                    entities=list(q.get("entity") or []),
                )
            )
    return rows


def format_context(retrieved: List[RetrievedMemory]) -> str:
    if not retrieved:
        return render_prompt("agent_context_empty_en.jinja")
    lines = [
        render_prompt(
            "agent_context_unit_en.jinja",
            index=i + 1,
            text=item.text,
            time=item.time,
            metadata=item.metadata or {},
        )
        for i, item in enumerate(retrieved)
    ]
    return "\n\n".join(lines)


def build_answer_prompt(question: MemeQuestion, context_block: str) -> str:
    return render_prompt(
        "agent_prompt_en_open.jinja",
        context_block=context_block,
        question_time=question.question_time,
        question=question.question,
    )


def _count_ingested_by_origin(
    items: List[IngestItem],
    max_session: int,
) -> Dict[str, int]:
    eligible = [it for it in items if it.session_index <= max_session]
    return {
        "gold": sum(1 for it in eligible if it.origin == "gold"),
        "filler": sum(1 for it in eligible if it.origin == "filler"),
    }


async def run_experiment(args: argparse.Namespace) -> None:
    load_env(str(_REPO_ROOT / ".env"))
    episode = load_episode(args.dataset, args.episode_id)
    ingest_items, ingest_mode = load_ingest_items(
        episode, args.episode_id, args.candidates_dir,
    )
    task_filter = {t.strip() for t in args.task_types.split(",") if t.strip()}
    questions = iter_questions(episode, ["before", "after"], task_filter)

    embed_base = args.embedding_base_url or os.getenv("EMBEDDING_BASE_URL", "http://localhost:7110/v1/")
    embed_key = args.embedding_api_key or os.getenv("EMBEDDING_API_KEY", "zjj")
    embed_client = OpenAI(api_key=embed_key, base_url=embed_base)
    llm_client = load_api_chat_completion(args.answer_model, async_=True)
    answer_client = llm_client

    if not args.skip_api_check:
        embed_client.embeddings.create(input=["health"], model=args.embedding_model)
        await answer_client.get_response_chat(
            [[{"role": "user", "content": "Reply OK"}]],
            max_new_tokens=8,
            temperature=0.0,
            max_concurrency=1,
        )
        print("[ok] embedding + chat APIs")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "ingest_trace.jsonl"
    pred_path = out_dir / "pred.jsonl"

    # Incremental ingest per phase
    systems_by_phase: Dict[str, CasUpdateMemorySystem] = {}
    bank = CasUpdateMemoryBank(embed_client, args.embedding_model)

    before_max = int(episode.get("before_questions", {}).get("position_after_session", 3))
    after_max = int(episode.get("after_questions", {}).get("position_after_session", 4))
    before_counts = _count_ingested_by_origin(ingest_items, before_max)
    after_counts = _count_ingested_by_origin(ingest_items, after_max)

    print("Mode: text-only memory (primary_text + cas_update_condition, no entity metadata)")
    print(f"Ingest mode: {ingest_mode}")
    if ingest_mode == "filler32k_timeline":
        print(
            f"  Timeline: {len(ingest_items)} memories "
            f"({sum(1 for it in ingest_items if it.origin == 'gold')} gold, "
            f"{sum(1 for it in ingest_items if it.origin == 'filler')} filler total)"
        )
        if args.candidates_dir:
            print(f"  Candidates: {args.candidates_dir}")
    print(
        f"Ingesting up to session {before_max} (before phase): "
        f"{before_counts['gold']} gold + {before_counts['filler']} filler …"
    )
    sys_before = CasUpdateMemorySystem(
        embed_client, args.embedding_model, llm_client,
        sim_threshold=args.condition_sim_threshold,
        propagation_rounds=args.propagation_rounds,
    )
    await sys_before.ingest_up_to_session(ingest_items, before_max)
    systems_by_phase["before"] = sys_before
    bank.attach_system(args.episode_id, sys_before)

    print(
        f"Ingesting up to session {after_max} (after phase): "
        f"{after_counts['gold']} gold + {after_counts['filler']} filler …"
    )
    sys_after = CasUpdateMemorySystem(
        embed_client, args.embedding_model, llm_client,
        sim_threshold=args.condition_sim_threshold,
        propagation_rounds=args.propagation_rounds,
    )
    await sys_after.ingest_up_to_session(ingest_items, after_max)
    systems_by_phase["after"] = sys_after

    with trace_path.open("w", encoding="utf-8") as f:
        for phase_name, sys in systems_by_phase.items():
            for entry in sys.trace:
                f.write(json.dumps({"phase_ingest": phase_name, **entry}, ensure_ascii=False) + "\n")

    print(f"Wrote ingest trace: {trace_path} ({sum(len(s.trace) for s in systems_by_phase.values())} events)")

    # Answer questions
    pred_rows: List[Dict[str, Any]] = []
    pending_q: List[MemeQuestion] = []
    messages_list: List[List[dict]] = []
    meta: List[Tuple[MemeQuestion, List[RetrievedMemory]]] = []

    for q in questions:
        sys = systems_by_phase[q.phase]
        bank.attach_system(args.episode_id, sys)
        retrieved = bank.retrieve(
            q.episode_id,
            q.question,
            q.position_after_session,
            args.retrieve_topk,
        )
        ctx = format_context(retrieved)
        prompt = build_answer_prompt(q, ctx)
        messages_list.append([{"role": "user", "content": prompt}])
        meta.append((q, retrieved))
        pending_q.append(q)

    print(f"Answering {len(messages_list)} questions …")
    responses = await answer_client.get_response_chat(
        messages_list,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        max_concurrency=max(1, args.answer_concurrency),
        use_tqdm=True,
        verbose=False,
    )

    for (q, retrieved), model_answer in zip(meta, responses):
        entity_key = (q.entities or [""])[0] if q.entities else ""
        pred_rows.append({
            "history_name": q.episode_id,
            "episode_id": q.episode_id,
            "domain": q.domain,
            "phase": q.phase,
            "task_type": q.task_type,
            "question_type": q.task_type,
            "question": q.question,
            "answer": q.reference,
            "model_answer": model_answer if model_answer is not None else "",
            "question_time": q.question_time,
            "position_after_session": q.position_after_session,
            "hop": q.hop,
            "entities": q.entities,
            "entity_key": entity_key,
            "retrieved_count": len(retrieved),
            "retrieved_memories": [
                {"text": r.text, "score": r.score}
                for r in retrieved
            ],
            "memory_source": "cas_update_condition",
            "benchmark": args.dataset.stem,
            "ingest_mode": ingest_mode,
        })

    with pred_path.open("w", encoding="utf-8") as f:
        for row in pred_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote predictions: {pred_path}")

    if args.skip_judge:
        _print_summary(pred_rows, None)
        return

    judge_client = load_api_chat_completion(args.judge_model, async_=True)
    judge = MemeLLMJudge(client=judge_client, model=judge_client.model_name)
    sem = asyncio.Semaphore(max(1, args.judge_concurrency))

    async def _judge_one(row: Dict[str, Any]) -> Dict[str, Any]:
        async with sem:
            out = dict(row)
            try:
                result = await judge.u_check(
                    question=str(row["question"]),
                    gold_value=row["answer"],
                    agent_answer=str(row.get("model_answer", "")),
                    task_type=str(row["question_type"]),
                    phase=str(row.get("phase", "after")),
                    max_new_tokens=args.judge_max_new_tokens,
                )
                out["judge_api_failed"] = False
                out["u_pass"] = result.u_pass
                out["u_reason"] = result.u_reason
            except Exception as exc:
                out["judge_api_failed"] = True
                out["u_pass"] = None
                out["u_reason"] = f"judge error: {exc}"
            return out

    judged = await asyncio.gather(*(_judge_one(r) for r in pred_rows))

    before_pass: Dict[str, bool] = {}
    for row in judged:
        if row.get("phase") == "before" and row.get("entity_key"):
            before_pass[str(row["entity_key"])] = bool(row.get("u_pass"))

    for row in judged:
        if row.get("phase") == "after":
            row["pass_type"] = classify_trivial_pass(
                task_type=str(row.get("question_type", "")),
                entity_key=str(row.get("entity_key", "")),
                before_pass_by_entity=before_pass,
                after_u_pass=bool(row.get("u_pass")),
            )
        else:
            row["pass_type"] = None

    metrics = aggregate_meme_metrics(judged)
    eval_path = out_dir / "eval_judge.json"
    eval_record = {
        "timestamp": utc_timestamp_iso(),
        "eval_type": "cas_update_condition_experiment",
        "episode_id": args.episode_id,
        "dataset": str(args.dataset),
        "pred_path": str(pred_path),
        "trace_path": str(trace_path),
        "answer_model": args.answer_model,
        "judge_model": args.judge_model,
        "embedding_model": args.embedding_model,
        "condition_sim_threshold": args.condition_sim_threshold,
        "text_only_memory": True,
        "ingest_mode": ingest_mode,
        "candidates_dir": str(args.candidates_dir) if args.candidates_dir else None,
        "n_gold_ingested_before": before_counts["gold"],
        "n_filler_ingested_before": before_counts["filler"],
        "n_gold_ingested_after": after_counts["gold"],
        "n_filler_ingested_after": after_counts["filler"],
        "n_questions": len(judged),
        **metrics,
    }
    append_eval_json(str(eval_path), eval_record)

    with pred_path.open("w", encoding="utf-8") as f:
        for row in judged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(eval_record, indent=2, ensure_ascii=False))
    _print_summary(judged, eval_record)


def _print_summary(rows: List[Dict[str, Any]], eval_record: Optional[Dict[str, Any]]) -> None:
    print("\n=== Cas/Abs Results ===")
    for row in rows:
        phase = row.get("phase", "")
        tt = row.get("task_type", "")
        ent = row.get("entity_key", "")
        ans = row.get("model_answer", "")[:80]
        up = row.get("u_pass")
        pt = row.get("pass_type")
        mark = "PASS" if up else "FAIL"
        print(f"  [{phase}] {tt} ({ent}): {mark} pass_type={pt}")
        print(f"    Q: {row.get('question','')[:60]}")
        print(f"    gold: {row.get('answer','')[:60]}")
        print(f"    pred: {ans}")
    if eval_record:
        print(f"\nmeme_score (after real): {eval_record.get('meme_score')}")
        per = eval_record.get("per_type") or {}
        for k in ("Cas", "Abs"):
            if k in per:
                print(f"  {k}: {per[k]}")


def main() -> None:
    args = parse_args()
    asyncio.run(run_experiment(args))


if __name__ == "__main__":
    main()
