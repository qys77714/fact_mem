"""
MEMEBenchmark — loader for the MEME (Multi-Entity and Evolving Memory Evaluation) dataset.

Three variants are supported via DEFAULT_BENCHMARK_DATASETS:
  meme_nofiller   → data/raw_data/MEME/meme_nofiller.json   (100 episodes, evidence only)
  meme_filler32k  → data/raw_data/MEME/meme_filler32k.json  (100 episodes, ~32k filler)
  meme_filler128k → data/raw_data/MEME/meme_filler128k.json ( 40 episodes, ~128k filler)

Session metadata written into ChatSession.metadata:
  {
    "type":       "evidence" | "filler",
    "gold_facts": [{"fact_id", "entity", "value", "fact_text", ...}, ...],  # None for filler
  }

Evaluation questions include both before_questions and after_questions (MEME-public protocol):
  - phase=before: asked after sessions up to before position (pre-change state check)
  - phase=after:  asked after all sessions including change/delete events
  QuestionItem.metadata carries phase, max_session_index (1-based), entity_key, hop, etc.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseBenchmark, ChatSession, ChatTurn, MemoryEpisode, QuestionItem

logger = logging.getLogger(__name__)

TRIVIAL_PASS_TASKS = frozenset({"Cas", "Abs", "Del"})


def _task_base(task_type: str) -> str:
    return str(task_type or "").split(" (")[0]


def _entity_key_from_question(q: Dict[str, Any]) -> str:
    ev = q.get("entity_values") or {}
    if isinstance(ev, dict) and ev:
        return str(next(iter(ev.keys())))
    ent = q.get("entity")
    if isinstance(ent, list) and ent:
        return str(ent[0])
    if isinstance(ent, str) and ent:
        return ent
    return ""


def _resolve_gold_answer(
    q: Dict[str, Any],
    gold_answer_lookup: Dict[Tuple[str, str], str],
    phase: str,
) -> str:
    """Before questions use expected_answer; after questions match tasks by question text."""
    ref = q.get("expected_answer")
    if ref is not None and str(ref).strip():
        return str(ref)
    task_type = str(q.get("task_type", ""))
    question_text = str(q.get("question", ""))
    return gold_answer_lookup.get((task_type, question_text), "")


def _max_session_index_1based(position_after_session: int) -> int:
    """MEME position_after_session is 0-based last included session → 1-based inclusive cutoff."""
    return int(position_after_session) + 1


class MEMEBenchmark(BaseBenchmark):
    """
    Loader for MEME benchmark JSON files (all three variants share the same schema).

    Raw MEME files are loaded directly without a preprocessed intermediate file,
    since the session type / gold_facts metadata cannot be losslessly represented
    in the standard LME preprocessed format.
    """

    def _load_data(self) -> None:
        path = Path(self.file_path)
        if not path.exists():
            raise FileNotFoundError(f"MEME data file not found: {path}")

        logger.info("Loading MEME benchmark from %s", path)
        with path.open("r", encoding="utf-8") as f:
            raw_episodes: List[Dict[str, Any]] = json.load(f)

        for ep in raw_episodes:
            episode = self._convert_episode(ep)
            self.episodes.append(episode)

        logger.info("Loaded %d MEME episodes from %s", len(self.episodes), path)

    def _convert_episode(self, ep: Dict[str, Any]) -> MemoryEpisode:
        episode_id: str = ep["episode_id"]

        sessions: List[ChatSession] = []
        for raw_sess in ep.get("sessions", []):
            sess_type: str = raw_sess.get("type", "filler")
            timestamp: str = raw_sess.get("timestamp", "")
            conversation: List[Dict[str, Any]] = raw_sess.get("conversation", [])

            turns = [
                ChatTurn(
                    speaker=turn.get("role", "user"),
                    content=turn.get("content", ""),
                )
                for turn in conversation
            ]

            raw_gold_facts = raw_sess.get("gold_facts")
            gold_facts: List[Dict[str, Any]] = (
                raw_gold_facts if isinstance(raw_gold_facts, list) else []
            )

            session_meta: Dict[str, Any] = {
                "type": sess_type,
                "gold_facts": gold_facts,
            }
            sessions.append(
                ChatSession(
                    session_date=timestamp,
                    turns=turns,
                    metadata=session_meta,
                )
            )

        gold_answer_lookup: Dict[tuple, str] = {}
        for task in ep.get("tasks", []):
            key = (task.get("type", ""), task.get("question_template", ""))
            gold_answer_lookup[key] = str(task.get("gold_answer", ""))

        qas: List[QuestionItem] = []
        phase_blocks: Dict[str, Any] = {}

        for phase in ("before", "after"):
            block = ep.get(f"{phase}_questions", {})
            if not block or not isinstance(block, dict):
                continue
            pos = int(block.get("position_after_session", -1))
            q_time = str(block.get("timestamp", ""))
            max_sess = _max_session_index_1based(pos) if pos >= 0 else len(sessions)
            phase_blocks[phase] = {
                "position_after_session": pos,
                "max_session_index": max_sess,
                "timestamp": q_time,
            }
            for idx, q in enumerate(block.get("questions") or []):
                task_type: str = str(q.get("task_type", ""))
                question_text: str = str(q.get("question", ""))
                answer = _resolve_gold_answer(q, gold_answer_lookup, phase)
                entity_key = _entity_key_from_question(q)
                qi_meta: Dict[str, Any] = {
                    "phase": phase,
                    "question_id": f"{phase}:{_task_base(task_type)}:{idx}",
                    "max_session_index": max_sess,
                    "entity_key": entity_key,
                    "entity_values": q.get("entity_values") or {},
                }
                if "hop" in q:
                    qi_meta["hop"] = q["hop"]
                qas.append(
                    QuestionItem(
                        question=question_text,
                        answer=answer,
                        question_time=q_time,
                        question_type=task_type,
                        metadata=qi_meta,
                    )
                )

        return MemoryEpisode(
            history_name=episode_id,
            sessions=sessions,
            qas=qas,
            metadata={
                "domain": ep.get("domain", ""),
                "phase_blocks": phase_blocks,
                "root": ep.get("root", ""),
            },
        )
