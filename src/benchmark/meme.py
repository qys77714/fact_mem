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

Evaluation questions (qas) come from after_questions only; before_questions are stored
in episode-level metadata for optional trivial-pass filtering by downstream steps.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from .base import BaseBenchmark, ChatSession, ChatTurn, MemoryEpisode, QuestionItem

logger = logging.getLogger(__name__)


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

        # --- sessions ---
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

            # gold_facts is a list for evidence sessions, None for filler
            raw_gold_facts = raw_sess.get("gold_facts")
            gold_facts: List[Dict[str, Any]] = raw_gold_facts if isinstance(raw_gold_facts, list) else []

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

        # Build gold_answer lookup from tasks: (task_type, question_template) → gold_answer.
        # after_questions[].expected_answer is always None in the dataset; the authoritative
        # answer is stored in tasks[].gold_answer and matched via question text.
        gold_answer_lookup: Dict[tuple, str] = {}
        for task in ep.get("tasks", []):
            key = (task.get("type", ""), task.get("question_template", ""))
            gold_answer_lookup[key] = str(task.get("gold_answer", ""))

        # --- after_questions → qas ---
        qas: List[QuestionItem] = []
        after_q_block = ep.get("after_questions", {})
        after_timestamp = after_q_block.get("timestamp", "")
        for q in after_q_block.get("questions", []):
            task_type: str = q.get("task_type", "")
            question_text: str = q.get("question", "")
            # Look up gold_answer from tasks by (type, question_template)
            answer = gold_answer_lookup.get((task_type, question_text), "")
            qi_meta: Dict[str, Any] = {}
            if "hop" in q:
                qi_meta["hop"] = q["hop"]
            qas.append(
                QuestionItem(
                    question=question_text,
                    answer=answer,
                    question_time=after_timestamp,
                    question_type=task_type,
                    metadata=qi_meta,
                )
            )

        return MemoryEpisode(
            history_name=episode_id,
            sessions=sessions,
            qas=qas,
        )
