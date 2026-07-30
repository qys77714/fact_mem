"""MEME benchmark loader.

将 ``meme_filler32k.json`` 转换为标准 ``MemoryEpisode`` 格式。
每 episode = 一个 MemoryEpisode，含 ~21 sessions 和 ~7 after questions。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from .base import BaseBenchmark, ChatSession, ChatTurn, MemoryEpisode, QuestionItem

logger = logging.getLogger(__name__)


class MEMEBenchmark(BaseBenchmark):
    """MEME (Multi-Entity Evolving Memory) 评测基准。

    原始 JSON 结构（每项一个 episode）：::

        {
          "episode_id": "pl_001",
          "domain": "personal_life",
          "root": "health_condition",
          "root_change": {"before": "...", "after": "..."},
          "entities": {...},
          "tasks": [...],
          "sessions": [
            {
              "session_id": "sharegpt_14177",
              "type": "filler",
              "timestamp": "2023/03/03 (Fri) 11:55",
              "conversation": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
              ]
            }
          ],
          "before_questions": {
            "timestamp": "2023/03/19 (Sun) 00:18",
            "position_after_session": 17,
            "questions": [{"task_type": "Cas", "question": "...", "expected_answer": "..."}]
          },
          "after_questions": {
            "timestamp": "2023/03/27 (Mon) 06:29",
            "position_after_session": 23,
            "questions": [{"task_type": "Tr", "question": "...", "gold_answer": "..."}]
          }
        }

    设计决策：
    - 只加载 after_questions（before 完全不参与）
    - extract 范围：全部 session（不限于 position_after_session 之前）
    - 每 episode 多 QA（LoCoMo 风格）
    """

    def _load_data(self) -> None:
        file_path = Path(self.file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"MEME data file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            raw_list: List[Dict[str, Any]] = json.load(f)

        if not isinstance(raw_list, list):
            raise ValueError(
                f"MEME data root must be a JSON array, got {type(raw_list).__name__}"
            )

        for raw_ep in raw_list:
            episode = self._convert_episode(raw_ep)
            self.episodes.append(episode)

        logger.info(
            "MEMEBenchmark: loaded %d episodes from %s",
            len(self.episodes),
            file_path,
        )

    @staticmethod
    def _convert_episode(raw: Dict[str, Any]) -> MemoryEpisode:
        """将一条 MEME episode 转换为 MemoryEpisode。"""

        # ---- 1. 解析 sessions ----
        sessions: List[ChatSession] = []
        for s in raw.get("sessions", []):
            turns: List[ChatTurn] = []
            for turn in s.get("conversation", []):
                turns.append(ChatTurn(
                    speaker=str(turn.get("role", "Unknown")).strip(),
                    content=str(turn.get("content", "")),
                ))

            session_meta: Dict[str, Any] = {
                "type": str(s.get("type", "")),
                "session_id": str(s.get("session_id", "")),
            }
            if s.get("evidence_type"):
                session_meta["evidence_type"] = s["evidence_type"]

            sessions.append(ChatSession(
                session_date=str(s.get("timestamp", "")),
                turns=turns,
                metadata=session_meta,
            ))

        # ---- 2. 解析 after_questions ----
        after_block: Dict[str, Any] = raw.get("after_questions", {})
        question_time = str(after_block.get("timestamp", ""))

        qas: List[QuestionItem] = []
        for q in after_block.get("questions", []):
            meta: Dict[str, Any] = {
                "entity": q.get("entity", []),
                "entity_values": q.get("entity_values", {}),
            }
            if "hop" in q:
                meta["hop"] = q["hop"]

            qas.append(QuestionItem(
                question=str(q.get("question", "")).strip(),
                answer=str(q.get("gold_answer", "")).strip(),
                question_time=question_time,
                question_type=str(q.get("task_type", "")).strip(),
                metadata=meta,
            ))

        # ---- 3. 组装 Episode ----
        episode_id = str(raw.get("episode_id", "")).strip()

        return MemoryEpisode(
            history_name=episode_id if episode_id else f"meme_{len(sessions)}sessions",
            sessions=sessions,
            qas=qas,
            metadata={
                "benchmark": "meme",
                "domain": str(raw.get("domain", "")),
                "root": str(raw.get("root", "")),
            },
        )
