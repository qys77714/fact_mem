"""LoCoMo benchmark loader.

将 ``locomo10.json``（10 段双人长对话）转换为标准 ``MemoryEpisode`` 格式。
每段 conversation = 1 个 episode，包含按时间排序的 session 对话和 QA 题目。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from .base import BaseBenchmark, ChatSession, ChatTurn, MemoryEpisode, QuestionItem

logger = logging.getLogger(__name__)


def _session_sort_key(key: str) -> int:
    """从 'session_N' 提取数字 N 用于排序。"""
    if key.startswith("session_"):
        tail = key[len("session_"):]
        if tail.isdigit():
            return int(tail)
    return 10 ** 9


class LoCoMoBenchmark(BaseBenchmark):
    """LoCoMo 双人对话评测基准。

    原始 JSON 结构（每项一条 conversation）：::

        {
          "sample_id": "conv-0",
          "conversation": {
            "speaker_a": "Caroline",
            "speaker_b": "Melanie",
            "session_1": [{"speaker": "Caroline", "dia_id": "D1:1", "text": "..."}, ...],
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            ...
          },
          "qa": [
            {"question": "When did Caroline ...?", "answer": "7 May 2023",
             "evidence": ["D1:3"], "category": 2},
            ...
          ]
        }
    """

    def _load_data(self) -> None:
        file_path = Path(self.file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"LoCoMo data file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            raw_list: List[Dict[str, Any]] = json.load(f)

        if not isinstance(raw_list, list):
            raise ValueError(f"LoCoMo data root must be a JSON array, got {type(raw_list).__name__}")

        for conv in raw_list:
            episode = self._parse_conversation(conv)
            self.episodes.append(episode)

        logger.info("LoCoMoBenchmark: loaded %d episodes from %s", len(self.episodes), file_path)

    def _parse_conversation(self, conv: Dict[str, Any]) -> MemoryEpisode:
        conv_data: Dict[str, Any] = conv.get("conversation", {})
        speaker_a = str(conv_data.get("speaker_a", "")).strip()
        speaker_b = str(conv_data.get("speaker_b", "")).strip()

        # ---- 1. 收集 session（只取有对话数据的，跳过仅有 date_time 的）----
        session_keys: List[str] = []
        for key in conv_data:
            if key.startswith("session_") and not key.endswith("_date_time"):
                if isinstance(conv_data[key], list) and len(conv_data[key]) > 0:
                    session_keys.append(key)
        session_keys.sort(key=_session_sort_key)

        sessions: List[ChatSession] = []
        for sk in session_keys:
            dt_key = f"{sk}_date_time"
            session_date = str(conv_data.get(dt_key, "")).strip()
            turns_raw: List[Dict[str, Any]] = conv_data.get(sk, [])
            if not isinstance(turns_raw, list):
                turns_raw = []

            turns: List[ChatTurn] = []
            for turn in turns_raw:
                speaker = str(turn.get("speaker", "Unknown")).strip()
                text = str(turn.get("text", "")).strip()
                dia_id = str(turn.get("dia_id", "")).strip()
                blip_caption = str(turn.get("blip_caption", "")).strip()
                img_url = turn.get("img_url", None)

                metadata: Dict[str, Any] = {"dia_id": dia_id}
                if blip_caption:
                    metadata["blip_caption"] = blip_caption
                if img_url:
                    metadata["img_url"] = img_url

                turns.append(ChatTurn(
                    speaker=speaker,
                    content=text,
                    metadata=metadata,
                ))

            sessions.append(ChatSession(
                session_date=session_date,
                turns=turns,
                metadata={"session_key": sk},
            ))

        # ---- 2. 组装 QA ----
        qa_list: List[Dict[str, Any]] = conv.get("qa", [])
        if not isinstance(qa_list, list):
            qa_list = []

        qas: List[QuestionItem] = []
        for qa in qa_list:
            evidence = qa.get("evidence", [])
            if not isinstance(evidence, list):
                evidence = []

            qas.append(QuestionItem(
                question=str(qa.get("question", "")).strip(),
                answer=str(qa.get("answer", "")).strip(),
                question_time="",  # LoCoMo QA 无独立 question_date
                question_type=str(qa.get("category", "")).strip(),
                metadata={
                    "evidence": evidence,
                },
            ))

        # ---- 3. 组装 Episode ----
        sample_id = str(conv.get("sample_id", "")).strip()
        history_name = sample_id if sample_id else f"conv_{speaker_a}_{speaker_b}"

        return MemoryEpisode(
            history_name=history_name,
            sessions=sessions,
            qas=qas,
            metadata={
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "benchmark": "locomo",
            },
        )
