import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any
from .base import BaseBenchmark, MemoryEpisode, ChatSession, ChatTurn, QuestionItem

logger = logging.getLogger(__name__)


def _session_index_from_key(key: str) -> int:
    match = re.match(r"^session_(\d+)$", str(key))
    if not match:
        return 10**9
    return int(match.group(1))


def _map_locomo_category_to_question_type(category: Any) -> str | None:
    # 按用户提供规则: 0~4
    mapping = {
        0: "single-hop",
        1: "multi-hop",
        2: "temporal-reasoning",
        3: "open-domain-knowledge",
        4: "adversarial",
        # 兼容部分数据可能出现的 5
        5: "adversarial",
    }
    if isinstance(category, int):
        return mapping.get(category)

    if isinstance(category, str) and category.isdigit():
        return mapping.get(int(category))

    return None

class LocomoBenchmark(BaseBenchmark):
    """
    针对 LoCoMo 数据集的 Loader
    """

    def __init__(self, file_path: str, lang: str = "en"):
        raw_path = Path(file_path)

        if "raw_data" in raw_path.parts:
            parts = list(raw_path.parts)
            idx = parts.index("raw_data")
            parts[idx] = "preprocessed"
            if "converted" not in raw_path.stem:
                self.preprocessed_path = Path(*parts).with_name(f"{raw_path.stem}_converted.json")
            else:
                self.preprocessed_path = Path(*parts)

            self._preprocess_if_needed(raw_path, self.preprocessed_path)
            super().__init__(str(self.preprocessed_path), lang)
        else:
            super().__init__(file_path, lang)

    def _preprocess_if_needed(self, raw_path: Path, preprocessed_path: Path):
        if preprocessed_path.exists():
            logger.info(f"Preprocessed file already exists: {preprocessed_path}, skipping conversion.")
            return

        logger.info(f"Converting raw file {raw_path} to {preprocessed_path}")
        with raw_path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)

        converted = self._convert_raw_to_standard(raw_data)

        preprocessed_path.parent.mkdir(parents=True, exist_ok=True)
        with preprocessed_path.open("w", encoding="utf-8") as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)

    def _convert_raw_to_standard(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []

        for item in raw_data:
            conversation = item.get("conversation", {}) or {}
            sample_id = item.get("sample_id", "unknown_history")

            qa_list = item.get("qa", []) or []

            # evidence 合并成一组，用于 turn-level has_answer
            evidence_set: set[str] = set()
            for qa in qa_list:
                for e in (qa.get("evidence", []) or []):
                    if isinstance(e, str):
                        evidence_set.add(e)

            qas: list[dict[str, Any]] = []
            for qa in qa_list:
                # 兼容 category / categoty
                category = qa.get("category", qa.get("categoty", None))

                qa_item: dict[str, Any] = {
                    "question_id": str(uuid.uuid4()),
                    "question": qa.get("question", ""),
                    "answer": str(qa.get("answer", "")),
                    "question_time": "",
                    "question_type": _map_locomo_category_to_question_type(category),
                }
                if qa.get("options") is not None:
                    qa_item["options"] = qa.get("options")
                if qa.get("golden_option") is not None:
                    qa_item["golden_option"] = qa.get("golden_option")
                qas.append(qa_item)

            chat_time: dict[str, str] = {}
            chat_history: dict[str, list[dict[str, Any]]] = {}

            # 找到所有 session_n
            session_keys = [
                k for k in conversation.keys()
                if isinstance(k, str) and re.match(r"^session_\d+$", k)
            ]
            session_keys = sorted(session_keys, key=_session_index_from_key)

            for idx, old_session_key in enumerate(session_keys, start=1):
                new_session_key = f"session_{idx}"
                date_key = f"{old_session_key}_date_time"

                chat_time[new_session_key] = conversation.get(date_key, "")

                turns = []
                for turn in (conversation.get(old_session_key, []) or []):
                    dia_id = turn.get("dia_id", "")
                    metadata = {
                        "dia_id": dia_id,
                        "has_answer": dia_id in evidence_set,
                    }

                    # 保留其余 dataset-specific 字段到 metadata
                    for k, v in turn.items():
                        if k not in {"speaker", "text", "dia_id"}:
                            metadata[k] = v

                    turns.append(
                        {
                            "speaker": turn.get("speaker", "Unknown"),
                            "content": turn.get("text", ""),
                            "metadata": metadata,
                        }
                    )

                chat_history[new_session_key] = turns

            converted.append(
                {
                    "history_id": str(sample_id),
                    "QAs": qas,
                    "chat_time": chat_time,
                    "chat_history": chat_history,
                }
            )

        return converted

    def _load_data(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load LoCoMo data from {self.file_path}: {e}")
            raise
            
        for item in raw_data:
            # 1. 组装 Sessions
            sessions = []
            chat_times = item.get("chat_time", [])
            chat_histories = item.get("chat_history", [])

            if isinstance(chat_times, dict) and isinstance(chat_histories, dict):
                session_keys = sorted(chat_histories.keys(), key=_session_index_from_key)
                for sk in session_keys:
                    turn_list = chat_histories.get(sk, []) or []
                    session_time = chat_times.get(sk, "")

                    turns = []
                    for turn in turn_list:
                        turns.append(ChatTurn(
                            speaker=turn.get("speaker", "Unknown"),
                            content=turn.get("content", turn.get("text", "")),
                            metadata=turn.get("metadata", {}) or {}
                        ))
                    sessions.append(ChatSession(session_date=session_time, turns=turns))
            else:
                # 兼容旧 list 格式
                for idx, turn_list in enumerate(chat_histories):
                    chat_time = chat_times[idx] if idx < len(chat_times) else ""

                    turns = [
                        ChatTurn(
                            speaker=turn.get("speaker", "Unknown"),
                            content=turn.get("content", turn.get("text", "")),
                            metadata=turn.get("metadata", {}) or {}
                        )
                        for turn in turn_list
                    ]
                    sessions.append(ChatSession(session_date=chat_time, turns=turns))
            
            # 2. 组装 Questions
            qas = []
            qa_items = item.get("QAs", item.get("qa", []))
            for qa_data in qa_items:
                metadata = {}
                if "category" in qa_data:
                    metadata["category"] = qa_data["category"]
                if "categoty" in qa_data:
                    metadata["categoty"] = qa_data["categoty"]
                if "evidence" in qa_data:
                    metadata["evidence"] = qa_data["evidence"]
                if "question_id" in qa_data:
                    metadata["question_id"] = qa_data["question_id"]
                if "golden_option" in qa_data:
                    metadata["golden_option"] = qa_data["golden_option"]

                qas.append(QuestionItem(
                    question=qa_data.get("question", ""),
                    answer=str(qa_data.get("answer", "")),  # 强制转str, 有时答案是数字
                    question_time=qa_data.get("question_time", ""), # LoCoMo如果缺失则为空
                    options=qa_data.get("options", None),
                    question_type=qa_data.get(
                        "question_type",
                        _map_locomo_category_to_question_type(qa_data.get("category", qa_data.get("categoty", None)))
                    ),
                    metadata=metadata
                ))
                
            # 3. 产生 Episode
            episode = MemoryEpisode(
                history_name=item.get("history_id", item.get("history_name", item.get("sample_id", "unknown_history"))),
                sessions=sessions,
                qas=qas
            )
            self.episodes.append(episode)
