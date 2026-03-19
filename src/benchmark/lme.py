import json
import logging
import uuid
from typing import Any
from datetime import datetime
from pathlib import Path
from .base import BaseBenchmark, MemoryEpisode, ChatSession, ChatTurn, QuestionItem
from utils.date_utils import parse_chat_time as _parse_chat_time

logger = logging.getLogger(__name__)

class LMEBenchmark(BaseBenchmark):
    """
    针对 LongMemEval 系列数据集的 Loader
    (适用于 lme_oracle, lme_m, lme_s 等)
    自动处理从 raw_data 到 preprocessed 的转换。
    """
    
    def __init__(self, file_path: str, lang: str = "en"):
        # 拦截 file_path 检查并在有需要时进行预处理
        raw_path = Path(file_path)
        
        # 匹配到 raw_data，自动导向到 preprocessed 目录
        if "raw_data" in raw_path.parts:
            parts = list(raw_path.parts)
            idx = parts.index("raw_data")
            parts[idx] = "preprocessed"
            # 为了防止冲突或是名称重复，我们把预处理后的文件在原名的基础上补 _converted
            if "converted" not in raw_path.stem:
                self.preprocessed_path = Path(*parts).with_name(f"{raw_path.stem}_converted.json")
            else:
                self.preprocessed_path = Path(*parts)
                
            self._preprocess_if_needed(raw_path, self.preprocessed_path)
            
            # 使用预处理后的路径调用父类加载逻辑
            super().__init__(str(self.preprocessed_path), lang)
        else:
            # 否则，它可能是直接传的 preprocessed 文件
            super().__init__(file_path, lang)
            
    def _preprocess_if_needed(self, raw_path: Path, preprocessed_path: Path):
        """将原始 A 格式数据转换为 B 格式并保存至 preprocessed 文件夹"""
        if preprocessed_path.exists():
            logger.info(f"Preprocessed file already exists: {preprocessed_path}, skipping conversion.")
            return

        logger.info(f"Converting raw file {raw_path} to {preprocessed_path}")
        with raw_path.open("r", encoding="utf-8") as f:
            data_a = json.load(f)

        data_b = self._convert_dataset_a_to_b(data_a)

        preprocessed_path.parent.mkdir(parents=True, exist_ok=True)
        with preprocessed_path.open("w", encoding="utf-8") as f:
            json.dump(data_b, f, ensure_ascii=False, indent=2)

    def _convert_dataset_a_to_b(self, data_a: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """将 LongMemEval raw_data 转换为标准 preprocessed 格式。"""
        data_b: list[dict[str, Any]] = []

        for item in data_a:
            generated_question_id = str(uuid.uuid4())
            question = item.get("question", "")
            answer = item.get("answer", "")
            question_date = item.get("question_date", "")
            question_type = item.get("question_type", None)
            options = item.get("options", None)
            golden_option = item.get("golden_option", None)

            haystack_dates = item.get("haystack_dates", []) or []
            haystack_sessions = item.get("haystack_sessions", []) or []

            # 先按索引配对时间与会话，再按时间升序排序
            paired = []
            max_len = max(len(haystack_dates), len(haystack_sessions))
            for i in range(max_len):
                chat_time = haystack_dates[i] if i < len(haystack_dates) else ""
                session = haystack_sessions[i] if i < len(haystack_sessions) else []

                converted_session = []
                for turn in session:
                    metadata = {}
                    if "has_answer" in turn:
                        metadata["has_answer"] = turn.get("has_answer")

                    converted_session.append(
                        {
                            "speaker": turn.get("role", "Unknown"),
                            "content": turn.get("content", ""),
                            "metadata": metadata,
                        }
                    )
                paired.append((chat_time, converted_session))

            paired.sort(key=lambda x: _parse_chat_time(x[0]))

            qa_dict: dict[str, Any] = {
                "question_id": generated_question_id,
                "question": question,
                "answer": str(answer),
                "question_time": question_date,
                "question_type": question_type,
            }
            if options is not None:
                qa_dict["options"] = options
            if golden_option is not None:
                qa_dict["golden_option"] = golden_option

            chat_time_dict: dict[str, str] = {}
            chat_history_dict: dict[str, list[dict[str, Any]]] = {}
            for idx, (chat_time, converted_session) in enumerate(paired, start=1):
                session_key = f"session_{idx}"
                chat_time_dict[session_key] = chat_time
                chat_history_dict[session_key] = converted_session

            b_item = {
                "history_id": item.get("question_id", "unknown_id"),
                "QAs": [qa_dict],
                "chat_time": chat_time_dict,
                "chat_history": chat_history_dict,
            }
            data_b.append(b_item)

        return data_b

    def _load_data(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load LongMemEval data from {self.file_path}: {e}")
            raise
            
        for item in raw_data:
            # 1. 组装 Sessions
            sessions = []
            chat_times = item.get("chat_time", [])
            chat_histories = item.get("chat_history", [])

            # 新标准格式: dict(session_i -> ...)
            if isinstance(chat_times, dict) and isinstance(chat_histories, dict):
                def _session_order(k: str) -> tuple[int, str]:
                    if isinstance(k, str) and k.startswith("session_"):
                        tail = k.split("session_", 1)[1]
                        if tail.isdigit():
                            return (int(tail), k)
                    return (10**9, str(k))

                session_keys = sorted(chat_histories.keys(), key=_session_order)
                for sk in session_keys:
                    turn_list = chat_histories.get(sk, []) or []
                    session_time = chat_times.get(sk, "")

                    turns = []
                    for turn in turn_list:
                        speaker = turn.get("speaker", turn.get("role", "Unknown"))
                        metadata = turn.get("metadata", {}) or {}
                        if "has_answer" in turn and "has_answer" not in metadata:
                            metadata["has_answer"] = turn.get("has_answer")

                        turns.append(ChatTurn(
                            speaker=speaker,
                            content=turn.get("content", ""),
                            metadata=metadata
                        ))
                    sessions.append(ChatSession(session_date=session_time, turns=turns))
            else:
                # 旧格式: list
                for idx, turn_list in enumerate(chat_histories):
                    # 兼容部分数据 chat_time 长度不一
                    session_time = chat_times[idx] if idx < len(chat_times) else ""

                    turns = []
                    for turn in turn_list:
                        speaker = turn.get("speaker", turn.get("role", "Unknown"))
                        turns.append(ChatTurn(
                            speaker=speaker,
                            content=turn.get("content", ""),
                            metadata={"has_answer": turn.get("has_answer")} if "has_answer" in turn else {}
                        ))
                    sessions.append(ChatSession(session_date=session_time, turns=turns))

            # 2. 组装 Questions
            qas = []
            qa_items = item.get("QAs", item.get("qa", []))
            for qa_data in qa_items:
                metadata = {}
                if "question_id" in qa_data:
                    metadata["question_id"] = qa_data.get("question_id")
                if "golden_option" in qa_data:
                    metadata["golden_option"] = qa_data.get("golden_option")

                qas.append(QuestionItem(
                    question=qa_data.get("question", ""),
                    answer=str(qa_data.get("answer", "")),
                    question_time=qa_data.get("question_time", ""),
                    options=qa_data.get("options", None),
                    question_type=qa_data.get("question_type", None),
                    metadata=metadata
                ))
                
            # 3. 产生 Episode
            history_name = item.get("history_id", item.get("history_name", item.get("question_id", "unknown_history")))
            
            episode = MemoryEpisode(
                history_name=history_name,
                sessions=sessions,
                qas=qas
            )
            self.episodes.append(episode)
