import json
import logging
from typing import List
from .base import BaseBenchmark, MemoryEpisode, ChatSession, ChatTurn, QuestionItem

logger = logging.getLogger(__name__)

class EventBenchmark(BaseBenchmark):
    """
    针对 LifeMemBench_event 等事件类数据集的 Loader
    """
    def _load_data(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load Event data from {self.file_path}: {e}")
            raise
            
        for item in raw_data:
            # 1. 组装 Sessions
            sessions = []
            chat_times = item.get("chat_time", [])
            chat_histories = item.get("chat_history", [])
            
            for idx, turn_list in enumerate(chat_histories):
                session_time = chat_times[idx] if idx < len(chat_times) else ""
                
                turns = [
                    ChatTurn(
                        speaker=turn.get("speaker", "Unknown"), 
                        content=turn.get("content", "")
                    ) 
                    for turn in turn_list
                ]
                sessions.append(ChatSession(session_date=session_time, turns=turns))
            
            # 2. 组装 Questions
            qas = []
            for qa_data in item.get("qa", []):
                metadata = {}
                if "golden_option" in qa_data:
                    metadata["golden_option"] = qa_data["golden_option"]
                if "evidence_date" in qa_data:
                    metadata["evidence_date"] = qa_data["evidence_date"]
                if "timescale" in qa_data:
                    metadata["timescale"] = qa_data["timescale"]
                
                qas.append(QuestionItem(
                    question=qa_data.get("question", ""),
                    answer=str(qa_data.get("answer", "")),
                    question_time=qa_data.get("question_time", ""),
                    options=qa_data.get("options", None),
                    question_type=qa_data.get("question_type", None),
                    metadata=metadata
                ))
                
            # 3. 产生 Episode
            episode = MemoryEpisode(
                history_name=item.get("history_name", "unknown_history"),
                sessions=sessions,
                qas=qas
            )
            self.episodes.append(episode)
