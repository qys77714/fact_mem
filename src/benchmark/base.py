from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

@dataclass
class ChatTurn:
    """单轮对话"""
    speaker: str
    content: str
    time: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ChatSession:
    """单次对话会话 (Session)"""
    session_date: str
    turns: List[ChatTurn]

@dataclass
class QuestionItem:
    """单个测试问题"""
    question: str
    answer: str  # 真实答案 (用于离线对齐衡量)
    question_time: str
    options: Optional[List[str]] = None  # 如果是选择题则有选项
    question_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MemoryEpisode:
    """
    一个完整的评测用例 (Episode/Case)
    包含了记忆喂入阶段 (history) 和 抽查答疑阶段 (qas)
    """
    history_name: str  # 对应原代码的 question_id 或 history_name
    sessions: List[ChatSession]  # 按时间线排列的记忆块
    qas: List[QuestionItem]      # 需要模型回答的问题列表

class BaseBenchmark:
    """所有 Benchmark 的基类"""
    def __init__(self, file_path: str, lang: str = "en"):
        self.file_path = file_path
        self.lang = lang
        self.episodes: List[MemoryEpisode] = []
        self._load_data()

    def _load_data(self):
        """
        子类必须实现此方法：读取 self.file_path，
        并将解析后的对象存入 self.episodes
        """
        raise NotImplementedError

    def __iter__(self):
        """让 Benchmark 可以直接被 for 循环遍历"""
        return iter(self.episodes)
    
    def __len__(self):
        return len(self.episodes)
