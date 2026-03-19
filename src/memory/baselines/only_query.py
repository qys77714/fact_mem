from typing import List, Optional
from benchmark.base import ChatSession
from memory.base import BaseMemorySystem, RetrievedMemory

class OnlyQueryMemorySystem(BaseMemorySystem):
    """
    无任何外部记忆的最基础 Baseline (类似于 Zero-Shot)。
    只用来做对照消融实验。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def store_session(self, history_name: str, session_idx: int, session: ChatSession) -> None:
        """什么也不保存"""
        pass

    def retrieve(self, history_name: str, query: str, current_time: str, top_k: int = 5) -> List[RetrievedMemory]:
        """每次返回空"""
        return []
