from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from memory.base import BaseMemorySystem, RetrievedMemory
from memory.storage.local_faiss import LocalFaissDatabase
from benchmark.base import ChatSession

class FullContextMemorySystem(BaseMemorySystem):
    """
    全量上下文基线记忆系统：
    把整个 Session 当成大文本存入（不需要 Embedding）。
    Retrieve 时忽略 Query 和 Top_k，直接将库里所有的记录按时间返回拼成终极长 Prompt。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._databases = {}

    def episode_storage_path(self, history_name: str) -> Optional[Path]:
        return self.persisted_data_root() / history_name

    def _get_database(self, history_name: str) -> LocalFaissDatabase:
        namespace = history_name
        if namespace not in self._databases:
            self._databases[namespace] = LocalFaissDatabase(
                namespace=namespace,
                database_root=self.database_root
            )
        return self._databases[namespace]

    def store_session(self, history_name: str, session_idx: int, session: ChatSession) -> None:
        db = self._get_database(history_name)
        
        mem_content = "\n".join(f"**{turn.speaker}**: {turn.content}" for turn in session.turns)
        if not mem_content.strip():
            return
            
        text = mem_content.strip()
        source_index = f"session_{session_idx}"
        time = session.session_date
        metadata = {"turns": len(session.turns)}
        
        db.add(text, source_index, time, metadata)

    def retrieve(self, history_name: str, query: str, current_time: str, top_k: int = 5) -> List[RetrievedMemory]:
        db = self._get_database(history_name)
        # 不做筛选，直接把所有的原样全给你拿出来
        # 返回的所有记忆最好按时间从新到旧（或者由业务层决定）
        # 这里返回 descending=False 也就是升序排列，让最早的记忆在前面
        return db.list_all_memories(sort_by_time=True, descending=False)
