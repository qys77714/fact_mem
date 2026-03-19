from typing import List, Iterable, Union
import numpy as np
from memory.base import BaseMemorySystem, RetrievedMemory
from memory.storage.local_faiss import LocalFaissDatabase
from benchmark.base import ChatSession

class RagMemorySystem(BaseMemorySystem):
    """
    RAG 基线记忆系统：
    不对对话进行反思或压缩，简单地将对话切分并带 Embedding 存入。
    由本系统负责调用外部 embedding 接口，并进行批量提速。
    
    granularity 控制切分粒度:
    - "all": 整个 session 作为一条记忆存入。
    - int (如 1, 3, 5): 将 session 每 N 个 turn 打包在一起作为一条记忆存入 (1 相当于逐 turn 存入)。
    """
    def __init__(self, granularity: Union[str, int] = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(granularity, "isdigit") and granularity.isdigit():
            granularity = int(granularity)
            
        if granularity != "all" and not isinstance(granularity, int):
            raise ValueError("granularity 必须是 'all' 或者正整数。")
        if isinstance(granularity, int) and granularity <= 0:
            raise ValueError("作为整数的 granularity 必须大于 0。")
            
        self.granularity = granularity
        self._databases = {}

    def _get_database(self, history_name: str) -> LocalFaissDatabase:
        namespace = f"rag_{self.granularity}_{history_name}"
        if namespace not in self._databases:
            self._databases[namespace] = LocalFaissDatabase(
                namespace=namespace,
                database_root=self.database_root
            )
        return self._databases[namespace]

    def _embed_texts(self, inputs: Iterable[str]) -> np.ndarray:
        from utils.embed_utils import embed_texts
        return embed_texts(self.embed_client, inputs, self.embed_model_name)

    def store_session(self, history_name: str, session_idx: int, session: ChatSession) -> None:
        db = self._get_database(history_name)
        
        pending_items = []
        
        if self.granularity == "all":
            mem_content = "\n".join(f"**{turn.speaker}**: {turn.content}" for turn in session.turns)
            if mem_content.strip():
                text = mem_content.strip()
                source_index = f"session_{session_idx}"
                time = session.session_date
                metadata = {"turns": len(session.turns), "granularity": "all"}
                pending_items.append((text, source_index, time, metadata))

        elif isinstance(self.granularity, int):
            chunk_size = self.granularity
            turns = session.turns
            
            for i in range(0, len(turns), chunk_size):
                chunk_turns = turns[i : i + chunk_size]
                chunk_content = "\n".join(
                    f"**{t.speaker}**: {t.content.strip()}" 
                    for t in chunk_turns if t.content.strip()
                )
                if not chunk_content: 
                    continue
                    
                text = chunk_content
                # 表示这一块来源于哪几句
                end_idx = i + len(chunk_turns) - 1
                source_index = f"session_{session_idx}-turn_{i}_to_{end_idx}" if chunk_size > 1 else f"session_{session_idx}-turn_{i}"
                
                # 时间取这块的第一句的时间，或者使用整个 session 的时间
                time = chunk_turns[0].time if chunk_turns[0].time else session.session_date
                metadata = {"speaker": chunk_turns[0].speaker, "granularity": f"chunk_{chunk_size}"}
                
                pending_items.append((text, source_index, time, metadata))

        if not pending_items:
            return

        # 核心优化：一次性批量计算整个 session/turns 的 embedding
        # RAG 使用基类默认：仅 text，不拼接 metadata
        texts_to_embed = [
            self.build_text_for_embedding(text, metadata=metadata)
            for text, _, _, metadata in pending_items
        ]
        embeddings = self._embed_texts(texts_to_embed)

        # 配合向量将其塞入底层数据库
        for idx, (text, source_index, time, metadata) in enumerate(pending_items):
            db.add(text, source_index, time, metadata, embedding=embeddings[idx])

    def retrieve(self, history_name: str, query: str, current_time: str, top_k: int = 5) -> List[RetrievedMemory]:
        db = self._get_database(history_name)
        # 获取查询的 embedding
        query_embedding = self._embed_texts([query])
        if query_embedding.size == 0:
            return []
        return db.search(query_embedding[0], top_k)
