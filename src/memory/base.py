from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
from benchmark.base import ChatSession


def _session_progress_tick(session_progress: Optional[Any], n: int = 1) -> None:
    """If ``session_progress`` has an ``update`` method (e.g. tqdm), call ``update(n)``."""
    if session_progress is None or n <= 0:
        return
    update = getattr(session_progress, "update", None)
    if callable(update):
        update(n)


@dataclass
class RetrievedMemory:
    """检索到的单条记忆"""
    memory_id: str
    text: str              # 记忆主体内容
    source_index: str      # 原文索引 (如: session_1-turn_0)
    time: str              # 该记忆代表的时间
    score: float           # 检索相似度得分
    metadata: Dict[str, Any] = field(default_factory=dict) # 其他相关元信息

class BaseMemorySystem:
    """所有记忆体系统的抽象基类。
    它的工作只包含：提取记忆（store_session）和召回记忆（retrieve）。不包含拼装 Prompt 和调用生成答案。
    """
    
    def __init__(
        self, 
        embed_client=None, 
        embed_model_name: str = "default",
        llm_client=None,                 # 有的Memory系统(如 amem)由于需要在存入前压缩反思，因此持有生成模型 client
        database_root: Optional[str] = None
    ):
        self.embed_client = embed_client
        self.embed_model_name = embed_model_name
        self.llm_client = llm_client
        self.database_root = database_root
        
    def store_session(self, history_name: str, session_idx: int, session: ChatSession) -> None:
        """
        接收一条新会话并将其结构化地存入记忆系统。
        不同系统有不同切分策略：
        - RAG系统：往往将每一次 turn 作为一条记录直接存入，或者每 N 个 turn 切分为一条块。
        - AMem系统 / Mem0系统：调用 LLM 对 Session 进行反思，抽取出 Insight 之后再存入。
        
        :param history_name: 用户或该段对话历史的 ID (如: conv-26 或 user_1)
        :param session_idx: 该会话在整个对话生命周期中的序号
        :param session: 我们刚刚写的标准化的 ChatSession 结构
        """
        raise NotImplementedError

    def store_episode(
        self,
        history_name: str,
        sessions: List[ChatSession],
        *,
        session_progress: Optional[Any] = None,
    ) -> None:
        """按 session 顺序写入一整段 episode（默认实现为逐个 store_session）。"""
        for session_idx, session in enumerate(sessions, start=1):
            self.store_session(history_name, session_idx, session)
            _session_progress_tick(session_progress, 1)

    def retrieve(self, history_name: str, query: str, current_time: str, top_k: int = 5) -> List[RetrievedMemory]:
        """
        根据问题和此时此刻的时间，从数据库检索相关记忆。
        :param history_name: 要检索的对应历史 ID
        :param query: 用户提出的问题或用于查询的具体语句
        :param current_time: 发出该提问的时间，用以做记忆时效性惩罚等
        :param top_k: 需要获取的条数
        """
        raise NotImplementedError

    def clear(self, history_name: str) -> None:
        """
        清理当前用户(history_name)的缓存或数据库空间，例如重置该用户的上下文状态。
        """
        pass

    def episode_storage_path(self, history_name: str) -> Optional[Path]:
        """
        若该记忆系统把 episode 持久化到磁盘，返回其向量库目录（与 LocalFaissDatabase 的 dataset 目录一致）；
        无持久化（如 only_query）则返回 None，流水线将把记忆视为「无需标记、始终就绪」且跳过写 marker。
        """
        return None

    def memory_ready_marker_path(self, history_name: str) -> Optional[Path]:
        base = self.episode_storage_path(history_name)
        return (base / ".memory_ready.json") if base is not None else None

    def persisted_data_root(self) -> Path:
        """与 LocalFaissDatabase 一致的 database_root 解析（用于拼接 namespace 目录）。"""
        return Path(self.database_root) if self.database_root else Path("MemDB/LocalStore")

    def build_text_for_embedding(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """
        构建用于 embedding 的文本。子类可重写以自定义策略。
        - mem0: 仅 text
        - amem: text + metadata (context, keywords, tags, summary)
        - RAG: 仅 text
        """
        return text

    def format_retrieved_for_context(
        self, retrieved: List[RetrievedMemory], language: str = "zh"
    ) -> str:
        """
        将检索到的记忆列表组装成用于回答问题的 context_block 字符串。
        子类可重写此方法，以自定义组装方式（如加入 time、metadata 等）。

        :param retrieved: 检索到的记忆列表
        :param language: 语言 ("zh" 或 "en")
        :return: 组装好的 context 字符串
        """
        from prompts import render_prompt

        if not retrieved:
            template = "agent_context_empty_zh.jinja" if language == "zh" else "agent_context_empty_en.jinja"
            return render_prompt(template)

        unit_template = "agent_context_unit_zh.jinja" if language == "zh" else "agent_context_unit_en.jinja"
        context_lines = [
            render_prompt(unit_template, index=idx + 1, text=item.text)
            for idx, item in enumerate(retrieved)
        ]
        return "\n\n".join(context_lines)
