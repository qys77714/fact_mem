from typing import List, Optional
from benchmark.base import QuestionItem
from memory.base import RetrievedMemory

class BaseAgent:
    """
    智能体的核心抽象类。
    它的主要职责是：接收来自数据集的问题（QuestionItem），
    调用底层的 Memory System 获取相关背景，组装成 Prompt 给到大模型，最后返回生成的答案字符串。
    """
    def answer_question(
        self,
        history_name: str,
        question: QuestionItem,
        top_k: int = 5
    ) -> str:
        raise NotImplementedError

    async def batch_answer_questions(
        self,
        history_name: str,
        questions: List[QuestionItem],
        top_k: int = 5
    ) -> List[str]:
        raise NotImplementedError
