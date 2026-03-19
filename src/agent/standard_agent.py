from typing import List, Optional

from benchmark.base import QuestionItem
from memory.base import BaseMemorySystem, RetrievedMemory
from memory.tracing import MemoryTraceLogger
from .base import BaseAgent
from prompts import render_prompt

# 假定我们需要一个 tokenizer 来截断上下文
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

def trim_context(context: str, tokenizer, max_tokens: int) -> str:
    if tokenizer is None:
        return context
    
    # 简单的截断逻辑
    encoded = tokenizer(context, add_special_tokens=False, return_attention_mask=False)
    token_ids = encoded.get("input_ids", [])
    original_len = len(token_ids)

    if original_len <= max_tokens:
        return context

    trimmed_ids = token_ids[:max_tokens]
    trimmed_text = tokenizer.decode(trimmed_ids, skip_special_tokens=True)
    return trimmed_text

class StandardAgent(BaseAgent):
    """
    最标准的 QA Agent：
    1. 根据问题从 memory_system 获取 context
    2. 依据 context 和 QA 文本组装 Prompt
    3. 调用传入的 chat_model 进行推理。
    """
    def __init__(
        self,
        memory_system: BaseMemorySystem,
        chat_model,
        memory_token_limit: int = 2048,
        language: str = "zh",
        trace_log_dir: Optional[str] = None,
    ):
        self.memory_system = memory_system
        self.chat_model = chat_model
        self.memory_token_limit = memory_token_limit
        self.language = language
        self.trace: Optional[MemoryTraceLogger] = (
            MemoryTraceLogger(method="agent", log_dir=trace_log_dir) if trace_log_dir else None
        )

        if AutoTokenizer is not None:
            # 默认使用 Qwen tokenizer 近似统计
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        else:
            self.tokenizer = None

    def _build_prompt(self, question: QuestionItem, context_block: str) -> str:
        options_text = "\n".join(question.options) if question.options else ""
        if self.language == "zh":
            template = "agent_prompt_zh_mcq.jinja" if question.options else "agent_prompt_zh_open.jinja"
        else:
            template = "agent_prompt_en_mcq.jinja" if question.options else "agent_prompt_en_open.jinja"

        return render_prompt(
            template,
            context_block=context_block,
            question_time=question.question_time,
            question=question.question,
            options_text=options_text,
        )

    async def batch_answer_questions(
        self,
        history_name: str,
        questions: List[QuestionItem],
        top_k: int = 5
    ) -> List[str]:
        messages_list: List[List[dict]] = []
        # tracing 用：每道题对应 (question, retrieved, prompt)
        trace_data: List[tuple] = []

        for q in questions:
            # 1. 构造用于去数据库里检索的真实查询语句（可以带上时间增强）
            q_for_retrieval = (
                f"- 当前日期: {q.question_time}\n" if self.language == "zh" else f"- Current Date: {q.question_time}\n"
            ) + (f"- 问题: {q.question}" if self.language == "zh" else f"- Question: {q.question}")

            # 2. 调用具体的 memory subsystem 进行取回
            retrieved: List[RetrievedMemory] = self.memory_system.retrieve(
                history_name=history_name,
                query=q_for_retrieval,
                current_time=str(q.question_time),
                top_k=top_k
            )

            # 3. 将取回的所有背景组合（由 memory_system 自定义组装方式，可包含 text/time/metadata）
            context_block = self.memory_system.format_retrieved_for_context(
                retrieved, language=self.language
            )

            # 4. 根据模型上限截断上下文（保险措施）
            context_block = trim_context(
                context_block,
                tokenizer=self.tokenizer,
                max_tokens=self.memory_token_limit
            )

            # 5. 拼装成最终的 prompt
            prompt = self._build_prompt(q, context_block)
            messages_list.append([{"role": "user", "content": prompt}])

            if self.trace:
                trace_data.append((q, retrieved, prompt))

        # 6. 交给内部的生成模型进行并发推理
        responses = await self.chat_model.get_response_chat(
            messages_list,
            max_concurrency=20,
            max_new_tokens=1024,
            temperature=0.0,
            use_tqdm=True,
            verbose=True,
        )

        # 7. 每道题一条日志（问题、检索记忆、prompt、response）
        if self.trace:
            for (q, retrieved, prompt), resp in zip(trace_data, responses):
                self.trace.log_question_answer(
                    history_name=history_name,
                    question_id=str(q.metadata.get("question_id", history_name)),
                    question=q.question,
                    question_time=str(q.question_time),
                    retrieved=retrieved,
                    prompt=prompt,
                    response=resp,
                )

        return responses

    def answer_question(
        self,
        history_name: str,
        question: QuestionItem,
        top_k: int = 5
    ) -> str:
        raise NotImplementedError("针对单个问题的同步调用尚未实现，请在主流程中使用 await batch_answer_questions")

