from typing import List, Optional
import os

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
        trace_method: str = "agent",
        answer_concurrency: int = 2,
        show_time: bool = True,
    ):
        self.memory_system = memory_system
        self.chat_model = chat_model
        self.memory_token_limit = memory_token_limit
        self.language = language
        self.answer_concurrency = max(1, int(answer_concurrency))
        self.show_time = show_time
        # Per-episode files under log_dir (e.g. ``{history_name}.jsonl``); resume/cleanup aligns with pred JSONL.
        self.trace: Optional[MemoryTraceLogger] = (
            MemoryTraceLogger(
                method=trace_method,
                log_dir=trace_log_dir,
                use_experiment_naming=True,
            )
            if trace_log_dir
            else None
        )

        if AutoTokenizer is not None:
            # 仅用于上下文 token 计数；路径可经 ANSWER_TOKENIZER_PATH 覆盖，
            # 默认指向本地 Qwen3-8B（原硬编码 /mnt/data_oss 在未挂载机器上不可用）。
            tok_path = os.environ.get(
                "ANSWER_TOKENIZER_PATH", "/data/zjj/models/Qwen/Qwen3-8B"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                tok_path,
                trust_remote_code=True,
            )
        else:
            self.tokenizer = None

    def _build_prompt(self, question: QuestionItem, context_block: str) -> str:
        return render_prompt(
            "pipeline_answer.jinja",
            context_block=context_block,
            question_time=question.question_time,
            question=question.question,
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
            # 1. 检索 query：仅用问题正文
            q_for_retrieval = q.question

            # 2. 调用具体的 memory subsystem 进行取回
            retrieved: List[RetrievedMemory] = self.memory_system.retrieve(
                history_name=history_name,
                query=q_for_retrieval,
                current_time=str(q.question_time),
                top_k=top_k
            )

            max_sess = q.metadata.get("max_session_index")
            if max_sess is not None:
                try:
                    cutoff = int(max_sess)
                    filtered: List[RetrievedMemory] = []
                    for mem in retrieved:
                        meta = mem.metadata or {}
                        sess_idx = meta.get("lme_session_index")
                        if sess_idx is None:
                            filtered.append(mem)
                            continue
                        try:
                            if int(sess_idx) <= cutoff:
                                filtered.append(mem)
                        except (TypeError, ValueError):
                            filtered.append(mem)
                    retrieved = filtered
                except (TypeError, ValueError):
                    pass

            # 3. 将取回的所有背景组合（由 memory_system 自定义组装方式，可包含 text/time/metadata）
            context_block = self.memory_system.format_retrieved_for_context(
                retrieved, language=self.language, show_time=self.show_time
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
            max_concurrency=self.answer_concurrency,
            max_new_tokens=1024,
            temperature=0.0,
            use_tqdm=True,
            verbose=True,
        )

        # 7. 每道题一条日志（问题、检索记忆、prompt、response）
        if self.trace:
            ep_trace = self.trace.get_logger_for(history_name)
            for (q, retrieved, prompt), resp in zip(trace_data, responses):
                ep_trace.log_question_answer(
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

