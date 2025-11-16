import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from generate.load_api_llm import load_api_chat_completion
from memory import *
from memory.mem0 import Mem0MemoryMethod
from openai import OpenAI
from transformers import AutoTokenizer


def trim_context(context: str, tokenizer: AutoTokenizer, max_tokens: int) -> str:
    encoded = tokenizer(context, add_special_tokens=False, return_attention_mask=False)
    token_ids = encoded["input_ids"]
    original_len = len(token_ids)

    if original_len <= max_tokens:
        print("Retrieved Context Trimming: No")
        return context

    trimmed_ids = token_ids[:max_tokens]  # Retain the first max_tokens tokens
    trimmed_text = tokenizer.decode(trimmed_ids, skip_special_tokens=True)
    print(f"Retrieved Context Trimming: Yes, before={original_len}, after={len(trimmed_ids)}")
    return trimmed_text

class MemorySystem:
    def __init__(
        self,
        method_name: str,
        chat_model: str,
        embed_model_name: str,
        embed_client: OpenAI,
        context_token_limit: int = 2048,
        database_root: Optional[str] = None,
    ) -> None:
        registry = {
            "full_context": FullContextMemoryMethod,
            "only_query": OnlyQueryMemoryMethod,
            "rag": RagMemoryMethod,
            "mem0": Mem0MemoryMethod,
        }
        if method_name not in registry:
            raise ValueError(f"未知的记忆方法: {method_name}")
        if not embed_model_name:
            raise ValueError("必须提供 embed_model_name。")

        self.chat_model = load_api_chat_completion(chat_model, async_=True)
        self.chat_model_context_token_limit = context_token_limit

        method_cls = registry[method_name]
        method_kwargs = {
            "embed_model_name": embed_model_name,
            "embed_client": embed_client,
            
            "database_root": database_root,
        }

        if method_name == "mem0":
            manager_model = load_api_chat_completion(chat_model, async_=False)
            method_kwargs.update({
                "llm_client": manager_model,
                "related_memory_top_k": 5,
            })

        self.method = method_cls(**method_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    def store_history(
            self, 
            history_name: str, 
            chat_history: List[List[Dict[str, str]]],
            chat_dates: List,
            granularity: str = "turn"
    ) -> None:
        self.method.store_history(history_name, chat_history, chat_dates, granularity=granularity)

    def retrieve_context(
        self,
        history_name: str,
        question_text: str,
        top_k: int = 5,
    ) -> List[RetrievedMemory]:
        return self.method.retrieve(history_name, question_text, top_k)

    async def answer_question(
        self,
        history_name: str,
        questions: List[str],
        question_dates: List,
        
        top_k: int = 5,
    ):
        if len(question_dates) != len(questions):
            raise ValueError("question_date 的数量需要与 question_text 对齐。")

        messages_list = []
        for q, date in zip(questions, question_dates):
            instruction = (
                "You are a memory-augmented assistant. "
                "Use the retrieved memory units to provide accurate and context-aware answers to the user's questions. "
                "If no relevant memory is found, rely on general knowledge to respond."
            )  
            header = (
                f"### Question Details\n"
                f"- Current Date: {date}\n"
                f"- Question: {q}"
            )
            q_for_retrieval = (
                f"- Current Date: {date}\n"
                f"- Question: {q}"
            )

            retrieved = self.retrieve_context(history_name, q_for_retrieval, top_k)
            if retrieved:
                context_lines = [
                    f"### Retrieved Memory Unit {idx + 1}\n{item.text}"
                    for idx, item in enumerate(retrieved)
                ]
                context_block = "\n\n".join(context_lines)
            else:
                context_block = "### Retrieved Memory Units\nNo relevant memory found."

            context_block = trim_context(
                context_block, 
                tokenizer=self.tokenizer, 
                max_tokens=self.chat_model_context_token_limit - 1024
            )

            prompt = f"{instruction}\n{context_block}\n{header}"
            messages_list.append(
                [
                    {
                        "role": "user",
                        "content": prompt
                    },
                ]
            )
        
        responses = await self.chat_model.get_response_chat(
            messages_list, 
            max_concurrency=10, 
            max_new_tokens=1024, 
            temperature=0.0,
            use_tqdm=True,
            verbose=True,
        )
        return responses