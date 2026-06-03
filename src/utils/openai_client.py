import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Any, List, Optional, Tuple

from openai import AsyncOpenAI, OpenAI
from tqdm.asyncio import tqdm

from utils.common_utils import configure_logging
from utils.thinking_text import split_embedded_thinking


def _is_qwen_family_model(model_name: str) -> bool:
    name = (model_name or "").lower()
    return "qwen" in name or "qwq" in name


def _prepare_extra_body_for_model(
    model_name: str,
    existing: dict | None,
    *,
    qwen_enable_thinking: bool,
) -> dict | None:
    """Prepare extra_body per model capability.

    Only Qwen-like serving stacks should receive chat_template_kwargs.enable_thinking.
    For non-Qwen models (e.g. gpt-5.4), explicitly drop chat_template_kwargs to avoid
    unknown-parameter errors.
    """
    merged = dict(existing) if existing else {}

    if not _is_qwen_family_model(model_name):
        merged.pop("chat_template_kwargs", None)
        return merged or None

    chat_kw = dict(merged.get("chat_template_kwargs") or {})
    chat_kw["enable_thinking"] = bool(qwen_enable_thinking)
    merged["chat_template_kwargs"] = chat_kw

    return merged or None


def _openai_assistant_message_text(msg: Any) -> Optional[str]:
    """Prefer message content; if empty, fall back to reasoning_content.

    Qwen3/vLLM may embed CoT in content (e.g. redacted_thinking XML) with an empty reasoning_content
    field; strip embedded blocks so callers see the final answer text (e.g. judge yes/no).
    """
    c = getattr(msg, "content", None)
    c = c.strip() if isinstance(c, str) else ""
    rc_raw = getattr(msg, "reasoning_content", None)
    rc = rc_raw.strip() if isinstance(rc_raw, str) else ""

    emb_think, c_rest = split_embedded_thinking(c)
    if emb_think:
        if not rc:
            rc = emb_think
        c = c_rest

    out = c if c else rc
    return out if out else None


def _legacy_log_enabled() -> bool:
    return os.getenv("OPENAI_CLIENT_LEGACY_LOG", "0").strip().lower() in {"1", "true", "yes", "on"}


class OpenAIClient:
    def __init__(
        self,
        api_key=None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model
        self.logger = logging.getLogger(__name__)
        if _legacy_log_enabled():
            time.sleep(1)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[4:-7]
            self.logger = configure_logging(
                log_file_path=os.path.join("logs", f"{model}_False_{ts}.log"),
                override=True,
            )

    def get_response_chat(
        self,
        messages,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        verbose: bool = False,
        **kargs,
    ) -> str:
        response = None
        try:
            kargs = dict(kargs)
            qwen_enable_thinking = False
            if self.model_name in ["gpt-4o-mini"]:
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        **kargs,
                    )
                except Exception as e:
                    print(f"Error during completion: {e}")
                    return None
            else:
                user_extra = kargs.pop("extra_body", None)
                qwen_enable_thinking = bool(kargs.pop("qwen_enable_thinking", False))
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    extra_body=_prepare_extra_body_for_model(
                        self.model_name,
                        user_extra,
                        qwen_enable_thinking=qwen_enable_thinking,
                    ),
                    **kargs,
                )

            _msg = completion.choices[0].message
            response = _openai_assistant_message_text(_msg)

            if verbose:
                self.logger.info("=========")
                self.logger.info(f"prompt: {messages[-1]['content']}\n" + "-" * 20)
                self.logger.info(f"response: {response}")
                self.logger.info("=========")
        except Exception as e:
            error_str = str(e)
            if "data_inspection_failed" in error_str or "inappropriate content" in error_str:
                print("请求内容安全检查失败，跳过生成")
                return None

        return response


class AsyncOpenAIClient:
    def __init__(
        self,
        api_key=None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
    ):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model
        self.logger = logging.getLogger(__name__)
        if _legacy_log_enabled():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[4:-7]
            self.logger = configure_logging(
                log_file_path=os.path.join("logs", f"{model}_True_{ts}.log"),
                override=True,
            )

    async def process_request(
        self,
        index: int,
        messages: List,
        semaphore: asyncio.Semaphore,
        max_tokens: int = 4,
        temperature: float = 0.0,
        max_retries: int = 50,
        verbose: bool = False,
        qwen_enable_thinking: bool = False,
        return_raw_message: bool = False,
        **kwargs,
    ) -> Tuple[int, str]:
        async with semaphore:
            for attempt in range(max_retries):
                try:
                    kwargs = dict(kwargs)
                    if self.model_name in ["gpt-4o-mini"]:
                        completion = await self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            **kwargs,
                        )
                    else:
                        user_extra = kwargs.pop("extra_body", None)
                        completion = await self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            extra_body=_prepare_extra_body_for_model(
                                self.model_name,
                                user_extra,
                                qwen_enable_thinking=qwen_enable_thinking,
                            ),
                            **kwargs,
                        )
                    _msg = completion.choices[0].message
                    response = _openai_assistant_message_text(_msg)
                    if return_raw_message:
                        content_raw = getattr(_msg, "content", None)
                        reasoning_raw = getattr(_msg, "reasoning_content", None)
                        content = content_raw.strip() if isinstance(content_raw, str) else ""
                        reasoning_content = (
                            reasoning_raw.strip() if isinstance(reasoning_raw, str) else ""
                        )
                        emb_think, content_after = split_embedded_thinking(content)
                        if emb_think and not reasoning_content:
                            reasoning_content = emb_think
                        if emb_think:
                            content = content_after
                        return (
                            index,
                            {
                                "response": response,
                                "content": content,
                                "reasoning_content": reasoning_content,
                            },
                        )

                    if verbose:
                        self.logger.info("=========")
                        self.logger.info(f"prompt: {messages[-1]['content']}\n" + "-" * 20)
                        self.logger.info(f"response: {response}")
                        self.logger.info("=========")
                    return (index, response)

                except Exception as e:
                    error_str = str(e)
                    if "data_inspection_failed" in error_str or "inappropriate content" in error_str:
                        print(f"请求 {index} 内容安全检查失败，跳过生成")
                        return (index, None)

                    print(f"请求 {index} 第 {attempt + 1} 次失败，错误: {str(e)}")
                    wait_time = min(60, 2**attempt)
                    await asyncio.sleep(wait_time)
            return (index, None)

    async def get_response_chat(
        self,
        messages_list,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        max_concurrency: int = 100,
        use_tqdm: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> List[str]:
        kwargs = dict(kwargs)
        qwen_enable_thinking = bool(kwargs.pop("qwen_enable_thinking", False))
        return_raw_message = bool(kwargs.pop("return_raw_message", False))
        semaphore = asyncio.Semaphore(max_concurrency)
        tasks = [
            asyncio.create_task(
                self.process_request(
                    idx,
                    msg,
                    semaphore,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    verbose=verbose,
                    qwen_enable_thinking=qwen_enable_thinking,
                    return_raw_message=return_raw_message,
                    **kwargs,
                )
            )
            for idx, msg in enumerate(messages_list)
        ]

        if use_tqdm:
            results = await tqdm.gather(*tasks)
        else:
            results = await asyncio.gather(*tasks)

        sorted_results = sorted(results, key=lambda x: x[0])
        return [r[1] for r in sorted_results]


__all__ = ["OpenAIClient", "AsyncOpenAIClient"]
