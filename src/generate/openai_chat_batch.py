import os
import asyncio
from openai import AsyncOpenAI
from typing import List, Tuple
from tqdm.asyncio import tqdm
from common_utils import configure_logging
from datetime import datetime
from openai import OpenAI
import time

class OpenAIClient:
    def __init__(
            self,
            api_key=None, 
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini"
    ):
        self.client = OpenAI(
            api_key=api_key, 
            base_url=base_url
        )
        self.model_name = model
        time.sleep(1)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[4:-7]  # 毫秒精度
        self.logger = configure_logging(
            log_file_path=os.path.join("logs", f"{model}_False_{ts}.log"),
            override=True
        )

    def get_response_chat(
            self,
            messages,
            max_new_tokens=1024,
            temperature=0.0,
            verbose=False,
            **kargs
    ) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
            **kargs
        )

        response = completion.choices[0].message.content

        if verbose:
            self.logger.info("=========")
            self.logger.info(f"prompt: {messages[-1]['content']}\n" + "-"*20)
            self.logger.info(f"response: {response}")
            self.logger.info("=========")

        return response

class AsyncOpenAIClient:
    def __init__(
            self,
            api_key=None, 
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini"
    ):
        self.client = AsyncOpenAI(
            api_key=api_key, 
            base_url=base_url
        )
        self.model_name = model
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[4:-7]  # 毫秒精度
        self.logger = configure_logging(
            log_file_path=os.path.join("logs", f"{model}_True_{ts}.log"),
            override=True
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
            **kwargs
    ) -> Tuple[int, str]:
        async with semaphore:
            for attempt in range(max_retries):
                try:
                    completion = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        extra_body={
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                        **kwargs
                    )
                    response = completion.choices[0].message.content
                    
                    if verbose:
                        self.logger.info("=========")
                        self.logger.info(f"prompt: {messages[-1]['content']}\n" + "-"*20)
                        self.logger.info(f"response: {response}")
                        self.logger.info("=========")
                    return (index, response)
                
                except Exception as e:
                    error_str = str(e)
                    if 'data_inspection_failed' in error_str or 'inappropriate content' in error_str:
                        print(f"请求 {index} 内容安全检查失败，跳过生成")
                        return (index, None)  # 返回 None 表示跳过

                    print(f"请求 {index} 第 {attempt+1} 次失败，错误: {str(e)}")
                    wait_time = min(60, 2 ** attempt)
                    await asyncio.sleep(wait_time)
            return (index, None)

    async def get_response_chat(
            self,
            messages_list,
            max_new_tokens=1024,
            temperature=0.0,
            max_concurrency=100,
            use_tqdm=False,
            verbose=False,
            **kwargs
    ) -> List[str]:
        semaphore = asyncio.Semaphore(max_concurrency)
        tasks = [
            asyncio.create_task(
                self.process_request(
                    idx, msg, semaphore,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    verbose=verbose,
                    **kwargs
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

        # if logger != None:
        #     logger.info(f"message:\n{messages_list[0]}\n{'--'*50}")
        #     logger.info(f"response:\n{sorted_results[0][1]}\n{'--'*50}")

if __name__ == "__main__":
    async def test():
        client = AsyncOpenAIClient(
            api_key="zjj", 
            base_url="http://localhost:7104/v1/",
            model="gpt-4o-mini"
        )
        messages_list = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"}
            ],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ]
        ]
        responses = await client.get_response_chat(
            messages_list,
            max_new_tokens=50,
            temperature=0.7,
            max_concurrency=2,
            use_tqdm=True,
            verbose=True
        )
        for resp in responses:
            print("Response:", resp)

    asyncio.run(test())