import os
import asyncio
from openai import AsyncOpenAI
from typing import List, Tuple
from tqdm.asyncio import tqdm
from common_utils import configure_logging
from datetime import datetime
from openai import OpenAI
import time

import random
from openai import AzureOpenAI
from azure.identity import get_bearer_token_provider, AzureCliCredential


class OpenAIClient:
    def __init__(
            self,
            api_key=None, 
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini"
    ):
        self.client, self.model_name = get_client(model_name=model)
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
        if self.model_name in ["gpt-4o-mini"]:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                **kargs
            )
        else:
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
        self.client, self.model_name = get_client(model_name=model)
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
                    if self.model_name in ["gpt-4o-mini", "gpt-4o"]:
                        completion = await self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            **kwargs
                        )
                    else:
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


# uv pip install openai azure.identity

def get_endpoints():
    gpt_4o = [
        {
            "endpoints": "https://conversationhubeastus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        {
            "endpoints": "https://conversationhubeastus2.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        {
            "endpoints": "https://conversationhubnorthcentralus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        {
            "endpoints": "https://conversationhubsouthcentralus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        {
            "endpoints": "https://conversationhubwestus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        # {
        #     "endpoints": "https://conversationhubwestus3.openai.azure.com/",
        #     "speed": 150,
        #     "model": "gpt-4o"
        # },
        {
            "endpoints": "https://readineastus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        {
            "endpoints": "https://readineastus2.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        {
            "endpoints": "https://readinnorthcentralus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        # {
        #     "endpoints": "https://readinsouthcentralus.openai.azure.com/",
        #     "speed": 150,
        #     "model": "gpt-4o"
        # },
        {
            "endpoints": "https://readinwestus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        # {
        #     "endpoints": "https://readinwestus3.openai.azure.com/",
        #     "speed": 150,
        #     "model": "gpt-4o"
        # },
        {
            "endpoints": "https://conversationhubeastus.openai.azure.com/",
            "speed": 450,
            "model": "gpt-4o-global"
        },
        {
            "endpoints": "https://conversationhubeastus2.openai.azure.com/",
            "speed": 450,
            "model": "gpt-4o-global"
        },
        {
            "endpoints": "https://conversationhubnorthcentralus.openai.azure.com/",
            "speed": 450,
            "model": "gpt-4o-global"
        },
        {
            "endpoints": "https://conversationhubsouthcentralus.openai.azure.com/",
            "speed": 450,
            "model": "gpt-4o-global"
        },
        # {
        #     "endpoints": "https://conversationhubwestus.openai.azure.com/",
        #     "speed": 450,
        #     "model": "gpt-4o-global"
        # },
        {
            "endpoints": "https://readineastus.openai.azure.com/",
            "speed": 450,
            "model": "gpt-4o-global"
        },
        {
            "endpoints": "https://readineastus2.openai.azure.com/",
            "speed": 450,
            "model": "gpt-4o-global"
        },
        {
            "endpoints": "https://readinnorthcentralus.openai.azure.com/",
            "speed": 450,
            "model": "gpt-4o-global"
        },
        {
            "endpoints": "https://readinwestus.openai.azure.com/",
            "speed": 450,
            "model": "gpt-4o-global"
        },
    ]
    gpt_4_turbo = [
        {
            "endpoints": "https://conversationhubswedencentral.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4-turbo"
        },
        {
            "endpoints": "https://conversationhubeastus2.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        {
            "endpoints": "https://readineastus2.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
    ]
    gpt_5 = [
        {
            "endpoints": "https://conversationhubeastus2.openai.azure.com/",
            "speed": 150,
            "model": "gpt-5-global"
        },
    ]
    return {
        "gpt-4o": gpt_4o,
        "gpt-4-turbo": gpt_4_turbo,
        "gpt-5": gpt_5,
    }
def select_endpoint(model_name: str) -> dict:
    azure_endpoints = get_endpoints()
    entries = azure_endpoints[model_name]
    candidates = [e for e in entries if e.get("speed", 0) > 0 and e.get("endpoints")]
    weights = [e["speed"] for e in candidates]
    chosen = random.choices(candidates, weights=weights, k=1)[0]
    return chosen
def get_client(
        model_name="gpt-4o",
        tenant_id="72f988bf-86f1-41af-91ab-2d7cd011db47",
        api_version="2024-12-01-preview",
        max_retries=5,
    ):
    azure_ad_token_provider = get_bearer_token_provider(
        AzureCliCredential(tenant_id=tenant_id),
        "https://cognitiveservices.azure.com/.default"
    )
    selected = select_endpoint(model_name)
    client = AzureOpenAI(
        azure_endpoint=selected["endpoints"],
        azure_ad_token_provider=azure_ad_token_provider,
        api_version=api_version,
        max_retries=max_retries,
    )
    return client, selected["model"]