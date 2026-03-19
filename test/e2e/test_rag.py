import sys
import os
import asyncio
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root / "src"))

import numpy as np
from benchmark.base import ChatSession, ChatTurn, QuestionItem
from memory.baselines.rag import RagMemorySystem
from agent.standard_agent import StandardAgent

class MockEmbedClient:
    class Embeddings:
        def __init__(self, dim=1024):
            self.dim = dim
            
        def create(self, input, model):
            class Item:
                def __init__(self, i, emb):
                    self.index = i
                    self.embedding = emb
            class Resp:
                def __init__(self, data):
                    self.data = data
            
            data = []
            for i, text in enumerate(input):
                # 简单地为不同文本生成一个固定的、或随机的但确定的向量
                np.random.seed(hash(text) % (2**32))
                emb = np.random.randn(self.dim).tolist()
                data.append(Item(i, emb))
            return Resp(data)
            
    def __init__(self, dim=1024):
        self.embeddings = self.Embeddings(dim)

class MockChatModel:
    async def get_response_chat(
        self, 
        messages_list, 
        **kwargs
    ):
        responses = []
        for msgs in messages_list:
            last_prompt = msgs[-1]["content"] if msgs else ""
            print(f"---> Received Prompt:\n{last_prompt}\n" + "-"*40)
            
            if "选项" in last_prompt:
                responses.append("最终答案: 电吉他")
            else:
                responses.append("根据您的历史记录，大约是3000左右。")
            
        return responses

async def main():
    print("=== Testing RAG Agent E2E ===")
    
    test_db_dir = project_root / "test" / "test_db_e2e"
    import shutil
    if test_db_dir.exists():
        shutil.rmtree(test_db_dir)
        
    print("1. Initializing Subsystems")
    rag_memory = RagMemorySystem(
        embed_client=MockEmbedClient(dim=128),
        embed_model_name="mock-model",
        granularity=3,
        database_root=str(test_db_dir)
    )
    
    agent = StandardAgent(
        memory_system=rag_memory,
        chat_model=MockChatModel(),
        memory_token_limit=2048,
        language="zh"
    )
    agent.tokenizer = None 
    
    print("2. Storing Chat Session (Mocking Data)")
    session = ChatSession(
        session_date="2024-05-01 10:00:00",
        turns=[
            ChatTurn(speaker="user", content="你好，我最近想学点乐器。", time="10:00:00"),
            ChatTurn(speaker="assistant", content="好啊，你想学什么？", time="10:00:05"),
            ChatTurn(speaker="user", content="吉他或者钢琴吧，但我更喜欢吉他，尤其是电吉他。", time="10:00:15"),
            ChatTurn(speaker="assistant", content="电吉他很酷哦，准备报班还是自学？", time="10:00:20"),
            ChatTurn(speaker="user", content="我想先自学，你能给我推荐几个型号吗？预算在3000左右。", time="10:00:45"),
        ]
    )
    
    rag_memory.store_session("test_user_e2e", session_idx=0, session=session)
    print("✓ Session context stored.")
    
    print("3. Asking Questions")
    q1 = QuestionItem(
        question="我最倾向于学什么乐器？",
        question_time="2024-05-02 10:00:00",
        answer="电吉他",
        options=["钢琴", "木吉他", "电吉他", "架子鼓"]
    )
    q2 = QuestionItem(
        question="我买吉他的预算是多少？",
        question_time="2024-05-02 10:01:00",
        answer="3000左右"
    )
    
    responses = await agent.batch_answer_questions(
        history_name="test_user_e2e",
        questions=[q1, q2],
        top_k=2 
    )
    
    for q, r in zip([q1, q2], responses):
        print(f"\nQ: {q.question}")
        print(f"Gold: {q.answer}")
        print(f"Agent Replied: {r}")
        
    print("\n✓ E2E Pipeline completed successfully!")
    shutil.rmtree(test_db_dir)

if __name__ == "__main__":
    asyncio.run(main())
