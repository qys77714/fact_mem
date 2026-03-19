import urllib.request
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os
import shutil
import tempfile
import sys
from pathlib import Path
import asyncio
import pytest

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root / "src"))

from benchmark.base import ChatSession, ChatTurn
from memory.amem import AMemMemorySystem
from test.e2e.test_rag import MockChatModel, MockEmbedClient

@dataclass
class EvalDatasetMock:
    sessions: List[ChatSession]

class ExtendedMockChatModel(MockChatModel):
    def get_response_chat(self, msg, *args, **kwargs):
        content = msg[0]['content'] if isinstance(msg, list) else msg
        format_name = str(getattr(kwargs.get('response_format'), 'name', ''))
        
        if "metadata" in format_name:
            return '{"context": "Test Context", "keywords": ["mock", "test"], "tags": ["tag1"], "summary": "test note"}'
        if "evolution" in format_name:
            return '{"should_evolve": false}'
        if "query" in format_name:
            return '{"keywords": ["test", "keyword"]}'
            
        return super().get_response_chat(msg, *args, **kwargs)

@pytest.mark.asyncio
async def test_amem_pipeline():
    tmp_path = tempfile.mkdtemp()
    try:
        s1 = ChatSession(
            session_date="2023-01-01",
            turns=[
                ChatTurn(time="1", speaker="zjj", content="我最近很想学吉他呢。"),
                ChatTurn(time="2", speaker="assistant", content="好啊，你想学木吉他还是电吉他？"),
                ChatTurn(time="3", speaker="zjj", content="电吉他吧，感觉比较帅。")
            ]
        )
        s2 = ChatSession(
            session_date="2023-01-02",
            turns=[
                ChatTurn(time="1", speaker="zjj", content="我今天去买了一把电吉他！就是有点重。"),
                ChatTurn(time="2", speaker="assistant", content="太棒了，你选了什么牌子？"),
                ChatTurn(time="3", speaker="zjj", content="Fender，红色的，绝美。")
            ]
        )
        
        db_path = os.path.join(tmp_path, "amem_db")
        os.makedirs(db_path, exist_ok=True)
        
        # Override to sync for AMem testing
        class SyncExtendedMockChatModel:
            def get_response_chat(self, msg, *args, **kwargs):
                format_name = str(getattr(kwargs.get('response_format'), 'name', ''))
                
                if "metadata" in format_name:
                    return '{"context": "Test Context", "keywords": ["mock", "test"], "tags": ["tag1"], "summary": "test note"}'
                if "evolution" in format_name:
                    return '{"should_evolve": false}'
                if "query" in format_name:
                    return '{"keywords": ["test", "keyword"]}'
                    
                return "Mock sync response"
                
        mock_llm = SyncExtendedMockChatModel()
        mock_embed = MockEmbedClient(dim=768)
        
        system = AMemMemorySystem(
            embed_model_name="mock-embed",
            embed_client=mock_embed,
            llm_client=mock_llm,
            database_root=db_path,
            related_memory_top_k=2,
            enable_evolution=True,
        )
        
        system.store_session("test_user_amem", 1, s1)
        system.store_session("test_user_amem", 2, s2)
        
        db = system._get_database("test_user_amem")
        count = len(db.list_all_memories())
        assert count == 2, f"Should have 2 session contexts, got {count}"
        
        mem = system.retrieve("test_user_amem", "我买了什么牌子的吉他？", "2023-01-03", top_k=2)
        assert len(mem) > 0
        print("AMem System Pipeline tests passed successfully!")
    finally:
        shutil.rmtree(tmp_path)

if __name__ == "__main__":
    asyncio.run(test_amem_pipeline())
