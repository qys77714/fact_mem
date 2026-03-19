import json
import tempfile

import pytest

import agent.standard_agent as sa
from agent.standard_agent import StandardAgent, trim_context
from benchmark.base import QuestionItem
from memory.base import RetrievedMemory


class _DummyMemory:
    def __init__(self, payload):
        self.payload = payload

    def retrieve(self, history_name, query, current_time, top_k=5):
        return self.payload

    def format_retrieved_for_context(self, retrieved, language="zh"):
        from prompts import render_prompt
        if not retrieved:
            t = "agent_context_empty_zh.jinja" if language == "zh" else "agent_context_empty_en.jinja"
            return render_prompt(t)
        unit = "agent_context_unit_zh.jinja" if language == "zh" else "agent_context_unit_en.jinja"
        return "\n\n".join(
            render_prompt(unit, index=i + 1, text=r.text) for i, r in enumerate(retrieved)
        )


class _DummyChat:
    def __init__(self):
        self.last_messages_list = None

    async def get_response_chat(self, messages_list, **kwargs):
        self.last_messages_list = messages_list
        return ["ok" for _ in messages_list]


def test_trim_context_without_tokenizer():
    text = "abc"
    assert trim_context(text, tokenizer=None, max_tokens=1) == text


@pytest.mark.asyncio
async def test_standard_agent_batch_answer_questions_zh_with_memory():
    sa.AutoTokenizer = None
    retrieved = [
        RetrievedMemory(
            memory_id="m1",
            text="<MemoryContent>我喜欢电吉他</MemoryContent>",
            source_index="s1",
            time="2024-01-01",
            score=0.9,
            metadata={},
        )
    ]
    mem = _DummyMemory(retrieved)
    chat = _DummyChat()

    agent = StandardAgent(memory_system=mem, chat_model=chat, memory_token_limit=2048, language="zh")
    agent.tokenizer = None

    questions = [
        QuestionItem(
            question="我喜欢什么乐器？",
            answer="电吉他",
            question_time="2024-01-02",
            options=["钢琴", "电吉他"],
        )
    ]

    answers = await agent.batch_answer_questions("u1", questions, top_k=1)
    assert answers == ["ok"]
    prompt = chat.last_messages_list[0][0]["content"]
    assert "检索到的记忆单元" in prompt
    assert "最终答案" in prompt


@pytest.mark.asyncio
async def test_standard_agent_batch_answer_questions_en_no_memory():
    sa.AutoTokenizer = None
    mem = _DummyMemory([])
    chat = _DummyChat()

    agent = StandardAgent(memory_system=mem, chat_model=chat, memory_token_limit=2048, language="en")
    agent.tokenizer = None

    questions = [
        QuestionItem(
            question="What instrument do I prefer?",
            answer="electric guitar",
            question_time="2024-01-02",
        )
    ]

    await agent.batch_answer_questions("u1", questions, top_k=2)
    prompt = chat.last_messages_list[0][0]["content"]
    assert "No relevant memory found" in prompt
    assert "Please give a short answer" in prompt


def test_standard_agent_sync_method_not_implemented():
    sa.AutoTokenizer = None
    agent = StandardAgent(memory_system=_DummyMemory([]), chat_model=_DummyChat())
    with pytest.raises(NotImplementedError):
        agent.answer_question(
            "u1",
            QuestionItem(question="q", answer="a", question_time="t"),
        )


@pytest.mark.asyncio
async def test_standard_agent_tracing_logs_retrieval_and_llm():
    """Verify that when trace_log_dir is set, agent writes scope, retrieval, and llm_interaction events."""
    sa.AutoTokenizer = None
    retrieved = [
        RetrievedMemory(
            memory_id="m1",
            text="test memory",
            source_index="s1",
            time="2024-01-01",
            score=0.9,
            metadata={},
        )
    ]
    mem = _DummyMemory(retrieved)
    chat = _DummyChat()

    with tempfile.TemporaryDirectory() as tmpdir:
        agent = StandardAgent(
            memory_system=mem,
            chat_model=chat,
            memory_token_limit=2048,
            language="zh",
            trace_log_dir=tmpdir,
        )
        agent.tokenizer = None

        questions = [
            QuestionItem(
                question="测试问题",
                answer="答案",
                question_time="2024-01-02",
                metadata={"question_id": "q1"},
            )
        ]

        answers = await agent.batch_answer_questions("u1", questions, top_k=1)
        assert answers == ["ok"]

        log_files = list(agent.trace.log_dir.glob("*.jsonl"))
        assert len(log_files) == 1
        events = [json.loads(line) for line in log_files[0].read_text(encoding="utf-8").strip().split("\n")]
        assert len(events) == 1
        evt = events[0]
        assert evt["event_type"] == "question_answer"
        assert evt["question"] == "测试问题"
        assert evt["prompt"]
        assert evt["response"] == "ok"
        assert len(evt["retrieved"]) == 1
        assert evt["retrieved"][0]["memory_id"] == "m1"
