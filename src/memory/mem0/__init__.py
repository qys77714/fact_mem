from __future__ import annotations

import json
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import random
from .prompts import *
from tqdm import tqdm
from memory import MemoryDatabase, RetrievedMemory, BaseMemoryMethod
from openai import OpenAI

logger = logging.getLogger(__name__)


class Mem0MemoryMethod(BaseMemoryMethod):
    def __init__(
        self,
        embed_model_name: str,
        llm_client = None,
        embed_client: Optional["OpenAI"] = None,
        database_root: Optional[str] = None,
        related_memory_top_k: int = 5,
    ) -> None:
        super().__init__(
            embed_model_name=embed_model_name,
            storage_namespace="mem0",
            embed_client=embed_client,
            database_root=database_root,
        )
        if llm_client is None:
            raise ValueError("llm_client must be provided for Mem0MemoryMethod.")
        
        self.llm = llm_client
        self.fact_retrieval_prompt = FACT_RETRIEVAL_PROMPT.strip()
        self.related_memory_top_k = max(1, related_memory_top_k)

    def store_history(
        self,
        history_name: str,
        chat_history: List[List[Dict[str, str]]],
        chat_dates: List[Any],
        granularity: str = "session",
    ) -> None:
        if granularity not in {"session", "turn"}:
            raise ValueError("granularity must be 'session' or 'turn'")

        database = self._get_database(history_name)
        dates = chat_dates or []

        for session_idx, turn_idx, turns in tqdm(self._iter_work_items(chat_history, granularity), desc=f"Storing history {history_name}"):
            session_date=dates[session_idx]

            transcript = self._format_session_transcript(turns, session_date)
            if not transcript.strip():
                continue
        
            facts = self._extract_facts(transcript)
            if not facts:
                continue
            
            metadata_base = self._build_metadata(
                history_name=history_name,
                session_idx=session_idx,
                turn_idx=turn_idx,
                session_date=session_date
            )

            old_memory_json, temp_uuid_mapping = self._collect_related_memories(database, facts, session_date)
            operations = self._decide_memory_operations(facts, old_memory_json)

            self._apply_memory_changes(database, operations, temp_uuid_mapping, metadata_base)

    def retrieve(
        self,
        history_name: str,
        question_text: str,
        top_k: int,
    ) -> List["RetrievedMemory"]:
        database = self._get_database(history_name)
        return database.search(question_text, top_k)

    def _iter_work_items(
        self,
        chat_history: List[List[Dict[str, str]]],
        granularity: str,
    ) -> List[Tuple[int, Optional[int], List[Dict[str, str]], int]]:
        items: List[Tuple[int, Optional[int], List[Dict[str, str]], int]] = []
        for session_idx, session in enumerate(chat_history):
            if not session:
                continue
            if granularity == "session":
                items.append((session_idx, None, session))
            else:
                for turn_idx, turn in enumerate(session):
                    items.append((session_idx, turn_idx, [turn]))
        return items

    def _format_session_transcript(self, turns: List[Dict[str, str]], session_date: Any) -> str:
        lines: List[str] = []
        for turn in turns:
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            speaker = (turn.get("speaker") or "unknown").strip()
            lines.append(f"**{speaker}**: {content}")
        return f"对话日期：{session_date}\n" + "\n".join(lines)

    def _extract_facts(self, transcript: str) -> List[str]:
        messages = [
            {"role": "user", "content": f"{self.fact_retrieval_prompt}\n# 输入文本：\n{transcript}"},
        ]
        raw_response = self.llm.get_response_chat(
            messages, 
            max_new_tokens=2048,
            temperature=0,
            response_format=FACT_RETRIEVAL_RESPONSE_FORMAT,
            verbose=True
        )
        facts = self._parse_fact_response(raw_response)
        return list(dict.fromkeys(facts))

    def _collect_related_memories(
        self,
        database: "MemoryDatabase",
        facts: List[str],
        session_date: Any,
    ) -> Tuple[Optional[str], Dict[str, str]]:
        aggregated: "OrderedDict[str, RetrievedMemory]" = OrderedDict()
        for fact in facts:
            fact_text = (
                f"- Memory Date: {session_date}\n"
                f"- Memory Content: \n"
                f"<MemoryContent>\n"
                f"{fact}\n"
                f"</MemoryContent>"
            )

            for memory in database.search(fact_text, self.related_memory_top_k):
                metadata = getattr(memory, "metadata", {}) or {}
                memory_id = metadata.get("memory_id") or metadata.get("id")
                if not memory_id or memory_id in aggregated:
                    continue
                aggregated[memory_id] = memory

        if not aggregated:
            return None, {}
                
        if len(aggregated) > 20:
            # 根据 memory.score 降序排序，取前20个
            sorted_items = sorted(aggregated.items(), key=lambda item: item[1].score, reverse=True)
            selected_items = sorted_items[:20]
            aggregated = OrderedDict(selected_items)

        temp_uuid_mapping = {str(idx): memory_id for idx, memory_id in enumerate(aggregated.keys())}

        serialized_payload = {
            temp_id: aggregated[memory_id].text.split("<MemoryContent>")[1].split("</MemoryContent>")[0]
            for temp_id, memory_id in temp_uuid_mapping.items()
        }
        return json.dumps(serialized_payload, ensure_ascii=False, indent=2), temp_uuid_mapping

    def _decide_memory_operations(
        self,
        facts: List[str],
        retrieved_old_memory_json: Optional[str],
    ) -> List[Dict[str, Any]]:
        if not facts:
            return []
        response_content = json.dumps({"facts": facts}, ensure_ascii=False, indent=2)
        update_prompt = get_update_memory_messages(
            retrieved_old_memory_json,
            response_content,
        )
        raw_response = self.llm.get_response_chat(
            [{"role": "user", "content": update_prompt}], 
            max_new_tokens=2048,
            temperature=0,
            response_format=UPDATE_MEMORY_RESPONSE_FORMAT,
            verbose=True
        )
        return self._parse_memory_changes(raw_response)

    def _apply_memory_changes(
        self,
        database: "MemoryDatabase",
        operations: List[Dict[str, Any]],
        temp_uuid_mapping: Dict[str, str],
        metadata_base: Dict[str, Any],
    ) -> None:
        for operation in operations:
            event = (operation.get("event") or "").upper()
            text = (operation.get("text") or "").strip()
            op_id = operation.get("id")
            if event == "ADD":
                mem_text = (
                    f"- Memory Date: {metadata_base['date']}\n"
                    f"- Memory Content: \n"
                    f"<MemoryContent>\n"
                    f"{text}\n"
                    f"</MemoryContent>"
                )
                database.add(mem_text, dict(metadata_base))
            elif event == "UPDATE":
                target_id = temp_uuid_mapping.get(str(op_id))
                if not target_id or not text:
                    continue

                mem_text = (
                    f"- Memory Date: {metadata_base['date']}\n"
                    f"- Memory Content: \n"
                    f"<MemoryContent>\n"
                    f"{text}\n"
                    f"</MemoryContent>"
                )
                success = database.update_memory(target_id, mem_text, dict(metadata_base))
                if not success:
                    logger.warning("Mem0 update failed: memory_id=%s", target_id)
            elif event == "DELETE":
                target_id = temp_uuid_mapping.get(str(op_id))
                if not target_id:
                    continue
                if not database.delete(target_id):
                    logger.warning("Mem0 delete failed: memory_id=%s", target_id)

    def _parse_fact_response(self, raw_response: str) -> List[str]:
        payload = self._safe_json_loads(raw_response)
        if not payload:
            logger.warning("Mem0 fact extraction returned empty or invalid JSON.")
            return []
        facts = payload.get("facts")
        if not isinstance(facts, list):
            return []
        normalized: List[str] = []
        for fact in facts:
            if isinstance(fact, str):
                cleaned = fact.strip()
                if cleaned:
                    normalized.append(cleaned)
        return normalized

    def _parse_memory_changes(self, raw_response: str) -> List[Dict[str, Any]]:
        payload = self._safe_json_loads(raw_response)
        if not payload:
            logger.warning("Mem0 memory update response is empty or invalid.")
            return []
        memory_ops = payload.get("memory")
        if not isinstance(memory_ops, list):
            return []
        normalized_ops: List[Dict[str, Any]] = []
        for op in memory_ops:
            if not isinstance(op, dict):
                continue
            normalized_ops.append(
                {
                    "id": op.get("id"),
                    "text": op.get("text"),
                    "event": (op.get("event") or "").upper(),
                    "old_memory": op.get("old_memory"),
                }
            )
        return normalized_ops

    def _sanitize_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        if not metadata:
            return sanitized
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                sanitized[key] = value
            else:
                sanitized[key] = str(value)
        return sanitized

    def _build_metadata(
        self,
        history_name: str,
        session_idx: Optional[int],
        turn_idx: Optional[int],
        session_date: Optional[Any],
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "method": self.storage_namespace,
            "history_name": history_name,
            "source": "mem0",
        }
        if session_idx is not None:
            metadata["session"] = session_idx
        if turn_idx is not None:
            metadata["turn"] = turn_idx
        if session_date is not None:
            metadata["date"] = session_date

        return metadata

    def _resolve_session_date(self, chat_dates: List[Any], index: int) -> Optional[Any]:
        if not chat_dates:
            return None
        if 0 <= index < len(chat_dates):
            return chat_dates[index]
        return chat_dates[-1]

    def _safe_json_loads(self, value: Any) -> Optional[Dict[str, Any]]:
        if isinstance(value, dict):
            return value
        if not isinstance(value, str):
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            start = value.find("{")
            end = value.rfind("}")
            if start != -1 and end != -1 and start < end:
                try:
                    return json.loads(value[start : end + 1])
                except json.JSONDecodeError:
                    logger.warning("Mem0 failed to parse JSON payload: %s", value)
        return None