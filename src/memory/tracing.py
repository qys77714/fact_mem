from __future__ import annotations

import json
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class MemoryTraceLogger:
    """Structured JSONL tracer for LLM interactions and memory operations."""

    def __init__(self, method: str, log_dir: str = "logs/memory_trace") -> None:
        self.method = method
        self.run_id = f"{method}-{uuid.uuid4().hex[:8]}"
        self._seq = 0
        self._llm_seq = 0
        self._memop_seq = 0
        self._retrieval_seq = 0
        self._scope_seq = 0

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"{method}_{timestamp}_{self.run_id}.jsonl"

    def _next(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}-{self._seq:06d}"

    def _to_jsonable(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._to_jsonable(v) for v in value]
        if is_dataclass(value):
            return self._to_jsonable(asdict(value))
        if hasattr(value, "__dict__"):
            return self._to_jsonable(vars(value))
        return str(value)

    def _write(self, payload: Dict[str, Any]) -> None:
        base = {
            "event_id": self._next("evt"),
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
            "method": self.method,
        }
        base.update(payload)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self._to_jsonable(base), ensure_ascii=False) + "\n")

    def create_scope(
        self,
        purpose: str,
        *,
        parent_scope_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        self._scope_seq += 1
        scope_id = f"scope-{self._scope_seq:06d}"
        self._write(
            {
                "event_type": "scope_start",
                "scope_id": scope_id,
                "parent_scope_id": parent_scope_id,
                "purpose": purpose,
                "summary": f"Start scope: {purpose}",
                "metadata": metadata or {},
            }
        )
        return scope_id

    def close_scope(self, scope_id: str, *, status: str = "ok", metadata: Optional[Dict[str, Any]] = None) -> None:
        self._write(
            {
                "event_type": "scope_end",
                "scope_id": scope_id,
                "status": status,
                "summary": f"End scope: {scope_id} ({status})",
                "metadata": metadata or {},
            }
        )

    def log_llm_interaction(
        self,
        *,
        purpose: str,
        messages: Any,
        response: Any,
        scope_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> str:
        self._llm_seq += 1
        interaction_id = f"llm-{self._llm_seq:06d}"
        self._write(
            {
                "event_type": "llm_interaction",
                "interaction_id": interaction_id,
                "scope_id": scope_id,
                "purpose": purpose,
                "status": "error" if error else "ok",
                "summary": f"LLM interaction for {purpose}",
                "metadata": metadata or {},
                "request": {"messages": messages},
                "response": response,
                "error": error,
            }
        )
        return interaction_id

    def log_memory_operation(
        self,
        *,
        operation: str,
        memory_id: Optional[str],
        scope_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        after: Optional[Dict[str, Any]] = None,
        status: str = "ok",
    ) -> str:
        self._memop_seq += 1
        op_id = f"memop-{self._memop_seq:06d}"
        self._write(
            {
                "event_type": "memory_operation",
                "operation_id": op_id,
                "scope_id": scope_id,
                "operation": operation.upper(),
                "memory_id": memory_id,
                "status": status,
                "summary": f"Memory {operation.upper()} ({status})",
                "metadata": metadata or {},
                "before": before,
                "after": after,
            }
        )
        return op_id

    def log_retrieval(
        self,
        *,
        purpose: str,
        query: str,
        retrieved: List[Any],
        scope_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        self._retrieval_seq += 1
        retrieval_id = f"retrieval-{self._retrieval_seq:06d}"
        self._write(
            {
                "event_type": "retrieval",
                "retrieval_id": retrieval_id,
                "scope_id": scope_id,
                "purpose": purpose,
                "summary": f"Retrieval for {purpose}",
                "query": query,
                "retrieved": retrieved,
                "metadata": metadata or {},
            }
        )
        return retrieval_id

    def log_question_answer(
        self,
        *,
        history_name: str,
        question_id: str,
        question: str,
        question_time: str,
        retrieved: List[Any],
        prompt: str,
        response: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """一条问题对应一条 JSONL 记录，包含检索记忆、prompt、response。"""
        self._write(
            {
                "event_type": "question_answer",
                "history_name": history_name,
                "question_id": question_id,
                "question": question,
                "question_time": question_time,
                "retrieved": retrieved,
                "prompt": prompt,
                "response": response,
                "metadata": metadata or {},
            }
        )
