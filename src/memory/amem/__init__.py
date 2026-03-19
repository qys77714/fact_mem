from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING, Union
import numpy as np

from memory.base import BaseMemorySystem, RetrievedMemory
from memory.storage.local_faiss import LocalFaissDatabase
from memory.tracing import MemoryTraceLogger
from benchmark.base import ChatSession

from .prompts import (
    build_metadata_prompt,
    build_evolution_prompt,
    build_query_prompt,
    METADATA_RESPONSE_FORMAT,
    EVOLUTION_RESPONSE_FORMAT,
    QUERY_RESPONSE_FORMAT,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from openai import OpenAI


@dataclass
class MemoryInsights:
    context: str
    keywords: List[str]
    tags: List[str]
    summary: str

    @classmethod
    def from_payload(cls, payload: Optional[Dict[str, Any]]) -> "MemoryInsights":
        context = str(payload.get("context") or "General context")
        summary = str(payload.get("summary") or context)
        keywords = _normalize_string_list(payload.get("keywords")) or ["general"]
        tags = _normalize_string_list(payload.get("tags")) or ["general"]
        return cls(context=context, keywords=keywords, tags=tags, summary=summary)


def _normalize_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        cleaned = (
            value.replace("，", ",")
            .replace("、", ",")
            .replace("|", ",")
        )
        return [item.strip() for item in cleaned.split(",") if item.strip()]
    return []


class AMemMemorySystem(BaseMemorySystem):
    def __init__(
        self,
        embed_model_name: str,
        llm_client,
        embed_client: Optional["OpenAI"] = None,
        database_root: Optional[str] = None,
        related_memory_top_k: int = 5,
        metadata_temperature: float = 0.0,
        evolution_temperature: float = 0.2,
        enable_evolution: bool = True,
        language: str = "en",
        granularity: Union[str, int] = "all",
    ) -> None:
        if llm_client is None:
            raise ValueError("llm_client must be provided for AMemMemorySystem.")
        super().__init__(
            embed_model_name=embed_model_name,
            embed_client=embed_client,
            llm_client=llm_client,
            database_root=database_root,
        )
        self.llm = llm_client
        self.related_memory_top_k = max(1, related_memory_top_k)
        self.metadata_temperature = metadata_temperature
        self.evolution_temperature = evolution_temperature
        self.enable_evolution = enable_evolution
        self.language = language
        self.granularity = self._parse_granularity(granularity)
        self.trace = MemoryTraceLogger(method="amem")
        self._databases = {}

    @staticmethod
    def _parse_granularity(granularity: Union[str, int]) -> Union[str, int]:
        if isinstance(granularity, str):
            g = granularity.strip().lower()
            if g == "all":
                return "all"
            if g.isdigit():
                granularity = int(g)
            else:
                raise ValueError("AMem granularity must be 'all' or a positive integer.")

        if isinstance(granularity, int) and granularity > 0:
            return granularity
        raise ValueError("AMem granularity must be 'all' or a positive integer.")

    def _get_database(self, history_name: str) -> LocalFaissDatabase:
        namespace = f"amem_{self.granularity}_{history_name}"
        if namespace not in self._databases:
            self._databases[namespace] = LocalFaissDatabase(
                namespace=namespace,
                database_root=self.database_root
            )
        return self._databases[namespace]
        
    def _embed_texts(self, inputs: List[str]) -> np.ndarray:
        from utils.embed_utils import embed_texts
        return embed_texts(self.embed_client, inputs, self.embed_model_name)

    def build_text_for_embedding(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """AMem: 对 text + metadata (context, keywords, tags, summary) 做 embedding。"""
        meta = metadata or {}
        context = str(meta.get("context") or "").strip()
        keywords = _normalize_string_list(meta.get("keywords"))
        tags = _normalize_string_list(meta.get("tags"))
        summary = str(meta.get("summary") or "").strip()

        if not context and not keywords and not tags and not summary:
            return text

        content = text.strip()

        keywords_line = ", ".join(keywords) if keywords else ""
        tags_line = ", ".join(tags) if tags else ""
        parts = []
        if context:
            parts.append(f"- Memory Context: {context}")
        if keywords_line:
            parts.append(f"- Memory Keywords: {keywords_line}")
        if tags_line:
            parts.append(f"- Memory Tags: {tags_line}")
        if summary:
            parts.append(f"- Memory Summary: {summary}")
        parts.append("- Memory Content:")
        parts.append(content)
        return "\n".join(parts)

    def store_session(self, history_name: str, session_idx: int, session: ChatSession) -> None:
        database = self._get_database(history_name)
        session_date = session.session_date

        session_scope = self.trace.create_scope(
            "amem_store_session",
            metadata={
                "history_name": history_name,
                "session_idx": session_idx,
                "session_date": str(session_date),
                "granularity": self.granularity,
            },
        )

        if self.granularity == "all":
            work_items = [(None, None, session.turns)]
        else:
            chunk_size = int(self.granularity)
            work_items = []
            for i in range(0, len(session.turns), chunk_size):
                chunk_turns = session.turns[i : i + chunk_size]
                if chunk_turns:
                    work_items.append((i, i + len(chunk_turns) - 1, chunk_turns))

        for turn_start, turn_end, turns in work_items:
            chunk_scope = self.trace.create_scope(
                "amem_store_chunk",
                parent_scope_id=session_scope,
                metadata={
                    "turn_start": turn_start,
                    "turn_end": turn_end,
                    "turn_count": len(turns),
                },
            )
            content_block = self._format_memory_content(turns)
            if not content_block.strip():
                self.trace.close_scope(chunk_scope, status="skip", metadata={"reason": "empty_content"})
                continue

            transcript = self._format_session_transcript(turns, session_date)
            insights = self._extract_memory_insights(transcript, trace_scope_id=chunk_scope)

            metadata = {
                "method": "amem",
                "history_name": history_name,
                "session": session_idx,
                "date": session_date,
            }
            if turn_start is not None:
                metadata["turn_start"] = turn_start
                metadata["turn_end"] = turn_end

            metadata.update(
                {
                    "context": insights.context,
                    "keywords": insights.keywords,
                    "tags": insights.tags,
                    "summary": insights.summary,
                    "granularity": self.granularity,
                }
            )
            
            search_query = "\n".join(
                [
                    insights.context,
                    ", ".join(insights.keywords),
                    content_block,
                ]
            )
            # Find closest records to possibly apply evolution
            query_emb = self._embed_texts([search_query])[0]
            neighbors = database.search(query_emb, self.related_memory_top_k)
            
            if neighbors:
                metadata["related_ids"] = [
                    mem.memory_id
                    for mem in neighbors
                ]
                if self.enable_evolution:
                    metadata = self._maybe_apply_evolution(
                        database=database,
                        metadata=metadata,
                        neighbors=neighbors,
                        trace_scope_id=chunk_scope,
                    )

            # 只存 content_block（与 RAG 一致），date/context/keywords/tags/summary 在 time 和 metadata 中
            text_to_store = content_block.strip()

            if turn_start is None:
                source_index = f"session_{session_idx}"
            elif turn_start == turn_end:
                source_index = f"session_{session_idx}-turn_{turn_start}"
            else:
                source_index = f"session_{session_idx}-turn_{turn_start}_to_{turn_end}"
            
            # Embed 时用 text + metadata 构建完整串用于检索
            text_for_embed = self.build_text_for_embedding(text_to_store, metadata=metadata)
            mem_emb = self._embed_texts([text_for_embed])[0]
            memory_id = database.add(text_to_store, source_index, str(session_date), metadata, embedding=mem_emb)
            self.trace.log_memory_operation(
                operation="ADD",
                memory_id=memory_id,
                scope_id=chunk_scope,
                metadata={
                    "history_name": history_name,
                    "session_idx": session_idx,
                    "source_index": source_index,
                },
                after={
                    "text": text_to_store,
                    "source_index": source_index,
                    "time": str(session_date),
                    "metadata": metadata,
                },
                status="ok",
            )
            self.trace.close_scope(chunk_scope, status="ok", metadata={"stored_memory_id": memory_id})

        self.trace.close_scope(session_scope, status="ok")

    def retrieve(
        self,
        history_name: str,
        query: str,
        current_time: str,
        top_k: int = 5,
    ) -> List[RetrievedMemory]:
        if top_k <= 0 or not query:
            return []
        database = self._get_database(history_name)
        augmented_query = self._build_retrieval_query(query)
        
        query_embedding = self._embed_texts([augmented_query])
        if query_embedding.size == 0:
            return []
            
        return database.search(query_embedding[0], top_k)

    def format_retrieved_for_context(
        self, retrieved: List[RetrievedMemory], language: str = "zh"
    ) -> str:
        """A-Mem 自定义组装：text + time + metadata（context/keywords/tags）"""
        from prompts import render_prompt

        if not retrieved:
            template = "agent_context_empty_zh.jinja" if language == "zh" else "agent_context_empty_en.jinja"
            return render_prompt(template)

        unit_template = "amem_context_unit_zh.jinja" if language == "zh" else "amem_context_unit_en.jinja"
        context_lines = [
            render_prompt(
                unit_template,
                index=idx + 1,
                text=item.text,
                time=item.time,
                metadata=item.metadata or {},
            )
            for idx, item in enumerate(retrieved)
        ]
        return "\n\n".join(context_lines)

    def _extract_memory_insights(self, transcript: str, trace_scope_id: Optional[str] = None) -> MemoryInsights:
        prompt = build_metadata_prompt(transcript, self.language)
        messages = [{"role": "user", "content": prompt}]
        try:
            raw = self.llm.get_response_chat(
                messages,
                max_new_tokens=2048,
                temperature=self.metadata_temperature,
                response_format=METADATA_RESPONSE_FORMAT,
                verbose=False,
            )
            if hasattr(raw, '__iter__') and not isinstance(raw, str):
                raw = raw[0]
                
            if not raw:
                raise ValueError("Empty response from LLM for metadata extraction.")

            self.trace.log_llm_interaction(
                purpose="amem_extract_metadata",
                messages=messages,
                response=raw,
                scope_id=trace_scope_id,
                metadata={"temperature": self.metadata_temperature},
            )
            
            payload = self._safe_json_loads(raw) or {}
            return MemoryInsights.from_payload(payload)
        
        except Exception as exc:
            self.trace.log_llm_interaction(
                purpose="amem_extract_metadata",
                messages=messages,
                response=None,
                scope_id=trace_scope_id,
                metadata={"temperature": self.metadata_temperature},
                error=str(exc),
            )
            logger.warning("AMem metadata extraction failed: %s", exc)
            return MemoryInsights(
                context="General context",
                keywords=["general"],
                tags=["general"],
                summary="General memory",
            )

    def _maybe_apply_evolution(
        self,
        database: LocalFaissDatabase,
        metadata: Dict[str, Any],
        neighbors: List[RetrievedMemory],
        trace_scope_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not neighbors:
            return metadata

        neighbor_summary = self._prepare_neighbor_summary(neighbors)
        prompt = build_evolution_prompt(
            context=metadata.get("context", ""),
            summary=metadata.get("summary", ""),
            keywords=", ".join(metadata.get("keywords", [])),
            tags=", ".join(metadata.get("tags", [])),
            neighbor_summary=neighbor_summary,
            language=self.language,
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            raw = self.llm.get_response_chat(
                messages,
                max_new_tokens=2048,
                temperature=self.evolution_temperature,
                response_format=EVOLUTION_RESPONSE_FORMAT,
                verbose=False,
            )
            if hasattr(raw, '__iter__') and not isinstance(raw, str):
                raw = raw[0]
                
            if not raw:
                raise ValueError("Empty response from LLM for evolution.")

            self.trace.log_llm_interaction(
                purpose="amem_evolution_decision",
                messages=messages,
                response=raw,
                scope_id=trace_scope_id,
                metadata={"temperature": self.evolution_temperature, "neighbor_count": len(neighbors)},
            )
        except Exception as exc:
            self.trace.log_llm_interaction(
                purpose="amem_evolution_decision",
                messages=messages,
                response=None,
                scope_id=trace_scope_id,
                metadata={"temperature": self.evolution_temperature, "neighbor_count": len(neighbors)},
                error=str(exc),
            )
            logger.warning("AMem evolution LLM call failed: %s", exc)
            return metadata

        payload = self._safe_json_loads(raw)
        if not payload or not payload.get("should_evolve"):
            return metadata

        new_context = payload.get("new_note_context")
        if isinstance(new_context, str) and new_context.strip():
            metadata["context"] = new_context.strip()

        new_tags = _normalize_string_list(payload.get("new_note_tags"))
        if new_tags:
            metadata["tags"] = new_tags

        neighbor_updates = payload.get("neighbor_updates") or []
        if neighbor_updates:
            self._apply_neighbor_updates(database, neighbors, neighbor_updates, trace_scope_id=trace_scope_id)
        return metadata

    def _apply_neighbor_updates(
        self,
        database: LocalFaissDatabase,
        neighbors: List[RetrievedMemory],
        updates: Iterable[Dict[str, Any]],
        trace_scope_id: Optional[str] = None,
    ) -> None:
        neighbor_map = {
            mem.memory_id: mem
            for mem in neighbors
        }
        alias_to_uuid = getattr(self, "_neighbor_id_map", {}) or {}
        
        texts_to_embed = []
        updates_meta = []

        for update in updates:
            alias = update.get("memory_id")
            memory_id = alias_to_uuid.get(alias, alias)
            target = neighbor_map.get(memory_id)
            if not target:
                continue
                
            metadata_updates: Dict[str, Any] = {}
            new_context = update.get("context")
            if isinstance(new_context, str) and new_context.strip():
                metadata_updates["context"] = new_context.strip()
            new_tags = _normalize_string_list(update.get("tags"))
            if new_tags:
                metadata_updates["tags"] = new_tags
                
            if metadata_updates:
                merged_meta = {**(target.metadata or {}), **metadata_updates}
                text_for_embed = self.build_text_for_embedding(target.text, metadata=merged_meta)
                texts_to_embed.append(text_for_embed)
                updates_meta.append((memory_id, target.text, metadata_updates, target.time))
                
        if not texts_to_embed:
            return
            
        embeddings = self._embed_texts(texts_to_embed)
        for i, (m_id, text, m_updates, t_time) in enumerate(updates_meta):
            success = database.update_memory(
                memory_id=m_id, 
                new_text=text, 
                new_source_index=None,
                new_time=t_time,
                metadata_updates=m_updates,
                new_embedding=embeddings[i]
            )
            self.trace.log_memory_operation(
                operation="UPDATE",
                memory_id=m_id,
                scope_id=trace_scope_id,
                metadata={"reason": "neighbor_update"},
                after={"text": text, "time": t_time, "metadata_updates": m_updates},
                status="ok" if success else "failed",
            )

    def _build_retrieval_query(self, question_text: str) -> str:
        keywords = self._generate_query_keywords(question_text)
        if keywords:
            return f"{question_text}\nKeywords: {', '.join(keywords)}"
        return question_text

    def _generate_query_keywords(self, question_text: str) -> List[str]:
        prompt = build_query_prompt(question_text, language=self.language)
        messages = [{"role": "user", "content": prompt}]
        try:
            raw = self.llm.get_response_chat(
                messages,
                max_new_tokens=256,
                temperature=0.0,
                response_format=QUERY_RESPONSE_FORMAT,
                verbose=False,
            )
            if hasattr(raw, '__iter__') and not isinstance(raw, str):
                raw = raw[0]
                
            if not raw:
                raise ValueError("Empty response from LLM for keyword generation.")

            self.trace.log_llm_interaction(
                purpose="amem_query_keywords",
                messages=messages,
                response=raw,
                metadata={"temperature": 0.0},
            )
                
            payload = self._safe_json_loads(raw) or {}
            keywords = payload.get("keywords")
            normalized = _normalize_string_list(keywords)
            if normalized:
                return normalized[:8]
        except Exception as exc:
            self.trace.log_llm_interaction(
                purpose="amem_query_keywords",
                messages=messages,
                response=None,
                metadata={"temperature": 0.0},
                error=str(exc),
            )
            logger.warning("AMem query keyword generation failed: %s", exc)
        tokens = [token.strip() for token in question_text.split() if len(token.strip()) > 2]
        return tokens[:5]

    def _render_memory_text(
        self,
        session_date: Optional[Any],
        insights: MemoryInsights,
        content_block: str,
    ) -> str:
        date_str = str(session_date or "unknown date")
        keywords_line = ", ".join(insights.keywords)
        tags_line = ", ".join(insights.tags)
        return (
            f"- Memory Date: {date_str}\n"
            f"- Memory Context: {insights.context}\n"
            f"- Memory Keywords: {keywords_line}\n"
            f"- Memory Tags: {tags_line}\n"
            f"- Memory Summary: {insights.summary}\n"
            f"- Memory Content:\n"
            f"{content_block.strip()}"
        )

    def _format_session_transcript(
        self,
        turns: List[Any],
        session_date: Optional[Any],
    ) -> str:
        lines = [f"对话日期：{session_date or 'unknown'}"]
        for turn in turns:
            speaker = (turn.speaker or "unknown").strip()
            content = (turn.content or "").strip()
            if content:
                lines.append(f"**{speaker}**: {content}")
        return "\n".join(lines)

    def _format_memory_content(self, turns: List[Any]) -> str:
        lines: List[str] = []
        for turn in turns:
            speaker = (turn.speaker or "unknown").strip()
            content = (turn.content or "").strip()
            if content:
                lines.append(f"{speaker}: {content}")
        return "\n".join(lines)

    def _prepare_neighbor_summary(self, neighbors: List[RetrievedMemory]) -> str:
        summaries = []
        self._neighbor_id_map: Dict[str, str] = {}
        for idx, mem in enumerate(neighbors):
            alias = str(idx)
            memory_uuid = mem.memory_id
            self._neighbor_id_map[alias] = memory_uuid
            
            date = mem.metadata.get("date") or "unknown"
            context = mem.metadata.get("context") or ""
            keywords = _normalize_string_list(mem.metadata.get("keywords"))
            tags = _normalize_string_list(mem.metadata.get("tags"))
            content = self._extract_memory_body(mem.text)
            
            summaries.append(
                f"[id={alias}] date={date}\n"
                f"Context: {context}\n"
                f"Keywords: {', '.join(keywords)}\n"
                f"Tags: {', '.join(tags)}\n"
                f"Content: {content}"
            )
        return "\n\n".join(summaries)

    def _extract_memory_body(self, text: str) -> str:
        return text.strip()

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
            if start != -1 and end != -1 and start <= end:
                try:
                    return json.loads(value[start : end + 1])
                except json.JSONDecodeError:
                    pass
            logger.warning("AMem failed to parse JSON payload: %s", value)
            return None
