from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

from memory import BaseMemoryMethod, MemoryDatabase, RetrievedMemory
from .prompt import (
    build_metadata_prompt,
    build_evolution_prompt,
    build_query_prompt,
    METADATA_RESPONSE_FORMAT,
    EVOLUTION_RESPONSE_FORMAT,
    QUERY_RESPONSE_FORMAT,
)

logger = logging.getLogger(__name__)


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


class AMemMemoryMethod(BaseMemoryMethod):
    def __init__(
        self,
        embed_model_name: str,
        llm_client,
        embed_client = None,
        database_root: Optional[str] = None,
        related_memory_top_k: int = 5,
        metadata_temperature: float = 0.0,
        evolution_temperature: float = 0.2,
        enable_evolution: bool = True,
        language: str = "en",
    ) -> None:
        if llm_client is None:
            raise ValueError("llm_client must be provided for AMemMemoryMethod.")
        super().__init__(
            embed_model_name=embed_model_name,
            embed_client=embed_client,
            llm_client=llm_client,
            storage_namespace="amem",
            database_root=database_root,
        )
        self.llm = llm_client
        self.related_memory_top_k = max(1, related_memory_top_k)
        self.metadata_temperature = metadata_temperature
        self.evolution_temperature = evolution_temperature
        self.enable_evolution = enable_evolution
        self.language = language

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
        work_items = self._iter_work_items(chat_history, granularity)

        for session_idx, turn_idx, turns in tqdm(
            work_items,
            desc=f"Storing AMem history {history_name}",
        ):
            session_date = self._resolve_session_date(dates, session_idx)
            content_block = self._format_memory_content(turns)
            if not content_block.strip():
                continue

            transcript = self._format_session_transcript(turns, session_date)
            insights = self._extract_memory_insights(transcript)

            metadata = self._build_metadata(
                history_name=history_name,
                session_idx=session_idx,
                turn_idx=turn_idx,
                session_date=session_date,
            )
            metadata.update(
                {
                    "context": insights.context,
                    "keywords": insights.keywords,
                    "tags": insights.tags,
                    "summary": insights.summary,
                    "granularity": granularity,
                }
            )

            neighbors = self._fetch_related_memories(
                database=database,
                insights=insights,
                content_block=content_block,
            )
            if neighbors:
                metadata["related_ids"] = [
                    mem.metadata.get("memory_id")
                    for mem in neighbors
                    if mem.metadata.get("memory_id")
                ]
                if self.enable_evolution:
                    metadata = self._maybe_apply_evolution(
                        database=database,
                        metadata=metadata,
                        neighbors=neighbors,
                    )

            memory_text = self._render_memory_text(
                session_date=session_date,
                insights=insights,
                content_block=content_block,
            )
            database.add(memory_text, metadata)

    def retrieve(
        self,
        history_name: str,
        question_text: str,
        top_k: int,
    ) -> List[RetrievedMemory]:
        if top_k <= 0 or not question_text:
            return []
        database = self._get_database(history_name)
        query = self._build_retrieval_query(question_text)
        return database.search(query, top_k)

    def _extract_memory_insights(self, transcript: str) -> MemoryInsights:
        prompt = build_metadata_prompt(transcript, self.language)
        try:
            raw = self.llm.get_response_chat(
                [{"role": "user", "content": prompt}],
                max_new_tokens=2048,
                temperature=self.metadata_temperature,
                response_format=METADATA_RESPONSE_FORMAT,
                verbose=True,
            )
            if not raw:
                raise ValueError("Empty response from LLM for metadata extraction.")
            
            payload = self._safe_json_loads(raw) or {}
            return MemoryInsights.from_payload(payload)
        
        except Exception as exc:
            logger.warning("AMem metadata extraction failed: %s", exc)
            return MemoryInsights(
                context="General context",
                keywords=["general"],
                tags=["general"],
                summary="General memory",
            )

    def _fetch_related_memories(
        self,
        database: MemoryDatabase,
        insights: MemoryInsights,
        content_block: str,
    ) -> List[RetrievedMemory]:
        search_query = "\n".join(
            [
                insights.context,
                ", ".join(insights.keywords),
                content_block,
            ]
        )
        return database.search(search_query, self.related_memory_top_k)

    def _maybe_apply_evolution(
        self,
        database: MemoryDatabase,
        metadata: Dict[str, Any],
        neighbors: List[RetrievedMemory]
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
        try:
            raw = self.llm.get_response_chat(
                [{"role": "user", "content": prompt}],
                max_new_tokens=2048,
                temperature=self.evolution_temperature,
                response_format=EVOLUTION_RESPONSE_FORMAT,
                verbose=True,
            )
            if not raw:
                raise ValueError("Empty response from LLM for evolution.")
        except Exception as exc:
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
            self._apply_neighbor_updates(database, neighbors, neighbor_updates)
        return metadata

    def _apply_neighbor_updates(
        self,
        database: MemoryDatabase,
        neighbors: List[RetrievedMemory],
        updates: Iterable[Dict[str, Any]],
    ) -> None:
        neighbor_map = {
            mem.metadata.get("memory_id"): mem
            for mem in neighbors
            if mem.metadata.get("memory_id")
        }
        alias_to_uuid = getattr(self, "_neighbor_id_map", {}) or {}
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
                try:
                    database.update_memory(memory_id, target.text, metadata_updates)
                except Exception as exc:
                    logger.warning(
                        "AMem neighbor update failed for %s: %s",
                        memory_id,
                        exc,
                    )

    def _build_retrieval_query(self, question_text: str) -> str:
        keywords = self._generate_query_keywords(question_text)
        if keywords:
            return f"{question_text}\nKeywords: {', '.join(keywords)}"
        return question_text

    def _generate_query_keywords(self, question_text: str) -> List[str]:
        prompt = build_query_prompt(question_text, language=self.language)
        try:
            raw = self.llm.get_response_chat(
                [{"role": "user", "content": prompt}],
                max_new_tokens=256,
                temperature=0.0,
                response_format=QUERY_RESPONSE_FORMAT,
                verbose=True,
            )
            if not raw:
                raise ValueError("Empty response from LLM for keyword generation.")
            # Parse the response to extract keywords
            payload = self._safe_json_loads(raw) or {}
            keywords = payload.get("keywords")
            normalized = _normalize_string_list(keywords)
            if normalized:
                return normalized[:8]
        except Exception as exc:
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
            f"<MemoryContent>\n{content_block.strip()}\n</MemoryContent>"
        )

    def _format_session_transcript(
        self,
        turns: List[Dict[str, str]],
        session_date: Optional[Any],
    ) -> str:
        lines = [f"对话日期：{session_date or 'unknown'}"]
        for turn in turns:
            speaker = (turn.get("speaker") or "unknown").strip()
            content = (turn.get("content") or "").strip()
            if content:
                lines.append(f"**{speaker}**: {content}")
        return "\n".join(lines)

    def _format_memory_content(self, turns: List[Dict[str, str]]) -> str:
        lines: List[str] = []
        for turn in turns:
            speaker = (turn.get("speaker") or "unknown").strip()
            content = (turn.get("content") or "").strip()
            if content:
                lines.append(f"{speaker}: {content}")
        return "\n".join(lines)

    def _prepare_neighbor_summary(self, neighbors: List[RetrievedMemory]) -> str:
        summaries = []
        self._neighbor_id_map: Dict[str, str] = {}
        for idx, mem in enumerate(neighbors):
            alias = str(idx)
            memory_uuid = mem.metadata.get("memory_id") or "unknown"
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

    def _iter_work_items(
        self,
        chat_history: List[List[Dict[str, str]]],
        granularity: str,
    ) -> List[Tuple[int, Optional[int], List[Dict[str, str]]]]:
        items: List[Tuple[int, Optional[int], List[Dict[str, str]]]] = []
        for session_idx, session in enumerate(chat_history):
            if not session:
                continue
            if granularity == "session":
                items.append((session_idx, None, session))
            else:
                for turn_idx, turn in enumerate(session):
                    items.append((session_idx, turn_idx, [turn]))
        return items

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
            "source": "amem",
        }
        if session_idx is not None:
            metadata["session"] = session_idx
        if turn_idx is not None:
            metadata["turn"] = turn_idx
        if session_date is not None:
            metadata["date"] = session_date
        return metadata

    def _resolve_session_date(
        self,
        chat_dates: List[Any],
        index: int,
    ) -> Optional[Any]:
        if not chat_dates:
            return None
        if 0 <= index < len(chat_dates):
            return chat_dates[index]
        return chat_dates[-1]

    def _extract_memory_body(self, text: str) -> str:
        start = text.find("<MemoryContent>")
        end = text.find("</MemoryContent>")
        if start != -1 and end != -1 and start < end:
            return text[start + len("<MemoryContent>") : end].strip()
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
            if start != -1 and end != -1 and start < end:
                try:
                    return json.loads(value[start : end + 1])
                except json.JSONDecodeError:
                    logger.warning("AMem failed to parse JSON payload: %s", value)
        return None