from datetime import datetime

from prompts import render_prompt
from .schemas import FACT_RETRIEVAL_RESPONSE_FORMAT, UPDATE_MEMORY_RESPONSE_FORMAT


def build_fact_retrieval_system_prompt(
    user_name: str = "user",
    language: str = "zh",
    dialogue_format: str = "user_assistant",
) -> str:
    """Build system prompt for fact extraction (aligned with Mem0 USER_MEMORY_EXTRACTION_PROMPT)."""
    df = (dialogue_format or "user_assistant").strip().lower()
    if df == "named_speakers":
        template = "mem0_fact_retrieval_multi_zh.jinja" if language == "zh" else "mem0_fact_retrieval_multi_en.jinja"
    elif df == "user_assistant":
        template = "mem0_fact_retrieval_zh.jinja" if language == "zh" else "mem0_fact_retrieval_en.jinja"
    else:
        raise ValueError(
            "dialogue_format must be 'user_assistant' or 'named_speakers', "
            f"got {dialogue_format!r}"
        )
    return render_prompt(
        template,
        user_name=user_name,
        today_date=datetime.now().strftime("%Y-%m-%d"),
    )


def build_fact_retrieval_prompt(
    user_name: str,
    language: str = "zh",
    dialogue_format: str = "user_assistant",
) -> str:
    """Legacy: returns full prompt for single-message mode. Prefer build_fact_retrieval_system_prompt + user 'Input:\\n{transcript}'."""
    return build_fact_retrieval_system_prompt(
        user_name=user_name,
        language=language,
        dialogue_format=dialogue_format,
    )


def get_update_memory_messages_en(retrieved_old_memory_dict, response_content, custom_update_memory_prompt=None):
    if retrieved_old_memory_dict:
        current_memory_part = render_prompt(
            "mem0_current_memory_part_en.jinja",
            retrieved_old_memory_dict=str(retrieved_old_memory_dict),
        )
    else:
        current_memory_part = render_prompt("mem0_current_memory_empty_en.jinja")

    return render_prompt(
        "mem0_update_memory_default_en.jinja",
        current_memory_part=current_memory_part,
        response_content=response_content,
    )


def get_update_memory_messages_zh(retrieved_old_memory_dict, response_content, custom_update_memory_prompt=None):
    if retrieved_old_memory_dict:
        current_memory_part = render_prompt(
            "mem0_current_memory_part_zh.jinja",
            retrieved_old_memory_dict=str(retrieved_old_memory_dict),
        )
    else:
        current_memory_part = render_prompt("mem0_current_memory_empty_zh.jinja")

    return render_prompt(
        "mem0_update_memory_default_zh.jinja",
        current_memory_part=current_memory_part,
        response_content=response_content,
    )


def build_update_memory_messages(retrieved_old_memory_dict, response_content, language="zh", custom_update_memory_prompt=None):
    if language == "zh":
        return get_update_memory_messages_zh(retrieved_old_memory_dict, response_content, custom_update_memory_prompt)
    return get_update_memory_messages_en(retrieved_old_memory_dict, response_content, custom_update_memory_prompt)