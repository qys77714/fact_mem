import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def process_longmemeval(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processed = []
    for item in records:
        qa_entry = {
            "question": item.get("question"),
            "answer": item.get("answer"),
            "question_time": item.get("question_date"),
        }
        chat_time = item.get("haystack_dates") or []
        chat_time = sorted(chat_time)
        chat_history = []
        for session in item.get("haystack_sessions") or []:
            converted_session = []
            for message in session or []:
                converted_session.append(
                    {
                        "speaker": message.get("role"),
                        "content": message.get("content"),
                    }
                )
            chat_history.append(converted_session)
        processed.append(
            {
                "history_name": item.get("question_id"),
                "qa": [qa_entry],
                "chat_time": chat_time,
                "chat_history": chat_history,
            }
        )
    return processed


def process_locomos(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processed = []
    session_pattern = re.compile(r"session_(\d+)$")
    for item in records:
        conversation = item.get("conversation") or {}
        session_indices = set()
        for key in conversation.keys():
            match = session_pattern.match(key)
            if match:
                session_indices.add(int(match.group(1)))
        session_indices = sorted(session_indices)
        chat_time: List[Any] = []
        chat_history: List[List[Dict[str, Any]]] = []
        for idx in session_indices:
            chat_time.append(conversation.get(f"session_{idx}_date_time"))
            session_messages = conversation.get(f"session_{idx}") or []
            converted_session = [
                {"speaker": message.get("speaker"), "content": message.get("text")}
                for message in session_messages
            ]
            chat_history.append(converted_session)
        processed.append(
            {
                "history_name": item.get("sample_id"),
                "qa": item.get("qa") or [],
                "chat_time": chat_time,
                "chat_history": chat_history,
            }
        )
    return processed


def determine_processor(input_path: Path):
    name = input_path.name.lower()
    if "longmemeval" in name:
        return process_longmemeval
    if "locomo" in name:
        return process_locomos
    raise ValueError("文件名需包含 'longmemeval' 或 'locomo'")


def main():
    parser = argparse.ArgumentParser(description="转换历史记录文件格式")
    parser.add_argument("--input_path", type=Path, help="输入 JSON 文件路径")
    parser.add_argument("--output_path", type=Path, help="输出 JSON 文件路径")
    args = parser.parse_args()

    data = load_json(args.input_path)
    if isinstance(data, dict):
        records = data.get("data") or data.get("records") or []
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError("输入 JSON 须为对象或数组")

    processor = determine_processor(args.input_path)
    result = processor(records)
    dump_json(args.output_path, result)


if __name__ == "__main__":
    main()