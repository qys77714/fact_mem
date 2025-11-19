import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

from openai import OpenAI
from memory.MemorySystem import MemorySystem
import asyncio
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="lme_s")
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", default="full_context")
    parser.add_argument("--chat_model", required=True)
    parser.add_argument("--embed_model_name", required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--granularity", type=str, default="turn")
    parser.add_argument("--context_token_limit", type=int, default=8196)
    parser.add_argument("--run_mode", choices=["offline", "advanced"], default="offline")
    return parser.parse_args()


TASK_TO_DATASET = {
    "lme_oracle": "data/preprocessed/longmemeval_oracle.json",
    "lme_m": "data/preprocessed/longmemeval_m_cleaned.json",
    "lme_s": "data/preprocessed/longmemeval_s_cleaned.json",
    "locomo": "data/preprocessed/locomo10.json",
    "lmb_event": "data/preprocessed/LifeMemBench_event.json"
}


def _normalize_date(date_value: Any) -> str:
    if date_value is None:
        return ""
    if isinstance(date_value, str):
        return date_value[:10]
    return str(date_value)[:10]


def _chunk_chat_by_day(chat_history: List[Any], chat_dates: List[Any]) -> OrderedDict[str, Dict[str, List[Any]]]:
    day_map: OrderedDict[str, Dict[str, List[Any]]] = OrderedDict()
    for idx, message in enumerate(chat_history):
        timestamp = chat_dates[idx] if idx < len(chat_dates) else (chat_dates[-1] if chat_dates else "")
        day_key = _normalize_date(timestamp)
        bucket = day_map.setdefault(day_key, {"chat": [], "dates": []})
        bucket["chat"].append(message)
        bucket["dates"].append(timestamp)
    return day_map


def _group_questions_by_day(qa_list: List[Dict[str, Any]]) -> OrderedDict[str, Dict[str, List[Any]]]:
    question_map: OrderedDict[str, Dict[str, List[Any]]] = OrderedDict()
    for idx, qa in enumerate(qa_list):
        day_key = _normalize_date(qa.get("question_time"))
        bucket = question_map.setdefault(day_key, {"indices": [], "questions": [], "dates": [], "mcq_options": []})
        bucket["indices"].append(idx)
        bucket["questions"].append(qa.get("question"))
        bucket["dates"].append(qa.get("question_time"))
        bucket["mcq_options"].append(qa.get("options", None))
    return question_map


def _build_output_path(base_path: Path, suffix: str) -> Path:
    if base_path.suffix:
        return base_path.with_name(f"{base_path.stem}{suffix}{base_path.suffix}")
    return base_path.with_name(f"{base_path.name}{suffix}")


def _write_records(handle: TextIO, task: str, entry: Dict[str, Any], qa_subset: List[Dict[str, Any]], responses: List[str]) -> int:
    assert len(qa_subset) == len(responses), f"Length mismatch: {len(qa_subset)} vs {len(responses)}"

    written = 0
    if task.startswith("lme"):
        hypothesis = responses[0]
        for _ in qa_subset:
            record = {"question_id": entry.get("history_name"), "hypothesis": hypothesis}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
        return written

    if task.startswith("lmb"):
        for idx, qa in enumerate(qa_subset):
            answer_text = responses[idx]
            record = {
                "question": qa.get("question"),
                "answer": qa.get("answer"),
                "model_answer": answer_text,
                "question_type": qa.get("question_type"),
                "options": qa.get("options"),
                "golden_option": qa.get("golden_option")
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
        return written
    
    if task.startswith("locomo"):
        for idx, qa in enumerate(qa_subset):
            answer_text = responses[idx]
            record = {
                "question": qa.get("question"),
                "answer": qa.get("answer"),
                "model_answer": answer_text
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
        return written


async def run_offline(args: argparse.Namespace) -> None:
    dataset_path = TASK_TO_DATASET.get(args.task, "")
    with open(dataset_path, "r", encoding="utf-8") as file:
        entries = json.load(file)

    memory_system = MemorySystem(
        method_name=args.method,
        chat_model=args.chat_model,
        embed_model_name=args.embed_model_name,
        embed_client=OpenAI(api_key="zjj", base_url="http://localhost:7100/v1/"),
        database_root=f"MemDB/{args.task}/{args.chat_model}_{args.granularity}/{args.method}",
        context_token_limit=args.context_token_limit
    )
    stored_histories = set()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for entry in tqdm(entries, desc="Processing entries (offline)"):
            history_name = str(entry.get("history_name"))
            if history_name not in stored_histories:
                chat_history = entry.get("chat_history", [])
                chat_dates = entry.get("chat_time", [])
                memory_system.store_history(history_name, chat_history, chat_dates, granularity=args.granularity)
                stored_histories.add(history_name)

            qa_list = entry.get("qa", []) or []

            questions = [qa.get("question") for qa in qa_list]
            dates = [qa.get("question_time") for qa in qa_list]
            responses = await memory_system.answer_question_cn(
                history_name=history_name,
                questions=questions,
                question_dates=dates,
                top_k=args.topk,
            )
            _write_records(out_f, args.task, entry, qa_list, responses)


async def main_online(
    memory_system: MemorySystem,
    entry: Dict[str, Any],
    args: argparse.Namespace,
    online_handle: TextIO,
    online_handle_mcq: TextIO
) -> None:
    history_name = str(entry.get("history_name"))
    chat_history = entry.get("chat_history", []) or []
    chat_dates = entry.get("chat_time", []) or []
    qa_list = entry.get("qa", []) or []

    day_chunks = _chunk_chat_by_day(chat_history, chat_dates)
    question_map = _group_questions_by_day(qa_list)

    for day, payload in day_chunks.items():
        print(payload["dates"][0][:10])
        memory_system.store_history(
            history_name,
            payload["chat"],
            payload["dates"],
            granularity=args.granularity
        )
        question_payload = question_map.pop(day, None)
        if question_payload and question_payload["questions"]:
            # 非选择题
            responses = await memory_system.answer_question_cn(
                history_name=history_name,
                questions=question_payload["questions"],
                question_dates=question_payload["dates"],
                mcq_options=None,
                top_k=args.topk,
            )
            qa_subset = [qa_list[idx] for idx in question_payload["indices"]]
            if _write_records(online_handle, args.task, entry, qa_subset, responses):
                online_handle.flush()

            # 选择题
            if question_payload['mcq_options'][0] != None:
                responses_mcq = await memory_system.answer_question_cn(
                    history_name=history_name,
                    questions=question_payload["questions"],
                    question_dates=question_payload["dates"],
                    mcq_options=question_payload['mcq_options'],
                    top_k=args.topk,
                )
                if _write_records(online_handle_mcq, args.task, entry, qa_subset, responses_mcq):
                    online_handle_mcq.flush()
            


async def main_advanced(args: argparse.Namespace) -> None:
    dataset_path = TASK_TO_DATASET.get(args.task, "")
    with open(dataset_path, "r", encoding="utf-8") as file:
        entries = json.load(file)

    memory_system = MemorySystem(
        method_name=args.method,
        chat_model=args.chat_model,
        embed_model_name=args.embed_model_name,
        embed_client=OpenAI(api_key="zjj", base_url="http://localhost:7100/v1/"),
        database_root=f"MemDB/{args.task}/{args.chat_model}_{args.granularity}/{args.method}",
        context_token_limit=args.context_token_limit
    )

    base_path = Path(args.output)
    online_path = _build_output_path(base_path, "_online")
    offline_path = _build_output_path(base_path, "_offline")
    online_path_mcq = _build_output_path(base_path, "_online_mcq")
    offline_path_mcq = _build_output_path(base_path, "_offline_mcq")

    online_path.parent.mkdir(parents=True, exist_ok=True)
    offline_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        open(online_path, "w", encoding="utf-8") as online_f,
        open(offline_path, "w", encoding="utf-8") as offline_f,
        open(online_path_mcq, "w", encoding="utf-8") as online_f_mcq,
        open(offline_path_mcq, "w", encoding="utf-8") as offline_f_mcq
    ):
        for entry in tqdm(entries, desc="Processing entries (update)"):
            history_name = str(entry.get("history_name"))
            qa_list = entry.get("qa", []) or []
            questions = [qa.get("question") for qa in qa_list]
            question_dates = [qa.get("question_time") for qa in qa_list]
            mcq_options = [qa.get("options", None) for qa in qa_list]
            # print(mcq_options)

            await main_online(memory_system, entry, args, online_f, online_f_mcq)
            
            # 非选择题
            offline_responses = await memory_system.answer_question_cn(
                    history_name=history_name,
                    questions=questions,
                    question_dates=question_dates,
                    mcq_options=None,
                    top_k=args.topk,
                )
            
            _write_records(offline_f, args.task, entry, qa_list, offline_responses)
            
            # 选择题
            if mcq_options[0] != None:
                offline_responses_mcq = await memory_system.answer_question_cn(
                    history_name=history_name,
                    questions=questions,
                    question_dates=question_dates,
                    mcq_options=mcq_options,
                    top_k=args.topk,
                )
                _write_records(offline_f_mcq, args.task, entry, qa_list, offline_responses_mcq)


async def main() -> None:
    args = parse_args()
    if args.run_mode == "advanced":
        await main_advanced(args)
    else:
        await run_offline(args)


if __name__ == "__main__":
    asyncio.run(main())