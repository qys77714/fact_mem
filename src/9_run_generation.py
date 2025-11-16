import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List
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
    return parser.parse_args()

TASK_TO_DATASET = {
    "lme_oracle": "data/preprocessed/longmemeval_oracle.json",
    "lme_m": "data/preprocessed/longmemeval_m_cleaned.json",
    "lme_s": "data/preprocessed/longmemeval_s_cleaned.json",
    "locomo": "data/preprocessed/locomo10.json",
    "lmb_event": "data/preprocessed/LifeMemBench_event.json"
}

async def main() -> None:
    args = parse_args()
    with open(TASK_TO_DATASET.get(args.task, ""), 'r', encoding='utf-8') as file:
        entries = json.load(file)
    memory_system = MemorySystem(
        method_name=args.method, 
        chat_model=args.chat_model,
        embed_model_name=args.embed_model_name,
        embed_client=OpenAI(
            api_key="zjj", 
            base_url="http://localhost:7104/v1/"
        ),
        database_root=f"MemDB/{args.task}/{args.chat_model}_{args.granularity}/{args.method}",
        context_token_limit=8196
    )
    stored_histories = set()

    with open(args.output, "w", encoding="utf-8") as out_f:
        for entry in tqdm(entries, desc="Processing entries"):
            history_name = str(entry.get("history_name"))
            if history_name not in stored_histories:
                chat_history = entry.get("chat_history", [])
                chat_dates = entry.get("chat_time", [])
                memory_system.store_history(history_name, chat_history, chat_dates, granularity=args.granularity)
                stored_histories.add(history_name)

            questions = [x["question"] for x in entry.get("qa", [])]
            dates = [x["question_time"] for x in entry.get("qa", [])]
            responses = await memory_system.answer_question(
                history_name=history_name,
                questions=questions,
                question_dates=dates,
                top_k=args.topk,
            )

            for idx, qa in enumerate(entry.get("qa", [])):
                if args.task.startswith("lme"):
                    record = {"question_id": entry.get("history_name"), "hypothesis": responses[0]}
                elif args.task.startswith("locomo") or args.task.startswith("lmb"):
                    record = {
                        "question": qa.get("question"),
                        "answer": qa.get("answer"),
                        "model_answer": responses[idx]
                    }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")



if __name__ == "__main__":
    asyncio.run(main())