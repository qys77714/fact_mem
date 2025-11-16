#!/usr/bin/env bash
task=lmb_event
chat_model=Qwen2.5-7B-Instruct
embed_model_name=qwen3-embedding-0.6b
method=rag
topk=20
context_token_limit=32768
granularity=session
output="experiment/${task}_${chat_model}_${method}_top${topk}_${granularity}.jsonl"


python src/1_run_generation.py \
  --task $task \
  --output $output \
  --chat_model "$chat_model" \
  --embed_model_name "$embed_model_name" \
  --method "$method" \
  --topk "$topk" \
  --granularity "$granularity" \
  --context_token_limit "$context_token_limit" \
  --run_mode offline