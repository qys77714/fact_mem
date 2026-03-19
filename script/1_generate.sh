#!/usr/bin/env bash
set -euo pipefail

benchmark=test
answer_model=qwen3-max
manager_model=qwen3-max
embedding_model=qwen3-embedding-8b
embedding_base_url=http://localhost:7110/v1/
embedding_api_key=zjj
method=mem0
retrieve_topk=10
memory_token_limit=32768
memory_granularity=all
output="experiment/${benchmark}_${manager_model}_${method}_top${retrieve_topk}_${memory_granularity}.jsonl"
agent_trace_dir="logs/answer_agent_trace"

python src/pipeline_generate.py \
  --benchmark "$benchmark" \
  --output $output \
  --answer_model "$answer_model" \
  --manager_model "$manager_model" \
  --embedding_model "$embedding_model" \
  --embedding_base_url "$embedding_base_url" \
  --embedding_api_key "$embedding_api_key" \
  --method "$method" \
  --retrieve_topk "$retrieve_topk" \
  --memory_granularity "$memory_granularity" \
  --memory_token_limit "$memory_token_limit" \
  --agent_trace_dir "$agent_trace_dir"
