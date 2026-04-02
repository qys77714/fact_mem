#!/usr/bin/env bash
# exp_01 + main-result baselines: Append-Only, Recency-Only, mem0 (Direct-Decision), relmem (Current System).
# Same embedding / retrieve_topk / memory_granularity / answer_model; LLM 方法需 manager_model。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

benchmark=locomo
answer_model=Qwen2.5-7B-Instruct
manager_model=Qwen2.5-7B-Instruct
manager_max_new_tokens=8192
embedding_model=qwen3-embedding-8b
embedding_base_url=http://localhost:7110/v1/
embedding_api_key=zjj
retrieve_topk=20
memory_token_limit=4096
memory_granularity=4
# Agent traces: ${agent_trace_dir}/<experiment_name>/agent_*.jsonl — same <experiment_name> as logs/memory_trace/<experiment_name>/
agent_trace_dir="logs/answer_agent_trace"
parallel_episodes=10
mem0_extract_concurrency=5
relmem_relation_concurrency=5
answer_concurrency=10

output_append_only="experiment/${benchmark}_gran${memory_granularity}_append_only_${manager_model}_top${retrieve_topk}.jsonl"
output_recency_only="experiment/${benchmark}_gran${memory_granularity}_recency_only_${answer_model}_top${retrieve_topk}.jsonl"
output_mem0="experiment/${benchmark}_gran${memory_granularity}_mem0_${manager_model}_top${retrieve_topk}.jsonl"
output_relmem="experiment/${benchmark}_gran${memory_granularity}_relmem_${manager_model}_top${retrieve_topk}.jsonl"

# python src/pipeline_generate.py \
#   --benchmark "$benchmark" \
#   --output "$output_append_only" \
#   --answer_model "$answer_model" \
#   --manager_model "$manager_model" \
#   --manager_max_new_tokens "$manager_max_new_tokens" \
#   --embedding_model "$embedding_model" \
#   --embedding_base_url "$embedding_base_url" \
#   --embedding_api_key "$embedding_api_key" \
#   --method append_only \
#   --retrieve_topk "$retrieve_topk" \
#   --memory_granularity "$memory_granularity" \
#   --memory_token_limit "$memory_token_limit" \
#   --agent_trace_dir "$agent_trace_dir" \
#   --parallel_episodes "$parallel_episodes" \
#   --mem0-extract-concurrency "$mem0_extract_concurrency"

# python src/pipeline_generate.py \
#   --benchmark "$benchmark" \
#   --output "$output_recency_only" \
#   --answer_model "$answer_model" \
#   --embedding_model "$embedding_model" \
#   --embedding_base_url "$embedding_base_url" \
#   --embedding_api_key "$embedding_api_key" \
#   --method recency_only \
#   --retrieve_topk "$retrieve_topk" \
#   --memory_granularity "$memory_granularity" \
#   --memory_token_limit "$memory_token_limit" \
#   --agent_trace_dir "$agent_trace_dir" \
#   --parallel_episodes "$parallel_episodes"

# python src/pipeline_generate.py \
#   --benchmark "$benchmark" \
#   --output "$output_mem0" \
#   --answer_model "$answer_model" \
#   --manager_model "$manager_model" \
#   --manager_max_new_tokens "$manager_max_new_tokens" \
#   --embedding_model "$embedding_model" \
#   --embedding_base_url "$embedding_base_url" \
#   --embedding_api_key "$embedding_api_key" \
#   --method mem0 \
#   --retrieve_topk "$retrieve_topk" \
#   --memory_granularity "$memory_granularity" \
#   --memory_token_limit "$memory_token_limit" \
#   --agent_trace_dir "$agent_trace_dir" \
#   --parallel_episodes "$parallel_episodes" \
#   --mem0-extract-concurrency "$mem0_extract_concurrency"

python src/pipeline_generate.py \
  --benchmark "$benchmark" \
  --output "$output_relmem" \
  --answer_model "$answer_model" \
  --manager_model "$manager_model" \
  --manager_max_new_tokens "$manager_max_new_tokens" \
  --embedding_model "$embedding_model" \
  --embedding_base_url "$embedding_base_url" \
  --embedding_api_key "$embedding_api_key" \
  --method relmem \
  --retrieve_topk "$retrieve_topk" \
  --memory_granularity "$memory_granularity" \
  --memory_token_limit "$memory_token_limit" \
  --agent_trace_dir "$agent_trace_dir" \
  --parallel_episodes "$parallel_episodes" \
  --mem0-extract-concurrency "$mem0_extract_concurrency" \
  --relmem-relation-concurrency "$relmem_relation_concurrency" \
  --answer-concurrency "$answer_concurrency"

echo "Done. Compare JSONL outputs and logs/memory_trace/<experiment_name>/ then run pipeline_evaluate on each."
