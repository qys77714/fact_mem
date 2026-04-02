uv run python script/case_study_ab_diff.py \
  --input-a experiment/locomo_gran4_append_only_Qwen2.5-7B-Instruct_top100000.jsonl \
  --input-b experiment/locomo_gran4_relmem_Qwen2.5-7B-Instruct_top100000.jsonl \
  --output experiment/case_study_a_win_b_lose.jsonl \
  --agent-trace-a logs/answer_agent_trace/append_only.jsonl \
  --agent-trace-b logs/answer_agent_trace/relmem.jsonl