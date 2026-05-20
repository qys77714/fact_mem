#!/usr/bin/env bash
# MEME gold_facts oracle eval.
#
# Prerequisites (separate terminals):
#   vllm_model_runner_4090/script/0_run_model.sh      # gemma4-26B @ :7111
#   vllm_model_runner_4090/script/0_run_embedding.sh  # qwen3-embedding @ :7110
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

uv run python test/meme/meme_gold_facts_eval.py \
  --dataset "${ROOT}/data/raw_data/MEME/meme_nofiller.json" \
  --answer-model gemma4-26B \
  --judge-model gemma4-26B \
  --embedding-model qwen3-embedding-8b \
  --phases after \
  --retrieve-topk 20 \
  --answer-concurrency 8 \
  --judge-concurrency 8 \
  --html \
  "$@"
