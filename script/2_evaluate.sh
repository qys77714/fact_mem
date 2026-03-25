#!/usr/bin/env bash
set -euo pipefail

# 仓库根目录（无论从哪执行本脚本）
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 定义输入路径数组
INPUT_PATHS=(
  experiment/locomo_gran4_mem0_nodel_Qwen3.5-27B-FP8_top20.jsonl
  experiment/locomo_gran4_mem0_Qwen3.5-27B-FP8_top20.jsonl
)

JUDGE_MODEL=Qwen3.5-27B-FP8
# 仅写入 eval_judge.jsonl / --csv 的 benchmark 字段；不传也可从路径推断 locomo
BENCHMARK=locomo

# 并行时两个进程各默认 max_concurrency=20，易打满 API；可按需调低
MAX_CONCURRENCY=4

# 可选：统一追加到 CSV，便于对比多次运行
# CSV_OUT="$ROOT/experiment/eval_summary.csv"

echo "开始并行执行..."
pids=()

for input_path in "${INPUT_PATHS[@]}"; do
  cmd=(
    uv run python src/pipeline_evaluate.py
    --input "$input_path"
    --judge_model "$JUDGE_MODEL"
    --benchmark "$BENCHMARK"
    --max_concurrency "$MAX_CONCURRENCY"
    --write_back
  )
  # 若需 CSV：取消下一行注释，并在上方设置 CSV_OUT
  # cmd+=(--csv "$CSV_OUT")

  "${cmd[@]}" &
  pids+=($!)
  echo "后台任务启动，进程ID: ${pids[-1]}  input=$input_path"
  sleep 2
done

echo "等待所有后台任务完成..."
for pid in "${pids[@]}"; do
  wait "$pid"
  echo "进程 $pid 已完成"
done
