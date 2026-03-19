#!/usr/bin/env bash
set -euo pipefail

# 定义输入路径数组
INPUT_PATHS=(
/data/zjj/project/easy_mem/experiment/test_qwen3-max_mem0_top10_2.jsonl
)

JUDGE_MODEL=qwen3-max
BENCHMARK=lme

# 并行执行（放到后台）
echo "开始并行执行..."
pids=()  # 存储进程ID

for input_path in "${INPUT_PATHS[@]}"; do
    python src/pipeline_evaluate.py \
        --input "$input_path" \
        --benchmark "$BENCHMARK" \
        --judge_model "$JUDGE_MODEL" \
        --write_back &
    pids+=($!)
    echo "后台任务启动，进程ID: ${pids[-1]}"
    sleep 2
done

# 等待所有后台任务完成
echo "等待所有后台任务完成..."
for pid in "${pids[@]}"; do
    wait "$pid"
    echo "进程 $pid 已完成"
done