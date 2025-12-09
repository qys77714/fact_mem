# 定义输入路径数组
INPUT_PATHS=(
    "experiment/qwen-plus_MemOS_offline_mcq.jsonl"
    "experiment/qwen-plus_MemOS_online_mcq.jsonl"
    "experiment/qwen-plus_MemOS_offline_oqa.jsonl"
    "experiment/qwen-plus_MemOS_online_oqa.jsonl"
    "experiment/qwen3-8b_MemOS_offline_mcq.jsonl"
    "experiment/qwen3-8b_MemOS_online_mcq.jsonl"
    "experiment/qwen3-8b_MemOS_offline_oqa.jsonl"
    "experiment/qwen3-8b_MemOS_online_oqa.jsonl"
)

JUDGE_MODEL=qwen3-32b

# 并行执行（放到后台）
echo "开始并行执行..."
pids=()  # 存储进程ID

for input_path in "${INPUT_PATHS[@]}"; do
    python src/2_evaluate_qa.py \
        --input_path "$input_path" \
        --evaluate_task lmb \
        --judge_model $JUDGE_MODEL \
        --use_cot &
    pids+=($!)  # 保存进程ID
    echo "后台任务启动，进程ID: ${pids[-1]}"
    sleep 2
done

# 等待所有后台任务完成
echo "等待所有后台任务完成..."
for pid in "${pids[@]}"; do
    wait "$pid"
    echo "进程 $pid 已完成"
done