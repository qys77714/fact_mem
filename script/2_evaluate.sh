# 定义输入路径数组
INPUT_PATHS=(
    "experiment/lmb_event_Qwen3-8B_amem_top20_session_online.jsonl"
    # "experiment/lmb_event_Qwen3-8B_rag_top20_session_offline_mcq.jsonl"
    # "experiment/lmb_event_Qwen3-8B_rag_top20_session_online.jsonl"
    # "experiment/lmb_event_Qwen3-8B_rag_top20_session_online_mcq.jsonl"
)

JUDGE_MODEL=gpt-4o-mini

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