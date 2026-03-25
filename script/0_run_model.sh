# 定义可用的 GPU 和端口
gpus=(1,2,3,4)
ports=(7111)
model_name=Qwen3.5-27B-FP8
MAX_MODEL_LEN=32768
GPU_MEM_UTIL=0.9
MAX_NUM_BATCHED_TOKENS=$((4096 * 8))

# 循环遍历 GPU 和端口
for i in ${!gpus[@]}; do
    gpu_count=$(echo ${gpus[$i]} | awk -F',' '{print NF}')
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    export CUDA_VISIBLE_DEVICES=${gpus[$i]} \
        && vllm serve /data/zjj/models/Qwen/${model_name} \
        --served-model-name ${model_name} \
        --host 0.0.0.0 \
        --port ${ports[$i]} \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
        --dtype bfloat16 \
        --tensor-parallel-size ${gpu_count} \
        --gpu-memory-utilization "${GPU_MEM_UTIL}" \
        --api-key zjj &
done

wait