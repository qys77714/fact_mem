# 定义可用的 GPU 和端口
gpus=(0,1)
ports=(7111)
model_name=Qwen3-30B-A3B-Thinking-2507
MAX_MODEL_LEN=32768
GPU_MEM_UTIL=0.8
MAX_NUM_BATCHED_TOKENS=$((4096 * 8))

# Some vendor libraries dlopen("libcuda.so") directly, so expose CUDA libs explicitly.
export CUDA_PATH=/usr/local/cuda
export PPU_SDK=/usr/local/PPU_SDK
export PATH=${PPU_SDK}/bin:${CUDA_PATH}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# 循环遍历 GPU 和端口
for i in ${!gpus[@]}; do
    gpu_count=$(echo ${gpus[$i]} | awk -F',' '{print NF}')
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    export CUDA_VISIBLE_DEVICES=${gpus[$i]} \
        && vllm serve /mnt/data_oss/models/${model_name} \
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