# 定义可用的 GPU 和端口
gpus=(1,2,3,4)
ports=(7111)
model_name=gemma-4-26B-A4B-it
MAX_MODEL_LEN=8196
GPU_MEM_UTIL=0.90
MAX_NUM_BATCHED_TOKENS=$((4096 * 8))

# 循环遍历 GPU 和端口
for i in ${!gpus[@]}; do
    gpu_count=$(echo ${gpus[$i]} | awk -F',' '{print NF}')
    # Triton: this image's triton defaults to a PPU toolchain (llvm-irformatter + ptxas --ppu-backend-options). On stock NVIDIA + CUDA that fails (e.g. ptxas exit 127). Force vanilla CUDA path.
    export TRITON_WITH_CUDA=1
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
    export PPU_SDK="${PPU_SDK:-$CUDA_HOME}"
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    export CUDA_VISIBLE_DEVICES=${gpus[$i]} \
        && uv run --no-sync vllm serve /data/zjj/models/${model_name} \
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