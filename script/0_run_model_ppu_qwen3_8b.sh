# 定义可用的 GPU 和端口
gpus=(1)
ports=(7112)
model_name=Qwen3-8B
MAX_MODEL_LEN=32768
GPU_MEM_UTIL=0.9
MAX_NUM_BATCHED_TOKENS=$((4096 * 32))

# Some vendor libraries dlopen("libcuda.so") directly, so expose CUDA libs explicitly.
export CUDA_PATH=/usr/local/cuda
export PPU_SDK=/usr/local/PPU_SDK
export PATH=${PPU_SDK}/bin:${CUDA_PATH}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# 循环遍历 GPU 和端口
for i in ${!gpus[@]}; do
    gpu_count=$(echo ${gpus[$i]} | awk -F',' '{print NF}')
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    # tool_llm / OpenAI tools + tool_choice=auto 需要 vLLM 开启自动工具解析（否则 400）：
    #   "auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser
    # Qwen3-8B Instruct 与 Qwen2.5 类似时用 hermes；若是 Qwen3-Coder 系列可改为 qwen3_coder。
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
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        --api-key zjj &
done

wait