# Some vendor libraries dlopen("libcuda.so") directly, so expose CUDA libs explicitly.
export CUDA_PATH=/usr/local/cuda
export PPU_SDK=/usr/local/PPU_SDK
export PATH=${PPU_SDK}/bin:${CUDA_PATH}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RERANK_CHAT_TEMPLATE="${SCRIPT_DIR}/templates/qwen3_reranker.jinja"

gpus=(1)
ports=(7114)
model_name=Qwen3-Reranker-0.6B

for i in ${!gpus[@]}; do
    gpu_count=$(echo ${gpus[$i]} | awk -F',' '{print NF}')
    CUDA_VISIBLE_DEVICES=${gpus[$i]} vllm serve /mnt/data_oss/models/${model_name} \
        --task score \
        --served-model-name ${model_name} \
        --port ${ports[$i]} \
        --gpu-memory-utilization 0.9 \
        --max-model-len 32768 \
        --tensor-parallel-size ${gpu_count} \
        --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}' \
        --chat-template "${RERANK_CHAT_TEMPLATE}" \
        --api-key zjj
done
