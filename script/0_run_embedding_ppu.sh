# Some vendor libraries dlopen("libcuda.so") directly, so expose CUDA libs explicitly.
export CUDA_PATH=/usr/local/cuda
export PPU_SDK=/usr/local/PPU_SDK
export PATH=${PPU_SDK}/bin:${CUDA_PATH}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

gpus=(0)
ports=(7110)

for i in ${!gpus[@]}; do
    gpu_count=$(echo ${gpus[$i]} | awk -F',' '{print NF}')
    CUDA_VISIBLE_DEVICES=${gpus[$i]} vllm serve /mnt/data_oss/models/Qwen3-Embedding-8B \
        --task embed \
        --served-model-name qwen3-embedding-8b \
        --port ${ports[$i]} \
        --gpu-memory-utilization 0.9 \
        --max-model-len 32768 \
        --tensor-parallel-size ${gpu_count} \
        --api-key zjj
done