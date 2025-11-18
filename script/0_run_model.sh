# 定义可用的 GPU 和端口
gpus=(7)
ports=(7107)
model_name=Qwen2.5-7B-Instruct

# 循环遍历 GPU 和端口
for i in ${!gpus[@]}; do
    gpu_count=$(echo ${gpus[$i]} | awk -F',' '{print NF}')
    export CUDA_VISIBLE_DEVICES=${gpus[$i]} \
        && vllm serve /mnt/data/models/${model_name} \
        --served-model-name ${model_name} \
        --host 0.0.0.0 \
        --port ${ports[$i]} \
        --max-model-len 32768 \
        --max_num_batched_tokens 32768 \
        --dtype bfloat16 \
        --tensor-parallel-size ${gpu_count} \
        --gpu-memory-utilization 0.9 \
        --api-key zjj &
done

wait