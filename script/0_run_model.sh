# 定义可用的 GPU 和端口
gpus=(1,2,3,4,5,6)
ports=(7111)
model_name=Qwen3.5-27B

# 循环遍历 GPU 和端口
for i in ${!gpus[@]}; do
    gpu_count=$(echo ${gpus[$i]} | awk -F',' '{print NF}')
    export CUDA_VISIBLE_DEVICES=${gpus[$i]} \
        && vllm serve /data/zjj/models/Qwen/${model_name} \
        --served-model-name ${model_name} \
        --host 0.0.0.0 \
        --port ${ports[$i]} \
        --max-model-len 32768 \
        --max-num-batched-tokens $((4096 * 40)) \
        --dtype bfloat16 \
        --tensor-parallel-size ${gpu_count} \
        --gpu-memory-utilization 0.9 \
        --api-key zjj &
done

wait