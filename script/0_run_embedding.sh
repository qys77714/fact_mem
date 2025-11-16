CUDA_VISIBLE_DEVICES=7 vllm serve /data/zjj/models/Qwen/Qwen3-Embedding-0.6B \
  --task embed \
  --served-model-name qwen3-embedding-0.6b \
  --port 7104 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 32768 \
  --api-key zjj