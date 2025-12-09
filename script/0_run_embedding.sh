CUDA_VISIBLE_DEVICES=7 vllm serve /data/zjj/models/Qwen/Qwen3-Embedding-8B \
  --task embed \
  --served-model-name qwen3-embedding-8b \
  --port 7107 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 \
  --api-key zjj