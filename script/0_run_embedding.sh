CUDA_VISIBLE_DEVICES=4 vllm serve /mnt/data/models/Qwen3-Embedding-8B \
  --task embed \
  --served-model-name qwen3-embedding-8b \
  --port 7104 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 \
  --api-key zjj