MAX_NUM_BATCHED_TOKENS=$((4096 * 32))

CUDA_VISIBLE_DEVICES=0 vllm serve /data/zjj/models/Qwen/Qwen3-Embedding-0.6B \
  --runner pooling \
  --served-model-name qwen3-embedding-0.6b \
  --port 7110 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 \
  --max-num-seqs 1024 \
  --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
  --api-key zjj