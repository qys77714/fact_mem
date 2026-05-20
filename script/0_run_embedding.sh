DEFAULT_EMBEDDING_MODEL=/mnt/data_oss/models/Qwen3-Embedding-8B
MODEL_PATH="${EMBEDDING_MODEL_PATH:-$DEFAULT_EMBEDDING_MODEL}"
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
	echo "error: '${MODEL_PATH}' is not a valid model dir (missing config.json)." >&2
	exit 1
fi

CUDA_VISIBLE_DEVICES=0 vllm serve "${MODEL_PATH}" \
  --task embed \
  --served-model-name qwen3-embedding-8b \
  --port 7110 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 \
  --api-key zjj
