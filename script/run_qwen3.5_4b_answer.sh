#!/bin/bash
# Qwen3.5-4B: answer + evaluate，全部串行
LOG_DIR="/data/zjj/project_26/fact_mem/artifacts/logs/qwen3.5_ingest"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

run_one() {
    local config="$1" name="$2"
    local logfile="${LOG_DIR}/${TS}_answer_${name}.log"
    echo "[$(date '+%H:%M:%S')] Starting: $name"
    uv run --no-sync python run_exp_lme.py --config "$config" --stages generate,evaluate > "$logfile" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then echo "[$(date '+%H:%M:%S')] ✓ $name done"
    else echo "[$(date '+%H:%M:%S')] ✗ $name FAILED (rc=$rc)"; fi
}

METHODS=("rd_addall" "mem0" "evm")
N_VALUES=(0 2 4 6 8)

for N in "${N_VALUES[@]}"; do
    for method in "${METHODS[@]}"; do
        config="config/exp_N${N}_qwen3.5-4b_${method}.yaml"
        name="N${N}_4b_${method}"
        run_one "$config" "$name"
    done
done

echo "[$(date '+%H:%M:%S')] ✅ 4B answer+evaluate all done!"
