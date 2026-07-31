#!/bin/bash
# gemma4-e4b + Qwen3.5-4B: tl1024 answer+evaluate，串行
LOG_DIR="/data/zjj/project_26/fact_mem/artifacts/logs/tl1024"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

run_one() {
    local config="$1" name="$2"
    local logfile="${LOG_DIR}/${TS}_${name}.log"
    echo "[$(date '+%H:%M:%S')] $name"
    uv run --no-sync python run_exp_lme.py --config "$config" --stages generate,evaluate > "$logfile" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then echo "[$(date '+%H:%M:%S')]   ✓ $name"
    else echo "[$(date '+%H:%M:%S')]   ✗ $name FAILED (rc=$rc)"; fi
}

MODELS=("gemma4-e4b" "qwen3.5-4b")
METHODS=("rd_addall" "mem0" "evm")
N_VALUES=(0 2 4 6 8)

for model in "${MODELS[@]}"; do
    echo "========================================="
    echo "[$(date '+%H:%M:%S')] Model: $model"
    echo "========================================="
    for N in "${N_VALUES[@]}"; do
        for method in "${METHODS[@]}"; do
            config="config/exp_N${N}_${model}_${method}_tl1024.yaml"
            name="${model}_N${N}_${method}_tl1024"
            run_one "$config" "$name"
        done
    done
done

echo "[$(date '+%H:%M:%S')] ✅ tl1024 all done!"
