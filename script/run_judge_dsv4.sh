#!/bin/bash
# Qwen3.5-4B (tl256+512+1024) + gemma4-e4b (tl1024) — judge only
# 全部串行，judge=deepseek-v4-flash
LOG_DIR="/data/zjj/project_26/fact_mem/artifacts/logs/judge_dsv4"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

run_judge() {
    local config="$1" name="$2"
    local logfile="${LOG_DIR}/${TS}_${name}.log"
    echo "[$(date '+%H:%M:%S')] $name"
    uv run --no-sync python run_exp_lme.py --config "$config" --stages evaluate > "$logfile" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then echo "[$(date '+%H:%M:%S')]   ✓ $name"
    else echo "[$(date '+%H:%M:%S')]   ✗ $name FAILED (rc=$rc)"; fi
}

METHODS=("rd_addall" "mem0" "evm")
N_VALUES=(0 2 4 6 8)

echo "========================================="
echo "[$(date '+%H:%M:%S')] Phase 1: Qwen3.5-4B tl256+512"
echo "========================================="
for N in "${N_VALUES[@]}"; do
    for method in "${METHODS[@]}"; do
        run_judge "config/exp_N${N}_qwen3.5-4b_${method}.yaml" "q35_4b_N${N}_${method}_256-512"
    done
done

echo "========================================="
echo "[$(date '+%H:%M:%S')] Phase 2: Qwen3.5-4B tl1024"
echo "========================================="
for N in "${N_VALUES[@]}"; do
    for method in "${METHODS[@]}"; do
        run_judge "config/exp_N${N}_qwen3.5-4b_${method}_tl1024.yaml" "q35_4b_N${N}_${method}_1024"
    done
done

echo "========================================="
echo "[$(date '+%H:%M:%S')] Phase 3: gemma4-e4b tl1024"
echo "========================================="
for N in "${N_VALUES[@]}"; do
    for method in "${METHODS[@]}"; do
        run_judge "config/exp_N${N}_gemma4-e4b_${method}_tl1024.yaml" "g4e4b_N${N}_${method}_1024"
    done
done

echo "[$(date '+%H:%M:%S')] ✅ All judge done!"
