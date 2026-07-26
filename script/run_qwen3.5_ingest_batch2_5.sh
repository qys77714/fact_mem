#!/bin/bash
# ================================================================
# Qwen3.5 ingest 自动调度器 (Batch 2-5)
# 在 Batch 1 (N=0,2) 完成后自动继续
# 用法: bash script/run_qwen3.5_ingest_batch2_5.sh
# ================================================================
set -e

LOG_DIR="/data/zjj/project_26/fact_mem/artifacts/logs/qwen3.5_ingest"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

run_one() {
    local config="$1" name="$2"
    local logfile="${LOG_DIR}/${TIMESTAMP}_${name}.log"
    echo "[$(date '+%H:%M:%S')] Starting: $name"
    uv run --no-sync python run_exp_lme.py --config "$config" --stages ingest > "$logfile" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] ✓ $name done"
    else
        echo "[$(date '+%H:%M:%S')] ✗ $name FAILED (rc=$rc)"
    fi
}

# ================================================================
# 等待 Batch 1 (N=0,2) 完成
# ================================================================
echo "[$(date '+%H:%M:%S')] Waiting for Batch 1 (N=0,2) to finish..."
while true; do
    RUNNING=$(ps aux | grep -c "ingest_candidates.py.*relation_decision" | tr -d ' ')
    if [ "$RUNNING" -le 1 ]; then
        echo "[$(date '+%H:%M:%S')] Batch 1 appears done ($RUNNING ingest_candidates remaining)"
        break
    fi
    echo "[$(date '+%H:%M:%S')] Batch 1 still running ($RUNNING ingest_candidates alive)"
    sleep 60
done
sleep 10  # extra buffer

# ================================================================
# Phase 1: RD — Batch 2 (N=4,6)
# ================================================================
echo "========================================="
echo "[$(date '+%H:%M:%S')] Phase 1: RD ingest — Batch 2 (N=4,6)"
echo "========================================="
run_one "config/exp_N4_qwen3.5-4b_rd_addall.yaml" "rd_N4_4b" &
run_one "config/exp_N6_qwen3.5-4b_rd_addall.yaml" "rd_N6_4b" &
run_one "config/exp_N4_qwen3.5-9b_rd_addall.yaml" "rd_N4_9b" &
run_one "config/exp_N6_qwen3.5-9b_rd_addall.yaml" "rd_N6_9b" &
wait

# ================================================================
# Phase 1: RD — Batch 3 (N=8)
# ================================================================
echo "========================================="
echo "[$(date '+%H:%M:%S')] Phase 1: RD ingest — Batch 3 (N=8)"
echo "========================================="
run_one "config/exp_N8_qwen3.5-4b_rd_addall.yaml" "rd_N8_4b" &
run_one "config/exp_N8_qwen3.5-9b_rd_addall.yaml" "rd_N8_9b" &
wait

# ================================================================
# Phase 2: Mem0 (all N parallel)
# ================================================================
echo "========================================="
echo "[$(date '+%H:%M:%S')] Phase 2: Mem0 ingest (all N parallel)"
echo "========================================="
for N in 0 2 4 6 8; do
    run_one "config/exp_N${N}_qwen3.5-4b_mem0.yaml" "mem0_N${N}_4b" &
    run_one "config/exp_N${N}_qwen3.5-9b_mem0.yaml" "mem0_N${N}_9b" &
done
wait

# ================================================================
# Phase 3: EverMemOS (all N parallel)
# ================================================================
echo "========================================="
echo "[$(date '+%H:%M:%S')] Phase 3: EverMemOS ingest (all N parallel)"
echo "========================================="
for N in 0 2 4 6 8; do
    run_one "config/exp_N${N}_qwen3.5-4b_evm.yaml" "evm_N${N}_4b" &
    run_one "config/exp_N${N}_qwen3.5-9b_evm.yaml" "evm_N${N}_9b" &
done
wait

echo "========================================="
echo "[$(date '+%H:%M:%S')] ✅ All Qwen3.5 ingest complete!"
echo "========================================="
