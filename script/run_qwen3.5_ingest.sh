#!/bin/bash
# ================================================================
# Qwen3.5-4B + Qwen3.5-9B ingest 脚本
# 方法串行: RD → Mem0 → EverMemOS
# RD 内部分批: N=0,2 → N=4,6 → N=8
# Mem0 / EverMemOS: 全部 N 并行
# ================================================================
set -e

LOG_DIR="artifacts/logs/qwen3.5_ingest"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

run_ingest() {
    local config="$1"
    local name="$2"
    local logfile="${LOG_DIR}/${TIMESTAMP}_${name}.log"
    echo "[$(date '+%H:%M:%S')] Starting: $name"
    uv run --no-sync python run_exp_lme.py --config "$config" --stages ingest > "$logfile" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] ✓ $name done"
    else
        echo "[$(date '+%H:%M:%S')] ✗ $name FAILED (rc=$rc) — see $logfile"
    fi
    return $rc
}

# ================================================================
# Phase 1: RD (relation_decision + add_all)
# ================================================================
echo "========================================="
echo "[$(date '+%H:%M:%S')] Phase 1: RD ingest — Batch 1 (N=0,2)"
echo "========================================="
run_ingest "config/exp_N0_qwen3.5-4b_rd_addall.yaml" "rd_N0_4b" &
run_ingest "config/exp_N2_qwen3.5-4b_rd_addall.yaml" "rd_N2_4b" &
run_ingest "config/exp_N0_qwen3.5-9b_rd_addall.yaml" "rd_N0_9b" &
run_ingest "config/exp_N2_qwen3.5-9b_rd_addall.yaml" "rd_N2_9b" &
wait

echo "========================================="
echo "[$(date '+%H:%M:%S')] Phase 1: RD ingest — Batch 2 (N=4,6)"
echo "========================================="
run_ingest "config/exp_N4_qwen3.5-4b_rd_addall.yaml" "rd_N4_4b" &
run_ingest "config/exp_N6_qwen3.5-4b_rd_addall.yaml" "rd_N6_4b" &
run_ingest "config/exp_N4_qwen3.5-9b_rd_addall.yaml" "rd_N4_9b" &
run_ingest "config/exp_N6_qwen3.5-9b_rd_addall.yaml" "rd_N6_9b" &
wait

echo "========================================="
echo "[$(date '+%H:%M:%S')] Phase 1: RD ingest — Batch 3 (N=8)"
echo "========================================="
run_ingest "config/exp_N8_qwen3.5-4b_rd_addall.yaml" "rd_N8_4b" &
run_ingest "config/exp_N8_qwen3.5-9b_rd_addall.yaml" "rd_N8_9b" &
wait

# ================================================================
# Phase 2: Mem0
# ================================================================
echo "========================================="
echo "[$(date '+%H:%M:%S')] Phase 2: Mem0 ingest (all N in parallel)"
echo "========================================="
for N in 0 2 4 6 8; do
    run_ingest "config/exp_N${N}_qwen3.5-4b_mem0.yaml" "mem0_N${N}_4b" &
    run_ingest "config/exp_N${N}_qwen3.5-9b_mem0.yaml" "mem0_N${N}_9b" &
done
wait

# ================================================================
# Phase 3: EverMemOS
# ================================================================
echo "========================================="
echo "[$(date '+%H:%M:%S')] Phase 3: EverMemOS ingest (all N in parallel)"
echo "========================================="
for N in 0 2 4 6 8; do
    run_ingest "config/exp_N${N}_qwen3.5-4b_evm.yaml" "evm_N${N}_4b" &
    run_ingest "config/exp_N${N}_qwen3.5-9b_evm.yaml" "evm_N${N}_9b" &
done
wait

echo "========================================="
echo "[$(date '+%H:%M:%S')] All ingest complete!"
echo "========================================="
