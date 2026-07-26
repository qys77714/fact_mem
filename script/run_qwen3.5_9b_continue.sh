#!/bin/bash
# ================================================================
# Qwen3.5-9B 独立调度: 等 RD(N0,2) 完成 → RD(N4,6→N8) → Mem0 → EverMemOS
# ================================================================
LOG_DIR="/data/zjj/project_26/fact_mem/artifacts/logs/qwen3.5_ingest"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

run_one() {
    local config="$1" name="$2"
    local logfile="${LOG_DIR}/${TS}_${name}.log"
    echo "[$(date '+%H:%M:%S')] Starting: $name"
    uv run --no-sync python run_exp_lme.py --config "$config" --stages ingest > "$logfile" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then echo "[$(date '+%H:%M:%S')] ✓ $name done"
    else echo "[$(date '+%H:%M:%S')] ✗ $name FAILED (rc=$rc)"; fi
}

# ---- 等当前 RD N0,2 完成（已在外部启动） ----
echo "[$(date '+%H:%M:%S')] 9B: 等待 RD N0,N2 完成..."
while true; do
    running=$(ps aux | grep "ingest_candidates.*Qwen3.5-9B.*relation_decision" | grep -v grep | wc -l)
    if [ "$running" -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] 9B: RD N0,N2 已完成"
        break
    fi
    echo "[$(date '+%H:%M:%S')] 9B: RD N0,N2 仍在跑 ($running 进程)"
    sleep 60
done

# ---- RD Batch 2: N=4,6 ----
echo "========================================="
echo "[$(date '+%H:%M:%S')] 9B: RD Batch 2 (N=4,6)"
echo "========================================="
run_one "config/exp_N4_qwen3.5-9b_rd_addall.yaml" "rd_N4_9b" &
run_one "config/exp_N6_qwen3.5-9b_rd_addall.yaml" "rd_N6_9b" &
wait

# ---- RD Batch 3: N=8 ----
echo "========================================="
echo "[$(date '+%H:%M:%S')] 9B: RD Batch 3 (N=8)"
echo "========================================="
run_one "config/exp_N8_qwen3.5-9b_rd_addall.yaml" "rd_N8_9b"

# ---- Mem0 (all N parallel) ----
echo "========================================="
echo "[$(date '+%H:%M:%S')] 9B: Mem0 (all N parallel)"
echo "========================================="
pids=""
for N in 0 2 4 6 8; do
    run_one "config/exp_N${N}_qwen3.5-9b_mem0.yaml" "mem0_N${N}_9b" &
    pids="$pids $!"
done
wait $pids

# ---- EverMemOS (all N parallel) ----
echo "========================================="
echo "[$(date '+%H:%M:%S')] 9B: EverMemOS (all N parallel)"
echo "========================================="
pids=""
for N in 0 2 4 6 8; do
    run_one "config/exp_N${N}_qwen3.5-9b_evm.yaml" "evm_N${N}_9b" &
    pids="$pids $!"
done
wait $pids

echo "[$(date '+%H:%M:%S')] ✅ 9B all done!"
