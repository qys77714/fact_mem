#!/bin/bash
# ================================================================
# Qwen3.5-4B 独立调度: RD(N4,6→N8) → Mem0 → EverMemOS
# 4B 的 RD N0,2 已完成，当前 N4,6 正在跑（此脚本等待它们完成再继续）
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

# ---- 等当前 RD N4,6 完成（已在外部启动） ----
echo "[$(date '+%H:%M:%S')] 4B: 等待 RD N4,N6 完成..."
while true; do
    running=$(ps aux | grep "ingest_candidates.*Qwen3.5-4B.*relation_decision" | grep -v grep | wc -l)
    if [ "$running" -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] 4B: RD N4,N6 已完成"
        break
    fi
    echo "[$(date '+%H:%M:%S')] 4B: RD N4,N6 仍在跑 ($running 进程)"
    sleep 60
done

# ---- RD Batch 3: N=8 ----
echo "========================================="
echo "[$(date '+%H:%M:%S')] 4B: RD Batch 3 (N=8)"
echo "========================================="
run_one "config/exp_N8_qwen3.5-4b_rd_addall.yaml" "rd_N8_4b"
# 注意: 这里不用 &+wait，N8 单独跑，等它结束再继续

# ---- Mem0 (all N parallel) ----
echo "========================================="
echo "[$(date '+%H:%M:%S')] 4B: Mem0 (all N parallel)"
echo "========================================="
pids=""
for N in 0 2 4 6 8; do
    run_one "config/exp_N${N}_qwen3.5-4b_mem0.yaml" "mem0_N${N}_4b" &
    pids="$pids $!"
done
wait $pids

# ---- EverMemOS (all N parallel) ----
echo "========================================="
echo "[$(date '+%H:%M:%S')] 4B: EverMemOS (all N parallel)"
echo "========================================="
pids=""
for N in 0 2 4 6 8; do
    run_one "config/exp_N${N}_qwen3.5-4b_evm.yaml" "evm_N${N}_4b" &
    pids="$pids $!"
done
wait $pids

echo "[$(date '+%H:%M:%S')] ✅ 4B all done!"
