#!/bin/bash
# ================================================================
# 一键启动：gemma4-12b-it × LME Hybrid 全实验
#
# 前置条件：
#   1. .env 已配置（DEEPSEEK_API_KEY 等）
#   2. 模型已挂载并启动：
#      - gemma4-12b-it（ingest 用，.env 中 PORT_GEMMA4_12B）
#      - gemma4-26B-A4B（answer 用，.env 中 PORT_GEMMA4_26B）
#      - qwen3-embedding-0.6b（embedding，.env 中 EMBEDDING_BASE_URL）
#   3. 数据已解压到项目根目录
#   4. deepseek-v4-flash API key 已配（DEEPSEEK_API_KEY）
#
# 实验流程：
#   RD ingest（N=0,2,4,6,8 串行）
#   → Mem0 ingest（N=0,2,4,6,8 并行）
#   → EverMemOS ingest（N=0,2,4,6,8 并行）
#   → Answer + Judge（全部 15 组）
# ================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_ROOT/config/collab"

cd "$PROJECT_ROOT"

echo "========================================"
echo "  协作者一键实验脚本"
echo "  模型: gemma4-12b (ingest) + gemma4-26B (answer) + deepseek-v4-flash (judge)"
echo "  基准: LME Hybrid (N=0,2,4,6,8)"
echo "  方法: RD / Mem0 / EverMemOS"
echo "========================================"

# ---- Step 1: 生成配置文件 ----
echo ""
echo "=== [1/5] 生成配置文件 ==="
PYTHONPATH=src uv run --no-sync python script/generate_collab_configs.py

# ---- Step 2: RD ingest（串行：N 级别逐个跑，共用 gemma4-12b）----
echo ""
echo "=== [2/5] RD ingest（串行 N=0,2,4,6,8）==="
for N in 0 2 4 6 8; do
    echo "--- RD N=$N ---"
    uv run --no-sync python run_exp_lme.py \
        --config "$CONFIG_DIR/exp_N${N}_rd.yaml" \
        --stages extract,ingest
    echo "--- RD N=$N 完成 ---"
done

# ---- Step 3: Mem0 ingest（并行 N=0,2,4,6,8，共用 gemma4-12b）----
echo ""
echo "=== [3/5] Mem0 ingest（并行 N=0,2,4,6,8）==="
PIDS_MEM0=()
for N in 0 2 4 6 8; do
    uv run --no-sync python run_exp_lme.py \
        --config "$CONFIG_DIR/exp_N${N}_mem0.yaml" \
        --stages extract,ingest \
        > "$PROJECT_ROOT/logs/mem0_N${N}.log" 2>&1 &
    PIDS_MEM0+=($!)
    echo "  Mem0 N=$N → PID $!"
done
echo "  等待 Mem0 全部完成（PIDs: ${PIDS_MEM0[*]}）..."
for pid in "${PIDS_MEM0[@]}"; do
    wait "$pid" && echo "  PID $pid 完成" || echo "  PID $pid 失败！"
done

# ---- Step 4: EverMemOS ingest（并行 N=0,2,4,6,8，共用 gemma4-12b）----
echo ""
echo "=== [4/5] EverMemOS ingest（并行 N=0,2,4,6,8）==="
PIDS_EVM=()
for N in 0 2 4 6 8; do
    uv run --no-sync python run_exp_lme.py \
        --config "$CONFIG_DIR/exp_N${N}_evm.yaml" \
        --stages extract,ingest \
        > "$PROJECT_ROOT/logs/evm_N${N}.log" 2>&1 &
    PIDS_EVM+=($!)
    echo "  EverMemOS N=$N → PID $!"
done
echo "  等待 EverMemOS 全部完成（PIDs: ${PIDS_EVM[*]}）..."
for pid in "${PIDS_EVM[@]}"; do
    wait "$pid" && echo "  PID $pid 完成" || echo "  PID $pid 失败！"
done

# ---- Step 5: Answer + Judge（全部并行）----
echo ""
echo "=== [5/5] Answer + Judge（15 组并行）==="
PIDS_AJ=()
for method in rd mem0 evm; do
    for N in 0 2 4 6 8; do
        uv run --no-sync python run_exp_lme.py \
            --config "$CONFIG_DIR/exp_N${N}_${method}.yaml" \
            --stages generate,evaluate \
            > "$PROJECT_ROOT/logs/answer_${method}_N${N}.log" 2>&1 &
        PIDS_AJ+=($!)
        echo "  Answer+Judge ${method} N=$N → PID $!"
    done
done
echo "  等待 Answer+Judge 全部完成（${#PIDS_AJ[@]} 个任务）..."
for pid in "${PIDS_AJ[@]}"; do
    wait "$pid" && echo "  PID $pid 完成" || echo "  PID $pid 失败！"
done

echo ""
echo "========================================"
echo "  全部实验完成！结果在 artifacts/runs/ 下"
echo "========================================"
