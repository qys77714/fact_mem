#!/bin/bash
# ============================================================
# MEME Benchmark 一键实验脚本
# ============================================================
# 前提：
#   1. 已安装依赖（uv sync）
#   2. 已配置 .env（模型端口、API key）
#   3. 模型服务已启动（embedding + manager + answer）
#
# 用法：
#   bash script/run_meme_experiments.sh                    # 运行全部 MEME 实验
#   bash script/run_meme_experiments.sh --dry-run          # 只打印不执行
#   bash script/run_meme_experiments.sh --help             # 查看帮助
# ============================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ---- 默认参数 ----
STAGES="${STAGES:-ingest,generate,evaluate}"
DRY_RUN=false

# ---- 解析命令行参数 ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stages)
            STAGES="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        --help|-h)
            echo "用法: bash script/run_meme_experiments.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --stages LIST    阶段列表（默认：ingest,generate,evaluate）"
            echo "  --dry-run        只打印不执行"
            echo ""
            echo "示例:"
            echo "  bash script/run_meme_experiments.sh"
            echo "  bash script/run_meme_experiments.sh --stages ingest"
            echo "  bash script/run_meme_experiments.sh --dry-run"
            exit 0
            ;;
        *)
            echo "未知参数: $1"; exit 1 ;;
    esac
done

# ---- 数据下载 ----
echo "============================================================"
echo "  Step 1/2: 检查数据"
echo "============================================================"

if [ ! -f "data/raw_data/MEME/meme_filler32k.json" ]; then
    echo "[下载] MEME 数据集..."
    if [ ! -f "easy-mem-data.zip" ]; then
        wget https://huggingface.co/datasets/Qys77/easy-mem-data/resolve/main/easy-mem-data.zip
    fi
    unzip -n easy-mem-data.zip -d .
    echo "[OK] 数据集就绪"
else
    echo "[OK] 数据集已存在"
fi

if [ ! -d "artifacts/stages/candidates/ff157d29" ]; then
    echo "[下载] MEME 候选记忆..."
    if [ ! -f "easy-mem-candidates-meme.zip" ]; then
        wget https://huggingface.co/datasets/Qys77/easy-mem-data/resolve/main/easy-mem-candidates-meme.zip
    fi
    unzip -o easy-mem-candidates-meme.zip -d .
    echo "[OK] 候选记忆就绪"
else
    echo "[OK] 候选记忆已存在"
fi

# ---- MEME 配置列表 ----
CONFIGS=(
    "config/meme_default.yaml"
    "config/meme_e4b.yaml"
    "config/meme_q35.yaml"
    "config/meme_qwen35_9b.yaml"
    "config/meme_gemma4-12b.yaml"
)

# ---- 运行实验 ----
echo ""
echo "============================================================"
echo "  Step 2/2: 运行 MEME 实验 (stages: $STAGES)"
echo "============================================================"

for config in "${CONFIGS[@]}"; do
    echo ""
    echo "------------------------------------------------------------"
    echo "  Config: $config"
    echo "------------------------------------------------------------"

    if $DRY_RUN; then
        echo "  [DRY-RUN] uv run --no-sync python run_exp_lme.py --config $config --stages $STAGES"
    else
        uv run --no-sync python run_exp_lme.py --config "$config" --stages "$STAGES"
        echo "  [完成] $config"
    fi
done

echo ""
echo "============================================================"
echo "  全部完成！共运行 ${#CONFIGS[@]} 个配置"
echo "============================================================"
