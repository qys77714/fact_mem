#!/bin/bash
# 直接对已有 pred.jsonl 跑 judge (deepseek-v4-flash)
# Qwen3.5-4B tl256+512+1024 + gemma4-e4b tl1024
LOG_DIR="/data/zjj/project_26/fact_mem/artifacts/logs/judge_dsv4"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

judge_one() {
    local pred="$1" name="$2"
    local dir=$(dirname "$pred")
    local run_dir=$(echo "$dir" | grep -oP '.*/answer/[^/]+/[^/]+')
    local judge_dir="${dir/answer/judge}"
    mkdir -p "$judge_dir"
    local judged="${judge_dir}/judged.jsonl"
    local metrics="${judge_dir}/metrics.json"
    local logfile="${LOG_DIR}/${TS}_${name}.log"

    echo "[$(date '+%H:%M:%S')] $name"
    uv run --no-sync python src/pipeline_lme_evaluate.py \
        --input "$pred" \
        --output "$judged" \
        --metrics-output "$metrics" \
        --judge_model deepseek-v4-flash \
        --benchmark lme_s \
        --max_concurrency 20 \
        --max_new_tokens 512 \
        --use_cot \
        --judge-template pipeline_judge.jinja \
        > "$logfile" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then echo "[$(date '+%H:%M:%S')]   ✓ $name"
    else echo "[$(date '+%H:%M:%S')]   ✗ $name FAILED (rc=$rc)"; fi
}

echo "========================================="
echo "[$(date '+%H:%M:%S')] Phase 1: Qwen3.5-4B tl256+512+1024"
echo "========================================="

for N in 0 2 4 6 8; do
    for method in add_all relation_decision mem0 evermemos; do
        for tl in 256 512 1024; do
            # 找 pred.jsonl
            pred=$(find artifacts/runs -path "*filler_n${N}_qwen3-5-4b*tl${tl}*/answer/${method}/*/pred.jsonl" 2>/dev/null | head -1)
            if [ -n "$pred" ]; then
                lines=$(wc -l < "$pred")
                if [ "$lines" -ge 500 ]; then
                    # 检查是否已有 deepseek-v4-flash 的 metrics
                    q35_judge_dir="$(dirname "$pred")"
                    q35_judge_dir="${q35_judge_dir/answer/judge}"
                    if [ -f "${q35_judge_dir}/metrics.json" ]; then
                        jm=$(python3 -c "import json; print(json.load(open('${q35_judge_dir}/metrics.json')).get('judge_model',''))" 2>/dev/null)
                        if [ "$jm" = "deepseek-v4-flash" ]; then
                            echo "[$(date '+%H:%M:%S')]   skip N${N}_${method}_tl${tl} (已有 dsv4)"
                            continue
                        fi
                    fi
                    judge_one "$pred" "q35_N${N}_${method}_tl${tl}"
                fi
            fi
        done
    done
done

echo "========================================="
echo "[$(date '+%H:%M:%S')] Phase 2: gemma4-e4b tl1024"
echo "========================================="

for N in 0 2 4 6 8; do
    for method in add_all relation_decision mem0 evermemos; do
        pred=$(find artifacts/runs -path "*filler_n${N}_gemma4-e4b*tl1024*/answer/${method}/*/pred.jsonl" 2>/dev/null | head -1)
        if [ -n "$pred" ]; then
            lines=$(wc -l < "$pred")
            if [ "$lines" -ge 500 ]; then
                g4_judge_dir="$(dirname "$pred")"
                g4_judge_dir="${g4_judge_dir/answer/judge}"
                if [ -f "${g4_judge_dir}/metrics.json" ]; then
                    jm=$(python3 -c "import json; print(json.load(open('${g4_judge_dir}/metrics.json')).get('judge_model',''))" 2>/dev/null)
                    if [ "$jm" = "deepseek-v4-flash" ]; then
                        echo "[$(date '+%H:%M:%S')]   skip N${N}_${method}_tl1024 (已有 dsv4)"
                        continue
                    fi
                fi
                judge_one "$pred" "g4e4b_N${N}_${method}_tl1024"
            fi
        fi
    done
done

echo "[$(date '+%H:%M:%S')] ✅ All judge done!"
