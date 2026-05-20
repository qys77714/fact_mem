#!/usr/bin/env bash
# =============================================================================
# run_exp.sh — LME 实验流水线（抽取 → 灌库 → 融合 → 生成 → Judge → HTML）
#
# 用法:
#   ./script/run_exp.sh
#   RUN_EXP_CONFIG=/path/to/custom.yaml ./script/run_exp.sh
#   ./script/run_exp.sh --config script/run_exp.config.yaml [额外参数透传给生成阶段]
#
# 修改实验: 优先改下方「可调参数」区块；并行度 / token / 路径见 run_exp.config.yaml
# =============================================================================
set -euo pipefail
export PYTHONPATH=src

# --- 终端样式（仅在有颜色的 TTY 下启用）---
if [[ -t 1 ]]; then
  _R=$'\033[0m'
  _B=$'\033[1m'
  _D=$'\033[2m'
  _C=$'\033[36m'
  _G=$'\033[32m'
  _Y=$'\033[33m'
  _M=$'\033[35m'
else
  _R= _B= _D= _C= _G= _Y= _M=
fi

_run_title() { echo "${_C}${_B}$*${_R}"; }
_run_sub()   { echo "${_D}  $*${_R}"; }
_run_step()  { echo ""; echo "${_G}${_B}▶ $*${_R}"; }
_run_ok()    { echo "${_D}  ${_R}${_G}✓${_R} ${_D}$*${_R}"; }
_run_warn()  { echo "${_Y}$*${_R}" >&2; }
_run_err()   { echo "${_M}$*${_R}" >&2; }

# =============================================================================
# 仓库根目录
# =============================================================================
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# =============================================================================
# CLI：配置文件 + 透传参数（传给 pipeline_generate）
# =============================================================================
run_exp_config="${RUN_EXP_CONFIG:-}"
eval_forward=()
while (($#)); do
  case "$1" in
    --config=*)
      run_exp_config="${1#*=}"
      shift
      ;;
    --config)
      if (($# < 2)); then
        _run_err "错误：--config 需要路径参数"
        exit 1
      fi
      run_exp_config="$2"
      shift 2
      ;;
    *)
      eval_forward+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$run_exp_config" ]]; then
  run_exp_config="$repo_root/script/run_exp.config.yaml"
fi
if [[ ! -f "$run_exp_config" ]]; then
  _run_err "错误：找不到配置文件 ${run_exp_config}"
  exit 1
fi

# =============================================================================
# 可调参数（常用改这里；路径/并发/token 在 YAML）
# =============================================================================
# --- Benchmark ---
benchmark=lme_s
question_types=""
apply_language="en"

# --- 模型 ---
extract_model="Qwen3-32B"
manager_model="Qwen3-32B"
answer_model="Qwen3-32B"
judge_model="Qwen3-32B"
embedding_model="qwen3-embedding-8b"

# --- 版本标签 ---
candidate_suffix="0406"
exp_suffix="0416"

# --- 1) 候选抽取 --2-
memory_granularity=4

# --- 2) 候选灌库（检索规模）---
relation_related_top_k=3
mem0_related_top_k=3
mem0_related_aggregate_max=10

# --- 3) 关系包融合（默认与 manager 一致；可单独覆盖）---
fusion_model="${manager_model}"

# --- 4) 生成 / 答题（混合检索）---
answer_retrieve_topk=50
answer_hybrid_dense_weight=0.8
answer_hybrid_bm25_weight=0.2
answer_hybrid_pool_mult=4

# --- 5) Judge（Qwen3 Judge 服务端思考链）---
judge_pipeline_opts=()
if [[ "$judge_model" == "Qwen3-32B" ]]; then
  judge_pipeline_opts+=(--judge-qwen-thinking)
fi

# =============================================================================
# 派生变量 + 加载 YAML（输出 dir_* / tok_* / 并发等到当前 shell）
# =============================================================================
model_tag="$(printf '%s' "$extract_model" | tr '/ :\\' '____' | tr -cd '[:alnum:]_.-')"
if [[ -z "$model_tag" ]]; then
  _run_err "错误：model_tag 为空，请检查 extract_model=${extract_model}"
  exit 1
fi

export benchmark extract_model candidate_suffix manager_model exp_suffix

_run_title "加载配置"
_run_sub "文件: ${run_exp_config}"
# shellcheck disable=SC1090
eval "$(python -u script/run_exp_load_config.py "$run_exp_config")"
_run_ok "路径与并行参数已注入环境"

# YAML: debug.evaluate_llm_judge_print_one_sample 在 eval 之后才生效
if [[ "${evaluate_llm_judge_print_one_sample:-0}" == "1" ]]; then
  judge_pipeline_opts+=(--print-one-sample)
fi

# =============================================================================
# 流水线
# =============================================================================
extract_py_args=(
  --benchmark "$benchmark"
  --output "$dir_memdb_extract_candidates"
  --model "$extract_model"
  --suffix "$candidate_suffix"
  --question-types "$question_types"
  --memory-granularity "$memory_granularity"
  --max-new-tokens "$tok_extract_candidates_max_new_tokens"
  --chunk-concurrency "$extract_candidates_chunk_concurrency"
)

_run_step "候选记忆抽取 → ${dir_memdb_extract_candidates}"
python -u src/pipeline/extract_candidates.py "${extract_py_args[@]}"

ingest_shared=(
  --benchmark "$benchmark"
  --question-types "$question_types"
  --candidate-extract-model "$model_tag"
  --candidate-suffix "$candidate_suffix"
  --candidates-dir "${dir_memdb_extract_candidates}"
  --manager-model "$manager_model"
  --embedding-model "${embedding_model}"
  --language "${apply_language}"
  --relation-concurrency "${ingest_candidates_relation_concurrency}"
  --relation-max-new-tokens "${tok_ingest_candidates_relation_max_new_tokens}"
  --manager-max-new-tokens "${tok_ingest_candidates_manager_max_new_tokens}"
)

fuse_shared=(
  --manager-model "$fusion_model"
  --embedding-model "${embedding_model}"
  --language "${apply_language}"
  --fuse-max-new-tokens "$tok_fuse_relation_decision_fused_max_new_tokens"
  --episode-concurrency "$fuse_relation_decision_fused_episode_concurrency"
  --package-concurrency "$fuse_relation_decision_fused_package_concurrency"
)

_run_step "灌库 relation_decision → ${dir_memdb_ingest_relation_decision}"
python -u src/pipeline/ingest_candidates.py \
  --update-method relation_decision \
  --trust-apply-marker \
  --database-root "${dir_memdb_ingest_relation_decision}" \
  --trace-log-dir "${dir_logs_memory_trace_relation_decision}" \
  "${ingest_shared[@]}" \
  --related-top-k "${relation_related_top_k}" \
  --relation-episode-concurrency "${ingest_relation_decision_episode_concurrency}"

# 从 relation_decision 只读拷贝各 episode 到 relation_decision_fused 再融合；已融合 episode 自动跳过
_run_step "关系包融合: ${dir_memdb_ingest_relation_decision} → ${dir_memdb_ingest_relation_decision_fused}"
python -u src/pipeline/fuse_lme_memory_bundles.py \
  --database-root "${dir_memdb_ingest_relation_decision}" \
  --fused-output-root "${dir_memdb_ingest_relation_decision_fused}" \
  "${fuse_shared[@]}"

_run_step "对照页: relation_decision → 融合 → viewer/${viewer_run_tag}/relation_decision_to_fusion.html"
python -u viewer/build_lme_fusion_bundle_map_html.py --run-tag "${viewer_run_tag}"

# echo "=== 灌库 mem0 -> ${dir_memdb_ingest_mem0} ==="
# python -u src/pipeline/ingest_candidates.py \
#   --update-method mem0 \
#   --trust-apply-marker \
#   --database-root "${dir_memdb_ingest_mem0}" \
#   --trace-log-dir "${dir_logs_memory_trace_mem0}" \
#   "${ingest_shared[@]}" \
#   --mem0-related-top-k "${mem0_related_top_k}" \
#   --mem0-related-aggregate-max "${mem0_related_aggregate_max}" \
#   --mem0-episode-concurrency "${ingest_mem0_episode_concurrency}"

_run_step "灌库 add_all → ${dir_memdb_ingest_add_all}"
python -u src/pipeline/ingest_candidates.py \
  --update-method add_all \
  --trust-apply-marker \
  --database-root "${dir_memdb_ingest_add_all}" \
  --trace-log-dir "${dir_logs_memory_trace_add_all}" \
  "${ingest_shared[@]}" \
  --add-all-episode-concurrency "${ingest_add_all_episode_concurrency}"

mkdir -p "${dir_experiment_run_root}"

common_py=(
  src/pipeline_generate.py
  --benchmark "$benchmark"
  --method lme_prebuilt
  --prebuilt-memory
  --answer_model "$answer_model"
  --embedding_model "$embedding_model"
  --question-types "$question_types"
  --parallel_episodes "$generate_lme_prebuilt_parallel_episodes"
  --answer-concurrency "$generate_lme_prebuilt_answer_concurrency"
  --retrieve_topk "$answer_retrieve_topk"
  --memory_token_limit "$tok_generate_lme_prebuilt_memory_token_limit"
  --hybrid-bm25-dense
  --hybrid-dense-weight "$answer_hybrid_dense_weight"
  --hybrid-bm25-weight "$answer_hybrid_bm25_weight"
  --hybrid-pool-mult "$answer_hybrid_pool_mult"
)

_run_step "生成预测: relation_decision_fused → ${file_experiment_pred_relation_decision_jsonl}"
_run_sub "agent trace: ${dir_logs_agent_trace_relation_decision}"
python -u "${common_py[@]}" \
  --agent_trace_dir "${dir_logs_agent_trace_relation_decision}" \
  --database_root "${dir_memdb_ingest_relation_decision_fused}" \
  --hybrid-full-corpus-pool \
  --output "${file_experiment_pred_relation_decision_jsonl}" \
  "${eval_forward[@]}"

# echo "=== [答题/生成预测] mem0 库 -> ${file_experiment_pred_mem0_jsonl}（agent trace: ${dir_logs_agent_trace_mem0}）==="
# python -u "${common_py[@]}" \
#   --agent_trace_dir "${dir_logs_agent_trace_mem0}" \
#   --database_root "${dir_memdb_ingest_mem0}" \
#   --output "${file_experiment_pred_mem0_jsonl}" \
#   "${eval_forward[@]}"

_run_step "生成预测: add_all → ${file_experiment_pred_add_all_jsonl}"
_run_sub "agent trace: ${dir_logs_agent_trace_add_all}"
python -u "${common_py[@]}" \
  --agent_trace_dir "${dir_logs_agent_trace_add_all}" \
  --database_root "${dir_memdb_ingest_add_all}" \
  --output "${file_experiment_pred_add_all_jsonl}" \
  "${eval_forward[@]}"

_run_step "LLM Judge（多文件并行，写回预测 JSONL）"
judge_inputs=()
for pred_file in "${file_experiment_pred_relation_decision_jsonl}" "${file_experiment_pred_add_all_jsonl}"; do
  if [[ -f "$pred_file" ]]; then
    judge_inputs+=("$pred_file")
  fi
done
if ((${#judge_inputs[@]} > 0)); then
  python -u src/pipeline_evaluate.py \
    --input "${judge_inputs[@]}" \
    --judge_model "$judge_model" \
    --benchmark "$benchmark" \
    "${judge_pipeline_opts[@]}" \
    --max_new_tokens "$tok_evaluate_llm_judge_max_new_tokens" \
    --max_concurrency "$evaluate_llm_judge_max_concurrency" \
    --write_back
  _run_ok "Judge 完成（已 --write_back）"
else
  _run_warn "Judge 跳过：未找到可用的预测 JSONL"
fi

_run_step "HTML: add_all vs relation_decision → viewer/${viewer_run_tag}/add_all_vs_relation_decision.html"
python -u viewer/build_lme_case_study_html.py --run-tag "${viewer_run_tag}" --experiment-dir "${dir_experiment_run_root}"

_run_step "HTML: 多方法答题对照 → viewer/${viewer_run_tag}/methods_answer_compare.html"
python -u viewer/build_lme_methods_answer_compare_html.py --run-tag "${viewer_run_tag}" --experiment-dir "${dir_experiment_run_root}"

# =============================================================================
# 结束摘要
# =============================================================================
echo ""
_run_title "完成"
_run_sub "benchmark=${benchmark}  |  答题模型=${answer_model}  |  Judge=${judge_model}"
_run_sub "预测与评分目录: ${dir_experiment_run_root}/"
echo ""
_run_sub "viewer/${viewer_run_tag}/relation_decision_to_fusion.html"
_run_sub "viewer/${viewer_run_tag}/methods_answer_compare.html"
_run_sub "viewer/${viewer_run_tag}/add_all_vs_relation_decision.html"
