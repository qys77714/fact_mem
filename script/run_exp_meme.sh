#!/usr/bin/env bash
# =============================================================================
# run_exp_meme.sh — MEME 4-Phase 对齐实验流水线
#
# 与官方 MEME-public run_episode 协议完全对齐：
#   Phase 1: ingest sessions 0 → before_pos → MemDB/{hn}_before
#   Phase 2: 答 before_questions（relation_decision 用 fused {hn}_before）
#   Phase 3: copy {hn}_before → {hn}_after；增量 ingest sessions before_pos+1 → after_pos
#   Phase 4: 答 after_questions（relation_decision 用 fused {hn}_after）
#
# 与 run_exp.sh 的区别：
#   - 抽取候选步骤相同（MEME evidence session 直接用 gold_facts，filler 走 LLM）
#   - 灌库 + 答题合并为 pipeline_meme_4phase.py（每方法一次调用）
#   - Judge 使用 pipeline_meme_evaluate.py（task-specific prompts + trivial-pass）
#   - 实验输出目录带 _meme4p 后缀
#
# 用法:
#   ./script/run_exp_meme.sh
#   RUN_EXP_CONFIG=/path/to/custom.yaml ./script/run_exp_meme.sh
# =============================================================================
set -euo pipefail
export PYTHONPATH=src

{

if [[ -t 1 ]]; then
  _R=$'\033[0m'; _B=$'\033[1m'; _D=$'\033[2m'; _C=$'\033[36m'; _G=$'\033[32m'; _Y=$'\033[33m'; _M=$'\033[35m'
else
  _R= _B= _D= _C= _G= _Y= _M=
fi

_run_title() { echo "${_C}${_B}$*${_R}"; }
_run_sub()   { echo "${_D}  $*${_R}"; }
_run_step()  { echo ""; echo "${_G}${_B}▶ $*${_R}"; }
_run_ok()    { echo "${_D}  ${_R}${_G}✓${_R} ${_D}$*${_R}"; }
_run_warn()  { echo "${_Y}$*${_R}" >&2; }
_run_err()   { echo "${_M}$*${_R}" >&2; }

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_exp_config="${RUN_EXP_CONFIG:-}"
eval_forward=()
while (($#)); do
  case "$1" in
    --config=*) run_exp_config="${1#*=}"; shift ;;
    --config)
      if (($# < 2)); then _run_err "错误：--config 需要路径参数"; exit 1; fi
      run_exp_config="$2"; shift 2 ;;
    *) eval_forward+=("$1"); shift ;;
  esac
done

if [[ -z "$run_exp_config" ]]; then
  run_exp_config="$repo_root/script/run_exp_meme.config.yaml"
fi
if [[ ! -f "$run_exp_config" ]]; then
  _run_err "错误：找不到配置文件 ${run_exp_config}"
  exit 1
fi

# =============================================================================
# 可调参数
# =============================================================================
benchmark=meme_filler32k
question_types=""
apply_language="en"

extract_model="gemma4-26B"
manager_model="gemma4-26B"
answer_model="gemma4-26B"
judge_model="gpt-4o-mini"
embedding_model="qwen3-embedding-8b"

candidate_suffix="0519_as3"
exp_suffix="0521_v2"

memory_granularity=4
extract_turn_overlap=0

related_top_k=3
mem0_related_top_k=3
mem0_related_aggregate_max=10

amac_threshold=0.3

model_tag="$(printf '%s' "$extract_model" | tr '/ :\\' '____' | tr -cd '[:alnum:]_.-')"
export benchmark extract_model candidate_suffix manager_model answer_model exp_suffix

_run_title "加载 MEME 4-Phase 实验配置"
_run_sub "文件: ${run_exp_config}"
cfg_exports=""
if ! cfg_exports="$(python -u script/run_exp_load_config.py "$run_exp_config")"; then
  _run_err "错误：配置加载失败"
  exit 1
fi
# shellcheck disable=SC1090
eval "$cfg_exports"
_run_ok "路径、并行、prompts 已注入环境"

# =============================================================================
# 候选抽取（与 run_exp.sh 相同）
# =============================================================================
extract_py_args=(
  --benchmark "$benchmark"
  --output "$dir_memdb_extract_candidates"
  --model "$extract_model"
  --suffix "$candidate_suffix"
  --question-types "$question_types"
  --memory-granularity "$memory_granularity"
  --turn-overlap "$extract_turn_overlap"
  --language "${apply_language}"
  --max-new-tokens "$tok_extract_candidates_max_new_tokens"
  --chunk-concurrency "$extract_candidates_chunk_concurrency"
)
if [[ -n "${mem_extract_template:-}" ]]; then
  extract_py_args+=(--mem-extract-template "${mem_extract_template}")
fi
if [[ "${mem_extract_aspects_only:-0}" == "1" || "${mem_extract_aspects_only,,}" == "true" ]]; then
  extract_py_args+=(--mem-extract-aspects-only)
fi
for i in 1 2 3; do
  v="mem_extract_aspect_template_${i}"
  if [[ -n "${!v:-}" ]]; then
    extract_py_args+=(--mem-extract-extra-template "${!v}")
  fi
done

_run_step "候选记忆抽取 → ${dir_memdb_extract_candidates}"
python -u src/pipeline/extract_candidates.py "${extract_py_args[@]}"

# =============================================================================
# 公共 4-Phase 参数
# =============================================================================
mkdir -p "${dir_experiment_run_root}"

# 公共参数（ingest + fuse + answer，不含 update-method / database-root / output）
common_4phase=(
  --benchmark "$benchmark"
  --candidates-dir "$dir_memdb_extract_candidates"
  --answer-model "$answer_model"
  --embedding-model "$embedding_model"
  --manager-model "$manager_model"
  --language "${apply_language}"
  --retrieve-topk 50
  --memory-token-limit "${tok_memory_token_limit}"
  --answer-concurrency "${meme_4phase_answer_concurrency}"
  --hybrid-bm25-dense
  --hybrid-dense-weight 0.8
  --hybrid-bm25-weight 0.2
  --hybrid-pool-mult 4
  --relation-concurrency 50
  --relation-max-new-tokens "${tok_ingest_candidates_relation_max_new_tokens}"
  --manager-max-new-tokens "${tok_ingest_candidates_manager_max_new_tokens}"
  --related-top-k "${related_top_k}"
  --mem0-related-top-k "${mem0_related_top_k}"
  --mem0-related-aggregate-max "${mem0_related_aggregate_max}"
  --fuse-max-new-tokens "${tok_fuse_max_new_tokens}"
  --fusion-package-concurrency "${fuse_package_concurrency}"
  --no-memory-time
)

# relation_decision prompt overrides
if [[ -n "${relation_classification_system_en:-}" ]]; then
  common_4phase+=(--relation-system-template-en "${relation_classification_system_en}")
fi
if [[ -n "${relation_classification_system_zh:-}" ]]; then
  common_4phase+=(--relation-system-template-zh "${relation_classification_system_zh}")
fi
if [[ -n "${relation_classification_user:-}" ]]; then
  common_4phase+=(--relation-user-template "${relation_classification_user}")
fi
if [[ -n "${fusion_bundle_template_en:-}" ]]; then
  common_4phase+=(--fusion-bundle-template-en "${fusion_bundle_template_en}")
fi
if [[ -n "${fusion_bundle_template_zh:-}" ]]; then
  common_4phase+=(--fusion-bundle-template-zh "${fusion_bundle_template_zh}")
fi
if [[ -n "${fusion_edge_labels_template_en:-}" ]]; then
  common_4phase+=(--fusion-edge-labels-template-en "${fusion_edge_labels_template_en}")
fi
if [[ -n "${fusion_edge_labels_template_zh:-}" ]]; then
  common_4phase+=(--fusion-edge-labels-template-zh "${fusion_edge_labels_template_zh}")
fi

# =============================================================================
# 4-Phase 灌库 + 答题（每方法一次调用 pipeline_meme_4phase.py）
# =============================================================================

# _run_step "4-Phase relation_decision → ${file_experiment_pred_relation_decision_jsonl}"
# _run_sub "DB: ${dir_memdb_4phase_relation_decision}  fused: ${dir_memdb_4phase_relation_decision_fused}"
# _run_sub "trace: ${dir_logs_memory_trace_relation_decision}"
# python -u src/pipeline_meme_4phase.py \
#   "${common_4phase[@]}" \
#   --update-method relation_decision \
#   --database-root "${dir_memdb_4phase_relation_decision}" \
#   --fused-database-root "${dir_memdb_4phase_relation_decision_fused}" \
#   --trace-log-dir "${dir_logs_memory_trace_relation_decision}" \
#   --parallel-episodes "${meme_4phase_parallel_episodes_relation_decision}" \
#   --output "${file_experiment_pred_relation_decision_jsonl}" \
#   "${eval_forward[@]}"

# _run_step "4-Phase mem0 → ${file_experiment_pred_mem0_jsonl}"
# _run_sub "DB: ${dir_memdb_4phase_mem0}  trace: ${dir_logs_memory_trace_mem0}"
# python -u src/pipeline_meme_4phase.py \
#   "${common_4phase[@]}" \
#   --update-method mem0 \
#   --database-root "${dir_memdb_4phase_mem0}" \
#   --trace-log-dir "${dir_logs_memory_trace_mem0}" \
#   --parallel-episodes "${meme_4phase_parallel_episodes_mem0}" \
#   --output "${file_experiment_pred_mem0_jsonl}" \
#   "${eval_forward[@]}"

# _run_step "4-Phase zep → ${file_experiment_pred_zep_jsonl}"
# _run_sub "DB: ${dir_memdb_4phase_zep}  trace: ${dir_logs_memory_trace_zep}"
# python -u src/pipeline_meme_4phase.py \
#   "${common_4phase[@]}" \
#   --update-method zep \
#   --database-root "${dir_memdb_4phase_zep}" \
#   --trace-log-dir "${dir_logs_memory_trace_zep}" \
#   --parallel-episodes "${meme_4phase_parallel_episodes_zep}" \
#   --output "${file_experiment_pred_zep_jsonl}" \
#   "${eval_forward[@]}"

_run_step "4-Phase amac → ${file_experiment_pred_amac_jsonl}"
_run_sub "DB: ${dir_memdb_4phase_amac}  trace: ${dir_logs_memory_trace_amac}"
python -u src/pipeline_meme_4phase.py \
  "${common_4phase[@]}" \
  --update-method amac \
  --database-root "${dir_memdb_4phase_amac}" \
  --trace-log-dir "${dir_logs_memory_trace_amac}" \
  --parallel-episodes "${meme_4phase_parallel_episodes_amac}" \
  --amac-threshold "${amac_threshold}" \
  --output "${file_experiment_pred_amac_jsonl}" \
  "${eval_forward[@]}"

# _run_step "4-Phase add_all → ${file_experiment_pred_add_all_jsonl}"
# _run_sub "DB: ${dir_memdb_4phase_add_all}  trace: ${dir_logs_memory_trace_add_all}"
# python -u src/pipeline_meme_4phase.py \
#   "${common_4phase[@]}" \
#   --update-method add_all \
#   --database-root "${dir_memdb_4phase_add_all}" \
#   --trace-log-dir "${dir_logs_memory_trace_add_all}" \
#   --parallel-episodes "${meme_4phase_parallel_episodes_add_all}" \
#   --output "${file_experiment_pred_add_all_jsonl}" \
#   "${eval_forward[@]}"

_run_step "4-Phase evermemos → ${file_experiment_pred_evermemos_jsonl}"
_run_sub "DB: ${dir_memdb_4phase_evermemos}  trace: ${dir_logs_memory_trace_evermemos}"
python -u src/pipeline_meme_4phase.py \
  "${common_4phase[@]}" \
  --update-method evermemos \
  --database-root "${dir_memdb_4phase_evermemos}" \
  --trace-log-dir "${dir_logs_memory_trace_evermemos}" \
  --parallel-episodes "${meme_4phase_parallel_episodes_evermemos}" \
  --output "${file_experiment_pred_evermemos_jsonl}" \
  "${eval_forward[@]}"

# =============================================================================
# MEME Judge（task-specific prompts + trivial-pass 过滤）
# =============================================================================
_run_step "MEME Judge → eval_meme_judge.json"
meme_judge_concurrency="${evaluate_meme_judge_max_concurrency:-8}"
meme_judge_tokens="${tok_evaluate_meme_judge_max_new_tokens:-512}"

judge_inputs=()
for pred_file in \
    "${file_experiment_pred_relation_decision_jsonl}" \
    "${file_experiment_pred_mem0_jsonl}" \
    "${file_experiment_pred_zep_jsonl}" \
    "${file_experiment_pred_amac_jsonl}" \
    "${file_experiment_pred_add_all_jsonl}" \
    "${file_experiment_pred_evermemos_jsonl}"; do
  [[ -f "$pred_file" ]] && judge_inputs+=("$pred_file")
done

if ((${#judge_inputs[@]} > 0)); then
  python -u src/pipeline_meme_evaluate.py \
    --input "${judge_inputs[@]}" \
    --judge_model "$judge_model" \
    --benchmark "$benchmark" \
    --max_concurrency "$meme_judge_concurrency" \
    --max_new_tokens "$meme_judge_tokens" \
    --write_back
  _run_ok "MEME Judge 完成（已 --write_back）"
else
  _run_warn "Judge 跳过：未找到可用的预测 JSONL"
fi

# =============================================================================
# HTML（可选，pred.jsonl 中已含 is_correct = u_pass 字段名不同，仅供参考）
# =============================================================================
_run_step "HTML: 多方法答题对照 → viewer/${viewer_run_tag}/methods_answer_compare.html"
python -u viewer/build_lme_methods_answer_compare_html.py \
  --run-tag "${viewer_run_tag}" \
  --experiment-dir "${dir_experiment_run_root}" || _run_warn "HTML 生成跳过"

echo ""
_run_title "MEME 4-Phase 实验完成"
_run_sub "benchmark=${benchmark}  answer=${answer_model}  judge=${judge_model}"
_run_sub "输出目录: ${dir_experiment_run_root}/"
_run_sub "评分文件: ${dir_experiment_run_root}/eval_meme_judge.json"
_run_sub "指标说明: meme_score = after 题准确率（分母=after_total）；Cas/Abs/Del after 仅 real 算对"
_run_sub "meme_score_raw = after 题 raw u_pass；meme_score_judge_totals = (before+after) 如 judge.py"
_run_sub "Cas/Abs/Del after 细拆: real / trivial / knew_but_failed / never_knew"
_run_sub "DB 目录结构: {method}_4p/{hn}_before / {hn}_after（episodes × 2 phases）"

exit 0
}
