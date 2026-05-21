#!/usr/bin/env bash
# =============================================================================
# run_exp.sh — LME 实验流水线（抽取 → 灌库 → 生成 → Judge → HTML）
# 说明：关系包融合是 relation_decision 灌库方法的后续子步骤，不是与灌库并列的通用阶段。
#
# 用法:
#   ./script/run_exp.sh
#   RUN_EXP_CONFIG=/path/to/custom.yaml ./script/run_exp.sh
#   ./script/run_exp.sh --config script/run_exp.config.yaml [额外参数透传给生成阶段]
#
# 修改实验: 优先改下方「可调参数」与 run_exp.config.yaml；并行度 / token / 路径 / 全部 Jinja 模板名见 YAML 的 parallel / token_limits / prompts
# =============================================================================
set -euo pipefail
export PYTHONPATH=src

# 将整个脚本体包在 { ... } 组合命令里：Bash 会先把花括号块完整解析完再执行，
# 这样即便脚本在长时间步骤（如抽取、灌库）运行期间被编辑保存，也不会因为
# 文件偏移错位导致 "line N: xxx: command not found" 之类的幽灵错误。
{

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
# 可选值:
#   lme_o / lme_s / lme_m       — LongMemEval 系列
#   locomo                       — LoCoMo
#   meme_nofiller                — MEME, 纯证据会话（无 filler 噪声，约 4.6 MB）
#   meme_filler32k               — MEME, ~32k filler（默认基准配置，约 20 MB）
#   meme_filler128k              — MEME, ~128k filler（压力测试子集，40 episodes，约 27 MB）
# MEME 说明: evidence 会话直接使用 gold_facts 作为候选记忆（无 LLM 抽取），
#            filler 会话走 LLM 三方面抽取（与 lme_s 相同 0_mem_extract_v2.jinja）；
#            评测题来自 after_questions（ER/Agg/Tr/Del/Cas/Abs 六类任务）。
benchmark=lme_s
question_types=""
# 须与 YAML prompts.mem_extract_aspect_template_* 的 *_en / *_zh 一致（三方面抽取时尤需注意）
apply_language="en"

# --- 模型 ---
extract_model="gemma4-26B"
manager_model="gemma4-26B"
answer_model="gemma4-26B"
judge_model="gpt-4o-mini"
embedding_model="qwen3-embedding-8b"

# --- 版本标签（换抽取策略时请改 candidate_suffix，否则会沿用 extract_progress.state 跳过已标记 episode）---
candidate_suffix="0507_as3"
exp_suffix="0521"

# --- 1) 候选抽取（粗粒度、窗口；模板名在 run_exp.config.yaml -> prompts）---
memory_granularity=4
# 滑动窗口：相邻块共享的 turn 数（0=不重叠）。须小于 memory_granularity。
extract_turn_overlap=0

# --- 2) 候选灌库（检索规模）---
relation_related_top_k=3
mem0_related_top_k=3
mem0_related_aggregate_max=10
amac_threshold=0.3

# relation_decision 专用：关系包融合模型（默认与 manager 一致；融合模板在 YAML prompts）
fusion_model="${manager_model}"

# --- 3) 生成 / 答题（混合检索）---
# 仅答题与 Judge：按题库中 question_type 比例分层抽样（0=全量；与 --question-types 可同时用）
answer_stratified_sample=500
answer_sample_seed=43
# 若已有完整 pred.jsonl、只想对 Judge 做分层抽样，可设此项（0=与答题一致）
judge_stratified_sample=0
judge_sample_seed=43

answer_retrieve_topk=50
# 召回记忆是否展示时间（1=展示；0=不展示，对应 --no-memory-time）
answer_show_memory_time=0
# 精排：需先启动 script/0_run_reranker_ppu.sh（vLLM /v1/score）；粗排与精排均为 top answer_retrieve_topk
answer_rerank_qwen3_vllm=0
answer_hybrid_dense_weight=0.8
answer_hybrid_bm25_weight=0.2
answer_hybrid_pool_mult=4

# --- 4) Judge（Qwen3 Judge 服务端思考链；模板在 YAML prompts）---
# judge_pipeline_opts=()
# if [[ "$judge_model" == "Qwen3-32B" ]]; then
#   judge_pipeline_opts+=(--judge-qwen-thinking)
# fi

judge_pipeline_opts=(--use_cot)

# =============================================================================
# 派生变量 + 加载 YAML（输出 dir_* / tok_* / 并发 / prompts 等到当前 shell）
# =============================================================================
model_tag="$(printf '%s' "$extract_model" | tr '/ :\\' '____' | tr -cd '[:alnum:]_.-')"
if [[ -z "$model_tag" ]]; then
  _run_err "错误：model_tag 为空，请检查 extract_model=${extract_model}"
  exit 1
fi

export benchmark extract_model candidate_suffix manager_model answer_model exp_suffix

_run_title "加载配置"
_run_sub "文件: ${run_exp_config}"
cfg_exports=""
if ! cfg_exports="$(python -u script/run_exp_load_config.py "$run_exp_config")"; then
  _run_err "错误：配置加载失败，请先修复上面的 unresolved placeholder 或 YAML 格式问题。"
  exit 1
fi
# shellcheck disable=SC1090
eval "$cfg_exports"
_run_ok "路径、并行、prompts 已注入环境"
if [[ "${mem_extract_aspects_only:-0}" == "1" || "${mem_extract_aspects_only,,}" == "true" ]]; then
  _run_sub "候选抽取: 三方面模板（--mem-extract-aspects-only）language=${apply_language}"
else
  _run_sub "候选抽取: 仅主模板 language=${apply_language}"
fi

# YAML: debug.evaluate_llm_judge_print_one_sample 在 eval 之后才生效
if [[ "${evaluate_llm_judge_print_one_sample:-0}" == "1" ]]; then
  judge_pipeline_opts+=(--print-one-sample)
fi
judge_pipeline_opts+=(--stratified-sample-n "${judge_stratified_sample}" --stratified-sample-seed "${judge_sample_seed}")

# =============================================================================
# 流水线：抽取 → 灌库（relation_decision 含关系包融合 / mem0 / zep / add_all / amac）→ 生成 → Judge
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
if [[ -n "${mem_extract_template}" ]]; then
  extract_py_args+=(--mem-extract-template "${mem_extract_template}")
fi
if [[ "${mem_extract_aspects_only:-0}" == "1" || "${mem_extract_aspects_only,,}" == "true" ]]; then
  extract_py_args+=(--mem-extract-aspects-only)
fi
if [[ -n "${mem_extract_aspect_template_1:-}" ]]; then
  extract_py_args+=(--mem-extract-extra-template "${mem_extract_aspect_template_1}")
fi
if [[ -n "${mem_extract_aspect_template_2:-}" ]]; then
  extract_py_args+=(--mem-extract-extra-template "${mem_extract_aspect_template_2}")
fi
if [[ -n "${mem_extract_aspect_template_3:-}" ]]; then
  extract_py_args+=(--mem-extract-extra-template "${mem_extract_aspect_template_3}")
fi
# LoCoMo：候选抽取用具名说话人转写（与 0_mem_extract_locomo*.jinja 一致）；其它 benchmark 仍走 extract_candidates 默认 user_assistant
# MEME：evidence 会话由 MEMEBenchmark 在 Python 侧自动识别并跳过 LLM（直接用 gold_facts），无需额外 CLI 标志
case "${benchmark,,}" in
  locomo*) extract_py_args+=(--dialogue-format named_speakers) ;;
  meme*)   : ;;  # no extra flags; hybrid extraction is auto-detected from benchmark name
esac

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

ingest_relation_decision_prompt_args=()
if [[ -n "${relation_classification_system_en:-}" ]]; then
  ingest_relation_decision_prompt_args+=(--relation-system-template-en "${relation_classification_system_en}")
fi
if [[ -n "${relation_classification_system_zh:-}" ]]; then
  ingest_relation_decision_prompt_args+=(--relation-system-template-zh "${relation_classification_system_zh}")
fi
if [[ -n "${relation_classification_user:-}" ]]; then
  ingest_relation_decision_prompt_args+=(--relation-user-template "${relation_classification_user}")
fi

# ----- relation_decision（关系分类写库 → 关系包融合；仅此灌库策略含融合）-----
_run_step "relation_decision（1/2 关系分类写库）→ ${dir_memdb_ingest_relation_decision}"
python -u src/pipeline/ingest_candidates.py \
  --update-method relation_decision \
  --trust-apply-marker \
  --database-root "${dir_memdb_ingest_relation_decision}" \
  --trace-log-dir "${dir_logs_memory_trace_relation_decision}" \
  "${ingest_shared[@]}" \
  "${ingest_relation_decision_prompt_args[@]}" \
  --related-top-k "${relation_related_top_k}" \
  --relation-episode-concurrency "${ingest_relation_decision_episode_concurrency}"

# _run_step "HTML: relation_decision 特殊关系 → viewer/${viewer_run_tag}/relation_decision_special_relations.html"
# python -u viewer/build_lme_relation_decision_special_html.py \
#   --trace-log-dir "${dir_logs_memory_trace_relation_decision}" \
#   --out "${repo_root}/viewer/${viewer_run_tag}/relation_decision_special_relations.html"

# 从 relation_decision 只读拷贝各 episode 到 relation_decision_fused 再融合；已融合 episode 自动跳过
_run_step "relation_decision（2/2 关系包融合）→ ${dir_memdb_ingest_relation_decision_fused}"
_run_sub "自 ${dir_memdb_ingest_relation_decision} 只读拷贝后融合；非 mem0/zep/add_all/amac 的通用步骤"
fuse_cli_extra=()
if [[ -n "${fusion_bundle_template_en}" ]]; then
  fuse_cli_extra+=(--fusion-bundle-template-en "${fusion_bundle_template_en}")
fi
if [[ -n "${fusion_bundle_template_zh}" ]]; then
  fuse_cli_extra+=(--fusion-bundle-template-zh "${fusion_bundle_template_zh}")
fi
if [[ -n "${fusion_edge_labels_template_en:-}" ]]; then
  fuse_cli_extra+=(--fusion-edge-labels-template-en "${fusion_edge_labels_template_en}")
fi
if [[ -n "${fusion_edge_labels_template_zh:-}" ]]; then
  fuse_cli_extra+=(--fusion-edge-labels-template-zh "${fusion_edge_labels_template_zh}")
fi
python -u src/pipeline/fuse_lme_memory_bundles.py \
  --database-root "${dir_memdb_ingest_relation_decision}" \
  --fused-output-root "${dir_memdb_ingest_relation_decision_fused}" \
  "${fuse_cli_extra[@]}" \
  "${fuse_shared[@]}"

# _run_step "对照页: relation_decision → 融合 → viewer/${viewer_run_tag}/relation_decision_to_fusion.html"
# python -u viewer/build_lme_fusion_bundle_map_html.py \
#   --source-root "${dir_memdb_ingest_relation_decision}" \
#   --fused-root "${dir_memdb_ingest_relation_decision_fused}" \
#   --out "${repo_root}/viewer/${viewer_run_tag}/relation_decision_to_fusion.html"

_run_step "灌库 add_all → ${dir_memdb_ingest_add_all}"
python -u src/pipeline/ingest_candidates.py \
  --update-method add_all \
  --trust-apply-marker \
  --database-root "${dir_memdb_ingest_add_all}" \
  --trace-log-dir "${dir_logs_memory_trace_add_all}" \
  "${ingest_shared[@]}" \
  --add-all-episode-concurrency "${ingest_add_all_episode_concurrency}"

_run_step "灌库 mem0 → ${dir_memdb_ingest_mem0}"
python -u src/pipeline/ingest_candidates.py \
  --update-method mem0 \
  --trust-apply-marker \
  --database-root "${dir_memdb_ingest_mem0}" \
  --trace-log-dir "${dir_logs_memory_trace_mem0}" \
  "${ingest_shared[@]}" \
  --mem0-related-top-k "${mem0_related_top_k}" \
  --mem0-related-aggregate-max "${mem0_related_aggregate_max}" \
  --mem0-episode-concurrency "${ingest_mem0_episode_concurrency}"

# _run_step "灌库 zep → ${dir_memdb_ingest_zep}"
# python -u src/pipeline/ingest_candidates.py \
#   --update-method zep \
#   --trust-apply-marker \
#   --database-root "${dir_memdb_ingest_zep}" \
#   --trace-log-dir "${dir_logs_memory_trace_zep}" \
#   "${ingest_shared[@]}" \
#   --zep-episode-concurrency "${ingest_zep_episode_concurrency}"

_run_step "灌库 amac → ${dir_memdb_ingest_amac}"
ingest_amac_cli=(
  --update-method amac
  --trust-apply-marker
  --database-root "${dir_memdb_ingest_amac}"
  --trace-log-dir "${dir_logs_memory_trace_amac}"
  "${ingest_shared[@]}"
  --ingest-obs-granularity "${memory_granularity}"
  --ingest-obs-turn-overlap "${extract_turn_overlap}"
  --amac-episode-concurrency "${ingest_amac_episode_concurrency}"
  --amac-threshold "${amac_threshold}"
)
case "${benchmark,,}" in
  locomo*) ingest_amac_cli+=(--ingest-obs-dialogue-format named_speakers) ;;
  meme*)   : ;;
esac
python -u src/pipeline/ingest_candidates.py "${ingest_amac_cli[@]}"

_run_step "灌库 evermemos → ${dir_memdb_ingest_evermemos}"
python -u src/pipeline/ingest_candidates.py \
  --update-method evermemos \
  --trust-apply-marker \
  --database-root "${dir_memdb_ingest_evermemos}" \
  --trace-log-dir "${dir_logs_memory_trace_evermemos}" \
  "${ingest_shared[@]}" \
  --evermemos-episode-concurrency "${ingest_evermemos_episode_concurrency}"

# MEME：~694 道题全量评测（不分层抽样），覆盖 6 种任务类型
case "${benchmark,,}" in
  meme*) answer_stratified_sample=0 ;;
esac

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
  --answer-stratified-sample "$answer_stratified_sample"
  --answer-sample-seed "$answer_sample_seed"
)
if [[ "${answer_show_memory_time:-1}" == "0" ]]; then
  common_py+=(--no-memory-time)
fi
if [[ "${answer_rerank_qwen3_vllm:-0}" == "1" ]]; then
  common_py+=(--rerank-qwen3-vllm --rerank-top-k "${answer_retrieve_topk}")
  _run_sub "答题精排: 粗排 top${answer_retrieve_topk} → Qwen3-Reranker 精排 top${answer_retrieve_topk}（需先启动 script/0_run_reranker_ppu.sh）"
fi

_run_step "生成预测: relation_decision_fused → ${file_experiment_pred_relation_decision_jsonl}"
_run_sub "agent trace: ${dir_logs_agent_trace_relation_decision}"
python -u "${common_py[@]}" \
  --agent_trace_dir "${dir_logs_agent_trace_relation_decision}" \
  --database_root "${dir_memdb_ingest_relation_decision_fused}" \
  --hybrid-full-corpus-pool \
  --output "${file_experiment_pred_relation_decision_jsonl}" \
  "${eval_forward[@]}"

_run_step "生成预测: mem0 → ${file_experiment_pred_mem0_jsonl}"
_run_sub "agent trace: ${dir_logs_agent_trace_mem0}"
python -u "${common_py[@]}" \
  --agent_trace_dir "${dir_logs_agent_trace_mem0}" \
  --database_root "${dir_memdb_ingest_mem0}" \
  --output "${file_experiment_pred_mem0_jsonl}" \
  "${eval_forward[@]}"

_run_step "生成预测: zep → ${file_experiment_pred_zep_jsonl}"
_run_sub "agent trace: ${dir_logs_agent_trace_zep}"
python -u "${common_py[@]}" \
  --agent_trace_dir "${dir_logs_agent_trace_zep}" \
  --database_root "${dir_memdb_ingest_zep}" \
  --output "${file_experiment_pred_zep_jsonl}" \
  "${eval_forward[@]}"

_run_step "生成预测: amac → ${file_experiment_pred_amac_jsonl}"
_run_sub "agent trace: ${dir_logs_agent_trace_amac}"
python -u "${common_py[@]}" \
  --agent_trace_dir "${dir_logs_agent_trace_amac}" \
  --database_root "${dir_memdb_ingest_amac}" \
  --output "${file_experiment_pred_amac_jsonl}" \
  "${eval_forward[@]}"

_run_step "生成预测: add_all → ${file_experiment_pred_add_all_jsonl}"
_run_sub "agent trace: ${dir_logs_agent_trace_add_all}"
python -u "${common_py[@]}" \
  --agent_trace_dir "${dir_logs_agent_trace_add_all}" \
  --database_root "${dir_memdb_ingest_add_all}" \
  --output "${file_experiment_pred_add_all_jsonl}" \
  "${eval_forward[@]}"

_run_step "生成预测: evermemos → ${file_experiment_pred_evermemos_jsonl}"
_run_sub "agent trace: ${dir_logs_agent_trace_evermemos}"
python -u "${common_py[@]}" \
  --agent_trace_dir "${dir_logs_agent_trace_evermemos}" \
  --database_root "${dir_memdb_ingest_evermemos}" \
  --output "${file_experiment_pred_evermemos_jsonl}" \
  "${eval_forward[@]}"

_run_step "LLM Judge（多文件并行，写回预测 JSONL）"
judge_inputs=()
for pred_file in \
    "${file_experiment_pred_relation_decision_jsonl}" \
    "${file_experiment_pred_mem0_jsonl}" \
    "${file_experiment_pred_zep_jsonl}" \
    "${file_experiment_pred_amac_jsonl}" \
    "${file_experiment_pred_add_all_jsonl}" \
    "${file_experiment_pred_evermemos_jsonl}"; do
  if [[ -f "$pred_file" ]]; then
    judge_inputs+=("$pred_file")
  fi
done
if ((${#judge_inputs[@]} > 0)); then
  judge_py_args=(
    -u src/pipeline_evaluate.py
    --input "${judge_inputs[@]}"
    --judge_model "$judge_model"
    --benchmark "$benchmark"
  )
  [[ -n "${judge_oqa_template:-}" ]] && judge_py_args+=(--judge-oqa-template "${judge_oqa_template}")
  [[ -n "${judge_mcq_template:-}" ]] && judge_py_args+=(--judge-mcq-template "${judge_mcq_template}")
  [[ -n "${judge_system_template:-}" ]] && judge_py_args+=(--judge-system-template "${judge_system_template}")
  judge_py_args+=(
    "${judge_pipeline_opts[@]}"
    --max_new_tokens "$tok_evaluate_llm_judge_max_new_tokens"
    --max_concurrency "$evaluate_llm_judge_max_concurrency"
    --write_back
  )
  python "${judge_py_args[@]}"
  _run_ok "Judge 完成（已 --write_back）"
else
  _run_warn "Judge 跳过：未找到可用的预测 JSONL"
fi

# _run_step "HTML: add_all vs relation_decision → viewer/${viewer_run_tag}/add_all_vs_relation_decision.html"
# python -u viewer/build_lme_case_study_html.py --run-tag "${viewer_run_tag}" --experiment-dir "${dir_experiment_run_root}"

_run_step "HTML: 多方法答题对照 → viewer/${viewer_run_tag}/methods_answer_compare.html"
python -u viewer/build_lme_methods_answer_compare_html.py --run-tag "${viewer_run_tag}" --experiment-dir "${dir_experiment_run_root}"

# =============================================================================
# 结束摘要
# =============================================================================
echo ""
_run_title "完成"
_run_sub "benchmark=${benchmark}  |  答题模型=${answer_model}  |  Judge=${judge_model}"
_run_sub "预测与评分目录: ${dir_experiment_run_root}/"
_run_sub "  relation_decision_fused: ${file_experiment_pred_relation_decision_jsonl}"
_run_sub "  mem0:                    ${file_experiment_pred_mem0_jsonl}"
_run_sub "  zep:                     ${file_experiment_pred_zep_jsonl}"
_run_sub "  amac:                    ${file_experiment_pred_amac_jsonl}"
_run_sub "  add_all:                 ${file_experiment_pred_add_all_jsonl}"
_run_sub "  evermemos:               ${file_experiment_pred_evermemos_jsonl}"
echo ""
# _run_sub "viewer/${viewer_run_tag}/relation_decision_special_relations.html"
# _run_sub "viewer/${viewer_run_tag}/relation_decision_to_fusion.html"
_run_sub "  methods_answer_compare:    viewer/${viewer_run_tag}/methods_answer_compare.html"
# _run_sub "viewer/${viewer_run_tag}/add_all_vs_relation_decision.html"

exit 0
}
