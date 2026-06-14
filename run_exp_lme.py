#!/usr/bin/env python3
"""
fact_memory LME 实验流水线（Python 入口）

用法：
  python run_exp_lme.py [--config config/lme.yaml] [--stages extract,ingest,generate,evaluate]

比较多种方法：在 config/lme.yaml 的 methods 下将需要比较的方法 enabled 设为 true，
再运行此脚本即可——ingest 和 generate 会依次为每种方法执行，evaluate 自动汇总所有 pred 文件。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# 确保可以从 src/ 导入 utils.config
_REPO_ROOT = Path(__file__).parent.resolve()
_SRC = _REPO_ROOT / "src"
sys.path.insert(0, str(_SRC))

from utils.config import ExperimentConfig  # noqa: E402


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _run(args: list, *, check: bool = True) -> subprocess.CompletedProcess:
    """在 src/ PYTHONPATH 下运行 Python 脚本，实时输出。"""
    env = {**os.environ, "PYTHONPATH": str(_SRC)}
    cmd = [sys.executable, "-u"] + [str(a) for a in args]
    print(f"\n\033[1;32m▶\033[0m {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, env=env)
    if check and result.returncode != 0:
        print(f"\033[35m错误：命令退出码 {result.returncode}\033[0m", file=sys.stderr)
        sys.exit(result.returncode)
    return result


def _run_zep_with_restart(args: list, max_restarts: int = 600) -> None:
    """运行 zep 灌库，遇 SIGSEGV/SIGABRT 自动重启（续传由 --trust-apply-marker 保证幂等）。"""
    env = {**os.environ, "PYTHONPATH": str(_SRC)}
    cmd = [sys.executable, "-u"] + [str(a) for a in args]
    restarts = 0
    while True:
        result = subprocess.run(cmd, env=env)
        if result.returncode == 0:
            break
        if result.returncode in (134, 139):  # SIGABRT / SIGSEGV
            restarts += 1
            if restarts >= max_restarts:
                print(
                    f"灌库 zep 崩溃次数达到上限 {max_restarts}，终止。",
                    file=sys.stderr,
                )
                sys.exit(1)
            print(
                f"灌库 zep 进程崩溃（exit {result.returncode}，第 {restarts} 次重启）…",
                flush=True,
            )
        else:
            print(f"灌库 zep 以非崩溃错误退出（exit {result.returncode}），终止。", file=sys.stderr)
            sys.exit(result.returncode)


def _title(text: str) -> None:
    print(f"\n\033[1;36m{'=' * 60}\033[0m")
    print(f"\033[1;36m  {text}\033[0m")
    print(f"\033[1;36m{'=' * 60}\033[0m", flush=True)


# ---------------------------------------------------------------------------
# 流水线各阶段
# ---------------------------------------------------------------------------

def stage_extract(cfg: ExperimentConfig) -> None:
    _title(f"抽取候选记忆 → {cfg.candidates_dir}")

    args = [
        _SRC / "pipeline" / "extract_candidates.py",
        "--benchmark", cfg.experiment.benchmark,
        "--output", cfg.candidates_dir,
        "--model", cfg.models.extract,
        "--suffix", cfg.extract.candidate_suffix,
        "--memory-granularity", cfg.extract.granularity,
        "--turn-overlap", cfg.extract.turn_overlap,
        "--language", cfg.extract.language,
        "--max-new-tokens", cfg.token_limits.extract_max_new_tokens,
        "--chunk-concurrency", cfg.parallel.extract_chunk_concurrency,
    ]
    args.append("--mem-extract-aspects-only")
    for t in cfg.extract.aspect_templates:
        args += ["--mem-extract-extra-template", t]

    _run(args)


def stage_ingest(cfg: ExperimentConfig) -> None:
    methods = cfg.enabled_methods
    if not methods:
        print("ingest 跳过：没有 enabled 的方法", flush=True)
        return

    ingest_shared = [
        "--benchmark", cfg.experiment.benchmark,
        "--candidate-extract-model", cfg._safe_tag(cfg.models.extract),
        "--candidate-suffix", cfg.extract.candidate_suffix,
        "--candidates-dir", cfg.candidates_dir,
        "--manager-model", cfg.models.manager,
        "--embedding-model", cfg.models.embedding,
        "--language", cfg.extract.language,
        "--relation-concurrency", cfg.parallel.ingest_relation_concurrency,
        "--relation-max-new-tokens", cfg.token_limits.ingest_relation_max_new_tokens,
        "--manager-max-new-tokens", cfg.token_limits.ingest_manager_max_new_tokens,
        "--trust-apply-marker",
    ]

    for method_name in methods:
        db_root = cfg.ingest_dir(method_name)
        _title(f"灌库 {method_name} → {db_root}")

        concurrency_cfg = cfg.parallel.ingest_episode_concurrency
        episode_concurrency = getattr(concurrency_cfg, method_name, 1)
        method_cfg = getattr(cfg.methods, method_name)

        base_args = [
            _SRC / "pipeline" / "ingest_candidates.py",
            "--update-method", method_name,
            "--database-root", db_root,
        ] + ingest_shared

        if method_name == "amac":
            extra = [
                "--amac-episode-concurrency", episode_concurrency,
                "--amac-threshold", method_cfg.threshold,
                "--amac-weights", method_cfg.weights,
                "--amac-recency-decay-per-step", method_cfg.recency_decay_per_step,
                "--amac-novelty-max-existing", method_cfg.novelty_max_existing,
                "--ingest-obs-granularity", cfg.extract.granularity,
                "--ingest-obs-turn-overlap", cfg.extract.turn_overlap,
            ]
            if method_cfg.skip_utility:
                extra.append("--amac-skip-utility")
            _run(base_args + extra)

        elif method_name == "relation_decision":
            p = cfg.prompts
            extra = [
                "--relation-episode-concurrency", episode_concurrency,
                "--related-top-k", method_cfg.related_top_k,
            ]
            if p.relation_system_en:
                extra += ["--relation-system-template-en", p.relation_system_en]
            if p.relation_system_zh:
                extra += ["--relation-system-template-zh", p.relation_system_zh]
            if p.relation_user:
                extra += ["--relation-user-template", p.relation_user]
            _run(base_args + extra)
            _stage_fuse(cfg)

        elif method_name == "mem0":
            extra = [
                "--mem0-episode-concurrency", episode_concurrency,
                "--mem0-related-top-k", method_cfg.related_top_k,
                "--mem0-related-aggregate-max", method_cfg.related_aggregate_max,
            ]
            _run(base_args + extra)

        elif method_name == "add_all":
            _run(base_args + ["--add-all-episode-concurrency", episode_concurrency])

        elif method_name == "zep":
            _run_zep_with_restart(base_args + ["--zep-episode-concurrency", episode_concurrency])

        elif method_name == "evermemos":
            extra = [
                "--evermemos-episode-concurrency", episode_concurrency,
                "--evermemos-similarity-threshold", method_cfg.similarity_threshold,
                "--evermemos-max-time-gap-days", method_cfg.max_time_gap_days,
            ]
            _run(base_args + extra)


def _stage_fuse(cfg: ExperimentConfig) -> None:
    """relation_decision 专用：关系包融合。"""
    p = cfg.prompts
    ec = cfg.parallel.ingest_episode_concurrency
    fusion_model = cfg.methods.relation_decision.fusion_model or cfg.models.manager

    _title(f"关系包融合 → {cfg.ingest_dir('relation_decision_fused')}")

    args = [
        _SRC / "pipeline" / "fuse_lme_memory_bundles.py",
        "--database-root", cfg.ingest_dir("relation_decision"),
        "--fused-output-root", cfg.ingest_dir("relation_decision_fused"),
        "--manager-model", fusion_model,
        "--embedding-model", cfg.models.embedding,
        "--language", cfg.extract.language,
        "--fuse-max-new-tokens", cfg.token_limits.fusion_max_new_tokens,
        "--episode-concurrency", ec.fusion_episodes,
        "--package-concurrency", ec.fusion_packages,
    ]
    if p.fusion_bundle_en:
        args += ["--fusion-bundle-template-en", p.fusion_bundle_en]
    if p.fusion_bundle_zh:
        args += ["--fusion-bundle-template-zh", p.fusion_bundle_zh]
    if p.fusion_edge_labels_en:
        args += ["--fusion-edge-labels-template-en", p.fusion_edge_labels_en]
    if p.fusion_edge_labels_zh:
        args += ["--fusion-edge-labels-template-zh", p.fusion_edge_labels_zh]
    _run(args)


def stage_generate(cfg: ExperimentConfig) -> None:
    methods = cfg.enabled_methods
    if not methods:
        print("generate 跳过：没有 enabled 的方法", flush=True)
        return

    g = cfg.generate
    # MEME：全量评测，不分层抽样
    sample = 0 if cfg.experiment.benchmark.lower().startswith("meme") else g.answer_stratified_sample

    common_args = [
        "--method", "lme_prebuilt",
        "--benchmark", cfg.experiment.benchmark,
        "--answer_model", cfg.models.answer,
        "--embedding_model", cfg.models.embedding,
        "--retrieve_topk", g.retrieve_topk,
        "--memory_token_limit", g.memory_token_limit,
        "--parallel_episodes", cfg.parallel.generate_parallel_episodes,
        "--answer-concurrency", cfg.parallel.generate_answer_concurrency,
        "--answer-stratified-sample", sample,
        "--answer-sample-seed", g.answer_sample_seed,
    ]
    if not g.show_memory_time:
        common_args.append("--no-memory-time")
    if g.hybrid.enabled:
        common_args += [
            "--hybrid-bm25-dense",
            "--hybrid-dense-weight", g.hybrid.dense_weight,
            "--hybrid-bm25-weight", g.hybrid.bm25_weight,
            "--hybrid-pool-mult", g.hybrid.pool_mult,
        ]
    if g.rerank.enabled:
        common_args += ["--rerank-qwen3-vllm", "--rerank-top-k", g.retrieve_topk]

    cfg.experiment_run_root.mkdir(parents=True, exist_ok=True)

    for method_name in methods:
        # relation_decision uses fused DB
        db_root = cfg.ingest_dir(
            "relation_decision_fused" if method_name == "relation_decision" else method_name
        )
        output = cfg.pred_file(method_name)
        trace_dir = cfg.experiment_run_root / "agent_trace" / method_name

        _title(f"生成预测: {method_name} → {output}")

        _run([
            _SRC / "pipeline_lme_generate.py",
        ] + common_args + [
            "--database_root", db_root,
            "--output", output,
            "--agent_trace_dir", trace_dir,
        ])


def stage_evaluate(cfg: ExperimentConfig) -> None:
    pred_files = [
        str(cfg.pred_file(m))
        for m in cfg.enabled_methods
        if cfg.pred_file(m).exists()
    ]
    if not pred_files:
        print("Judge 跳过：未找到可用的预测 JSONL", flush=True)
        return

    _title(f"LLM Judge ({len(pred_files)} 文件)")

    e = cfg.evaluate
    p = cfg.prompts

    args = [
        _SRC / "pipeline_lme_evaluate.py",
        "--input", *pred_files,
        "--judge_model", cfg.models.judge,
        "--benchmark", cfg.experiment.benchmark,
        "--max_concurrency", cfg.parallel.evaluate_max_concurrency,
        "--max_new_tokens", cfg.token_limits.evaluate_max_new_tokens,
        "--write_back",
    ]
    if e.use_cot:
        args.append("--use_cot")
    if e.judge_stratified_sample > 0:
        args += [
            "--stratified-sample-n", e.judge_stratified_sample,
            "--stratified-sample-seed", e.judge_sample_seed,
        ]
    if cfg.debug.evaluate_print_one_sample:
        args.append("--print-one-sample")
    if p.judge_oqa:
        args += ["--judge-oqa-template", p.judge_oqa]
    if p.judge_mcq:
        args += ["--judge-mcq-template", p.judge_mcq]
    if p.judge_system:
        args += ["--judge-system-template", p.judge_system]

    _run(args)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

_ALL_STAGES = ("extract", "ingest", "generate", "evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="fact_memory 实验流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python run_exp_lme.py                             # 跑所有阶段\n"
            "  python run_exp_lme.py --stages ingest,generate   # 只跑 ingest + generate\n"
            "  python run_exp_lme.py --config config/lme.yaml   # 使用自定义配置\n"
        ),
    )
    parser.add_argument(
        "--config",
        default="config/lme.yaml",
        help="实验配置 YAML（默认 config/lme.yaml）",
    )
    parser.add_argument(
        "--stages",
        default=",".join(_ALL_STAGES),
        help=f"逗号分隔的运行阶段（默认全部）：{', '.join(_ALL_STAGES)}",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"错误：找不到配置文件 {config_path}", file=sys.stderr)
        sys.exit(1)

    cfg = ExperimentConfig.from_yaml(config_path)
    stages = {s.strip() for s in args.stages.split(",")}

    # 切到项目根目录（路径均为相对路径）
    os.chdir(_REPO_ROOT)

    _title(
        f"fact_memory | benchmark={cfg.experiment.benchmark} | suffix={cfg.experiment.suffix}"
        f" | methods={cfg.enabled_methods or ['(none)']}"
    )

    if "extract" in stages:
        stage_extract(cfg)
    if "ingest" in stages:
        stage_ingest(cfg)
    if "generate" in stages:
        stage_generate(cfg)
    if "evaluate" in stages:
        stage_evaluate(cfg)

    _title("完成")
    print(f"  预测目录：{cfg.experiment_run_root}", flush=True)
    for method in cfg.enabled_methods:
        p = cfg.pred_file(method)
        status = "✓" if p.exists() else "✗"
        print(f"  {status} {method}: {p}", flush=True)


if __name__ == "__main__":
    main()
