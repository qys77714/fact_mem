#!/usr/bin/env python3
"""
fact_memory MEME 4-Phase 实验流水线

与官方 MEME-public 协议对齐：每个 episode 分 4 个阶段执行，
ingest + answer 合并在 pipeline_meme_4phase.py 内完成。

用法：
  python run_exp_meme.py [--config config/meme.yaml] [--stages extract,run,evaluate]

阶段说明：
  extract  — 候选记忆抽取（三方面模板，与 run_exp_lme.py 相同）
  run      — 各 enabled 方法的 4-phase 灌库+答题（pipeline_meme_4phase.py）
  evaluate — MEME Judge（pipeline_meme_evaluate.py，task-specific prompts + trivial-pass 过滤）
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.resolve()
_SRC = _REPO_ROOT / "src"
sys.path.insert(0, str(_SRC))

from utils.config import MemeExperimentConfig  # noqa: E402


# ---------------------------------------------------------------------------
# 工具函数（与 run_exp_lme.py 相同）
# ---------------------------------------------------------------------------

def _run(args: list, *, check: bool = True) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(_SRC)}
    cmd = [sys.executable, "-u"] + [str(a) for a in args]
    print(f"\n\033[1;32m▶\033[0m {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, env=env)
    if check and result.returncode != 0:
        print(f"\033[35m错误：命令退出码 {result.returncode}\033[0m", file=sys.stderr)
        sys.exit(result.returncode)
    return result


def _run_zep_with_restart(args: list, max_restarts: int = 600) -> None:
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
                print(f"zep 崩溃次数达到上限 {max_restarts}，终止。", file=sys.stderr)
                sys.exit(1)
            print(f"zep 崩溃（exit {result.returncode}，第 {restarts} 次重启）…", flush=True)
        else:
            print(f"zep 以非崩溃错误退出（exit {result.returncode}），终止。", file=sys.stderr)
            sys.exit(result.returncode)


def _title(text: str) -> None:
    print(f"\n\033[1;36m{'=' * 60}\033[0m")
    print(f"\033[1;36m  {text}\033[0m")
    print(f"\033[1;36m{'=' * 60}\033[0m", flush=True)


# ---------------------------------------------------------------------------
# 流水线各阶段
# ---------------------------------------------------------------------------

def stage_extract(cfg: MemeExperimentConfig) -> None:
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
        "--mem-extract-aspects-only",
    ]
    for t in cfg.extract.aspect_templates:
        args += ["--mem-extract-extra-template", t]

    _run(args)


def stage_run(cfg: MemeExperimentConfig) -> None:
    """4-phase 灌库 + 答题（每方法一次 pipeline_meme_4phase.py 调用）。"""
    methods = cfg.enabled_methods
    if not methods:
        print("run 跳过：没有 enabled 的方法", flush=True)
        return

    r = cfg.run
    p = cfg.prompts
    tl = cfg.token_limits
    par = cfg.parallel

    # 所有方法共享的参数
    common = [
        "--benchmark", cfg.experiment.benchmark,
        "--candidates-dir", cfg.candidates_dir,
        "--answer-model", cfg.models.answer,
        "--embedding-model", cfg.models.embedding,
        "--manager-model", cfg.models.manager,
        "--language", cfg.extract.language,
        "--retrieve-topk", r.retrieve_topk,
        "--memory-token-limit", r.memory_token_limit,
        "--answer-concurrency", r.answer_concurrency,
        "--relation-concurrency", par.ingest_relation_concurrency,
        "--evermemos-cluster-concurrency", par.evermemos_cluster_concurrency,
        "--relation-max-new-tokens", tl.ingest_relation_max_new_tokens,
        "--manager-max-new-tokens", tl.ingest_manager_max_new_tokens,
        "--fuse-max-new-tokens", tl.fusion_max_new_tokens,
        "--fusion-package-concurrency", par.fuse_package_concurrency,
    ]
    if not r.show_memory_time:
        common.append("--no-memory-time")
    if r.hybrid.enabled:
        common += [
            "--hybrid-bm25-dense",
            "--hybrid-dense-weight", r.hybrid.dense_weight,
            "--hybrid-bm25-weight", r.hybrid.bm25_weight,
            "--hybrid-pool-mult", r.hybrid.pool_mult,
        ]

    # relation_decision prompt overrides
    if p.relation_system_en:
        common += ["--relation-system-template-en", p.relation_system_en]
    if p.relation_system_zh:
        common += ["--relation-system-template-zh", p.relation_system_zh]
    if p.relation_user:
        common += ["--relation-user-template", p.relation_user]
    if p.fusion_bundle_en:
        common += ["--fusion-bundle-template-en", p.fusion_bundle_en]
    if p.fusion_bundle_zh:
        common += ["--fusion-bundle-template-zh", p.fusion_bundle_zh]
    if p.fusion_edge_labels_en:
        common += ["--fusion-edge-labels-template-en", p.fusion_edge_labels_en]
    if p.fusion_edge_labels_zh:
        common += ["--fusion-edge-labels-template-zh", p.fusion_edge_labels_zh]

    cfg.experiment_run_root.mkdir(parents=True, exist_ok=True)

    for method_name in methods:
        method_cfg = getattr(cfg.methods, method_name)
        ep_concurrency = getattr(cfg.parallel.parallel_episodes, method_name, 1)
        db_root = cfg.ingest_4phase_dir(method_name)
        output = cfg.pred_file(method_name)
        trace_dir = cfg.experiment_run_root / "memory_trace" / method_name

        _title(f"4-Phase {method_name} → {output}")

        method_args = [
            _SRC / "pipeline_meme_4phase.py",
        ] + common + [
            "--update-method", method_name,
            "--database-root", db_root,
            "--output", output,
            "--parallel-episodes", ep_concurrency,
            "--trace-log-dir", trace_dir,
        ]

        # relation_decision: auto-derives fused root (method_4p_fused) if not passed
        # method-specific params
        if method_name == "amac":
            method_args += [
                "--amac-threshold", method_cfg.threshold,
                "--amac-weights", method_cfg.weights,
                "--amac-recency-decay-per-step", method_cfg.recency_decay_per_step,
                "--amac-novelty-max-existing", method_cfg.novelty_max_existing,
            ]
            if method_cfg.skip_utility:
                method_args.append("--amac-skip-utility")
        elif method_name == "relation_decision":
            method_args += [
                "--related-top-k", method_cfg.related_top_k,
                "--fused-database-root", cfg.ingest_4phase_fused_dir(method_name),
            ]
            if getattr(method_cfg, "cascade_enabled", True):
                method_args.append("--cascade-enabled")
            else:
                method_args.append("--no-cascade-enabled")
            if getattr(method_cfg, "deletion_enabled", True):
                method_args.append("--deletion-enabled")
            else:
                method_args.append("--no-deletion-enabled")
            if getattr(method_cfg, "topic_aggregation_enabled", True):
                method_args.append("--topic-aggregation")
            else:
                method_args.append("--no-topic-aggregation")
            method_args += [
                "--condition-sim-threshold", getattr(method_cfg, "condition_sim_threshold", 0.5),
                "--pairwise-sim-threshold", getattr(method_cfg, "pairwise_sim_threshold", 0.7),
            ]
        elif method_name == "mem0":
            method_args += [
                "--mem0-related-top-k", method_cfg.related_top_k,
                "--mem0-related-aggregate-max", method_cfg.related_aggregate_max,
            ]
        elif method_name == "evermemos":
            method_args += [
                "--evermemos-similarity-threshold", method_cfg.similarity_threshold,
                "--evermemos-max-time-gap-days", method_cfg.max_time_gap_days,
            ]

        if method_name == "zep":
            _run_zep_with_restart(method_args)
        else:
            _run(method_args)


def stage_evaluate(cfg: MemeExperimentConfig) -> None:
    pred_files = [
        str(cfg.pred_file(m))
        for m in cfg.enabled_methods
        if cfg.pred_file(m).exists()
    ]
    if not pred_files:
        print("MEME Judge 跳过：未找到可用的预测 JSONL", flush=True)
        return

    _title(f"MEME Judge ({len(pred_files)} 文件)")

    e = cfg.evaluate
    args = [
        _SRC / "pipeline_meme_evaluate.py",
        "--input", *pred_files,
        "--judge_model", cfg.models.judge,
        "--benchmark", cfg.experiment.benchmark,
        "--max_concurrency", e.judge_max_concurrency,
        "--max_new_tokens", e.judge_max_new_tokens,
        "--write_back",
    ]
    _run(args)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

_ALL_STAGES = ("extract", "run", "evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="fact_memory MEME 4-Phase 实验流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python run_exp_meme.py                         # 跑所有阶段\n"
            "  python run_exp_meme.py --stages run,evaluate   # 跳过抽取\n"
            "  python run_exp_meme.py --config config/meme.yaml\n"
        ),
    )
    parser.add_argument(
        "--config",
        default="config/meme.yaml",
        help="实验配置 YAML（默认 config/meme.yaml）",
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

    cfg = MemeExperimentConfig.from_yaml(config_path)
    stages = {s.strip() for s in args.stages.split(",")}

    os.chdir(_REPO_ROOT)

    _title(
        f"fact_memory MEME 4-Phase | benchmark={cfg.experiment.benchmark}"
        f" | suffix={cfg.experiment.suffix}"
        f" | methods={cfg.enabled_methods or ['(none)']}"
    )

    if "extract" in stages:
        stage_extract(cfg)
    if "run" in stages:
        stage_run(cfg)
    if "evaluate" in stages:
        stage_evaluate(cfg)

    _title("完成")
    print(f"  输出目录：{cfg.experiment_run_root}", flush=True)
    print(f"  评分文件：{cfg.experiment_run_root}/eval_meme_judge.json", flush=True)
    for method in cfg.enabled_methods:
        p = cfg.pred_file(method)
        status = "✓" if p.exists() else "✗"
        print(f"  {status} {method}: {p}", flush=True)


if __name__ == "__main__":
    main()
