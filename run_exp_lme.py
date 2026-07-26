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
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

# 确保可以从 src/ 导入 utils.config
_REPO_ROOT = Path(__file__).parent.resolve()
_SRC = _REPO_ROOT / "src"
sys.path.insert(0, str(_SRC))

from utils.config import ExperimentConfig  # noqa: E402
from utils.experiment_artifacts import ArtifactLayout, ExperimentIdentity  # noqa: E402


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


def _run_zep_with_restart(
    args: list, max_restarts: int = 600, *, trust_on_retry: bool = False
) -> None:
    """运行 zep 灌库；可仅在崩溃重试时附加 apply marker。"""
    env = {**os.environ, "PYTHONPATH": str(_SRC)}
    command_args = [str(a) for a in args]
    restarts = 0
    while True:
        cmd = [sys.executable, "-u"] + command_args
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
            if trust_on_retry and "--trust-apply-marker" not in command_args:
                command_args.append("--trust-apply-marker")
        else:
            print(f"灌库 zep 以非崩溃错误退出（exit {result.returncode}），终止。", file=sys.stderr)
            sys.exit(result.returncode)


def _title(text: str) -> None:
    print(f"\n\033[1;36m{'=' * 60}\033[0m")
    print(f"\033[1;36m  {text}\033[0m")
    print(f"\033[1;36m{'=' * 60}\033[0m", flush=True)


# ---------------------------------------------------------------------------
# 新 artifact 布局：ExperimentIdentity / ArtifactLayout 接线
# ---------------------------------------------------------------------------

def _build_layout(
    cfg: ExperimentConfig,
    config_path: Path,
    artifacts_root: Path,
    *,
    repo_root: Path = _REPO_ROOT,
    stage_nonce: Optional[str] = None,
) -> ArtifactLayout:
    """由 ``cfg`` + 配置文件路径推导出本次运行的 ``ArtifactLayout``。

    ``repo_root`` 默认取模块级 ``_REPO_ROOT``（真实仓库根），测试可传入
    ``tmp_path`` 隔离，避免污染真实仓库下的 artifacts 目录。

    ``stage_nonce``：仅在 ``replication.scope == "full_pipeline"`` 的重复变体下
    非空（通常传 ``variant.variant_id``），使该变体的 candidate/ingest 强制
    独立，不与其他变体共享同一份候选/灌库产物。
    """
    identity = ExperimentIdentity(
        resolved_config=cfg.model_dump(mode="json"),
        source_config_path=config_path,
        repo_root=repo_root,
        artifacts_root=Path(artifacts_root) / "runs",
    )
    return ArtifactLayout(
        identity=identity,
        template_root=_SRC / "prompts" / "templates",
        stages_root=Path(artifacts_root) / "stages",
        stage_nonce=stage_nonce,
    )


def _write_stages_json(cfg: ExperimentConfig, layout: ArtifactLayout) -> Path:
    """在 run_root 下写出 stages.json，记录 candidate/各 method 的 stage manifest，便于追溯 stage ID。"""
    layout.run_root.mkdir(parents=True, exist_ok=True)
    data = {
        "run_id": layout.identity.run_id,
        "candidate": layout.candidate_manifest(),
        "methods": {
            method: {
                "candidate": layout.candidate_manifest(),
                "ingest": layout.ingest_manifest(method),
                "answer": layout.answer_manifest(method),
                "judge": layout.judge_manifest(method),
            }
            for method in cfg.enabled_methods
        },
    }
    stages_path = layout.run_root / "stages.json"
    stages_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return stages_path


def _write_attempt_json(
    layout: ArtifactLayout, attempt_dir: Path, requested_stages: set[str], argv: list[str]
) -> Path:
    """为本次执行写入可复现的 attempt metadata（原子替换）。"""
    path = attempt_dir / "attempt.json"
    payload = {
        "schema_version": 1,
        "attempt_id": attempt_dir.name,
        "run_id": layout.identity.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "requested_stages": sorted(requested_stages),
        "argv": argv,
    }
    fd, temporary_path = tempfile.mkstemp(
        prefix=".attempt.json.", suffix=".tmp", dir=attempt_dir
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise
    return path


# ---------------------------------------------------------------------------
# 流水线各阶段
# ---------------------------------------------------------------------------

def stage_extract(cfg: ExperimentConfig, layout: Optional[ArtifactLayout] = None) -> None:
    output = layout.candidate_dir if layout is not None else cfg.candidates_dir
    _title(f"抽取候选记忆 → {output}")

    args = [
        _SRC / "pipeline" / "extract_candidates.py",
        "--benchmark", cfg.experiment.benchmark,
        "--output", output,
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

    if layout is None:
        _run(args)
    else:
        with layout.candidate_lock():
            _run(args)
            layout.write_candidate_stage_manifest()


def stage_ingest(
    cfg: ExperimentConfig,
    layout: Optional[ArtifactLayout] = None,
    methods: Optional[list] = None,
) -> None:
    """灌库；``methods`` 缺省时使用 ``cfg.enabled_methods``。

    ``methods`` 用于 variant 调度层在 ``replication.scope == "answer_judge"``
    下跳过已经在本次运行中灌过库的 ``(method, ingest_dir)`` 组合，只对尚未
    执行过的方法子集调用本函数。
    """
    if methods is None:
        methods = cfg.enabled_methods
    if not methods:
        print("ingest 跳过：没有 enabled 的方法", flush=True)
        return

    candidates_dir = layout.candidate_dir if layout is not None else cfg.candidates_dir

    ingest_shared = [
        "--benchmark", cfg.experiment.benchmark,
        "--candidate-extract-model", cfg._safe_tag(cfg.models.extract),
        "--candidate-suffix", cfg.extract.candidate_suffix,
        "--candidates-dir", candidates_dir,
        "--manager-model", cfg.models.manager,
        "--embedding-model", cfg.models.embedding,
        "--language", cfg.extract.language,
        "--relation-concurrency", cfg.parallel.ingest_relation_concurrency,
        "--relation-max-new-tokens", cfg.token_limits.ingest_relation_max_new_tokens,
        "--manager-max-new-tokens", cfg.token_limits.ingest_manager_max_new_tokens,
    ]
    if layout is None:
        # legacy：沿用旧语义，无条件校验/跳过均依赖 apply marker
        ingest_shared.append("--trust-apply-marker")
    # 新布局：不再通用信任 apply marker，改由 pipeline 的 fingerprint 校验兜底
    # （fingerprint 天然吸收 candidates_dir/manager 等配置变化）；仅 zep 崩溃可恢复
    # 续传时单独信任 apply marker（见下方 zep 分支）。

    for method_name in methods:
        db_root = layout.ingest_dir(method_name) if layout is not None else cfg.ingest_dir(method_name)
        _title(f"灌库 {method_name} → {db_root}")

        concurrency_cfg = cfg.parallel.ingest_episode_concurrency
        episode_concurrency = getattr(concurrency_cfg, method_name, 1)
        method_cfg = getattr(cfg.methods, method_name)

        base_args = [
            _SRC / "pipeline" / "ingest_candidates.py",
            "--update-method", method_name,
            "--database-root", db_root,
        ] + ingest_shared
        if layout is not None:
            base_args += ["--trace-log-dir", layout.ingest_dir(method_name) / "trace"]

        def run_method() -> None:
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
                extra += ["--relation-backend", getattr(method_cfg, "backend", "llm") or "llm"]
                active_relations = getattr(method_cfg, "active_relations", None)
                if active_relations:
                    extra += ["--active-relations", ",".join(active_relations)]
                fusion_enabled = getattr(method_cfg, "fusion_enabled", True)
                if not fusion_enabled:
                    extra.append("--no-fusion")
                # 保留用户已有 RD backend/threshold/fuse 参数传递。
                cond_th = getattr(method_cfg, "condition_sim_threshold", None)
                if cond_th is not None:
                    extra += ["--condition-sim-threshold", str(cond_th)]
                pair_th = getattr(method_cfg, "pairwise_sim_threshold", None)
                if pair_th is not None:
                    extra += ["--pairwise-sim-threshold", str(pair_th)]
                if p.relation_user_en:
                    extra += ["--relation-system-template-en", p.relation_user_en]
                if p.relation_user_zh:
                    extra += ["--relation-system-template-zh", p.relation_user_zh]
                _run(base_args + extra)

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
                zep_args = base_args + ["--zep-episode-concurrency", episode_concurrency]
                _run_zep_with_restart(zep_args, trust_on_retry=layout is not None)

            elif method_name == "evermemos":
                extra = [
                    "--evermemos-episode-concurrency", episode_concurrency,
                    "--evermemos-similarity-threshold", method_cfg.similarity_threshold,
                    "--evermemos-max-time-gap-days", method_cfg.max_time_gap_days,
                ]
                _run(base_args + extra)

        if layout is None:
            run_method()
        else:
            with layout.ingest_lock(method_name):
                run_method()
                layout.write_ingest_stage_manifest(method_name)


def stage_generate(cfg: ExperimentConfig, layout: Optional[ArtifactLayout] = None, benchmark_override: Optional[str] = None) -> None:
    methods = cfg.enabled_methods
    if not methods:
        print("generate 跳过：没有 enabled 的方法", flush=True)
        return

    g = cfg.generate
    # MEME：全量评测，不分层抽样
    benchmark = benchmark_override or cfg.experiment.benchmark
    sample = 0 if benchmark.lower().startswith("meme") else g.answer_stratified_sample

    common_args = [
        "--method", "prebuilt",
        "--benchmark", benchmark,
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

    if layout is None:
        cfg.experiment_run_root.mkdir(parents=True, exist_ok=True)

    for method_name in methods:
        # relation_decision：答题读灌库目录（同库），用 --answer-mode 只检索融合记忆 C + 孤立原子
        if layout is not None:
            db_root = layout.ingest_dir(method_name)
            output = layout.answer_dir(method_name) / "pred.jsonl"
            trace_dir = layout.answer_dir(method_name) / "agent_trace"
        else:
            db_root = cfg.ingest_dir(method_name)
            output = cfg.pred_file(method_name)
            trace_dir = cfg.experiment_run_root / "agent_trace" / method_name

        _title(f"生成预测: {method_name} → {output}")

        method_args = list(common_args)
        if method_name == "relation_decision":
            method_args.append("--answer-mode")

        _run([
            _SRC / "pipeline_lme_generate.py",
        ] + method_args + [
            "--database_root", db_root,
            "--output", output,
            "--agent_trace_dir", trace_dir,
        ])


def _evaluate_common_args(cfg: ExperimentConfig, benchmark_override: Optional[str] = None, judge_model_override: Optional[str] = None) -> list:
    e = cfg.evaluate
    p = cfg.prompts
    benchmark = benchmark_override or cfg.experiment.benchmark
    judge_model = judge_model_override or cfg.models.judge
    args = [
        "--judge_model", judge_model,
        "--benchmark", benchmark,
        "--max_concurrency", cfg.parallel.evaluate_max_concurrency,
        "--max_new_tokens", cfg.token_limits.evaluate_max_new_tokens,
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
    if p.judge_template:
        args += ["--judge-template", p.judge_template]
    return args


def stage_evaluate(cfg: ExperimentConfig, layout: Optional[ArtifactLayout] = None, benchmark_override: Optional[str] = None, judge_model_override: Optional[str] = None) -> None:
    if layout is not None:
        # 新布局：按 method 单独调用，各自独立的 --output/--metrics-output，绝不写回输入文件
        ran_any = False
        for method_name in cfg.enabled_methods:
            pred_path = layout.answer_dir(method_name) / "pred.jsonl"
            if not pred_path.exists():
                continue
            ran_any = True
            judge_dir = layout.judge_dir(method_name)
            _title(f"LLM Judge: {method_name} → {judge_dir}")

            args = [
                _SRC / "pipeline_lme_evaluate.py",
                "--input", pred_path,
            ] + _evaluate_common_args(cfg, benchmark_override, judge_model_override) + [
                "--output", judge_dir / "judged.jsonl",
                "--metrics-output", judge_dir / "metrics.json",
            ]
            _run(args)

        if not ran_any:
            print("Judge 跳过：未找到可用的预测 JSONL", flush=True)
        return

    # legacy：单次多输入调用，写回输入文件
    pred_files = [
        str(cfg.pred_file(m))
        for m in cfg.enabled_methods
        if cfg.pred_file(m).exists()
    ]
    if not pred_files:
        print("Judge 跳过：未找到可用的预测 JSONL", flush=True)
        return

    _title(f"LLM Judge ({len(pred_files)} 文件)")

    args = [
        _SRC / "pipeline_lme_evaluate.py",
        "--input", *pred_files,
    ] + _evaluate_common_args(cfg, benchmark_override, judge_model_override) + [
        "--write_back",
    ]
    _run(args)


# ---------------------------------------------------------------------------
# variant 调度：token-limit sweep x replication
# ---------------------------------------------------------------------------

_ALL_STAGES = ("extract", "ingest", "generate", "evaluate")


def _run_variants_new_layout(
    cfg: ExperimentConfig,
    config_path: Path,
    artifacts_root: Path,
    stages: set,
    argv: list,
    benchmark_override: Optional[str] = None,
    judge_model_override: Optional[str] = None,
) -> None:
    """新布局下按 ``cfg.experiment_variants()`` 逐个 variant 执行流水线。

    调度规则：
    - ``replication.scope == "answer_judge"``（默认）：variant 之间不带
      ``stage_nonce``，extract/ingest 天然按 candidate_dir / (method, ingest_dir)
      内容寻址去重——同一份 candidate_dir 只 extract 一次，同一个
      ``(method, ingest_dir)`` 只 ingest 一次；generate/evaluate 每个 variant
      都独立执行一次，各自落到自己的 answer/judge 目录。
    - ``replication.scope == "full_pipeline"``：每个 variant 的
      ``stage_nonce = variant.variant_id``，使其 candidate_dir/ingest_dir 与
      其他所有 variant（包括同一 token limit 的其他 repeat）都不同，因而
      extract/ingest 对每个 variant 都会各自执行一次。
    """
    variants = cfg.experiment_variants()
    extracted_candidate_dirs: set = set()
    ingested_keys: set = set()
    summary: list = []

    for variant in variants:
        variant_cfg = variant.config
        stage_nonce = (
            variant.variant_id
            if variant_cfg.replication.scope == "full_pipeline"
            else None
        )
        layout = _build_layout(
            variant_cfg, config_path, artifacts_root, stage_nonce=stage_nonce
        )
        layout.identity.materialize()
        attempt_dir = layout.new_attempt_dir()
        _write_attempt_json(layout, attempt_dir, stages, argv)
        _write_stages_json(variant_cfg, layout)

        _title(f"variant={variant.variant_id} | run_root={layout.run_root}")

        if "extract" in stages:
            if layout.candidate_dir in extracted_candidate_dirs:
                print(f"  extract 跳过（本次运行已完成）: {layout.candidate_dir}", flush=True)
            else:
                stage_extract(variant_cfg, layout)
                extracted_candidate_dirs.add(layout.candidate_dir)

        if "ingest" in stages:
            methods = variant_cfg.enabled_methods
            if not methods:
                print("ingest 跳过：没有 enabled 的方法", flush=True)
            else:
                pending = []
                for method in methods:
                    key = (method, layout.ingest_dir(method))
                    if key in ingested_keys:
                        print(
                            f"  ingest[{method}] 跳过（本次运行已完成）: "
                            f"{layout.ingest_dir(method)}",
                            flush=True,
                        )
                    else:
                        pending.append(method)
                        ingested_keys.add(key)
                if pending:
                    stage_ingest(variant_cfg, layout, methods=pending)

        if "generate" in stages:
            stage_generate(variant_cfg, layout, benchmark_override)
        if "evaluate" in stages:
            stage_evaluate(variant_cfg, layout, benchmark_override, judge_model_override)

        summary.append((variant.variant_id, variant_cfg, layout))

    _title("完成")
    for variant_id, variant_cfg, layout in summary:
        print(f"  variant={variant_id} run_root={layout.run_root}", flush=True)
        print(f"    candidate: {layout.candidate_dir}", flush=True)
        for method in variant_cfg.enabled_methods:
            print(
                f"    {method}: ingest={layout.ingest_dir(method)}"
                f" answer={layout.answer_dir(method)} judge={layout.judge_dir(method)}",
                flush=True,
            )


def _run_variants_legacy(cfg: ExperimentConfig, stages: set) -> None:
    """legacy 布局下按 ``cfg.experiment_variants()`` 逐个 variant 执行流水线。

    legacy 语义下没有内容寻址的共享 stage 目录，因此不做去重调度：每个
    variant 都各走一遍旧的 ``cfg.candidates_dir`` / ``cfg.ingest_dir`` /
    ``cfg.pred_file`` 路径（不同 token limit 天然落在不同的
    ``experiment_run_root`` 下）。
    """
    variants = cfg.experiment_variants()
    summary: list = []

    for variant in variants:
        variant_cfg = variant.config
        _title(f"variant={variant.variant_id} (legacy layout)")

        if "extract" in stages:
            stage_extract(variant_cfg, None)
        if "ingest" in stages:
            stage_ingest(variant_cfg, None)
        if "generate" in stages:
            stage_generate(variant_cfg, None)
        if "evaluate" in stages:
            stage_evaluate(variant_cfg, None)

        summary.append((variant.variant_id, variant_cfg))

    _title("完成")
    for variant_id, variant_cfg in summary:
        print(f"  variant={variant_id} 预测目录：{variant_cfg.experiment_run_root}", flush=True)
        for method in variant_cfg.enabled_methods:
            p = variant_cfg.pred_file(method)
            status = "✓" if p.exists() else "✗"
            print(f"    {status} {method}: {p}", flush=True)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


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
    parser.add_argument(
        "--artifacts-root",
        default="artifacts",
        help="新 artifact 布局的根目录（默认 artifacts；runs/stages 分别落在其下）",
    )
    parser.add_argument(
        "--legacy-layout",
        action="store_true",
        help="使用旧的 cfg.candidates_dir/ingest_dir/pred_file 路径语义，不物化新布局",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="覆盖 YAML 中的 experiment.benchmark（例如 --benchmark lme_s_golden 使用 470 题数据集）",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="覆盖 YAML 中的 models.judge（例如 --judge-model deepseek-v4-flash）",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        print(f"错误：找不到配置文件 {config_path}", file=sys.stderr)
        sys.exit(1)

    cfg = ExperimentConfig.from_yaml(config_path)
    benchmark_override = args.benchmark  # 仅用于 generate/evaluate，不改变 ingest 指纹
    judge_model_override = args.judge_model  # 仅用于 evaluate
    stages = {s.strip() for s in args.stages.split(",") if s.strip()}

    # 切到项目根目录（路径均为相对路径）
    os.chdir(_REPO_ROOT)

    _title(
        f"fact_memory | benchmark={cfg.experiment.benchmark} | suffix={cfg.experiment.suffix}"
        f" | methods={cfg.enabled_methods or ['(none)']}"
    )

    if args.legacy_layout:
        _run_variants_legacy(cfg, stages)
    else:
        _run_variants_new_layout(cfg, config_path, Path(args.artifacts_root), stages, sys.argv[1:], benchmark_override, judge_model_override)


if __name__ == "__main__":
    main()
