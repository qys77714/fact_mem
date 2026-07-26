"""run_exp_lme.py 新 artifact 布局接入的离线单元测试（TDD）。

覆盖范围：
1. ``_build_layout`` helper：tl256/tl512 得到不同 run/answer 路径，
   相同 candidate/ingest 路径。
2. ``stage_ingest`` 新布局：不同方法传不同 database root；
   非 zep 方法不含 ``--trust-apply-marker``，zep 含；trace 落在 ingest dir 下。
3. ``stage_generate`` 新布局：使用 answer 目录下的 pred.jsonl / agent_trace。
4. ``stage_evaluate`` 新布局：每个已有 pred 单独调用一次，
   携带 ``--output`` / ``--metrics-output``，绝不携带 ``--write_back``。
5. legacy 布局（``layout=None``）：stage_* 仍使用
   ``cfg.candidates_dir`` / ``cfg.ingest_dir`` / ``cfg.pred_file``，
   evaluate 仍是单次多输入调用 + ``--write_back``。

不发起任何模型/网络调用：只 monkeypatch ``run_exp_lme._run`` 与
``run_exp_lme._run_zep_with_restart``；不依赖已有数据库或模型服务。
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import run_exp_lme  # noqa: E402
from utils.config import ExperimentConfig  # noqa: E402
from utils.experiment_artifacts import ArtifactLayout  # noqa: E402


def _make_cfg(
    *,
    token_limit: int = 512,
    methods: Optional[dict] = None,
    benchmark: str = "lme_s",
    suffix: str = "exp001",
    sweep: Optional[dict] = None,
    replication: Optional[dict] = None,
) -> ExperimentConfig:
    if methods is None:
        methods = {"add_all": {"enabled": True}}
    data: dict[str, Any] = {
        "experiment": {"benchmark": benchmark, "suffix": suffix},
        "models": {
            "extract": "gemma4-26B",
            "manager": "gemma4-26B",
            "answer": "gemma4-26B",
            "judge": "qwen3-max",
            "embedding": "qwen3-embedding-0.6b",
        },
        "extract": {
            "candidate_suffix": "unit_test",
            "granularity": "4",
            "turn_overlap": "0",
            "language": "en",
            "aspect_templates": [],
        },
        "methods": methods,
        "generate": {"memory_token_limit": token_limit},
    }
    if sweep is not None:
        data["sweep"] = sweep
    if replication is not None:
        data["replication"] = replication
    return ExperimentConfig.model_validate(data)


def _write_config_yaml(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("placeholder: true\n", encoding="utf-8")
    return config_path


def _build_layout_for(cfg: ExperimentConfig, tmp_path: Path) -> ArtifactLayout:
    config_path = _write_config_yaml(tmp_path)
    return run_exp_lme._build_layout(
        cfg, config_path, tmp_path / "artifacts", repo_root=tmp_path
    )


def _record_run(monkeypatch: pytest.MonkeyPatch) -> list:
    calls: list = []
    monkeypatch.setattr(
        run_exp_lme, "_run", lambda args, **kw: calls.append([str(a) for a in args])
    )
    return calls


def _record_zep(monkeypatch: pytest.MonkeyPatch) -> list:
    calls: list = []
    monkeypatch.setattr(
        run_exp_lme,
        "_run_zep_with_restart",
        lambda args, **kw: calls.append([str(a) for a in args]),
    )
    return calls


# ---------------------------------------------------------------------------
# 1. _build_layout: tl256/tl512
# ---------------------------------------------------------------------------


class TestBuildLayout:
    def test_tl256_tl512_different_run_and_answer_same_candidate_and_ingest(
        self, tmp_path
    ):
        cfg256 = _make_cfg(token_limit=256, methods={"add_all": {"enabled": True}})
        cfg512 = _make_cfg(token_limit=512, methods={"add_all": {"enabled": True}})

        layout256 = _build_layout_for(cfg256, tmp_path)
        layout512 = _build_layout_for(cfg512, tmp_path)

        assert isinstance(layout256, ArtifactLayout)
        assert layout256.run_root != layout512.run_root
        assert layout256.candidate_dir == layout512.candidate_dir
        assert layout256.ingest_dir("add_all") == layout512.ingest_dir("add_all")
        assert layout256.answer_dir("add_all") != layout512.answer_dir("add_all")

    def test_build_layout_returns_artifact_layout_rooted_under_artifacts_root(
        self, tmp_path
    ):
        cfg = _make_cfg()
        config_path = _write_config_yaml(tmp_path)
        layout = run_exp_lme._build_layout(
            cfg, config_path, tmp_path / "myartifacts", repo_root=tmp_path
        )
        assert layout.run_root.is_relative_to(tmp_path / "myartifacts" / "runs")
        assert layout.candidate_dir.is_relative_to(tmp_path / "myartifacts" / "stages")


# ---------------------------------------------------------------------------
# 2. stage_ingest 新布局
# ---------------------------------------------------------------------------


class TestStageIngestNewLayout:
    def test_different_methods_get_different_db_roots_and_without_trust_marker(
        self, tmp_path, monkeypatch
    ):
        cfg = _make_cfg(
            methods={"add_all": {"enabled": True}, "zep": {"enabled": True}}
        )
        layout = _build_layout_for(cfg, tmp_path)

        run_calls = _record_run(monkeypatch)
        zep_calls = _record_zep(monkeypatch)

        run_exp_lme.stage_ingest(cfg, layout)

        assert len(run_calls) == 1
        assert len(zep_calls) == 1
        add_all_args = run_calls[0]
        zep_args = zep_calls[0]

        add_all_db_root = str(layout.ingest_dir("add_all"))
        zep_db_root = str(layout.ingest_dir("zep"))
        assert add_all_db_root != zep_db_root
        assert add_all_db_root in add_all_args
        assert zep_db_root in zep_args

        assert "--trust-apply-marker" not in add_all_args
        assert "--trust-apply-marker" not in zep_args

        assert str(layout.ingest_dir("add_all") / "trace") in add_all_args
        assert str(layout.ingest_dir("zep") / "trace") in zep_args

        # 灌库共用参数使用 layout.candidate_dir，而不是 cfg.candidates_dir
        assert str(layout.candidate_dir) in add_all_args
        assert str(layout.candidate_dir) in zep_args
        assert str(cfg.candidates_dir) not in add_all_args

    def test_relation_decision_llm_backend_threshold_fuse_args_preserved(
        self, tmp_path, monkeypatch
    ):
        """确保新布局接入不影响用户未提交的 RD backend/threshold/fuse 参数传递。"""
        cfg = _make_cfg(
            methods={
                "relation_decision": {
                    "enabled": True,
                    "backend": "llm",
                    "condition_sim_threshold": 0.42,
                    "pairwise_sim_threshold": 0.77,
                    "fusion_enabled": False,
                }
            }
        )
        layout = _build_layout_for(cfg, tmp_path)
        run_calls = _record_run(monkeypatch)

        run_exp_lme.stage_ingest(cfg, layout)

        assert len(run_calls) == 1
        args = run_calls[0]
        assert "--relation-backend" in args
        assert args[args.index("--relation-backend") + 1] == "llm"
        assert "--condition-sim-threshold" in args
        assert args[args.index("--condition-sim-threshold") + 1] == "0.42"
        assert "--pairwise-sim-threshold" in args
        assert args[args.index("--pairwise-sim-threshold") + 1] == "0.77"
        assert "--no-fusion" in args
        assert "--fuse-template-version" in args
        assert args[args.index("--fuse-template-version") + 1] == "_v2"


# ---------------------------------------------------------------------------
# 3. stage_generate 新布局
# ---------------------------------------------------------------------------


class TestStageGenerateNewLayout:
    def test_uses_answer_dir_for_pred_and_trace(self, tmp_path, monkeypatch):
        cfg = _make_cfg(methods={"add_all": {"enabled": True}})
        layout = _build_layout_for(cfg, tmp_path)

        run_calls = _record_run(monkeypatch)
        run_exp_lme.stage_generate(cfg, layout)

        assert len(run_calls) == 1
        args = run_calls[0]

        expected_db_root = str(layout.ingest_dir("add_all"))
        expected_output = str(layout.answer_dir("add_all") / "pred.jsonl")
        expected_trace = str(layout.answer_dir("add_all") / "agent_trace")

        assert expected_db_root in args
        assert expected_output in args
        assert expected_trace in args
        assert str(cfg.pred_file("add_all")) not in args


# ---------------------------------------------------------------------------
# 4. stage_evaluate 新布局
# ---------------------------------------------------------------------------


class TestStageEvaluateNewLayout:
    def test_calls_once_per_method_with_output_metrics_no_write_back(
        self, tmp_path, monkeypatch
    ):
        cfg = _make_cfg(
            methods={"add_all": {"enabled": True}, "zep": {"enabled": True}}
        )
        layout = _build_layout_for(cfg, tmp_path)

        for method in ("add_all", "zep"):
            pred_path = layout.answer_dir(method) / "pred.jsonl"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            pred_path.write_text("", encoding="utf-8")

        run_calls = _record_run(monkeypatch)
        run_exp_lme.stage_evaluate(cfg, layout)

        assert len(run_calls) == 2
        for method in ("add_all", "zep"):
            pred_path = str(layout.answer_dir(method) / "pred.jsonl")
            judged_path = str(layout.judge_dir(method) / "judged.jsonl")
            metrics_path = str(layout.judge_dir(method) / "metrics.json")

            matching = [c for c in run_calls if pred_path in c]
            assert len(matching) == 1
            call = matching[0]
            assert judged_path in call
            assert metrics_path in call
            assert "--write_back" not in call
            assert "--output" in call
            assert "--metrics-output" in call

    def test_skips_methods_without_pred_file(self, tmp_path, monkeypatch):
        cfg = _make_cfg(methods={"add_all": {"enabled": True}})
        layout = _build_layout_for(cfg, tmp_path)

        run_calls = _record_run(monkeypatch)
        run_exp_lme.stage_evaluate(cfg, layout)
        assert run_calls == []


# ---------------------------------------------------------------------------
# 5. legacy 布局（layout=None）保持原有行为
# ---------------------------------------------------------------------------


class TestLegacyLayoutUnchanged:
    def test_stage_extract_legacy_uses_cfg_candidates_dir(self, monkeypatch):
        cfg = _make_cfg()
        run_calls = _record_run(monkeypatch)

        run_exp_lme.stage_extract(cfg, None)

        assert len(run_calls) == 1
        assert str(cfg.candidates_dir) in run_calls[0]

    def test_stage_ingest_legacy_uses_cfg_paths_and_trust_marker_for_all(
        self, monkeypatch
    ):
        cfg = _make_cfg(
            methods={"add_all": {"enabled": True}, "zep": {"enabled": True}}
        )
        run_calls = _record_run(monkeypatch)
        zep_calls = _record_zep(monkeypatch)

        run_exp_lme.stage_ingest(cfg, None)

        assert str(cfg.candidates_dir) in run_calls[0]
        assert str(cfg.ingest_dir("add_all")) in run_calls[0]
        assert "--trust-apply-marker" in run_calls[0]

        assert str(cfg.ingest_dir("zep")) in zep_calls[0]
        assert "--trust-apply-marker" in zep_calls[0]

    def test_stage_generate_legacy_uses_cfg_pred_file(self, tmp_path, monkeypatch):
        cfg = _make_cfg(methods={"add_all": {"enabled": True}})
        monkeypatch.chdir(tmp_path)
        run_calls = _record_run(monkeypatch)

        run_exp_lme.stage_generate(cfg, None)

        assert len(run_calls) == 1
        assert str(cfg.pred_file("add_all")) in run_calls[0]
        assert str(cfg.ingest_dir("add_all")) in run_calls[0]

    def test_stage_evaluate_legacy_single_call_with_write_back(
        self, tmp_path, monkeypatch
    ):
        cfg = _make_cfg(
            methods={"add_all": {"enabled": True}, "zep": {"enabled": True}}
        )
        monkeypatch.chdir(tmp_path)
        for method in ("add_all", "zep"):
            pred_path = cfg.pred_file(method)
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            pred_path.write_text("", encoding="utf-8")

        run_calls = _record_run(monkeypatch)
        run_exp_lme.stage_evaluate(cfg, None)

        assert len(run_calls) == 1
        call = run_calls[0]
        assert "--write_back" in call
        assert "--output" not in call
        assert "--metrics-output" not in call
        assert str(cfg.pred_file("add_all")) in call
        assert str(cfg.pred_file("zep")) in call


# ---------------------------------------------------------------------------
# 6. shared-stage lock/manifest + main safety
# ---------------------------------------------------------------------------


class TestSharedStageSafety:
    def test_relation_decision_yaml_rejects_unsupported_classifier_backend(self, tmp_path):
        config_path = tmp_path / "invalid-backend.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "methods": {
                        "relation_decision": {"enabled": True, "backend": "classifier"}
                    }
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="backend"):
            ExperimentConfig.from_yaml(config_path)

    def test_stage_extract_runner_executes_while_candidate_lock_is_held(
        self, tmp_path, monkeypatch
    ):
        cfg = _make_cfg()
        layout = _build_layout_for(cfg, tmp_path)
        entered = False

        class Lock:
            def __enter__(self):
                nonlocal entered
                entered = True

            def __exit__(self, *_):
                nonlocal entered
                entered = False

        monkeypatch.setattr(ArtifactLayout, "candidate_lock", lambda self: Lock(), raising=False)
        calls = []
        monkeypatch.setattr(
            run_exp_lme, "_run", lambda args, **kw: calls.append(entered)
        )

        run_exp_lme.stage_extract(cfg, layout)

        assert calls == [True]

    def test_stage_ingest_runner_executes_while_method_lock_is_held(
        self, tmp_path, monkeypatch
    ):
        cfg = _make_cfg()
        layout = _build_layout_for(cfg, tmp_path)
        entered = False

        class Lock:
            def __enter__(self):
                nonlocal entered
                entered = True

            def __exit__(self, *_):
                nonlocal entered
                entered = False

        monkeypatch.setattr(
            ArtifactLayout, "ingest_lock", lambda self, method: Lock(), raising=False
        )
        calls = []
        monkeypatch.setattr(
            run_exp_lme, "_run", lambda args, **kw: calls.append(entered)
        )

        run_exp_lme.stage_ingest(cfg, layout)

        assert calls == [True]

    def test_zep_retries_add_trust_marker_only_after_crash(self, monkeypatch):
        seen_commands = []
        returncodes = iter((139, 0))

        def fake_run(cmd, **kwargs):
            seen_commands.append(cmd)
            return subprocess.CompletedProcess(cmd, next(returncodes))

        monkeypatch.setattr(run_exp_lme.subprocess, "run", fake_run)

        run_exp_lme._run_zep_with_restart(["fake.py"], trust_on_retry=True)

        assert "--trust-apply-marker" not in seen_commands[0]
        assert "--trust-apply-marker" in seen_commands[1]

    def test_zep_legacy_retry_does_not_change_supplied_args(self, monkeypatch):
        seen_commands = []
        returncodes = iter((139, 0))

        def fake_run(cmd, **kwargs):
            seen_commands.append(cmd)
            return subprocess.CompletedProcess(cmd, next(returncodes))

        monkeypatch.setattr(run_exp_lme.subprocess, "run", fake_run)

        run_exp_lme._run_zep_with_restart(["fake.py", "--trust-apply-marker"])

        assert seen_commands[0] == seen_commands[1]

    def test_main_uses_absolute_config_path_before_chdir_and_writes_attempt(
        self, tmp_path, monkeypatch
    ):
        cfg = _make_cfg()
        config_path = tmp_path / "relative-config.yaml"
        config_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")
        artifacts_root = tmp_path / "artifacts"
        captured = {}
        original_build_layout = run_exp_lme._build_layout

        def capture_layout(cfg_arg, config_arg, artifacts_arg, **kwargs):
            captured["config_path"] = config_arg
            return original_build_layout(cfg_arg, config_arg, artifacts_arg, **kwargs)

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(run_exp_lme, "_build_layout", capture_layout)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_exp_lme.py",
                "--config",
                "relative-config.yaml",
                "--artifacts-root",
                str(artifacts_root),
                "--stages",
                "",
            ],
        )

        run_exp_lme.main()

        assert captured["config_path"] == config_path.resolve()
        attempt_files = list((artifacts_root / "runs").glob("*/attempts/*/attempt.json"))
        assert len(attempt_files) == 1
        attempt = json.loads(attempt_files[0].read_text(encoding="utf-8"))
        assert attempt["attempt_id"] == attempt_files[0].parent.name
        assert attempt["requested_stages"] == []
        assert attempt["argv"] == sys.argv[1:]


# ---------------------------------------------------------------------------
# 7. main() x cfg.experiment_variants()：token-limit sweep 与统计重复接线
# ---------------------------------------------------------------------------


def _arg_value(call: list, flag: str) -> str:
    return call[call.index(flag) + 1]


def _calls_for(calls: list, script_name: str) -> list:
    return [c for c in calls if any(script_name in a for a in c)]


def _run_main(
    cfg: ExperimentConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    stages: str = "extract,ingest,generate,evaluate",
    legacy: bool = False,
) -> list:
    """通过 ``run_exp_lme.main()`` 跑一遍完整入口，只 mock 掉真正的子进程调用。

    ``_run`` 的 mock 会在遇到 generate 阶段调用时把 ``--output`` 对应的
    pred.jsonl 落一个空文件，好让紧跟着的 evaluate 阶段能找到输入文件。
    额外把 ``run_exp_lme._REPO_ROOT`` 换成 ``tmp_path``，避免 legacy 布局下
    对 ``cfg.experiment_run_root`` 等相对路径的 mkdir 落到真实仓库里。
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8"
    )
    artifacts_root = tmp_path / "artifacts"

    calls: list = []

    def fake_run(args, **kw):
        str_args = [str(a) for a in args]
        calls.append(str_args)
        if any("pipeline_lme_generate.py" in a for a in str_args):
            output = Path(_arg_value(str_args, "--output"))
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("", encoding="utf-8")

    def fake_zep_run(args, **kw):
        calls.append([str(a) for a in args])

    monkeypatch.setattr(run_exp_lme, "_run", fake_run)
    monkeypatch.setattr(run_exp_lme, "_run_zep_with_restart", fake_zep_run)
    monkeypatch.setattr(run_exp_lme, "_REPO_ROOT", tmp_path)

    argv = [
        "run_exp_lme.py",
        "--config", str(config_path),
        "--artifacts-root", str(artifacts_root),
        "--stages", stages,
    ]
    if legacy:
        argv.append("--legacy-layout")
    monkeypatch.setattr(sys, "argv", argv)

    run_exp_lme.main()
    return calls


class TestMainExperimentVariants:
    def test_sweep_answer_judge_dedupes_extract_and_ingest_but_not_generate_evaluate(
        self, tmp_path, monkeypatch
    ):
        cfg = _make_cfg(
            methods={"add_all": {"enabled": True}},
            sweep={"memory_token_limits": [256, 512]},
            replication={"count": 2, "scope": "answer_judge"},
        )
        calls = _run_main(cfg, tmp_path, monkeypatch)

        extract_calls = _calls_for(calls, "extract_candidates.py")
        ingest_calls = _calls_for(calls, "ingest_candidates.py")
        generate_calls = _calls_for(calls, "pipeline_lme_generate.py")
        evaluate_calls = _calls_for(calls, "pipeline_lme_evaluate.py")

        assert len(extract_calls) == 1
        assert len(ingest_calls) == 1
        assert len(generate_calls) == 4
        assert len(evaluate_calls) == 4

        answer_paths = {_arg_value(c, "--output") for c in generate_calls}
        assert len(answer_paths) == 4

        # 256 与 512 两个 token limit 必须复用同一个 add_all ingest 库。
        db_roots = {_arg_value(c, "--database_root") for c in generate_calls}
        assert len(db_roots) == 1
        assert db_roots == {_arg_value(ingest_calls[0], "--database-root")}

    def test_full_pipeline_repeats_run_extract_and_ingest_once_per_repeat(
        self, tmp_path, monkeypatch
    ):
        cfg = _make_cfg(
            methods={"add_all": {"enabled": True}},
            replication={"count": 2, "scope": "full_pipeline"},
        )
        calls = _run_main(cfg, tmp_path, monkeypatch)

        extract_calls = _calls_for(calls, "extract_candidates.py")
        ingest_calls = _calls_for(calls, "ingest_candidates.py")
        generate_calls = _calls_for(calls, "pipeline_lme_generate.py")
        evaluate_calls = _calls_for(calls, "pipeline_lme_evaluate.py")

        assert len(extract_calls) == 2
        assert len(ingest_calls) == 2
        assert len(generate_calls) == 2
        assert len(evaluate_calls) == 2

        extract_outputs = {_arg_value(c, "--output") for c in extract_calls}
        ingest_roots = {_arg_value(c, "--database-root") for c in ingest_calls}
        assert len(extract_outputs) == 2
        assert len(ingest_roots) == 2

    def test_stages_subset_still_applies_across_variants(self, tmp_path, monkeypatch):
        cfg = _make_cfg(
            methods={"add_all": {"enabled": True}},
            sweep={"memory_token_limits": [256, 512]},
        )
        calls = _run_main(cfg, tmp_path, monkeypatch, stages="ingest")

        assert len(_calls_for(calls, "ingest_candidates.py")) == 1
        assert _calls_for(calls, "extract_candidates.py") == []
        assert _calls_for(calls, "pipeline_lme_generate.py") == []
        assert _calls_for(calls, "pipeline_lme_evaluate.py") == []

    def test_no_enabled_methods_skips_ingest_and_generate_safely(
        self, tmp_path, monkeypatch
    ):
        cfg = _make_cfg(
            methods={"add_all": {"enabled": False}},
            sweep={"memory_token_limits": [256, 512]},
        )
        calls = _run_main(cfg, tmp_path, monkeypatch)

        assert _calls_for(calls, "ingest_candidates.py") == []
        assert _calls_for(calls, "pipeline_lme_generate.py") == []
        assert _calls_for(calls, "pipeline_lme_evaluate.py") == []

    def test_legacy_layout_sweep_does_not_raise_and_uses_distinct_pred_paths(
        self, tmp_path, monkeypatch
    ):
        cfg = _make_cfg(
            methods={"add_all": {"enabled": True}},
            sweep={"memory_token_limits": [256, 512]},
        )
        calls = _run_main(cfg, tmp_path, monkeypatch, legacy=True)

        generate_calls = _calls_for(calls, "pipeline_lme_generate.py")
        assert len(generate_calls) == 2

        pred_paths = {_arg_value(c, "--output") for c in generate_calls}
        assert len(pred_paths) == 2
