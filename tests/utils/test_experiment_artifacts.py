"""Tests for src/utils/experiment_artifacts.py (实验身份与阶段指纹).

TDD：这些测试先于实现编写，应在实现落地前全部失败（ImportError）。
"""

from __future__ import annotations

import copy
import json
import math
import re
import threading
from pathlib import Path

import pytest
import yaml

from utils.experiment_artifacts import (
    ArtifactLayout,
    ExperimentIdentity,
    answer_fingerprint,
    candidate_fingerprint,
    canonical_json,
    ingest_fingerprint,
    judge_fingerprint,
    sha256_hash,
    short_hash,
)


def _base_config() -> dict:
    """一个具有代表性的嵌套配置，形状对齐 config/lme.yaml。"""
    return {
        "experiment": {"benchmark": "lme_s", "suffix": "exp001"},
        "models": {
            "extract": "gemma4-26B",
            "manager": "gemma4-26B",
            "answer": "gemma4-26B",
            "judge": "qwen3-max",
            "embedding": "qwen3-embedding-0.6b",
        },
        "extract": {
            "candidate_suffix": "0615_unified",
            "granularity": "4",
            "turn_overlap": "0",
            "language": "en",
            "aspect_templates": ["0_mem_extract_aspect_unified_en.jinja"],
        },
        "methods": {
            "relation_decision": {
                "enabled": True,
                "related_top_k": 3,
                "backend": "classifier",
                "fusion_model": "",
                "condition_sim_threshold": 0.5,
                "pairwise_sim_threshold": 0.5,
                "fusion_enabled": True,
                "active_relations": None,
            },
            "add_all": {"enabled": False},
        },
        "generate": {
            "retrieve_topk": 50,
            "memory_token_limit": 512,
            "answer_stratified_sample": 500,
            "answer_sample_seed": 43,
            "show_memory_time": True,
            "hybrid": {"enabled": True, "dense_weight": 0.8, "bm25_weight": 0.2, "pool_mult": 4},
        },
        "evaluate": {"use_cot": True, "judge_stratified_sample": 0, "judge_sample_seed": 43},
        "token_limits": {
            "extract_max_new_tokens": 2048,
            "ingest_relation_max_new_tokens": 256,
            "ingest_manager_max_new_tokens": 2048,
            "fusion_max_new_tokens": 512,
            "evaluate_max_new_tokens": 512,
        },
        "prompts": {
            "relation_system_en": "RD_0_relation_classify.jinja",
            "relation_system_zh": "RD_0_relation_classify.jinja",
            "judge_oqa": "pipeline_eval_oqa.jinja",
            "judge_mcq": "pipeline_eval_mcq.jinja",
            "judge_system": "pipeline_eval_system.jinja",
        },
    }


# ---------------------------------------------------------------------------
# 1. canonical JSON / sha256 hash 基础语义
# ---------------------------------------------------------------------------


class TestCanonicalHash:
    def test_dict_key_order_does_not_change_hash(self):
        a = {"a": 1, "b": {"x": 1, "y": 2}, "c": [1, 2, 3]}
        b = {"c": [1, 2, 3], "b": {"y": 2, "x": 1}, "a": 1}
        assert canonical_json(a) == canonical_json(b)
        assert sha256_hash(a) == sha256_hash(b)

    def test_different_content_changes_hash(self):
        a = {"a": 1}
        b = {"a": 2}
        assert sha256_hash(a) != sha256_hash(b)

    def test_canonical_json_is_valid_json_and_deterministic(self):
        cfg = _base_config()
        text1 = canonical_json(cfg)
        text2 = canonical_json(copy.deepcopy(cfg))
        assert text1 == text2
        # round-trips as JSON
        parsed = json.loads(text1)
        assert parsed["experiment"]["benchmark"] == "lme_s"

    def test_path_is_normalized_to_string(self):
        a = {"p": Path("a") / "b" / "c"}
        b = {"p": "a/b/c"}
        assert canonical_json(a) == canonical_json(b)

    def test_tuple_normalized_like_list_but_order_preserved(self):
        a = {"t": (1, 2, 3)}
        b = {"t": [1, 2, 3]}
        c = {"t": (3, 2, 1)}
        assert canonical_json(a) == canonical_json(b)
        assert canonical_json(a) != canonical_json(c)

    def test_set_is_sorted_regardless_of_insertion_order(self):
        a = {"s": set()}
        a["s"].update([3, 1, 2])
        b = {"s": set()}
        b["s"].update([2, 3, 1])
        assert canonical_json(a) == canonical_json(b)
        # sorted representation should be [1, 2, 3]
        assert json.loads(canonical_json(a))["s"] == [1, 2, 3]

    def test_short_hash_length_and_determinism(self):
        cfg = _base_config()
        h1 = short_hash(cfg, length=8)
        h2 = short_hash(copy.deepcopy(cfg), length=8)
        assert h1 == h2
        assert len(h1) == 8
        assert re.fullmatch(r"[0-9a-f]{8}", h1)

        h_full = sha256_hash(cfg)
        assert len(h_full) == 64
        assert h_full.startswith(h1)

    def test_nan_is_rejected(self):
        with pytest.raises(ValueError):
            canonical_json({"score": math.nan})


# ---------------------------------------------------------------------------
# 2. ExperimentIdentity: slug / run_id / run_root
# ---------------------------------------------------------------------------


class TestExperimentIdentity:
    def test_slug_is_readable_and_path_safe(self, tmp_path):
        identity = ExperimentIdentity(
            resolved_config=_base_config(),
            source_config_path=None,
            repo_root=tmp_path,
        )
        slug = identity.slug
        assert slug
        # 路径安全：不含斜杠、空格、反斜杠
        assert "/" not in slug and "\\" not in slug and " " not in slug
        # 可读：包含 benchmark 关键片段
        assert "lme_s" in slug or "lme-s" in slug

    def test_run_id_format_slug_dash_dash_hash8(self, tmp_path):
        identity = ExperimentIdentity(
            resolved_config=_base_config(),
            source_config_path=None,
            repo_root=tmp_path,
        )
        run_id = identity.run_id
        m = re.fullmatch(r"(?P<slug>.+)--(?P<hash>[0-9a-f]{8})", run_id)
        assert m is not None
        assert m.group("slug") == identity.slug
        assert m.group("hash") == short_hash(_base_config(), length=8)

    def test_run_root_under_artifacts_root_relative_to_repo_root(self, tmp_path):
        identity = ExperimentIdentity(
            resolved_config=_base_config(),
            source_config_path=None,
            repo_root=tmp_path,
            artifacts_root=Path("artifacts/runs"),
        )
        expected = tmp_path / "artifacts" / "runs" / identity.run_id
        assert identity.run_root == expected

    def test_different_config_gives_different_run_id(self, tmp_path):
        cfg1 = _base_config()
        cfg2 = _base_config()
        cfg2["models"]["judge"] = "gpt-4o-mini"
        id1 = ExperimentIdentity(resolved_config=cfg1, source_config_path=None, repo_root=tmp_path)
        id2 = ExperimentIdentity(resolved_config=cfg2, source_config_path=None, repo_root=tmp_path)
        assert id1.run_id != id2.run_id

    def test_constructor_defensively_copies_resolved_config(self, tmp_path):
        cfg = _base_config()
        identity = ExperimentIdentity(resolved_config=cfg, source_config_path=None, repo_root=tmp_path)
        original_run_id = identity.run_id

        cfg["models"]["judge"] = "mutated-after-construction"
        cfg["methods"]["relation_decision"]["enabled"] = False

        assert identity.run_id == original_run_id


# ---------------------------------------------------------------------------
# 3. 阶段指纹：最小依赖
# ---------------------------------------------------------------------------


class TestStageFingerprints:
    def test_embedding_changes_ingest_and_answer_but_not_candidate_or_judge_only(self):
        cfg1 = _base_config()
        cfg2 = _base_config()
        cfg2["models"]["embedding"] = "qwen3-embedding-8b"
        method = "relation_decision"

        assert candidate_fingerprint(cfg1) == candidate_fingerprint(cfg2)
        ingest_id1 = ingest_fingerprint(cfg1, method)
        ingest_id2 = ingest_fingerprint(cfg2, method)
        assert ingest_id1 != ingest_id2

        answer_id1 = answer_fingerprint(cfg1, method, ingest_id1)
        answer_id2 = answer_fingerprint(cfg2, method, ingest_id2)
        assert answer_id1 != answer_id2
        assert judge_fingerprint(cfg1, method, answer_id1) != judge_fingerprint(
            cfg2, method, answer_id2
        )

    def test_candidate_template_content_changes_candidate_fingerprint(self, tmp_path):
        cfg = _base_config()
        templates = tmp_path / "templates"
        templates.mkdir()
        template = templates / cfg["extract"]["aspect_templates"][0]
        template.write_text("extract version one", encoding="utf-8")
        first = candidate_fingerprint(cfg, template_root=templates)

        template.write_text("extract version two", encoding="utf-8")
        assert candidate_fingerprint(cfg, template_root=templates) != first

    def test_ingest_template_content_changes_ingest_fingerprint(self, tmp_path):
        cfg = _base_config()
        templates = tmp_path / "templates"
        templates.mkdir()
        template = templates / cfg["prompts"]["relation_system_en"]
        template.write_text("relation version one", encoding="utf-8")
        first = ingest_fingerprint(cfg, "relation_decision", template_root=templates)

        template.write_text("relation version two", encoding="utf-8")
        assert ingest_fingerprint(cfg, "relation_decision", template_root=templates) != first

    def test_judge_template_content_changes_judge_fingerprint(self, tmp_path):
        cfg = _base_config()
        templates = tmp_path / "templates"
        templates.mkdir()
        template = templates / cfg["prompts"]["judge_oqa"]
        template.write_text("judge version one", encoding="utf-8")
        first = judge_fingerprint(
            cfg, "relation_decision", "answer123", template_root=templates
        )

        template.write_text("judge version two", encoding="utf-8")
        assert judge_fingerprint(
            cfg, "relation_decision", "answer123", template_root=templates
        ) != first

    def test_missing_template_has_stable_explicit_fingerprint_marker(self, tmp_path):
        cfg = _base_config()
        cfg["extract"]["aspect_templates"] = ["does-not-exist.jinja"]
        first = candidate_fingerprint(cfg, template_root=tmp_path)
        second = candidate_fingerprint(copy.deepcopy(cfg), template_root=tmp_path)
        assert first == second

    def test_token_limit_and_topk_do_not_affect_candidate_or_ingest(self):
        cfg1 = _base_config()
        cfg2 = _base_config()
        cfg2["generate"]["memory_token_limit"] = 256
        cfg2["generate"]["retrieve_topk"] = 20

        assert candidate_fingerprint(cfg1) == candidate_fingerprint(cfg2)
        assert ingest_fingerprint(cfg1, "relation_decision") == ingest_fingerprint(
            cfg2, "relation_decision"
        )

    def test_token_limit_changes_answer_fingerprint(self):
        cfg1 = _base_config()
        cfg2 = _base_config()
        cfg2["generate"]["memory_token_limit"] = 256

        ingest_id = ingest_fingerprint(cfg1, "relation_decision")
        a1 = answer_fingerprint(cfg1, "relation_decision", ingest_id)
        a2 = answer_fingerprint(cfg2, "relation_decision", ingest_id)
        assert a1 != a2

    def test_retrieve_topk_changes_answer_fingerprint(self):
        cfg1 = _base_config()
        cfg2 = _base_config()
        cfg2["generate"]["retrieve_topk"] = 5

        ingest_id = ingest_fingerprint(cfg1, "relation_decision")
        a1 = answer_fingerprint(cfg1, "relation_decision", ingest_id)
        a2 = answer_fingerprint(cfg2, "relation_decision", ingest_id)
        assert a1 != a2

    def test_relation_decision_threshold_changes_ingest_but_not_candidate(self):
        cfg1 = _base_config()
        cfg2 = _base_config()
        cfg2["methods"]["relation_decision"]["condition_sim_threshold"] = 0.9

        assert candidate_fingerprint(cfg1) == candidate_fingerprint(cfg2)
        assert ingest_fingerprint(cfg1, "relation_decision") != ingest_fingerprint(
            cfg2, "relation_decision"
        )

    def test_relation_decision_pairwise_threshold_changes_ingest(self):
        cfg1 = _base_config()
        cfg2 = _base_config()
        cfg2["methods"]["relation_decision"]["pairwise_sim_threshold"] = 0.9

        assert ingest_fingerprint(cfg1, "relation_decision") != ingest_fingerprint(
            cfg2, "relation_decision"
        )

    def test_threshold_change_does_not_affect_other_method_ingest(self):
        cfg1 = _base_config()
        cfg2 = _base_config()
        cfg2["methods"]["relation_decision"]["condition_sim_threshold"] = 0.9

        # add_all 的 ingest 指纹不应受 relation_decision 阈值变化影响
        assert ingest_fingerprint(cfg1, "add_all") == ingest_fingerprint(cfg2, "add_all")

    def test_judge_model_change_only_affects_judge(self):
        cfg1 = _base_config()
        cfg2 = _base_config()
        cfg2["models"]["judge"] = "gpt-4o-mini"

        method = "relation_decision"
        assert candidate_fingerprint(cfg1) == candidate_fingerprint(cfg2)

        ingest_id1 = ingest_fingerprint(cfg1, method)
        ingest_id2 = ingest_fingerprint(cfg2, method)
        assert ingest_id1 == ingest_id2

        answer_id1 = answer_fingerprint(cfg1, method, ingest_id1)
        answer_id2 = answer_fingerprint(cfg2, method, ingest_id2)
        assert answer_id1 == answer_id2

        judge_id1 = judge_fingerprint(cfg1, method, answer_id1)
        judge_id2 = judge_fingerprint(cfg2, method, answer_id2)
        assert judge_id1 != judge_id2

    def test_judge_fingerprint_changes_when_answer_id_changes(self):
        cfg = _base_config()
        j1 = judge_fingerprint(cfg, "relation_decision", "aaaaaaaa")
        j2 = judge_fingerprint(cfg, "relation_decision", "bbbbbbbb")
        assert j1 != j2

    def test_missing_optional_fields_use_stable_defaults(self):
        """缺失字段时应使用稳定默认值，而不是抛异常或每次不同。"""
        minimal = {"experiment": {"benchmark": "lme_s"}}
        f1 = candidate_fingerprint(minimal)
        f2 = candidate_fingerprint(copy.deepcopy(minimal))
        assert f1 == f2

        i1 = ingest_fingerprint(minimal, "relation_decision")
        i2 = ingest_fingerprint(copy.deepcopy(minimal), "relation_decision")
        assert i1 == i2


# ---------------------------------------------------------------------------
# 4. materialize(): 写入 run.yaml / manifest.json，幂等 & 冲突保护
# ---------------------------------------------------------------------------


class TestMaterialize:
    def test_materialize_creates_two_loadable_files(self, tmp_path):
        cfg = _base_config()
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        identity = ExperimentIdentity(
            resolved_config=cfg,
            source_config_path=cfg_path,
            repo_root=tmp_path,
        )
        result = identity.materialize()

        manifest_path = identity.run_root / "manifest.json"
        run_yaml_path = identity.run_root / "run.yaml"
        assert manifest_path.exists()
        assert run_yaml_path.exists()

        manifest_from_json = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_from_yaml = yaml.safe_load(run_yaml_path.read_text(encoding="utf-8"))

        for manifest in (manifest_from_json, manifest_from_yaml):
            assert manifest["run_id"] == identity.run_id
            assert manifest["slug"] == identity.slug
            assert manifest["resolved_config_hash"] == identity.resolved_config_hash
            assert manifest["resolved_config"]["experiment"]["benchmark"] == "lme_s"
            assert manifest["source_config_path"] == str(cfg_path.resolve())
            assert "created_at" in manifest and manifest["created_at"]
            assert "git_commit" in manifest  # 值可能为 None，但键必须存在
            assert "schema_version" in manifest

        assert result.run_root == identity.run_root
        assert result.manifest_path == manifest_path
        assert result.run_yaml_path == run_yaml_path

    def test_materialize_is_idempotent_when_config_hash_matches(self, tmp_path):
        cfg = _base_config()
        identity = ExperimentIdentity(resolved_config=cfg, source_config_path=None, repo_root=tmp_path)

        result1 = identity.materialize()
        first_created_at = result1.manifest["created_at"]

        # 用相同配置重新构造一个新的 identity 对象，模拟"重新运行同一实验"
        identity_again = ExperimentIdentity(
            resolved_config=copy.deepcopy(cfg), source_config_path=None, repo_root=tmp_path
        )
        result2 = identity_again.materialize()

        assert result2.run_root == result1.run_root
        # 复用已有 manifest，不应产生不同的 created_at（未被覆盖）
        assert result2.manifest["created_at"] == first_created_at

    def test_materialize_rejects_overwrite_when_config_hash_differs(self, tmp_path):
        cfg = _base_config()
        identity = ExperimentIdentity(resolved_config=cfg, source_config_path=None, repo_root=tmp_path)
        identity.materialize()

        manifest_path = identity.run_root / "manifest.json"
        original_text = manifest_path.read_text(encoding="utf-8")

        # 手动破坏 manifest 中的 hash，模拟"同一 run_root 但内容对不上"
        tampered = json.loads(original_text)
        tampered["resolved_config_hash"] = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
        manifest_path.write_text(json.dumps(tampered), encoding="utf-8")

        with pytest.raises(ValueError):
            identity.materialize()

        # 不能覆盖：文件内容应保持"被篡改后"的状态,而不是被新 manifest 覆盖
        after_text = manifest_path.read_text(encoding="utf-8")
        assert json.loads(after_text)["resolved_config_hash"] == tampered["resolved_config_hash"]

    def test_materialize_does_not_require_git(self, tmp_path, monkeypatch):
        """repo_root 不是 git 仓库（或系统没有 git）时也必须正常工作，git_commit 为 None。"""
        monkeypatch.setenv("PATH", "")  # 模拟找不到 git 可执行文件
        cfg = _base_config()
        identity = ExperimentIdentity(resolved_config=cfg, source_config_path=None, repo_root=tmp_path)
        result = identity.materialize()
        assert result.manifest["git_commit"] is None

    def test_materialize_rebuilds_missing_run_yaml_from_existing_manifest(self, tmp_path):
        identity = ExperimentIdentity(
            resolved_config=_base_config(), source_config_path=None, repo_root=tmp_path
        )
        first = identity.materialize()
        first.run_yaml_path.unlink()

        reused = identity.materialize()
        assert reused.reused is True
        assert yaml.safe_load(reused.run_yaml_path.read_text(encoding="utf-8")) == first.manifest

    def test_materialize_rejects_invalid_existing_manifest_json(self, tmp_path):
        identity = ExperimentIdentity(
            resolved_config=_base_config(), source_config_path=None, repo_root=tmp_path
        )
        identity.run_root.mkdir(parents=True)
        (identity.run_root / "manifest.json").write_text("{not JSON", encoding="utf-8")

        with pytest.raises(ValueError, match="invalid manifest.json"):
            identity.materialize()

    def test_materialize_concurrent_same_identity_leaves_consistent_files(self, tmp_path):
        identity = ExperimentIdentity(
            resolved_config=_base_config(), source_config_path=None, repo_root=tmp_path
        )
        results = []
        errors = []

        def materialize() -> None:
            try:
                results.append(identity.materialize())
            except Exception as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        threads = [threading.Thread(target=materialize) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors
        assert len(results) == 2
        assert any(not result.reused for result in results)
        manifest = json.loads((identity.run_root / "manifest.json").read_text(encoding="utf-8"))
        run_yaml = yaml.safe_load((identity.run_root / "run.yaml").read_text(encoding="utf-8"))
        assert manifest == run_yaml


# ---------------------------------------------------------------------------
# 5. ArtifactLayout: token limit 变化下的 ingest 复用
# ---------------------------------------------------------------------------


def _identity(tmp_path: Path, cfg: dict) -> ExperimentIdentity:
    return ExperimentIdentity(resolved_config=cfg, source_config_path=None, repo_root=tmp_path)


class TestArtifactLayoutStagesRoot:
    def test_stages_root_defaults_to_repo_root_artifacts_stages(self, tmp_path):
        identity = _identity(tmp_path, _base_config())
        layout = ArtifactLayout(identity=identity, template_root=None)
        assert layout.stages_root == tmp_path / "artifacts" / "stages"
        # 全局阶段根必须独立于 run_root（run_root 落在 artifacts/runs/<run_id> 下）
        assert layout.stages_root != identity.run_root
        assert identity.run_root.is_relative_to(tmp_path / "artifacts" / "runs")

    def test_candidate_dir_layout(self, tmp_path):
        identity = _identity(tmp_path, _base_config())
        layout = ArtifactLayout(identity=identity, template_root=None)
        expected = layout.stages_root / "candidates" / layout.candidate_id
        assert layout.candidate_dir == expected
        assert layout.candidate_id == candidate_fingerprint(identity.resolved_config)

    def test_ingest_dir_layout(self, tmp_path):
        identity = _identity(tmp_path, _base_config())
        layout = ArtifactLayout(identity=identity, template_root=None)
        method = "relation_decision"
        expected = layout.stages_root / "ingest" / method / layout.ingest_id(method)
        assert layout.ingest_dir(method) == expected
        assert layout.ingest_id(method) == ingest_fingerprint(identity.resolved_config, method)

    def test_answer_and_judge_dir_under_run_root(self, tmp_path):
        identity = _identity(tmp_path, _base_config())
        layout = ArtifactLayout(identity=identity, template_root=None)
        method = "relation_decision"

        expected_answer = identity.run_root / "answer" / method / layout.answer_id(method)
        expected_judge = identity.run_root / "judge" / method / layout.judge_id(method)
        assert layout.answer_dir(method) == expected_answer
        assert layout.judge_dir(method) == expected_judge

        ingest_id = layout.ingest_id(method)
        assert layout.answer_id(method) == answer_fingerprint(
            identity.resolved_config, method, ingest_id
        )
        assert layout.judge_id(method) == judge_fingerprint(
            identity.resolved_config, method, layout.answer_id(method)
        )

    def test_template_root_is_forwarded_to_fingerprints(self, tmp_path):
        templates = tmp_path / "templates"
        templates.mkdir()
        cfg = _base_config()
        template = templates / cfg["extract"]["aspect_templates"][0]
        template.write_text("v1", encoding="utf-8")

        identity = _identity(tmp_path, cfg)
        layout_with_root = ArtifactLayout(identity=identity, template_root=templates)
        layout_without_root = ArtifactLayout(identity=identity, template_root=None)

        assert layout_with_root.candidate_id == candidate_fingerprint(
            identity.resolved_config, template_root=templates
        )
        # 有无 template_root 应该导致不同的 candidate_id（未知模板内容 vs 已知内容）
        assert layout_with_root.candidate_id != layout_without_root.candidate_id

    def test_token_limit_change_keeps_candidate_and_ingest_dir_identical(self, tmp_path):
        """核心场景：只改 generate.memory_token_limit（256 -> 512），
        run_id / answer_dir / judge_dir 必须变化，但 candidate_dir 与
        relation_decision 的 ingest_dir 必须完全相同（复用同一 ingest 库）。
        """
        cfg_256 = _base_config()
        cfg_256["generate"]["memory_token_limit"] = 256
        cfg_512 = copy.deepcopy(cfg_256)
        cfg_512["generate"]["memory_token_limit"] = 512

        identity_256 = _identity(tmp_path, cfg_256)
        identity_512 = _identity(tmp_path, cfg_512)
        layout_256 = ArtifactLayout(identity=identity_256, template_root=None)
        layout_512 = ArtifactLayout(identity=identity_512, template_root=None)

        assert identity_256.run_id != identity_512.run_id

        method = "relation_decision"
        assert layout_256.candidate_dir == layout_512.candidate_dir
        assert layout_256.ingest_dir(method) == layout_512.ingest_dir(method)

        assert layout_256.answer_dir(method) != layout_512.answer_dir(method)
        assert layout_256.judge_dir(method) != layout_512.judge_dir(method)

    def test_relation_decision_threshold_change_only_moves_ingest_and_downstream(self, tmp_path):
        """RD 阈值变化：candidate_dir 不变，但 ingest_dir 及下游 answer/judge 都变化。"""
        cfg1 = _base_config()
        cfg2 = _base_config()
        cfg2["methods"]["relation_decision"]["condition_sim_threshold"] = 0.9

        identity1 = _identity(tmp_path, cfg1)
        identity2 = _identity(tmp_path, cfg2)
        layout1 = ArtifactLayout(identity=identity1, template_root=None)
        layout2 = ArtifactLayout(identity=identity2, template_root=None)

        method = "relation_decision"
        assert layout1.candidate_dir == layout2.candidate_dir
        assert layout1.ingest_dir(method) != layout2.ingest_dir(method)
        assert layout1.answer_dir(method) != layout2.answer_dir(method)
        assert layout1.judge_dir(method) != layout2.judge_dir(method)

    def test_add_all_and_relation_decision_ingest_dirs_are_isolated(self, tmp_path):
        identity = _identity(tmp_path, _base_config())
        layout = ArtifactLayout(identity=identity, template_root=None)

        rd_dir = layout.ingest_dir("relation_decision")
        add_all_dir = layout.ingest_dir("add_all")

        assert rd_dir != add_all_dir
        # 不只是 id 不同，父目录（method 分片）也必须不同，保证物理隔离
        assert rd_dir.parent != add_all_dir.parent
        assert rd_dir.parent.name == "relation_decision"
        assert add_all_dir.parent.name == "add_all"

    def test_answer_and_judge_dirs_isolated_across_methods(self, tmp_path):
        identity = _identity(tmp_path, _base_config())
        layout = ArtifactLayout(identity=identity, template_root=None)
        assert layout.answer_dir("relation_decision") != layout.answer_dir("add_all")
        assert layout.judge_dir("relation_decision") != layout.judge_dir("add_all")


class TestArtifactLayoutMethodSafety:
    @pytest.mark.parametrize(
        "bad_method",
        ["", "..", "../escape", "a/b", "a\\b", "a b", ".", "rd/../../etc"],
    )
    def test_unsafe_method_names_are_rejected(self, tmp_path, bad_method):
        identity = _identity(tmp_path, _base_config())
        layout = ArtifactLayout(identity=identity, template_root=None)
        with pytest.raises(ValueError):
            layout.ingest_dir(bad_method)
        with pytest.raises(ValueError):
            layout.answer_dir(bad_method)
        with pytest.raises(ValueError):
            layout.judge_dir(bad_method)

    def test_unsafe_method_cannot_escape_stages_root(self, tmp_path):
        identity = _identity(tmp_path, _base_config())
        layout = ArtifactLayout(identity=identity, template_root=None)
        with pytest.raises(ValueError):
            layout.ingest_dir("../../outside")


class TestArtifactLayoutStageManifests:
    def test_candidate_manifest_has_no_upstream(self, tmp_path):
        identity = _identity(tmp_path, _base_config())
        layout = ArtifactLayout(identity=identity, template_root=None)
        manifest = layout.candidate_manifest()

        assert manifest["schema_version"]
        assert manifest["stage"] == "candidate"
        assert manifest["stage_id"] == layout.candidate_id
        assert manifest["run_id"] == identity.run_id
        assert manifest.get("method") is None
        assert manifest["upstream_stage_ids"] == []
        assert manifest["resolved_config_hash"] == identity.resolved_config_hash

    def test_ingest_manifest_upstream_is_candidate(self, tmp_path):
        identity = _identity(tmp_path, _base_config())
        layout = ArtifactLayout(identity=identity, template_root=None)
        method = "relation_decision"
        manifest = layout.ingest_manifest(method)

        assert manifest["stage"] == "ingest"
        assert manifest["stage_id"] == layout.ingest_id(method)
        assert manifest["method"] == method
        assert manifest["upstream_stage_ids"] == [layout.candidate_id]
        assert manifest["run_id"] == identity.run_id
        assert manifest["resolved_config_hash"] == identity.resolved_config_hash

    def test_answer_manifest_upstream_is_ingest(self, tmp_path):
        identity = _identity(tmp_path, _base_config())
        layout = ArtifactLayout(identity=identity, template_root=None)
        method = "relation_decision"
        manifest = layout.answer_manifest(method)

        assert manifest["stage"] == "answer"
        assert manifest["stage_id"] == layout.answer_id(method)
        assert manifest["method"] == method
        assert manifest["upstream_stage_ids"] == [layout.ingest_id(method)]

    def test_judge_manifest_upstream_is_answer(self, tmp_path):
        identity = _identity(tmp_path, _base_config())
        layout = ArtifactLayout(identity=identity, template_root=None)
        method = "relation_decision"
        manifest = layout.judge_manifest(method)

        assert manifest["stage"] == "judge"
        assert manifest["stage_id"] == layout.judge_id(method)
        assert manifest["method"] == method
        assert manifest["upstream_stage_ids"] == [layout.answer_id(method)]

    def test_stage_manifests_are_not_persisted_to_disk(self, tmp_path):
        """manifest 只是内存 mapping，layout 本身不应该落盘（留给编排层）。"""
        identity = _identity(tmp_path, _base_config())
        layout = ArtifactLayout(identity=identity, template_root=None)
        layout.candidate_manifest()
        layout.ingest_manifest("relation_decision")
        layout.answer_manifest("relation_decision")
        layout.judge_manifest("relation_decision")
        assert not layout.stages_root.exists()
        assert not identity.run_root.exists()


class TestArtifactLayoutAttemptDir:
    def test_new_attempt_dir_is_under_run_root_attempts(self, tmp_path):
        identity = _identity(tmp_path, _base_config())
        layout = ArtifactLayout(identity=identity, template_root=None)

        attempt_dir = layout.new_attempt_dir()
        assert attempt_dir.parent == identity.run_root / "attempts"
        assert attempt_dir.exists() and attempt_dir.is_dir()

    def test_new_attempt_dir_generates_distinct_ids_each_call(self, tmp_path):
        identity = _identity(tmp_path, _base_config())
        layout = ArtifactLayout(identity=identity, template_root=None)

        first = layout.new_attempt_dir()
        second = layout.new_attempt_dir()
        assert first != second
        assert first.exists() and second.exists()

    def test_new_attempt_dir_does_not_touch_other_stage_dirs(self, tmp_path):
        identity = _identity(tmp_path, _base_config())
        layout = ArtifactLayout(identity=identity, template_root=None)

        layout.new_attempt_dir()
        assert not layout.candidate_dir.exists()
        assert not layout.stages_root.exists() or not (layout.stages_root / "candidates").exists()


class TestArtifactLayoutSharedStageSafety:
    def test_candidate_lock_serializes_two_threads_outside_stage_data_dir(self, tmp_path):
        layout = ArtifactLayout(_identity(tmp_path, _base_config()), template_root=None)
        entered = []
        first_entered = threading.Event()
        release_first = threading.Event()

        def first() -> None:
            with layout.candidate_lock():
                entered.append("first")
                first_entered.set()
                release_first.wait(timeout=2)

        def second() -> None:
            first_entered.wait(timeout=2)
            with layout.candidate_lock():
                entered.append("second")

        first_thread = threading.Thread(target=first)
        second_thread = threading.Thread(target=second)
        first_thread.start()
        second_thread.start()
        first_entered.wait(timeout=2)
        # The second thread must still be blocked on the same file lock.
        assert entered == ["first"]
        release_first.set()
        first_thread.join(timeout=2)
        second_thread.join(timeout=2)

        assert entered == ["first", "second"]
        lock_files = list((layout.stages_root / "locks").glob("*"))
        assert len(lock_files) == 1
        assert lock_files[0].is_relative_to(layout.stages_root / "locks")
        assert not lock_files[0].is_relative_to(layout.candidate_dir)

    def test_shared_stage_manifest_is_created_after_stage_and_reused_across_token_limits(
        self, tmp_path
    ):
        cfg_256 = _base_config()
        cfg_256["generate"]["memory_token_limit"] = 256
        cfg_512 = copy.deepcopy(cfg_256)
        cfg_512["generate"]["memory_token_limit"] = 512
        layout_256 = ArtifactLayout(_identity(tmp_path, cfg_256), template_root=None)
        layout_512 = ArtifactLayout(_identity(tmp_path, cfg_512), template_root=None)

        with layout_256.candidate_lock():
            candidate_path = layout_256.write_candidate_stage_manifest()
        with layout_256.ingest_lock("relation_decision"):
            ingest_path = layout_256.write_ingest_stage_manifest("relation_decision")
        with layout_512.candidate_lock():
            assert layout_512.write_candidate_stage_manifest() == candidate_path
        with layout_512.ingest_lock("relation_decision"):
            assert layout_512.write_ingest_stage_manifest("relation_decision") == ingest_path

        manifest = json.loads(ingest_path.read_text(encoding="utf-8"))
        assert manifest["stage"] == "ingest"
        assert manifest["stage_id"] == layout_256.ingest_id("relation_decision")
        assert manifest["method"] == "relation_decision"
        assert manifest["upstream_stage_ids"] == [layout_256.candidate_id]
        assert manifest["producer_run_id"] == layout_256.identity.run_id
        assert manifest["producer_resolved_config_hash"] == layout_256.identity.resolved_config_hash
        assert manifest["producer_run_root"] == str(layout_256.run_root)
        assert manifest["created_at"]

    def test_shared_stage_manifest_rejects_incompatible_existing_manifest(self, tmp_path):
        layout = ArtifactLayout(_identity(tmp_path, _base_config()), template_root=None)
        path = layout.candidate_dir / "stage_manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "stage": "ingest",
                    "stage_id": layout.candidate_id,
                    "method": None,
                    "upstream_stage_ids": [],
                }
            ),
            encoding="utf-8",
        )

        with layout.candidate_lock(), pytest.raises(ValueError, match="incompatible"):
            layout.write_candidate_stage_manifest()


# ---------------------------------------------------------------------------
# 6. ArtifactLayout.stage_nonce：full_pipeline 重复统计的强制去重
# ---------------------------------------------------------------------------


class TestArtifactLayoutStageNonce:
    def test_nonce_changes_candidate_and_ingest_id(self, tmp_path):
        identity = _identity(tmp_path, _base_config())
        base_layout = ArtifactLayout(identity=identity, template_root=None)
        nonce_layout = ArtifactLayout(identity=identity, template_root=None, stage_nonce="r00")

        assert nonce_layout.candidate_id != base_layout.candidate_id
        method = "relation_decision"
        assert nonce_layout.ingest_id(method) != base_layout.ingest_id(method)
        assert nonce_layout.candidate_dir != base_layout.candidate_dir
        assert nonce_layout.ingest_dir(method) != base_layout.ingest_dir(method)

    def test_different_nonces_give_different_ids(self, tmp_path):
        identity = _identity(tmp_path, _base_config())
        layout_a = ArtifactLayout(identity=identity, template_root=None, stage_nonce="r00")
        layout_b = ArtifactLayout(identity=identity, template_root=None, stage_nonce="r01")

        assert layout_a.candidate_id != layout_b.candidate_id
        assert layout_a.ingest_dir("add_all") != layout_b.ingest_dir("add_all")

    def test_empty_nonce_behaves_like_none(self, tmp_path):
        identity = _identity(tmp_path, _base_config())
        layout_none = ArtifactLayout(identity=identity, template_root=None, stage_nonce=None)
        layout_empty = ArtifactLayout(identity=identity, template_root=None, stage_nonce="")

        assert layout_none.candidate_id == layout_empty.candidate_id
        assert layout_none.ingest_dir("add_all") == layout_empty.ingest_dir("add_all")

    def test_nonce_propagates_to_answer_and_judge(self, tmp_path):
        identity = _identity(tmp_path, _base_config())
        base_layout = ArtifactLayout(identity=identity, template_root=None)
        nonce_layout = ArtifactLayout(identity=identity, template_root=None, stage_nonce="r00")
        method = "relation_decision"

        assert nonce_layout.answer_id(method) != base_layout.answer_id(method)
        assert nonce_layout.judge_id(method) != base_layout.judge_id(method)
        # answer/judge 必须依赖 nonce 之后的 ingest_id，而不是原始 ingest_fingerprint。
        assert nonce_layout.answer_id(method) == answer_fingerprint(
            identity.resolved_config, method, nonce_layout.ingest_id(method)
        )
        assert nonce_layout.answer_id(method) != answer_fingerprint(
            identity.resolved_config, method, ingest_fingerprint(identity.resolved_config, method)
        )

    def test_nonce_preserves_method_safety_locks_and_manifests(self, tmp_path):
        identity = _identity(tmp_path, _base_config())
        layout = ArtifactLayout(identity=identity, template_root=None, stage_nonce="r00")

        with pytest.raises(ValueError):
            layout.ingest_dir("../escape")
        with pytest.raises(ValueError):
            layout.answer_dir("a/b")
        with pytest.raises(ValueError):
            layout.judge_dir("")

        with layout.candidate_lock():
            path = layout.write_candidate_stage_manifest()
        assert path.exists()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["stage_id"] == layout.candidate_id

        with layout.ingest_lock("relation_decision"):
            ingest_path = layout.write_ingest_stage_manifest("relation_decision")
        ingest_manifest = json.loads(ingest_path.read_text(encoding="utf-8"))
        assert ingest_manifest["stage_id"] == layout.ingest_id("relation_decision")
        assert ingest_manifest["upstream_stage_ids"] == [layout.candidate_id]

    def test_nonce_does_not_change_run_root(self, tmp_path):
        identity = _identity(tmp_path, _base_config())
        base_layout = ArtifactLayout(identity=identity, template_root=None)
        nonce_layout = ArtifactLayout(identity=identity, template_root=None, stage_nonce="r00")

        assert base_layout.run_root == nonce_layout.run_root
        assert base_layout.answer_dir("add_all").is_relative_to(base_layout.run_root)
        assert nonce_layout.answer_dir("add_all").is_relative_to(nonce_layout.run_root)
