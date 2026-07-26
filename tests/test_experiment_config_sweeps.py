"""统计重复（replication）与 token-limit 矩阵（sweep）的纯配置/规划层单测（TDD）。

只测试 ``ExperimentConfig`` 上新增的 ``replication`` / ``sweep`` 配置节，以及
``replication_specs()`` / ``experiment_variants()`` 两个规划方法。不涉及
runner、pipeline 或用户 config YAML 的任何改动。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.config import (  # noqa: E402
    ExperimentConfig,
    ExperimentVariant,
    ReplicationSpec,
)


# ---------------------------------------------------------------------------
# 1. replication_specs()
# ---------------------------------------------------------------------------


class TestReplicationSpecs:
    def test_default_count_one_returns_single_spec_index_zero(self):
        cfg = ExperimentConfig()
        specs = cfg.replication_specs()
        assert specs == [ReplicationSpec(index=0, seed=cfg.generate.answer_sample_seed)]

    def test_no_seeds_defaults_to_answer_sample_seed_plus_index(self):
        cfg = ExperimentConfig.model_validate(
            {
                "generate": {"answer_sample_seed": 43},
                "replication": {"count": 3},
            }
        )
        specs = cfg.replication_specs()
        assert specs == [
            ReplicationSpec(index=0, seed=43),
            ReplicationSpec(index=1, seed=44),
            ReplicationSpec(index=2, seed=45),
        ]

    def test_explicit_seeds_used_verbatim(self):
        cfg = ExperimentConfig.model_validate(
            {"replication": {"count": 3, "seeds": [100, 200, 300]}}
        )
        specs = cfg.replication_specs()
        assert specs == [
            ReplicationSpec(index=0, seed=100),
            ReplicationSpec(index=1, seed=200),
            ReplicationSpec(index=2, seed=300),
        ]

    def test_seeds_length_mismatch_raises_value_error(self):
        with pytest.raises(ValueError, match="seeds"):
            ExperimentConfig.model_validate(
                {"replication": {"count": 2, "seeds": [1, 2, 3]}}
            )

    def test_seeds_can_be_arbitrary_ints_including_negative(self):
        cfg = ExperimentConfig.model_validate(
            {"replication": {"count": 2, "seeds": [-5, 0]}}
        )
        specs = cfg.replication_specs()
        assert specs == [
            ReplicationSpec(index=0, seed=-5),
            ReplicationSpec(index=1, seed=0),
        ]

    def test_count_zero_raises_value_error(self):
        with pytest.raises(ValueError):
            ExperimentConfig.model_validate({"replication": {"count": 0}})

    def test_count_negative_raises_value_error(self):
        with pytest.raises(ValueError):
            ExperimentConfig.model_validate({"replication": {"count": -1}})

    def test_scope_defaults_to_answer_judge(self):
        cfg = ExperimentConfig()
        assert cfg.replication.scope == "answer_judge"

    def test_scope_accepts_full_pipeline(self):
        cfg = ExperimentConfig.model_validate(
            {"replication": {"scope": "full_pipeline"}}
        )
        assert cfg.replication.scope == "full_pipeline"

    def test_scope_invalid_value_raises_value_error(self):
        with pytest.raises(ValueError):
            ExperimentConfig.model_validate({"replication": {"scope": "bogus"}})


# ---------------------------------------------------------------------------
# 2. sweep 配置校验
# ---------------------------------------------------------------------------


class TestSweepConfigValidation:
    def test_memory_token_limits_default_empty(self):
        cfg = ExperimentConfig()
        assert cfg.sweep.memory_token_limits == []

    def test_zero_token_limit_raises_value_error(self):
        with pytest.raises(ValueError):
            ExperimentConfig.model_validate({"sweep": {"memory_token_limits": [0]}})

    def test_negative_token_limit_raises_value_error(self):
        with pytest.raises(ValueError):
            ExperimentConfig.model_validate(
                {"sweep": {"memory_token_limits": [256, -128]}}
            )

    def test_duplicates_deduped_preserving_write_order(self):
        cfg = ExperimentConfig.model_validate(
            {"sweep": {"memory_token_limits": [512, 256, 512, 128, 256]}}
        )
        assert cfg.sweep.memory_token_limits == [512, 256, 128]


# ---------------------------------------------------------------------------
# 3. experiment_variants()
# ---------------------------------------------------------------------------


class TestExperimentVariants:
    def test_default_config_yields_single_variant(self):
        cfg = ExperimentConfig()
        variants = cfg.experiment_variants()
        assert len(variants) == 1
        variant = variants[0]
        assert isinstance(variant, ExperimentVariant)
        assert variant.memory_token_limit == cfg.generate.memory_token_limit
        assert variant.replication == ReplicationSpec(
            index=0, seed=cfg.generate.answer_sample_seed
        )
        assert variant.variant_id == f"tl{cfg.generate.memory_token_limit}-r00-s{cfg.generate.answer_sample_seed}"

    def test_two_token_limits_times_three_repeats_yields_six_variants_token_outer(
        self,
    ):
        cfg = ExperimentConfig.model_validate(
            {
                "sweep": {"memory_token_limits": [256, 512]},
                "replication": {"count": 3},
                "generate": {"answer_sample_seed": 43},
            }
        )
        variants = cfg.experiment_variants()
        assert len(variants) == 6

        token_limit_seq = [v.memory_token_limit for v in variants]
        assert token_limit_seq == [256, 256, 256, 512, 512, 512]

        repeat_index_seq = [v.replication.index for v in variants]
        assert repeat_index_seq == [0, 1, 2, 0, 1, 2]

        seed_seq = [v.replication.seed for v in variants]
        assert seed_seq == [43, 44, 45, 43, 44, 45]

    def test_variant_ids_are_unique_and_follow_expected_format(self):
        cfg = ExperimentConfig.model_validate(
            {
                "sweep": {"memory_token_limits": [256, 512]},
                "replication": {"count": 3},
                "generate": {"answer_sample_seed": 43},
            }
        )
        variants = cfg.experiment_variants()
        variant_ids = [v.variant_id for v in variants]
        assert variant_ids == [
            "tl256-r00-s43",
            "tl256-r01-s44",
            "tl256-r02-s45",
            "tl512-r00-s43",
            "tl512-r01-s44",
            "tl512-r02-s45",
        ]
        assert len(set(variant_ids)) == len(variant_ids)

    def test_each_variant_config_has_matching_memory_token_limit_and_seed(self):
        cfg = ExperimentConfig.model_validate(
            {
                "sweep": {"memory_token_limits": [256, 512]},
                "replication": {"count": 2},
                "generate": {"answer_sample_seed": 10},
            }
        )
        variants = cfg.experiment_variants()
        for variant in variants:
            assert variant.config.generate.memory_token_limit == variant.memory_token_limit
            assert variant.config.generate.answer_sample_seed == variant.replication.seed

    def test_original_config_is_not_mutated(self):
        cfg = ExperimentConfig.model_validate(
            {
                "sweep": {"memory_token_limits": [256, 512]},
                "replication": {"count": 3},
                "generate": {"memory_token_limit": 384, "answer_sample_seed": 43},
            }
        )
        cfg.experiment_variants()
        assert cfg.generate.memory_token_limit == 384
        assert cfg.generate.answer_sample_seed == 43
        assert cfg.sweep.memory_token_limits == [256, 512]

    def test_variant_configs_are_independent_deep_copies(self):
        cfg = ExperimentConfig.model_validate(
            {
                "sweep": {"memory_token_limits": [256, 512]},
                "replication": {"count": 1},
            }
        )
        variants = cfg.experiment_variants()
        variants[0].config.generate.memory_token_limit = 999999
        assert variants[1].config.generate.memory_token_limit == 512
        assert cfg.generate.memory_token_limit != 999999

    def test_sweep_empty_uses_current_generate_memory_token_limit(self):
        cfg = ExperimentConfig.model_validate({"generate": {"memory_token_limit": 777}})
        variants = cfg.experiment_variants()
        assert len(variants) == 1
        assert variants[0].memory_token_limit == 777


# ---------------------------------------------------------------------------
# 4. 兼容性：from_yaml / 现有字段行为不变
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_from_yaml_without_new_sections_still_works(self, tmp_path):
        config_path = tmp_path / "legacy.yaml"
        config_path.write_text(
            "experiment:\n  benchmark: lme_s\n  suffix: exp001\n",
            encoding="utf-8",
        )
        cfg = ExperimentConfig.from_yaml(config_path)
        assert cfg.replication.count == 1
        assert cfg.sweep.memory_token_limits == []
        assert cfg.experiment_variants()[0].memory_token_limit == cfg.generate.memory_token_limit
