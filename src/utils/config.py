"""ExperimentConfig: typed Pydantic model for config/lme.yaml.

Used by run_exp_lme.py to read the unified config and build CLI args for each stage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Leaf config sections
# ---------------------------------------------------------------------------


class ExperimentMeta(BaseModel):
    benchmark: str = "lme_s"
    suffix: str = "exp001"


class ModelsConfig(BaseModel):
    extract: str = "gemma4-26B"
    manager: str = "Qwen3-4B"
    answer: str = "gemma4-26B"
    judge: str = "qwen3-max"
    embedding: str = "qwen3-embedding-8b"


class ExtractConfig(BaseModel):
    candidate_suffix: str = "default"
    granularity: str = "4"
    turn_overlap: str = "0"
    language: str = "en"
    aspect_templates: List[str] = Field(default_factory=list)

    model_config = {"coerce_numbers_to_str": True}


class HybridConfig(BaseModel):
    enabled: bool = True
    dense_weight: float = 0.8
    bm25_weight: float = 0.2
    pool_mult: int = 4


class GenerateConfig(BaseModel):
    retrieve_topk: int = 50
    memory_token_limit: int = 512
    answer_stratified_sample: int = 500
    answer_sample_seed: int = 43
    show_memory_time: bool = False
    hybrid: HybridConfig = Field(default_factory=HybridConfig)


class EvaluateConfig(BaseModel):
    use_cot: bool = True
    judge_stratified_sample: int = 0
    judge_sample_seed: int = 43


class IngestEpisodeConcurrencyConfig(BaseModel):
    relation_decision: int = 100
    mem0: int = 50
    add_all: int = 100
    zep: int = 50
    amac: int = 100
    evermemos: int = 5
    fusion_episodes: int = 100
    fusion_packages: int = 10


class ParallelConfig(BaseModel):
    extract_chunk_concurrency: int = 100
    ingest_relation_concurrency: int = 10
    ingest_episode_concurrency: IngestEpisodeConcurrencyConfig = Field(
        default_factory=IngestEpisodeConcurrencyConfig
    )
    generate_parallel_episodes: int = 100
    generate_answer_concurrency: int = 2
    evaluate_max_concurrency: int = 5


class TokenLimitsConfig(BaseModel):
    extract_max_new_tokens: int = 2048
    ingest_relation_max_new_tokens: int = 256
    ingest_manager_max_new_tokens: int = 2048
    fusion_max_new_tokens: int = 512
    evaluate_max_new_tokens: int = 2048


class DebugConfig(BaseModel):
    evaluate_print_one_sample: bool = False


class ReplicationConfig(BaseModel):
    """统计重复配置：控制同一实验重复跑几遍（不同随机种子）。"""

    count: int = Field(default=1, ge=1)
    scope: Literal["answer_judge", "full_pipeline"] = "answer_judge"
    seeds: Optional[List[int]] = None

    @model_validator(mode="after")
    def _check_seeds_length(self) -> "ReplicationConfig":
        if self.seeds is not None and len(self.seeds) != self.count:
            raise ValueError(
                "replication.seeds 长度"
                f"({len(self.seeds)}) 必须等于 replication.count({self.count})"
            )
        return self


class SweepConfig(BaseModel):
    """token-limit 矩阵配置：控制要扫哪些 memory_token_limit 取值。"""

    memory_token_limits: List[int] = Field(default_factory=list)

    @field_validator("memory_token_limits")
    @classmethod
    def _validate_and_dedupe(cls, value: List[int]) -> List[int]:
        deduped: List[int] = []
        seen: set[int] = set()
        for limit in value:
            if limit <= 0:
                raise ValueError(f"sweep.memory_token_limits 中的值必须 > 0，got {limit}")
            if limit not in seen:
                seen.add(limit)
                deduped.append(limit)
        return deduped


class PromptsConfig(BaseModel):
    relation_user_en: str = "RD_0_relation_classify.jinja"
    relation_user_zh: str = "RD_0_relation_classify.jinja"
    judge_template: str = "pipeline_judge.jinja"


# ---------------------------------------------------------------------------
# Per-method configs
# ---------------------------------------------------------------------------


class AmacMethodConfig(BaseModel):
    enabled: bool = False
    threshold: float = 0.55
    weights: str = "0.1,0.1,0.1,0.1,0.6"
    skip_utility: bool = False
    recency_decay_per_step: float = 0.12
    novelty_max_existing: int = 64


class ZepMethodConfig(BaseModel):
    enabled: bool = False


class RelationDecisionMethodConfig(BaseModel):
    enabled: bool = False
    related_top_k: int = 3
    backend: Literal["llm"] = "llm"
    fusion_model: str = ""
    condition_sim_threshold: float = 0.5
    pairwise_sim_threshold: float = 0.7
    fusion_enabled: bool = True
    active_relations: Optional[list[str]] = None  # 消融：限定的关系类型，None=全部生效


class Mem0MethodConfig(BaseModel):
    enabled: bool = False
    related_top_k: int = 3
    related_aggregate_max: int = 10


class AddAllMethodConfig(BaseModel):
    enabled: bool = False


class EverMemosMethodConfig(BaseModel):
    enabled: bool = False
    similarity_threshold: float = 0.65
    max_time_gap_days: float = 7.0


class MethodsConfig(BaseModel):
    amac: AmacMethodConfig = Field(default_factory=AmacMethodConfig)
    zep: ZepMethodConfig = Field(default_factory=ZepMethodConfig)
    relation_decision: RelationDecisionMethodConfig = Field(
        default_factory=RelationDecisionMethodConfig
    )
    mem0: Mem0MethodConfig = Field(default_factory=Mem0MethodConfig)
    add_all: AddAllMethodConfig = Field(default_factory=AddAllMethodConfig)
    evermemos: EverMemosMethodConfig = Field(default_factory=EverMemosMethodConfig)


# ---------------------------------------------------------------------------
# Runtime-only planning objects (not part of the YAML schema, never persisted)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplicationSpec:
    """单次重复运行的规划信息：第几次重复 + 使用的随机种子。"""

    index: int
    seed: int


@dataclass(frozen=True)
class ExperimentVariant:
    """一次 token-limit x replication 组合的规划信息（纯规划层，不落 YAML）。"""

    config: "ExperimentConfig"
    memory_token_limit: int
    replication: ReplicationSpec
    variant_id: str


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

_METHOD_ORDER = ("amac", "zep", "relation_decision", "mem0", "add_all", "evermemos")


class ExperimentConfig(BaseModel):
    experiment: ExperimentMeta = Field(default_factory=ExperimentMeta)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    extract: ExtractConfig = Field(default_factory=ExtractConfig)
    methods: MethodsConfig = Field(default_factory=MethodsConfig)
    generate: GenerateConfig = Field(default_factory=GenerateConfig)
    evaluate: EvaluateConfig = Field(default_factory=EvaluateConfig)
    parallel: ParallelConfig = Field(default_factory=ParallelConfig)
    token_limits: TokenLimitsConfig = Field(default_factory=TokenLimitsConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    replication: ReplicationConfig = Field(default_factory=ReplicationConfig)
    sweep: SweepConfig = Field(default_factory=SweepConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    # ------------------------------------------------------------------
    # Replication / sweep planning (pure config-level, no I/O)
    # ------------------------------------------------------------------

    def replication_specs(self) -> list[ReplicationSpec]:
        """展开 replication 配置为具体的 (index, seed) 列表。"""
        seeds = self.replication.seeds
        if seeds:
            return [ReplicationSpec(index=i, seed=seed) for i, seed in enumerate(seeds)]
        base_seed = self.generate.answer_sample_seed
        return [
            ReplicationSpec(index=i, seed=base_seed + i)
            for i in range(self.replication.count)
        ]

    def experiment_variants(self) -> list[ExperimentVariant]:
        """展开 sweep x replication 笛卡尔积为具体的实验变体列表。

        顺序：token limit 外层、repeat 内层。每个 variant 携带深复制后的
        ``ExperimentConfig``（不修改 ``self``），并设置对应的
        ``generate.memory_token_limit`` / ``generate.answer_sample_seed``。
        """
        token_limits = self.sweep.memory_token_limits or [self.generate.memory_token_limit]
        specs = self.replication_specs()

        variants: list[ExperimentVariant] = []
        for token_limit in token_limits:
            for spec in specs:
                variant_cfg = self.model_copy(deep=True)
                variant_cfg.generate.memory_token_limit = token_limit
                variant_cfg.generate.answer_sample_seed = spec.seed
                variant_id = f"tl{token_limit}-r{spec.index:02d}-s{spec.seed}"
                variants.append(
                    ExperimentVariant(
                        config=variant_cfg,
                        memory_token_limit=token_limit,
                        replication=spec,
                        variant_id=variant_id,
                    )
                )
        return variants

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_tag(name: str) -> str:
        tag = re.sub(r"[/:\\\s]+", "_", str(name).strip())
        tag = re.sub(r"[^a-zA-Z0-9_.-]+", "", tag)
        return tag or "model"

    @property
    def candidates_dir(self) -> Path:
        b = self.experiment.benchmark
        m = self._safe_tag(self.models.extract)
        s = self.extract.candidate_suffix
        return Path("MemDB") / "candidates" / f"{b}_{m}_{s}"

    @property
    def ingest_run_root(self) -> Path:
        b = self.experiment.benchmark
        s = self.extract.candidate_suffix
        m = self._safe_tag(self.models.manager)
        e = self.experiment.suffix
        return Path("MemDB") / "ingest" / f"{b}_cand{s}_{m}_{e}"

    def ingest_dir(self, method: str) -> Path:
        return self.ingest_run_root / method

    @property
    def experiment_run_root(self) -> Path:
        b = self.experiment.benchmark
        s = self.extract.candidate_suffix
        mgr = self._safe_tag(self.models.manager)
        ans = self._safe_tag(self.models.answer)
        tl = self.generate.memory_token_limit
        e = self.experiment.suffix
        return Path("experiment") / f"{b}_cand{s}_{mgr}_{ans}_tl{tl}_{e}"

    def pred_file(self, method: str) -> Path:
        return self.experiment_run_root / f"pred_{method}.jsonl"

    @property
    def enabled_methods(self) -> list[str]:
        return [
            name
            for name in _METHOD_ORDER
            if getattr(getattr(self.methods, name), "enabled", False)
        ]
