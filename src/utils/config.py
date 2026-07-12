"""ExperimentConfig: typed Pydantic model for config/lme.yaml.

Used by run_exp_lme.py to read the unified config and build CLI args for each stage.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field


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


class PromptsConfig(BaseModel):
    relation_system_en: str = "lme_relation_classification_system_en_v2.jinja"
    relation_system_zh: str = "lme_relation_classification_system_zh_v2.jinja"
    relation_user: str = "lme_relation_classification_user.jinja"
    fusion_bundle_en: str = "fuse_memory_bundle_en_v3.jinja"
    fusion_bundle_zh: str = ""
    fusion_edge_labels_en: str = "fuse_memory_bundle_edge_labels_en_v2.jinja"
    fusion_edge_labels_zh: str = "fuse_memory_bundle_edge_labels_zh_v2.jinja"
    judge_oqa: str = "pipeline_eval_oqa.jinja"
    judge_mcq: str = "pipeline_eval_mcq.jinja"
    judge_system: str = "pipeline_eval_system.jinja"


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
    backend: str = "classifier"       # "classifier" | "llm"
    fusion_model: str = ""
    cascade_enabled: bool = True
    deletion_enabled: bool = True
    condition_sim_threshold: float = 0.5
    pairwise_sim_threshold: float = 0.7
    fusion_enabled: bool = True


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
# Root config
# ---------------------------------------------------------------------------

_METHOD_ORDER = ("amac", "zep", "relation_decision", "mem0", "add_all", "evermemos")


# ---------------------------------------------------------------------------
# MEME-specific config models
# ---------------------------------------------------------------------------


class MemeRunConfig(BaseModel):
    """4-phase ingest + answer 参数（对应 pipeline_meme_4phase.py 的公共参数）。"""
    retrieve_topk: int = 50
    memory_token_limit: int = 512
    answer_concurrency: int = 10
    show_memory_time: bool = False
    hybrid: HybridConfig = Field(default_factory=HybridConfig)


class MemeEvaluateConfig(BaseModel):
    judge_max_concurrency: int = 8
    judge_max_new_tokens: int = 512


class MemeParallelEpisodesConfig(BaseModel):
    """4-phase 各方法 per-episode 并发数。"""
    relation_decision: int = 20
    mem0: int = 4
    add_all: int = 20
    zep: int = 10
    amac: int = 10
    evermemos: int = 4


class MemeParallelConfig(BaseModel):
    extract_chunk_concurrency: int = 100
    ingest_relation_concurrency: int = 50    # relation_decision 单 episode 内关系对并发
    evermemos_cluster_concurrency: int = 8
    fuse_package_concurrency: int = 4
    parallel_episodes: MemeParallelEpisodesConfig = Field(
        default_factory=MemeParallelEpisodesConfig
    )


class MemeExperimentConfig(BaseModel):
    experiment: ExperimentMeta = Field(default_factory=ExperimentMeta)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    extract: ExtractConfig = Field(default_factory=ExtractConfig)
    methods: MethodsConfig = Field(default_factory=MethodsConfig)
    run: MemeRunConfig = Field(default_factory=MemeRunConfig)
    evaluate: MemeEvaluateConfig = Field(default_factory=MemeEvaluateConfig)
    parallel: MemeParallelConfig = Field(default_factory=MemeParallelConfig)
    token_limits: TokenLimitsConfig = Field(default_factory=TokenLimitsConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MemeExperimentConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

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

    def ingest_4phase_dir(self, method: str) -> Path:
        """Unfused 4-phase DB root for ``method`` ({method}_4p)."""
        return self.ingest_run_root / f"{method}_4p"

    def ingest_4phase_fused_dir(self, method: str) -> Path:
        """Fused DB root (only for relation_decision)."""
        return self.ingest_run_root / f"{method}_4p_fused"

    @property
    def experiment_run_root(self) -> Path:
        b = self.experiment.benchmark
        s = self.extract.candidate_suffix
        mgr = self._safe_tag(self.models.manager)
        ans = self._safe_tag(self.models.answer)
        tl = self.run.memory_token_limit
        e = self.experiment.suffix
        return Path("experiment") / f"{b}_cand{s}_{mgr}_{ans}_tl{tl}_{e}_meme4p"

    def pred_file(self, method: str) -> Path:
        return self.experiment_run_root / f"pred_{method}.jsonl"

    @property
    def enabled_methods(self) -> list[str]:
        return [
            name
            for name in _METHOD_ORDER
            if getattr(getattr(self.methods, name), "enabled", False)
        ]


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

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

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
