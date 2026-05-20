"""Compatibility entrypoint and re-exports for scripts importing ``extract_lme_candidate_memory``."""

from benchmark.datasets import DEFAULT_BENCHMARK_DATASETS as BENCHMARK_TO_DATASET
from pipeline.extract_candidates import main
from pipeline.extract_candidates import episode_to_observation_chunks
from pipeline.extract_candidates import _normalize_memory_granularity
from pipeline.extract_candidates import _parse_json_from_llm

__all__ = [
    "BENCHMARK_TO_DATASET",
    "main",
    "episode_to_observation_chunks",
    "_normalize_memory_granularity",
    "_parse_json_from_llm",
]

if __name__ == "__main__":
    main()
