"""Default benchmark id → (data file path, language). Extend here or use --benchmark-file."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

# Used by pipeline_lme_generate, extract_candidates, and any tool that resolves paths by benchmark id.
DEFAULT_BENCHMARK_DATASETS: Dict[str, Tuple[str, str]] = {
    "test": ("data/preprocessed/test.json", "zh"),
    "lme_o": ("data/preprocessed/longmemeval_oracle_converted.json", "en"),
    "lme_s": ("data/preprocessed/longmemeval_s_cleaned_converted.json", "en"),
    "lme_s_golden": ("data/preprocessed/longmemeval_s_hybrid_golden_converted.json", "en"),
    "lme_m": ("data/preprocessed/longmemeval_m_cleaned_converted.json", "en"),
}


def resolve_benchmark_data_path(
    benchmark: str,
    benchmark_file: Optional[str] = None,
) -> Tuple[str, str]:
    """Return (file_path, lang). Raises ValueError if unknown benchmark and no file given."""
    if benchmark_file:
        return benchmark_file, "en"
    key = benchmark.strip()
    if key not in DEFAULT_BENCHMARK_DATASETS:
        keys = ", ".join(sorted(DEFAULT_BENCHMARK_DATASETS.keys()))
        raise ValueError(
            f"Unknown --benchmark {benchmark!r}. Use --benchmark-file or one of: {keys}"
        )
    return DEFAULT_BENCHMARK_DATASETS[key]
