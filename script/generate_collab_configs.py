#!/usr/bin/env python3
"""为协作者生成实验配置：gemma4-12b × LME Hybrid 全组合。

生成 3 方法 × 5 filler level = 15 个 YAML 配置文件。
用法：
    PYTHONPATH=src uv run --no-sync python script/generate_collab_configs.py
"""

from __future__ import annotations

from pathlib import Path
import yaml

_REPO_ROOT = Path(__file__).parent.parent.resolve()
CONFIG_DIR = _REPO_ROOT / "config" / "collab"

# 实验模型配置
EXTRACT_MODEL = "gemma4-12b"
MANAGER_MODEL = "gemma4-12b"
ANSWER_MODEL = "gemma4-26B"
JUDGE_MODEL = "deepseek-v4-flash"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"

FILLER_LEVELS = ["N0", "N2", "N4", "N6", "N8"]

# 各方法独立的配置
METHOD_CONFIGS = {
    "rd": {
        "add_all": {"enabled": False},
        "relation_decision": {
            "enabled": True,
            "backend": "llm",
            "related_top_k": 3,
            "fusion_model": "",
            "condition_sim_threshold": 0.5,
            "pairwise_sim_threshold": 0.5,
        },
        "mem0": {"enabled": False},
        "evermemos": {"enabled": False},
    },
    "mem0": {
        "add_all": {"enabled": False},
        "relation_decision": {"enabled": False},
        "mem0": {"enabled": True, "related_top_k": 3, "related_aggregate_max": 10},
        "evermemos": {"enabled": False},
    },
    "evm": {
        "add_all": {"enabled": False},
        "relation_decision": {"enabled": False},
        "mem0": {"enabled": False},
        "evermemos": {"enabled": True, "similarity_threshold": 0.65, "max_time_gap_days": 7.0},
    },
}

SHARED_PARALLEL = {
    "extract_chunk_concurrency": 100,
    "ingest_relation_concurrency": 20,
    "ingest_episode_concurrency": {
        "relation_decision": 10,
        "mem0": 10,
        "add_all": 100,
        "evermemos": 5,
        "fusion_episodes": 100,
        "fusion_packages": 10,
    },
    "generate_parallel_episodes": 50,
    "generate_answer_concurrency": 2,
    "evaluate_max_concurrency": 8,
}

SHARED_TOKEN_LIMITS = {
    "extract_max_new_tokens": 2048,
    "ingest_relation_max_new_tokens": 256,
    "ingest_manager_max_new_tokens": 2048,
    "fusion_max_new_tokens": 512,
    "evaluate_max_new_tokens": 512,
}

SHARED_PROMPTS = {
    "relation_user_en": "RD_0_relation_classify.jinja",
    "relation_user_zh": "RD_0_relation_classify.jinja",
    "judge_template": "pipeline_judge.jinja",
}


def build_config(filler_level: str, method_key: str, methods: dict) -> dict:
    suffix = f"collab_{filler_level}_{method_key}"

    return {
        "experiment": {
            "benchmark": "lme_s",
            "suffix": suffix,
        },
        "models": {
            "extract": EXTRACT_MODEL,
            "manager": MANAGER_MODEL,
            "answer": ANSWER_MODEL,
            "judge": JUDGE_MODEL,
            "embedding": EMBEDDING_MODEL,
        },
        "extract": {
            "candidate_suffix": f"hybrid_filler_{filler_level}",
            "granularity": "4",
            "turn_overlap": "0",
            "language": "en",
            "aspect_templates": ["0_mem_extract_aspect_unified_en.jinja"],
        },
        "methods": methods,
        "generate": {
            "retrieve_topk": 50,
            "memory_token_limit": 256,
            "answer_stratified_sample": 0,
            "answer_sample_seed": 43,
            "show_memory_time": True,
            "hybrid": {"enabled": False},
        },
        "evaluate": {
            "use_cot": True,
        },
        "sweep": {
            "memory_token_limits": [256],
        },
        "replication": {
            "count": 1,
            "scope": "answer_judge",
        },
        "parallel": SHARED_PARALLEL,
        "token_limits": SHARED_TOKEN_LIMITS,
        "prompts": SHARED_PROMPTS,
    }


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    for filler in FILLER_LEVELS:
        for method_key, methods in METHOD_CONFIGS.items():
            config = build_config(filler, method_key, methods)
            filename = f"exp_{filler}_{method_key}.yaml"
            filepath = CONFIG_DIR / filename

            with open(filepath, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    config, f,
                    sort_keys=False, allow_unicode=True,
                    default_flow_style=False, indent=2,
                )

            count += 1
            print(f"  ✓ {filename}")

    print(f"\n生成完成：{count} 个配置文件 → {CONFIG_DIR}/")


if __name__ == "__main__":
    main()
