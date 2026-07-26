#!/usr/bin/env python3
"""生成实验配置文件：3 模型 × 5 filler level × 3 方法组合 = 45 个 YAML。

用法：
    PYTHONPATH=src uv run --no-sync python script/generate_exp_configs.py
"""

from __future__ import annotations

from pathlib import Path
import yaml

_REPO_ROOT = Path(__file__).parent.parent.resolve()
CONFIG_DIR = _REPO_ROOT / "config"

MODELS = {
    "qwen3-4b": "Qwen3-4B",
    "qwen3-8b": "Qwen3-8B",
    "qwen3-32b": "Qwen3-32B",
}

FILLER_LEVELS = ["N0", "N2", "N4", "N6", "N8"]

# 方法组合定义
METHOD_GROUPS = {
    "rd_addall": {
        "add_all": {"enabled": True},
        "relation_decision": {
            "enabled": True,
            "backend": "llm",
            "related_top_k": 3,
            "fusion_model": "",
            "condition_sim_threshold": 0.5,
            "pairwise_sim_threshold": 0.5,
        },
        "mem0": {"enabled": False, "related_top_k": 3, "related_aggregate_max": 10},
        "evermemos": {"enabled": False, "similarity_threshold": 0.65, "max_time_gap_days": 7.0},
        "amac": {"enabled": False},
        "zep": {"enabled": False},
    },
    "mem0": {
        "add_all": {"enabled": False},
        "relation_decision": {"enabled": False},
        "mem0": {"enabled": True, "related_top_k": 3, "related_aggregate_max": 10},
        "evermemos": {"enabled": False, "similarity_threshold": 0.65, "max_time_gap_days": 7.0},
        "amac": {"enabled": False},
        "zep": {"enabled": False},
    },
    "evm": {
        "add_all": {"enabled": False},
        "relation_decision": {"enabled": False},
        "mem0": {"enabled": False, "related_top_k": 3, "related_aggregate_max": 10},
        "evermemos": {"enabled": True, "similarity_threshold": 0.65, "max_time_gap_days": 7.0},
        "amac": {"enabled": False},
        "zep": {"enabled": False},
    },
}

# 共享的配置段
SHARED_PARALLEL = {
    "extract_chunk_concurrency": 100,
    "ingest_relation_concurrency": 20,
    "ingest_episode_concurrency": {
        "relation_decision": 10,
        "mem0": 10,
        "add_all": 100,
        "zep": 50,
        "amac": 100,
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


def build_config(
    model_key: str,
    model_name: str,
    filler_level: str,
    method_group: str,
    methods: dict,
) -> dict:
    """构建单个实验 YAML 配置。"""
    suffix = f"hybrid_filler_{filler_level}_{model_key}_{method_group}"

    return {
        "experiment": {
            "benchmark": "lme_s",
            "suffix": suffix,
        },
        "models": {
            "extract": "gemma4-26B",
            "manager": model_name,
            "answer": "gemma4-26B",
            "judge": "qwen3-max",
            "embedding": "qwen3-embedding-0.6b",
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
            "memory_token_limit": 256,  # sweep 默认值
            "answer_stratified_sample": 0,
            "answer_sample_seed": 43,
            "show_memory_time": True,
            "hybrid": {
                "enabled": False,  # 只用 embedding，不用 BM25
                "dense_weight": 0.8,
                "bm25_weight": 0.2,
                "pool_mult": 4,
            },
        },
        "evaluate": {
            "use_cot": True,
            "judge_stratified_sample": 0,
            "judge_sample_seed": 43,
        },
        "sweep": {
            "memory_token_limits": [256, 512],
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

    config_count = 0
    for filler in FILLER_LEVELS:
        for model_key, model_name in MODELS.items():
            for method_key, methods in METHOD_GROUPS.items():
                config = build_config(model_key, model_name, filler, method_key, methods)
                filename = f"exp_{filler}_{model_key}_{method_key}.yaml"
                filepath = CONFIG_DIR / filename

                with open(filepath, "w", encoding="utf-8") as f:
                    yaml.safe_dump(
                        config,
                        f,
                        sort_keys=False,
                        allow_unicode=True,
                        default_flow_style=False,
                        indent=2,
                    )

                config_count += 1
                print(f"  ✓ {filename}")

    print(f"\n生成完成：{config_count} 个配置文件 → {CONFIG_DIR}")


if __name__ == "__main__":
    main()
