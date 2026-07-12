#!/usr/bin/env python3
"""
为 confusion memory 实验批量生成 config YAML。

用法:
  uv run --no-sync python script/gen_confusion_configs.py [--distractors 0,2,4,8]

输出:
  - config/lme_conf_v3_filler_N0.yaml
  - config/lme_conf_v3_filler_N2.yaml
  - config/lme_conf_v3_filler_N4.yaml
  - config/lme_conf_v3_filler_N8.yaml

每个 config 同时启用 add_all 和 relation_decision (backend=llm, v3 模板)。
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

YAML_TEMPLATE = """\
# fact_memory LME Confusion Memory 对比实验配置 (N={n})
# 自动生成: script/gen_confusion_configs.py
# 用法: PYTHONPATH=src uv run --no-sync python run_exp_lme.py --config config/lme_conf_v3_filler_N{n}.yaml --stages ingest,generate,evaluate

experiment:
  benchmark: lme_s
  suffix: confv3_filler_N{n}

models:
  extract: gemma4-26B
  manager: gemma4-26B
  answer: gemma4-26B
  judge: gpt-4o-mini
  embedding: qwen3-embedding-0.6b

extract:
  candidate_suffix: conf_v3_filler_N{n}   # 指向预制候选目录 MemDB/candidates/lme_conf_v3_filler_N{n}/
  granularity: 4
  turn_overlap: 0
  language: en
  aspect_templates:
    - "0_mem_extract_aspect_unified_en.jinja"

methods:
  add_all:
    enabled: true

  relation_decision:
    enabled: true
    backend: llm                       # 使用 gemma4-26B 直接分类
    related_top_k: 3
    fusion_model: ""
    cascade_enabled: false
    deletion_enabled: false
    condition_sim_threshold: 0.5
    pairwise_sim_threshold: 0.5

  amac:
    enabled: false

  zep:
    enabled: false

  mem0:
    enabled: false

  evermemos:
    enabled: false

generate:
  retrieve_topk: 50
  memory_token_limit: 512
  answer_stratified_sample: 0          # 0 = 全量 470 题
  answer_sample_seed: 43
  show_memory_time: true

  hybrid:
    enabled: true
    dense_weight: 0.8
    bm25_weight: 0.2
    pool_mult: 4

evaluate:
  use_cot: true
  judge_stratified_sample: 0
  judge_sample_seed: 43

parallel:
  extract_chunk_concurrency: 100
  ingest_relation_concurrency: 50
  ingest_episode_concurrency:
    relation_decision: 40
    mem0: 50
    add_all: 100
    zep: 50
    amac: 100
    evermemos: 5
    fusion_episodes: 100
    fusion_packages: 10
  generate_parallel_episodes: 50
  generate_answer_concurrency: 2
  evaluate_max_concurrency: 8

token_limits:
  extract_max_new_tokens: 2048
  ingest_relation_max_new_tokens: 256
  ingest_manager_max_new_tokens: 2048
  fusion_max_new_tokens: 512
  evaluate_max_new_tokens: 512

debug:
  evaluate_print_one_sample: false

prompts:
  relation_system_en: "lme_relation_classification_system_en_v3.jinja"
  relation_system_zh: "lme_relation_classification_system_zh_v2.jinja"
  relation_user: "lme_relation_classification_user.jinja"
  fusion_bundle_en: "fuse_memory_bundle_en_v3.jinja"
  fusion_bundle_zh: ""
  fusion_edge_labels_en: "fuse_memory_bundle_edge_labels_en_v2.jinja"
  fusion_edge_labels_zh: "fuse_memory_bundle_edge_labels_zh_v2.jinja"
  judge_oqa: "pipeline_eval_oqa.jinja"
  judge_mcq: "pipeline_eval_mcq.jinja"
  judge_system: "pipeline_eval_system.jinja"
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 confusion 实验 config")
    parser.add_argument("--distractors", default="0,2,4,8",
                        help="逗号分隔的 distractor 数量 (default: 0,2,4,8)")
    args = parser.parse_args()

    ns = [int(x.strip()) for x in args.distractors.split(",")]
    config_dir = _REPO / "config"

    for n in ns:
        yaml_content = YAML_TEMPLATE.format(n=n)
        # 清理多余空行
        yaml_content = yaml_content.strip() + "\n"
        out_path = config_dir / f"lme_conf_v3_filler_N{n}.yaml"
        with open(out_path, "w") as f:
            f.write(yaml_content)
        print(f"生成: {out_path}")

    print(f"\n完成。{len(ns)} 个 config 已写入 {config_dir}/")


if __name__ == "__main__":
    main()
