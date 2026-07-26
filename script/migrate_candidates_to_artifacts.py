#!/usr/bin/env python3
"""将 legacy MemDB/candidates/ 下的候选数据迁移到新 artifact 布局。

对每个 filler level (N0/N2/N4/N6/N8)，计算 content-addressed candidate_id，
将旧目录下的 episode JSON + extract_progress.state 复制过去，并写入 stage_manifest.json。

用法：
    PYTHONPATH=src uv run --no-sync python script/migrate_candidates_to_artifacts.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.resolve()
_SRC = _REPO_ROOT / "src"
sys.path.insert(0, str(_SRC))

from utils.config import ExperimentConfig
from utils.experiment_artifacts import ArtifactLayout, ExperimentIdentity

LEGACY_CANDIDATES_ROOT = _REPO_ROOT / "MemDB" / "candidates"
ARTIFACTS_ROOT = _REPO_ROOT / "artifacts"
FILLER_LEVELS = ["N0", "N2", "N4", "N6", "N8"]

# 所有迁移共享的基础 extract 配置（与将要生成的 YAML 保持一致）
BASE_EXTRACT_CONFIG = {
    "experiment": {"benchmark": "lme_s", "suffix": "_migration"},
    "models": {
        "extract": "gemma4-26B",
        "manager": "gemma4-26B",  # manager 不影响 candidate fingerprint
        "answer": "gemma4-26B",
        "judge": "qwen3-max",
        "embedding": "qwen3-embedding-0.6b",
    },
    "extract": {
        "candidate_suffix": "",  # 由各 N 覆盖
        "granularity": "4",
        "turn_overlap": "0",
        "language": "en",
        "aspect_templates": ["0_mem_extract_aspect_unified_en.jinja"],
    },
    "methods": {"add_all": {"enabled": True}},  # 至少一个 enabled method
    "generate": {
        "retrieve_topk": 50,
        "memory_token_limit": 256,
        "answer_stratified_sample": 0,
        "show_memory_time": True,
        "hybrid": {"enabled": False, "dense_weight": 0.8, "bm25_weight": 0.2, "pool_mult": 4},
    },
    "evaluate": {"use_cot": True},
    "token_limits": {
        "extract_max_new_tokens": 2048,
        "ingest_relation_max_new_tokens": 256,
        "ingest_manager_max_new_tokens": 2048,
        "fusion_max_new_tokens": 512,
        "evaluate_max_new_tokens": 512,
    },
    "sweep": {"memory_token_limits": []},
    "replication": {"count": 1, "scope": "answer_judge"},
}


def migrate_one_filler_level(filler_level: str) -> Path:
    """迁移单个 filler level 的候选数据，返回新布局下的 candidate_dir。"""
    legacy_dir = LEGACY_CANDIDATES_ROOT / f"lme_s_gemma4-26B_hybrid_filler_{filler_level}"
    if not legacy_dir.is_dir():
        raise FileNotFoundError(f"Legacy candidate dir not found: {legacy_dir}")

    # 构建与生产 config 一致的 extract 配置
    config_data = json.loads(json.dumps(BASE_EXTRACT_CONFIG))  # deep copy
    config_data["extract"]["candidate_suffix"] = f"hybrid_filler_{filler_level}"

    cfg = ExperimentConfig.model_validate(config_data)
    resolved = cfg.model_dump(mode="json")

    # 使用框架自身逻辑计算 candidate_id
    identity = ExperimentIdentity(
        resolved_config=resolved,
        source_config_path=_REPO_ROOT / "config" / "_migration_temp.yaml",
        repo_root=_REPO_ROOT,
        artifacts_root=ARTIFACTS_ROOT / "runs",
    )
    layout = ArtifactLayout(
        identity=identity,
        template_root=_SRC / "prompts" / "templates",
        stages_root=ARTIFACTS_ROOT / "stages",
    )

    candidate_id = layout.candidate_id
    candidate_dir = layout.candidate_dir

    print(f"\n  filler={filler_level}")
    print(f"    candidate_id: {candidate_id}")
    print(f"    candidate_dir: {candidate_dir}")

    if candidate_dir.exists():
        # 检查是否已有文件
        existing_files = list(candidate_dir.glob("*.json"))
        if existing_files:
            print(f"    ⏭  已有 {len(existing_files)} 个文件，跳过")
            return candidate_dir

    # 创建目录并复制文件
    candidate_dir.mkdir(parents=True, exist_ok=True)

    file_count = 0
    for src_file in sorted(legacy_dir.iterdir()):
        if src_file.is_file() and src_file.suffix == ".json":
            dst = candidate_dir / src_file.name
            shutil.copy2(src_file, dst)
            file_count += 1

    print(f"    ✓ 已复制 {file_count} 个文件")

    # 写入 stage_manifest.json
    with layout.candidate_lock():
        manifest_path = layout.write_candidate_stage_manifest()
        print(f"    ✓ stage_manifest: {manifest_path}")

    return candidate_dir


def main() -> None:
    print("=" * 60)
    print("  候选数据迁移: MemDB/candidates/ → artifacts/stages/candidates/")
    print("=" * 60)

    for level in FILLER_LEVELS:
        try:
            migrate_one_filler_level(level)
        except FileNotFoundError as e:
            print(f"\n  ✗ {e}")
            sys.exit(1)

    print(f"\n{'=' * 60}")
    print("  迁移完成")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
