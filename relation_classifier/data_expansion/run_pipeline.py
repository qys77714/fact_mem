"""端到端训练数据扩充流水线编排。"""

import os
import sys
import argparse
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG = os.path.join(HERE, "config.yaml")


def load_config(config_path):
    cfg = {}
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    return cfg


def main():
    ap = argparse.ArgumentParser(description="训练数据扩充流水线")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="YAML 配置文件")
    ap.add_argument("--steps", default="1,2,3,4", help="要运行的步骤，逗号分隔")
    ap.add_argument("--limit", type=int, default=0, help="Step 3 仅处理前 N 对（调试用）")
    ap.add_argument("--dry-run", action="store_true", help="仅打印配置，不实际运行")
    args = ap.parse_args()

    cfg = load_config(args.config)
    steps = [int(s.strip()) for s in args.steps.split(",")]

    # 解析路径
    repo_root = os.path.abspath(os.path.join(HERE, "..", ".."))

    persona_dir = cfg.get("personamem_dir", "data/raw_data/PersonaMem-v2/data/raw_data")
    if not os.path.isabs(persona_dir):
        persona_dir = os.path.join(repo_root, persona_dir)

    data_dir = cfg.get("data_dir", os.path.join(HERE, "data"))
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(repo_root, data_dir)

    mem_path = cfg.get("atomic_memories_path", os.path.join(data_dir, "personamem_atomic_memories.jsonl"))
    if not os.path.isabs(mem_path):
        mem_path = os.path.join(repo_root, mem_path)

    pairs_path = cfg.get("pairs_all_path", os.path.join(data_dir, "pairs_all.jsonl"))
    if not os.path.isabs(pairs_path):
        pairs_path = os.path.join(repo_root, pairs_path)

    judged_path = cfg.get("pairs_judged_path", os.path.join(data_dir, "pairs_with_judgments.jsonl"))
    if not os.path.isabs(judged_path):
        judged_path = os.path.join(repo_root, judged_path)

    train_path = cfg.get("training_data_path", os.path.join(data_dir, "training_data_expanded.jsonl"))
    if not os.path.isabs(train_path):
        train_path = os.path.join(repo_root, train_path)

    original_path = cfg.get("original_training_data", "non_ind.jsonl")
    if not os.path.isabs(original_path):
        original_path = os.path.join(repo_root, original_path)

    if args.dry_run:
        print("=== Dry Run ===")
        print(f"Config: {args.config}")
        print(f"Steps: {steps}")
        print(f"PersonaMem dir: {persona_dir}")
        print(f"Atomic memories: {mem_path}")
        print(f"Pairs: {pairs_path}")
        print(f"Judged: {judged_path}")
        print(f"Training data: {train_path}")
        print(f"Original data: {original_path}")
        return

    sys.path.insert(0, HERE)

    # Step 1
    if 1 in steps:
        print("=" * 60)
        print("Step 1: 原子记忆提取 & 主语改写")
        print("=" * 60)
        from step1_extract_preferences import extract_personamem_preferences
        extract_personamem_preferences(persona_dir, mem_path)

    # Step 2
    if 2 in steps:
        print("=" * 60)
        print("Step 2: (old, new) 配对构造")
        print("=" * 60)
        from step2_construct_pairs import build_all_pairs
        build_all_pairs(mem_path, pairs_path, cfg)

    # Step 3
    if 3 in steps:
        print("=" * 60)
        print("Step 3: 双裁判判断")
        print("=" * 60)
        from step3_dual_judge import judge_all_pairs

        if args.limit > 0:
            from step3_dual_judge import load_pairs
            pairs = load_pairs(pairs_path)[:args.limit]
            tmp_path = pairs_path + ".tmp_limit"
            import json
            with open(tmp_path, "w") as f:
                for p in pairs:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            pairs_path = tmp_path

        judge_all_pairs(pairs_path, judged_path, cfg)

    # Step 4
    if 4 in steps:
        print("=" * 60)
        print("Step 4: 生成训练数据")
        print("=" * 60)
        from step4_generate_training_data import generate_training_data
        generate_training_data(judged_path, original_path, train_path, cfg)

    print("=" * 60)
    print("流水线完成!")
    print(f"最终训练数据: {train_path}")


if __name__ == "__main__":
    main()
