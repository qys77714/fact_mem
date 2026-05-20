# MEME gold_facts 评测

本目录包含 MEME 数据集「仅用 gold_facts 作记忆库」的 oracle 实验代码与结果。

## 目录结构

```
test/meme/
├── meme_gold_facts_eval.py    # 主评测脚本
├── build_meme_gold_facts_html.py  # 从 pred.jsonl 生成 HTML 查看器
├── run_meme_gold_facts.sh     # 一键运行（默认 --retrieve-topk 20 --html）
├── output/                    # 评测输出（pred.jsonl、eval_judge.json、qa_viewer.html）
└── README.md
```

## 运行

```bash
cd fact_memory

# 需先启动 vLLM chat (:7111) 与 embedding (:7110)
bash test/meme/run_meme_gold_facts.sh

# oracle 上界（全量 gold_facts，无需 embedding）
uv run python test/meme/meme_gold_facts_eval.py --use-all-facts --html

# 仅生成 HTML
uv run python test/meme/build_meme_gold_facts_html.py \
  --pred test/meme/output/<run_dir>/pred.jsonl
```

数据集路径：`data/raw_data/MEME/meme_nofiller.json`（仓库根目录下）。
