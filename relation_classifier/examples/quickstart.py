"""最小可跑示例：单条 + 批量预测。

    cd relation_classifier
    CUDA_VISIBLE_DEVICES=1 python examples/quickstart.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classifier import RelationClassifier

clf = RelationClassifier()   # backbone 默认 /mnt/data_oss/models/Qwen3-0.6B
print(f"loaded: val_macro_f1={clf.ckpt_meta['val_macro_f1']:.4f} device={clf.device}\n")

# 单条
r = clf.predict("I live in Beijing.", "I live in Chaoyang District, Beijing.")
print("single:", r)

# 批量
pairs = [
    ("I work as a teacher.", "I am an educator."),       # 期望 EQV
    ("I live in Beijing.", "I moved to Shanghai."),      # 期望 CON
    ("I have a cat.", "I enjoy hiking on weekends."),     # 期望 IND
]
print("\nbatch:")
for (o, n), res in zip(pairs, clf.predict_batch(pairs)):
    print(f"  [{res['label']}] {o!r} -> {n!r}")
