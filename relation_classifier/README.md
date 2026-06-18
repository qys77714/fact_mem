# relation_classifier

原子化记忆关系五分类器（IND/EQV/OSN/NSO/CON）的可交付推理包。冻结 Qwen3-0.6B + 线性探测头，test Macro F1 ≈ 0.953。

**给 agent / 集成方看的完整说明在 [`AGENT.md`](AGENT.md)。**

## 快速开始

```bash
cd relation_classifier
CUDA_VISIBLE_DEVICES=1 python examples/quickstart.py
```

```python
from classifier import RelationClassifier
clf = RelationClassifier()
clf.predict("I live in Beijing.", "I moved to Shanghai.")   # -> {"label": "CON", ...}
```

## 要点

- 输入必须是**英文**记忆对 `(old, new)`，`old`/`new` 顺序不可颠倒。
- backbone 默认本地路径 `/mnt/data_oss/models/Qwen3-0.6B`，可经 `backbone_path` 参数或 `RC_BACKBONE_PATH` 环境变量覆盖。
- 依赖：`pip install -r requirements.txt`（torch / transformers / PyYAML）。

详见 `AGENT.md`。
