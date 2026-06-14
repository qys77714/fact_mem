# relation_classifier — 给 agent 的使用说明

输入一对**英文**记忆 `(old, new)`，输出它们的关系五分类。基于冻结的 Qwen3-0.6B + 训练好的线性探测头，test Macro F1 ≈ 0.936。

## 五类语义

所有记忆默认描述用户"现在"的状态。

| 标签 | 含义 |
|------|------|
| `IND` | 独立：不同属性，可同时成立（养猫 / 周末徒步） |
| `EQV` | 等价：同一事实不同表述（teacher / educator） |
| `OSN` | new ⊃ old，**新的更具体**（住北京 → 住北京朝阳区） |
| `NSO` | old ⊃ new，**旧的更具体**（住北京朝阳区 → 住北京） |
| `CON` | 矛盾：同一属性互斥（住北京 / 搬到上海） |

两个最易错的点：
- **顺序敏感**：`old`/`new` 不能颠倒。颠倒会让 OSN 与 NSO 互换。
- **时间变化算 CON**：换工作、搬家这类"先后变化"在本模型语义下判 `CON`，不是 IND。

## 用法一：Python 库

```python
import sys
sys.path.insert(0, "/绝对路径/relation_classifier")   # 或把本文件夹加进 PYTHONPATH
from classifier import RelationClassifier

clf = RelationClassifier()          # 加载一次，反复用；backbone 默认本地路径
clf.predict("I live in Beijing.", "I moved to Shanghai.")
# -> {"label": "CON", "label_id": 4, "probs": {"IND":.., "EQV":.., "OSN":.., "NSO":.., "CON":..}}

clf.predict_batch([("a", "b"), ("c", "d")])   # 批量更快
```

## 用法二：命令行

输入 jsonl 每行 `{"old": ..., "new": ...}`，输出每行追加 `label` + `probs`：

```bash
cd relation_classifier
CUDA_VISIBLE_DEVICES=1 python cli.py --input mem.jsonl --output preds.jsonl
```

最小可跑示例：`CUDA_VISIBLE_DEVICES=1 python examples/quickstart.py`

## 三条硬约束

1. **必须英文输入**。模型在英文数据上训练。中文记忆要先成对翻译成英文（old/new 一起翻译，保持信息量差异、措辞异同）再喂入，否则准确率会塌。
2. **backbone 走本地路径**。默认 `/mnt/data_oss/models/Qwen3-0.6B`。若目标机器路径不同，构造时传 `RelationClassifier(backbone_path="...")` 或设环境变量 `RC_BACKBONE_PATH`。
3. **`old`/`new` 顺序不可颠倒**（见上）。

## 已知局限

`IND`↔`CON` 边界最弱（IND recall 最低）。两句很短、没有共享字面线索时，"换城市/换工作"这类本应判 CON 的样本可能被误判成 IND。对这类结果建议留心或加规则兜底。

## 运行环境

- GPU 机器固定用 `CUDA_VISIBLE_DEVICES=1`。无 GPU 会自动退到 CPU（慢）。
- 依赖见 `requirements.txt`：torch / transformers / PyYAML。
- 首次构造会加载 1.5G backbone，约几秒到十几秒；之后实例可反复调用。

## 文件清单

```
classifier.py    核心库（自包含，RelationClassifier）
cli.py           命令行批量打标
config.yaml       推理配置（标签顺序、末token、分类头结构）
head_best.pt      分类头权重（2MB，val Macro F1=0.9407）
examples/         sample.jsonl + quickstart.py
requirements.txt  依赖
```
