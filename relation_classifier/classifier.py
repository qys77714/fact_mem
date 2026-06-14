"""relation_classifier — 原子化记忆关系五分类推理库（自包含）。

输入一对英文记忆 (old, new)，输出五分类关系：
  IND 独立 / EQV 等价 / OSN 新更具体 / NSO 旧更具体 / CON 矛盾

范式：冻结 Qwen3-0.6B 作固定特征器，只跑训练好的线性探测头。
本文件自包含，不依赖训练目录任何代码。详见 AGENT.md。
"""
import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))

# 标签顺序固定，对应分类头输出维度，切勿改动
LABELS = ["IND", "EQV", "OSN", "NSO", "CON"]
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

# backbone 默认共享路径（已挂载 data_oss 的机器可直接用）
DEFAULT_BACKBONE = "/mnt/data_oss/models/Qwen3-0.6B"


def format_pair(old: str, new: str) -> str:
    """与训练完全一致的输入拼接。改这里就会线上线下不一致。"""
    return f"old: {old}\nnew: {new}"


def gather_last_token(hidden, attention_mask):
    """从 [B,T,H] 取每条样本真实末 token（右 padding 安全）。返回 [B,H]。

    不可用 hidden[:,-1]——右 padding 下会取到 pad 位。
    """
    last_idx = attention_mask.sum(dim=1) - 1
    b = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[b, last_idx]


class ProbeHead(nn.Module):
    """线性探测分类头：Linear→ReLU→Dropout→Linear。结构须与训练一致。"""
    def __init__(self, hidden_size, mid_dim, num_classes, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class RelationClassifier:
    """老/新记忆对 → 五分类关系。

    一次加载（backbone + 分类头），可反复调用 predict / predict_batch。
    线程不安全（共享 backbone 前向）；多线程请各持实例或加锁。

    参数:
        backbone_path: Qwen3-0.6B 本地目录。默认 /mnt/data_oss/models/Qwen3-0.6B，
                       也可经环境变量 RC_BACKBONE_PATH 覆盖。
        ckpt_path:     分类头权重，默认包内 head_best.pt。
        config_path:   推理配置，默认包内 config.yaml。
        device:        默认 cuda 可用即用，否则 cpu。
    """

    def __init__(self, backbone_path=None, ckpt_path=None,
                 config_path=None, device=None):
        cfg_path = config_path or os.path.join(HERE, "config.yaml")
        self.cfg = yaml.safe_load(open(cfg_path))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = self.cfg["max_length"]

        self.backbone_path = (backbone_path
                              or os.environ.get("RC_BACKBONE_PATH")
                              or DEFAULT_BACKBONE)
        if not os.path.isdir(self.backbone_path):
            raise FileNotFoundError(
                f"backbone 目录不存在: {self.backbone_path}\n"
                f"请确认已挂载 data_oss，或传 backbone_path / 设 RC_BACKBONE_PATH。")

        from transformers import AutoTokenizer, AutoModel
        self.tok = AutoTokenizer.from_pretrained(self.backbone_path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"   # gather_last_token 依赖右 padding

        dt = getattr(torch, self.cfg["dtype"])
        self.backbone = AutoModel.from_pretrained(self.backbone_path, dtype=dt)
        self.backbone.eval().to(self.device)
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        ckpt_path = ckpt_path or os.path.join(HERE, "head_best.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"分类头权重不存在: {ckpt_path}")
        ck = torch.load(ckpt_path, map_location=self.device)
        self.head = ProbeHead(self.cfg["hidden_size"], self.cfg["mid_dim"],
                              self.cfg["num_classes"], self.cfg["dropout"])
        self.head.load_state_dict(ck["state_dict"])
        self.head.eval().to(self.device)
        self.ckpt_meta = {k: ck.get(k) for k in ("val_macro_f1", "epoch")}

    @torch.no_grad()
    def _features(self, texts):
        """texts: List[str] -> [N,H] fp32 特征（与训练抽取逐位对齐）。"""
        feats = []
        bs = self.cfg["extract_batch_size"]
        for i in range(0, len(texts), bs):
            enc = self.tok(texts[i:i + bs], padding=True, truncation=True,
                           max_length=self.max_length,
                           return_tensors="pt").to(self.device)
            out = self.backbone(**enc)
            f = gather_last_token(out.last_hidden_state, enc["attention_mask"])
            feats.append(f.float().cpu())
        return torch.cat(feats, dim=0)

    @torch.no_grad()
    def predict_batch(self, pairs, return_probs=True):
        """pairs: List[(old, new)] -> List[dict]。

        每个 dict: {"label", "label_id", "probs"(可选，按 LABELS 顺序的类别概率)}。
        """
        if not pairs:
            return []
        texts = [format_pair(o, n) for o, n in pairs]
        X = self._features(texts).to(self.device)
        prob = F.softmax(self.head(X), dim=1).cpu()
        ids = prob.argmax(1).tolist()
        results = []
        for row, cid in zip(prob, ids):
            r = {"label": ID2LABEL[cid], "label_id": cid}
            if return_probs:
                r["probs"] = {LABELS[j]: round(float(row[j]), 4)
                              for j in range(len(LABELS))}
            results.append(r)
        return results

    def predict(self, old, new, return_probs=True):
        """单条：(old, new) -> dict。"""
        return self.predict_batch([(old, new)], return_probs=return_probs)[0]
