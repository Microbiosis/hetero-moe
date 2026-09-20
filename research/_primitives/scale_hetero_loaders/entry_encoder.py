"""v37.0 — 自训练预测编码入口 (β 核心: 入口编码器参与端到端梯度)。

相对 v36 的进步:
  v36: 入口表征 c_struct = TabLDM 中间量 (冻结), 不为注入任务优化, 只取一次
  v37: 入口编码器 E(x_table) 可训练, 唯一训练信号是容器对目标的预测误差
        → E 必须自己发现"表格的哪些维度携带类别信息", 这才是预测编码语义

设计 (预测编码闭环):
  容器 h_base = MSA-4B 浅层(随机 token 文本, 与表格无关, 预计算一次)
  入口 E: x_table -> c  (MLP 6→256→512, 可训练)     ← β 核心: 这里学
  注入 P: c -> [B, 2560]  (Linear, 可训练)
  头  : h -> logits (Linear 2560->10)
  损失: CE(h_logits, y_table) 反传到 E+P+头; h_base 固定.
"""
import torch
import torch.nn as nn
import numpy as np


class EntryEncoder(nn.Module):
    """表格入口编码器 E: R^d_in → R^d_out. β 唯一新增的可训练入口组件."""

    def __init__(self, d_in, d_out, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.net(x)


def gen_table_classification(n, seed, n_classes=10):
    """合成符号规则任务: 类别由前三维符号模式决定 (表格信息必需).

    规则: cls = ((X[:,0] > 0)*4 + (X[:,1] > 0)*2 + (X[:,2] > 0)) % n_classes
    文本随机 token (与表格完全无关), 凸显入口贡献.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 6).astype(np.float32)
    y = ((X[:, 0] > 0).astype(int) * 4
         + (X[:, 1] > 0).astype(int) * 2
         + (X[:, 2] > 0).astype(int)) % n_classes
    return torch.tensor(X), torch.tensor(y, dtype=torch.long)


def container_text_repr(big_model, tok, n, s, seed):
    """随机 token → MSA-4B 浅层均值池化. 与表格无关 (β 的 h_base)."""
    rng = np.random.RandomState(seed + 777)
    ids = torch.tensor(rng.randint(100, 20000, (n, s)), dtype=torch.long)
    with torch.no_grad():
        h = big_model(ids).last_hidden_state.float().mean(dim=1)
    return h


def tabldm_passive_repr(n, seed, n_classes=10):
    """α2 对照: TabLDM 被动中间表征 (icl_predictor 输入), 逐样本冻结."""
    from tabldm import TabLDMClassifier
    import gc
    from .paths import TABLDM_CKPT
    clf = TabLDMClassifier(model_path=TABLDM_CKPT, device="cpu")
    rng = np.random.RandomState(42)
    X_train = rng.randn(12, 6).astype(np.float32)
    y_train = rng.randint(0, n_classes, 12)
    clf.fit(X_train, y_train)
    m = clf.model_
    reps = []
    h = m.icl_predictor.register_forward_hook(
        lambda mod, inp, out: reps.append(inp[0].detach() if isinstance(inp, tuple) else inp.detach()))
    Xq, yq = gen_table_classification(n, seed, n_classes)
    clf.predict_proba(Xq.numpy())
    h.remove()
    r = reps[0]
    rep_q = r[:, -n:, :].mean(dim=0).float()
    del clf, m
    gc.collect()
    return rep_q, yq