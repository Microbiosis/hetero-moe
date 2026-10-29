"""V56.0 — 维度作为容量分配 (capacity allocation), 而非容量通胀.

为什么这一代存在
----------------
v39 证明了「融合头必须非线性」: `concat + MLP` 解开 XOR (0.997), `add` 不能
(0.781)。但那一代有两个量从头到尾是**固定超参, 从未当成变量**:

    hidden = 128                    (融合头瓶颈宽度)
    ProjHead: c(512) → D_BIG(2560)  (入口被投到容器宽度)

于是四个问题一直没有答案:

  Q1 宽度是否是约束?    hidden 从 4 扫到 2048, 看 XOR 在哪个宽度被解开
  Q2 投影是否必要?      c(512)→2560 (v39 原样)  vs  保留原生 512
  Q3 空间被填满了吗?    训练后 hidden 激活的有效秩 vs hidden 宽度
  Q4 哪个模态在被用?    按块消融 (zero-out) 测准确率掉落

只有 Q3 给出「秩随宽度饱和」时, 「高维空间」才是**约束**而不是**比喻**。

与 v39 的关系
-------------
本代**不重新发明协议**: 数据 (Adult XOR)、容器表征 h_base、TabLDM 被动入口
c_passive 全部复用 v39/v36 的实现并冻结; 只把 hidden 宽度与投影方式变成变量。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "eff_rank_pr", "eff_rank_entropy", "rank_report",
    "make_head", "build_input", "train_head", "block_ablate", "n_params",
    "WIDTHS", "REPRS", "HEADS", "PREDICTIONS",
]

# ----------------------------------------------------------------------
# 1) 维度诊断
# ----------------------------------------------------------------------

def eff_rank_pr(X: torch.Tensor) -> float:
    """参与率 (participation ratio): (Σσ)² / Σσ² —— 越大说明占用越多方向。"""
    X = X.detach().float()
    X = X - X.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(X)
    s = s[s > 1e-12]
    if s.numel() == 0:
        return 0.0
    return float(s.sum() ** 2 / (s ** 2).sum())


def eff_rank_entropy(X: torch.Tensor) -> float:
    """熵有效秩: exp(−Σ p log p), p = σ/Σσ —— 对长尾更敏感。"""
    X = X.detach().float()
    X = X - X.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(X)
    s = s[s > 1e-12]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def rank_report(name: str, X: torch.Tensor) -> dict:
    """一个张量占了多少维 —— 这是「维度」这个概念的可执行形式。"""
    d = X.shape[1]
    pr, ent = eff_rank_pr(X), eff_rank_entropy(X)
    return {"name": name, "ambient_dim": d, "eff_rank_pr": round(pr, 2),
            "eff_rank_entropy": round(ent, 2),
            "occupied_frac": round(pr / d, 5) if d else None}


# ----------------------------------------------------------------------
# 2) 表征与融合头
# ----------------------------------------------------------------------

REPRS = ("proj", "native")
HEADS = ("linear", "mlp")
WIDTHS = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)

D_ENTRY = 512
D_BIG = 2560
N_CLS = 2


def build_input(h_base: torch.Tensor, c: torch.Tensor, repr_mode: str,
                P: nn.Module | None = None) -> tuple[torch.Tensor, int, int]:
    """把两侧拼成融合头的输入。

    proj   : c(512) --P--> 2560, 再拼 h_base(2560)        → 5120 维 (v39 原样)
    native : c(512) 原样保留, 再拼 h_base(2560)            → 3072 维

    返回 (x, d_hbase_block, d_entry_block) —— 块边界供消融使用。
    """
    if repr_mode == "proj":
        assert P is not None, "proj 模式需要传入投影 P"
        cp = P(c)
    elif repr_mode == "native":
        cp = c
    else:
        raise ValueError(f"未知 repr_mode={repr_mode!r}, 应为 {REPRS}")
    x = torch.cat([h_base, cp], dim=1)
    return x, h_base.shape[1], cp.shape[1]


def make_head(mode: str, d_in: int, hidden: int = 128, n_cls: int = N_CLS) -> nn.Module:
    """融合头。

    linear : Linear(d_in → n_cls)
             ⚠️ 数学上等价于 add 头: W₁·h + W₂·c + b = f(h) + g(c), XOR 无解。
             保留它是为了把「concat 本身」与「非线性」分开。
    mlp    : Linear(d_in → hidden) → GELU → Linear(hidden → n_cls)
    """
    if mode == "linear":
        return nn.Linear(d_in, n_cls)
    if mode == "mlp":
        return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, n_cls))
    raise ValueError(f"未知 head mode={mode!r}, 应为 {HEADS}")


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ----------------------------------------------------------------------
# 3) 训练 / 评估 / 消融
# ----------------------------------------------------------------------

def train_head(x: torch.Tensor, y: torch.Tensor, tr_idx: torch.Tensor,
               te_idx: torch.Tensor, hidden: int = 128, head_mode: str = "mlp",
               steps: int = 500, seed: int = 0, lr: float = 1e-3,
               batch: int = 64) -> tuple[nn.Module, float]:
    """训练一个融合头, 返回 (head, test_acc)。容器与入口全部冻结, 只有头在学。"""
    torch.manual_seed(seed)
    head = make_head(head_mode, x.shape[1], hidden)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(steps):
        b = tr_idx[torch.randperm(len(tr_idx))[:batch]]
        loss = F.cross_entropy(head(x[b]), y[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = float((head(x[te_idx]).argmax(1) == y[te_idx]).float().mean())
    return head, acc


def hidden_act_rank(head: nn.Module, x: torch.Tensor) -> float | None:
    """训练后 hidden 层激活占了几个方向 —— Q3 的核心读数。

    宽度 2048 但有效秩只有 ~20, 说明「空间在那里, 没有东西去填」。
    这个数与 hidden 宽度同时报告, 两者分叉处就是容量的真实边界。
    """
    if not isinstance(head, nn.Sequential):
        return None
    with torch.no_grad():
        h = F.gelu(head[0](x))
    return eff_rank_pr(h)


def block_ablate(head: nn.Module, x: torch.Tensor, y: torch.Tensor,
                 te_idx: torch.Tensor, d_hbase: int, d_entry: int) -> dict:
    """按模态块消融: 把某一块置零, 看准确率掉多少。

    Q4 的答案。若置零入口块后准确率不掉, 说明这个模态没被用 —— 这正是
    「注入真实信息 vs 不注入」差别消失时的指纹。
    """
    def acc_of(xx):
        with torch.no_grad():
            return float((head(xx[te_idx]).argmax(1) == y[te_idx]).float().mean())

    xa = x.clone()
    xa[:, :d_hbase] = 0.0                  # 抹掉容器侧
    xb = x.clone()
    xb[:, d_hbase:d_hbase + d_entry] = 0.0  # 抹掉入口侧
    base = acc_of(x)
    return {"full": base, "drop_hbase": base - acc_of(xa),
            "drop_entry": base - acc_of(xb)}


# ----------------------------------------------------------------------
# 4) 预登记判决 (先写死, 事后不得改)
# ----------------------------------------------------------------------

PREDICTIONS = {
    "P1": "linear 头在任何宽度下都解不开 XOR (acc ≤ 0.80) —— concat 本身不加价值, 只有非线性加",
    "P2": "MLP 在 hidden=16 时已达到 hidden=2048 的 95% 准确率 (维度不是约束)",
    "P3": "proj (c→2560) 与 native (c 保持 512) 的准确率差 ≤ 0.03 (线性投影不增信息)",
    "P4": "hidden 激活有效秩随宽度饱和: width 1024/2048 的有效秩 < 宽度的 5%",
}
