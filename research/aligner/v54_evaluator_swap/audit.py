"""V54.0 — 评价器换轨: MSE → {高斯 NLL, Cross-Entropy} (同网格, 不降级).

定位
----
v49–v53 的全部结论都建立在一个评价器 (MSE) 上。而 v53 的目标块是块状 one-hot,
于是 `MSE(预测全零) = 1/n_classes` 是**精确等式**; v53 真实 n_classes=6 时 = 0.1667。
在那个量级上比较"融合 vs 最优单专家", 比的是两个同样坏掉的模型。

本代**不缩小实验**: 数据/划分/探针/lift/种子/步数全部与 v53 同构,
只把评价器从一个变成三个, 且三档**共用同一份初始化** (配对)。

为什么是"配对"而不是"各跑一遍"
------------------------------
若三档各自采样, 差异里混着采样噪声。配对后任何两档的差异**只能**来自损失函数。
实现: 每个 (arm, D, seed) 下, 三档各自 `torch.manual_seed(seed)` 重建模型,
权重逐位相同; 批顺序由同一个 `Generator` 种子决定。

参数等量纪律 (v49 起立的规矩)
----------------------------
换评价器最容易踩的坑: 输出头变大会偷偷改变参数量, 于是"哪把尺子好"和"哪个模型大"
混在一起。三档必须逐参数同构:

    mse       输出 [n,S,D] → 直接 MSE                        +0 参数
    ce        [n,S,D] → block-mean pool → [n,S,K] → CE        +1 参数 (可学习温度 τ)
    gauss_nll [n,S,D] + Linear(D→1) 逐样本标量 logvar         +D+1 参数

CE 用 block 均值池化而非另接一个 D→K 线性头, 是为了让 lift 主体的输出张量
**逐字节相同**, 只有最后一步不同。τ 是唯一的额外参数 (softplus 保正)。
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用 v53 的数据 / 探针 / lift —— 这是"不降级"的结构性保证:
# 数据生成、值域分割、三族探针、残差抬升全部沿用, 不可能在 v54 里悄悄变掉。
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from research.aligner.v53_third_audit.audit import (  # noqa: E402
    CLASSES, N_CLASSES, ARMS, PREDICTIONS,
    make_split, build_targets, probe_accuracy_all, consistency,
    ResidualLift, LinearLift,
)

__all__ = ["CLASSES", "N_CLASSES", "ARMS", "EVALUATORS", "EvaluatorHead",
           "eval_loss", "baselines", "decode_accuracy", "calibration",
           "PREDICTIONS", "PREDICTIONS_V54",
           "make_split", "build_targets", "probe_accuracy_all",
           "consistency", "ResidualLift", "LinearLift"]

EVALUATORS = ("mse", "gauss_nll", "ce")


# ----------------------------------------------------------------------
# 1) 三档评价器
# ----------------------------------------------------------------------

class EvaluatorHead(nn.Module):
    """把 lift 的统一输出 [n,S,D] 映射到某一档评价器所需的量。

    三档共享同一个 forward 入口, 所以训练循环只有一处 `loss = ...` 不同。
    """

    def __init__(self, evaluator: str, d_shared: int, n_classes: int = N_CLASSES):
        super().__init__()
        if evaluator not in EVALUATORS:
            raise ValueError(f"unknown evaluator {evaluator!r}, expect {EVALUATORS}")
        self.evaluator = evaluator
        self.d_shared = d_shared
        self.n_classes = n_classes
        if d_shared % n_classes != 0:
            raise ValueError(f"d_shared={d_shared} 不能被 n_classes={n_classes} 整除")
        self.blk = d_shared // n_classes
        if evaluator == "gauss_nll":
            # 逐样本标量 logvar: 先对 token 维求均值再 D→1 (只加 D+1 个参数)
            self.logvar_head = nn.Linear(d_shared, 1)
            nn.init.zeros_(self.logvar_head.weight)
            nn.init.constant_(self.logvar_head.bias, -2.0)
        elif evaluator == "ce":
            # 唯一额外参数: 可学习温度 (softplus 保正), 让池化后的 [0,1] 值能当 logits
            self.tau_raw = nn.Parameter(torch.tensor(0.0))

    def extra_params(self) -> int:
        """相对 MSE 档多出来的参数量 (等量纪律审计用)。"""
        return sum(p.numel() for p in self.parameters())

    def forward(self, out: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.evaluator == "mse":
            return {"pred": out}
        if self.evaluator == "gauss_nll":
            return {"pred": out, "logvar": self.logvar_head(out.mean(dim=1)).squeeze(-1)}
        # CE 档必须把输出压到**单纯形**上, 否则输出幅度不受约束,
        # 那个张量的 MSE 与 MSE 档不可比 (冒烟实测: 未压时 fuse MSE 0.74 vs 0.17)。
        # 压法: block 均值 → softmax → 概率摊回到各自 block →
        #       pred 是"预测的 one-hot", 落点与目标同量纲, MSE 可直接互比。
        n, s, d = out.shape
        pooled = out.reshape(n, s, self.n_classes, self.blk).mean(dim=-1)
        tau = F.softplus(self.tau_raw) + 1e-3
        logits = pooled / tau
        probs = F.softmax(logits, dim=-1)                      # [n,S,K]
        pred = probs.unsqueeze(-1).expand(n, s, self.n_classes, self.blk).reshape(n, s, d)
        return {"pred": pred, "logits": logits, "pooled": pooled, "probs": probs, "tau": tau}


def eval_loss(head: EvaluatorHead, out: torch.Tensor, target: torch.Tensor,
              labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """返回 (可反传的 loss, 读数)。

    target: [n, S, D] 块状 one-hot; labels: [n] 类别下标。
    三档 loss **不可直接互比**(量纲不同); 互比要用 baselines() 归一。
    每档都同时报 `mse` —— 那是同一个输出张量的绝对误差, 跨档唯一可比的量。
    """
    q = head.forward(out)
    if head.evaluator == "mse":
        loss = F.mse_loss(q["pred"], target)
        v = float(loss.detach())
        return loss, {"loss": v, "mse": v}

    if head.evaluator == "gauss_nll":
        err = ((q["pred"] - target) ** 2).mean(dim=(1, 2))       # [n]
        lv = q["logvar"]
        loss = (0.5 * (torch.exp(-lv) * err + lv)).mean()
        with torch.no_grad():
            mse = float(F.mse_loss(q["pred"], target))
            cal = _spearman(lv.detach().cpu(), err.detach().cpu())
        return loss, {"loss": float(loss.detach()), "mse": mse,
                      "logvar_mean": float(lv.detach().mean()),
                      "logvar_err_spearman": cal}

    logits = q["logits"]                                          # [n, S, K]
    n, s, k = logits.shape
    lg = logits.reshape(-1, k)
    lb = labels.unsqueeze(1).expand(n, s).reshape(-1)
    loss = F.cross_entropy(lg, lb)
    with torch.no_grad():
        acc = float((lg.argmax(-1) == lb).float().mean())
        mse = float(F.mse_loss(q["pred"], target))
        tau = float(q["tau"].detach())
    return loss, {"loss": float(loss.detach()), "mse": mse, "ce_acc": acc, "tau": tau}


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """无 scipy 依赖的 Spearman (秩相关)。E4 自检: 方差头是否真在用。"""
    if a.numel() < 3:
        return 0.0
    # 常数输入下秩相关无定义; 若不设这个哨兵, argsort 会给出一个"看起来正常"的假相关
    if float(a.std()) < 1e-8 or float(b.std()) < 1e-8:
        return 0.0

    def rank(x: torch.Tensor) -> torch.Tensor:
        order = x.argsort()
        r = torch.empty_like(order, dtype=torch.float32)
        r[order] = torch.arange(len(x), dtype=torch.float32)
        return r

    ra, rb = rank(a), rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = float(ra.norm() * rb.norm())
    return float((ra * rb).sum() / denom) if denom > 1e-9 else 0.0


# ----------------------------------------------------------------------
# 2) 解析基线 (精确值, 不是估计)
# ----------------------------------------------------------------------

def baselines(evaluator: str, n_classes: int = N_CLASSES) -> dict[str, float]:
    """块状 one-hot 目标下, 零参数/无信息预测器在各档评价器上的**精确**得分。

    mse       : 全零 = 1/n;  预测类别均值 = (n-1)/n²
    gauss_nll : 最优常数 logvar 下 = 0.5·ln(MSE) (常数项省略, 与实现同口径)
    ce        : 均匀分布 = ln(n)
    """
    n = n_classes
    zero = 1.0 / n
    mean = (n - 1) / (n * n)
    if evaluator == "mse":
        return {"all_zero": zero, "predict_mean": mean}
    if evaluator == "gauss_nll":
        # 对 logvar 求最优: lv* = ln(err) → 0.5·(1 + ln(err))。漏掉那个 1 会低估 0.5
        return {"all_zero": 0.5 * (1.0 + math.log(zero)),
                "predict_mean": 0.5 * (1.0 + math.log(mean))}
    if evaluator == "ce":
        # 均匀分布 = 每 block 概率 1/n → 摊回 D 维后其 MSE 恰等于 predict_mean 基线
        # (这是内部一致性检查: 两条独立推导必须给出同一个数)
        return {"uniform": math.log(n), "perfect": 0.0,
                "uniform_pred_mse": mean}
    raise ValueError(evaluator)


def decode_accuracy(out: torch.Tensor, labels: torch.Tensor) -> float:
    """从统一输出 [n,S,D] 直接解码类别 (block 均值 argmax) —— 无参数读数。"""
    d = out.shape[-1]
    if d % N_CLASSES != 0:
        raise ValueError(f"d_shared={d} 不能被 n_classes={N_CLASSES} 整除; 块状目标要求等宽块")
    blk = d // N_CLASSES
    pooled = out.reshape(out.shape[0], out.shape[1], N_CLASSES, blk).mean(-1)
    pred = pooled.mean(dim=1).argmax(-1)
    return float((pred == labels).float().mean())


def calibration(logvar: torch.Tensor, err: torch.Tensor) -> float:
    """预测 logvar 与实际平方误差的 Spearman。≈0 ⇒ 方差头退化成常数。"""
    return _spearman(logvar.detach().cpu(), err.detach().cpu())


# ----------------------------------------------------------------------
# 3) 预登记判决 (先写死, 事后不得改)
# ----------------------------------------------------------------------

PREDICTIONS_V54 = {
    "E1": "有意义空间臂的绝对误差(mse 读数)在**三档全部**低于冻结随机臂 (3/3 档, ≥4/5 seeds)",
    "E2": "冻结随机臂在 CE 下准确率 ≈ 1/n_classes ±0.05 (chance); 且 NLL 显著差于常数预测器",
    "E3": "有意义空间下 '融合 vs 最优单专家' 的 CE 准确率差 ≤ 5pp (≥4/5 seeds), 否则须单独解释",
    "E4": "NLL 臂 logvar 与平方误差 Spearman ≥ 0.2 且为正 (否则方差头退化, E2/E3 的 NLL 读数作废)",
    "E5": "投影空间有效秩: 有意义空间 > 冻结随机空间, 且三档同量级",
    # ↓ 冒烟之后**新增** (不是修改 E1): E1 对 CE 档在结构上不成立, 理由见 E6 的说明
    "E6": "有意义空间臂在三档都严格低于**该档自己的零参数基线** "
          "(mse: predict_mean=0.138889 / gauss_nll: predict_mean=-0.486977 / "
          "ce: 均匀预测的等价 mse=0.138889)",
}

# E6 的由来 (2026-10-27, 冒烟阶段, 全网格开跑前):
#   CE 档的冻结随机臂拿到 mse 读数 0.1389 —— 恰等于 predict_mean 基线。
#   原因: CE 头在随机特征上饱和到**均匀分布**, 而均匀分布正是零参数最优常数预测器。
#   于是在 CE 下"随机冻结臂"不是坏模型, 它自动退化成一个聪明的常数预测器。
#   ⇒ E1 的 "learned < frozen" 在 CE 档不是一个合法的不变量 (它比的是两个都不是坏模型的东西)。
#   正确的跨档不变量是"learned < 该档零参数基线"。E6 补这一条; E1 保留不改, 作为记录。
