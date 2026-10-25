"""v48.0 — 投影机制的可判定变体 (冻结 / 可训练 / 全可训练)。

设计原则: **不修改 v33 的实现代码**, 只在 v48 侧改 `requires_grad`。
理由: 每代只加一个机制, 既有代的代码不应被后人改动 (否则 v33 的
"+26.1%" 就失去了可追溯性)。本模块只提供"参数自由度"的开关。

⚠️ 一个必须记录的实现事实 (来自 v33 fusion.py 的 `_compute_output`):
    out = x_shared + Σ_m α_m · FFN_m(γ_m ⊙ x_det + β_m),  其中 x_det = x_shared.detach()
  → **专家通路被 detach 了**, 梯度只经由**残差项**回传到 P_m。
  也就是说: 把 P_m 解冻后, 它的学习信号比"直觉上"弱得多 (只有残差这一条
  通路在驱动它)。这一点必须写进 SPEC, 否则一个 null result 会被误读成
  "可训练投影没用", 而真实原因可能是"梯度通路太窄"。
"""
from __future__ import annotations

import torch

__all__ = ["ARMS", "configure_arm", "count_trainable"]

# 臂定义: 只改"哪些参数允许被梯度更新", 不改任何结构、参数量或前向语义。
ARMS = {
    "A_frozen": {
        "train_proj": False,     # P_m 冻结 (v33 现状: 随机正交)
        "train_experts": False,  # 专家 FFN 冻结 (v33 现状: 随机)
        "label": "A 冻结 (v33 现状复现)",
    },
    "B_train_proj": {
        "train_proj": True,      # 只有 P_m 可训练 —— 「空间自己会学」
        "train_experts": False,
        "label": "B 投影可训练 (空间可变)",
    },
    "C_train_all": {
        "train_proj": True,
        "train_experts": True,   # 连专家 FFN 一起训
        "label": "C 投影+专家全可训练",
    },
}


def count_trainable(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def configure_arm(layer, arm: str):
    """把 v33 的 CrossModalFusionLayer 调到指定臂, 返回 (参数列表, 信息)。

    参数量在三个臂之间**完全相同** —— 解冻不增加任何参数, 只增加自由度。
    因此本对照不受 v41 审计指出的"容量混杂"影响: 差异只能来自"是否被训练"。
    """
    if arm not in ARMS:
        raise ValueError(f"未知臂: {arm}; 可选 {sorted(ARMS)}")
    cfg = ARMS[arm]

    # 先全部冻结 (幂等)
    for p in layer.P_m:
        p.requires_grad_(False)
    for pl in (layer.w_gates, layer.w_ups, layer.w_downs):
        for p in pl:
            p.requires_grad_(False)

    # v33 的可训练集合: 路由 + 每专家仿射
    params = [layer.W_router] + list(layer.gammas) + list(layer.betas) + list(layer.alphas)
    info = {
        "arm": arm,
        "label": cfg["label"],
        "n_trainable_params": sum(p.numel() for p in params),
        "proj_params_unfrozen": 0,
        "expert_params_unfrozen": 0,
        "total_params_unchanged": sum(p.numel() for p in layer.parameters()),
    }
    if cfg["train_proj"]:
        for p in layer.P_m:
            p.requires_grad_(True)
            params.append(p)
            info["proj_params_unfrozen"] += p.numel()
    if cfg["train_experts"]:
        for pl in (layer.w_gates, layer.w_ups, layer.w_downs):
            for p in pl:
                p.requires_grad_(True)
                params.append(p)
                info["expert_params_unfrozen"] += p.numel()
    info["n_trainable_params"] = sum(p.numel() for p in params)
    return params, info
