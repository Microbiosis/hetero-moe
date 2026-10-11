"""v39.0 — 融合头两种模式 (γ2 核心: 加性 vs 交互)。

背景 (为什么需要交互头):
    v9.0-γ (本项目 v38.0-α) 用**加性框架**:
        logits = head(h_base + P(c))        head = nn.Linear(D_BIG, N_CLS)
    展开即 logits = W·h_base + W·P(c) + b —— 文本侧贡献与表格侧贡献**线性求和**。
    XOR 标签 (a XOR b) 无法表示为两个一元函数之和 (经典 XOR 不可加性分解),
    故所有臂都被压在同一退化天花板 (~0.77, 而正确解需 ≥0.9)。
    γ 报告的 "C/D/E 三臂打平 0.755-0.767" 因此是**架构人工产物**, 非科学结论。

    γ2 (本项目 v39.0) 引入**交互框架**:
        logits = head(concat[h_base, P(c)])  head = MLP(D_BIG*2 → 128 → N_CLS)
    concat 后经 MLP 可表达两模态交互 → 能解 XOR。

可验证的判据 (tests/test_v39 已锁定):
    加性头在 XOR 上退化 (acc ≈ 0.5~0.77), 交互头可达 acc = 1.0。
"""
import torch
import torch.nn as nn


def make_head(mode, d_big, n_cls, hidden=128):
    """按 mode 构造融合头。

    Args:
        mode: "add"    — 加性: logits = Linear(h_base + c_proj)      (v38 旧框架)
              "concat" — 交互: logits = MLP(concat[h_base, c_proj])   (v39 γ2 框架)
        d_big: 容器维度 (h_base 与 c_proj 的维度)
        n_cls: 类别数
    """
    if mode == "add":
        return nn.Linear(d_big, n_cls)
    if mode == "concat":
        return nn.Sequential(
            nn.Linear(d_big * 2, hidden), nn.GELU(), nn.Linear(hidden, n_cls))
    raise ValueError(f"未知融合模式: {mode!r} (应为 'add' 或 'concat')")


def fuse(h_base, c_proj, mode):
    """按 mode 融合容器表征与投影后的入口表征。

    add:    h_base + c_proj                (线性叠加, 无法表达交互)
    concat: cat([h_base, c_proj], dim=1)   (保留两侧独立维度, 供 MLP 学交互)
    """
    if mode == "add":
        return h_base + c_proj
    if mode == "concat":
        return torch.cat([h_base, c_proj], dim=1)
    raise ValueError(f"未知融合模式: {mode!r}")


def logits_of(head, h_base, c_proj, mode):
    """融合 + 头前向 (训练与评估共用, 避免两处不一致)。"""
    return head(fuse(h_base, c_proj, mode))
