"""v40.0 — 分派式多教师的算子集.

背景
----
v17 的 ``MultiTeacherDistiller.forward`` (``v17_multi_teacher/distill.py:51``)
把**同一张量** ``y`` 别名进全部教师槽位, 并对多个目标同时回归:

    total = Σ_k  w_k · MSE(y, t_k)

``MANIFEST.json`` 的 ``root_cause_identified_later`` 把失败归因于
"同时多目标回归的实现形式" 而非多教师本身不可行, 并注明
"从未测过分派式多教师". 本模块提供**分派族算子**, 用于检验该归因.

与 MOPD 的关系
--------------
外部对照 MOPD (arXiv:2606.30406) 用 **per-prompt 单教师分派** 取代加权求和,
在 Qwen3-30B-A3B 上把归一化分数做到 0.9373 (Mix-RL 0.8818 / Cascade RL 0.7752).

**但前置条件不同**: MOPD 的教师是**领域专家**(math / IF / SWE), 输入域有划分,
分派信号唯一确定; v17 的教师是**对齐器阶梯三档** (attn / mixture / mom),
同数据同目标、只是架构不同, **能力完全重叠、没有域划分**.

因此本模块提供的分派信号不是域标签, 而是三种从教师自身导出的规则,
并**显式标注各自的泄漏状态** (对齐 v32 的 β_h 记账法):

===================  ====================  ==========================
函数                  分派信号              泄漏
===================  ====================  ==========================
``oracle_assign``    目标 MSE 最小者       **是** — 仅作上界, 禁止引用为结论
``consensus_assign`` 留一共识最近者        否
``cyclic_assign``    确定性轮转            否(但平凡)
===================  ====================  ==========================

设计约束
--------
L1 自包含: 本模块**只依赖 torch**, 不 import 任何 ``research.*`` 子包
(``research/distillation`` 在 ``check_self_containment.py`` 的
``ALLOWED_INTRA_LINE`` 中没有条目, 跨线 import 在 --strict 下会报错).
教师本身由 ``examples/run_v40_dispatch.py`` 从 v17 的 harness 构造.
"""
from __future__ import annotations

from typing import Dict, List, Mapping

import torch

__all__ = [
    "weighted_loss",
    "dispatch_loss",
    "teacher_mean",
    "oracle_assign",
    "consensus_assign",
    "cyclic_assign",
    "pairwise_disagreement",
    "teacher_target_mse",
]


# ── 损失算子 ──────────────────────────────────────────────────────────────

def weighted_loss(
    student: torch.Tensor,
    teachers: Mapping[str, torch.Tensor],
    weights: Mapping[str, float] | None = None,
) -> Dict[str, torch.Tensor]:
    """v17 基线的精确复刻: 全部教师对**同一**学生输出加权求和.

    与 ``v17_multi_teacher.distill.MultiTeacherDistiller.forward`` 逐行等价,
    差别只在批处理 (这里 ``[N, ...]``, 原实现逐样本 ``[1, ...]``).

    返回 ``{teacher_name: loss, "total": 加权和}``。
    """
    names = list(teachers)
    if not names:
        raise ValueError("teachers 不能为空")
    if weights is None:
        weights = {n: 1.0 / len(names) for n in names}

    per_teacher: Dict[str, torch.Tensor] = {}
    total: torch.Tensor | None = None
    for name in names:
        t = teachers[name].detach()
        if t.shape != student.shape:
            raise ValueError(
                f"教师 {name!r} 形状 {tuple(t.shape)} 与学生 {tuple(student.shape)} 不符"
            )
        loss = torch.nn.functional.mse_loss(student, t)
        per_teacher[name] = loss
        term = float(weights[name]) * loss
        total = term if total is None else total + term
    per_teacher["total"] = total
    return per_teacher


def dispatch_loss(
    student: torch.Tensor,
    teachers: Mapping[str, torch.Tensor],
    assign: List[str] | torch.Tensor,
    teacher_names: List[str] | None = None,
) -> torch.Tensor:
    """分派式蒸馏: **每个样本只有一个教师说话**.

    与 :func:`weighted_loss` 的区别是本质的 —— 不是多个目标同时回归,
    而是每个样本各自有一个目标, 梯度不来自冲突方向的叠加.

    参数
    ----
    student : ``[N, ...]``
    teachers : ``name -> [N, ...]`` (各教师对全部 N 个样本都有预测)
    assign : 长度 N 的教师名序列, ``assign[n]`` 是第 n 个样本的分派教师
    teacher_names : ``assign`` 用整数索引时的名称表; 传 ``None`` 表示用字符串

    返回标量 loss。
    """
    if student.dim() == 0:
        raise ValueError("student 至少要有一维")
    n = student.shape[0]
    if len(assign) != n:
        raise ValueError(f"assign 长度 {len(assign)} 与样本数 {n} 不符")
    if teacher_names is not None:
        assign = [teacher_names[int(i)] for i in assign]

    total: torch.Tensor | None = None
    for idx, name in enumerate(assign):
        if name not in teachers:
            raise KeyError(f"分派到未知教师 {name!r}; 可用: {sorted(teachers)}")
        t = teachers[name].detach()
        if t.shape != student.shape:
            raise ValueError(
                f"教师 {name!r} 形状 {tuple(t.shape)} 与学生 {tuple(student.shape)} 不符"
            )
        term = torch.nn.functional.mse_loss(student[idx], t[idx])
        total = term if total is None else total + term
    assert total is not None, "assign 为空时不应到达"
    return total / n


# ── 分派信号 ──────────────────────────────────────────────────────────────

def oracle_assign(
    teachers: Mapping[str, torch.Tensor],
    target: torch.Tensor,
) -> List[str]:
    """按**目标 MSE 最小**给每个样本选教师.

    ⚠️ **这个函数读取目标, 是泄漏的**。它是上界 (upper bound), 用来回答
    "**如果**分派信号完美, 收益上限有多大". 任何结论都不得基于它,
    正如 v32 把 β_h=1 的表观增益归因为泄漏项. 引用本函数结果时必须同时
    声明 ``leakage_audit.status`` 与 β 取值.
    """
    names = list(teachers)
    n = target.shape[0]
    out: List[str] = []
    for i in range(n):
        best, best_err = names[0], None
        for name in names:
            err = torch.nn.functional.mse_loss(teachers[name][i], target[i]).item()
            if best_err is None or err < best_err:
                best, best_err = name, err
        out.append(best)
    return out


def consensus_assign(teachers: Mapping[str, torch.Tensor]) -> List[str]:
    """按**留一共识**选教师: 取与"其余教师均值"最近的那个.

    不读目标, 因此**无泄漏**. 动机: 三个对齐器同数据同目标, 若它们互为
    噪声副本, 共识分派会退化成随机选择; 若某个教师系统性地更靠近共识,
    共识分派就等价于一个无监督的质量代理.

    要求至少 2 个教师 (留一需要"其余"非空)。
    """
    names = list(teachers)
    if len(names) < 2:
        raise ValueError(f"共识分派至少需要 2 个教师, 收到 {len(names)}")
    n = teachers[names[0]].shape[0]
    out: List[str] = []
    for i in range(n):
        best, best_err = names[0], None
        for name in names:
            others = [teachers[m][i] for m in names if m != name]
            consensus = torch.stack(others, dim=0).mean(dim=0)
            err = torch.nn.functional.mse_loss(teachers[name][i], consensus).item()
            if best_err is None or err < best_err:
                best, best_err = name, err
        out.append(best)
    return out


def cyclic_assign(teachers: Mapping[str, torch.Tensor]) -> List[str]:
    """确定性轮转分派 (平凡对照).

    存在的意义是提供**零信息基线**: 它与任何教师质量无关, 只看样本序号.
    若共识分派打不过它, 说明共识信号本身没有信息量.
    """
    names = list(teachers)
    n = teachers[names[0]].shape[0]
    return [names[i % len(names)] for i in range(n)]


# ── 诊断量 ────────────────────────────────────────────────────────────────

def teacher_mean(teachers: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """全体教师预测的逐元素均值 (共识目标)."""
    names = list(teachers)
    if not names:
        raise ValueError("teachers 不能为空")
    return torch.stack([teachers[n] for n in names], dim=0).mean(dim=0)


def pairwise_disagreement(teachers: Mapping[str, torch.Tensor]) -> float:
    """教师两两之间的平均平方距离.

    这个量回答 v17 遗留的开放问题: **教师之间到底冲突到什么程度**?
    - 若与各教师自身对目标的误差同量级 → 教师互为噪声副本,
      多教师在信息层面本就无增益, v17 的负结果不是实现问题
    - 若远小于 → 教师高度一致, 加权和的"冲突"是虚的,
      负结果更可能来自信号稀释
    """
    names = list(teachers)
    if len(names) < 2:
        return 0.0
    total, count = 0.0, 0
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            total += torch.nn.functional.mse_loss(
                teachers[names[a]], teachers[names[b]]
            ).item()
            count += 1
    return total / count


def teacher_target_mse(
    teachers: Mapping[str, torch.Tensor],
    target: torch.Tensor,
) -> Dict[str, float]:
    """每个教师 (以及共识目标) 对真实目标的 MSE.

    这是**不需要训练**的关键诊断: 若 ``mean`` 不优于任何单教师,
    则多教师携带的互补信息在目标层面就不存在.
    """
    out = {
        name: torch.nn.functional.mse_loss(t.detach(), target).item()
        for name, t in teachers.items()
    }
    out["__consensus_mean__"] = torch.nn.functional.mse_loss(
        teacher_mean({k: v.detach() for k, v in teachers.items()}), target
    ).item()
    return out
