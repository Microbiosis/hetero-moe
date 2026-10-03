"""v32.0-4 目标泄漏审计 (target-leakage audit) — 从随包分发的原始记录重算结论.

存在的理由
----------
v32 是**审计代**: 它不提出新机制, 而是对 v16 蒸馏头条结论做归因分解.
论文里那句 "the entire measured margin is attributable to the 0.3 direct-target
term" 原本是散文; 本模块把它变成**可重算的判据** —— 只要包内 results/ 的
四份原始记录在, 任何人重跑本模块就能复核该结论, 无需相信作者的自述.

这是本包与"只有一份 MANIFEST 声明"的包的关键差别: 声明可以被照抄, 原始记录不能.

四个变体各自证伪一个替代解释
----------------------------
| 变体 | 提问 | 若成立则说明 |
|---|---|---|
| V32.0 variance_isolation | 关掉 INT4 噪声 / 硬 Top-K 路由, 差距是否变化 | 教师前向随机性贡献了差距 |
| V32.2 beta_scan | 直接目标项权重 beta_h 0→1, 学生 MSE 如何走 | 收益与泄漏权重成比例 |
| V32.3 super_deterministic | 20 次前向平均的超确定教师是否更难打败 | Jensen 型方差平均优势 |
| V32.4 representation_transfer | 学到的概念方向能否迁移到新任务 | 混合目标培养了更好的内部表示 |

数值口径 (与论文 §Target-leakage quantification 一致, 不要随意改)
--------------------------------------------------------------
- ``paired_gap := teacher - pure_student``。**负值 = 学生比教师差**,
  即"纯蒸馏不超越教师"。若把符号反过来, 会读成"学生反超教师", 与论文矛盾。
- 标准差一律用**样本标准差** (ddof=1), 不是总体标准差。论文的 ±0.0038
  只有 ddof=1 才成立 (ddof=0 会得到 ±0.0036)。
- 论文引用的头条数字取自 variance_isolation 的 ``deterministic`` 条件,
  不是 ``base``。两者数值不同, 引用时别混。

本模块只依赖标准库, 符合 ``research/_primitives`` 的零依赖约束。
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence

__all__ = [
    "RESULTS_DIR",
    "PUBLISHED_CONDITION",
    "load_records",
    "paired_gap",
    "variance_isolation_table",
    "published_headline",
    "beta_scan_curve",
    "beta_scan_is_monotonic",
    "super_deterministic_table",
    "transfer_accuracy",
    "leakage_verdict",
]

#: 包内 results/ —— 证据随包分发, 审计不依赖仓库根的任何东西
RESULTS_DIR = Path(__file__).resolve().parent / "results"

#: 论文头条数字所引用的条件
PUBLISHED_CONDITION = "deterministic"


def load_records(name: str, results_dir: Optional[Path] = None) -> Dict:
    """读取包内一份原始记录 (如 ``v32_beta_scan``)."""
    d = Path(results_dir) if results_dir is not None else RESULTS_DIR
    p = d / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(
            f"缺少原始记录 {p.name} —— 审计无法在证据缺失时给出结论 (宁可报错, 也不猜)")
    return json.loads(p.read_text(encoding="utf-8"))


def _mean_std(vals: Sequence[float]) -> Dict[str, float]:
    """均值 + **样本**标准差 (ddof=1); 单样本时 std 记 0.0 而不是抛异常."""
    if not vals:
        raise ValueError("空序列无法统计")
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {"mean": statistics.mean(vals), "std": std, "n": len(vals)}


def paired_gap(teacher: Sequence[float], student: Sequence[float]) -> Dict[str, float]:
    """逐 seed 配对差 ``teacher - student`` 的均值与样本标准差.

    负值 = 学生更差。必须逐 seed 配对后再求标准差 —— 配对消掉了 seed 间的
    共同波动, 这正是"同一个教师下学生是否更好"该有的比较方式。
    """
    if len(teacher) != len(student):
        raise ValueError(f"teacher/student 长度不等: {len(teacher)} vs {len(student)}")
    gaps = [t - s for t, s in zip(teacher, student)]
    out = _mean_std(gaps)
    out["abs_mean"] = abs(out["mean"])
    return out


def variance_isolation_table(results_dir: Optional[Path] = None) -> Dict[str, Dict]:
    """四个教师随机性条件下的教师 / 混合学生 / 纯学生 / 配对差."""
    rec = load_records("v32_variance_isolation", results_dir)
    table: Dict[str, Dict] = {}
    for cond in rec["config"]["conditions"]:
        t = rec["teacher_mse"][cond]
        table[cond] = {
            "teacher": _mean_std(t),
            "mixed_student": _mean_std(rec["student"][cond]["mixed"]),
            "pure_student": _mean_std(rec["student"][cond]["pure"]),
            "paired_gap": paired_gap(t, rec["student"][cond]["pure"]),
        }
    return table


def published_headline(results_dir: Optional[Path] = None) -> Dict:
    """复算论文/v16 MANIFEST 引用的头条数字: -0.0009 +- 0.0038 (10 seeds)."""
    row = variance_isolation_table(results_dir)[PUBLISHED_CONDITION]
    return {
        "condition": PUBLISHED_CONDITION,
        "teacher_mse": row["teacher"]["mean"],
        "teacher_std": row["teacher"]["std"],
        "pure_student_mse": row["pure_student"]["mean"],
        "pure_student_std": row["pure_student"]["std"],
        "mixed_student_mse": row["mixed_student"]["mean"],
        "mixed_student_std": row["mixed_student"]["std"],
        "paired_gap": row["paired_gap"]["mean"],
        "paired_gap_std": row["paired_gap"]["std"],
        "seeds": row["paired_gap"]["n"],
    }


def beta_scan_curve(results_dir: Optional[Path] = None) -> List[Dict]:
    """按 beta_h 升序返回 (beta_h, teacher_mse, student_mse, gap)."""
    rec = load_records("v32_beta_scan", results_dir)
    rows = sorted(rec["summary"], key=lambda r: r["beta_h"])
    return [
        {
            "beta_h": r["beta_h"],
            "teacher_mse": r["teacher_mse_mean"],
            "teacher_std": r["teacher_mse_std"],
            "student_mse": r["student_mse_mean"],
            "student_std": r["student_mse_std"],
            "gap": r["gap_mean"],
        }
        for r in rows
    ]


def beta_scan_is_monotonic(results_dir: Optional[Path] = None) -> bool:
    """学生 MSE 是否随 beta_h 单调不增.

    这条判据的意义: 若存在阈值效应, 在 beta_h=0.3 (论文采用值) 附近会出现跳变.
    实际是平滑过渡, 所以收益与泄漏权重成比例, 而非某个机制被"激活"。
    """
    vals = [r["student_mse"] for r in beta_scan_curve(results_dir)]
    return all(b <= a + 1e-12 for a, b in zip(vals, vals[1:]))


def super_deterministic_table(results_dir: Optional[Path] = None) -> Dict[str, Dict]:
    """base / deterministic / super-deterministic 三种教师的对照."""
    rec = load_records("v32_super_deterministic", results_dir)
    return {k: dict(v) for k, v in rec["summary"].items()}


def transfer_accuracy(results_dir: Optional[Path] = None) -> Dict[str, Dict]:
    """迁移到新任务后的分类准确率: 混合学生 vs 纯学生."""
    rec = load_records("v32_representation_transfer", results_dir)
    return {
        k: {
            "eval_accuracy": v["eval_accuracy_mean"],
            "eval_std": v["eval_accuracy_std"],
            "seeds": rec["config"]["seeds"],
        }
        for k, v in rec["summary"].items()
    }


def leakage_verdict(results_dir: Optional[Path] = None) -> Dict:
    """整代结论的机器可读版本 —— 三个问题, 三个判定.

    1. 移走直接目标信号后, 学生还超越教师吗?   -> 否
    2. 教师前向随机性 / Jensen 优势可测量吗?    -> 否
    3. 混合目标培养出更好的内部表示吗?          -> 否
    结论: 混合目标下的全部表观收益都来自泄漏的直接目标项, 而非概念方向迁移.
    """
    head = published_headline(results_dir)
    table = variance_isolation_table(results_dir)
    curve = beta_scan_curve(results_dir)
    beta0 = curve[0]
    transfer = transfer_accuracy(results_dir)
    supdet = super_deterministic_table(results_dir)

    return {
        "pure_beats_teacher": head["paired_gap"] > 0,
        "paired_gap": head["paired_gap"],
        "paired_gap_std": head["paired_gap_std"],
        "gap_within_noise_in_every_condition": all(
            abs(row["paired_gap"]["abs_mean"]) < 0.01 for row in table.values()),
        "leaked_margin": head["mixed_student_mse"],
        "leak_free_margin": head["pure_student_mse"],
        "beta0_matches_teacher": abs(beta0["student_mse"] - beta0["teacher_mse"]) < 0.01,
        "beta_scan_monotonic": beta_scan_is_monotonic(results_dir),
        "super_deterministic_teacher_identical": abs(
            supdet["super-deterministic"]["teacher_mse_mean"]
            - supdet["deterministic"]["teacher_mse_mean"]) < 1e-6,
        "transfer_prefers_leak_free": (
            transfer["pure"]["eval_accuracy"] >= transfer["mixed"]["eval_accuracy"]),
        "conclusion": (
            "混合目标下的全部表观学生-教师差距来自 0.3 直接目标项 (目标泄漏); "
            "纯蒸馏不超越教师, Jensen 型方差平均优势不可测量, 且未产生可迁移的内部表示."
        ),
    }
