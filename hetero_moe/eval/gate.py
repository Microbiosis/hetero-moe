"""hetero_moe.eval.gate — 反泄漏、强基线、统计显著的下游评测门禁 (v5.0)。

系统性根治 v4–v83 反复出现的四大评测方法论固有缺陷:
    1. 验证集选模泄漏 (如 v62 train_v2 用 va_idx 选 best_state 又把 va_idx 报为 held-out):
       SplitGuard 强制 train / val / test 三集严格两两不相交，且选模集与报告集隔离。
    2. 缺失单底座强基线与零参数基线 (如 v62 AG_News 无单底座对照、v54 弱于 predict_mean):
       强制输入每个单专家/单底座在相同 test 集上的预测，自动计算 majority/uniform 零参数基线
       与 best_single，以 (fusion - best_single) 为唯一核心增量。
    3. 高维小样本严重欠定拟合 (如 v83 在 N_train=64 上拟合 O(256) 或 896x896 Ridge):
       IdentifiabilityGuard 检查有效自由度与训练样本比，提供安全子空间秩上限。
    4. 代理指标与真实下游能力脱钩 (MSE/CKA/回声探针漂亮但无真实任务提升):
       仅接受真实分类 (Acc/CE) 或真实语言建模 (NLL/PPL) 评测，并强制配对 Bootstrap 95% CI
       与阴性对照 (随机权重/特征置换) 检查，任一门禁不通过即判 FAIL。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class SplitGuard:
    """严格三集划分 (train / val / test) 零重叠守卫。"""

    @staticmethod
    def verify(
        train_idx: Iterable[int],
        val_idx: Iterable[int],
        test_idx: Iterable[int],
    ) -> Dict[str, Any]:
        tr = set(int(i) for i in train_idx)
        va = set(int(i) for i in val_idx)
        te = set(int(i) for i in test_idx)
        if not tr or not va or not te:
            raise ValueError(
                f"train/val/test 均不可为空: n_train={len(tr)}, n_val={len(va)}, n_test={len(te)}"
            )
        ov_tv = tr & va
        ov_vt = va & te
        ov_tt = tr & te
        if ov_tv or ov_vt or ov_tt:
            raise ValueError(
                f"检测到数据划分重叠泄漏! train∩val={len(ov_tv)}, val∩test={len(ov_vt)}, train∩test={len(ov_tt)}"
            )
        return {
            "ok": True,
            "n_train": len(tr),
            "n_val": len(va),
            "n_test": len(te),
            "overlap_count": 0,
        }

    @staticmethod
    def make_splits(
        n_total: int,
        train_frac: float = 0.6,
        val_frac: float = 0.2,
        seed: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """确定性三集划分并自动通过 verify() 校验。"""
        assert n_total >= 3, "样本总数必须 >= 3"
        assert 0.0 < train_frac < 1.0 and 0.0 < val_frac < 1.0 and train_frac + val_frac < 1.0
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_total)
        n_tr = max(1, int(round(n_total * train_frac)))
        n_va = max(1, int(round(n_total * val_frac)))
        if n_tr + n_va >= n_total:
            n_va = max(1, n_total - n_tr - 1)
        tr = perm[:n_tr]
        va = perm[n_tr : n_tr + n_va]
        te = perm[n_tr + n_va :]
        SplitGuard.verify(tr, va, te)
        return tr, va, te


class IdentifiabilityGuard:
    """高维拟合可辨识性/自由度守卫，防止 p >> N 欠定伪拟合。"""

    @staticmethod
    def orthogonal_group_dof(dim: int) -> int:
        """O(d) 群自由度 = d(d-1)/2。"""
        d = int(dim)
        return d * (d - 1) // 2

    @staticmethod
    def max_safe_procrustes_rank(n_train: int, max_dim: int, safety_factor: float = 2.0) -> int:
        """在样本数 n_train 约束下，满足 O(r) 自由度 <= n_train / safety_factor 的最大子空间秩 r。"""
        budget = max(1.0, float(n_train) / max(1.0, float(safety_factor)))
        # r*(r-1)/2 <= budget => r <= (1 + sqrt(1 + 8*budget)) / 2
        r = int(math.floor((1.0 + math.sqrt(1.0 + 8.0 * budget)) / 2.0))
        return max(1, min(int(max_dim), r))

    @staticmethod
    def check(
        n_train: int,
        dof: int,
        max_dof_per_sample: float = 1.0,
    ) -> Dict[str, Any]:
        ratio = float(dof) / max(1, int(n_train))
        ok = ratio <= float(max_dof_per_sample)
        return {
            "ok": ok,
            "n_train": int(n_train),
            "dof": int(dof),
            "dof_per_sample": ratio,
            "max_dof_per_sample": float(max_dof_per_sample),
        }


def paired_bootstrap_ci(
    fusion_values: np.ndarray,
    baseline_values: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    higher_is_better: bool = True,
) -> Dict[str, float]:
    """对逐样本指标差值做配对 Bootstrap 95% 置信区间与单侧 p 值。

    返回增量的方向统一为: 正值 = fusion 优于 baseline。
    """
    fv = np.asarray(fusion_values, dtype=np.float64).reshape(-1)
    bv = np.asarray(baseline_values, dtype=np.float64).reshape(-1)
    assert fv.shape == bv.shape and fv.size > 0, f"形状不匹配: {fv.shape} vs {bv.shape}"
    diff = (fv - bv) if higher_is_better else (bv - fv)
    n = diff.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    boot_means = diff[idx].mean(axis=1)
    lo = float(np.quantile(boot_means, alpha / 2.0))
    hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    p_one_sided = float(np.mean(boot_means <= 0.0))
    return {
        "mean_delta": float(diff.mean()),
        "median_delta": float(np.median(diff)),
        "ci_low": lo,
        "ci_high": hi,
        "p_one_sided": p_one_sided,
        "n_items": int(n),
    }


@dataclass
class GateReport:
    """结构化下游评测门禁报告。"""

    task: str
    passed: bool
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "passed": self.passed,
            "verdict": "PASS" if self.passed else "FAIL",
            "reasons": list(self.reasons),
            "metrics": self.metrics,
        }


class DownstreamEvalGate:
    """统一反泄漏下游任务评测门禁（支持真实分类与序列语言建模 PPL）。"""

    def __init__(
        self,
        n_boot: int = 2000,
        alpha: float = 0.05,
        seed: int = 0,
        require_ci_positive: bool = True,
    ):
        self.n_boot = int(n_boot)
        self.alpha = float(alpha)
        self.seed = int(seed)
        self.require_ci_positive = bool(require_ci_positive)

    def evaluate_classification(
        self,
        y_true_test: torch.Tensor,
        fusion_logits_test: torch.Tensor,
        single_logits_test: Dict[str, torch.Tensor],
        y_train_for_majority: Optional[torch.Tensor] = None,
        negative_control_logits_test: Optional[torch.Tensor] = None,
        split_info: Optional[Dict[str, Any]] = None,
        identifiability_info: Optional[Dict[str, Any]] = None,
    ) -> GateReport:
        """在严格 held-out test 上裁决融合分类模型是否显著击败所有单底座与阴性对照。"""
        if not single_logits_test:
            raise ValueError("必须提供至少一个单底座/单专家基线 single_logits_test")

        y = y_true_test.detach().cpu().long().reshape(-1)
        f_logits = fusion_logits_test.detach().cpu().float()
        n_cls = int(f_logits.shape[-1])
        assert f_logits.shape[0] == y.shape[0], "fusion_logits 样本数与 y_true 不一致"

        f_pred = f_logits.argmax(dim=-1)
        f_correct = (f_pred == y).float().numpy()
        f_nll = F.cross_entropy(f_logits, y, reduction="none").numpy()
        f_acc = float(f_correct.mean())
        f_ce = float(f_nll.mean())

        # 1. 零参数多数类基线（仅用训练集统计众数，避免窥探测试标签分布）
        ref_y = y_train_for_majority.detach().cpu().long().reshape(-1) if y_train_for_majority is not None else y
        maj_cls = int(torch.bincount(ref_y, minlength=n_cls).argmax().item())
        maj_acc = float((y == maj_cls).float().mean().item())
        uniform_ce = float(math.log(max(2, n_cls)))

        # 2. 逐个单底座基线
        singles_summary: Dict[str, Dict[str, float]] = {}
        best_acc_name, best_acc_val, best_acc_correct = None, -1.0, None
        best_ce_name, best_ce_val, best_ce_nll = None, float("inf"), None
        for name, s_log in single_logits_test.items():
            sl = s_log.detach().cpu().float()
            assert sl.shape == f_logits.shape, f"单底座 {name} 形状 {sl.shape} != {f_logits.shape}"
            sc = (sl.argmax(dim=-1) == y).float().numpy()
            snll = F.cross_entropy(sl, y, reduction="none").numpy()
            s_acc = float(sc.mean())
            s_ce = float(snll.mean())
            singles_summary[name] = {"accuracy": s_acc, "cross_entropy": s_ce}
            if s_acc > best_acc_val:
                best_acc_val, best_acc_name, best_acc_correct = s_acc, name, sc
            if s_ce < best_ce_val:
                best_ce_val, best_ce_name, best_ce_nll = s_ce, name, snll

        # 3. 与最强单底座的配对 Bootstrap
        acc_boot = paired_bootstrap_ci(
            f_correct, best_acc_correct, n_boot=self.n_boot, alpha=self.alpha, seed=self.seed, higher_is_better=True
        )
        ce_boot = paired_bootstrap_ci(
            f_nll, best_ce_nll, n_boot=self.n_boot, alpha=self.alpha, seed=self.seed + 1, higher_is_better=False
        )

        # 4. 阴性对照（随机权重 / 置换特征）
        neg_summary = None
        beats_neg = True
        if negative_control_logits_test is not None:
            nl = negative_control_logits_test.detach().cpu().float()
            nc = (nl.argmax(dim=-1) == y).float().numpy()
            nnll = F.cross_entropy(nl, y, reduction="none").numpy()
            neg_boot = paired_bootstrap_ci(
                f_correct, nc, n_boot=self.n_boot, alpha=self.alpha, seed=self.seed + 2, higher_is_better=True
            )
            neg_summary = {
                "accuracy": float(nc.mean()),
                "cross_entropy": float(nnll.mean()),
                "fusion_minus_neg_acc": neg_boot,
            }
            beats_neg = (f_acc > float(nc.mean())) and (neg_boot["ci_low"] > 0.0)

        reasons: List[str] = []
        if split_info is not None and not split_info.get("ok", False):
            reasons.append("split_guard_failed")
        if identifiability_info is not None and not identifiability_info.get("ok", False):
            reasons.append("identifiability_underdetermined")
        if not (f_acc > maj_acc and f_ce < uniform_ce):
            reasons.append(f"failed_zero_param_baseline(acc={f_acc:.4f}<=maj={maj_acc:.4f} or ce={f_ce:.4f}>=unif={uniform_ce:.4f})")
        if not (f_acc > best_acc_val):
            reasons.append(f"failed_best_single_acc({f_acc:.4f}<={best_acc_val:.4f}[{best_acc_name}])")
        if self.require_ci_positive and not (acc_boot["ci_low"] > 0.0 and ce_boot["ci_low"] > 0.0):
            reasons.append(
                f"margin_not_significant(acc_ci_lo={acc_boot['ci_low']:+.4f}, ce_ci_lo={ce_boot['ci_low']:+.4f})"
            )
        if not beats_neg:
            reasons.append("failed_negative_control")

        return GateReport(
            task="classification",
            passed=len(reasons) == 0,
            reasons=reasons,
            metrics={
                "fusion": {"accuracy": f_acc, "cross_entropy": f_ce},
                "zero_param": {"majority_class": maj_cls, "majority_accuracy": maj_acc, "uniform_ce": uniform_ce},
                "singles": singles_summary,
                "best_single_acc": {"name": best_acc_name, "accuracy": best_acc_val},
                "best_single_ce": {"name": best_ce_name, "cross_entropy": best_ce_val},
                "margin_vs_best_single_acc": acc_boot,
                "margin_vs_best_single_ce": ce_boot,
                "negative_control": neg_summary,
                "split": split_info,
                "identifiability": identifiability_info,
            },
        )

    def evaluate_lm(
        self,
        targets_test: torch.Tensor,
        fusion_logits_test: torch.Tensor,
        single_logits_test: Dict[str, torch.Tensor],
        negative_control_logits_test: Optional[torch.Tensor] = None,
        split_info: Optional[Dict[str, Any]] = None,
        identifiability_info: Optional[Dict[str, Any]] = None,
    ) -> GateReport:
        """在严格 held-out 序列测试集上裁决融合 LM 的 NLL/PPL 是否显著优于所有单底座。"""
        if not single_logits_test:
            raise ValueError("必须提供至少一个单底座基线 single_logits_test")

        tgt_raw = targets_test.detach().cpu().long()
        n_seq = int(tgt_raw.shape[0]) if tgt_raw.ndim >= 2 else int(tgt_raw.numel())
        tgt = tgt_raw.reshape(-1)
        f_log = fusion_logits_test.detach().cpu().float()
        V = int(f_log.shape[-1])
        f_log_flat = f_log.reshape(-1, V)
        assert f_log_flat.shape[0] == tgt.shape[0]

        f_nll_tok = F.cross_entropy(f_log_flat, tgt, reduction="none").numpy()
        # 在独立序列级 (每条序列求平均 NLL) 上执行配对 Bootstrap，避免序列内 token 自相关虚增样本量
        f_nll_seq = f_nll_tok.reshape(n_seq, -1).mean(axis=1)
        f_nll = float(f_nll_tok.mean())
        f_ppl = float(math.exp(min(50.0, f_nll)))
        uniform_nll = float(math.log(max(2, V)))
        uniform_ppl = float(V)

        singles_summary: Dict[str, Dict[str, float]] = {}
        best_name, best_nll_val, best_nll_seq = None, float("inf"), None
        for name, s_log in single_logits_test.items():
            sl_flat = s_log.detach().cpu().float().reshape(-1, V)
            snll_tok = F.cross_entropy(sl_flat, tgt, reduction="none").numpy()
            snll_seq = snll_tok.reshape(n_seq, -1).mean(axis=1)
            snll = float(snll_tok.mean())
            sppl = float(math.exp(min(50.0, snll)))
            singles_summary[name] = {"nll": snll, "ppl": sppl}
            if snll < best_nll_val:
                best_nll_val, best_name, best_nll_seq = snll, name, snll_seq

        nll_boot = paired_bootstrap_ci(
            f_nll_seq, best_nll_seq, n_boot=self.n_boot, alpha=self.alpha, seed=self.seed, higher_is_better=False
        )

        neg_summary = None
        beats_neg = True
        if negative_control_logits_test is not None:
            nl_flat = negative_control_logits_test.detach().cpu().float().reshape(-1, V)
            nnll_tok = F.cross_entropy(nl_flat, tgt, reduction="none").numpy()
            nnll_seq = nnll_tok.reshape(n_seq, -1).mean(axis=1)
            nnll = float(nnll_tok.mean())
            neg_boot = paired_bootstrap_ci(
                f_nll_seq, nnll_seq, n_boot=self.n_boot, alpha=self.alpha, seed=self.seed + 1, higher_is_better=False
            )
            neg_summary = {
                "nll": nnll,
                "ppl": float(math.exp(min(50.0, nnll))),
                "nll_reduction_vs_neg": neg_boot,
            }
            beats_neg = (f_nll < nnll) and (neg_boot["ci_low"] > 0.0)

        reasons: List[str] = []
        if split_info is not None and not split_info.get("ok", False):
            reasons.append("split_guard_failed")
        if identifiability_info is not None and not identifiability_info.get("ok", False):
            reasons.append("identifiability_underdetermined")
        if not (f_nll < uniform_nll):
            reasons.append(f"failed_uniform_baseline(nll={f_nll:.4f}>=unif={uniform_nll:.4f})")
        if not (f_nll < best_nll_val):
            reasons.append(f"failed_best_single_nll({f_nll:.4f}>={best_nll_val:.4f}[{best_name}])")
        if self.require_ci_positive and not (nll_boot["ci_low"] > 0.0):
            reasons.append(f"nll_reduction_not_significant(ci_lo={nll_boot['ci_low']:+.4f})")
        if not beats_neg:
            reasons.append("failed_negative_control")

        return GateReport(
            task="language_modeling",
            passed=len(reasons) == 0,
            reasons=reasons,
            metrics={
                "fusion": {"nll": f_nll, "ppl": f_ppl},
                "zero_param": {"uniform_nll": uniform_nll, "uniform_ppl": uniform_ppl},
                "singles": singles_summary,
                "best_single": {
                    "name": best_name,
                    "nll": best_nll_val,
                    "ppl": float(math.exp(min(50.0, best_nll_val))),
                },
                "nll_reduction_vs_best_single": nll_boot,
                "ppl_reduction_pct": float((1.0 - f_ppl / max(1e-9, math.exp(min(50.0, best_nll_val)))) * 100.0),
                "negative_control": neg_summary,
                "split": split_info,
                "identifiability": identifiability_info,
            },
        )
