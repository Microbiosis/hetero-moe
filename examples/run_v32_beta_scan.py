"""V32.2 — β_h continuous transition scan (连续过渡扫描).

扫描 β_h ∈ [0.0, 1.0] 步长 0.05, 对每个 β_h:
    w_teacher = 1 − β_h,  w_target = β_h

用确定性教师 (soft_route=True, no_quant=True) 训练学生, 5 seeds.
打印表格: β_h | teacher MSE | student MSE | gap (teacher-student) | vs pure

用法:
    python3 examples/run_v32_beta_scan.py
    V32_SEEDS=3 V32_STEPS=20 python3 examples/run_v32_beta_scan.py   # 冒烟
    JSON 结果: results/v32_beta_scan.json

本模块顶部的机制部分 (STE 补丁/上下文/学生目标) 不依赖 transformers,
可独立单测; 重导入 (编码器/学生/数据) 全部延迟到 main().
"""
import contextlib
import json
import os
import statistics
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import torch.nn.functional as F

import research.aligner.v10_embedding.fusion as _fusion_mod


# ======================================================================
# 机制部分 (纯 torch, 无 transformers 依赖) — 可独立单测
# ======================================================================

class _SoftRouterSTE:
    """连续化路由: 前向直接返回 softmax 权重 (无 Top-K 硬选择)."""

    @staticmethod
    def apply(alpha, k):
        return alpha


class _NoQuantSTE:
    """关闭 INT4: 恒等映射."""

    @staticmethod
    def apply(x, bits=4, group_size=128):
        return x


@contextlib.contextmanager
def teacher_determinism(soft_route: bool, no_quant: bool):
    """推理期确定性开关: 对 v10_embedding.fusion 命名空间内的 STE 类打补丁.

    (SparseRouterSTE / FakeQuantSTE 仅被该模块引用; 对齐器与 attention 池不涉及.)
    """
    saved = (_fusion_mod.SparseRouterSTE, _fusion_mod.FakeQuantSTE)
    if soft_route:
        _fusion_mod.SparseRouterSTE = _SoftRouterSTE
    if no_quant:
        _fusion_mod.FakeQuantSTE = _NoQuantSTE
    try:
        yield
    finally:
        _fusion_mod.SparseRouterSTE, _fusion_mod.FakeQuantSTE = saved


def stack_by_class(modal_seqs, modal_indices, targets, n_cls):
    """把样本按类堆叠成 [n_cls, S, D] 张量组 (与 run_v16 教师接口一致)."""
    h_by_modal = [[] for _ in range(n_cls)]
    t_by_modal = [[] for _ in range(n_cls)]
    for i, m in enumerate(modal_indices):
        h_by_modal[m].append(modal_seqs[i])
        t_by_modal[m].append(targets[i])
    for cls in range(n_cls):
        h_by_modal[cls] = torch.stack(h_by_modal[cls], dim=0)
        t_by_modal[cls] = torch.stack(t_by_modal[cls], dim=0)
    return h_by_modal, t_by_modal


def teacher_outputs_per_sample(teacher, modal_seqs, modal_indices, n_cls, d_shared, seq_len):
    """单次教师前向, 输出按全局样本序排列的 [S, D_shared] 列表 (与 v16 对齐)."""
    dummy_targets = [torch.zeros(seq_len, d_shared)] * len(modal_seqs)
    h_by_modal, _ = stack_by_class(modal_seqs, modal_indices, dummy_targets, n_cls)
    with torch.no_grad():
        y_teacher = teacher(h_by_modal)  # [B, S, d_shared]
    return [y_teacher[s].clone() for _ in range(n_cls) for s in range(y_teacher.shape[0])]


def train_eval_student(student, bert_input_ids, targets, teacher_outputs,
                       w_teacher, vocab_size, steps, seq_len, lr):
    """v16 train_eval_bert 的参数化版本: w_teacher·MSE(y, y_T) + (1-w)·MSE(y, t).

    w_teacher=0.7 即论文口径 mixed; 1.0 即纯蒸馏 pure (无直接目标泄漏).
    评估与 v16 相同: 对每个样本重采样全新 input_ids, 对 target 计 MSE.
    """
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        losses = []
        for sample_idx, input_ids in bert_input_ids:
            y = student(input_ids.unsqueeze(0))            # [1, S, D_shared]
            t = targets[sample_idx].unsqueeze(0)
            t_teacher = teacher_outputs[sample_idx].unsqueeze(0).detach()
            loss = (w_teacher * F.mse_loss(y, t_teacher)
                    + (1.0 - w_teacher) * F.mse_loss(y, t))
            losses.append(loss)
        total = sum(losses) / len(losses)
        total.backward()
        opt.step()
    fuse_loss = 0.0
    n_eval = 0
    with torch.no_grad():
        for sample_idx, _ in bert_input_ids:
            eval_ids = torch.randint(0, vocab_size, (seq_len,), dtype=torch.long)
            y = student(eval_ids.unsqueeze(0))
            t = targets[sample_idx].unsqueeze(0)
            fuse_loss += F.mse_loss(y, t).item()
            n_eval += 1
    return fuse_loss / n_eval


def mean_std(vals):
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return float("nan"), float("nan")
    return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0


# ======================================================================
# 实验主流程 (重导入延迟至此: 编码器/学生/数据依赖 transformers)
# ======================================================================

def main():
    from examples.run_v16_bert_distill import (
        make_teacher, make_student, prepare_data, train_teacher,
        eval_teacher, encode_modal_to_input_ids,
    )
    from examples.run_v8_full import (
        load_encoders, make_pools, D_SHARED, N_CLS, STEPS, LR, SEEDS,
    )

    seeds = int(os.environ.get("V32_SEEDS", SEEDS))
    steps = int(os.environ.get("V32_STEPS", STEPS))

    # β_h 扫描: 0.0, 0.1, 0.2, ..., 1.0  → 11 个值 (step=0.1)
    beta_values = [round(i * 0.1, 1) for i in range(11)]
    assert beta_values[0] == 0.0 and beta_values[-1] == 1.0, \
        "β_h range must span [0.0, 1.0]"
    assert len(beta_values) == 11, \
        f"Expected 11 β_h values, got {len(beta_values)}"

    print("=" * 78)
    print("V32.2 — β_h continuous transition scan (连续过渡扫描)")
    print("=" * 78)
    print(f"β_h ∈ [0.0, 1.0] step=0.1  ({len(beta_values)} values)")
    print(f"{seeds} seeds  |  steps={steps}  |  deterministic teacher (soft_route + no_quant)")
    print()

    encoders = load_encoders()
    attn_pools = make_pools(encoders)
    bert_tok = encoders[0][0]
    seq_len = 16

    # 用列表收集每个 β_h 的指标 (跨 seeds)
    teacher_mse_by_beta = {bh: [] for bh in beta_values}
    student_mse_by_beta = {bh: [] for bh in beta_values}
    raw_rows = []

    for seed in range(seeds):
        torch.manual_seed(seed)
        print(f"--- seed {seed} ---")
        modal_seqs, modal_indices, targets, _ = prepare_data(seed, encoders)

        # 教师按论文口径训练一次 (硬路由 + INT4 + STE)
        teacher = make_teacher(attn_pools, encoders)
        train_teacher(teacher, modal_seqs, modal_indices, targets)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        bert_input_ids = encode_modal_to_input_ids(modal_seqs, modal_indices, bert_tok)

        # 确定性教师前向: 对所有 β_h 共享同一个教师输出
        with teacher_determinism(soft_route=True, no_quant=True):
            t_mse = eval_teacher(teacher, modal_seqs, modal_indices, targets)
            t_out = teacher_outputs_per_sample(
                teacher, modal_seqs, modal_indices, N_CLS, D_SHARED, seq_len)

        print(f"  [teacher deterministic] fuse={t_mse:.4f}")

        for beta_h in beta_values:
            w_teacher = round(1.0 - beta_h, 2)
            w_target  = beta_h

            student = make_student()
            s_mse = train_eval_student(
                student, bert_input_ids, targets, t_out,
                w_teacher=w_teacher, vocab_size=bert_tok.vocab_size,
                steps=steps, seq_len=seq_len, lr=LR,
            )

            teacher_mse_by_beta[beta_h].append(t_mse)
            student_mse_by_beta[beta_h].append(s_mse)
            raw_rows.append({
                "seed": seed,
                "beta_h": beta_h,
                "w_teacher": w_teacher,
                "w_target": w_target,
                "teacher_mse": t_mse,
                "student_mse": s_mse,
            })
            print(f"  [seed {seed} β_h={beta_h:.2f}]  w_T={w_teacher:.2f} w_t={w_target:.2f}  "
                  f"student={s_mse:.4f}")

    # ---- 汇总统计 ----
    summary = []
    for beta_h in beta_values:
        tm, ts = mean_std(teacher_mse_by_beta[beta_h])
        sm, ss = mean_std(student_mse_by_beta[beta_h])
        gap = tm - sm  # teacher - student (正值 = student better)
        summary.append({
            "beta_h": beta_h,
            "w_teacher": round(1.0 - beta_h, 2),
            "w_target": beta_h,
            "teacher_mse_mean": tm,
            "teacher_mse_std": ts,
            "student_mse_mean": sm,
            "student_mse_std": ss,
            "gap_mean": gap,
        })

    # vs pure: 以 β_h=1.0 (纯目标, w_teacher=0.0) 为基准
    pure_student_mean = summary[-1]["student_mse_mean"]
    for row in summary:
        row["vs_pure"] = row["student_mse_mean"] - pure_student_mean

    # ---- 打印表格 ----
    print()
    print("=" * 78)
    print(f"{'β_h':>5} | {'w_T':>5} | {'teacher MSE':>14} | {'student MSE':>14} | "
          f"{'gap(T-S)':>10} | {'vs pure':>10}")
    print("-" * 78)
    for row in summary:
        print(
            f"{row['beta_h']:>5.2f} | {row['w_teacher']:>5.2f} | "
            f"{row['teacher_mse_mean']:>7.4f}±{row['teacher_mse_std']:.4f} | "
            f"{row['student_mse_mean']:>7.4f}±{row['student_mse_std']:.4f} | "
            f"{row['gap_mean']:>+10.4f} | {row['vs_pure']:>+10.4f}"
        )
    print("=" * 78)
    print(f"\nPure baseline (β_h=1.0, w_teacher=0.0): student MSE = {pure_student_mean:.4f}")
    print()

    # ---- 过渡分析 ----
    beta_0 = summary[0]
    beta_1 = summary[-1]
    gap_0 = beta_0["gap_mean"]
    gap_1 = beta_1["gap_mean"]
    print("过渡分析:")
    print(f"  β_h=0.0 (w_teacher=1.0, pure distill)  gap = {gap_0:+.4f}")
    print(f"  β_h=1.0 (w_teacher=0.0, pure target)   gap = {gap_1:+.4f}")
    if gap_0 > gap_1:
        print("  → 纯蒸馏 (β_h=0) 的 gap 大于纯目标 (β_h=1): 教师信号有效")
    else:
        print("  → 纯目标 (β_h=1) 的 gap 更大或相等: 教师信号弱/无")

    best_row = max(summary, key=lambda r: r["gap_mean"])
    print(f"  → 最大 gap 出现在 β_h={best_row['beta_h']:.2f}  (gap={best_row['gap_mean']:+.4f})")
    print()

    # ---- 落盘 ----
    out_dir = Path(_REPO_ROOT) / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "v32_beta_scan.json"
    payload = {
        "config": {
            "seeds": seeds,
            "steps": steps,
            "d_shared": D_SHARED,
            "n_cls": N_CLS,
            "beta_values": beta_values,
            "teacher_mode": "deterministic (soft_route=True, no_quant=True)",
        },
        "summary": summary,
        "raw": raw_rows,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"JSON 已写入 {out_path}")
    print(f"  {seeds} seeds × {len(beta_values)} β_h values = {seeds * len(beta_values)} runs")


if __name__ == "__main__":
    main()
