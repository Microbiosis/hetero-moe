"""V32.3 — super-deterministic teacher test: 教师输出平均对蒸馏质量的影响.

3 种教师推理条件 × 2 种学生目标 × 5 seeds:
    base           — 原始硬 Top-K 路由 + INT4 伪量化 (无任何修改)
    deterministic  — soft_router=True, no_quant=True (v32 已有)
    super-deterministic — deterministic + 教师输出平均
                           对同一输入做 20 次前向 (每次加微量随机噪声),
                           取平均后作为蒸馏目标

2 种学生目标:
    mixed (w_teacher=0.7) — 0.7·MSE(y, y_T) + 0.3·MSE(y, target)
    pure  (w_teacher=1.0) — 纯蒸馏: 仅看教师输出

环境变量:
    V32_SEEDS  — 种子数 (默认 5)
    V32_STEPS  — 训练步数 (默认来自 run_v8_full.STEPS = 100)
    V32_AVG    — 平均次数 (默认 20)

用法:
    python3 examples/run_v32_super_deterministic.py
    V32_SEEDS=5 V32_STEPS=100 python3 examples/run_v32_super_deterministic.py
    JSON 结果: results/v32_super_deterministic.json
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
# 机制部分 (纯 torch, 无 transformers 依赖)
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
    """推理期确定性开关: 对 v10_embedding.fusion 命名空间内的 STE 类打补丁."""
    saved = (_fusion_mod.SparseRouterSTE, _fusion_mod.FakeQuantSTE)
    if soft_route:
        _fusion_mod.SparseRouterSTE = _SoftRouterSTE
    if no_quant:
        _fusion_mod.FakeQuantSTE = _NoQuantSTE
    try:
        yield
    finally:
        _fusion_mod.SparseRouterSTE, _fusion_mod.FakeQuantSTE = saved


# ======================================================================
# 条件 & 目标定义
# ======================================================================

CONDITIONS = {
    #                  soft_route, no_quant
    "base":             (False, False),
    "deterministic":    (True,  True),
    # super-deterministic 在推理时额外做输出平均
    "super-deterministic": None,  # 特殊处理
}

OBJECTIVES = {"mixed": 0.7, "pure": 1.0}


def stack_by_class(modal_seqs, modal_indices, targets, n_cls):
    """把样本按类堆叠成 [6, S, D] 张量组 (与 v16 教师接口一致)."""
    h_by_modal = [[] for _ in range(n_cls)]
    t_by_modal = [[] for _ in range(n_cls)]
    for i, m in enumerate(modal_indices):
        h_by_modal[m].append(modal_seqs[i])
        t_by_modal[m].append(targets[i])
    for cls in range(n_cls):
        h_by_modal[cls] = torch.stack(h_by_modal[cls], dim=0)
        t_by_modal[cls] = torch.stack(t_by_modal[cls], dim=0)
    return h_by_modal, t_by_modal


def eval_teacher_v32(teacher, modal_seqs, modal_indices, targets, n_cls):
    """教师 MSE 评估: 与 v16.eval_teacher 对齐."""
    h_by_modal, t_by_modal = stack_by_class(modal_seqs, modal_indices, targets, n_cls)
    fuse_loss = 0.0
    n = 0
    with torch.no_grad():
        y = teacher(h_by_modal)  # [6, S, D_shared]
        for c in range(n_cls):
            for s in range(6):
                fuse_loss += F.mse_loss(y[s:s + 1], t_by_modal[c][s:s + 1]).item()
                n += 1
    return fuse_loss / n


def get_teacher_outputs_per_sample(teacher, modal_seqs, modal_indices, n_cls, d_shared, seq_len):
    """单次教师前向, 输出按全局样本序排列的 [S, D_shared] 列表."""
    dummy_targets = [torch.zeros(seq_len, d_shared)] * len(modal_seqs)
    h_by_modal, _ = stack_by_class(modal_seqs, modal_indices, dummy_targets, n_cls)
    with torch.no_grad():
        y = teacher(h_by_modal)  # [6, S, D_shared]
    return [y[s].clone() for s in range(y.shape[0])]


def get_super_det_teacher_outputs(teacher, modal_seqs, modal_indices, n_cls, d_shared,
                                   seq_len, n_avg=20):
    """super-deterministic 教师输出: 20 次前向 (每次加随机噪声) 取平均.

    对每个样本独立加噪声, 然后用 teacher_determinism(deterministic) 做前向,
    最后对 n_avg 次输出取平均. 输入噪声标准差固定为 1e-4 (远小于模态信号幅度).
    """
    dummy_targets = [torch.zeros(seq_len, d_shared)] * len(modal_seqs)
    h_by_modal_base, _ = stack_by_class(modal_seqs, modal_indices, dummy_targets, n_cls)

    # 预分配累加器 [6, S, D_shared]
    acc = torch.zeros(6, seq_len, d_shared)

    with teacher_determinism(soft_route=True, no_quant=True):
        for i in range(n_avg):
            # 对每个样本加微量随机噪声, 使前向路径略有不同 (模拟 batch/stochastic noise)
            noisy_h = [
                torch.stack([
                    h_by_modal_base[c][s].clone() + 1e-4 * torch.randn_like(h_by_modal_base[c][s])
                    for s in range(h_by_modal_base[c].shape[0])
                ], dim=0)
                for c in range(n_cls)
            ]
            with torch.no_grad():
                y = teacher(noisy_h)  # [6, S, D_shared]
            acc = acc + y

    avg_y = acc / n_avg
    return [avg_y[s].clone() for s in range(avg_y.shape[0])]


def train_eval_student_v323(student, bert_input_ids, targets, teacher_outputs,
                             w_teacher, vocab_size, steps, seq_len, lr):
    """参数化学生训练/评估: w_teacher·MSE(y, y_T) + (1-w)·MSE(y, t)."""
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        losses = []
        for sample_idx, input_ids in bert_input_ids:
            y = student(input_ids.unsqueeze(0))               # [1, S, D_shared]
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
    vals = [v for v in vals if not (np.isnan(v) or np.isinf(v))]
    if not vals:
        return float("nan"), float("nan")
    return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0


# ======================================================================
# 实验主流程
# ======================================================================

def main():
    # 延迟导入: 依赖 transformers
    from examples.run_v16_bert_distill import (
        make_teacher, make_student, prepare_data, train_teacher,
        eval_teacher, encode_modal_to_input_ids,
    )
    from examples.run_v8_full import (
        load_encoders, make_pools, D_SHARED, N_CLS, STEPS, LR, SEEDS,
    )

    seeds = int(os.environ.get("V32_SEEDS", SEEDS))
    steps = int(os.environ.get("V32_STEPS", STEPS))
    n_avg = int(os.environ.get("V32_AVG", 20))
    seq_len = 16

    print("=" * 80)
    print("V32.3 — super-deterministic teacher test")
    print("=" * 80)
    print(f"  3 教师条件 × 2 学生目标 × {seeds} seeds  (steps={steps}, n_avg={n_avg})")
    print()
    print("  条件说明:")
    print("    base              — 原始: 硬 Top-K 路由 + INT4 伪量化")
    print("    deterministic     — soft_router=True, no_quant=True")
    print(f"    super-deterministic — deterministic + {n_avg}× 前向输出平均")
    print()

    encoders = load_encoders()
    attn_pools = make_pools(encoders)
    bert_tok = encoders[0][0]

    # 结果存储
    teacher_mse = {c: [] for c in CONDITIONS}
    student_mse = {c: {o: [] for o in OBJECTIVES} for c in CONDITIONS}

    raw = {
        "config": {
            "seeds": seeds,
            "steps": steps,
            "n_avg": n_avg,
            "d_shared": D_SHARED,
            "n_cls": N_CLS,
            "seq_len": seq_len,
            "conditions": list(CONDITIONS.keys()),
            "objectives": OBJECTIVES,
        },
        "seeds": [],
        "teacher_mse": {c: [] for c in CONDITIONS},
        "student_mse": {c: {o: [] for o in OBJECTIVES} for c in CONDITIONS},
    }

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

        seed_data = {"seed": seed, "teacher": {}, "student": {}}

        for cond in CONDITIONS:
            if cond == "super-deterministic":
                # super-deterministic: deterministic 补丁 + 20 次前向平均
                with teacher_determinism(soft_route=True, no_quant=True):
                    t_mse = eval_teacher_v32(teacher, modal_seqs, modal_indices, targets, N_CLS)
                    t_out = get_super_det_teacher_outputs(
                        teacher, modal_seqs, modal_indices, N_CLS, D_SHARED, seq_len, n_avg=n_avg
                    )
            else:
                soft_route, no_quant = CONDITIONS[cond]
                with teacher_determinism(soft_route=soft_route, no_quant=no_quant):
                    t_mse = eval_teacher_v32(teacher, modal_seqs, modal_indices, targets, N_CLS)
                    t_out = get_teacher_outputs_per_sample(
                        teacher, modal_seqs, modal_indices, N_CLS, D_SHARED, seq_len
                    )

            teacher_mse[cond].append(t_mse)
            raw["teacher_mse"][cond].append(t_mse)
            seed_data["teacher"][cond] = round(t_mse, 6)
            print(f"  [teacher {cond:>22}] fuse={t_mse:.4f}")

            for obj, w in OBJECTIVES.items():
                student = make_student()
                s_mse = train_eval_student_v323(
                    student, bert_input_ids, targets, t_out,
                    w_teacher=w, vocab_size=bert_tok.vocab_size, steps=steps,
                    seq_len=seq_len, lr=LR,
                )
                student_mse[cond][obj].append(s_mse)
                raw["student_mse"][cond][obj].append(s_mse)
                seed_data["student"][f"{cond}/{obj}"] = round(s_mse, 6)
                print(f"  [student {cond:>22} {obj:>5}] fuse={s_mse:.4f}")

        raw["seeds"].append(seed_data)

    # ---- 汇总表 ----
    print()
    print("=" * 80)
    print(f"{'condition':>24} | {'teacher MSE':>12} | {'mixed(0.7)':>12} | {'pure(1.0)':>12} | {'gap(pure)':>10}")
    print("-" * 80)

    summary = {}
    for cond in CONDITIONS:
        tm, ts = mean_std(teacher_mse[cond])
        mm, ms = mean_std(student_mse[cond]["mixed"])
        pm, ps = mean_std(student_mse[cond]["pure"])
        gap = tm - pm  # 教师 - 纯学生 (正 = 学生优于教师)
        print(f"{cond:>24} | {tm:>8.4f}±{ts:<.4f} | {mm:>8.4f}±{ms:<.4f} | "
              f"{pm:>8.4f}±{ps:<.4f} | {gap:>+8.4f}")
        summary[cond] = {
            "teacher_mse_mean": round(tm, 6),
            "teacher_mse_std":  round(ts, 6),
            "mixed_mean":       round(mm, 6),
            "mixed_std":        round(ms, 6),
            "pure_mean":        round(pm, 6),
            "pure_std":         round(ps, 6),
            "gap_pure_mean":    round(gap, 6),
        }

    print("=" * 80)

    # ---- 对比分析 ----
    print()
    print("对比分析:")
    t_base, _ = mean_std(teacher_mse["base"])
    t_det, _  = mean_std(teacher_mse["deterministic"])
    t_sd, _   = mean_std(teacher_mse["super-deterministic"])
    print(f"  教师 MSE:  base={t_base:.4f}  deterministic={t_det:.4f}  "
          f"super-deterministic={t_sd:.4f}")

    for obj in OBJECTIVES:
        s_base_m, _ = mean_std(student_mse["base"][obj])
        s_det_m, _  = mean_std(student_mse["deterministic"][obj])
        s_sd_m, _   = mean_std(student_mse["super-deterministic"][obj])
        print(f"  student [{obj}] MSE:  base={s_base_m:.4f}  det={s_det_m:.4f}  "
              f"super-det={s_sd_m:.4f}")

    # gap(pure) = teacher MSE - pure student MSE
    for cond in CONDITIONS:
        tm, _ = mean_std(teacher_mse[cond])
        pm, _ = mean_std(student_mse[cond]["pure"])
        gap = tm - pm
        print(f"  gap(pure, {cond}) = teacher - pure_student = {tm:.4f} - {pm:.4f} = {gap:+.4f}")

    # ---- 落盘 ----
    out_dir = Path(_REPO_ROOT) / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "v32_super_deterministic.json"
    payload = {
        "config": raw["config"],
        "summary": summary,
        "per_seed": raw,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nJSON 已写入: {out_path}")


if __name__ == "__main__":
    main()
