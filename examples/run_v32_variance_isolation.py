"""V32.0 — 方差隔离: 学生-教师差距的归因分解 (评审路线图 E1, 对应 DA CRITICAL 1).

背景: 五维评审的魔鬼代言人主张 "学生超出教师 47.5%/50.5%" 可能主要来自
教师输出管线的离散化伪影 (硬 Top-K 路由 + INT4 伪量化) 与学生混合目标中
0.3 直接目标项的泄漏, 而非"内化概念方向"的证据. 本实验把差距分解干净:

    4 种教师推理条件 (同一批按论文口径训练好的教师, 仅切换推理期前向):
        base           — 原版: 硬 Top-K 路由 + INT4 伪量化
        no-int4        — 关 INT4 (恒等), 保留硬路由
        soft-route     — 关硬路由 (连续 softmax 权重), 保留 INT4
        deterministic  — 两者全关 (最干净的教师函数)

    2 种学生目标 (同一 TinyBERT+LoRA 学生):
        mixed (w_teacher=0.7) — 论文口径: 0.7·MSE(y, y_T) + 0.3·MSE(y, target)
        pure  (w_teacher=1.0) — 纯蒸馏: 学生只看教师输出, 无直接目标泄漏

    验收判据 (打印, 由数据说话):
        A. 教师自身误差中的离散化伪影 = MSE_teacher(base) − MSE_teacher(deterministic)
        B. 纯蒸馏学生在 deterministic 教师下是否仍优于教师:
           是 → 存在真实传递信号; 否 → 标题裕度 = 伪影 + 泄漏 (DA 主张成立)
        C. mixed 与 pure 的差 = 0.3 直接目标项的贡献 (泄漏量化)

    5 seeds × (1 教师训练 + 4 条件 × 2 学生) = 5 教师训练 + 40 学生训练.
    所有指标 mean ± std (响应评审 R2-1).

用法:
    python3 examples/run_v32_variance_isolation.py
    V32_SEEDS=2 V32_STEPS=20 python3 examples/run_v32_variance_isolation.py   # 冒烟
    JSON 结果落盘: results/v32_variance_isolation.json

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


CONDITIONS = {
    #              soft_route, no_quant
    "base":          (False, False),
    "no-int4":       (False, True),
    "soft-route":    (True,  False),
    "deterministic": (True,  True),
}

OBJECTIVES = {"mixed": 0.7, "pure": 1.0}


def stack_by_class(modal_seqs, modal_indices, targets, n_cls):
    """把样本按类堆叠成 [6, S, D] 张量组 (与 run_v16 教师接口一致)."""
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


def train_eval_student_v32(student, bert_input_ids, targets, teacher_outputs,
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
    print("=" * 78)
    print("V32.0 — 方差隔离: 学生-教师差距的归因分解 (E1 / DA CRITICAL 1)")
    print("=" * 78)
    print(f"{len(CONDITIONS)} 教师推理条件 × 2 学生目标 × {seeds} seeds "
          f"(steps={steps}, teacher 训练每 seed 一次, 论文口径)")
    print()

    encoders = load_encoders()
    attn_pools = make_pools(encoders)
    bert_tok = encoders[0][0]
    seq_len = 16

    teacher_mse = {c: [] for c in CONDITIONS}
    student_mse = {c: {o: [] for o in OBJECTIVES} for c in CONDITIONS}
    raw = {"config": {"seeds": seeds, "steps": steps, "d_shared": D_SHARED,
                      "conditions": CONDITIONS, "objectives": OBJECTIVES},
           "seeds": [], "teacher_mse": {c: [] for c in CONDITIONS},
           "student": {c: {o: [] for o in OBJECTIVES} for c in CONDITIONS}}

    for seed in range(seeds):
        torch.manual_seed(seed)
        print(f"--- seed {seed} ---")
        modal_seqs, modal_indices, targets, _ = prepare_data(seed, encoders)

        # 教师按论文口径训练一次 (硬路由 + INT4 + STE), 之后仅切换推理条件
        teacher = make_teacher(attn_pools, encoders)
        train_teacher(teacher, modal_seqs, modal_indices, targets)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        bert_input_ids = encode_modal_to_input_ids(modal_seqs, modal_indices, bert_tok)

        for cond, (soft_route, no_quant) in CONDITIONS.items():
            with teacher_determinism(soft_route, no_quant):
                t_mse = eval_teacher(teacher, modal_seqs, modal_indices, targets)
                t_out = teacher_outputs_per_sample(
                    teacher, modal_seqs, modal_indices, N_CLS, D_SHARED, seq_len)
            teacher_mse[cond].append(t_mse)
            raw["teacher_mse"][cond].append(t_mse)
            print(f"  [teacher {cond:>13}] fuse={t_mse:.4f}")

            for obj, w in OBJECTIVES.items():
                student = make_student()
                s_mse = train_eval_student_v32(
                    student, bert_input_ids, targets, t_out,
                    w_teacher=w, vocab_size=bert_tok.vocab_size, steps=steps,
                    seq_len=seq_len, lr=LR,
                )
                student_mse[cond][obj].append(s_mse)
                raw["student"][cond][obj].append(s_mse)
                print(f"  [student {cond:>13} {obj:>5}] fuse={s_mse:.4f}")
        raw["seeds"].append(seed)

    # ---- 汇总 ----
    out_dir = Path(_REPO_ROOT) / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "v32_variance_isolation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print(f"{'condition':>15} | {'teacher':>16} | {'stu mixed(0.7)':>16} | {'stu pure(1.0)':>16}")
    print("-" * 78)
    for cond in CONDITIONS:
        tm, ts = mean_std(teacher_mse[cond])
        mm, ms = mean_std(student_mse[cond]["mixed"])
        pm, ps = mean_std(student_mse[cond]["pure"])
        print(f"{cond:>15} | {tm:>7.4f} ± {ts:.4f} | {mm:>7.4f} ± {ms:.4f} | {pm:>7.4f} ± {ps:.4f}")
    print("=" * 78)

    # ---- 归因判据 ----
    t_base, _ = mean_std(teacher_mse["base"])
    t_det, _ = mean_std(teacher_mse["deterministic"])
    s_base_mixed, _ = mean_std(student_mse["base"]["mixed"])
    s_det_mixed, _ = mean_std(student_mse["deterministic"]["mixed"])
    s_det_pure, _ = mean_std(student_mse["deterministic"]["pure"])

    print("\n归因分解 (mean):")
    artifact = t_base - t_det
    print(f"  A. 教师自身误差中的离散化伪影   = MSE(base) − MSE(deterministic) "
          f"= {t_base:.4f} − {t_det:.4f} = {artifact:+.4f} "
          f"({(artifact / max(t_base, 1e-9)) * 100:+.1f}% of teacher base)")
    gap_base = t_base - s_base_mixed
    gap_det_mixed = t_det - s_det_mixed
    gap_det_pure = t_det - s_det_pure
    print(f"  B. gap(mixed, base)   = {gap_base:+.4f}   ← 论文标题裕度的复现")
    print(f"     gap(mixed, deterministic) = {gap_det_mixed:+.4f}")
    print(f"     gap(pure,   deterministic) = {gap_det_pure:+.4f}   ← 纯蒸馏、最干净教师")
    leak = gap_det_mixed - gap_det_pure
    print(f"  C. 0.3 直接目标项泄漏 = gap(mixed,det) − gap(pure,det) = {leak:+.4f}")

    print("\n判据解读 (对照 microfusion_peer_review.md R0-1 / DA CRITICAL 1):")
    if gap_det_pure <= 0:
        print("  ✗ 纯蒸馏学生在最干净教师下不再优于教师 → 标题裕度 = 离散化伪影 + 泄漏;")
        print("    论文应把学生-教师差距重述为上界伪影, 以 E1 的分解表替代标题数字。")
    elif gap_det_pure < 0.5 * gap_base:
        print("  △ 存在真实传递信号但被伪影+泄漏显著放大; 论文应以 gap_det_pure 为主张,")
        print("    并报告 A/B/C 分解。")
    else:
        print("  ✓ 纯蒸馏学生在最干净教师下仍保留大部分裕度 → 真实传递信号成立;")
        print("    论文应以该条件下的数字为主张并附分解表。")
    print(f"\nJSON 已写入 {out_path}")


if __name__ == "__main__":
    main()
