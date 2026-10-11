"""v42.0 — 多模态蒸馏泄漏审计 (β_h 连续过渡扫描, runner).

审计对象: ``v21_multimodal`` 声称"多模态学生蒸馏后 (0.1578) 比 v13 教师 (0.3190) 优 50.5%"。

泄漏通道: v21 的学生损失是::

    loss = 0.7 · MSE(y_student, teacher) + 0.3 · MSE(y_student, target)

这与 ``v16_bert_student`` 的结构**完全相同**, 即 v32 已判定为泄漏源的那个系数
(``w_teacher = 0.7 ⇒ β_h = 0.3``)。v32 在 v16 上证明: 把 β_h 压到 0, 学生回到教师水平,
表观裕度归零。本代把该扫描移植到 v21。

协议复用: 教师构造、教师训练、学生结构、多模态输入构造、评估口径
**全部直接 import** ``examples/run_v21_multimodal_distill.py``, 只把学生损失参数化。

用法::

    python3 examples/run_v42_leakage_audit.py                # 11 个 β_h × 5 seeds
    V42_SEEDS=1 python3 examples/run_v42_leakage_audit.py    # 冒烟
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from research.multimodal.v21_multimodal import MultiModalStudent
from research.multimodal.v42_leakage_audit import (
    blend_loss,
    linearity_report,
    linearity_verdict,
)

from examples.run_v21_multimodal_distill import (
    get_multimodal_inputs,
    make_teacher,
    prepare_teacher_data,
)
from examples.run_v8_full import D_SHARED, N_CLS, SEEDS, STEPS, LR, load_encoders, make_pools
from research.routing_evolution.v8_central import V8Trainer as TeacherTrainer

BETAS = [round(i * 0.1, 1) for i in range(11)]     # v32.2 同口径: [0.0, 1.0] 步长 0.1


def run_seed(seed, encoders, attn_pools):
    """一个 seed: 教师训一次, 学生对每个 β_h 各训一次。"""
    torch.manual_seed(seed)

    # ── 教师 (与 v21 完全同构, 只训一次并冻结) ──────────────────────────
    modal_seqs, modal_indices, targets = prepare_teacher_data(seed, encoders)
    teacher = make_teacher(attn_pools, encoders)
    trainer = TeacherTrainer(teacher, lr=LR, phase=2)
    h_by_modal = [[] for _ in range(N_CLS)]
    t_by_modal = [[] for _ in range(N_CLS)]
    for i, m in enumerate(modal_indices):
        h_by_modal[m].append(modal_seqs[i])
        t_by_modal[m].append(targets[i])
    for c in range(N_CLS):
        h_by_modal[c] = torch.stack(h_by_modal[c], dim=0)
        t_by_modal[c] = torch.stack(t_by_modal[c], dim=0)
    for _ in range(STEPS):
        trainer.optimizer.zero_grad(set_to_none=True)
        y = teacher(h_by_modal)
        loss = sum(F.mse_loss(y, t_by_modal[c]) for c in range(N_CLS)) / N_CLS
        loss.backward()
        trainer.optimizer.step()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        y_teacher = teacher(h_by_modal)
        teacher_outs = [y_teacher[s] for s in range(6)]
        # 教师基线 (v21 的 in-sample 口径)
        fuse_t = 0.0
        for c in range(N_CLS):
            for s in range(6):
                fuse_t += F.mse_loss(y_teacher[s:s + 1], t_by_modal[c][s:s + 1]).item()
        teacher_mse = fuse_t / 18

    # ── 学生 × β_h ──────────────────────────────────────────────────────
    out = {}
    for beta in BETAS:
        torch.manual_seed(seed)                       # 学生初始化与 β 无关
        student = MultiModalStudent(d_out=D_SHARED, lora_rank=8)
        opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=LR)

        for step in range(STEPS):
            opt.zero_grad(set_to_none=True)
            inputs = get_multimodal_inputs(seed + step, batch_size=6)
            y_student, _ = student(inputs["text_inputs"], inputs["image_pixels"])
            S_text = y_student.size(1)
            tgt = t_by_modal[0][:, :S_text, :]
            teacher_targets = torch.cat(
                [teacher_outs[s].unsqueeze(0) for s in range(6)], dim=0)[:, :S_text, :]
            loss = blend_loss(y_student, teacher_targets, tgt, beta_h=beta)
            loss.backward()
            opt.step()

        # 评估: 与 v21 完全一致 (3 次 held-out 多模态输入, 对 BERT 模态 target)
        fuse = 0.0
        with torch.no_grad():
            for b in range(3):
                inputs = get_multimodal_inputs(seed + 1000 + b, batch_size=6)
                y_s, _ = student(inputs["text_inputs"], inputs["image_pixels"])
                S_text = y_s.size(1)
                tgt = t_by_modal[0][:, :S_text, :]
                sl = sum(F.mse_loss(y_s[s:s + 1], tgt[s:s + 1]).item() for s in range(6)) / 6
                fuse += sl / 3
        out[beta] = {"student": fuse, "teacher": teacher_mse}
    return out


def main():
    seeds = int(os.environ.get("V42_SEEDS", SEEDS))
    print("=" * 86)
    print("V42.0 — 多模态蒸馏泄漏审计: β_h 连续过渡扫描")
    print("=" * 86)
    print(f"审计对象: v21 的 +50.5% 主张 (学生 0.1578 vs 教师 0.3190)")
    print(f"泄漏通道: v21 学生损失 = 0.7·MSE(y,teacher) + 0.3·MSE(y,target) ⇒ β_h = 0.3")
    print(f"β_h ∈ {BETAS[0]}..{BETAS[-1]} step=0.1 ({len(BETAS)} 点) × {seeds} seeds")
    print()

    encoders = load_encoders()
    attn_pools = make_pools(encoders)

    raw = {b: [] for b in BETAS}
    teacher_all = []
    for seed in range(seeds):
        try:
            r = run_seed(seed, encoders, attn_pools)
            for b in BETAS:
                raw[b].append(r[b]["student"])
            teacher_all.append(r[BETAS[0]]["teacher"])
            print(f"  [seed {seed}] 教师 {r[BETAS[0]]['teacher']:.4f} | " +
                  "  ".join(f"β={b}:{r[b]['student']:.4f}" for b in (0.0, 0.3, 1.0)))
        except Exception as e:
            print(f"  [seed {seed}] FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            for b in BETAS:
                raw[b].append(float("nan"))

    # ── 汇总 + 裁决 ──────────────────────────────────────────────────────
    means = [statistics.mean([v for v in raw[b] if not np.isnan(v)]) for b in BETAS]
    meds = [statistics.median([v for v in raw[b] if not np.isnan(v)]) for b in BETAS]
    sds = [statistics.stdev([v for v in raw[b] if not np.isnan(v)])
           if len([v for v in raw[b] if not np.isnan(v)]) > 1 else 0.0 for b in BETAS]
    noise = statistics.mean(sds)

    print("\n" + "=" * 86)
    print(f"{'β_h':>5} | {'蒸馏权重':>9} | {'学生 均值':>10} {'中位数':>9} {'sd':>8} | {'vs 教师':>9}")
    print("-" * 86)
    t_mean = statistics.mean([t for t in teacher_all if not np.isnan(t)])
    for b, mu, me, sd in zip(BETAS, means, meds, sds):
        vs = (t_mean - mu) / max(t_mean, 1e-9) * 100
        print(f"{b:>5.1f} | {1-b:>9.1f} | {mu:>10.4f} {me:>9.4f} {sd:>8.4f} | {vs:>+8.1f}%")
    print("-" * 86)
    print(f"{'教师 (v21 口径)':>5} | {'-':>9} | {t_mean:>10.4f}")

    rep = linearity_report(BETAS, means, noise=noise)
    verdict = linearity_verdict(rep)
    print(f"\n=== 线性诊断 (均值曲线, 噪声 = 各 β_h 种子间 sd 的均值 = {noise:.4f}) ===")
    print(f"  slope={rep['slope']:+.4f}  intercept={rep['intercept']:.4f}  R²={rep['r2']:.4f}")
    print(f"  最大残差={rep['max_abs_residual']:.4f} ({rep['residual_over_noise']:.2f}× 噪声)")
    print(f"  最大二阶差分={rep['max_second_diff']:.4f} ({rep['second_over_noise']:.2f}× 噪声)")
    print(f"  单调递减: {rep['monotone']}")
    print(f"  裁决: **{verdict}**  (linear = 纯插值无机制 / threshold = 有折点 / inconclusive = 读不出)")

    at0, at1 = means[0], means[-1]
    print(f"\n=== 裕度裁决 ===")
    print(f"  v21 原始口径 (β_h=0.3): 学生 {statistics.mean([v for v in raw[0.3]]):.4f}")
    print(f"  β_h=0.0 (纯蒸馏):        学生 {at0:.4f}   教师 {t_mean:.4f}   "
          f"vs 教师 {(t_mean-at0)/t_mean*100:+.1f}%")
    print(f"  β_h=1.0 (纯目标):        学生 {at1:.4f}")
    print(f"  关闭泄漏通道后的裕度变化: "
          f"{(statistics.mean([v for v in raw[0.3]]) - at0):+.4f} MSE")
    print("=" * 86)

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "research", "multimodal", "v42_leakage_audit", "results", "raw.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"experiment": "v42_leakage_audit", "seeds": seeds, "betas": BETAS,
                   "teacher_mse": teacher_all,
                   "raw": {str(b): raw[b] for b in BETAS},
                   "means": means, "medians": meds, "sds": sds,
                   "noise": noise, "report": rep, "verdict": verdict,
                   "teacher_mean": t_mean}, f, indent=2, ensure_ascii=False)
    print(f"写入: {out}")


if __name__ == "__main__":
    main()
