"""V17.0 — 多教师蒸馏到真实 BERT 学生 (5 seeds).

5 变体 × 5 seeds = 25 run:
    teacher-single       — v13 mixture 单教师蒸馏 (v16 baseline)
    teacher-multi-2      — v10 attn + v13 mixture 双教师等权蒸馏
    teacher-multi-3      — v10 attn + v13 mixture + v15 mom 三教师等权蒸馏
    teacher-multi-3-weighted — 三教师加权 (v13 权重 0.5, v10/v15 各 0.25)
    teacher-best-single  — 已知最优单教师 (v15 mom)

核心判断: 多教师是否优于单教师 (v16 的 0.135)?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import statistics

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from research._primitives.attention import AttnPool
from research.routing_evolution.v8_central import V8Trainer as TeacherTrainer
from research.aligner.v10_embedding import AlignedFusionLayer, CrossArchAttnAligner
from research.aligner.v13_mixture import MixtureAligner
from research.aligner.v15_nested import MoMAligner
from research.distillation.v16_bert_student import BERTStudent
from research.distillation.v17_multi_teacher import MultiTeacherDistiller
from examples.run_v8_full import (
    load_encoders, make_pools, encode_text, encode_code, encode_image,
    D_SHARED, N_CLS, B, S, STEPS, LR, SEEDS,
)


def make_teacher(aligner_kind, attn_pools, encoders):
    (_, _, D_bert, _), (_, _, D_llama, _), (_, _, D_vit, _) = encoders
    modal_dims = [D_bert, D_llama, D_vit]
    if aligner_kind == "attn":
        aligner = CrossArchAttnAligner(modal_dims=modal_dims, d_shared=D_SHARED, num_heads=4)
    elif aligner_kind == "mixture":
        aligner = MixtureAligner(modal_dims=modal_dims, d_shared=D_SHARED,
                                  n_shared_experts=4, num_heads=4)
    elif aligner_kind == "mom":
        aligner = MoMAligner(modal_dims=modal_dims, d_shared=D_SHARED,
                              n_inner=2, n_outer=2, num_heads=4)
    return AlignedFusionLayer(
        d_shared=D_SHARED, d_ff=4 * D_SHARED,
        attn_pools=attn_pools, modal_dims=modal_dims,
        aligner=aligner, c1_alpha=0.1,
    )


def prepare_data(seed, encoders):
    (bert_tok, bert, _, _), (llama_tok, llama, _, _), (vit_proc, vit, _, _) = encoders
    texts = ["cat dog bird", "animal pet wild", "feline canine fowl"]
    h_text = encode_text(texts, bert_tok, bert)
    codes = ["def hello():", "return True", "pass None"]
    h_code = encode_code(codes, llama_tok, llama)
    img = np.random.randint(0, 256, (224, 224, 3))
    h_img = encode_image(img, vit_proc, vit)
    modal_seqs, modal_indices = [], []
    for cls, h_block in enumerate([h_text, h_code, h_img]):
        n = h_block.shape[0]
        for i in range(6):
            src = h_block[i % n]
            g = torch.Generator().manual_seed(seed * 10 + cls * 6 + i)
            modal_seqs.append(src + 0.1 * torch.randn(src.shape, generator=g))
            modal_indices.append(cls)
    targets = []
    for cls in range(N_CLS):
        ds = cls * (D_SHARED // N_CLS); de = (cls + 1) * (D_SHARED // N_CLS)
        t = torch.zeros(S, D_SHARED); t[:, ds:de] = 1.0
        for _ in range(6):
            targets.append(t)
    return modal_seqs, modal_indices, targets


def train_teacher(teacher, modal_seqs, modal_indices, targets):
    trainer = TeacherTrainer(teacher, lr=LR, phase=2)
    h_by_modal = [[] for _ in range(N_CLS)]
    t_by_modal = [[] for _ in range(N_CLS)]
    for i, m in enumerate(modal_indices):
        h_by_modal[m].append(modal_seqs[i])
        t_by_modal[m].append(targets[i])
    for cls in range(N_CLS):
        h_by_modal[cls] = torch.stack(h_by_modal[cls], dim=0)
        t_by_modal[cls] = torch.stack(t_by_modal[cls], dim=0)
    for _ in range(STEPS):
        trainer.optimizer.zero_grad(set_to_none=True)
        y = teacher(h_by_modal)
        loss = sum(F.mse_loss(y, t_by_modal[c]) for c in range(N_CLS)) / N_CLS
        loss.backward()
        trainer.optimizer.step()


def run_seed(mode, seed, encoders, attn_pools, bert_tok):
    torch.manual_seed(seed)
    (bert_tok_, bert, _, _), (llama_tok, llama, _, _), (vit_proc, vit, _, _) = encoders
    print(f"  [seed {seed}] mode={mode} ...")
    modal_seqs, modal_indices, targets = prepare_data(seed, encoders)

    # 训练所有教师
    teachers = {}
    if mode == "teacher-single":
        teacher_kinds = ["mixture"]
    elif mode == "teacher-best-single":
        teacher_kinds = ["mom"]
    elif mode == "teacher-multi-2":
        teacher_kinds = ["attn", "mixture"]
    elif mode in ("teacher-multi-3", "teacher-multi-3-weighted"):
        teacher_kinds = ["attn", "mixture", "mom"]
    else:
        raise ValueError(mode)

    for kind in teacher_kinds:
        t = make_teacher(kind, attn_pools, encoders)
        train_teacher(t, modal_seqs, modal_indices, targets)
        t.eval()
        for p in t.parameters():
            p.requires_grad = False
        teachers[kind] = t

    # 收集教师输出 (按 BERT 模态样本 6 个)
    h_by_modal = [[] for _ in range(N_CLS)]
    for i, m in enumerate(modal_indices):
        h_by_modal[m].append(modal_seqs[i])
    for cls in range(N_CLS):
        h_by_modal[cls] = torch.stack(h_by_modal[cls], dim=0)
    # 教师对 BERT 模态 (cls=0) 6 个样本的输出
    teacher_outs_per_sample = {kind: [] for kind in teacher_kinds}
    with torch.no_grad():
        for kind, t in teachers.items():
            y = t(h_by_modal)  # [6, S, D_shared]
            for s in range(6):
                teacher_outs_per_sample[kind].append(y[s])  # [S, D_shared]

    # 权重
    if mode == "teacher-multi-3-weighted":
        weights = {"attn": 0.25, "mixture": 0.5, "mom": 0.25}
    else:
        weights = None   # 默认等权
    distiller = MultiTeacherDistiller(teacher_kinds, weights)

    # 学生: 真实 BERT
    student = BERTStudent(model_name="huawei-noah/TinyBERT_General_4L_312D",
                            d_out=D_SHARED, lora_rank=8)
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=LR)
    vocab_size = bert_tok.vocab_size

    # 训练
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        losses = []
        for s in range(6):  # 6 个 BERT 模态样本
            input_ids = torch.randint(0, vocab_size, (S,), dtype=torch.long)
            y = student(input_ids.unsqueeze(0))   # [1, S, D_shared]
            # 多教师输出
            student_outs = {kind: y for kind in teacher_kinds}  # 同一学生输出对所有教师
            teacher_outs = {kind: teacher_outs_per_sample[kind][s].unsqueeze(0)
                            for kind in teacher_kinds}
            d_losses = distiller(student_outs, teacher_outs)
            losses.append(d_losses["total"])
        total = sum(losses) / len(losses)
        total.backward()
        opt.step()

    # 评估: 新 sample (避免过拟合)
    fuse_loss = 0.0
    n_eval = 0
    with torch.no_grad():
        for s in range(6):
            input_ids = torch.randint(0, vocab_size, (S,), dtype=torch.long)
            y = student(input_ids.unsqueeze(0))
            t = targets[s].unsqueeze(0)
            fuse_loss += F.mse_loss(y, t).item()
            n_eval += 1
        fuse_loss /= n_eval
    return fuse_loss


MODES = ["teacher-single", "teacher-best-single", "teacher-multi-2",
         "teacher-multi-3", "teacher-multi-3-weighted"]


def main():
    print("=" * 78)
    print("V17.0 — 多教师蒸馏到真实 BERT 学生")
    print("=" * 78)
    print("5 变体 × 5 seeds = 25 run")
    print("核心判断: 多教师是否优于 v16 的单教师 (0.135)?")
    print()
    encoders = load_encoders()
    attn_pools = make_pools(encoders)
    bert_tok = encoders[0][0]
    results = {m: [] for m in MODES}

    print(f"\n每个 mode × {SEEDS} seeds")
    for mode in MODES:
        for seed in range(SEEDS):
            try:
                fuse = run_seed(mode, seed, encoders, attn_pools, bert_tok)
                results[mode].append(fuse)
                print(f"  [{mode:>26} seed {seed}] fuse={fuse:.4f}")
            except Exception as e:
                print(f"  [{mode:>26} seed {seed}] FAILED: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                results[mode].append(float("nan"))

    # 汇总
    print("\n" + "=" * 78)
    print(f"{'mode':>28} | {'fuse MSE':>9} | {'vs single':>11}")
    print("-" * 78)
    single_mean = statistics.mean([m for m in results["teacher-single"] if not np.isnan(m)])

    for m in MODES:
        vals = [x for x in results[m] if not np.isnan(x)]
        if not vals:
            print(f"{m:>28} | {'N/A':>9} | {'N/A':>11}")
            continue
        m_mean = statistics.mean(vals)
        gain = (single_mean - m_mean) / max(single_mean, 1e-9) * 100
        print(f"{m:>28} | {m_mean:>9.4f} | {gain:>+10.1f}%")
    print("=" * 78)
    print()
    print("结论:")
    f_single = statistics.mean([m for m in results["teacher-single"] if not np.isnan(m)])
    f_best = statistics.mean([m for m in results["teacher-best-single"] if not np.isnan(m)])
    f_2 = statistics.mean([m for m in results["teacher-multi-2"] if not np.isnan(m)])
    f_3 = statistics.mean([m for m in results["teacher-multi-3"] if not np.isnan(m)])
    f_3w = statistics.mean([m for m in results["teacher-multi-3-weighted"] if not np.isnan(m)])
    print(f"  single (v13 mixture, v16):          {f_single:.4f}")
    print(f"  best single (v15 mom):              {f_best:.4f}")
    print(f"  multi-2 (v10+v13):                   {f_2:.4f}")
    print(f"  multi-3 (v10+v13+v15):               {f_3:.4f}")
    print(f"  multi-3 weighted (v13 权重 0.5):     {f_3w:.4f}")
    print()
    if min(f_2, f_3, f_3w) < f_single:
        best_multi = min(f_2, f_3, f_3w)
        improvement = (f_single - best_multi) / f_single * 100
        print(f"✅ 多教师蒸馏有效: 最佳 ({best_multi:.4f}) 比单教师 ({f_single:.4f}) 改善 {improvement:.1f}%")
    else:
        print(f"⚠ 多教师蒸馏未带来增益 (单教师 {f_single:.4f} 已足够)")


if __name__ == "__main__":
    main()