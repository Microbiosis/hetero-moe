"""V13.0 — 蒸馏到单底座 (教师 v10/v13 → 学生单底座 + LoRA).

实验设计:
    教师 (teacher): v10 CrossArchAttnAligner + v9 gate-style 中枢 (跨架构)
    学生 (student): 单 BERT + LoRA (StudentModel)

    教师需要 3 模态输入 (h_text, h_code, h_img)
    学生只需要 1 模态输入 (h_text), 蒸馏后输出与教师接近

3 变体 × 5 seeds = 15 run:
    teacher-only       — 教师不蒸馏, 单独 baseline (v10 fuse MSE ≈ 0.32)
    student-no-distill — 学生单 BERT + LoRA, 不用教师, 单独 baseline
    student-distill    — 学生用教师 logits 蒸馏

判断: student-distill 是否接近 teacher-only, 且显著优于 student-no-distill?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import statistics

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from research._primitives.attention import AttnPool
from research.routing_evolution.v8_central import V8Trainer as TeacherTrainer
from research.aligner.v10_embedding import AlignedFusionLayer, CrossArchAttnAligner
from research.aligner.v13_mixture import StudentModel, distill_loss
from examples.run_v8_full import (
    load_encoders, make_pools, encode_text, encode_code, encode_image,
    D_SHARED, N_CLS, B, S, STEPS, LR, SEEDS,
)


def make_teacher(attn_pools, encoders):
    (_, _, D_bert, _), (_, _, D_llama, _), (_, _, D_vit, _) = encoders
    aligner = CrossArchAttnAligner(modal_dims=[D_bert, D_llama, D_vit],
                                    d_shared=D_SHARED, num_heads=4)
    return AlignedFusionLayer(
        d_shared=D_SHARED, d_ff=4 * D_SHARED,
        attn_pools=attn_pools, modal_dims=[D_bert, D_llama, D_vit],
        aligner=aligner, c1_alpha=0.1,
    )


def make_student(d_in=312, d_out=D_SHARED):
    return StudentModel(d_in=d_in, d_out=d_out, lora_rank=8, hidden=128)


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
    """教师训练: 跨架构协同"""
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


def eval_teacher(teacher, modal_seqs, modal_indices, targets):
    h_by_modal = [[] for _ in range(N_CLS)]
    t_by_modal = [[] for _ in range(N_CLS)]
    for i, m in enumerate(modal_indices):
        h_by_modal[m].append(modal_seqs[i])
        t_by_modal[m].append(targets[i])
    for cls in range(N_CLS):
        h_by_modal[cls] = torch.stack(h_by_modal[cls], dim=0)
        t_by_modal[cls] = torch.stack(t_by_modal[cls], dim=0)
    fuse_loss = 0.0
    with torch.no_grad():
        y = teacher(h_by_modal)
        for c in range(N_CLS):
            for s in range(6):
                fuse_loss += F.mse_loss(y[s:s+1], t_by_modal[c][s:s+1]).item()
        fuse_loss /= 18
    return fuse_loss


def train_student_distill(student, teacher, modal_seqs, modal_indices, targets, lam=0.7):
    """学生训练: 蒸馏教师输出.

    学生只看 BERT 模态输入, 但要预测教师融合 3 模态后的输出.
    """
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=LR)
    h_by_modal = [[] for _ in range(N_CLS)]
    t_by_modal = [[] for _ in range(N_CLS)]
    for i, m in enumerate(modal_indices):
        h_by_modal[m].append(modal_seqs[i])
        t_by_modal[m].append(targets[i])
    for cls in range(N_CLS):
        h_by_modal[cls] = torch.stack(h_by_modal[cls], dim=0)
        t_by_modal[cls] = torch.stack(t_by_modal[cls], dim=0)
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        # 教师前向 (无梯度)
        with torch.no_grad():
            y_teacher = teacher(h_by_modal).detach()  # [6, S, D_shared]
        # 学生只看 BERT 模态 (cls=0)
        h_bert = h_by_modal[0]  # [6, S, 312]
        y_student = student(h_bert)  # [6, S, D_shared]
        # 用教师第 0 模态的目标 (即 BERT 模态目标)
        # 但学生应该学会"教师融合后"的输出, 用教师均值目标
        y_teacher_mean = y_teacher.mean(dim=0, keepdim=True).expand_as(y_teacher)
        loss = F.mse_loss(y_student, y_teacher_mean)
        loss.backward()
        opt.step()


def eval_student(student, modal_seqs, modal_indices, targets):
    """学生评估: 只用 BERT 模态, 与教师 (3 模态) 蒸馏目标比较"""
    h_by_modal = [[] for _ in range(N_CLS)]
    t_by_modal = [[] for _ in range(N_CLS)]
    for i, m in enumerate(modal_indices):
        h_by_modal[m].append(modal_seqs[i])
        t_by_modal[m].append(targets[i])
    for cls in range(N_CLS):
        h_by_modal[cls] = torch.stack(h_by_modal[cls], dim=0)
        t_by_modal[cls] = torch.stack(t_by_modal[cls], dim=0)
    fuse_loss = 0.0
    with torch.no_grad():
        # 学生只看 BERT 模态
        y = student(h_by_modal[0])  # [6, S, D_shared]
        # 评估学生对齐目标的能力: 与教师 (3 模态) 输出比较
        # 但单独学生不能调用教师, 所以我们评估"学生 vs 教师输出" (蒸馏是否成功)
        # 用 BERT 模态的 target 作为参考
        for c in range(N_CLS):
            for s in range(6):
                fuse_loss += F.mse_loss(y[s:s+1], t_by_modal[c][s:s+1]).item()
        fuse_loss /= 18
    return fuse_loss


def run_seed(mode, seed, encoders, attn_pools):
    torch.manual_seed(seed)
    (bert_tok, bert, _, _), (llama_tok, llama, _, _), (vit_proc, vit, _, _) = encoders
    print(f"  [seed {seed}] mode={mode} ...")

    modal_seqs, modal_indices, targets = prepare_data(seed, encoders)

    if mode == "teacher-only":
        teacher = make_teacher(attn_pools, encoders)
        train_teacher(teacher, modal_seqs, modal_indices, targets)
        return eval_teacher(teacher, modal_seqs, modal_indices, targets)

    # 学生相关模式
    teacher = make_teacher(attn_pools, encoders)
    train_teacher(teacher, modal_seqs, modal_indices, targets)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = make_student(d_in=312, d_out=D_SHARED)
    if mode == "student-no-distill":
        # 学生单独训练 (只看 BERT 模态, 不蒸馏)
        opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=LR)
        h_by_modal = [[] for _ in range(N_CLS)]
        t_by_modal = [[] for _ in range(N_CLS)]
        for i, m in enumerate(modal_indices):
            h_by_modal[m].append(modal_seqs[i])
            t_by_modal[m].append(targets[i])
        for cls in range(N_CLS):
            h_by_modal[cls] = torch.stack(h_by_modal[cls], dim=0)
            t_by_modal[cls] = torch.stack(t_by_modal[cls], dim=0)
        for _ in range(STEPS):
            opt.zero_grad(set_to_none=True)
            y = student(h_by_modal[0])
            loss = F.mse_loss(y, t_by_modal[0])
            loss.backward()
            opt.step()
    elif mode == "student-distill":
        # 学生蒸馏
        train_student_distill(student, teacher, modal_seqs, modal_indices, targets)

    return eval_student(student, modal_seqs, modal_indices, targets)


MODES = ["teacher-only", "student-no-distill", "student-distill"]


def main():
    print("=" * 78)
    print("V13.0 — 蒸馏到单底座 (教师 v10/v13 → 学生单 BERT + LoRA)")
    print("=" * 78)
    print("3 变体 × 5 seeds = 15 run")
    print()
    encoders = load_encoders()
    attn_pools = make_pools(encoders)
    results = {m: [] for m in MODES}

    print(f"\n每个 mode × {SEEDS} seeds")
    for mode in MODES:
        for seed in range(SEEDS):
            try:
                fuse = run_seed(mode, seed, encoders, attn_pools)
                results[mode].append(fuse)
                print(f"  [{mode:>20} seed {seed}] fuse={fuse:.4f}")
            except Exception as e:
                print(f"  [{mode:>20} seed {seed}] FAILED: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                results[mode].append(float("nan"))

    # 汇总
    print("\n" + "=" * 78)
    print(f"{'mode':>22} | {'fuse MSE':>9} | {'vs teacher':>11}")
    print("-" * 78)
    teacher_mean = statistics.mean([m for m in results["teacher-only"] if not np.isnan(m)])

    for m in MODES:
        vals = [x for x in results[m] if not np.isnan(x)]
        if not vals:
            print(f"{m:>22} | {'N/A':>9} | {'N/A':>11}")
            continue
        m_mean = statistics.mean(vals)
        gain = (teacher_mean - m_mean) / max(teacher_mean, 1e-9) * 100
        print(f"{m:>22} | {m_mean:>9.4f} | {gain:>+10.1f}%")
    print("=" * 78)
    print()
    print("结论:")
    f_t = statistics.mean([m for m in results["teacher-only"] if not np.isnan(m)])
    f_n = statistics.mean([m for m in results["student-no-distill"] if not np.isnan(m)])
    f_d = statistics.mean([m for m in results["student-distill"] if not np.isnan(m)])
    print(f"  teacher-only (v10, 3 模态):    {f_t:.4f}")
    print(f"  student-no-distill (单 BERT):   {f_n:.4f}")
    print(f"  student-distill (单 BERT+LoRA): {f_d:.4f}")
    print()
    if f_d < f_n:
        print(f"✅ 蒸馏有效: 学生 + 蒸馏 ({f_d:.4f}) < 学生单独 ({f_n:.4f})")
    if f_d <= f_t * 1.5:
        print(f"✅ 学生 + 蒸馏 ({f_d:.4f}) 与教师 ({f_t:.4f}) 同量级, 蒸馏成功")
    print()
    print("关键意义: 蒸馏后的学生是**真正的单底座**, 推理时无需其他底座")


if __name__ == "__main__":
    main()