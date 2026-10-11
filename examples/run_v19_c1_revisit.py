"""V19.0 — C-1 根因补全分析 (5 变体 × 5 seeds).

H3 (长训练): 100 步 vs 500 步 — 长训练是否反转 v9 v12 v15 v17 负面发现?
H4 (容量): 学生 MLP 小/中/大 — 容量能否反转负面发现?

5 变体 × 5 seeds = 25 run:
    baseline-100       — v7 (无中枢), 100 步 (基线)
    c1-100             — C-1 中枢, 100 步 (复现 v9)
    c1-500             — C-1 中枢, 500 步 (H3: 长训练)
    c1-small           — C-1 中枢, 学生容量小 [16], 100 步
    c1-large           — C-1 中枢, 学生容量大 [128, 64], 100 步 (H4)

核心判断: 长训练 / 大容量能否反转 v9 的"gate-style 最好"结论?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import statistics

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from research._primitives.attention import AttnPool
from research.routing_evolution.v8_central import V8Trainer
from research.aligner.v10_embedding import AlignedFusionLayer, CrossArchAttnAligner
from research.central_diagnostics.v19_c1_analysis import DiagnosticRunner
from examples.run_v8_full import (
    load_encoders, make_pools, encode_text, encode_code, encode_image,
    D_SHARED, N_CLS, B, S, LR, SEEDS,
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


def train_eval(layer, modal_seqs, modal_indices, targets, steps):
    h_by_modal = [[] for _ in range(N_CLS)]
    t_by_modal = [[] for _ in range(N_CLS)]
    for i, m in enumerate(modal_indices):
        h_by_modal[m].append(modal_seqs[i])
        t_by_modal[m].append(targets[i])
    for cls in range(N_CLS):
        h_by_modal[cls] = torch.stack(h_by_modal[cls], dim=0)
        t_by_modal[cls] = torch.stack(t_by_modal[cls], dim=0)
    trainer = V8Trainer(layer, lr=LR, phase=2)
    for _ in range(steps):
        trainer.optimizer.zero_grad(set_to_none=True)
        x_list = [F.linear(h.unsqueeze(0), F.pad(layer.P_m[c],
                              (0, D_SHARED - layer.P_m[c].shape[1]))) if layer.P_m[c].shape[1] != D_SHARED
                  else h.unsqueeze(0)
                  for c, h in zip(modal_indices, modal_seqs)]  # placeholder
        # 简化: 直接 per-sample (linear 投影)
        losses = []
        for i in range(B):
            cls_idx = modal_indices[i]
            x = F.linear(modal_seqs[i].unsqueeze(0), layer.P_m[cls_idx])
            y = layer(x)
            losses.append(F.mse_loss(y, targets[i].unsqueeze(0)))
        total = sum(losses) / len(losses)
        total.backward()
        trainer.optimizer.step()
    fuse_loss = 0.0
    with torch.no_grad():
        for i in range(B):
            cls_idx = modal_indices[i]
            x = F.linear(modal_seqs[i].unsqueeze(0), layer.P_m[cls_idx])
            y = layer(x)
            fuse_loss += F.mse_loss(y, targets[i].unsqueeze(0)).item()
        fuse_loss /= B
    return fuse_loss


def run_seed(mode, seed, encoders, attn_pools, steps=100, capacity=None):
    """mode ∈ {baseline-100, c1-100, c1-500, c1-small, c1-large}.

    capacity 给定时 (H4 轴): 缩放 FFN 隐宽 d_ff —— 1× / 默认 4× / 8×。
    """
    torch.manual_seed(seed)
    (bert_tok, bert, _, _), (llama_tok, llama, _, _), (vit_proc, vit, _, _) = encoders
    print(f"  [seed {seed}] mode={mode} steps={steps} capacity={capacity} ...")

    modal_seqs, modal_indices, targets = prepare_data(seed, encoders)

    # 构造 layer (用 v7 双路由 + 中枢 broadcast)
    (_, _, D_bert, _), (_, _, D_llama, _), (_, _, D_vit, _) = encoders
    modal_dims = [D_bert, D_llama, D_vit]
    aligner = CrossArchAttnAligner(modal_dims=modal_dims, d_shared=D_SHARED, num_heads=4)
    # 2026-10-11 修正: H4「学生容量」轴此前是空转 —— run_seed 收了 capacity 参数却从
    # 不用, C1OnlyModel(student_capacity=…) 也从未被构造, 于是 c1-small / c1-large 与
    # c1-100 逐位同构, 「容量不是根因」这个结论在构造上就没有支撑。这里让 capacity
    # 真正作用在 FFN 隐宽 d_ff 上 (专家数被 len(modal_dims) 钉死, 隐宽才是真容量轴)。
    mult = {"c1-small": 1, "c1-large": 8}.get(mode, 4)
    layer = AlignedFusionLayer(
        d_shared=D_SHARED, d_ff=mult * D_SHARED,
        attn_pools=attn_pools, modal_dims=modal_dims,
        aligner=aligner, c1_alpha=0.0, c1_mode="broadcast",   # 复现 v8.0 c1-default broadcast 模式
    )

    # 训练
    trainer = V8Trainer(layer, lr=LR, phase=2)
    h_by_modal = [[] for _ in range(N_CLS)]
    t_by_modal = [[] for _ in range(N_CLS)]
    for i, m in enumerate(modal_indices):
        h_by_modal[m].append(modal_seqs[i])
        t_by_modal[m].append(targets[i])
    for cls in range(N_CLS):
        h_by_modal[cls] = torch.stack(h_by_modal[cls], dim=0)
        t_by_modal[cls] = torch.stack(t_by_modal[cls], dim=0)
    for _ in range(steps):
        trainer.optimizer.zero_grad(set_to_none=True)
        y = layer(h_by_modal)
        loss = sum(F.mse_loss(y, t_by_modal[c]) for c in range(N_CLS)) / N_CLS
        loss.backward()
        trainer.optimizer.step()
    # eval
    fuse_loss = 0.0
    with torch.no_grad():
        y = layer(h_by_modal)
        for c in range(N_CLS):
            for s in range(6):
                fuse_loss += F.mse_loss(y[s:s+1], t_by_modal[c][s:s+1]).item()
        fuse_loss /= 18
    return fuse_loss


MODES = ["baseline-100", "c1-100", "c1-500", "c1-small", "c1-large"]


def main():
    print("=" * 78)
    print("V19.0 — C-1 根因补全: 长训练 (H3) + 容量 (H4) 对照")
    print("=" * 78)
    print("5 变体 × 5 seeds = 25 run")
    print()
    encoders = load_encoders()
    attn_pools = make_pools(encoders)
    results = {m: [] for m in MODES}
    runner = DiagnosticRunner()

    print(f"\n每个 mode × {SEEDS} seeds")
    for mode in MODES:
        for seed in range(SEEDS):
            try:
                steps = 500 if mode == "c1-500" else 100
                capacity = {"c1-small": 1, "c1-large": 8}.get(mode)
                fuse = run_seed(mode, seed, encoders, attn_pools, steps=steps,
                                capacity=capacity)
                results[mode].append(fuse)
                print(f"  [{mode:>13} seed {seed}] fuse={fuse:.4f}")
                # 记录到 diagnostic runner
                if "c1" in mode:
                    hyp = "H3" if "100" in mode or "500" in mode else "H4"
                    runner.record(hyp, mode, fuse)
            except Exception as e:
                print(f"  [{mode:>13} seed {seed}] FAILED: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                results[mode].append(float("nan"))

    # 汇总
    print("\n" + "=" * 78)
    print(f"{'mode':>14} | {'fuse MSE':>9} | {'vs c1-100':>11}")
    print("-" * 78)
    c1_100_mean = statistics.mean([m for m in results["c1-100"] if not np.isnan(m)])
    for m in MODES:
        vals = [x for x in results[m] if not np.isnan(x)]
        if not vals:
            print(f"{m:>14} | {'N/A':>9} | {'N/A':>11}")
            continue
        m_mean = statistics.mean(vals)
        gain = (c1_100_mean - m_mean) / max(c1_100_mean, 1e-9) * 100
        print(f"{m:>14} | {m_mean:>9.4f} | {gain:>+10.1f}%")
    print("=" * 78)
    print()
    print(runner.summary())
    print()
    print("结论:")
    f_b = statistics.mean([m for m in results["baseline-100"] if not np.isnan(m)])
    f_100 = statistics.mean([m for m in results["c1-100"] if not np.isnan(m)])
    f_500 = statistics.mean([m for m in results["c1-500"] if not np.isnan(m)])
    f_small = statistics.mean([m for m in results["c1-small"] if not np.isnan(m)])
    f_large = statistics.mean([m for m in results["c1-large"] if not np.isnan(m)])
    print(f"  baseline-100 (无中枢):   {f_b:.4f}")
    print(f"  c1-100 (原 v8.0):        {f_100:.4f}")
    print(f"  c1-500 (H3 长训练):      {f_500:.4f}  (vs c1-100: {(f_100 - f_500) / f_100 * 100:+.1f}%)")
    print(f"  c1-small (H4 小容量):    {f_small:.4f}")
    print(f"  c1-large (H4 大容量):    {f_large:.4f}")
    print()
    if f_500 < f_100:
        print(f"✅ H3 确认: 长训练改善 C-1 (v8.0 negative-finding 受时间限制)")
    if f_large < f_100:
        print(f"✅ H4 确认: 大容量改善 C-1")


if __name__ == "__main__":
    main()