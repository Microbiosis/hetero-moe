#!/usr/bin/env python3
"""exp86_assembled_train.py — **组件级组装原型**（用户思路：拆→挑→组装）。

## 这是什么

不是把两个模型的**输出特征**首尾相接（那是阶段 D 的诊断实验），
而是**把基模的组件（权重）搬进我们自己的骨架，组装成一个新架构，随训练更新**。

```
我们自研的骨架（exp80，D=64，每时间片一个 token）
  EEG (B,63,250)
   ├─ ch_mix / enc / t_down              ← 我们的 CNN 编码器
   ├─ [ 组装块 ]                          ← ★ 本实验新增
   │    ├─ LN → MHA(64)                   ← 嫁接自 signal-jepa encoder 第 0 层
   │    │                                    （逐键同形，零改动直接搬）
   │    └─ LN → FFN                        ← 嫁接自 donor（biot / mirepnet）
   │         ↑ 两侧学习投影 64→d→64        （维度不同 ⇒ 必须训练）
   ├─ VQ(K,64) → codes
   └─ decoder → 重建波形
```

## 两个问题一次回答

1. **组装后的新架构自己有多强？**（重建层 RSA 是否超过 S1a 的 +0.0372）
2. **语义能不能写进离散码本？**（S1b 的核心判据）

⇒ 第 2 点要靠**两层语义监督**（关键细节，写错就测不到）：

| 损失 | 挂在哪 | 梯度到哪 | 作用 |
|---|---|---|---|
| `L_nce_z` | 量化**前**的 `z.mean(1)` | encoder | 让编码器的 token 分配更可分图像 |
| `L_nce_q` | **码本查表结果** `E[codes].mean(1)` | **码本本身** | 让码本的几何朝语义走 |

⚠️ 若只把语义损失挂在 straight-through 的量化向量上，**梯度会被 detach 挡住**
⇒ 码本学不到语义 ⇒ 实验会假阴性。
⇒ 诚实判据因此是**纯离散读出**（`onehot_mean` / `code_ids`）：
若它们也变显著 ⇒ **token 身份**本身带语义（而不只是码本向量带）。

## 读出与监控

- 重建 MSE、in-batch 图像检索准确率（64 选 1）、码本利用率与熵。

## 用法

    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp86_assembled_train.py \
      -v hf://buckets/illusion-brain/training-data:/training:ro \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor cpu-upgrade --timeout 180m -d \
      --donor-ffn none --steps 6000 --out-dir /domains/runs/illusionbrain-align/exp86/ffn-none
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

EXP_DIR = Path(__file__).resolve().parent
_PKG_CANDIDATES = [EXP_DIR / "models", EXP_DIR.parent / "models", EXP_DIR]
PKG_DIR = next((c for c in _PKG_CANDIDATES if (c / "registry.py").exists()),
               _PKG_CANDIDATES[0])
for _p in (str(EXP_DIR), str(PKG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import base_specs as BS            # noqa: E402
import domain_adapters as DA       # noqa: E402

TRAIN_PT = Path("/training/eeg/things-eeg/Preprocessed_data_250Hz_whiten")
FEAT_TRAIN = "/training/eeg/eeg-image-decode/ViT-H-14_features_train.pt"
MODELS_DIR = Path("/training/models")
SEED = 0
K_CODES = 1024
D_MODEL = 64
T_STRIDE = 8
NHEAD = 4

# 供体组件（由 exp86b_extract_donors.py 抽好落桶）⇒ 训练进程**不需要 braindecode**
# ⇒ 一次组装里可以混装**不同 braindecode 版本**的模型的组件
DONOR_DIR = Path("/domains/runs/illusionbrain-align/donors")
ATTN_DONOR = "signal-jepa"          # 逐键同形的唯一注意力供体（MHA 64）
FFN_DONORS = ("none", "mirepnet", "biot", "eegdino-small")


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 组装模型
# ---------------------------------------------------------------------------
def build_assembled(n_ch=63, n_times=250, n_codes=K_CODES, dim=D_MODEL,
                    attn_state=None, ffn_donor=None):
    """组装模型工厂（返回 nn.Module 实例）。

    骨架与我们自研 tokenizer 一致（CNN 编码 → 时间片 token → VQ → 解码），
    中间插一个**组装块**：
      - 注意力：从 donor **逐键同形**地搬进来（signal-jepa encoder 第 0 层）
      - FFN：维度不同 ⇒ 两侧学习投影 + donor 的 FFN 权重（可训练）
    """
    import torch
    import torch.nn.functional as F
    from torch import nn

    from exp80_eeg_tokenizer_train import VQ

    class _Assembled(nn.Module):
        kind = "assembled"

        def __init__(self):
            super().__init__()
            self.dim = dim
            self.n_times = n_times
            self.ch_mix = nn.Conv1d(n_ch, dim, 1)
            self.enc = nn.Sequential(
                nn.Conv1d(dim, dim, 5, padding=2), nn.GELU(), nn.GroupNorm(8, dim),
                nn.Conv1d(dim, dim, 5, padding=2), nn.GELU(), nn.GroupNorm(8, dim))
            self.t_down = nn.Conv1d(dim, dim, T_STRIDE, stride=T_STRIDE)
            # --- 组装块 ---
            self.use_attn = attn_state is not None
            if self.use_attn:
                self.norm1 = nn.LayerNorm(dim)
                self.attn = nn.MultiheadAttention(dim, NHEAD, batch_first=True)
                self.attn.load_state_dict(attn_state, strict=True)   # 逐键同形 ⇒ 直接搬
            self.use_ffn = ffn_donor is not None
            if self.use_ffn:
                d = ffn_donor["d"]
                h = ffn_donor["h"]
                self.norm2 = nn.LayerNorm(dim)
                self.ffn_in = nn.Linear(dim, d)            # 入口学习投影
                self.ffn_w1 = nn.Linear(d, h)
                self.ffn_w2 = nn.Linear(h, d)
                self.ffn_out = nn.Linear(d, dim)           # 出口学习投影
                with torch.no_grad():
                    self.ffn_w1.weight.copy_(ffn_donor["w1"])
                    self.ffn_w1.bias.copy_(ffn_donor["b1"])
                    self.ffn_w2.weight.copy_(ffn_donor["w2"])
                    self.ffn_w2.bias.copy_(ffn_donor["b2"])
            # --- VQ（同 exp80：直线估计 + EMA + 死码重启）---
            self.vq = VQ(n_codes, dim)
            # --- decoder ---
            self.dec_up = nn.ConvTranspose1d(dim, dim, T_STRIDE, stride=T_STRIDE)
            self.dec_out = nn.Conv1d(dim, n_ch, 1)
            # --- 语义头（只用于训练时的对比损失）---
            self.head_z = nn.Linear(dim, 256)
            self.head_img = nn.Linear(1024, 256)

        # --- 编码 ---
        def encode(self, x):
            h = self.ch_mix(x)
            h = self.enc(h)
            h = self.t_down(h)
            z = h.transpose(1, 2)                    # (B, T, D)
            if self.use_attn:
                a, _ = self.attn(self.norm1(z), self.norm1(z), self.norm1(z),
                                 need_weights=False)
                z = z + a
            if self.use_ffn:
                u = self.norm2(z)
                u = self.ffn_in(u)
                u = self.ffn_w2(F.gelu(self.ffn_w1(u)))
                z = z + self.ffn_out(u)
            return z

        def decode(self, zq, n_times):
            h = zq.transpose(1, 2)
            h = self.dec_up(h)
            h = F.pad(h, (0, max(0, n_times - h.shape[-1])))[:, :, :n_times]
            return self.dec_out(h)

        def forward(self, x):
            z = self.encode(x)
            zq, vq_loss, codes = self.vq(z)
            rec = self.decode(zq, x.shape[-1])
            return rec, vq_loss, codes, z, zq

    return _Assembled()


# ---------------------------------------------------------------------------
# 组件加载（只读桶里的 .pt ⇒ 不需要 braindecode）
# ---------------------------------------------------------------------------
def load_donor(name):
    fp = DONOR_DIR / f"{name}.pt"
    if not fp.exists():
        raise FileNotFoundError(
            f"缺少供体组件 {fp} ⇒ 先跑 exp86b_extract_donors.py")
    import torch
    return torch.load(fp, map_location="cpu", weights_only=False)


def extract_attn_state(donor_name):
    """取 donor 的注意力 state_dict（逐键同形 ⇒ 直接搬）。"""
    d = load_donor(donor_name)
    if "attn" not in d:
        raise RuntimeError(f"{donor_name} 没有抽到注意力")
    return {k: v.clone() for k, v in d["attn"]["state"].items()}


def extract_ffn_donor(donor_name):
    """从 donor 组件里取 FFN（h, d, w1, b1, w2, b2）。"""
    d = load_donor(donor_name)
    if "ffn" not in d:
        raise RuntimeError(f"{donor_name} 没有抽到 FFN")
    f = d["ffn"]
    return {"h": f["h"], "d": f["d"], "w1": f["w1"].clone(), "b1": f["b1"].clone(),
            "w2": f["w2"].clone(), "b2": f["b2"].clone(),
            "src": f"{donor_name}:{f.get('src', '')}"}


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def info_nce(a, b, temp=0.07):
    """InfoNCE（a、b 各自 L2 归一化）⇒ 返回 (loss, in-batch 检索准确率)。"""
    import torch
    import torch.nn.functional as F
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.t() / temp
    tgt = torch.arange(a.shape[0], device=a.device)
    loss = F.cross_entropy(logits, tgt)
    acc = (logits.argmax(1) == tgt).float().mean()
    return loss, float(acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor-ffn", default="none", choices=list(FFN_DONORS))
    ap.add_argument("--attn-donor", default=ATTN_DONOR)
    ap.add_argument("--freeze-attn", action="store_true",
                    help="嫁接来的注意力冻结（只当固定组件用）")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--codes", type=int, default=K_CODES)
    ap.add_argument("--lambda-nce", type=float, default=0.5)
    ap.add_argument("--lambda-vq", type=float, default=0.25)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--max-seconds", type=int, default=0)
    ap.add_argument("--out-dir", default="/domains/runs/illusionbrain-align/exp86/assembled")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"=== 组装原型：骨架(自研 D={D_MODEL}) + 注意力({args.attn_donor}) "
        f"+ FFN({args.donor_ffn}) + 语义监督 ===")

    # --- 数据：train split（白化 EEG + 同序的图像特征）---
    fp = TRAIN_PT / f"sub-{args.subject:02d}" / "train.pt"
    try:
        obj = torch.load(fp, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        obj = torch.load(fp, map_location="cpu", weights_only=False)
    eeg = obj["eeg"]
    log(f"  train split EEG {tuple(eeg.shape)}")
    ftr = torch.load(FEAT_TRAIN, map_location="cpu", weights_only=False)
    IMG = ftr["img_features"].numpy().astype(np.float32)   # (16540, 1024)
    log(f"  图像特征 {IMG.shape}")

    n_img, n_rep = eeg.shape[0], eeg.shape[1]

    # --- 组装（组件从桶里读 ⇒ 不需要 braindecode）---
    attn_state = extract_attn_state(args.attn_donor)
    log(f"  嫁接注意力 {args.attn_donor}：键 {sorted(attn_state)}")
    ffn_donor = None
    if args.donor_ffn != "none":
        ffn_donor = extract_ffn_donor(args.donor_ffn)
        log(f"  嫁接 FFN {ffn_donor['src']}：{ffn_donor['d']}→"
            f"{ffn_donor['h']}→{ffn_donor['d']}（两侧学习投影 "
            f"{D_MODEL}↔{ffn_donor['d']}）")

    model = build_assembled(n_codes=args.codes, attn_state=attn_state,
                            ffn_donor=ffn_donor)
    if args.freeze_attn:
        for p in model.attn.parameters():
            p.requires_grad_(False)
        log("  嫁接的注意力**冻结**")
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  可训练参数 {n_par:,}")

    groups = [
        {"params": [p for n, p in model.named_parameters()
                    if not n.startswith(("vq.", "head_"))], "lr": args.lr},
        {"params": list(model.vq.parameters()), "lr": args.lr},
        {"params": list(model.head_z.parameters()) + list(model.head_img.parameters()),
         "lr": args.lr},
    ]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)

    t0 = time.time()
    hist = []
    for step in range(1, args.steps + 1):
        # 采样：一个批次 = B 个（图像, 重复）试次，并取对应图像特征
        im_idx = rng.integers(0, n_img, size=args.batch)
        rp_idx = rng.integers(0, n_rep, size=args.batch)
        rows = [np.asarray(eeg[int(i), int(r)]) for i, r in zip(im_idx, rp_idx)]
        x = torch.from_numpy(np.stack(rows).astype(np.float32))
        img = torch.from_numpy(IMG[im_idx])

        rec, vq_loss, codes, z, zq = model(x)
        l_rec = F.mse_loss(rec, x)
        # 语义损失：两条路（编码器 + 码本）
        nce_z, acc_z = info_nce(model.head_z(z.mean(1)), model.head_img(img))
        # 码本查表结果（梯度直达码本 ⇒ 不被 straight-through 的 detach 挡住）
        emb = model.vq.lookup(codes)
        nce_q, acc_q = info_nce(model.head_z(emb.mean(1)), model.head_img(img))
        loss = l_rec + args.lambda_nce * (nce_z + nce_q) + args.lambda_vq * vq_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()

        if step % 50 == 0:
            with torch.no_grad():
                model.vq.restart_dead(z.detach())
        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                u = model.vq.usage(codes)
            hist.append({"step": step, "loss": float(loss.item()),
                         "rec_mse": float(l_rec.item()),
                         "nce_z": float(nce_z.item()), "nce_q": float(nce_q.item()),
                         "retrieval_acc": acc_q,
                         "utilization": u["utilization"],
                         "entropy": u["entropy_nats"]})
            log(f"  step {step:>5} | loss {float(loss.item()):.4f} "
                f"| 重建 {float(l_rec.item()):.4f} "
                f"| nce_z {float(nce_z.item()):.3f} nce_q {float(nce_q.item()):.3f} "
                f"| 检索准确率 {acc_q:.3f} "
                f"| 码利用率 {u['utilization']:.1%} 熵 {u['entropy_nats']:.2f} "
                f"| {time.time()-t0:.0f}s")
        if args.max_seconds and (time.time() - t0) > args.max_seconds:
            log(f"  到达时间上限 {args.max_seconds}s，提前收尾")
            break

    ck = {"state_dict": model.state_dict(), "args": vars(args),
          "n_codes": args.codes, "d_model": D_MODEL, "arch": "assembled",
          "donor_ffn": args.donor_ffn, "hist": hist}
    out = out_dir / "tokenizer_assembled.pt"
    torch.save(ck, out)
    log(f"权重写入 {out}（{out.stat().st_size/1e6:.1f} MB）")
    (out_dir / "train_log.json").write_text(
        json.dumps({"args": vars(args), "hist": hist}, indent=2,
                   ensure_ascii=False, default=str), encoding="utf-8")
    log(f"总耗时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:
        log(f"✗ {type(e).__name__}: {e}")
        log(traceback.format_exc()[-1500:])
        raise SystemExit(1)
