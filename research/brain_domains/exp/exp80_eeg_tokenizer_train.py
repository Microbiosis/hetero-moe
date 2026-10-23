#!/usr/bin/env python3
"""exp80_eeg_tokenizer_train.py — S1：自训 EEG 离散 tokenizer（量化码本路线）。

## 为什么必须自己写一个（而不是复用某个基模）

exp75 的结论：**7 个基模里只有 braintokenizer 是「时间片级」token**——
它的输出是 (B, 19ch, 512 步) ⇒ 每个时间步一个码。
其余 6 个都是「**每通道一个 token、没有时间轴**」：

| 模型 | token 形状 | 时间轴？ |
|---|---|---|
| cbramod | (B, 63, 200) = 63 个通道位置 | ❌ |
| signal-jepa | (B, 62, 64) | ❌ |
| mirepnet | (B, 61, 256) | ❌ |
| biot / eegdino / braintokenizer(量化码是 (B,19,512) ✅) | | |

⇒ 脑原生的 tokenizer 必须**每时间片一个 token**（Chameleon 的图像 VQ 那样），
否则没法做「原生交错序列」（脑段要是一串 token）。

## 架构（照 braintokenizer 的骨架，但自训）

```
EEG (B, 63ch, 250 样本 @250Hz)
  ├─ ch_mix   Conv1d(63→64, k=1)            通道混合
  ├─ enc     2 × [Conv1d(64, k=5) + GELU + GroupNorm]
  ├─ t_down  Conv1d(64→64, k=8, stride=8)   时间下采样 250 → 31 个时间片
  ├─ VQ      码本 (K=1024, d=64)            直线估计 + EMA 更新
  └─ dec     ConvTranspose1d(64, k=8, s=8) → Conv1d(64→63, k=1) → 重建波形
```

**训练目标 = 掩码重建**：随机遮住 50% 的时间片，decoder 只看**可见 token**
（被遮位置换成可学习的 mask embedding），重建整段波形。
⇒ 码本被迫编码**上下文相关**的信息，而不是逐点的信号幅度。

## 判据（预先写死，跑完只看这些）

1. **量化码的 RSA 显著**（p<0.05），且不低于 braintokenizer 的 +0.0155；
2. **码本利用率** > 50%（避免塌成位置编码）；
3. **对照 A**：同一 encoder **不量化**（连续特征）的 RSA ⇒ 量化是否真的有用；
4. **对照 B**：**随机码本**（不训练）的 RSA ⇒ 「量化这个操作」本身有没有用。

## 数据

桶内 `/training/eeg/things-eeg/Preprocessed_data_250Hz_whiten/sub-XX/train.pt`
（**小写** things-eeg；已白化、250 Hz、63 通道 ⇒ 正好是我们的输入域；
实测 train.pt = (16540, 4, 63, 250) float16 ⇒ **真正的训练集**。
用 `torch.load(mmap=True)` 避免把 4.2 GB 的数组整个读进内存）。

⚠️ 顺带记一笔：桶里的 `omnibrain4d/eeg_train/` 目录名有误导 ——
它装的是 **test split**（1654×80，实测 2,084,040,260 ÷ (63×250×2) = 132,320），
不是训练集 ⇒ 评测脚本拿它当「训练集」会泄漏（见 FINDINGS §十九）。

## 用法

    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp80_eeg_tokenizer_train.py \
      -v hf://buckets/illusion-brain/training-data:/training:ro \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor cpu-upgrade --timeout 180m -d --steps 4000 --out-dir /domains/runs/illusionbrain-align/exp80
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
import torch
from torch import nn
import torch.nn.functional as F

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

EXP_DIR = Path(__file__).resolve().parent
# ⚠️ 桶里是**小写** `things-eeg`（写成 `things-EEG` 会列到空目录 ⇒ 静默取不到数据）
TRAIN_ROOT = Path("/training/eeg/things-eeg/Preprocessed_data_250Hz_whiten")
OUT_ROOT = Path("/domains/runs/illusionbrain-align/exp80")
SEED = 0
K_CODES = 1024
D_MODEL = 64
T_STRIDE = 8


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def open_train(subject):
    """打开 train.pt（mmap，避免 4.2 GB 全量进内存）。返回 (eeg数组, 取样函数)。"""
    import torch
    fp = TRAIN_ROOT / f"sub-{subject:02d}" / "train.pt"
    if not fp.exists():
        raise FileNotFoundError(str(fp))
    try:
        obj = torch.load(fp, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        obj = torch.load(fp, map_location="cpu", weights_only=False)
    log(f"  train.pt keys：{list(obj.keys()) if isinstance(obj, dict) else type(obj)}")
    eeg = obj["eeg"] if isinstance(obj, dict) and "eeg" in obj else obj
    log(f"  eeg：{tuple(eeg.shape)} {getattr(eeg, 'dtype', None)}")
    return obj, eeg


def sample_batch(eeg, rng, bs):
    """随机取 bs 个试次（eeg 的前两轴是 [n_img, n_rep]）。

    ⚠️ `train.pt` 里的 `eeg` 是 **numpy 数组**（实测 keys: eeg/label/img/text/
    session/ch_names/times）⇒ 直接 `torch.stack(numpy_rows)` 会报
    「expected Tensor as element 0」⇒ 统一先 `torch.as_tensor`。
    """
    import torch
    n_img, n_rep = eeg.shape[0], eeg.shape[1]
    flat_n = n_img * n_rep
    idx = rng.integers(0, flat_n, size=bs)
    rows = []
    for i in idx:
        im, rp = divmod(int(i), n_rep)
        rows.append(torch.as_tensor(np.asarray(eeg[im, rp])))
    x = torch.stack(rows).float()
    return x          # (bs, C, T)


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
class VQ(nn.Module):
    """向量量化：直线估计 + EMA 码本更新（标准 VQ-VAE 做法）。"""

    def __init__(self, n_codes=K_CODES, dim=D_MODEL, decay=0.99, eps=1e-5):
        super().__init__()
        self.n_codes, self.dim, self.decay, self.eps = n_codes, dim, decay, eps
        self.embedding = nn.Embedding(n_codes, dim)
        self.embedding.weight.data.uniform_(-1.0 / n_codes, 1.0 / n_codes)
        self.register_buffer("ema", self.embedding.weight.data.clone())
        self.register_buffer("cluster_size", torch.zeros(n_codes))
        self.register_buffer("embed_avg", self.embedding.weight.data.clone())

    def lookup(self, idx):
        return self.embedding(idx)

    def forward(self, z):
        """z: (B, T, D) → (z_q, loss, idx)"""
        B, T, D = z.shape
        flat = z.reshape(-1, D)
        d = (flat.pow(2).sum(1, keepdim=True)
             - 2 * flat @ self.embedding.weight.t()
             + self.embedding.weight.pow(2).sum(1))
        idx = d.argmin(1)
        zq = self.embedding(idx).view(B, T, D)
        # EMA 更新
        with torch.no_grad():
            onehot = F.one_hot(idx, self.n_codes).type(flat.dtype)
            self.cluster_size.mul_(self.decay).add_(onehot.sum(0), alpha=1 - self.decay)
            self.embed_avg.mul_(self.decay).add_(onehot.t() @ flat, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cs = (self.cluster_size + self.eps) / (n + self.n_codes * self.eps) * n
            self.embedding.weight.data.copy_(self.embed_avg / cs.unsqueeze(1))
            self.ema.copy_(self.embedding.weight.data)
        loss = ((zq.detach() - z) ** 2).mean()          # commitment
        zq = z + (zq - z).detach()                     # 直线估计
        return zq, loss, idx.view(B, T)

    @torch.no_grad()
    def restart_dead(self, z, frac=0.05):
        """**死码重启**：把几乎没被用过的码重新初始化成当前批次的 z 样本。

        为什么必须做：EMA 更新里，cluster_size≈0 的码其 embed_avg/cs 会放大成噪声
        ⇒ 那些码再也不被选中 ⇒ **永久死亡** ⇒ 码本利用率一路掉到个位数
        （本地自测实测：初始化时 100%，4 步后 1.6%）。
        重启把它们拉回「当前数据里真实存在的向量」⇒ 利用率能维持。
        """
        flat = z.reshape(-1, self.dim)
        thresh = max(1.0, frac * self.cluster_size.sum() / self.n_codes)
        dead = (self.cluster_size < thresh).nonzero(as_tuple=True)[0]
        if dead.numel() == 0:
            return 0
        pick = torch.randint(0, flat.shape[0], (dead.numel(),), device=flat.device)
        self.embedding.weight.data[dead] = flat[pick]
        self.embed_avg[dead] = flat[pick]
        self.cluster_size[dead] = 1.0
        self.ema[dead] = flat[pick]
        return int(dead.numel())

    @torch.no_grad()
    def init_from_data(self, z):
        """用**第一批数据**初始化码本（VQ-VAE 经典做法）。

        为什么必须做：随机初始化的码本在第一步会被「最远的那几个码」吃掉
        ⇒ 立刻塌成 1~2 个码（本地自测实测利用率掉到 1.6%、熵≈0）
        ⇒ 后面 EMA 再怎么更新也救不回来（死码的 cluster_size 永远是 0）。
        """
        flat = z.reshape(-1, self.dim)
        n = flat.shape[0]
        if n < self.n_codes:
            flat = flat.repeat((self.n_codes // n) + 1, 1)[:self.n_codes]
        perm = torch.randperm(flat.shape[0], device=flat.device)[:self.n_codes]
        self.embedding.weight.data.copy_(flat[perm])
        self.embed_avg.copy_(self.embedding.weight.data)
        self.cluster_size.fill_(1.0)
        self.ema.copy_(self.embedding.weight.data)
        return self.embedding.weight.data.clone()

    @torch.no_grad()
    def usage(self, idx):
        cnt = torch.bincount(idx.reshape(-1), minlength=self.n_codes).float()
        used = int((cnt > 0).sum())
        p = cnt / cnt.sum().clamp(min=1)
        nz = p[p > 0]
        ent = float(-(nz * nz.log()).sum())
        return {"used": used, "total": self.n_codes,
                "utilization": used / self.n_codes,
                "entropy_nats": ent,
                "entropy_max": float(math.log(max(used, 2)))}


class MaskEmbed(nn.Module):
    """被遮时间片的占位 embedding（decoder 侧）。"""

    def __init__(self, dim=D_MODEL):
        super().__init__()
        self.emb = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.emb, std=0.02)

    def forward(self, x, mask):
        """x: (B,T,D)；mask: (B,T) bool，True = 被遮。"""
        return torch.where(mask.unsqueeze(-1), self.emb.expand_as(x), x)


class EEGTokenizer(nn.Module):
    def __init__(self, n_ch=63, n_times=250, n_codes=K_CODES, dim=D_MODEL):
        super().__init__()
        self.ch_mix = nn.Conv1d(n_ch, dim, 1)
        self.enc = nn.Sequential(
            nn.Conv1d(dim, dim, 5, padding=2), nn.GELU(), nn.GroupNorm(8, dim),
            nn.Conv1d(dim, dim, 5, padding=2), nn.GELU(), nn.GroupNorm(8, dim))
        self.t_down = nn.Conv1d(dim, dim, T_STRIDE, stride=T_STRIDE)
        self.vq = VQ(n_codes, dim)
        self.mask_emb = MaskEmbed(dim)
        self.dec_up = nn.ConvTranspose1d(dim, dim, T_STRIDE, stride=T_STRIDE)
        self.dec_out = nn.Conv1d(dim, n_ch, 1)
        self.n_times = n_times

    def encode(self, x):
        h = self.ch_mix(x)
        h = self.enc(h)
        h = self.t_down(h)              # (B, D, T')
        return h.transpose(1, 2)        # (B, T', D)

    def decode(self, zq, n_times):
        h = zq.transpose(1, 2)           # (B, D, T')
        h = self.dec_up(h)              # (B, D, T'*8)
        h = F.pad(h, (0, max(0, n_times - h.shape[-1])))[:, :, :n_times]
        return self.dec_out(h)          # (B, C, T)

    def forward(self, x, mask_ratio=0.5, generator=None):
        z = self.encode(x)              # (B, T', D)
        zq, vq_loss, idx = self.vq(z)
        # 掩码：随机选 mask_ratio 的时间片
        B, T, D = z.shape
        if mask_ratio > 0:
            noise = torch.rand(B, T, device=z.device, generator=generator)
            thresh = torch.quantile(noise, mask_ratio, dim=1, keepdim=True)
            mask = noise <= thresh
        else:
            mask = torch.zeros(B, T, dtype=torch.bool, device=z.device)
        zq_in = self.mask_emb(zq, mask)
        rec = self.decode(zq_in, x.shape[-1])
        return rec, vq_loss, idx, mask, z


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def train(args):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"=== S1：自训 EEG 离散 tokenizer ===")
    log(f"步数 {args.steps} | batch {args.batch} | 码本 K={args.codes} d={D_MODEL} "
        f"| 掩码率 {args.mask_ratio} | 训练被试 {args.subject}")

    if getattr(args, "self_test", False):
        log("  [自测] 用**有结构**的合成数据代替真实 train.pt")
        # 4 个固定模式 + 噪声 ⇒ 编码器应该学会区分它们
        #（纯随机噪声的「正确最优」就是常数 ⇒ 必然塌陷，自测没有信息量）
        pat = torch.randn(4, 63, 250) * 1.0
        eeg = pat[torch.randint(0, 4, (64, 4))] + torch.randn(64, 4, 63, 250) * 0.3
    else:
        obj, eeg = open_train(args.subject)
    model = EEGTokenizer(n_codes=args.codes)
    nparam = sum(p.numel() for p in model.parameters())
    log(f"模型参数量 {nparam:,}")
    opt = torch.optim.AdamW([
        {"params": list(model.ch_mix.parameters()) + list(model.enc.parameters())
         + list(model.t_down.parameters()) + list(model.dec_up.parameters())
         + list(model.dec_out.parameters()) + list(model.mask_emb.parameters()),
         "lr": args.lr},
    ], weight_decay=1e-4)

    t0 = time.time()
    hist = []
    for step in range(1, args.steps + 1):
        x = sample_batch(eeg, rng, args.batch)
        if step == 1:
            with torch.no_grad():
                z0 = model.encode(x)
                model.vq.init_from_data(z0)
            log(f"  码本已用第一批数据初始化（{args.codes} 个码）")
        rec, vq_loss, idx, mask, z = model(x, mask_ratio=args.mask_ratio)
        if step % args.restart_every == 0:
            with torch.no_grad():
                ndead = model.vq.restart_dead(z.detach())
            if ndead:
                log(f"  step {step:>5} | 死码重启 {ndead} 个")
        rec_loss = F.mse_loss(rec, x)
        loss = rec_loss + args.vq_weight * vq_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                u = model.vq.usage(idx)
                zstd = float(z.detach().reshape(-1, z.shape[-1]).std(0).mean())
                # ⚠️ **被遮位置的 MSE**才是与 loss 同口径的量：
                #    loss = 全波形 MSE，而被遮位置才是难的那部分。
                #    基线也要同口径（「时间轴均值」在被遮位置上的 MSE），
                #    否则「重建 MSE vs 基线」根本不能比（实测踩过：
                #    全波形基线 0.54 vs 含被遮位置的 loss 0.94 ⇒ 假象）。
                m = mask.unsqueeze(1)
                rec_mse_masked = float(((rec - x) ** 2)[m].mean())
                base_mse_masked = float(
                    ((x.mean(dim=2, keepdim=True).expand_as(x) - x) ** 2)[m].mean())
            rec_mse = float(rec_loss.item())
            base_mse = float(F.mse_loss(
                x, x.mean(dim=2, keepdim=True).expand_as(x)).item())
            hist.append({"step": step, "loss": float(loss.item()),
                         "rec_mse": rec_mse, "base_mse": base_mse,
                         "rec_mse_masked": rec_mse_masked,
                         "base_mse_masked": base_mse_masked,
                         "vq_loss": float(vq_loss.item()),
                         "utilization": u["utilization"],
                         "entropy": u["entropy_nats"], "z_std": zstd})
            log(f"  step {step:>5} | loss {float(loss.item()):.4f} "
                f"| 全波形 MSE {rec_mse:.4f}（基线 {base_mse:.4f}） "
                f"| **被遮位置 MSE {rec_mse_masked:.4f}"
                f"（基线 {base_mse_masked:.4f}）** "
                f"| vq {float(vq_loss.item()):.4f} | z_std {zstd:.4f} "
                f"| 码利用率 {u['utilization']:.1%} 熵 {u['entropy_nats']:.2f}"
                f" | {time.time()-t0:.0f}s")
        if args.max_seconds and (time.time() - t0) > args.max_seconds:
            log(f"  到达时间上限 {args.max_seconds}s，提前收尾")
            break

    # --- 保存 ---
    ckpt = {"state_dict": model.state_dict(), "args": vars(args),
            "n_codes": args.codes, "d_model": D_MODEL,
            "n_tokens": (250 + T_STRIDE - 1) // T_STRIDE,
            "hist": hist}
    fp = out_dir / f"tokenizer_sub-{args.subject:02d}.pt"
    torch.save(ckpt, fp)
    log(f"权重写入 {fp}（{fp.stat().st_size/1e6:.1f} MB）")
    (out_dir / f"train_log_sub-{args.subject:02d}.json").write_text(
        json.dumps({"args": vars(args), "hist": hist}, indent=2,
                   ensure_ascii=False, default=str), encoding="utf-8")
    log(f"总耗时 {time.time()-t0:.0f}s")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=1, help="用哪个被试的 train 集训练")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--codes", type=int, default=K_CODES)
    ap.add_argument("--mask-ratio", type=float, default=0.5)
    ap.add_argument("--vq-weight", type=float, default=0.25)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--restart-every", type=int, default=20,
                    help="每多少步做一次死码重启")
    ap.add_argument("--max-seconds", type=int, default=0, help="时间上限（秒），0=不限")
    ap.add_argument("--out-dir", default=str(OUT_ROOT))
    ap.add_argument("--self-test", action="store_true",
                    help="用合成数据把训练循环跑通（不碰网络）")
    args = ap.parse_args()
    try:
        raise SystemExit(train(args))
    except SystemExit:
        raise
    except Exception as e:
        log(f"✗ {type(e).__name__}: {e}")
        log(traceback.format_exc()[-1500:])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
