#!/usr/bin/env python3
"""exp87_shared_vocab.py — **S1b-v3：共享词典**（跨模态码预测）。

## 要解决什么

`FINDINGS_TIER0_RSA.md` §二十三·四 测到的事实：

| 臂 | 学出来的码本 | 随机码本 | Δr（学 − 随机） |
|---|---:|---:|---:|
| v2-none（嫁接注意力） | +0.0294 | +0.0495 | **−0.0180（1/10）** |
| v2-plain（不嫁接注意力） | +0.0221 | +0.0149 | +0.0068（8/10，p=0.055） |

⇒ **学出来的码本没有被语义组织**：语义在连续表征 `z` 里，离散化只是顺带把它带过去。
⇒ 根因（可定位到代码）：`exp80` 的 VQ 每次 forward 都做
`embedding.weight.data.copy_(embed_avg / cs)` ⇒ **码向量被 EMA 聚合覆盖**
⇒ 任何加在码本上的语义梯度都会被下一次 forward 抹掉
⇒ 码本**只服务重建目标**，永远学不到语义。

## 这一版改三件事

1. **码本吃语义梯度**（`VQShared`）：码本改由梯度更新（不再被 EMA 覆盖），
   跨模态码预测的交叉熵**直接作用在码向量上** ⇒ 码本的几何朝语义走。
2. **跨模态码预测**：每个 token 位置预测**图像侧的码**（K 路分类）
   ⇒ 梯度到码本：码向量被拉向「该图像簇的 z」；
   ⇒ 梯度到 encoder：z 被拉向「该图像的码」⇒ **token 身份变得可分图像**。
3. **逐 token**（不池化）：31 个位置全部参与 ⇒ 语义不会被位置平均冲掉。

## 三个臂（单因子分解）

| 臂 | 码本机制 | CE 的码本路径 | 回答什么 |
|---|---|---|---|
| `v3-plain-ema` | EMA（= v2 的机制） | 吃梯度（但被 EMA 抹掉） | 「加 CE 本身有没有用」 |
| `v3-plain-grad-nograd` | 梯度 | **不**吃梯度（logits 用 detach 过的码本） | 「是不是必须让码本吃语义梯度」 |
| `v3-plain-grad` | 梯度 | 吃梯度 | **候选配方** |

（基线是 `v2-plain`：EMA 码本 + 无 CE。）

## 硬门槛

**「学出来的码本」必须显著优于「随机码本」**（同 encoder、同维度、同池化，
与 §二十三·四 完全同一套读出与对照口径）。
本轮没过 ⇒ 「统一词表」还只是「离散分布恰好带了语义」。

## 用法

    # 本地自测（合成数据 + 合成图像码，不碰网络/桶）
    python exp87_shared_vocab.py --self-test --steps 200 --out-dir /tmp/exp87-self

    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp87_shared_vocab.py \
      -v hf://buckets/illusion-brain/training-data:/training:ro \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor cpu-upgrade --timeout 120m -d \
      --vq grad --sem-to-codebook 1 --image-codes /domains/runs/illusionbrain-align/shared-vocab/image_codes_k1024.pt \
      --out-dir /domains/runs/illusionbrain-align/exp87/grad
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

TRAIN_PT = Path("/training/eeg/things-eeg/Preprocessed_data_250Hz_whiten")
FEAT_TRAIN = "/training/eeg/eeg-image-decode/ViT-H-14_features_train.pt"
SEED = 0
K_CODES = 1024
D_MODEL = 64
T_STRIDE = 8
NHEAD = 4


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 共享词典的量化器
# ---------------------------------------------------------------------------
def build_vqshared(balance=False, eps_bal=0.05, sinkhorn_iters=3):
    """返回 `VQShared` 类（梯度版码本）。

    **关键差异（相对 exp80 的 VQ）**：**不再**用 EMA 覆盖 `embedding.weight`
    ⇒ 码向量能保住语义梯度。
    其余（键名、死码重启、lookup 接口）与 exp80 的 VQ **保持一致**
    ⇒ 同一个评测脚本能加载两种 checkpoint。

    `balance=True` ⇒ 用 **Sinkhorn 平衡分配**（切塌缩，见 VQShared.forward 的注释）。
    """
    import torch
    import torch.nn.functional as F
    from torch import nn

    class VQShared(nn.Module):
        """共享词典 VQ：码本由**梯度**组织（不是 EMA 聚合）。"""

        kind = "vq_shared"

        def __init__(self, n_codes=K_CODES, dim=D_MODEL, decay=0.99, eps=1e-5,
                     balance=False, eps_bal=0.05, sinkhorn_iters=3):
            super().__init__()
            self.n_codes, self.dim, self.decay, self.eps = n_codes, dim, decay, eps
            self.balance = balance          # 是否用 Sinkhorn 平衡分配
            self.eps_bal = eps_bal          # Sinkhorn 的 ε（越小越硬）
            self.sinkhorn_iters = sinkhorn_iters
            self.embedding = nn.Embedding(n_codes, dim)
            self.embedding.weight.data.normal_(0, 0.05)
            # ⚠️ 键名与 exp80 的 VQ 一致 ⇒ checkpoint 可互换加载
            self.register_buffer("ema", self.embedding.weight.data.clone())
            self.register_buffer("cluster_size", torch.zeros(n_codes))
            self.register_buffer("embed_avg", self.embedding.weight.data.clone())

        def lookup(self, idx):
            return self.embedding(idx)

        @torch.no_grad()
        def init_from_data(self, z):
            """同 exp80：用第一批数据初始化码本（防早期塌码）。"""
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

        def forward(self, z):
            """z: (B,T,D) → (z_q_straight_through, 码本损失, 承诺损失, idx)。

            **双路梯度**（经典 VQ-VAE）：
            - 直线估计 `z_q = z + (e − z).detach()` ⇒ 重建损失的梯度到 **encoder**
            - 码本损失 `||sg[e] − z||²` ⇒ 梯度到**码向量**（e 被拉向 z）
            - 承诺损失 `||e − sg[z]||²` ⇒ 梯度到 **z**（z 被拉向 e）

            ⚠️ **平衡分配（Sinkhorn）**：不加它会**塌缩**。
            实测（v3 四个臂）：码向量两两余弦 → **+0.90～+0.97**、有效秩 → **1.1/64**、
            语义 CE 卡在随机水平 ⇒ 跨模态码预测命中率 = 随机。
            机制：跨模态 CE 把 **z 拉向码**，而码本损失把 **码拉向 z** ⇒ 互相强化 ⇒ 一起塌。
            ⇒ Sinkhorn 让分配**均衡**（每个码拿到大致相同份额的 token）
            ⇒ 塌缩被切断（SwAV 的标准解法）。
            """
            B, T, D = z.shape
            flat = z.reshape(-1, D)
            d = (flat.pow(2).sum(1, keepdim=True)
                 - 2 * flat @ self.embedding.weight.t()
                 + self.embedding.weight.pow(2).sum(1))
            if self.balance:
                idx = self._sinkhorn_assign(d, flat)
            else:
                idx = d.argmin(1)
            zq_e = self.embedding(idx).view(B, T, D)
            l_cb = ((zq_e.detach() - z) ** 2).mean()        # 码本损失（梯度到 e）
            l_cm = ((zq_e - z.detach()) ** 2).mean()        # 承诺损失（梯度到 z）
            # 使用统计（只用于死码重启；**不改**码向量）
            with torch.no_grad():
                onehot = F.one_hot(idx, self.n_codes).type(flat.dtype)
                self.cluster_size.mul_(self.decay).add_(onehot.sum(0),
                                                        alpha=1 - self.decay)
                self.embed_avg.mul_(self.decay).add_(onehot.t() @ flat,
                                                     alpha=1 - self.decay)
            zq = z + (zq_e - z).detach()                   # 直线估计
            return zq, l_cb, l_cm, idx.view(B, T)

        @torch.no_grad()
        def _sinkhorn_assign(self, d, flat):
            """平衡分配（Sinkhorn）：让每个码拿到大致相同的份额。

            ⚠️ **affinity 必须用余弦相似度**（SwAV 标准做法）：
            平方距离做 affinity 时 `exp(-d/ε)` 会**下溢**（d~10、ε=0.05 ⇒ exp(-200)）
            ⇒ 只有最近的码活下来 ⇒ 分配反而更不均衡（实测：活跃码 = 1）。
            ⇒ 用有界的余弦，并先减最大值防溢出。
            """
            import torch
            zn = torch.nn.functional.normalize(flat, dim=-1)
            en = torch.nn.functional.normalize(self.embedding.weight, dim=-1)
            sim = zn @ en.t()                       # (N, K) ∈ [-1, 1]
            Q = torch.exp((sim - sim.max(dim=1, keepdim=True).values) / self.eps_bal)
            Q = Q / Q.sum().clamp(min=1e-12)
            n = Q.shape[0]
            for _ in range(self.sinkhorn_iters):
                Q = Q / Q.sum(dim=1, keepdim=True).clamp(min=1e-12) * n
                Q = Q / Q.sum(dim=0, keepdim=True).clamp(min=1e-12) * n
            return Q.argmax(dim=1)

        @torch.no_grad()
        def restart_dead(self, z, frac=0.05):
            """死码重启：把几乎没被用过的码重置成当前批次的 z 样本。"""
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
        def usage(self, idx):
            cnt = torch.bincount(idx.reshape(-1), minlength=self.n_codes).float()
            used = int((cnt > 0).sum())
            p = cnt / cnt.sum().clamp(min=1)
            nz = p[p > 0]
            ent = float(-(nz * torch.log(nz)).sum())
            return {"used": used, "total": self.n_codes,
                    "utilization": used / max(self.n_codes, 1),
                    "entropy_nats": ent}

    # 返回**工厂闭包**（把 balance 等设置带进去）⇒ `build_assembled` 只要一个
    # `(n_codes, dim) -> VQ` 的可调用对象。
    def factory(n_codes, dim):
        return VQShared(n_codes, dim, balance=balance, eps_bal=eps_bal,
                        sinkhorn_iters=sinkhorn_iters)

    factory.__name__ = f"VQShared(balance={balance})"
    return factory


# ---------------------------------------------------------------------------
# 语义损失
# ---------------------------------------------------------------------------
def code_logits(z, emb, tau=0.07, mode="none", target=None):
    """z: (B,T,D) / emb: (K,D) → (B,T,K) 余弦 logits（τ 缩放）。

    `mode` 决定**码本吃多少语义梯度**（这是 v3 的核心变量）：

    | mode | 码本的梯度来源 |
    |---|---|
    | `"none"` | 全部 detach ⇒ 码本只被 `l_cb`（拉向 z）更新 |
    | `"full"` | 全部可导 ⇒ 负类也给码本压力（理论上会把非目标码向量推向同一方向） |
    | `"positive"` | **只有正类（目标码）的码向量可导** |

    ⚠️ 本地**合成**自测里两个模式的「余弦相似度高、有效秩低」**不能**当作塌缩的证据：
    合成数据只有 8 张图 ⇒ 每个图一个码 ⇒ 逐批利用率本来就该是 ~0.8%；
    且死码是 `restart_dead` 灌进去的 z 副本 ⇒ 彼此高度相似 ⇒ 把死码算进去指标必假。
    ⇒ **几何指标只对「活跃码」算**（见 `codebook_geom`），真结论看实数据。
    """
    import torch.nn.functional as F
    zn = F.normalize(z, dim=-1)
    en = F.normalize(emb, dim=-1)
    if mode == "none":
        en_use = en.detach()
    elif mode == "full":
        en_use = en
    elif mode == "positive":
        if target is None:
            raise ValueError("mode='positive' 需要 target")
        B, T = z.shape[0], z.shape[1]
        # 目标码的相似度：可导（梯度只落到 en[target] 那几行）
        sim_pos = (zn * en[target].unsqueeze(1)).sum(-1)          # (B,T)
        # 其余码：detach（不给码本负类压力）
        sim_all = zn @ en.detach().t()                            # (B,T,K)
        # ⚠️ **沿 K 维 scatter**（dim=2）—— 沿 T 维 scatter 会把「码号」当成「位置」下标
        #    ⇒ 码号 ≥ T 时直接 RuntimeError；码号 < T 时**不报错但结果全错**
        #    （本地自测就漏在这里：合成数据只有 8 个码，恰好都 < T=31）
        idx_t = target.view(B, 1, 1).expand(B, T, 1)              # (B,T,1)
        logits = sim_all.scatter(2, idx_t, sim_pos.unsqueeze(-1))  # (B,T,K)
        return logits / tau
    else:
        raise ValueError(f"未知 mode={mode}")
    return (zn @ en_use.t()) / tau

def eval_crossmodal(model, codes_centroids, eeg_te, img_te, device="cpu"):
    """**跨模态码预测**：EEG 的 token 能不能直接报出图像侧的码。

    这是「统一词表」的**直接**判据（比 RSA 读出更贴定义）：
    同一本词典 ⇒ 同一个 token 在两个模态里表示同一个概念
    ⇒ 拿 EEG 的 token 去撞图像的码，命中率显著高于随机（1/K）就是「共用词典」。

    返回三个数：
    - `pos_acc`：逐位置命中率（31 个 token 各自报一个码）
    - `any_acc`：整幅至少一个位置命中（更宽松 ⇒ 只要语义进了码本就算）
    - `top1_hit`：按位置投票（多数票）出的码 == 图像的码
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    K = codes_centroids.shape[0]
    # 图像侧：test 图 → 最近质心（= 它的码）
    with torch.no_grad():
        sim = torch.from_numpy(img_te) @ torch.from_numpy(codes_centroids).T
        img_code = sim.argmax(dim=1).numpy()
    n_img, n_rep = eeg_te.shape[0], eeg_te.shape[1]
    flat = eeg_te.reshape(n_img * n_rep, eeg_te.shape[2], eeg_te.shape[3])
    tgt = np.repeat(img_code, n_rep)
    hits, anys, votes = 0, 0, 0
    bs = 64
    with torch.no_grad():
        for s in range(0, flat.shape[0], bs):
            e = min(s + bs, flat.shape[0])
            x = torch.from_numpy(np.ascontiguousarray(flat[s:e])).float().to(device)
            z = model.encode(x)
            out = model.vq(z)
            idx = out[3] if len(out) == 4 else out[2]   # VQShared 返回 4 元组，exp80 的 VQ 返回 3 元组
            idx = idx.reshape(e - s, -1).cpu().numpy()
            for r in range(idx.shape[0]):
                row = idx[r]
                t = int(tgt[s + r])
                nhit = int((row == t).sum())
                hits += nhit
                anys += 1 if nhit > 0 else 0
                # 多数票
                vals, cnt = np.unique(row, return_counts=True)
                if int(vals[cnt.argmax()]) == t:
                    votes += 1
    model.train()
    total_pos = flat.shape[0] * idx.shape[1]
    return {"pos_acc": hits / max(total_pos, 1),
            "any_acc": anys / max(flat.shape[0], 1),
            "top1_hit": votes / max(flat.shape[0], 1),
            "n_test": int(n_img),
            "random_pos_acc": 1.0 / K}


def self_test_logits():
    """`code_logits` 的**单元断言**：目标位置上的 logit 必须等于
    「z 与 e[target] 的余弦」/τ。

    为什么必须有这条：这一版曾把 scatter 沿**错的一维**做
    （沿 T 维 ⇒ 把「码号」当「位置」下标）
    ⇒ 码号 ≥ T 时 RuntimeError，码号 < T 时**不报错但结果全错**
    ⇒ 而本地合成自测只有 8 个码（0–7），恰好都 < T=31 ⇒ **自测没抓到**。
    ⇒ 这条断言直接盯住「目标位置的值」，任何 scatter 维度错误都会立刻暴露。
    """
    import torch
    g = torch.Generator().manual_seed(0)
    B, T, D, K = 4, 7, 16, 32
    z = torch.randn(B, T, D, generator=g)
    emb = torch.randn(K, D, generator=g)
    target = torch.tensor([0, 5, 31, 7])          # 故意含 ≥ T 的码号（31）
    tau = 0.07
    for mode in ("none", "full", "positive"):
        lg = code_logits(z, emb, tau=tau, mode=mode, target=target)
        assert lg.shape == (B, T, K), (mode, lg.shape)
        for b in range(B):
            for t in range(T):
                want = (torch.nn.functional.normalize(z[b, t], dim=-1)
                        * torch.nn.functional.normalize(emb[target[b]], dim=-1)
                        ).sum() / tau
                got = lg[b, t, target[b]]
                assert abs(float(got) - float(want)) < 1e-4, (
                    mode, b, t, float(got), float(want))
    log("  [自测] code_logits：三种 mode 的目标位置取值都等于直接余弦 ✓")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vq", default="grad", choices=["grad", "ema"],
                    help="码本机制：grad=梯度（吃语义梯度）；ema=exp80 的聚合机制")
    ap.add_argument("--sem-codebook-mode", default="positive",
                    choices=["none", "full", "positive"],
                    help="跨模态 CE 对**码本**的梯度路径："
                         "none=全 detach（码本不吃语义梯度）；"
                         "full=全可导（⚠️ 会塌）；positive=只有正类可导（修法）")
    ap.add_argument("--image-codes", default="",
                    help="图像侧离散码（exp87b 产出）；空 ⇒ 用随机码（仅自测）")
    ap.add_argument("--attn-donor", default="none")
    ap.add_argument("--donor-ffn", default="none")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--codes", type=int, default=K_CODES)
    ap.add_argument("--lambda-nce", type=float, default=0.5)
    ap.add_argument("--lambda-sem", type=float, default=1.0)
    ap.add_argument("--lambda-vq", type=float, default=0.25)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--balance", default="none", choices=["none", "sinkhorn"],
                    help="量化分配是否用 **Sinkhorn 平衡**（切塌缩；实测不加会塌）")
    ap.add_argument("--eps-bal", type=float, default=0.05,
                    help="Sinkhorn 的 ε（越小越硬；SwAV 用 0.05→0.01 退火）")
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--max-seconds", type=int, default=0)
    ap.add_argument("--out-dir", default="/domains/runs/illusionbrain-align/exp87/grad")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--eval-test", type=int, default=1,
                    help="训练完在 test split 上测**跨模态码预测**（统一词表的直接判据）")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    from exp86_assembled_train import build_assembled, info_nce
    from exp80_eeg_tokenizer_train import VQ

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.self_test:
        self_test_logits()

    vq_factory = (build_vqshared(balance=(args.balance == "sinkhorn"),
                                 eps_bal=args.eps_bal)
                  if args.vq == "grad" else VQ)
    log(f"=== S1b-v3 共享词典 | 码本机制 {args.vq} | "
        f"CE→码本 {args.sem_codebook_mode} | "
        f"注意力 {args.attn_donor} | FFN {args.donor_ffn} ===")

    # --- 数据 ---
    if args.self_test:
        g = torch.Generator().manual_seed(1234)
        pats = torch.randn(8, 63, 250, generator=g) * 1.0
        eeg = (pats[:, None].expand(8, 8, 63, 250)
               + torch.randn(8, 8, 63, 250, generator=g) * 0.3).numpy().astype(np.float16)
        IMG = (torch.randn(8, 1024, generator=g)
               + torch.randn(8, 1024, generator=g) * 0.1).numpy().astype(np.float32)
        # 合成图像码：8 张图 ⇒ 8 个簇
        img_codes = np.arange(8, dtype=np.int64)
        args.batch = min(args.batch, 8)
        log(f"  [自测] 合成 EEG {tuple(eeg.shape)} / 图像特征 {IMG.shape} / "
            f"图像码 {img_codes.shape}")
    else:
        fp = TRAIN_PT / f"sub-{args.subject:02d}" / "train.pt"
        try:
            obj = torch.load(fp, map_location="cpu", mmap=True, weights_only=False)
        except TypeError:
            obj = torch.load(fp, map_location="cpu", weights_only=False)
        eeg = obj["eeg"]
        log(f"  train split EEG {tuple(eeg.shape)}")
        ftr = torch.load(FEAT_TRAIN, map_location="cpu", weights_only=False)
        IMG = ftr["img_features"].numpy().astype(np.float32)
        log(f"  图像特征 {IMG.shape}")
        if not args.image_codes:
            raise SystemExit("✗ 需要 --image-codes（先跑 exp87b_image_codes.py）")
        cd = torch.load(args.image_codes, map_location="cpu", weights_only=False)
        img_codes = cd["codes"].astype(np.int64)
        log(f"  图像侧码本：K={cd['n_codes']}，码号 {img_codes.shape}")

    n_img, n_rep = eeg.shape[0], eeg.shape[1]

    model = build_assembled(n_codes=args.codes, attn_state=None,
                            ffn_donor=None, heads="v2", vq_factory=vq_factory)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  可训练参数 {n_par:,}")

    body_p, vq_p, head_p = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("vq."):
            vq_p.append(p)
        elif n.startswith("head_"):
            head_p.append(p)
        else:
            body_p.append(p)
    opt = torch.optim.AdamW([
        {"params": body_p, "lr": args.lr},
        {"params": vq_p, "lr": args.lr},
        {"params": head_p, "lr": args.lr},
    ], weight_decay=1e-4)
    log(f"  优化器分组：主体 {len(body_p)} / 码本 {len(vq_p)} / 语义头 {len(head_p)}")

    # 码本的几何摘要（训练后要回答「码本有没有被语义组织」）
    def codebook_geom():
        """几何摘要。

        ⚠️ **只看「活跃码」**：死码是 `restart_dead` 灌进去的当前批次 z 的副本
        ⇒ 在「每个图一个码」的配方里它们彼此高度相似 ⇒ 把死码算进去会让
        余弦相似度虚高、有效秩虚低（本地合成自测踩到过这个坑）。
        ⇒ 同时给出全局口径以便对照。
        """
        E = model.vq.embedding.weight.detach().float().numpy()
        K, D = E.shape
        with torch.no_grad():
            cnt = torch.bincount(codes.reshape(-1),
                                  minlength=model.vq.n_codes).float()
        active = (cnt > 0).numpy()
        n_act = int(active.sum())

        def g_of(M):
            if M.shape[0] < 2:
                return {"cos_mean": float("nan"), "eff_rank": 0.0}
            nrm = np.linalg.norm(M, axis=1)
            Mn = M / np.maximum(nrm[:, None], 1e-12)
            Gm = Mn @ Mn.T
            iu = np.triu_indices(M.shape[0], 1)
            lam = np.linalg.svd(M, compute_uv=False) ** 2
            return {"cos_mean": float(Gm[iu].mean()),
                    "eff_rank": float(lam.sum() ** 2 / max((lam ** 2).sum(), 1e-12))}

        ga = g_of(E[active])
        gg = g_of(E)
        nrm = np.linalg.norm(E, axis=1)
        return {"active_codes": n_act,
                "cos_mean_active": ga["cos_mean"],
                "eff_rank_active": ga["eff_rank"],
                "cos_mean_all": gg["cos_mean"],
                "eff_rank_all": gg["eff_rank"],
                "norm_mean": float(nrm.mean()),
                "var_max_over_mean": float(E.var(axis=0).max()
                                           / max(E.var(axis=0).mean(), 1e-12))}

    t0 = time.time()
    hist = []
    for step in range(1, args.steps + 1):
        if args.self_test:
            im_idx = rng.permutation(n_img)[:args.batch]
        else:
            im_idx = rng.integers(0, n_img, size=args.batch)
        rp_idx = rng.integers(0, n_rep, size=args.batch)
        rows = [np.asarray(eeg[int(i), int(r)]) for i, r in zip(im_idx, rp_idx)]
        x = torch.from_numpy(np.stack(rows).astype(np.float32))
        img = torch.from_numpy(IMG[im_idx])
        tgt_code = torch.from_numpy(img_codes[im_idx])          # (B,)

        if step == 1:
            with torch.no_grad():
                model.vq.init_from_data(model.encode(x))
            log("  码本已用第一批数据初始化")

        z = model.encode(x)
        if args.vq == "grad":
            zq, l_cb, l_cm, codes = model.vq(z)
        else:
            zq, l_vq, codes = model.vq(z)
            l_cb = torch.zeros((), device=z.device)
            l_cm = torch.zeros((), device=z.device)
        rec = model.decode(zq, x.shape[-1])
        l_rec = F.mse_loss(rec, x)

        # --- 跨模态码预测（逐 token）---
        logits = code_logits(z, model.vq.embedding.weight, tau=args.tau,
                             mode=args.sem_codebook_mode, target=tgt_code)
        l_sem = F.cross_entropy(logits.reshape(-1, args.codes),
                                tgt_code.repeat_interleave(z.shape[1]))

        # --- 语义对比（挂在重建信号上，与 v2 同一口径）---
        if args.lambda_nce > 0:
            rec_pooled = rec.mean(dim=1)
            nce_r, acc_r = info_nce(model.head_r(rec_pooled), model.head_img(img))
        else:
            nce_r = torch.zeros((), device=rec.device)
            acc_r = float("nan")

        l_vq = l_cb + l_cm
        loss = (l_rec + args.lambda_nce * nce_r + args.lambda_sem * l_sem
                + args.lambda_vq * l_vq)
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
            g = codebook_geom()
            hist.append({"step": step, "loss": float(loss.item()),
                         "rec_mse": float(l_rec.item()),
                         "nce_r": float(nce_r.item()),
                         "sem_ce": float(l_sem.item()),
                         "l_cb": float(l_cb.item()), "l_cm": float(l_cm.item()),
                         "retrieval_acc": acc_r,
                         "utilization": u["utilization"],
                         "entropy": u["entropy_nats"], **g})
            log(f"  step {step:>5} | loss {float(loss.item()):.4f} "
                f"| 重建 {float(l_rec.item()):.4f} "
                f"| nce_r {float(nce_r.item()):.3f} "
                f"| 语义CE {float(l_sem.item()):.4f}（随机 {math.log(args.codes):.3f}）"
                f"| 码本 cos(活跃) {g['cos_mean_active']:+.3f} "
                f"有效秩 {g['eff_rank_active']:.1f}/{D_MODEL}"
                f" | 活跃码 {g['active_codes']}"
                f" | {time.time()-t0:.0f}s")
        if args.max_seconds and (time.time() - t0) > args.max_seconds:
            log(f"  到达时间上限 {args.max_seconds}s，提前收尾")
            break

    ck = {"state_dict": model.state_dict(), "args": vars(args),
          "n_codes": args.codes, "d_model": D_MODEL, "arch": "assembled_v3",
          "vq": args.vq, "sem_codebook_mode": args.sem_codebook_mode,
          "donor_ffn": args.donor_ffn, "hist": hist}
    out = out_dir / "tokenizer_assembled.pt"
    torch.save(ck, out)
    log(f"权重写入 {out}（{out.stat().st_size/1e6:.1f} MB）")

    # --- 跨模态码预测（test split）⇒「统一词表」的直接判据 ---
    if not args.self_test and args.eval_test:
        try:
            import torch as _t
            cd = _t.load(args.image_codes, map_location="cpu", weights_only=False)
            cent = cd["centroids"].astype(np.float32)
            eeg_te = np.load("/domains/omnibrain4d/eeg/"
                             f"sub-{args.subject:02d}.npz")["eeg"]
            fte = _t.load("/training/eeg/eeg-image-decode/"
                          "ViT-H-14_features_test.pt",
                          map_location="cpu", weights_only=False)
            IMG_te = fte["img_features"].numpy().astype(np.float32)
            # 图像侧的质心是归一化特征上的 ⇒ 归一化后再算最近质心
            c = cent / np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-12)
            q = IMG_te / np.maximum(np.linalg.norm(IMG_te, axis=1, keepdims=True), 1e-12)
            st = eval_crossmodal(model, c.astype(np.float32), eeg_te, q.astype(np.float32))
            log(f"\n=== 跨模态码预测（test split {st['n_test']} 图）===")
            log(f"  逐位置命中 {st['pos_acc']:.4f}（随机 {st['random_pos_acc']:.5f}）")
            log(f"  整幅至少一位置命中 {st['any_acc']:.4f}")
            log(f"  多数票命中 {st['top1_hit']:.4f}")
            ck["crossmodal"] = st
            _t.save(ck, out)
        except Exception as e:
            log(f"  ⚠ 跨模态评测没跑成：{type(e).__name__}: {e}")

    (out_dir / "train_log.json").write_text(
        json.dumps({"args": vars(args), "hist": hist,
                    "crossmodal": ck.get("crossmodal")}, indent=2,
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
