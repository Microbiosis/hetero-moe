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

⇒ 第 2 点要靠**三层语义监督**（关键细节，写错就测不到）：

| 损失 | 挂在哪 | 梯度到哪 | 作用 |
|---|---|---|---|
| `L_nce_r` | **重建信号** `rec.mean(1)` | encoder（经 straight-through） | ★ 语义损失的主路：重建信号是 S1a 唯一带语义的读出 |
| `L_nce_z` | 量化**前**的 `z.mean(1)` | encoder | 让连续表征本身可分图像 |
| `L_nce_q` | **码本查表结果** `E[codes].mean(1)` | **码本本身** | 让码本的几何朝语义走 |

⚠️ 三条路**必须是三个独立的头**（`head_r`/`head_z`/`head_q`）：
三条路径的张量空间不同（重建 250 维 / 量化前 z 64 维 / 码本查表 64 维），
共用一个头会让梯度互相打架。
⚠️ 若只把语义损失挂在 straight-through 的量化向量上，**梯度会被 detach 挡住**
⇒ 码本学不到语义 ⇒ 实验会假阴性。
⇒ 诚实判据因此是**纯离散读出**（`onehot_mean` / `code_ids`）：
若它们也变显著 ⇒ **token 身份**本身带语义（而不只是码本向量带）。

⚠️ **第一版（v1）把语义损失挂在池化 token 上 ⇒ 完全学不动**：
S1a 实测池化 token 不带语义（`emb_mean`/`continuous` 都 0/10），
于是 NCE 卡在随机水平（ln64=4.159）、检索准确率 0.000–0.062。
⇒ v2 改挂在重建信号上（= exp81 里那个 +0.0372 的读出，**口径逐字一致**）。

⚠️ **语义头必须全部进优化器**：PyTorch 只更新传进 `optimizer` 的参数
⇒ 漏掉任何一个头 ⇒ 那条分支学不动 ⇒ 假阴性。
⇒ 本脚本按**构造**分组（每个 `requires_grad` 参数必落入恰好一组），
并在自测里断言「每个头的权重真的变了」。

## 读出与监控

- 重建 MSE、in-batch 图像检索准确率（64 选 1）、码本利用率与熵。

## 用法

    # 本地自测（合成数据 + 合成组件，不碰网络与桶）
    python exp86_assembled_train.py --self-test --steps 400 --out-dir /tmp/exp86-self

    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp86_assembled_train.py \
      -v hf://buckets/illusion-brain/training-data:/training:ro \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor cpu-upgrade --timeout 180m -d \
      --donor-ffn none --steps 6000 --out-dir /domains/runs/illusionbrain-align/exp86/ffn-none

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--donor-ffn` | none | 嫁接哪个供体的 FFN（none/mirepnet/biot/eegdino-small） |
| `--attn-donor` | signal-jepa | 嫁接哪个供体的注意力（唯一逐键同形的：MHA 64） |
| `--lambda-nce` | 0.5 | 语义损失总权重；**置 0 即「无语义监督」对照组** |
| `--codebook-init` | data | 码本初始化：`data`=用第一批数据初始化（exp80 纪律）；`random`=随机 |
| `--self-test` | — | 合成数据 + 合成组件跑通训练循环并自检（不碰网络/桶） |
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

# 组装模型只依赖 exp80 的 VQ 类 + 桶里的供体组件 .pt
# ⇒ **不需要** 基模注册表（base_specs / domain_adapters）：
#   那些 import 是从 exp80 抄来的历史遗留，从未在本脚本里被用到，
#   却会让脚本在找不到 models/ 时直接 ImportError（无谓的失败面）。

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
                    attn_state=None, ffn_donor=None, heads="v2", vq_factory=None):
    """组装模型工厂（返回 nn.Module 实例）。

    骨架与我们自研 tokenizer 一致（CNN 编码 → 时间片 token → VQ → 解码），
    中间插一个**组装块**：
      - 注意力：从 donor **逐键同形**地搬进来（signal-jepa encoder 第 0 层）
      - FFN：维度不同 ⇒ 两侧学习投影 + donor 的 FFN 权重（可训练）

    `heads`：
      - `"v1"`：语义头只有 head_z（量化前 z 与码本查表**共用**）⇒
        第一版（语义损失挂在池化 token 上 ⇒ 学不动，见 v2 的注释）
      - `"v2"`：head_r / head_z / head_q / head_img 四个独立头
        （语义损失挂在**重建信号**上）
    ⇒ 两版的 checkpoint 能被同一个评测脚本按元信息加载。

    `vq_factory`：可选的量化器工厂 `(n_codes, dim) -> nn.Module`。
      默认用 exp80 的 `VQ`（EMA 码本）；
      **S1b-v3（共享词典）**用它换掉 EMA 码本 ⇒ 码本吃语义梯度。
      ⚠️ 默认行为不变 ⇒ 已跑完的 v1/v2 臂不受影响。
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
            self.vq = vq_factory(n_codes, dim) if vq_factory else VQ(n_codes, dim)
            # --- decoder ---
            self.dec_up = nn.ConvTranspose1d(dim, dim, T_STRIDE, stride=T_STRIDE)
            self.dec_out = nn.Conv1d(dim, n_ch, 1)
            # --- 语义头（只用于训练时的对比损失）---
            # ⚠️ **三个头分开**（v2）：三条路径的张量空间不同
            #    （重建 250 维 / 量化前 z 64 维 / 码本查表 64 维），
            #    共用一个头会让梯度互相打架。
            if heads == "v2":
                self.head_r = nn.Linear(n_times, 256)   # 重建信号（信息在这儿）
                self.head_z = nn.Linear(dim, 256)       # 量化前 z
                self.head_q = nn.Linear(dim, 256)       # 码本查表结果
                self.head_img = nn.Linear(1024, 256)    # 图像特征
            else:
                # v1：量化前 z 与码本查表共用 head_z
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
    ap.add_argument("--attn-donor", default=ATTN_DONOR,
                    choices=[ATTN_DONOR, "none"],
                    help="嫁接哪个供体的注意力；'none' = **不嫁接注意力**"
                         "（用于「嫁接注意力到底有没有增益」的同配方对照）")
    ap.add_argument("--freeze-attn", action="store_true",
                    help="嫁接来的注意力冻结（只当固定组件用）")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--codes", type=int, default=K_CODES)
    ap.add_argument("--lambda-nce", type=float, default=0.5,
                    help="语义损失总权重；**置 0 = 无语义监督对照组**")
    ap.add_argument("--lambda-vq", type=float, default=0.25)
    ap.add_argument("--codebook-init", default="data", choices=["data", "random"],
                    help="码本初始化：data=用第一批数据初始化（exp80 纪律，"
                         "否则第一步就塌成 1~2 个码）；random=随机初始化")
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--max-seconds", type=int, default=0)
    ap.add_argument("--out-dir", default="/domains/runs/illusionbrain-align/exp86/assembled")
    ap.add_argument("--self-test", action="store_true",
                    help="合成数据 + 合成组件跑通训练循环并自检（不碰网络/桶）")
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
    if args.self_test:
        # 合成但**有结构**的数据：8 个固定模式，各带一个固定图像特征向量
        # ⇒ 语义监督若接线正确，NCE 必须能从 ln(8)=2.079 降下来
        # （纯噪声没有这个可学结构 ⇒ 测不出接线对错）
        g = torch.Generator().manual_seed(1234)
        pats = torch.randn(8, 63, 250, generator=g) * 1.0
        eeg = (pats[:, None].expand(8, 8, 63, 250)
               + torch.randn(8, 8, 63, 250, generator=g) * 0.3).numpy().astype(np.float16)
        IMG = (torch.randn(8, 1024, generator=g)
               + torch.randn(8, 1024, generator=g) * 0.1).numpy().astype(np.float32)
        log(f"  [自测] 合成 EEG {tuple(eeg.shape)} / 图像特征 {IMG.shape}")
    else:
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
    if args.self_test:
        # 自测批内不重复 ⇒ 批大小不能超过图像数（否则随机水平口径会算错）
        args.batch = min(args.batch, n_img)
        log(f"  [自测] 批大小 {args.batch}（随机水平 ln{args.batch}="
            f"{math.log(args.batch):.3f}）")

    # --- 组装（组件从桶里读 ⇒ 不需要 braindecode）---
    if args.self_test:
        # 合成供体：同形的 MHA(64) 与一个 64→128→64 的 FFN
        gg = torch.Generator().manual_seed(7)
        mha = torch.nn.MultiheadAttention(D_MODEL, NHEAD, batch_first=True)
        with torch.no_grad():
            for p_ in mha.parameters():
                p_.copy_(torch.randn(p_.shape, generator=gg) * 0.05)
        attn_state = {k: v.clone() for k, v in mha.state_dict().items()}
        ffn_donor = {"h": 128, "d": D_MODEL,
                     "w1": torch.randn(128, D_MODEL, generator=gg) * 0.05,
                     "b1": torch.zeros(128),
                     "w2": torch.randn(D_MODEL, 128, generator=gg) * 0.05,
                     "b2": torch.zeros(D_MODEL),
                     "src": "synthetic"}
        log("  [自测] 合成组件：MHA(64) + FFN(64→128→64)")
    else:
        attn_state = None
        if args.attn_donor != "none":
            attn_state = extract_attn_state(args.attn_donor)
            log(f"  嫁接注意力 {args.attn_donor}：键 {sorted(attn_state)}")
        else:
            log("  **不嫁接注意力**（同配方的「无嫁接」对照臂）")
        ffn_donor = None
        if args.donor_ffn != "none":
            ffn_donor = extract_ffn_donor(args.donor_ffn)
            log(f"  嫁接 FFN {ffn_donor['src']}：{ffn_donor['d']}→"
                f"{ffn_donor['h']}→{ffn_donor['d']}（两侧学习投影 "
                f"{D_MODEL}↔{ffn_donor['d']}）")

    model = build_assembled(n_codes=args.codes, attn_state=attn_state,
                            ffn_donor=ffn_donor, heads="v2")
    if args.freeze_attn:
        for p in model.attn.parameters():
            p.requires_grad_(False)
        log("  嫁接的注意力**冻结**")
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  可训练参数 {n_par:,}")

    # ⚠️ **参数分组按构造覆盖每一个可训练参数**（PyTorch 只更新传进 optimizer 的参数
    #    ⇒ 漏掉一个头 ⇒ 那条语义分支永远学不动 ⇒ 实验假阴性）。
    #    第一版就是漏了 head_r / head_q（只收了 head_z + head_img）。
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
    groups = [
        {"params": body_p, "lr": args.lr},
        {"params": vq_p, "lr": args.lr},
        {"params": head_p, "lr": args.lr},
    ]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    log(f"  优化器分组：主体 {len(body_p)} / 码本 {len(vq_p)} / 语义头 {len(head_p)} 个张量")
    # 自检：所有 requires_grad 参数必须恰好落在某一组里
    _in_opt = {id(p) for g in groups for p in g["params"]}
    _missing = [n for n, p in model.named_parameters()
                if p.requires_grad and id(p) not in _in_opt]
    if _missing:
        raise RuntimeError(f"这些参数没进优化器 ⇒ 永不更新：{_missing}")

    # 记录初始权重（自测用：验证每个头真的被更新了）
    w0 = {n: p.detach().clone() for n, p in model.named_parameters()
          if n.startswith("head_")}

    t0 = time.time()
    hist = []
    for step in range(1, args.steps + 1):
        # 采样：一个批次 = B 个（图像, 重复）试次，并取对应图像特征
        if args.self_test:
            im_idx = rng.permutation(n_img)[:args.batch]   # 自测：批内不重复
        else:
            im_idx = rng.integers(0, n_img, size=args.batch)
        rp_idx = rng.integers(0, n_rep, size=args.batch)
        rows = [np.asarray(eeg[int(i), int(r)]) for i, r in zip(im_idx, rp_idx)]
        x = torch.from_numpy(np.stack(rows).astype(np.float32))
        img = torch.from_numpy(IMG[im_idx])

        if step == 1 and args.codebook_init == "data":
            # 码本用**第一批数据**初始化（exp80 纪律）：
            # 随机初始化的码本第一步就被「最远的几个码」吃掉 ⇒ 塌成 1~2 个码
            # ⇒ 后面 EMA 再怎么更新也救不回来（死码 cluster_size 永远是 0）。
            with torch.no_grad():
                model.vq.init_from_data(model.encode(x))
            log("  码本已用第一批数据初始化")

        rec, vq_loss, codes, z, zq = model(x)
        l_rec = F.mse_loss(rec, x)
        # --- 语义损失：挂在**重建信号**上（关键！）---
        # ⚠️ S1a 实测：**池化 token 向量不带语义**（continuous 0/10、emb_mean 0/10），
        #    带语义的是**池化重建信号**（recon +0.0372）⇒ 语义损失必须挂在这儿，
        #    否则模型学不动（第一版挂在池化 token 上 ⇒ NCE 卡在随机水平 ln64=4.159、
        #    检索准确率 0.000~0.031）。
        #    梯度经 straight-through 回到 encoder ⇒ **token 分配**变得可分图像
        #    ⇒ 这才是「语义进离散 token」的路径。
        if args.lambda_nce > 0:
            rec_pooled = rec.mean(dim=1)              # (B, T) 沿通道轴均值 = 与 exp81 的 recon 读出逐字同口径
            nce_r, acc_r = info_nce(model.head_r(rec_pooled), model.head_img(img))
            nce_z, acc_z = info_nce(model.head_z(z.mean(1)), model.head_img(img))
            # 码本查表结果（梯度直达码本 ⇒ 不被 straight-through 的 detach 挡住）
            emb = model.vq.lookup(codes)
            nce_q, acc_q = info_nce(model.head_q(emb.mean(1)), model.head_img(img))
            loss = (l_rec + args.lambda_nce * (nce_r + nce_z + nce_q)
                    + args.lambda_vq * vq_loss)
        else:
            # 无语义监督对照组：不计算 NCE（省算力，也避免 head 被无意义的梯度带动）
            nce_r = nce_z = nce_q = torch.zeros((), device=rec.device)
            acc_r = acc_z = acc_q = float("nan")
            loss = l_rec + args.lambda_vq * vq_loss
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
                         "nce_r": float(nce_r.item()), "nce_z": float(nce_z.item()),
                         "nce_q": float(nce_q.item()),
                         "retrieval_acc": acc_r, "retrieval_acc_q": acc_q,
                         "utilization": u["utilization"],
                         "entropy": u["entropy_nats"]})
            log(f"  step {step:>5} | loss {float(loss.item()):.4f} "
                f"| 重建 {float(l_rec.item()):.4f} "
                f"| nce_r {float(nce_r.item()):.3f} "
                f"nce_z {float(nce_z.item()):.3f} nce_q {float(nce_q.item()):.3f} "
                f"| 检索准确率(重建) {acc_r:.3f} (码本) {acc_q:.3f} "
                f"| 码利用率 {u['utilization']:.1%} 熵 {u['entropy_nats']:.2f} "
                f"| {time.time()-t0:.0f}s")
        if args.max_seconds and (time.time() - t0) > args.max_seconds:
            log(f"  到达时间上限 {args.max_seconds}s，提前收尾")
            break

    # --- 自测断言（只在 --self-test 下生效）---
    if args.self_test:
        if args.lambda_nce > 0:
            # 自检 1：每个语义头的权重真的变了（抓「没进优化器」这类 bug）
            dead = []
            for n, p in model.named_parameters():
                if not n.startswith("head_"):
                    continue
                if float((p.detach() - w0[n]).abs().max()) == 0.0:
                    dead.append(n)
            if dead:
                raise RuntimeError(
                    f"这些语义头从头到尾没变过 ⇒ 没被更新：{dead}")
            log("  [自测] 所有语义头的权重都发生了变化 ✓")
            # 自检 2：语义损失必须真的在学（接线对错的可判据）
            last = hist[-1]
            rnd = math.log(args.batch)
            for k in ("nce_r", "nce_z", "nce_q"):
                if not (last[k] < rnd - 0.05):
                    raise RuntimeError(
                        f"{k}={last[k]:.4f} 没有低于随机水平 ln({args.batch})={rnd:.4f}"
                        f" ⇒ 语义损失这条线没学到东西（接线可能错了）")
            log(f"  [自测] 三条语义损失都低于随机水平 "
                f"(ln{args.batch}={rnd:.3f} → "
                f"{last['nce_r']:.3f}/{last['nce_z']:.3f}/{last['nce_q']:.3f}) ✓")

    ck = {"state_dict": model.state_dict(), "args": vars(args),
          "n_codes": args.codes, "d_model": D_MODEL, "arch": "assembled_v2",
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
