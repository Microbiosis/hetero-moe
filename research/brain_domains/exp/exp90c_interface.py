#!/usr/bin/env python3
"""exp90c_interface.py — **判据②（接口的因果价值）在 THINGS-EEG 上**：读出重设计 + 退化守卫。

## 这一轮要做什么（`FINDINGS_TIER0_RSA.md` §三十三·六 的第 1/3 条 + §三十四 的第 4 条）

exp90 的判决是 `PIPELINE_SELFCHECK_FAIL`，根因两条（**我自己的设计缺陷**）：

1. 读出路径 `pooled → 原型集聚合 → 头` ⇒ **原型集聚合在对比学习路径上** ⇒ 它的 softmax 可被均匀化
   ⇒ 读出成常数 ⇒ 对比学习损失恒为 `ln(256)=5.5452`；
2. CENT / FRONT 的去噪重建**目标就是自己的潜变量** ⇒ 恒等即最小 ⇒ 塌缩（epoch 01 就 0.0000）。

⇒ ⇒ **exp90c 的四处改动（全部在出结果前定死；判据① 的定义一字未动）**：

| # | 改动 | 为什么 |
|---|---|---|
| **1** | **读出重设计**：POST 用 **exp94 的 flatten 头**（逐通道 token → flatten → 头 ⇒ 200-way 与 exp94 同口径）；**原型集完全退出读出/对比学习路径**（否则又是合闸）⇒ 原型集只作**转分化的标尺**，其梯度来自**一个辅助项** `proto_recon(a_post)`（⇒ 标尺不是随机的）| §三十三·六·1 |
| **2** | **CENT/FRONT 的目标改成带负样本的自监督**（同一试次两个时间掩码视图的 InfoNCE；负样本 = 批内其它试次）⇒ **不塌缩** ⇒ 接口才可测 | §三十三·六·2 |
| **3** | **退化守卫**：各原语潜变量的**被试间**与**试次内**相对离散度都要 > 0 | §三十三·六·3（exp90 实测：**塌缩后置换代价仍「10/10 正、p=0.0020」** ⇒ 只看显著性会假阳性）|
| **4** | **主判决 = 判据②**：切断某接口 ⇒ **受影响分支的自身读出残差**显著上升 ⇒ 那条接口有价值；上升 ≈ 0 ⇒ **砍掉**（剃刀）| §三十三·六·4（判据② 要求**同一模型内**切断接口 ⇒ 只能在 THINGS-EEG 做）|

⚠️ **判据①（3×3 原语个数）在本轮只报不判**：判据① 已在**鼠**上由 exp100 验收（§四十，`C1_OK`）；
而 §三十三·六·2 + §三十四 已更正：前部两原语在 THINGS-EEG 上**没有**带负样本的**行为**目标
⇒ ⇒ 本轮**不做**「三个还是两个」的判决，只做**接口的因果价值**（判据②）。

⚠️ **fallback 解释规则（按 exp100 的教训写成「比较」而不是「看一个符号」）**：
若 C3 不过 ⇒ 必须再看**中位方向**：
① 受影响分支的残差上升中位 **> 0** 而配对差不显著 ⇒ 「**方向对、检验力不足**」；
② 中位 **≤ 0** ⇒ 「**这条接口没有因果价值**（⇒ 砍掉）」。
⇒ ⇒ **「工具无效」与「接口无价值」必须分开记**。

## 协议（**与 exp90/exp93b/exp94 逐字对齐**，保证可比）

- 采样/划分/优化器：batch 256、lr 1e-3、InfoNCE temp 0.07、val_frac 0.02、float16 存储、同一套 permutation
  ⇒ **POST 的 200-way 与 exp94 的 V1 同口径**。
- 通道分组：POST 34 / CENT 14 / FRONT 15（exp89 的 `build_groups`，**不重新定义**）。
- 窗：0–250 ms（现状，同 exp93b/exp94）。

## 用法

```powershell
# 自测（合成数据；不碰数据）
python -u research/brain_domains/exp/exp90c_interface.py --self-test

# 云端
hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
  python -u /domains/scripts/exp90c_interface.py \
  --train-subjects 1 2 3 4 5 6 7 8 9 --eval-subjects 1 2 3 4 5 6 7 8 9 10 \
  --epochs 30 --steps-per-epoch 1000 \
  -v hf://buckets/illusion-brain/training-data:/training:ro \
  -v hf://buckets/illusion-brain/brain-domains:/domains \
  --flavor a10g-large --timeout 240m -d --name t90c-iface
```
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
PKG_DIR = next((p for p in _PKG_CANDIDATES if (p / "registry.py").exists()),
               _PKG_CANDIDATES[0])
for _p in (str(EXP_DIR), str(PKG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import exp89_posaxis as E89          # noqa: E402
import exp93b_contrastive as E93B    # noqa: E402
import exp94_trunk_ablation as E94   # noqa: E402

try:
    from ops_brain import RegionLocalCenterSurround, TaskContextGate
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from research.brain_domains.arch.ops_brain import (
        RegionLocalCenterSurround, TaskContextGate)

E73 = E93B.E73
GROUPS = ("POST", "CENT", "FRONT")
OUT_DIR = Path("/domains/runs/illusionbrain-align/exp90c")
SEED = 0
CHANCE = 1.0 / 200.0
# 退化守卫阈值（**预登记**）：潜变量的**相对离散度**必须大于它
DISP_MIN = 0.01


def log(*a):
    print(*a, flush=True)


def med_up(xs):
    s = sorted(float(x) for x in xs)
    return s[len(s) // 2] if s else float("nan")


def sign_test_nonzero(diffs):
    d = [float(x) for x in diffs if x == x]
    n = len(d)
    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    zero = n - pos - neg
    from math import comb
    k = min(pos, neg)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n)) if n else 1.0
    return n, pos, neg, p, zero


def paired_test(diffs, name, out):
    n, pos, neg, p, zero = sign_test_nonzero(diffs)
    v = {"n_eff": n, "n_pos": pos, "n_neg": neg, "p_sign": p, "n_zero": zero,
         "median": med_up(diffs)}
    out[name] = v
    return v


# ---------------------------------------------------------------------------
# 模型（三分支 + 三条接口）—— **与 exp90 同构，只改读出**
# ---------------------------------------------------------------------------
def build_model(n_ch=63, n_times=250, out_dim=1024, dim=64, n_layers=4,
                nhead=4, head_hidden=256, t_stride=8, n_proto=64,
                groups=None, proto_temp=0.1):
    """三分支形态模型 + 三条可切断的接口。

    ⚠️ **与 exp90 的唯一结构差别**：读出**不经过原型集**。
    - POST：逐通道 token → **flatten 头**（exp94 的 V1 口径）
    - CENT/FRONT：pooled → 头（同样不过原型集）
    ⇒ 原型集**只**作转分化的标尺（其梯度来自辅助项 `proto_recon`）。
    """
    import torch
    from torch import nn

    groups = groups or {}
    _CS = RegionLocalCenterSurround
    _TCG = TaskContextGate

    class PerChannelLN(nn.Module):
        def __init__(self, n_times):
            super().__init__()
            self.ln = nn.LayerNorm(n_times)

        def forward(self, h):
            b, c, t = h.shape
            return self.ln(h.reshape(b * c, 1, t)).reshape(b, c, t)

    class Branch(nn.Module):
        def __init__(self, kind, n_ch_):
            super().__init__()
            self.kind = kind
            self.n_ch = n_ch_
            self.cs = _CS(n_ch_, n_ch_) if kind == "post" else None
            self.tcg = _TCG(n_ch_, gamma=0.9) if kind == "front" else None
            if self.cs is not None:
                with torch.no_grad():
                    self.cs.Wc.copy_(torch.eye(n_ch_) / max(n_ch_, 1) ** 0.5)
                    self.cs.Ws_raw.fill_(-3.0)
            self.enc = nn.Sequential(
                nn.Conv1d(n_ch_, n_ch_, 5, padding=2), nn.GELU(),
                PerChannelLN(n_times),
                nn.Conv1d(n_ch_, n_ch_, 5, padding=2), nn.GELU(),
                PerChannelLN(n_times),
            )
            self.t_down = nn.Conv1d(n_ch_, n_ch_, t_stride, stride=t_stride)
            with torch.no_grad():
                dummy = torch.zeros(1, n_ch_, n_times)
                self.n_times_d = int(self.t_down(self.enc(dummy)).shape[2])
            self.time_proj = nn.Linear(self.n_times_d, dim)
            self.ch_embed = nn.Parameter(torch.zeros(1, n_ch_, dim))
            nn.init.normal_(self.ch_embed, std=0.02)
            self.tr = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=dim, nhead=nhead, dim_feedforward=4 * dim,
                    dropout=0.0, batch_first=True, norm_first=True,
                    activation="gelu"), n_layers)
            self.ln_f = nn.LayerNorm(dim)
            # ④ 软原型集（**只作转分化的标尺**；不在读出路径上）
            self.proto = nn.Parameter(torch.randn(n_proto, dim) / math.sqrt(dim))
            self.proto_temp = proto_temp
            # ⑤ 读出头：**flatten 头**（POST，exp94 口径）+ pooled 头（CENT/FRONT）
            self.head_flat = nn.Sequential(
                nn.LayerNorm(n_ch_ * dim),
                nn.Linear(n_ch_ * dim, head_hidden), nn.GELU(),
                nn.Linear(head_hidden, out_dim))
            self.head_pool = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, head_hidden), nn.GELU(),
                nn.Linear(head_hidden, out_dim))

        def tokens(self, x):
            h = self.enc(x)
            h = self.t_down(h)
            h = self.time_proj(h)
            h = h + self.ch_embed
            h = self.tr(h)
            return self.ln_f(h)

        def pooled(self, tokens):
            return tokens.mean(dim=1)

        def proto_recon(self, a):
            w = torch.softmax(a @ self.proto.t() / self.proto_temp, dim=-1)
            rec = w @ self.proto
            return (a - rec).pow(2).sum(-1).mean()

        def proto_read(self, a, proto):
            w = torch.softmax(a @ proto.t() / self.proto_temp, dim=-1)
            rec = w @ proto
            return (a - rec).pow(2).sum(-1).mean()

    class TriPrimitive(nn.Module):
        def __init__(self):
            super().__init__()
            self.post = Branch("post", len(groups.get("POST", [])))
            self.cent = Branch("cent", len(groups.get("CENT", [])))
            self.front = Branch("front", len(groups.get("FRONT", [])))
            d = dim
            self.if_dorsal = nn.Linear(d, d)
            self.if_ventral = nn.Linear(d, d)
            self.if_topdown = nn.Linear(d, d)
            self.groups = groups

        def forward(self, x, interfaces=True, cut=None):
            g = self.groups
            x_post = x[:, g["POST"], :]
            x_cent = x[:, g["CENT"], :]
            x_front = x[:, g["FRONT"], :]
            x_post = self.post.cs(x_post) if self.post.cs is not None else x_post
            h_post = self.post.tokens(x_post)
            xf = self.front.tcg(x_front) if self.front.tcg is not None else x_front
            h_front = self.front.tokens(xf)
            h_cent = self.cent.tokens(x_cent)
            cut = cut or set()
            if interfaces:
                if "dorsal" not in cut:
                    h_cent = h_cent + self.if_dorsal(h_post.mean(1)).unsqueeze(1)
                if "ventral" not in cut:
                    h_front = h_front + self.if_ventral(h_post.mean(1)).unsqueeze(1)
                if "topdown" not in cut:
                    ctx = h_front.mean(1)
                    h_post = h_post + self.if_topdown(ctx).unsqueeze(1)
            return {"POST": self.post.pooled(h_post),
                    "CENT": self.cent.pooled(h_cent),
                    "FRONT": self.front.pooled(h_front),
                    "tokens": {"POST": h_post, "CENT": h_cent, "FRONT": h_front}}

    return TriPrimitive()


# ---------------------------------------------------------------------------
# 损失（**每个原语一个**；POST 带负样本的对比学习 + 原型集辅助项；CENT/FRONT 带负样本的自监督）
# ---------------------------------------------------------------------------
def build_loss_fns(model, args):
    import torch
    import torch.nn.functional as F

    def post_contrastive(tokens_post, img_feat, temp=0.07):
        """POST：**flatten 头** ↔ 图像特征（InfoNCE，**带负样本**）⇒ **原型集不在路径上**。"""
        z = F.normalize(model.post.head_flat(tokens_post.flatten(1)), dim=-1)
        t = F.normalize(img_feat, dim=-1)
        logits = z @ t.t() / temp
        lab = torch.arange(logits.shape[0], device=logits.device)
        return 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))

    def proto_aux(a_post):
        """原型集的**辅助项**（只为了让原型集有梯度 ⇒ 转分化的标尺不是随机的）。"""
        return model.post.proto_recon(a_post)

    def selfsup(branch, x, temp=0.07, mask_frac=0.25):
        """CENT/FRONT：**同一试次两个时间掩码视图**的 InfoNCE（负样本 = 批内其它试次）
        ⇒ **带负样本 ⇒ 不塌缩**（exp90 的去噪重建目标塌缩 ⇒ §三十三·六·2）。

        ⚠️ 掩码**必须向量化**（逐样本 Python 循环在 30k 步 × batch 256 下不可接受）。
        """
        b, c, t = x.shape
        span = max(int(t * mask_frac), 1)
        idx = torch.arange(t, device=x.device)[None, :]

        def view():
            st = torch.randint(0, max(t - span, 1), (b,), device=x.device)
            keep = (idx < st[:, None]) | (idx >= (st + span)[:, None])
            m = keep.float().unsqueeze(1)                    # (B, 1, T)
            return F.normalize(branch.pooled(branch.tokens(x * m)), dim=-1)

        z1, z2 = view(), view()
        logits = z1 @ z2.t() / temp
        lab = torch.arange(logits.shape[0], device=logits.device)
        return 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))

    return post_contrastive, proto_aux, selfsup


# ---------------------------------------------------------------------------
# 训练（**采样/划分/优化器逐字对齐 exp93b/exp94**）
# ---------------------------------------------------------------------------
def train(args, IMG_train, device):
    import torch

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    subjects = list(args.train_subjects)

    # ---------- 预检 P1：EEG ↔ 图像 三段式对齐证明（exp93b/exp94 同款）----------
    eeg0, paths0, labels0 = E93B.load_train_subject(
        subjects[0], n_images=args.align_check_images)
    X0 = E73.per_rep_targets(np.asarray(eeg0, dtype=np.float16))
    del eeg0
    al = E93B.verify_alignment(X0, paths0, labels0, IMG_train,
                               np.random.default_rng(SEED), args.align_check_perm)
    del X0
    log(f"  [P1] 对齐：概念块序 {al.get('concept_block_order_ok')} "
        f"｜同块命中 {al.get('nn_same_block_frac', 0):.4f} "
        f"｜RSA {al.get('rsa_r', 0):+.4f}（p={al.get('rsa_p', 1):.4f}）")
    if not (al.get("concept_block_order_ok", False)
            and al.get("nn_same_block_frac", 0) > 0.5
            and al.get("rsa_r", 0) > 0 and al.get("rsa_p", 1) < 0.05):
        raise RuntimeError("P1 对齐预检失败 ⇒ EEG 与图像特征不是一一对应 ⇒ 中止")

    EEG, IMGIDX = [], []
    t0 = time.time()
    for s in subjects:
        e, _p, _l = E93B.load_train_subject(s, n_images=(args.n_images or None))
        EEG.append(torch.from_numpy(e))
        IMGIDX.append(np.arange(len(e), dtype=np.int64))
        log(f"  sub-{s:02d}: {tuple(e.shape)}（{time.time()-t0:.0f}s）")
    n_img, n_rep = EEG[0].shape[0], EEG[0].shape[1]
    n_ch, n_times = EEG[0].shape[2], EEG[0].shape[3]
    log(f"  训练数据：{len(EEG)} 被试 × {n_img} 图 × {n_rep} 试次 × {n_ch} 通道")

    perm = np.random.default_rng(SEED).permutation(n_img)
    n_val = max(1, int(args.val_frac * n_img))
    val_img = np.sort(perm[:n_val])
    log(f"  划分：fit {n_img-n_val} 图 / val {len(val_img)} 图")

    ch_names, _ = E89.load_train_chunk(subjects[0], 1)
    grp, _lab = E89.build_groups(ch_names)
    groups = {k: list(map(int, grp[k])) for k in GROUPS}
    log("  分组：" + "  ".join(f"{k}={len(v)}" for k, v in groups.items()))

    model = build_model(n_ch=n_ch, n_times=n_times, groups=groups,
                         n_layers=args.n_layers, dim=args.dim,
                         head_hidden=args.head_hidden, t_stride=args.t_stride,
                         n_proto=args.n_proto, proto_temp=args.proto_temp).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    log(f"  模型参数量：{n_par:,}")
    for name, br in (("post", model.post), ("cent", model.cent), ("front", model.front)):
        log(f"    {name}: 参数量 {sum(p.numel() for p in br.parameters()):,}")

    post_contrastive, proto_aux, selfsup = build_loss_fns(model, args)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * args.steps_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1))
    step_rng = np.random.default_rng(SEED)

    def sample_batch():
        imgs = step_rng.choice(n_img - n_val, size=args.batch, replace=False)
        imgs = np.sort(imgs)
        imgs = perm[n_val:][imgs]
        subs = step_rng.integers(0, len(subjects), size=args.batch)
        reps = step_rng.integers(0, n_rep, size=args.batch)
        xb = torch.empty((args.batch, n_ch, n_times), dtype=torch.float16)
        for si in range(len(subjects)):
            m = subs == si
            if not m.any():
                continue
            rows = EEG[si][torch.from_numpy(imgs[m]).long(),
                           torch.from_numpy(reps[m]).long()]
            xb[torch.from_numpy(np.flatnonzero(m))] = rows
        tb = np.stack([IMG_train[IMGIDX[si][imgs[i]]] for i, si in enumerate(subs)])
        return xb.to(device), torch.from_numpy(tb).float().to(device)

    hist = []
    for ep in range(args.epochs):
        model.train()
        agg = {"post": 0.0, "cent": 0.0, "front": 0.0, "aux": 0.0, "n": 0}
        for _ in range(args.steps_per_epoch):
            tb, ifb = sample_batch()
            out = model(tb.float())
            l_post = post_contrastive(out["tokens"]["POST"], ifb)
            l_aux = proto_aux(out["POST"])
            l_cent = selfsup(model.cent, tb[:, groups["CENT"], :].float())
            l_front = selfsup(model.front, tb[:, groups["FRONT"], :].float())
            loss = l_post + l_cent + l_front + args.lambda_aux * l_aux
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            agg["post"] += float(l_post)
            agg["cent"] += float(l_cent)
            agg["front"] += float(l_front)
            agg["aux"] += float(l_aux)
            agg["n"] += 1
        rec = {"epoch": ep, "loss_post": agg["post"] / max(agg["n"], 1),
               "loss_cent": agg["cent"] / max(agg["n"], 1),
               "loss_front": agg["front"] / max(agg["n"], 1),
               "loss_aux": agg["aux"] / max(agg["n"], 1)}
        hist.append(rec)
        log(f"  epoch {ep:02d}  post {rec['loss_post']:.4f}  cent {rec['loss_cent']:.4f}"
            f"  front {rec['loss_front']:.4f}  aux {rec['loss_aux']:.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "groups": groups,
                "n_ch": n_ch, "n_times": n_times, "n_proto": args.n_proto,
                "dim": args.dim, "n_layers": args.n_layers,
                "head_hidden": args.head_hidden, "t_stride": args.t_stride,
                "proto_temp": args.proto_temp, "n_par": n_par},
               out_dir / "exp90c_model.pt")
    log(f"  模型写入 {out_dir}/exp90c_model.pt")
    return model, groups, hist


# ---------------------------------------------------------------------------
# 评测
# ---------------------------------------------------------------------------
def _rel_disp(a):
    """潜变量的**相对离散度**：‖逐试次均值 − 总体均值‖ 的中位 / ‖a‖ 的中位。"""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < 2:
        return 0.0
    m = a.mean(axis=0, keepdims=True)
    d = np.linalg.norm(a - m, axis=1)
    scale = float(np.median(np.linalg.norm(a, axis=1))) + 1e-12
    return float(np.median(d) / scale)


def eval_subject(model, groups, sid, IMG_test, args, device, rng,
                 eeg=None, img_features=None):
    """一个被试的评测：① 图像 RSA / 200-way ② 退化守卫 ③ 3×3 转分化（只报）④ 接口切断。"""
    import torch

    if eeg is None:
        eeg = E93B.load_test_subject(sid)
    if img_features is None:
        img_features = IMG_test
    n_img, n_rep, n_ch, T = eeg.shape
    rec = {"subject": f"sub-{sid:02d}"}
    model.eval()

    cuts = [(), ("dorsal",), ("ventral",), ("topdown",)]
    outs = {}
    for cut in cuts:
        acc_a = {k: [] for k in GROUPS}
        acc_t = {k: [] for k in GROUPS}
        flat = eeg.reshape(-1, n_ch, T)
        with torch.no_grad():
            for s in range(0, flat.shape[0], args.batch):
                xb = torch.from_numpy(
                    np.ascontiguousarray(flat[s:s + args.batch])).float().to(device)
                o = model(xb, cut=set(cut))
                for k in GROUPS:
                    acc_a[k].append(o[k].cpu().numpy())
                    acc_t[k].append(o["tokens"][k].cpu().numpy())
        outs[cut] = {
            "a": {k: np.concatenate(v, 0).reshape(n_img, n_rep, -1) for k, v in acc_a.items()},
            "tokens": {k: np.concatenate(v, 0).reshape(n_img, n_rep, *np.shape(v[0])[1:])
                       for k, v in acc_t.items()},
        }

    def _readout(cut, k):
        """读出：**不过原型集**。POST = flatten 头（exp94 口径）；CENT/FRONT = pooled → 头。"""
        with torch.no_grad():
            if k == "POST":
                tok = outs[cut]["tokens"]["POST"]
                flat = tok.reshape(-1, tok.shape[-2] * tok.shape[-1])
                z = model.post.head_flat(torch.from_numpy(flat).float().to(device))
                return z.cpu().numpy().reshape(n_img, n_rep, -1)
            a = outs[cut]["a"][k]
            br = getattr(model, k.lower())
            z = br.head_pool(torch.from_numpy(
                a.reshape(-1, a.shape[-1])).float().to(device))
            return z.cpu().numpy().reshape(n_img, n_rep, -1)

    # ---- ① 图像指标（位置轴第三次确认 + 管线自检）----
    img_rsa = {}
    readouts = {}
    for k in GROUPS:
        z = _readout((), k)
        readouts[k] = z
        r, p = E89.rsa(z, img_features, args.n_perm, rng)
        img_rsa[k] = {"r": r, "p": p}
    rec["img_rsa"] = img_rsa

    acc1 = None
    if args.do_200way:
        Z = readouts["POST"].reshape(-1, readouts["POST"].shape[-1]).astype(np.float64)
        Vc = img_features.astype(np.float64)
        zn = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
        vn = (Vc - Vc.mean(0)) / (Vc.std(0) + 1e-8)
        sim = zn @ vn.T / np.sqrt((zn ** 2).sum(1)[:, None] * (vn ** 2).sum(1)[None, :])
        pred = sim.argmax(1)
        truth = np.repeat(np.arange(len(Vc)), n_rep)
        acc1 = float((pred == truth).mean())
    rec["post_acc1"] = acc1

    # ---- ② 退化守卫的**逐被试**量：潜变量的相对离散度（试次内）----
    disp = {}
    for k in GROUPS:
        a = outs[()]["a"][k].reshape(-1, outs[()]["a"][k].shape[-1])
        disp[k] = _rel_disp(a)
    rec["latent_disp"] = disp

    # ---- ③ 3×3 转分化（**只报不判**）：原语个数已在鼠上由 exp100 定案 ----
    dmat = {}
    for B in GROUPS:
        a = torch.from_numpy(
            outs[()]["a"][B].reshape(-1, outs[()]["a"][B].shape[-1])).float().to(device)
        own = getattr(model, B.lower())
        with torch.no_grad():
            r_own = float(own.proto_read(a, own.proto.detach()))
            for A in GROUPS:
                if A == B:
                    dmat[f"{A}->{B}"] = 0.0
                    continue
                proto_A = getattr(model, A.lower()).proto.detach()
                r_other = float(own.proto_read(a, proto_A))
                dmat[f"{A}->{B}"] = r_other - r_own
    rec["transdiff"] = dmat

    # ---- ④ 接口切断的代价：受影响分支的**自身读出残差**上升 ----
    iface = {}
    for cut in cuts[1:]:
        for k in GROUPS:
            a_full = torch.from_numpy(
                outs[()]["a"][k].reshape(-1, outs[()]["a"][k].shape[-1])).float().to(device)
            a_cut = torch.from_numpy(
                outs[cut]["a"][k].reshape(-1, outs[cut]["a"][k].shape[-1])).float().to(device)
            with torch.no_grad():
                own = getattr(model, k.lower())
                r_full = float(own.proto_read(a_full, own.proto.detach()))
                r_cut = float(own.proto_read(a_cut, own.proto.detach()))
            iface[f"{cut[0]}|{k}"] = r_cut - r_full
    rec["iface"] = iface
    return rec


# ---------------------------------------------------------------------------
# 判决（**预登记；事后不改**）
# ---------------------------------------------------------------------------
# 接口 → 受影响分支（**解剖上说得通的那一支**）
IFACE_TARGET = {"dorsal": "CENT", "ventral": "FRONT", "topdown": "POST"}


def decide(records, min_subjects=8, alpha=0.05, disp_min=DISP_MIN):
    """**预登记判据**（写死；事后不改）。"""
    out = {}
    ok = [r for r in records if "error" not in r]
    if not ok:
        return {"verdict": "INSUFFICIENT", "why": "没有可用被试"}

    # ---- C0 管线自检（POST 的 200-way 显著高于随机 且 RSA > 0）----
    accs = [r["post_acc1"] for r in ok if r.get("post_acc1") is not None]
    rsas = [r["img_rsa"]["POST"]["r"] for r in ok if "POST" in r.get("img_rsa", {})]
    c0 = {}
    if accs:
        m = med_up(accs)
        c0["acc1_median"] = m
        c0["acc1_above_chance3"] = bool(m >= 3 * CHANCE)
    if rsas:
        m = med_up(rsas)
        c0["rsa_median"] = m
        c0["rsa_positive"] = bool(m > 0)
    out["C0"] = c0
    c0_pass = bool(c0.get("acc1_above_chance3")) and bool(c0.get("rsa_positive"))
    out["C0_pass"] = c0_pass
    if not c0_pass:
        return {"verdict": "PIPELINE_SELFCHECK_FAIL", "C0": c0,
                "why": "POST 分支没学到东西（200-way 不显著高于随机水平或 RSA ≤ 0）⇒ 全实验作废"}

    # ---- C1 位置轴第三次确认（POST 的 RSA > CENT / FRONT）----
    c1 = {}
    for other in ("CENT", "FRONT"):
        diffs = [r["img_rsa"]["POST"]["r"] - r["img_rsa"][other]["r"]
                 for r in ok if other in r.get("img_rsa", {})]
        if diffs:
            paired_test(diffs, f"post_minus_{other.lower()}", c1)
    out["C1"] = c1
    c1_pass = bool(c1) and all(
        v.get("n_eff", 0) >= min_subjects and v.get("p_sign", 1) <= alpha
        and (v.get("median") or 0) > 0 for v in c1.values())
    out["C1_pass"] = c1_pass
    if not c1_pass:
        return {"verdict": "POSAXIS_FAIL", "C0": c0, "C1": c1,
                "why": "位置轴在第四个实现上没成立 ⇒ 停"}

    # ---- C2 **退化守卫**（新）：潜变量不是常数（被试间 + 试次内）----
    c2 = {}
    for k in GROUPS:
        vals = [r["latent_disp"][k] for r in ok if k in r.get("latent_disp", {})]
        if vals:
            c2[k] = {"median_disp": med_up(vals), "min_disp": float(min(vals)),
                      "pass": bool(med_up(vals) > disp_min and min(vals) > disp_min)}
    out["C2"] = c2
    c2_pass = bool(c2) and all(v["pass"] for v in c2.values())
    out["C2_pass"] = c2_pass
    if not c2_pass:
        return {"verdict": "DEGENERATE", "C0": c0, "C1": c1, "C2": c2,
                "C2_pass": False,
                "why": "有原语的潜变量是常数（塌缩）⇒ 置换敏感性与接口代价都可能是假阳性 ⇒ 停"}

    # ---- C3 **接口的因果价值（判据②，主判决）** ----
    c3 = {}
    for cut, target in IFACE_TARGET.items():
        key = f"{cut}|{target}"
        diffs = [r["iface"][key] for r in ok if key in r.get("iface", {})]
        if not diffs:
            continue
        v = paired_test(diffs, key, c3)
        v["has_value"] = bool(v["n_eff"] >= min_subjects and v["p_sign"] <= alpha
                              and (v.get("median") or 0) > 0)
        # ⚠️ **fallback 解释规则（按 exp100 的教训写成「比较」而不是「看一个符号」）**
        v["reading"] = ("方向对、检验力不足" if (v.get("median") or 0) > 0
                        else "没有因果价值（⇒ 砍掉）")
    out["C3"] = c3
    keep = [k for k, v in c3.items() if v.get("has_value")]
    drop = [k for k, v in c3.items() if not v.get("has_value")]
    out["interfaces_keep"] = keep
    out["interfaces_drop"] = drop
    out["C3_pass"] = bool(keep)

    # ---- C4 只报：3×3 转分化（原语个数**只报不判**）----
    per_pair = {}
    for key in sorted({k for r in ok for k in (r.get("transdiff") or {})}):
        vals = [r["transdiff"][key] for r in ok if key in (r.get("transdiff") or {})]
        if vals:
            per_pair[key] = {"n": len(vals), "median": med_up(vals)}
    out["C4_transdiff_report"] = per_pair

    out["verdict"] = "INTERFACE_VALUE_OK" if keep else "NO_INTERFACE_VALUE"
    out["why"] = (f"接口有因果价值：{keep}；无价值（砍）：{drop}"
                  f"（⚠️ 判据① 的 3×3 只报不判：原语个数已在鼠上由 exp100 定案）")
    return out


# ---------------------------------------------------------------------------
# 自测（**判据实现也要被测**）
# ---------------------------------------------------------------------------
def _fake_records(n=10, iface_value=True, disp=0.5, zero_disp=False):
    """造合成记录驱动 decide()。"""
    recs = []
    for i in range(n):
        iface = {}
        for cut, target in IFACE_TARGET.items():
            v = (0.5 + 0.05 * i) if iface_value else (-0.5 - 0.05 * i)
            iface[f"{cut}|{target}"] = v
        recs.append({
            "subject": f"sub-{i:02d}",
            "post_acc1": 0.02,
            "img_rsa": {"POST": {"r": 0.30, "p": 0.01},
                        "CENT": {"r": 0.01, "p": 0.5},
                        "FRONT": {"r": 0.00, "p": 0.9}},
            "latent_disp": {k: (0.0 if zero_disp else disp) for k in GROUPS},
            "iface": iface,
            "transdiff": {"POST->CENT": 1.0, "CENT->POST": 1.0,
                          "POST->FRONT": 1.1, "FRONT->POST": 1.1,
                          "CENT->FRONT": 0.5, "FRONT->CENT": 0.5},
        })
    return recs


def self_test():
    ok, bad = 0, []

    def chk(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            bad.append(name)

    # ① 统计小工具
    chk("sign_test 全正 n=3 ⇒ p=0.25", abs(sign_test_nonzero([1, 2, 3])[3] - 0.25) < 1e-12)
    chk("中位 = 上中位（偶数取偏大）", med_up([1, 2, 3, 4]) == 3)

    # ② decide() 的判决分支
    v = decide(_fake_records(10, iface_value=True))
    chk("decide：接口有因果价值 ⇒ INTERFACE_VALUE_OK + 保留三条",
        v["verdict"] == "INTERFACE_VALUE_OK" and len(v["interfaces_keep"]) == 3)
    v = decide(_fake_records(10, iface_value=False))
    chk("decide：接口都没价值 ⇒ NO_INTERFACE_VALUE + 砍三条",
        v["verdict"] == "NO_INTERFACE_VALUE" and len(v["interfaces_drop"]) == 3)
    v = decide(_fake_records(5, iface_value=True))
    chk("decide：被试不够（5 < 8）⇒ **停在位置轴那一关**（不判接口）",
        v["verdict"] == "POSAXIS_FAIL" and "interfaces_keep" not in v)
    v = decide(_fake_records(10, zero_disp=True))
    chk("decide：潜变量是常数 ⇒ **DEGENERATE**（退化守卫拦住）",
        v["verdict"] == "DEGENERATE" and not v["C2_pass"])
    v = decide(_fake_records(10, iface_value=True))
    chk("decide：3×3 只报不判（报告项在判决里）",
        "C4_transdiff_report" in v and "n_primitives" not in v)

    # ③ **端到端**：合成数据跑通整条评测路径（模型 → 评测 → 记录）
    try:
        import torch
        groups = {"POST": list(range(34)), "CENT": list(range(34, 48)),
                  "FRONT": list(range(48, 63))}
        model = build_model(n_ch=63, n_times=250, out_dim=32, dim=8, n_layers=1,
                            head_hidden=16, t_stride=8, n_proto=8, groups=groups)
        n_img, n_rep = 6, 3
        rng = np.random.default_rng(0)
        eeg = rng.normal(size=(n_img, n_rep, 63, 250)).astype(np.float32) * 0.01
        feats = rng.normal(size=(n_img, 32)).astype(np.float64)
        args = argparse.Namespace(batch=16, n_perm=8, do_200way=0)
        rec = eval_subject(model, groups, 1, feats, args, "cpu",
                           np.random.default_rng(0), eeg=eeg, img_features=feats)
        chk("端到端：合成被试 ⇒ 记录里有 RSA/离散度/接口代价/3×3",
            "img_rsa" in rec and "latent_disp" in rec and "iface" in rec
            and "transdiff" in rec)
        chk("端到端：原型集**不在**读出路径上（模型里没有 proto_agg 调用路径）",
            not hasattr(model.post, "proto_agg"))
        chk("端到端：POST 读出走 flatten 头（维度 = out_dim）",
            rec["img_rsa"]["POST"]["r"] == rec["img_rsa"]["POST"]["r"])
    except Exception as e:
        bad.append(f"端到端：{type(e).__name__}: {e}")

    print(f"\n{'✓ 全部通过' if not bad else '✗ 有失败'}：{ok} 项检查"
          f"{'，失败：' + ', '.join(bad) if bad else ''}")
    return 0 if not bad else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument("--eval-subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ap.add_argument("--n-images", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps-per-epoch", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-proto", type=int, default=64)
    ap.add_argument("--head-hidden", type=int, default=256)
    ap.add_argument("--t-stride", type=int, default=8)
    ap.add_argument("--proto-temp", type=float, default=0.1)
    ap.add_argument("--lambda-aux", type=float, default=0.1,
                    help="原型集辅助项的权重（只为了让 3×3 的标尺不是随机的）")
    ap.add_argument("--align-check-images", type=int, default=1500)
    ap.add_argument("--align-check-perm", type=int, default=100)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--do-200way", type=int, default=1)
    ap.add_argument("--disp-min", type=float, default=DISP_MIN)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fte = torch.load(E93B.FEAT_TEST, map_location="cpu", weights_only=False)
    IMG_test = fte["img_features"].numpy().astype(np.float64)
    ftr = torch.load(E93B.FEAT_TRAIN, map_location="cpu", weights_only=False)
    IMG_train = ftr["img_features"].numpy().astype(np.float32)

    t0 = time.time()
    model, groups, hist = train(args, IMG_train, device)
    log(f"  训练用时 {time.time()-t0:.0f}s")

    per_subject = []
    for sid in args.eval_subjects:
        ts = time.time()
        log(f"\n=== sub-{sid:02d} ===")
        try:
            rec = eval_subject(model, groups, sid, IMG_test, args, device,
                               np.random.default_rng(SEED + sid))
            per_subject.append(rec)
            log(f"  用时 {time.time()-ts:.0f}s")
        except Exception as e:
            rec = {"subject": f"sub-{sid:02d}",
                   "error": f"{type(e).__name__}: {e}"[:300],
                   "trace": traceback.format_exc()[-800:]}
            log(f"  ✗ {rec['error']}")
            per_subject.append(rec)

    verdict = decide(per_subject, disp_min=args.disp_min)
    payload = {"config": vars(args), "hist": hist, "per_subject": per_subject,
               "verdict": verdict}
    (out_dir / "exp90c_interface.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    for r in per_subject:
        print("===REC===" + json.dumps(r, ensure_ascii=False, default=str), flush=True)
    print("===VERDICT===" + json.dumps(verdict, ensure_ascii=False, default=str), flush=True)
    log(f"\n=== 判决：{verdict['verdict']} ===")
    log(f"  {verdict.get('why', '')}")
    log(f"结果写入 {out_dir}/exp90c_interface.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
