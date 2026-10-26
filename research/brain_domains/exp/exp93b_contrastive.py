#!/usr/bin/env python3
"""exp93b_contrastive.py — **对比学习**的 EEG 编码器（exp93 的「换训练目标」那一半）。

## 为什么

exp92 的交叉验证给出 **A18**：我们自训的 tokenizer（S1a/S1b）**RSA 更高**，
但 200-way top-1 **不如手工 630-d 基线**。exp93a 负责回答「差距在协议/特征/模型哪一层」。

⇒ 若差距在**模型**这一层，正确的修法不是调超参，而是**换训练目标**：
S1a 的目标是**掩码波形重建**（纯自监督、单被试）——它被明确优化成「还原波形」，
**从未被要求保留一阶判别性**。NICE-EEG（arXiv:2308.13234, ICLR 2024）的标准做法是
**图像编码器 + EEG 编码器对比学习**（+ 空间注意力）。

⇒ 本脚本照标准做法实现：
- 损失：对称 InfoNCE（EEG 投影 vs ViT-H-14 图像特征，温度可调）
- 批次内**图像互不相同**（否则同一张图的不同重复会变成假负例）
- **多被试联合训练**（默认 9 个被试训练，留第 10 个做真跨被试泛化检验）

## 与脑域线的关系（归属）

本脚本是 **B 类（手段评估）**：回答「我们的底座够不够格」，
**不产生任何脑结构主张**。但它产出的编码器**可以**作为
「后部语义原语」的候选底座（前部原语仍走自监督路线 ⇒ 那是 exp90 的事）。

## 三条读出（口径不同，都要报）

| 读出 | 含义 | 对照 |
|---|---|---|
| `zeroshot` | cos(编码器输出, 图像特征) → argmax | **NICE-EEG 口径**（200-way 零样本检索） |
| `ridge_feat` | 编码器输出 → ridge → 图像空间 → cos | **A13 / exp92-C 口径**（可与 1.1% 直接比） |
| `rsa` | 编码器输出与图像特征的 RSA | 与 exp89 / exp92-A 的位置轴数字同口径 |

## 预检（不通过就中止，不产出结论）

- **P1 对齐预检**：630-d 手工特征在 train split 上的 RSA 必须显著为正
  ⇒ 图像特征与 EEG 行序对齐正确（这是整套评测的地基）。
  若 ≈ 0 ⇒ 对齐错了，**直接退出**，不训练。
- **P2 置换预检**：InfoNCE 在「打乱配对」下必须显著变差 ⇒ 损失实现方向没写反。

## 用法

    # 本地自测
    python -u research/brain_domains/exp/exp93b_contrastive.py --self-test

    # 云端（需要 GPU）
    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp93b_contrastive.py \
      --train-subjects 1,2,3,4,5,6,7,8,9 --eval-subjects 1,2,3,4,5,6,7,8,9,10 \
      --epochs 30 --steps-per-epoch 1000 \
      -v hf://buckets/illusion-brain/training-data:/training:ro \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor a10g-large --timeout 180m -d --name t93b-contrastive
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

import exp73_decode200 as E73   # noqa: E402
import exp89_posaxis as E89       # noqa: E402

TRAIN_PT = Path("/training/eeg/things-eeg/Preprocessed_data_250Hz_whiten")
TEST_4D = Path("/domains/omnibrain4d/eeg")
FEAT_TEST = "/training/eeg/eeg-image-decode/ViT-H-14_features_test.pt"
FEAT_TRAIN = "/training/eeg/eeg-image-decode/ViT-H-14_features_train.pt"
OUT_DIR = Path("/domains/runs/illusionbrain-align/exp93b")

SEED = 0
CHANCE = 1.0 / 200.0
DIM = 64
T_STRIDE = 8


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------

def load_train_subject(subject, n_images=None):
    """→ (eeg float16 (n_img, n_rep, C, T), img_paths list | None, labels | None)

    ⚠️ 实测 `img` 是**每图 4 条相同路径**（不是索引），形如
    'train_images/00001_aardvark/aardvark_01b.jpg'；`label` 是概念号 0..1653。
    """
    import torch
    fp = TRAIN_PT / f"sub-{subject:02d}" / "train.pt"
    try:
        obj = torch.load(fp, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        obj = torch.load(fp, map_location="cpu", weights_only=False)
    eeg = obj["eeg"]
    img_paths, labels = None, None
    if isinstance(obj, dict) and "img" in obj:
        cand = np.asarray(obj["img"])
        if cand.ndim == 2 and cand.shape[1] == eeg.shape[1]:
            img_paths = [str(x) for x in cand[:, 0]]
        elif cand.ndim == 1 and len(cand) == len(eeg):
            img_paths = [str(x) for x in cand]
    if isinstance(obj, dict) and "label" in obj:
        lab = np.asarray(obj["label"])
        if lab.ndim == 2 and lab.shape[1] == eeg.shape[1]:
            labels = lab[:, 0].astype(np.int64)
        elif lab.ndim == 1 and len(lab) == len(eeg):
            labels = lab.astype(np.int64)
    if n_images is not None:
        eeg = eeg[:n_images]
        if img_paths is not None:
            img_paths = img_paths[:n_images]
        if labels is not None:
            labels = labels[:n_images]
    eeg = np.asarray(eeg, dtype=np.float16)
    return eeg, img_paths, labels


def concept_of_path(p):
    """'train_images/00001_aardvark/aardvark_01b.jpg' → 0（概念号减一）。"""
    try:
        return int(p.split("/")[1][:5]) - 1
    except Exception:
        return -1


def verify_alignment(X, img_paths, labels, IMG_train, rng, n_perm):
    """**三段式对齐证明**（与 exp93a 同一套判据）。

    ① EEG 行 → 路径 → 概念号，必须逐行等于 `label`，且行序是「概念升序 × 每概念 10 张」
    ② 图像特征文件行序：每行在特征空间的最近邻必须落在**自己的 10 块**内
    ③ EEG 与图像特征的 RSA 必须显著
    """
    out = {}
    if img_paths is not None and labels is not None:
        con = np.array([concept_of_path(p) for p in img_paths])
        out["label_match_frac"] = float((con == labels).mean())
        out["concept_block_order_ok"] = bool(
            np.array_equal(con, np.arange(len(con)) // 10))
    F = IMG_train / np.maximum(np.linalg.norm(IMG_train, axis=1, keepdims=True), 1e-12)
    rr = np.random.default_rng(7)
    rows = rr.choice(len(F), size=min(2000, len(F)), replace=False)
    nn = (F[rows] @ F.T).argmax(axis=1)
    out["nn_same_block_frac"] = float((nn // 10 == rows // 10).mean())
    r, p = E89.rsa(X, IMG_train[:len(X)], n_perm, rng)
    out["rsa_r"], out["rsa_p"] = float(r), float(p)
    return out


def load_test_subject(subject):
    return np.load(TEST_4D / f"sub-{subject:02d}.npz")["eeg"]


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------

def build_model(n_ch=63, n_times=250, out_dim=1024, dim=DIM):
    import torch
    from torch import nn

    class EEGEncoder(nn.Module):
        """exp80 那种「每时间片一个 token」的躯干 + 一个全局读出头。

        躯干：ch_mix(1×1) → 2×[Conv1d(k=5)+GELU+GroupNorm] → 时间下采样 ×8
              ⇒ (B, n_ch, 250) → (B, dim, 31)
        头  ：31 个 token 展平 → MLP → 图像特征空间（外面再 L2 归一化）
        """

        def __init__(self):
            super().__init__()
            self.ch_mix = nn.Conv1d(n_ch, dim, 1)
            self.enc = nn.Sequential(
                nn.Conv1d(dim, dim, 5, padding=2), nn.GELU(), nn.GroupNorm(8, dim),
                nn.Conv1d(dim, dim, 5, padding=2), nn.GELU(), nn.GroupNorm(8, dim))
            self.t_down = nn.Conv1d(dim, dim, T_STRIDE, stride=T_STRIDE)
            # ⚠️ token 数**用一次 dummy 前向动态算**（手写 (L+k-1)//k 会算错：
            #    Conv1d 的真实输出是 floor((L-k)/s)+1）
            with torch.no_grad():
                dummy = torch.zeros(1, n_ch, n_times)
                self.n_tokens = int(self.t_down(self.enc(self.ch_mix(dummy))).shape[2])
            self.head = nn.Sequential(
                nn.LayerNorm(dim * self.n_tokens),
                nn.Linear(dim * self.n_tokens, 512), nn.GELU(),
                nn.Linear(512, out_dim))

        def forward(self, x):
            h = self.ch_mix(x)
            h = self.enc(h)
            h = self.t_down(h).transpose(1, 2)     # (B, T', dim)
            return self.head(h.flatten(1))

    return EEGEncoder()


def info_nce(z, t, temp):
    """对称 InfoNCE。z / t 已是 L2 归一化后的 (B, D)。"""
    import torch
    import torch.nn.functional as F
    logits = z @ t.t() / temp
    labels = torch.arange(z.shape[0], device=z.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


# ---------------------------------------------------------------------------
# 读出
# ---------------------------------------------------------------------------

def zeroshot_topk(Z, Vc, n_rep_te):
    """Z : (n_te, n_rep_te, D)；Vc : (n_te, D) 候选图像特征。"""
    n_te = Vc.shape[0]
    flat = Z.reshape(-1, Z.shape[-1]).astype(np.float64)
    fn = flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)
    cn = Vc / np.maximum(np.linalg.norm(Vc, axis=1, keepdims=True), 1e-12)
    sim = fn @ cn.T
    true = np.repeat(np.arange(n_te), n_rep_te)
    out = {"acc1_cos": float((sim.argmax(axis=1) == true).mean()),
           "acc5_cos": float((np.argsort(-sim, axis=1)[:, :5] == true[:, None]).any(axis=1).mean())}
    avg = Z.astype(np.float64).mean(axis=1)
    an = avg / np.maximum(np.linalg.norm(avg, axis=1, keepdims=True), 1e-12)
    out["acc1_featavg_cos"] = float(((an @ cn.T).argmax(axis=1) == np.arange(n_te)).mean())
    return out


def _ridge_solve(A0, B, lam, scale):
    d = A0.shape[0]
    A = A0.copy()
    A.flat[:: d + 1] += lam * scale
    try:
        return np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, B, rcond=None)[0]


def ridge_on_features(Xtr, Vtr, Xte, Vte, lam=1.0):
    """**A13 / exp92-C 口径**：特征 → ridge → 图像空间 → 余弦检索。

    Xtr : (n_img, n_rep, d)；Vtr : (n_img, D) 对应图像特征
    Xte : (n_te, n_rep_te, d)；Vte : (n_te, D) 候选图像特征
    """
    n_img, n_rep, d = Xtr.shape
    flat = Xtr.reshape(-1, d).astype(np.float64)
    mu = flat.mean(axis=0)
    sd = flat.std(axis=0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    Xbar = Xtr.mean(axis=1, dtype=np.float64)
    A0 = (flat.T @ flat) / n_rep
    scale = float(np.trace(A0)) / d
    v_mean = Vtr.mean(axis=0)
    W = _ridge_solve(A0, Xbar.T @ (Vtr - v_mean), lam, scale)
    Vte_c = Vte - v_mean
    n_te, n_rep_te = Xte.shape[0], Xte.shape[1]
    pred = Xte.reshape(-1, d) @ W
    pn = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12)
    cn = Vte_c / np.maximum(np.linalg.norm(Vte_c, axis=1, keepdims=True), 1e-12)
    cos = pn @ cn.T
    true = np.repeat(np.arange(n_te), n_rep_te)
    out = {"acc1_cos": float((cos.argmax(axis=1) == true).mean())}
    avg = pred.reshape(n_te, n_rep_te, -1).mean(axis=1)
    an = avg / np.maximum(np.linalg.norm(avg, axis=1, keepdims=True), 1e-12)
    out["acc1_featavg_cos"] = float(((an @ cn.T).argmax(axis=1) == np.arange(n_te)).mean())
    return out


def encode_all(model, eeg_flat, device, batch):
    """(N, C, T) → (N, D) 编码器输出（numpy）。"""
    import torch
    eeg_flat = np.asarray(eeg_flat)
    outs = []
    model.eval()
    with torch.no_grad():
        for s in range(0, eeg_flat.shape[0], batch):
            xb = torch.from_numpy(
                np.ascontiguousarray(eeg_flat[s:s + batch])).to(device).float()
            outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs, 0)


def eval_val(model, EEG, IMGIDX, subjects, val_img, IMG_train, device, batch):
    """train split 的 val 图上做跨图检索（**图像级验证集**，不碰 test）。"""
    import torch
    n_ch = EEG[0].shape[2]
    n_times = EEG[0].shape[3]
    n_rep = EEG[0].shape[1]
    accs = []
    for si in range(len(subjects)):
        rows = EEG[si][torch.from_numpy(val_img).long()].numpy()   # (n_val, n_rep, C, T)
        Z = encode_all(model, rows.reshape(-1, n_ch, n_times), device, batch)
        Z = Z.reshape(len(val_img), n_rep, -1)
        tgt = np.stack([IMG_train[IMGIDX[si][i]] for i in val_img])
        accs.append(zeroshot_topk(Z, tgt, n_rep))
    return {"acc1_cos": float(np.mean([a["acc1_cos"] for a in accs])),
            "acc1_featavg_cos": float(np.mean([a["acc1_featavg_cos"] for a in accs]))}


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def train(args, IMG_train, device):
    import torch
    import torch.nn.functional as F
    rng = np.random.default_rng(SEED)

    # ---------- 预检 P1：对齐（三段式证明，见 verify_alignment）----------
    log("[预检 P1] 630-d 手工特征的 train-split RSA + 图像特征行序核对")
    eeg0, img_paths0, labels0 = load_train_subject(args.train_subjects[0],
                                                    n_images=args.align_check_images)
    X = E73.per_rep_targets(np.asarray(eeg0, dtype=np.float16))
    del eeg0
    flat = X.reshape(-1, X.shape[-1])
    X = (X - flat.mean(axis=0)) / (flat.std(axis=0) + 1e-6)
    al = verify_alignment(X, img_paths0, labels0, IMG_train, rng,
                          args.align_check_perm)
    log(f"  ①路径概念号==label：{al.get('label_match_frac')} "
        f"｜行序=概念升序×10：{al.get('concept_block_order_ok')}")
    log(f"  ②特征行同块命中率：{al.get('nn_same_block_frac', 0):.4f}"
        f"｜③RSA={al.get('rsa_r', 0):+.4f}（p={al.get('rsa_p', 1):.4f}）")
    del X
    r_align, p_align = al["rsa_r"], al["rsa_p"]
    ident = al.get("concept_block_order_ok", False)
    if not (al.get("nn_same_block_frac", 0) > 0.5
            and r_align > 0 and p_align < 0.05):
        log("✗✗ 对齐预检失败 ⇒ 图像特征与 EEG 不是一一对应 ⇒ 中止，不训练。")
        return None

    # ---------- 载入训练数据 ----------
    subjects = list(args.train_subjects)
    log(f"[数据] 载入 {len(subjects)} 个被试：{subjects}")
    t0 = time.time()
    EEG, IMGIDX = [], []
    for s in subjects:
        e, _paths, _labels = load_train_subject(s)
        EEG.append(torch.from_numpy(e))
        # 位置索引（对齐已由 P1 三段式证明）
        IMGIDX.append(np.arange(len(e), dtype=np.int64))
        log(f"  sub-{s:02d}: {tuple(e.shape)}（累计 {time.time()-t0:.0f}s）")
    n_img, n_rep = EEG[0].shape[0], EEG[0].shape[1]
    n_ch, n_times = EEG[0].shape[2], EEG[0].shape[3]
    log(f"[数据] 每被试 {n_img} 图 × {n_rep} 重复 × {n_ch} 通道 × {n_times} 样本")

    perm = np.random.default_rng(SEED).permutation(n_img)
    n_val = max(1, int(args.val_frac * n_img))
    val_img = np.sort(perm[:n_val])
    log(f"[划分] fit {n_img-n_val} 图 / val {len(val_img)} 图（图像级，被试间一致）")

    # ---------- 模型 ----------
    model = build_model(n_ch=n_ch, n_times=n_times, out_dim=IMG_train.shape[1]).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    log(f"[模型] 参数量 {nparam:,}｜设备 {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * args.steps_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1))

    def sample_batch(step_rng):
        """一批**图像互不相同**的样本（避免同图不同重复变成假负例）。"""
        imgs = step_rng.choice(n_img - n_val, size=args.batch, replace=False)
        imgs = np.sort(imgs)                     # fit 集是 perm 的后段
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

    # ---------- 预检 P2：损失实现的方向（**用合成向量**，不用未训练的模型）----------
    # ⚠️ 不能用「未训练的模型 + 真配对 vs 置换配对」来检验：
    #    未训练时 z 与 t 独立 ⇒ 两种情形的损失都 ≈ log(B) ⇒ 会**误杀**。
    with torch.no_grad():
        t_syn = F.normalize(torch.randn(args.batch, IMG_train.shape[1]), dim=1)
        l_al = float(info_nce(t_syn, t_syn, args.temp))
        l_pe = float(info_nce(t_syn, t_syn[torch.randperm(len(t_syn))], args.temp))
    log(f"[预检 P2] InfoNCE 对齐={l_al:.4f} 置换={l_pe:.4f}（置换应明显更大）")
    if not (l_pe > l_al):
        log("✗✗ 置换预检失败 ⇒ 损失实现方向可能写反 ⇒ 中止")
        return None

    # ---------- 训练 ----------
    hist = []
    best = {"val_acc": -1.0, "epoch": -1}
    step_rng = np.random.default_rng(SEED)
    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for _ in range(args.steps_per_epoch):
            xb, tb = sample_batch(step_rng)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=(device.type == "cuda")):
                z = model(xb.float())
                loss = info_nce(F.normalize(z, dim=1), F.normalize(tb, dim=1), args.temp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(float(loss.item()))
        tr_loss = float(np.mean(losses))
        val = eval_val(model, EEG, IMGIDX, subjects, val_img, IMG_train,
                       device, args.batch)
        model.train()
        hist.append({"epoch": ep, "train_loss": tr_loss,
                     "val_acc1": val["acc1_cos"],
                     "val_acc1_featavg": val["acc1_featavg_cos"],
                     "elapsed_s": round(time.time() - t_start, 1)})
        if val["acc1_featavg_cos"] > best["val_acc"]:
            best = {"val_acc": val["acc1_featavg_cos"], "epoch": ep}
            torch.save({"state_dict": model.state_dict(), "args": vars(args),
                        "epoch": ep, "n_ch": n_ch, "n_times": n_times,
                        "out_dim": IMG_train.shape[1]},
                       Path(args.out_dir) / "encoder_best.pt")
        log(f"  epoch {ep:>3} | loss {tr_loss:.4f} | val acc1 {val['acc1_cos']:.4f} "
            f"featavg {val['acc1_featavg_cos']:.4f} | {time.time()-t_start:.0f}s")
        # ---------- 护栏：第 N 轮仍在随机水平 ⇒ 立刻中止（别烧几小时 GPU）----------
        if ep == args.guard_epoch and val["acc1_featavg_cos"] < args.min_val_acc1:
            log(f"✗✗ 第 {ep} 轮 val featavg={val['acc1_featavg_cos']:.4f} "
                f"仍低于阈值 {args.min_val_acc1} ⇒ 没学到东西 ⇒ 中止")
            return None
    torch.save({"state_dict": model.state_dict(), "args": vars(args),
                "epoch": args.epochs, "n_ch": n_ch, "n_times": n_times,
                "out_dim": IMG_train.shape[1]},
               Path(args.out_dir) / "encoder_last.pt")
    return {"hist": hist, "best": best, "n_img": n_img, "n_rep": n_rep,
            "n_ch": n_ch, "n_times": n_times, "nparam": nparam,
            "align_rsa": float(r_align), "align_p": float(p_align),
            "align": al,
            "train_subjects": subjects}


# ---------------------------------------------------------------------------
# 测试集评估
# ---------------------------------------------------------------------------

def eval_subject(model, sid, IMG_train, IMG_test, args, device):
    import torch
    eeg = load_test_subject(sid)
    n_img, n_rep, n_ch, T = eeg.shape
    flat = eeg.reshape(n_img * n_rep, n_ch, T)
    Z = encode_all(model, flat, device, args.batch).reshape(n_img, n_rep, -1)

    rec = {"subject": f"sub-{sid:02d}"}
    rec["zeroshot"] = zeroshot_topk(Z, IMG_test, n_rep)

    # ridge_feat（A13 口径）：需要该被试 train split 的编码器输出
    n_ridge = min(args.ridge_train_images, 16540)
    e_tr, _paths, _labels = load_train_subject(sid, n_images=n_ridge)
    tr_flat = e_tr.reshape(-1, n_ch, T)
    Ztr = encode_all(model, tr_flat, device, args.batch).reshape(n_ridge, e_tr.shape[1], -1)
    Vtr = IMG_train[:n_ridge]          # 位置对齐已由 P1 三段式证明
    rec["ridge_feat"] = ridge_on_features(Ztr, Vtr, Z, IMG_test)
    rec["ridge_feat"]["n_train_images"] = int(n_ridge)
    del e_tr, Ztr

    r, p = E89.rsa(Z, IMG_test, args.align_check_perm,
                   np.random.default_rng(SEED + sid))
    rec["rsa"] = {"r": float(r), "p": float(p)}
    return rec


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-subjects", type=int, nargs="+",
                    default=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument("--eval-subjects", type=int, nargs="+",
                    default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps-per-epoch", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--align-check-images", type=int, default=1500)
    ap.add_argument("--align-check-perm", type=int, default=100)
    ap.add_argument("--ridge-train-images", type=int, default=4000)
    ap.add_argument("--min-val-acc1", type=float, default=0.008,
                    help="第 guard-epoch 轮的 val（train-split）featavg 低于此值 ⇒ 提前中止")
    ap.add_argument("--guard-epoch", type=int, default=3,
                    help="在第几轮做「没学到东西就中止」的护栏检查")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    import torch
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"设备 {device}｜torch {torch.__version__}｜cuda 可用 {torch.cuda.is_available()}")
    if device.type != "cuda":
        log("⚠️ 没有 GPU ⇒ 训练会非常慢；建议 --flavor a10g-large 或 t4-small")

    fte = torch.load(FEAT_TEST, map_location="cpu", weights_only=False)
    IMG_test = fte["img_features"].numpy().astype(np.float64)
    ftr = torch.load(FEAT_TRAIN, map_location="cpu", weights_only=False)
    IMG_train = ftr["img_features"].numpy().astype(np.float64)
    log(f"图像特征 test {IMG_test.shape} | train {IMG_train.shape}")

    res = train(args, IMG_train, device)
    if res is None:
        log("✗ 预检未过 ⇒ 不产出结果")
        raise SystemExit(2)

    ck = torch.load(out_dir / "encoder_best.pt", map_location="cpu", weights_only=False)
    model = build_model(n_ch=ck["n_ch"], n_times=ck["n_times"], out_dim=ck["out_dim"])
    model.load_state_dict(ck["state_dict"])
    model = model.to(device).eval()
    log(f"[评测] 用 best epoch={ck['epoch']}（**按 train-split 验证集选出**，未用 test）")

    per_subject = []
    for sid in args.eval_subjects:
        try:
            rec = eval_subject(model, sid, IMG_train, IMG_test, args, device)
            per_subject.append(rec)
            log(f"  sub-{sid:02d}: zeroshot acc1={rec['zeroshot']['acc1_cos']:.4f} "
                f"featavg={rec['zeroshot']['acc1_featavg_cos']:.4f} "
                f"| ridge_feat={rec['ridge_feat']['acc1_cos']:.4f} "
                f"| RSA={rec['rsa']['r']:+.4f}")
        except Exception as e:
            per_subject.append({"subject": f"sub-{sid:02d}",
                                "error": f"{type(e).__name__}: {e}"[:300],
                                "trace": traceback.format_exc()[-800:]})
            log(f"  ✗ sub-{sid:02d}: {per_subject[-1]['error']}")

    n_err = sum(1 for r in per_subject if "error" in r)
    log(f"\n评测完成：{len(per_subject)-n_err}/{len(per_subject)} 成功"
        + (f"，**{n_err} 个报错**" if n_err else ""))
    if per_subject and n_err == len(per_subject):
        log("✗✗ 全部被试报错 ⇒ 结果不可用")

    (out_dir / "exp93b_contrastive.json").write_text(
        json.dumps({"config": vars(args), "train": res, "per_subject": per_subject},
                   indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"结果写入 {out_dir}/exp93b_contrastive.json")


def self_test():
    log("=== 自测：模型形状 + InfoNCE 方向 + 端到端可学性 ===")
    import torch
    import torch.nn.functional as F
    m = build_model(n_ch=63, n_times=250, out_dim=1024)
    with torch.no_grad():
        z = m(torch.randn(8, 63, 250))
    assert z.shape == (8, 1024), z.shape
    log(f"  ✓ 输出形状 {tuple(z.shape)}｜参数量 {sum(p.numel() for p in m.parameters()):,}")

    # 损失方向：用**合成向量**检验（未训练的模型无法区分对齐/置换，会误判）
    with torch.no_grad():
        t = F.normalize(torch.randn(8, 1024), dim=1)
        l_al = float(info_nce(t, t, 0.07))
        l_pe = float(info_nce(t, t[torch.randperm(8)], 0.07))
    log(f"  InfoNCE 对齐={l_al:.4f} 置换={l_pe:.4f}")
    assert l_pe > l_al, "置换后损失没有变大 ⇒ 方向写反"

    # 端到端：合成一个「EEG → 目标」的**可学**映射，训练若干步，损失必须明显下降
    torch.manual_seed(0)
    A = torch.randn(63, 1024) / 8.0
    x = torch.randn(64, 63, 250)
    tgt = x.mean(axis=2) @ A                 # (64, 1024)：从 EEG 可学出
    m2 = build_model(n_ch=63, n_times=250, out_dim=1024)
    opt = torch.optim.AdamW(m2.parameters(), lr=3e-3)
    l0 = None
    for i in range(120):
        loss = info_nce(F.normalize(m2(x), dim=1), F.normalize(tgt, dim=1), 0.07)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i == 0:
            l0 = float(loss)
    l1 = float(loss)
    log(f"  端到端可学性：loss {l0:.4f} → {l1:.4f}（随机水平 log(64)={math.log(64):.3f}）")
    assert l1 < l0 - 0.2, "合成可学任务上损失没明显下降 ⇒ 训练/损失实现有问题"
    log("  ✓ 损失方向正确 + 端到端能学 ⇒ 训练循环没写反")
    log("=== 自测通过 ===")


if __name__ == "__main__":
    main()
