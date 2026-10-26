#!/usr/bin/env python3
"""train_illusionbrain_eeg.py — **IllusionBrain-EEG**：EEG → CLIP 图像特征的非线性编码器。

## 这个模型要回答的问题

exp74 的结论：ViT-H-14（632 GFLOP/图）与 ViT-B-32（4.4 GFLOP/图）在
**线性 ridge** 读出下**打平**（144× 的算力买不到更好的脑解码）。

但那只回答了一半。到底是
 (a) 特征里**确实没有**更多信息（大模型也不再有），
还是 (b) 特征里有、只是**线性读出用不上**？

⇒ 把线性 ridge 换成**训练过的非线性编码器**，在**完全同一套协议**下再比一次。
本模型就是那个"训练过的非线性编码器"，它同时也是 IllusionBrain 平台
第一个自训模型。

## 模型

    EEG (630) ── LayerNorm
                ── Linear(630, 1024) ── GELU ── Dropout(0.1)
                ── Linear(1024, 1024) ── GELU ── LayerNorm
                ├─ head_vit_h14 : Linear(1024, 1024)   → CLIP ViT-H/14 空间
                └─ head_vit_b32 : Linear(1024,  512)   → CLIP ViT-B/32 空间

两个头**共享主干** ⇒ 两个特征空间的比较发生在**同一个模型内部**，
比 exp74"分别训两个 ridge"更干净。

**归一化统计量必须随权重一起存**（`norm.mu` / `norm.sd`）：
否则外部拿到模型也无法复现输入归一化，解码结果不可复现。

## 协议（与 exp73/exp74 **逐字一致**）

- 输入特征：基线校正 0–50 ms + 10 个 100 ms 窗 × 63 通道 = 630 维
  （`per_rep_targets` 与 exp73 逐字相同，脚本内有一致性自测）
- 训练集：train split 16540 图 × 4 重复；按**图像**留出 10% 作验证
  （**绝不碰 test split**；选 epoch 只看验证集）
- 评测：test split 200 图 × 80 重复 → 每个试次与 200 个候选做余弦 → argmax
  ⇒ top-1/top-5 数字**可与 exp74 的 ridge 基线逐行对照**

## 产出

- 每被试一个 checkpoint（`model.safetensors` + `pytorch_model.bin` + `config.json`）
  → 写到 `/domains/models/illusionbrain-eeg/sub-XX/`
- 汇总 JSON → `/domains/runs/tier0-encoding/exp75_train_encoder.json`

用法：
    python train_illusionbrain_eeg.py --subjects 1 2 3 --epochs 40
    python train_illusionbrain_eeg.py --smoke          # 1 被试 / 3 epoch / 冒烟
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
TRAIN_4D = Path("/domains/omnibrain4d/eeg_train")     # 16540 图 × 4 重复
TEST_4D = Path("/domains/omnibrain4d/eeg")             # 200 图 × 80 重复
FEAT_H14 = ("/training/eeg/eeg-image-decode/ViT-H-14_features_train.pt",
            "/training/eeg/eeg-image-decode/ViT-H-14_features_test.pt")
FEAT_B32 = ("/training/eeg/things-eeg/Preprocessed_data_250Hz_whiten/ViT-B-32_features_train.pt",
            "/training/eeg/things-eeg/Preprocessed_data_250Hz_whiten/ViT-B-32_features_test.pt")
OUT_ROOT = Path("/domains/models/illusionbrain-eeg")
OUT_JSON = Path("/domains/runs/tier0-encoding/exp75_train_encoder.json")

# ---------------------------------------------------------------------------
# 特征提取：**必须**与 exp73 逐字相同（脚本内有自测比对）
# ---------------------------------------------------------------------------
FS = 250.0
WIN = 25
BASELINE = 12
N_BIN = 10
N_CH = 63
D_IN = N_CH * N_BIN          # 630


def per_rep_targets(eeg, chunk=400):
    """(n_img, n_rep, n_ch, T) → (n_img, n_rep, n_ch*n_bin)，窗主序。"""
    n_img, n_rep, n_ch, T = eeg.shape
    n_bin = T // WIN
    out = np.empty((n_img, n_rep, n_ch * n_bin), dtype=np.float32)
    for s in range(0, n_img, chunk):
        e = eeg[s:s + chunk].astype(np.float32)
        e -= e[:, :, :, :BASELINE].mean(axis=3, keepdims=True)
        e = e.reshape(e.shape[0], n_rep, n_ch, n_bin, WIN).mean(axis=4)
        e = e.transpose(0, 1, 3, 2)
        out[s:s + chunk] = e.reshape(e.shape[0], n_rep, n_ch * n_bin)
        del e
    return out


# ---------------------------------------------------------------------------
# 特征文件读取（两种形态：矩阵 / 路径字典）
# ---------------------------------------------------------------------------
PATH_RE = re.compile(
    r"^(?P<split>[A-Za-z_]*images)/(?P<cid>\d+)_(?P<cname>[^/]+)/(?P<file>[^/]+)$")


def load_matrix(path, expect_rows):
    import torch
    f = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(f, dict):
        key = None
        for k in ("img_features", "features", "image_features"):
            v = f.get(k)
            if hasattr(v, "shape") and len(v.shape) == 2 and v.shape[0] == expect_rows:
                key = k
                break
        if key is None:
            for k, v in f.items():
                if hasattr(v, "shape") and len(v.shape) == 2 and v.shape[0] == expect_rows:
                    key = k
                    break
        if key is None:
            raise KeyError(f"{path}: 找不到 {expect_rows} 行的特征矩阵；键={list(f)[:8]}")
        arr = f[key].numpy()
    else:
        arr = f.numpy()
    arr = arr.astype(np.float64)
    if arr.shape[0] != expect_rows:
        raise ValueError(f"{path}: 行数 {arr.shape[0]} ≠ {expect_rows}")
    return arr


def path_dict_to_matrix(d, expect_rows, label=""):
    """路径字典 → 矩阵；行序由 (概念号, 文件名) 排序推导（= 桶内清单顺序）。"""
    if len(d) != expect_rows:
        raise ValueError(f"{label}: 图像数 {len(d)} ≠ {expect_rows}")
    items = []
    for k, v in d.items():
        m = PATH_RE.match(str(k))
        if not m:
            raise ValueError(f"{label}: 图像路径格式不对：{k!r}")
        items.append((int(m.group("cid")), m.group("file"), str(k), v))
    items.sort(key=lambda t: (t[0], t[1]))
    cids = sorted({t[0] for t in items})
    if cids != list(range(1, len(cids) + 1)):
        raise ValueError(f"{label}: 概念号不连续")
    per = {}
    for cid, _f, _k, _v in items:
        per[cid] = per.get(cid, 0) + 1
    if len(set(per.values())) != 1:
        raise ValueError(f"{label}: 各概念图数不一致")
    arr = np.stack([t[3].numpy() for t in items]).astype(np.float64)
    return arr, {"n_concepts": len(cids), "per_concept": per[cids[0]],
                 "first_key": items[0][2], "last_key": items[-1][2]}


def load_path_dict(path, expect_rows):
    import torch
    f = torch.load(path, map_location="cpu", weights_only=False)
    d = None
    for k in ("image_features", "img_features", "features"):
        if isinstance(f.get(k), dict):
            d = f[k]
            break
    if d is None:
        raise KeyError(f"{path}: 找不到 image_features 路径字典；键={list(f)[:8]}")
    return path_dict_to_matrix(d, expect_rows, label=str(path))


# ---------------------------------------------------------------------------
# safetensors：自写序列化（环境里没有 safetensors，且不想装依赖）
# 格式：<u64 LE header_len><JSON header><raw data>
# ---------------------------------------------------------------------------
def save_safetensors(path, tensors, metadata=None):
    header, offset, blobs = {}, 0, []
    for name, t in tensors.items():
        t = t.detach().to("cpu").contiguous()
        if t.dtype == torch_f32():
            dtype = "F32"
        elif t.dtype == torch_f64():
            dtype = "F64"
        else:
            raise TypeError(f"暂只支持 F32/F64，收到 {t.dtype}")
        raw = t.numpy().tobytes()
        header[name] = {"dtype": dtype, "shape": list(t.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for b in blobs:
            fh.write(b)
    os.replace(tmp, path)


def load_safetensors(path):
    """读回来做自校验。"""
    import torch
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n).decode("utf-8"))
        base = 8 + n
        out = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            s, e = meta["data_offsets"]
            fh.seek(base + s)
            raw = fh.read(e - s)
            dt = torch.float32 if meta["dtype"] == "F32" else torch.float64
            out[name] = torch.frombuffer(bytearray(raw), dtype=dt).reshape(meta["shape"])
        return out, header.get("__metadata__", {})


def torch_f32():
    import torch
    return torch.float32


def torch_f64():
    import torch
    return torch.float64


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
def build_model(hidden=1024, dropout=0.1):
    import torch.nn as nn
    from torch import nn as _nn

    class IllusionBrainEEG(_nn.Module):
        """EEG(630) → 两个 CLIP 图像特征空间。"""

        def __init__(self, d_in=D_IN, d_h=hidden, d_h14=1024, d_b32=512, p=dropout):
            super().__init__()
            self.cfg = {"d_in": d_in, "d_hidden": d_h, "d_h14": d_h14,
                        "d_b32": d_b32, "dropout": p}
            self.in_norm = _nn.LayerNorm(d_in)
            self.fc1 = _nn.Linear(d_in, d_h)
            self.fc2 = _nn.Linear(d_h, d_h)
            self.out_norm = _nn.LayerNorm(d_h)
            self.drop = _nn.Dropout(p)
            self.act = _nn.GELU()
            self.head_h14 = _nn.Linear(d_h, d_h14)
            self.head_b32 = _nn.Linear(d_h, d_b32)

        def forward(self, x):
            x = x.to(self.fc1.weight.dtype)      # 外部喂进来的 dtype 不一定一致
            h = self.in_norm(x)
            h = self.act(self.fc1(h))
            h = self.drop(h)
            h = self.act(self.fc2(h))
            h = self.out_norm(h)
            return self.head_h14(h), self.head_b32(h)

    return IllusionBrainEEG()


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_loss(pred, tgt):
    """1 - cos(pred, tgt)；tgt 已 L2 归一。"""
    import torch
    p = torch.nn.functional.normalize(pred, dim=-1)
    return (1.0 - (p * tgt).sum(-1)).mean()


def train_subject(sid, feats, args, device):
    """返回 (best_state, history, test_results)。"""
    import torch
    from torch import nn as _nn

    t0 = time.time()
    tr_fp = TRAIN_4D / f"sub-{sid:02d}.npz"
    te_fp = TEST_4D / f"sub-{sid:02d}.npz"
    if not tr_fp.exists() or not te_fp.exists():
        return None, None, None, None

    # ---- 数据 ----
    eeg = np.load(tr_fp)["eeg"]
    Xrep = per_rep_targets(eeg)
    del eeg
    mu = Xrep.reshape(-1, D_IN).mean(axis=0)
    sd = Xrep.reshape(-1, D_IN).std(axis=0) + 1e-6
    Xrep = (Xrep - mu) / sd
    n_img, n_rep, _ = Xrep.shape

    VH_tr, VB_tr = feats["ViT-H-14"][0], feats["ViT-B-32"][0]
    VH_te, VB_te = feats["ViT-H-14"][1], feats["ViT-B-32"][1]

    # ---- 按**图像**划分 train/val（试次不能跨划分）----
    rng = np.random.default_rng(args.seed + sid)
    perm = rng.permutation(n_img)
    n_val = max(1, int(round(args.val_frac * n_img)))
    val_imgs = np.sort(perm[:n_val])
    tr_imgs = np.sort(perm[n_val:])
    tr_imgs_t = torch.as_tensor(tr_imgs, device=device)
    val_imgs_t = torch.as_tensor(val_imgs, device=device)

    Xtr = torch.from_numpy(Xrep[tr_imgs]).to(device)          # (m,4,630)
    Xval = torch.from_numpy(Xrep[val_imgs]).to(device)
    del Xrep
    m = Xtr.shape[0]
    VH_tr_t = torch.from_numpy(VH_tr).to(device).float()
    VB_tr_t = torch.from_numpy(VB_tr).to(device).float()
    VH_tr_t = torch.nn.functional.normalize(VH_tr_t, dim=-1)
    VB_tr_t = torch.nn.functional.normalize(VB_tr_t, dim=-1)
    # 目标也按**训练子集**重排：训练循环里的 im 是子集内下标，
    # 直接拿它去索引全量目标会静默取错图（这种错不会报错，只会给出错的数）。
    VH_tr_sub = VH_tr_t[tr_imgs_t]
    VB_tr_sub = VB_tr_t[tr_imgs_t]

    # ---- 模型 ----
    model = build_model(hidden=args.hidden, dropout=args.dropout).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(1, (m * n_rep) // args.batch)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(0.05 * total_steps))

    def lr_at(step):
        if step < warmup:
            return step / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    best = {"loss": float("inf"), "epoch": -1, "state": None}
    history = []
    step = 0
    for ep in range(args.epochs):
        model.train()
        # 打乱（图像, 重复）两级索引
        order = torch.randperm(m * n_rep, device=device)
        run = 0.0
        nb = 0
        for b in range(steps_per_epoch):
            idx = order[b * args.batch:(b + 1) * args.batch]
            if idx.numel() == 0:
                break
            im = idx // n_rep
            rp = idx % n_rep
            xb = Xtr[im, rp]                                    # (B,630)
            th = VH_tr_sub[im]
            tb = VB_tr_sub[im]
            ph, pb = model(xb)
            loss = cosine_loss(ph, th) + cosine_loss(pb, tb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            run += float(loss.item())
            nb += 1
        tr_loss = run / max(1, nb)

        # ---- 验证（**只看留出的 train 图像**）----
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val_imgs_t), 256):
                im = val_imgs_t[i:i + 256]
                xb = Xval[im].reshape(-1, D_IN)
                ph, pb = model(xb)
                th = VH_tr_t[im].repeat_interleave(n_rep, dim=0)
                tb = VB_tr_t[im].repeat_interleave(n_rep, dim=0)
                vl += float((cosine_loss(ph, th) + cosine_loss(pb, tb)).item()) * xb.shape[0]
                vn += xb.shape[0]
        val_loss = vl / max(1, vn)
        history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": val_loss})
        improved = val_loss < best["loss"] - 1e-5
        if improved:
            best = {"loss": val_loss, "epoch": ep,
                    "state": {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}}
        if ep % max(1, args.epochs // 10) == 0 or ep == args.epochs - 1:
            print(f"  sub-{sid:02d} ep{ep:3d} train {tr_loss:.4f} val {val_loss:.4f}"
                  f"{'  *' if improved else ''}  [{time.time()-t0:.0f}s]", flush=True)
        if ep - best["epoch"] >= args.patience:
            print(f"  sub-{sid:02d} 早停（patience {args.patience}）", flush=True)
            break

    if best["state"] is None:
        best = {"loss": float("inf"), "epoch": -1,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    # ---- test 评测（与 exp73/74 同口径）----
    eeg = np.load(te_fp)["eeg"]
    Xte = per_rep_targets(eeg)
    del eeg
    Xte = (Xte - mu) / sd
    Xte_t = torch.from_numpy(Xte).to(device)
    del Xte
    model.load_state_dict(best["state"])
    model.eval()
    res = {}
    with torch.no_grad():
        ph_all, pb_all = [], []
        for i in range(0, Xte_t.shape[0], 64):
            xb = Xte_t[i:i + 64].reshape(-1, D_IN)
            ph, pb = model(xb)
            ph_all.append(ph)
            pb_all.append(pb)
        ph_all = torch.cat(ph_all).reshape(Xte_t.shape[0], Xte_t.shape[1], -1)
        pb_all = torch.cat(pb_all).reshape(Xte_t.shape[0], Xte_t.shape[1], -1)
    res = eval_200way(ph_all, pb_all, VH_te, VB_te, device)
    res["n_param"] = int(n_param)
    res["best_epoch"] = int(best["epoch"])
    res["best_val_loss"] = float(best["loss"])
    res["epochs_run"] = len(history)
    res["elapsed_s"] = round(time.time() - t0, 1)
    return best["state"], history, res, (mu, sd)


def eval_200way(ph, pb, VH_te, VB_te, device):
    """ph/pb: (n_img_te, n_rep, d)；候选: (n_img_te, d)。⇒ 200 选 1。"""
    import torch
    out = {}
    n_te, n_rep = ph.shape[0], ph.shape[1]
    true = torch.arange(n_te, device=device).repeat_interleave(n_rep)
    for tag, P, C in (("h14", ph, VH_te), ("b32", pb, VB_te)):
        Ct = torch.from_numpy(C).to(device).float()
        Pn = torch.nn.functional.normalize(P.reshape(-1, P.shape[-1]), dim=-1)
        Cn = torch.nn.functional.normalize(Ct, dim=-1)
        sim = Pn @ Cn.T
        pick = sim.argmax(dim=1)
        out[f"acc1_{tag}"] = float((pick == true).float().mean())
        top5 = sim.topk(5, dim=1).indices
        out[f"acc5_{tag}"] = float((top5 == true[:, None]).any(dim=1).float().mean())
        # 80 次重复平均后
        avg = torch.nn.functional.normalize(P.mean(dim=1), dim=-1)
        out[f"acc1_featavg_{tag}"] = float((avg @ Cn.T).argmax(dim=1)
                                            .eq(torch.arange(n_te, device=device))
                                            .float().mean())
    return out


# ---------------------------------------------------------------------------
# 自测（合成数据；本地 CPU 可跑）
# ---------------------------------------------------------------------------
def self_test():
    import torch
    print("=== 自测 ===")

    # (1) 特征提取必须与 exp73 **逐位**一致（否则与 exp73/74 的数字不可比）
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import exp73_decode200 as E73
        rng = np.random.default_rng(0)
        eeg = rng.normal(size=(5, 3, 63, 250)).astype(np.float16)
        a = per_rep_targets(eeg)
        b = E73.per_rep_targets(eeg)
        assert np.array_equal(a, b), "特征提取与 exp73 不一致"
        assert a.shape == (5, 3, 630), a.shape
        print("  (1) per_rep_targets 与 exp73 逐位一致，形状 (5,3,630) ✓")
    except ImportError:
        print("  (1) 跳过：同目录没有 exp73_decode200.py")

    # (2) safetensors 往返 + 结构核对
    t = {"w": torch.arange(12, dtype=torch.float32).reshape(3, 4),
         "mu": torch.linspace(-1, 1, 7)}
    p = Path(os.environ.get("TEMP", ".")) / "_selftest.safetensors"
    save_safetensors(p, t, {"k": "v"})
    back, meta = load_safetensors(p)
    assert set(back) == set(t)
    for k in t:
        assert torch.allclose(back[k], t[k]), k
    assert meta.get("k") == "v"
    raw = open(p, "rb").read()
    n = struct.unpack("<Q", raw[:8])[0]
    hdr = json.loads(raw[8:8 + n].decode("utf-8"))
    assert hdr["w"]["dtype"] == "F32" and hdr["w"]["shape"] == [3, 4], hdr
    p.unlink(missing_ok=True)
    print("  (2) safetensors 往返 + 头部结构 ✓")

    # (3) 端到端：合成一个**可学**的映射，训练后 val 损失必须下降；
    #     并且**目标行打乱后损失必须不降**（负对照 ⇒ 这个测试真能抓住索引错）
    rng = np.random.default_rng(1)
    n_img, n_rep, k_lat = 80, 4, 12
    C = rng.normal(size=(D_IN, k_lat)) / np.sqrt(k_lat)
    z = rng.normal(size=(n_img, k_lat))
    base = (C @ z.T).T                                   # (n_img, 630)，落在 12 维子空间
    X = (base[:, None, :] + 0.02 * rng.normal(size=(n_img, n_rep, D_IN))).astype(np.float32)
    Ah = rng.normal(size=(1024, k_lat)) / np.sqrt(k_lat)
    Ab = rng.normal(size=(512, k_lat)) / np.sqrt(k_lat)
    vh = Ah @ z.T
    vb = Ab @ z.T
    vh /= np.linalg.norm(vh, axis=0, keepdims=True)
    vb /= np.linalg.norm(vb, axis=0, keepdims=True)
    vh, vb = vh.T, vb.T                     # (n_img, d)

    class _A:
        seed = 0; val_frac = 0.2; epochs = 60; batch = 64; lr = 4e-3
        wd = 1e-2; hidden = 256; dropout = 0.0; patience = 999
    _run_synthetic_train(X, vh, vb, _A, "cpu")
    print("=== 自测通过 ===")


def _run_synthetic_train(X, vh, vb, A, device):
    """与 train_subject 同构的合成版：验证训练循环 + 目标索引 + 评测接线。"""
    import torch
    n_img, n_rep, _ = X.shape
    mu = X.reshape(-1, D_IN).mean(axis=0)
    sd = X.reshape(-1, D_IN).std(axis=0) + 1e-6
    Xn = (X - mu) / sd
    rng = np.random.default_rng(A.seed)
    perm = rng.permutation(n_img)
    n_val = max(1, int(round(A.val_frac * n_img)))
    val_imgs = np.sort(perm[:n_val])
    tr_imgs = np.sort(perm[n_val:])
    tr_imgs_t = torch.as_tensor(tr_imgs)
    val_imgs_t = torch.as_tensor(val_imgs)
    Xtr = torch.from_numpy(Xn[tr_imgs])
    Xval = torch.from_numpy(Xn[val_imgs])
    VH = torch.nn.functional.normalize(torch.from_numpy(vh).float(), dim=-1)
    VB = torch.nn.functional.normalize(torch.from_numpy(vb).float(), dim=-1)
    VH_sub, VB_sub = VH[tr_imgs_t], VB[tr_imgs_t]

    model = build_model(hidden=A.hidden, dropout=A.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=A.lr, weight_decay=A.wd)
    m = Xtr.shape[0]
    steps = max(1, (m * n_rep) // A.batch)
    total = steps * A.epochs
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))

    def val_loss_with(VH_t, VB_t):
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val_imgs_t), 256):
                j = val_imgs_t[i:i + 256]                  # 原始图像号
                xb = Xval[i:i + 256].reshape(-1, D_IN)     # 子集位置（与 j 同序）
                ph, pb = model(xb)
                th = VH_t[j].repeat_interleave(n_rep, dim=0)
                tb = VB_t[j].repeat_interleave(n_rep, dim=0)
                vl += float((cosine_loss(ph, th) + cosine_loss(pb, tb)).item()) * xb.shape[0]
                vn += xb.shape[0]
        model.train()
        return vl / max(1, vn)

    val_loss = lambda: val_loss_with(VH, VB)

    v0 = val_loss()
    step = 0
    for ep in range(A.epochs):
        order = torch.randperm(m * n_rep)
        for b in range(steps):
            idx = order[b * A.batch:(b + 1) * A.batch]
            if idx.numel() == 0:
                break
            im, rp = idx // n_rep, idx % n_rep
            xb = Xtr[im, rp]
            ph, pb = model(xb)
            loss = cosine_loss(ph, VH_sub[im]) + cosine_loss(pb, VB_sub[im])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
    v1 = val_loss()
    print(f"  (3) 合成训练：val 损失 {v0:.4f} → {v1:.4f}", flush=True)
    assert v1 < v0 * 0.6, f"合成可学映射没学会（val {v0:.4f}→{v1:.4f}），训练/索引有问题"

    # ---- 负对照：把目标行打乱，损失必须**不降** ⇒ 证明这个测试对索引敏感 ----
    perm = torch.randperm(n_img)
    v_perm = val_loss_with(VH[perm], VB[perm])
    print(f"      负对照（目标行打乱）：val 损失 {v_perm:.4f}"
          f"（应远高于 {v1:.4f}）", flush=True)
    assert v_perm > v1 * 1.5, "打乱目标后损失没上升 ⇒ 这个测试对索引不敏感（白测）"

    # 评测接线：预测 = 目标 ⇒ 候选 argmax 必须 100% 命中
    # ⚠️ eval_200way 的候选参数收 **numpy**（内部 torch.from_numpy），
    #    预测参数收 **torch** —— 之前这里把 numpy 喂给了 torch 的 normalize。
    model.eval()
    with torch.no_grad():
        ph, pb = model(Xval.reshape(-1, D_IN))
    ph = ph.reshape(len(val_imgs), n_rep, -1)
    pb = pb.reshape(len(val_imgs), n_rep, -1)
    vh_val = VH[val_imgs_t]          # 已归一化
    vb_val = VB[val_imgs_t]
    r = eval_200way(vh_val.unsqueeze(1).repeat(1, n_rep, 1),
                    vb_val.unsqueeze(1).repeat(1, n_rep, 1),
                    vh_val.cpu().numpy(), vb_val.cpu().numpy(), device)
    assert r["acc1_h14"] == 1.0 and r["acc5_h14"] == 1.0, r
    assert r["acc1_featavg_h14"] == 1.0, r
    print("  (4) eval_200way 接线：自比对 100% ✓")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--subjects", type=str, default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="1 被试 / 3 epoch / 不保存")
    ap.add_argument("--save", action="store_true", help="保存 checkpoint 到桶内")
    args = ap.parse_args()
    if args.smoke:
        args.subjects, args.epochs, args.patience = "1", 3, 99
    if args.self_test:
        self_test()
        return

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__} "
          f"cuda={torch.cuda.is_available()}", flush=True)
    if device == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    set_seed(args.seed)

    t0 = time.time()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # ---- 特征（两个模型）----
    feats = {
        "ViT-H-14": (load_matrix(FEAT_H14[0], 16540), load_matrix(FEAT_H14[1], 200)),
    }
    b_tr, b_meta = load_path_dict(FEAT_B32[0], 16540)
    b_te, b_meta_te = load_path_dict(FEAT_B32[1], 200)
    feats["ViT-B-32"] = (b_tr, b_te)
    print(f"特征: H-14 {feats['ViT-H-14'][0].shape} | B-32 {b_tr.shape} "
          f"（路径字典：{b_meta['n_concepts']} 概念 × {b_meta['per_concept']} 图）", flush=True)

    sids = [int(s) for s in args.subjects.split(",") if s.strip()]
    rows = []
    for sid in sids:
        state, hist, res, norm = train_subject(sid, feats, args, device)
        if res is None:
            print(f"  sub-{sid:02d}: 数据缺失，跳过", flush=True)
            continue
        mu, sd = norm
        print(f"  sub-{sid:02d} [H-14] acc1={res['acc1_h14']:.4f} acc5={res['acc5_h14']:.4f} "
              f"avg={res['acc1_featavg_h14']:.4f} | [B-32] acc1={res['acc1_b32']:.4f} "
              f"acc5={res['acc5_b32']:.4f} avg={res['acc1_featavg_b32']:.4f} "
              f"| 参数量 {res['n_param']} | {res['elapsed_s']}s", flush=True)
        rows.append({"subject": f"sub-{sid:02d}", **res})
        if args.save and not args.smoke:
            save_subject(sid, state, mu, sd, res, args)
        OUT_JSON.write_text(json.dumps({"per_subject": rows, "args": vars(args)},
                                       indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 汇总 ----
    summary = {}
    for tag, label in (("h14", "ViT-H-14"), ("b32", "ViT-B-32")):
        a = [r[f"acc1_{tag}"] for r in rows]
        if not a:
            continue
        summary[label] = {
            "n_subjects": len(a),
            "acc1_median": float(np.median(a)),
            "acc1_min": float(np.min(a)), "acc1_max": float(np.max(a)),
            "ratio_vs_random": float(np.median(a) / 0.005),
            "acc5_median": float(np.median([r[f"acc5_{tag}"] for r in rows])),
            "featavg_median": float(np.median([r[f"acc1_featavg_{tag}"] for r in rows])),
            "per_subject": {r["subject"]: r[f"acc1_{tag}"] for r in rows},
        }
    if len(rows) >= 2:
        diffs = [r["acc1_h14"] - r["acc1_b32"] for r in rows]
        k = int((np.asarray(diffs) > 0).sum())
        n = len(diffs)
        tot = 2 ** n
        p_obs = math.comb(n, k) / tot
        p_exact = min(1.0, sum(math.comb(n, i) / tot for i in range(n + 1)
                               if math.comb(n, i) / tot <= p_obs + 1e-15))
        summary["paired_H_minus_B"] = {
            "median_diff": float(np.median(diffs)),
            "mean_diff": float(np.mean(diffs)),
            "n_pos": k, "n_neg": n - k, "p_exact_two_sided": float(p_exact),
            "per_subject_diff": {r["subject"]: d for r, d in zip(rows, diffs)},
        }
    print("\n" + "=" * 84, flush=True)
    print("汇总：训练过的非线性编码器（对照 exp74 的线性 ridge）", flush=True)
    print("=" * 84, flush=True)
    for lab, s in summary.items():
        if lab == "paired_H_minus_B":
            continue
        print(f"  {lab:10s}: acc1 中位 {s['acc1_median']:.4f} "
              f"({s['ratio_vs_random']:.1f}× 随机) 范围 {s['acc1_min']:.4f}–{s['acc1_max']:.4f} "
              f"| top5 {s['acc5_median']:.4f} | 80次平均 {s['featavg_median']:.4f}", flush=True)
    p = summary.get("paired_H_minus_B")
    if p:
        print(f"  配对 H−B: 中位 {p['median_diff']:+.4f} 均值 {p['mean_diff']:+.4f} "
              f"| 正 {p['n_pos']} 负 {p['n_neg']} 精确双侧 p={p['p_exact_two_sided']:.4f}", flush=True)
    OUT_JSON.write_text(json.dumps({"summary": summary, "per_subject": rows,
                                    "args": vars(args)}, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\n耗时 {time.time()-t0:.0f}s", flush=True)


def save_subject(sid, state, mu, sd, res, args):
    """写一个被试的 checkpoint（house 格式）。"""
    import torch
    d = OUT_ROOT / f"sub-{sid:02d}"
    d.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.float() for k, v in state.items()}
    tensors["norm.mu"] = torch.from_numpy(mu.astype(np.float32))
    tensors["norm.sd"] = torch.from_numpy(sd.astype(np.float32))
    meta = {
        "format": "pt",
        "architecture": "IllusionBrainEEG",
        "subject": f"sub-{sid:02d}",
        "d_in": D_IN, "hidden": args.hidden, "dropout": args.dropout,
        "n_param": res["n_param"],
        "input": "EEG 63ch x 10 windows(100ms) = 630, baseline-corrected 0-50ms, "
                 "z-scored with norm.mu/norm.sd",
        "heads": "vit_h14(1024) + vit_b32(512), cosine loss",
        "test_acc1_h14": res["acc1_h14"], "test_acc1_b32": res["acc1_b32"],
        "best_epoch": res["best_epoch"], "best_val_loss": res["best_val_loss"],
    }
    save_safetensors(d / "model.safetensors", tensors, meta)
    torch.save({"state_dict": state, "norm_mu": mu.astype(np.float32),
                "norm_sd": sd.astype(np.float32), "config": meta},
               d / "pytorch_model.bin")
    (d / "config.json").write_text(json.dumps({
        "architecture": "IllusionBrainEEG", "d_in": D_IN, "d_hidden": args.hidden,
        "d_out_h14": 1024, "d_out_b32": 512, "dropout": args.dropout,
        "n_param": res["n_param"], "subject": f"sub-{sid:02d}",
        "input": meta["input"], "targets": ["clip_vit_h14", "clip_vit_b32"],
        "metric": "200-way-1 top-1 on THINGS-EEG test split (200 images x 80 reps)",
        "test_acc1_h14": res["acc1_h14"], "test_acc1_b32": res["acc1_b32"],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    # 自校验：safetensors 能读回来，且与 state_dict 一致
    back, _ = load_safetensors(d / "model.safetensors")
    assert set(back) == set(tensors), "safetensors 键集合不一致"
    for k, v in tensors.items():
        assert torch.allclose(back[k], v), f"safetensors 回读不一致: {k}"
    print(f"  已保存并自校验: {d}", flush=True)


if __name__ == "__main__":
    main()
