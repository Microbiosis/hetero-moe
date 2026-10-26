#!/usr/bin/env python3
"""exp81_tokenizer_eval.py — S1：自训 EEG 离散 tokenizer 的评测（**干净协议**）。

## 口径纪律（这一版和基线表的关系，已按 §十九 的结论更正）

⚠️ **这里曾写错过，保留说明以免再犯**：本节原来断言
「`/domains/omnibrain4d/eeg_train/` 装的是 test split ⇒ 指标 B 泄漏」——
那是**一次除法算错**造成的（把 66,160 写成 132,320，少除了 float16 的 2）。
exp85 用**数值级溯源**（不靠算术）复核后的结论是：

| 目录 | 实际内容 | 证据 |
|---|---|---|
| `/domains/omnibrain4d/eeg_train/sub-XX.npz` | **train split (16540, 4)** | 2,084,040,260 ÷ (63×250×2) = 66,160 = 16540×4 |
| `/domains/omnibrain4d/eeg/sub-XX.npz` | **test split (200, 80)** | 200 张**不在** train split 里 |

⇒ 拟合集（train split）与解码集（test split）**本就不重叠 ⇒ 指标 B 从来没有泄漏**
⇒ **基线表与阶段 D 变体表的指标 B 不需要重做**，
⇒ 本脚本的指标 B 数字**可以**和基线表直接比（口径相同）。

本脚本仍然用**直接的 train split**（`train.pt`）当拟合集，而不是 npz：
少一层目录名的坑（`eeg_train/` 这个名字本身就是误导源）。
旁证（exp84）：用真正 train split 重算 630-d 那一行得 **acc1 = 0.0111（2.3× 随机）**，
与基线表原协议的 **0.0111（2.2×）** 几乎一样 ⇒ 两个协议本来就都是干净的。

| | 拟合 | 解码 |
|---|---|---|
| 指标 A（RSA） | —（不训练） | test split 200 图 × 80 试次（与基线表同口径） |
| 指标 B（ridge 200 选 1） | **train split**（`train.pt` 前 N 图 × 4 试次，16540 张里取） | test split 200 图 |

## 评测对象与读出

自训 tokenizer 产出**每时间片一个 token**（31 个），与 braintokenizer 的
「19 位置 × 512 时间步」布局不同 ⇒ 读出必须重新定义：

| 读出 | 构造 | 维度 |
|---|---|---|
| `emb_mean` | 码本向量按 token 平均 | d=64（**主读出**） |
| `onehot_mean` | 每个 token 的 one-hot 码按 token 平均 | K=1024（纯离散） |
| `code_ids` | 31 个码号本身 | 31（码号顺序是任意的 ⇒ 预期无意义，作负对照） |
| `continuous` | **不量化**，同一 encoder 的特征按 token 平均 | 64（**对照 A**） |
| `random_codebook` | 同一 encoder + **随机码本**量化 | 64（**对照 B**） |

## 判据（预先写死）

1. `emb_mean` 的指标 A 显著（p<0.05），且不低于 braintokenizer 的 +0.0155；
2. 码本利用率 > 50%；
3. 指标 B（干净协议）高于随机；
4. `emb_mean` > `continuous`（量化有用）与 `emb_mean` > `random_codebook`（要靠训练）。

## 用法

    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp81_tokenizer_eval.py \
      -v hf://buckets/illusion-brain/training-data:/training:ro \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor cpu-upgrade --timeout 90m -d \
      --ckpt /domains/runs/illusionbrain-align/exp80/tokenizer_sub-01.pt
"""

from __future__ import annotations

import argparse
import json
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

import eval_brain_alignment as EVAL   # noqa: E402

TRAIN_PT = Path("/training/eeg/things-eeg/Preprocessed_data_250Hz_whiten")
TEST_4D = Path("/domains/omnibrain4d/eeg")
FEAT_TEST = "/training/eeg/eeg-image-decode/ViT-H-14_features_test.pt"
FEAT_TRAIN = "/training/eeg/eeg-image-decode/ViT-H-14_features_train.pt"
OUT_DIR = Path("/domains/runs/illusionbrain-align/exp81")
SEED = 0

READOUTS = ("recon", "emb_mean", "onehot_mean", "code_ids", "continuous",
            "random_codebook")


def log(*a):
    print(*a, flush=True)


def load_model(ckpt_path, seed=SEED):
    """加载自训 tokenizer + 造一个随机码本（对照 B）。

    ⚠️ **按 checkpoint 里的元信息自动选架构**：
      - `arch == "base"`      ⇒ exp80 的 EEGTokenizer（S1a）
      - `arch == "assembled"` ⇒ exp86 的组装模型（组件级组装的产物）
    ⇒ 两种产物共用同一个评测脚本、同一套读出口径、同一份判据。
    """
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ck.get("arch", "base")
    if arch.startswith("assembled"):
        from exp86_assembled_train import (build_assembled, extract_attn_state,
                                           extract_ffn_donor)
        a = ck.get("args", {})
        attn_name = a.get("attn_donor", "signal-jepa")
        # ⚠️ `attn_donor == "none"` ⇒ 那个 checkpoint **没有**注意力块
        #    （同配方的「不嫁接注意力」对照臂）⇒ 不能去读供体文件
        attn = None if attn_name == "none" else extract_attn_state(attn_name)
        ffn = None
        if ck.get("donor_ffn", "none") != "none":
            ffn = extract_ffn_donor(ck["donor_ffn"])
        heads = "v2" if arch in ("assembled_v2", "assembled_v3") else "v1"
        # ⚠️ `assembled_v3`（S1b-v3 共享词典）用的是**梯度版码本** `VQShared`
        #    ⇒ 评测时必须用同一个类重建，否则 state_dict 键位/结构对不上。
        #    （键名与 exp80 的 VQ 一致，但类不同 ⇒ 不能只靠键名自动兼容）
        vq_factory = None
        if arch == "assembled_v3" and ck.get("vq") == "grad":
            from exp87_shared_vocab import build_vqshared
            a = ck.get("args", {})
            vq_factory = build_vqshared(
                balance=(a.get("balance", "none") == "sinkhorn"),
                eps_bal=float(a.get("eps_bal", 0.05)))
        model = build_assembled(n_codes=int(ck["n_codes"]), attn_state=attn,
                                ffn_donor=ffn, heads=heads, vq_factory=vq_factory)
        log(f"  架构：{arch}（注意力 {attn_name} / "
            f"FFN {ck.get('donor_ffn')} / 码本 {ck.get('vq', 'ema')}）")
    else:
        from exp80_eeg_tokenizer_train import EEGTokenizer
        model = EEGTokenizer(n_codes=int(ck["n_codes"]))
        log("  架构：base")
    model.load_state_dict(ck["state_dict"])
    model.eval()
    g = torch.Generator().manual_seed(seed)
    rnd = torch.randn(int(ck["n_codes"]), model.vq.dim, generator=g) * 0.5
    return model, ck, rnd


def readouts_for(model, x, rnd_codebook):
    """一批 EEG → 各读出的池化特征。"""
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        z = model.encode(x)                                  # (B, T', D)
        zq, _loss, idx = model.vq(z)                         # zq: (B,T',D)
        codes = idx                                          # (B, T')
        K = model.vq.n_codes
        # 对照 B：随机码本
        flat = z.reshape(-1, z.shape[-1])
        d = (flat.pow(2).sum(1, keepdim=True) - 2 * flat @ rnd_codebook.t()
             + rnd_codebook.pow(2).sum(1))
        r_idx = d.argmin(1).view(codes.shape)
        # 随机码本对应的「嵌入」就是随机码本向量本身
        r_emb = rnd_codebook[r_idx]                          # (B,T',D)
        emb = model.vq.lookup(codes)                         # (B,T',D)
        onehot = F.one_hot(codes, K).float()                 # (B,T',K)
        # 重建层：解码器重建的 EEG（**与 braintokenizer 那一行同口径**：
        #          它的最终输出就是重建信号 (B,19,512)，沿通道轴池化 ⇒ (B,512)）
        # ⚠️ `model.decode` 内部**已经**做了一次 transpose ⇒ 这里不能再转
        #    （转两次会得到 (B,T',D) ⇒ ConvTranspose1d 报「31 channels instead of 64」）
        rec = model.decode(zq, x.shape[-1])                   # (B, C, T)
        out = {
            "emb_mean": emb.mean(dim=1),                     # (B, D)
            "onehot_mean": onehot.mean(dim=1),               # (B, K)
            "code_ids": codes.float(),                       # (B, T')
            "continuous": z.mean(dim=1),                     # (B, D)
            "random_codebook": r_emb.mean(dim=1),            # (B, D)
            "recon": rec.mean(dim=1),                        # (B, T)
        }
    return out


def extract(model, eeg, rnd_cb, batch_size=64, device="cpu"):
    """(n_img, n_rep, C, T) → ({readout: (n_img, n_rep, D)}, 码号直方图)"""
    import torch
    n_img, n_rep = eeg.shape[0], eeg.shape[1]
    flat = eeg.reshape(n_img * n_rep, eeg.shape[2], eeg.shape[3])
    acc = {k: [] for k in READOUTS}
    K = model.vq.n_codes
    hist = np.zeros(K, dtype=np.int64)
    for s in range(0, flat.shape[0], batch_size):
        sub = flat[s:s + batch_size]
        x = torch.from_numpy(np.ascontiguousarray(sub)).float().to(device)
        o = readouts_for(model, x, rnd_cb)
        for k in READOUTS:
            acc[k].append(o[k].cpu().numpy())
        # 码号直方图（**整个 test 集**，不是抽几张 ⇒ 利用率才可信）
        codes = o["code_ids"].round().long().cpu().numpy().reshape(-1)
        for v, c_ in zip(*np.unique(codes, return_counts=True)):
            if v < K:
                hist[v] += int(c_)
    feats = {k: np.concatenate(v, 0).reshape(n_img, n_rep, -1) for k, v in acc.items()}
    return feats, hist


def usage_from_hist(hist):
    """码本利用率 + 熵（**用整个 test 集的码号直方图**）。"""
    cnt = hist[hist > 0]
    total = max(int(hist.sum()), 1)
    p = cnt / total
    ent = float(-(p * np.log(p)).sum())
    return {"used": int(len(cnt)), "total": int(len(hist)),
            "utilization": len(cnt) / max(len(hist), 1),
            "entropy_nats": ent,
            "entropy_max": float(np.log(max(len(cnt), 2)))}


def load_train_split(subject, n_images):
    """train split 的前 n_images 张（每张 n_rep 次）—— 指标 B 的**拟合集**。

    ⚠️ `train.pt` 里的 eeg 可能是 **numpy 数组**（实测）⇒ 用 `np.asarray` 统一，
    别直接 `.numpy()`（numpy 数组没有那个方法）。
    """
    import torch
    fp = TRAIN_PT / f"sub-{subject:02d}" / "train.pt"
    try:
        obj = torch.load(fp, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        obj = torch.load(fp, map_location="cpu", weights_only=False)
    eeg = obj["eeg"]
    log(f"  train split：{tuple(eeg.shape)}")
    return np.asarray(eeg[:n_images], dtype=np.float32)


def eval_subject(model, rnd_cb, eeg_te, eeg_tr, IMG_test, IMG_train, args, rng):
    """一个被试的完整评测。

    `eeg_tr is None` ⇒ 只算指标 A（省掉 train split 的抽取与 ridge）。
    返回 {"rows": {...}, "codebook": {...}}。
    """
    rec = {"rows": {}}
    F_te, hist = extract(model, eeg_te, rnd_cb, args.batch_size, args.device)
    F_tr = None
    if eeg_tr is not None:
        F_tr, _h2 = extract(model, eeg_tr, rnd_cb, args.batch_size, args.device)
    for k in READOUTS:
        Xte = F_te[k]
        # ⚠️ `F_tr` 在只算指标 A 时是 **None** ⇒ 不能无条件索引
        #    （上一版这么写 ⇒ 10 个被试全废在 TypeError 上）
        Xtr = F_tr[k] if F_tr is not None else None
        a = EVAL.metric_a_rsa(Xte, IMG_test, args.n_perm, rng)
        entry = {"dim": int(Xte.shape[-1]),
                 "r_img": a["r_img"], "p_img": a["p_img"]}
        if Xtr is not None and IMG_train is not None:
            mu, sd = EVAL.zscore_fit(Xtr)
            rb = EVAL.metric_b_decode(
                EVAL.zscore_apply(Xtr, mu, sd), EVAL.zscore_apply(Xte, mu, sd),
                IMG_train[:args.train_images], IMG_test, rng,
                n_null=args.n_null)
            entry["acc1"] = rb["acc1_cos"]
            entry["null_acc1"] = float(np.median(rb["null_acc1_cos"]))
        rec["rows"][k] = entry
        btxt = (f"| B acc1={entry['acc1']:.4f} "
                f"(零中位 {entry['null_acc1']:.4f})"
                if "acc1" in entry else "| B 跳过")
        log(f"  {k:<18} dim={Xte.shape[-1]:<5} A r={a['r_img']:+.4f} "
            f"p={a['p_img']:.3f} {btxt}")
    rec["codebook"] = usage_from_hist(hist)
    u = rec["codebook"]
    log(f"  码本利用率：{u['used']}/{u['total']} = {u['utilization']:.1%} "
        f"| 熵 {u['entropy_nats']:.2f} nats")
    return rec


def self_test_assembled():
    """**组装架构**（v1 / v2 两种头）的加载与读出链路。

    为什么必须测：这一轮要跑 **9 个评测 Job**（v1 四臂 + v2 四臂 + 对照臂），
    而 `load_model` 对组装架构走的是**另一条分支**（`arch="assembled*"`）——
    供体组件读取、头的数量（v1 两个 / v2 四个）、checkpoint 键位、读出维度
    全都在这条分支上。
    ⇒ 一条没本地跑过的分支 = 9 个 Job 一起报废（真发生过：10 个被试全废在
    `--no-metric-b` 的 TypeError 上）。

    做法：造**合成供体**（MHA 64 的注意力 + 64→128→64 的 FFN），
    分别按 `assembled` / `assembled_v2` 存两个 checkpoint，
    再走**真实的 `load_model` 路径**加载 + 跑全部读出。
    """
    import tempfile

    import torch
    from torch import nn

    import exp86_assembled_train as E86

    log("--- 自测：组装架构（assembled / assembled_v2）---")
    with tempfile.TemporaryDirectory(prefix="exp81-assembled-") as td:
        tmp = Path(td)
        mha = nn.MultiheadAttention(64, 4, batch_first=True)
        torch.save({"attn": {"state": mha.state_dict(), "src": "synthetic"}},
                   tmp / "signal-jepa.pt")
        torch.save({"ffn": {"h": 128, "d": 64,
                            "w1": torch.randn(128, 64) * 0.05,
                            "b1": torch.zeros(128),
                            "w2": torch.randn(64, 128) * 0.05,
                            "b2": torch.zeros(64),
                            "src": "synthetic"}},
                   tmp / "mirepnet.pt")
        old_dir = E86.DONOR_DIR
        E86.DONOR_DIR = tmp
        try:
            for arch in ("assembled", "assembled_v2"):
                heads = "v2" if arch == "assembled_v2" else "v1"
                model = E86.build_assembled(
                    n_codes=64,
                    attn_state=E86.extract_attn_state("signal-jepa"),
                    ffn_donor=E86.extract_ffn_donor("mirepnet"),
                    heads=heads)
                ck = {"state_dict": model.state_dict(),
                      "args": {"attn_donor": "signal-jepa", "steps": 1},
                      "n_codes": 64, "d_model": 64, "arch": arch,
                      "donor_ffn": "mirepnet", "hist": []}
                fp = tmp / f"ck_{arch}.pt"
                torch.save(ck, fp)
                # ↓ 走**真实**的加载路径（不是手工搭模型）
                m2, ck2, rnd = load_model(fp)
                assert m2.vq.n_codes == 64, m2.vq.n_codes
                x = torch.randn(6, 63, 250)
                out = readouts_for(m2, x, rnd)
                for k in READOUTS:
                    assert k in out, f"{arch} 缺读出 {k}"
                    assert torch.isfinite(out[k]).all(), f"{arch} 的 {k} 有非有限值"
                assert out["onehot_mean"].shape[-1] == 64
                assert out["emb_mean"].shape[-1] == 64
                assert out["continuous"].shape[-1] == 64
                assert out["recon"].shape[-1] == 250
                log(f"  ✓ {arch}：加载 + {len(READOUTS)} 个读出全部通过"
                    f"（头 {heads}）")

            # 「不嫁接注意力」的对照臂：checkpoint 里**没有**注意力块
            model = E86.build_assembled(n_codes=64, attn_state=None,
                                        ffn_donor=None, heads="v2")
            assert not any(k.startswith(("attn.", "norm1."))
                           for k in model.state_dict()), "不该有注意力参数"
            ck = {"state_dict": model.state_dict(),
                  "args": {"attn_donor": "none", "steps": 1},
                  "n_codes": 64, "d_model": 64, "arch": "assembled_v2",
                  "donor_ffn": "none", "hist": []}
            fp = tmp / "ck_noattn.pt"
            torch.save(ck, fp)
            m2, _, rnd = load_model(fp)
            out = readouts_for(m2, torch.randn(6, 63, 250), rnd)
            for k in READOUTS:
                assert k in out and torch.isfinite(out[k]).all(), k
            log("  ✓ 不嫁接注意力（attn_donor=none）：加载 + 读出全部通过")
        finally:
            E86.DONOR_DIR = old_dir
    log("--- 组装架构自测通过 ---")


def self_test():
    """用**合成数据**把两条分支（算 B / 不算 B）都跑通。

    为什么必须本地自测：上一版改了 `--no-metric-b` 分支后**没本地跑**就发到云端
    ⇒ 10 个被试全废在 TypeError 上（`F_tr` 为 None 仍被索引）。
    """
    import torch
    from exp80_eeg_tokenizer_train import EEGTokenizer

    log("=== 自测：S1 评测链路（合成数据）===")
    rng = np.random.default_rng(5)
    n_img_te, n_img_tr, n_rep, d_img = 30, 40, 2, 32
    IMG_te = rng.normal(size=(n_img_te, d_img))
    IMG_tr = rng.normal(size=(n_img_tr, d_img))
    W = rng.normal(size=(d_img, 63 * 250)) * 0.05

    def make(IMG, n_rep):
        e = (IMG @ W + 0.25 * rng.normal(size=(IMG.shape[0], 63 * 250)))
        e = e.reshape(IMG.shape[0], 63, 250)
        return np.repeat(e[:, None], n_rep, axis=1).astype(np.float32)

    eeg_te, eeg_tr = make(IMG_te, n_rep), make(IMG_tr, n_rep)
    model = EEGTokenizer(n_codes=64)
    rnd_cb = torch.randn(64, model.vq.dim) * 0.5

    for no_b in (True, False):
        args = argparse.Namespace(train_images=n_img_tr, n_perm=20, n_null=1,
                                 batch_size=8, device="cpu", no_metric_b=no_b)
        rec = eval_subject(model, rnd_cb, eeg_te,
                            None if no_b else eeg_tr,
                            IMG_te, None if no_b else IMG_tr,
                            args, np.random.default_rng(1))
        assert set(rec["rows"]) == set(READOUTS), rec["rows"].keys()
        assert all(v["r_img"] == v["r_img"] for v in rec["rows"].values())
        if no_b:
            assert all("acc1" not in v for v in rec["rows"].values())
        else:
            assert all("acc1" in v for v in rec["rows"].values())
        log(f"  ✓ no_metric_b={no_b} 分支通过（{len(rec['rows'])} 个读出）")
    self_test_assembled()
    log("=== 自测通过 ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="",
                    help="自训 tokenizer 的 checkpoint（桶内路径）；"
                         "--self-test 时不需要")
    ap.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--train-images", type=int, default=2000,
                    help="指标 B 的拟合集用多少张 train split 的图")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-null", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-metric-b", action="store_true",
                    help="只算指标 A（RSA）。指标 B 的 ridge 很贵（试次级 Gram "
                         "8000² × 每读出 1+N 次拟合）而它对所有读出都在随机水平 "
                         "⇒ 主轮只跑 A，B 单独一轮补")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--self-test", action="store_true",
                    help="合成数据把评测链路跑通（含 --no-metric-b 两条分支）")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.ckpt:
        log("✗ 需要 --ckpt（或用 --self-test）")
        raise SystemExit(1)

    import torch

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ftr = torch.load(FEAT_TRAIN, map_location="cpu", weights_only=False)
    IMG_train = ftr["img_features"].numpy().astype(np.float64)
    fte = torch.load(FEAT_TEST, map_location="cpu", weights_only=False)
    IMG_test = fte["img_features"].numpy().astype(np.float64)
    log(f"图像特征：train {IMG_train.shape} / test {IMG_test.shape}")

    model, ck, rnd_cb = load_model(args.ckpt)
    log(f"✓ 自训 tokenizer：{args.ckpt} | 码本 K={model.vq.n_codes} "
        f"d={model.vq.dim} | token 数 {ck.get('n_tokens')} | "
        f"训练步数 {ck.get('args', {}).get('steps')}")

    sids = [int(s) for s in args.subjects.split(",") if s.strip()]
    rng = np.random.default_rng(SEED)
    per_subject = []
    for sid in sids:
        t0 = time.time()
        log(f"\n=== sub-{sid:02d} ===")
        rec = {"subject": f"sub-{sid:02d}", "rows": {}}
        try:
            eeg_te = np.load(TEST_4D / f"sub-{sid:02d}.npz")["eeg"]
            eeg_tr = None
            if not args.no_metric_b:
                eeg_tr = load_train_split(sid, args.train_images)
            rec.update(eval_subject(model, rnd_cb, eeg_te, eeg_tr,
                                    IMG_test, IMG_train, args, rng))
            del eeg_te
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
            rec["trace"] = traceback.format_exc()[-600:]
            log(f"  ✗ {rec['error']}")
        rec["elapsed_s"] = round(time.time() - t0, 1)
        per_subject.append(rec)

    # ---- 汇总（中位 + 显著被试数 + 配对对照）----
    summary = {}
    for k in READOUTS:
        rs = [r["rows"][k]["r_img"] for r in per_subject if k in r.get("rows", {})]
        ps = [r["rows"][k]["p_img"] for r in per_subject if k in r.get("rows", {})]
        ac = [r["rows"][k].get("acc1") for r in per_subject if k in r.get("rows", {})]
        nu = [r["rows"][k].get("null_acc1") for r in per_subject
              if k in r.get("rows", {})]
        ac = [x for x in ac if x is not None]
        nu = [x for x in nu if x is not None]
        if not rs:
            continue
        rs_s = sorted(rs)
        entry = {"n": len(rs), "r_median": rs_s[len(rs_s) // 2],
                 "p_median": sorted(ps)[len(ps) // 2],
                 "n_sig": sum(1 for p in ps if p < 0.05)}
        if ac:
            entry["acc1_median"] = sorted(ac)[len(ac) // 2]
            entry["null_median"] = sorted(nu)[len(nu) // 2]
        summary[k] = entry
        log(f"  {k:<18} 中位 A r={entry['r_median']:+.4f} "
            f"({entry['n_sig']}/{entry['n']} 显著)")
        if ac:
            log(f"  {'':<18}   B acc1={entry['acc1_median']:.4f} "
                f"(零中位 {entry['null_median']:.4f})")

    (out_dir / "tokenizer_eval.json").write_text(
        json.dumps({"ckpt": args.ckpt, "config": vars(args),
                    "per_subject": per_subject, "summary": summary},
                   indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"\n结果写入 {out_dir}/tokenizer_eval.json")


if __name__ == "__main__":
    main()
