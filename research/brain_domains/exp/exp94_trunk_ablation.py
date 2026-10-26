#!/usr/bin/env python3
"""exp94_trunk_ablation.py — **骨干（token 化方式）对照 + 交叉验证**。

## 为什么做这个

exp93 的结论全是**在同一个骨干上**得到的：

    S1a(exp80) / S1b(exp86v2) / exp93b(对比学习)
    ⇒ 躯干都是 `ch_mix(1×1) → 2×Conv1d(k=5) → ×8 → 31 token`

⇒ **骨干从未被单独变过** ⇒ 所有「目标 > 读出 > 组件」的排序都只在**这一个实现**上成立。
这是外部审稿人最可能打的地方，也是我们证据里**唯一**没被检验过的架构级维度。

外部依据（注册源一手）：
- **LaBraM**（ICLR 2024, arXiv:2405.18765）：`EEG channel patches` → Transformer；
  **VQ 神经频谱预测**训离散 tokenizer ⇒ **保留通道身份**。
- **NeuroLM**（ICLR 2025, arXiv:2409.00101）：`VQ 时频预测` 的**文本对齐**离散 tokenizer，
  **冻结** VQ encoder → LLM 多通道自回归 ⇒ 与我们的 G4「内部状态先于功能」同构。
⇒ 二者都**保留通道身份**；而我们的 `ch_mix(1×1)` **先把通道混掉** ⇒
⇒ 假设 H1：**通道身份被抹掉，是「组件有缺陷」的最可能形式**。

## 两个变体（**只差 token 化方式**，参数总量对齐）

| | V0（现状 = exp93b 骨干） | V1（保留通道身份） |
|---|---|---|
| 空间 | `ch_mix(1×1)` **先混 63 通道** | **不混**；逐通道独立时间卷积 |
| token | 31 个**时间** token（已混通道） | 63 个**通道** token（每通道时间 31→64 线性） |
| 跨通道 | 无（head 是 31 token 展平） | **Transformer 4 层跨通道注意力**（+ 可学习通道嵌入）|
| 头 | flatten(31×64)→512→1024 | flatten(63×64)→256→1024（**宽度调小以匹配参数量**）|
| 参数量 | ≈1.62 M | ≈1.54 M（**不更大** ⇒ 性能差异不可能来自容量）|

⚠️ **诚实的表述**：变的是「**空间-时间 token 化方式**」，不是「只改空间一个维度」
（V0 保留 31 个时间 token，V1 把每通道的时间压成一个 token）。
⇒ 预登记里按「空间-时间 token 化对照」记，不夸大成纯空间消融。

## 交叉验证矩阵（每个变体都跑全部线）

| # | 线 | 读出 | 测什么 |
|---|---|---|---|
| C1 | 200-way **单试次** top-1 | `zeroshot.acc1_cos` | 绝对档位 |
| C2 | 200-way **试次平均** top-1 | `zeroshot.acc1_featavg_cos` | 绝对档位（聚合后） |
| C3 | **RSA**（vs 图像特征） | `E89.rsa` | 表征对齐强度 |
| C4 | **跨被试** | sub-10（**没进过训练**） | 泛化 |
| C5 | **位置轴**（模型层） | POST/ANT/ALL 的 zeroshot + RSA | 位置轴结论**是否依赖骨干** |
| C6 | **地板**（同尺寸随机通道组） | POST > 全部随机组？ANT < 全部随机组？ | 排除「通道多就是好」 |
| C7 | **自一致性** | V0 vs exp93b 已发表值 | 本实验有没有实现漂移 |

## 预登记判据（写死在脚本里，事后不改）

1. **C7 自一致性**：V0 的 `zeroshot_acc1` 中位须落在 exp93b 已发表值 **0.0589 ± 0.020**，
   `rsa` 落在 **0.1273 ± 0.045** ⇒ 否则标 `self_consistency=FAIL`，
   **本实验所有结论作废**（实现漂移 ⇒ 数字不可信）。
2. **主判决 H1**：V1 在 **C1 与 C2 同时**优于 V0（配对符号检验方向一致）⇒
   「通道身份被抹掉是组件缺陷」成立 ⇒ **底座换掉**（直接用现成 patch 化实现，不自写）。
3. **交叉验证 H2（位置轴的骨干无关性）**：**C5/C6 上 V0 与 V1 必须给出同向的位置轴结论**
   （都是 POST > ANT，且 POST 高于/ANT 低于随机组）⇒
   ⇒ **A14 的模型层部分不是我们这一个骨干的产物**。
   ⚠️ 若 V1 上位置轴**塌掉** ⇒ ⇒ **A14 的模型层部分骨干依赖** ⇒ 这是更重要的发现，
   必须显式报告，不得当噪声。
4. **C3 的读出口径纪律（R4）**：C1/C2/C3 三条线的方向若**不一致** ⇒
   **不得**下「谁更强」的单句结论 ⇒ 必须逐条归因后再说。

## 用法

    # 本地自测（形状 + 端到端可学性）
    python -u research/brain_domains/exp/exp94_trunk_ablation.py --self-test

    # 云端（**一个 Job 内串行跑两个变体** ⇒ 同 flavor、同争用、同采样序列）
    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp94_trunk_ablation.py \
      --train-subjects 1,2,3,4,5,6,7,8,9 --eval-subjects 1,2,3,4,5,6,7,8,9,10 \
      --epochs 30 --steps-per-epoch 1000 \
      -v hf://buckets/illusion-brain/training-data:/training:ro \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor a10g-large --timeout 180m -d --name t94-trunk-ablation
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

for _v in ("OMP_NUM_THREADS", "OPENBLAS_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

EXP_DIR = Path(__file__).resolve().parent
_PKG_CANDIDATES = [EXP_DIR / "models", EXP_DIR.parent / "models", EXP_DIR]
PKG_DIR = next((c for c in _PKG_CANDIDATES if (c / "registry.py").exists()),
               _PKG_CANDIDATES[0])
for _p in (str(EXP_DIR), str(PKG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import exp93b_contrastive as E93B   # noqa: E402

E89 = E93B.E89
E73 = E93B.E73

OUT_DIR = Path("/domains/runs/illusionbrain-align/exp94-trunk")
SEED = 0
CHANCE = 1.0 / 200.0
GROUP_SIZE = 34
ANT_SIZE = 29
K_LEVELS = (5, 11, 17, 23, 29, 34)

# exp93b 已发表值（C7 自一致性锚点；**只用于量级检查**，不改任何数）
PUBLISHED_V0 = {"zeroshot_acc1": 0.0589, "zeroshot_acc1_featavg": 0.2300,
                 "rsa": 0.1273, "zeroshot_acc5": 0.1875}
SELF_CONSISTENCY_TOL = {"zeroshot_acc1": 0.020,
                         "zeroshot_acc1_featavg": 0.060,
                         "rsa": 0.045,
                         "zeroshot_acc5": 0.040}


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# V1：保留通道身份的 token 化
# ---------------------------------------------------------------------------

def build_model_v1(n_ch=63, n_times=250, out_dim=1024, dim=64,
                    n_layers=4, nhead=4, head_hidden=256, t_stride=8):
    """逐通道时间编码 → **每通道 1 个 token** → 跨通道 Transformer。

    与 V0 的差别只有 token 化方式：
      V0: ch_mix(1×1) 混通道 → 时间 conv → 31 个时间 token → flatten 头
      V1: **不混通道** → 逐通道时间 conv → 每通道 31→64 线性 → 63 个通道 token
          → 可学习通道嵌入 → Transformer 跨通道注意力 → flatten 头
    头宽度调小（512→256）以**匹配参数量** ⇒ V1 不比 V0 更大。
    """
    import torch
    from torch import nn

    class PerChannelLN(nn.Module):
        """**只沿时间归一化**（reshape 成 (B*C,1,T)）⇒ 绝不跨通道混合。

        LayerNorm 的归一化维度是最后两维；这里中间维是 1 ⇒ 实际只沿 T 归一化。
        """

        def __init__(self, n_times):
            super().__init__()
            self.ln = nn.LayerNorm(n_times)

        def forward(self, h):
            b, c, t = h.shape
            return self.ln(h.reshape(b * c, 1, t)).reshape(b, c, t)

    class ChannelTokenEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv1d(n_ch, n_ch, 5, padding=2), nn.GELU(),
                PerChannelLN(n_times),
                nn.Conv1d(n_ch, n_ch, 5, padding=2), nn.GELU(),
                PerChannelLN(n_times))
            self.t_down = nn.Conv1d(n_ch, n_ch, t_stride, stride=t_stride)
            with torch.no_grad():
                dummy = torch.zeros(1, n_ch, n_times)
                self.n_times = int(self.t_down(self.enc(dummy)).shape[2])
            self.time_proj = nn.Linear(self.n_times, dim)
            self.ch_embed = nn.Parameter(torch.zeros(1, n_ch, dim))
            nn.init.normal_(self.ch_embed, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=nhead, dim_feedforward=4 * dim,
                dropout=0.0, batch_first=True, norm_first=True,
                activation="gelu")
            self.tr = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(n_ch * dim),
                nn.Linear(n_ch * dim, head_hidden), nn.GELU(),
                nn.Linear(head_hidden, out_dim))

        def forward(self, x):
            h = self.t_down(self.enc(x))            # (B, C, T')
            h = self.time_proj(h)                   # (B, C, dim)  ← 沿时间投影，逐通道共享
            h = h + self.ch_embed
            h = self.tr(h)                          # 跨通道注意力
            return self.head(h.flatten(1))

    # PerChannelLN 的 LayerNorm 在构造时即建好（不做事后挂载）
    return ChannelTokenEncoder()


# ---------------------------------------------------------------------------
# 训练（**两个变体共用同一份数据、同一个采样序列** ⇒ 严格受控）
# ---------------------------------------------------------------------------

def train_variant(args, IMG_train, device, EEG, IMGIDX, subjects, val_img,
                  perm, n_img, n_rep, n_ch, n_times, variant, out_dir):
    import torch
    import torch.nn.functional as F

    if variant == "v0":
        model = E93B.build_model(n_ch=n_ch, n_times=n_times,
                                 out_dim=IMG_train.shape[1]).to(device)
    elif variant == "v1":
        model = build_model_v1(n_ch=n_ch, n_times=n_times,
                                out_dim=IMG_train.shape[1]).to(device)
    else:
        raise ValueError(variant)

    nparam = sum(p.numel() for p in model.parameters())
    log(f"[{variant}] 参数量 {nparam:,}｜设备 {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * args.steps_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1))

    def sample_batch(step_rng):
        imgs = step_rng.choice(n_img - len(val_img), size=args.batch, replace=False)
        imgs = np.sort(imgs)
        imgs = perm[len(val_img):][imgs]
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

    # ⚠️ **采样序列两个变体完全相同**（同 seed、同 perm、同 val_img）
    step_rng = np.random.default_rng(SEED)
    hist, best = [], {"val_acc": -1.0, "epoch": -1}
    t_start = time.time()
    t_fwd = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for _ in range(args.steps_per_epoch):
            xb, tb = sample_batch(step_rng)
            t0 = time.time()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=(device.type == "cuda")):
                z = model(xb.float())
                loss = E93B.info_nce(F.normalize(z, dim=1),
                                     F.normalize(tb, dim=1), args.temp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(float(loss.item()))
            t_fwd += time.time() - t0
        tr_loss = float(np.mean(losses))
        val = E93B.eval_val(model, EEG, IMGIDX, subjects, val_img, IMG_train,
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
                        "out_dim": IMG_train.shape[1], "variant": variant},
                       out_dir / f"encoder_{variant}_best.pt")
        log(f"  [{variant}] epoch {ep:>3} | loss {tr_loss:.4f} | val acc1 "
            f"{val['acc1_cos']:.4f} featavg {val['acc1_featavg_cos']:.4f} | "
            f"{time.time()-t_start:.0f}s")
        if ep == args.guard_epoch and val["acc1_featavg_cos"] < args.min_val_acc1:
            log(f"✗✗ [{variant}] 第 {ep} 轮 val featavg={val['acc1_featavg_cos']:.4f} "
                f"< {args.min_val_acc1} ⇒ 没学到东西 ⇒ 中止该变体")
            return None, None, hist

    torch.save({"state_dict": model.state_dict(), "args": vars(args),
                "epoch": args.epochs, "n_ch": n_ch, "n_times": n_times,
                "out_dim": IMG_train.shape[1], "variant": variant},
               out_dir / f"encoder_{variant}_last.pt")
    return model, best, hist


def reload_model(variant, ckpt_path, device, n_ch, n_times, out_dim):
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if variant == "v0":
        model = E93B.build_model(n_ch=ck["n_ch"], n_times=ck["n_times"],
                                 out_dim=ck["out_dim"])
    else:
        model = build_model_v1(n_ch=ck["n_ch"], n_times=ck["n_times"],
                                out_dim=ck["out_dim"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


# ---------------------------------------------------------------------------
# 位置轴（模型层）+ 地板：POST/ANT/ALL × 同尺寸随机通道组
# ---------------------------------------------------------------------------

def mask_channels(eeg, keep_idx):
    x = eeg.copy()
    m = np.ones(x.shape[2], dtype=bool)
    m[np.asarray(keep_idx)] = False
    x[:, :, m, :] = 0.0
    return x


def rand_groups(pool, size, n_draw, rng):
    out = []
    for _ in range(n_draw):
        out.append(list(rng.choice(pool, size=size, replace=False)))
    return out


def eval_posaxis(model, sid, post, ant, n_ch, IMG_test, args, device, rng):
    """模型层的位置轴 + 地板。

    ⚠️ **读出口径**：本实验的模型是**对比学习**训的 ⇒ 它的输出 z 就是被训练成
    与图像特征对齐的 ⇒ 拿 z 做 RSA **部分是循环的**（目标就是让 z·image 大）。
    ⇒⇒ 因此**主指标用 zeroshot 200-way acc1**（它是真实的检索任务），
    **RSA 只作次要参考**，并在此显式标注。
    ⇒ 另外：掩码（把不选中的通道置零）对对比学习模型是 **OOD 输入**
    ⇒ 与 exp92-A 的模型层掩码读出同属这一类，结论只在**模型之间可比**。
    """
    eeg = E93B.load_test_subject(sid)
    eeg = eeg[:, :args.pa_reps]
    n_img, n_rep = eeg.shape[0], eeg.shape[1]
    flat_dim = None
    res = {}

    named = [("ALL", list(range(n_ch))), ("POST", list(post)), ("ANT", list(ant))]
    rnd_post = rand_groups(list(ant) + list(post), GROUP_SIZE, args.pa_rand, rng)
    rnd_ant = rand_groups(list(ant) + list(post), ANT_SIZE, args.pa_rand, rng)
    plan = named + [(f"RND34_{i}", g) for i, g in enumerate(rnd_post)] \
        + [(f"RND29_{i}", g) for i, g in enumerate(rnd_ant)]

    for tag, idx in plan:
        xm = mask_channels(eeg, idx)
        Z = E93B.encode_all(model, xm.reshape(-1, n_ch, eeg.shape[3]),
                            device, args.batch).reshape(n_img, n_rep, -1)
        flat_dim = Z.shape[-1]
        acc = E93B.zeroshot_topk(Z, IMG_test, n_rep)
        rec = {"n_ch": len(idx), "acc1": acc["acc1_cos"],
               "acc1_featavg": acc["acc1_featavg_cos"]}
        if tag in ("ALL", "POST", "ANT"):
            r, p = E89.rsa(Z, IMG_test, args.pa_perm, rng)
            rec["rsa"], rec["rsa_p"] = float(r), float(p)
        res[tag] = rec
        log(f"    [{tag:<9}] n_ch={len(idx):<3} acc1={acc['acc1_cos']:.4f} "
            f"featavg={acc['acc1_featavg_cos']:.4f}"
            + (f" rsa={rec.get('rsa', 0):+.4f}" if "rsa" in rec else ""))

    post_v = [res[t]["acc1"] for t in res if t.startswith("RND34")]
    ant_v = [res[t]["acc1"] for t in res if t.startswith("RND29")]
    verdict = {
        "post_above_all_rand34": bool(res["POST"]["acc1"] > max(post_v)) if post_v else None,
        "ant_below_all_rand29": bool(res["ANT"]["acc1"] < min(ant_v)) if ant_v else None,
        "post_minus_ant": float(res["POST"]["acc1"] - res["ANT"]["acc1"]),
    }
    return {"groups": res, "verdict": verdict, "flat_dim": flat_dim}


# ---------------------------------------------------------------------------
# 单被试主评测（C1/C2/C3/C4）
# ---------------------------------------------------------------------------

def eval_subject(model, sid, IMG_train, IMG_test, args, device):
    eeg = E93B.load_test_subject(sid)
    n_img, n_rep, n_ch, T = eeg.shape
    Z = E93B.encode_all(model, eeg.reshape(-1, n_ch, T),
                        device, args.batch).reshape(n_img, n_rep, -1)
    rec = {"subject": f"sub-{sid:02d}", "zeroshot": E93B.zeroshot_topk(Z, IMG_test, n_rep)}
    n_ridge = min(args.ridge_train_images, 16540)
    e_tr, _p, _l = E93B.load_train_subject(sid, n_images=n_ridge)
    Ztr = E93B.encode_all(model, e_tr.reshape(-1, n_ch, T),
                          device, args.batch).reshape(n_ridge, e_tr.shape[1], -1)
    rec["ridge_feat"] = E93B.ridge_on_features(Ztr, IMG_train[:n_ridge], Z, IMG_test)
    rec["ridge_feat"]["n_train_images"] = int(n_ridge)
    del e_tr, Ztr
    r, p = E89.rsa(Z, IMG_test, args.align_check_perm, np.random.default_rng(SEED + sid))
    rec["rsa"] = {"r": float(r), "p": float(p)}
    return rec


# ---------------------------------------------------------------------------
# 汇总与预登记判据
# ---------------------------------------------------------------------------

def med_up(xs):
    s = sorted(float(x) for x in xs if x == x)
    return s[len(s) // 2] if s else None


def sign_test_p(diffs):
    """双侧符号检验（精确）。"""
    from math import comb
    d = [x for x in diffs if x == x]
    n = len(d)
    if n == 0:
        return None
    k = sum(1 for x in d if x > 0)
    lo = min(k, n - k)
    p = sum(comb(n, i) for i in range(lo + 1)) / (2 ** n) * 2
    return float(min(1.0, p))


def summarize(per_subject):
    out = {}
    if not per_subject:
        return out
    ok = [r for r in per_subject if "error" not in r]
    if not ok:
        return out
    out["zeroshot_acc1"] = med_up([r["zeroshot"]["acc1_cos"] for r in ok])
    out["zeroshot_acc1_featavg"] = med_up([r["zeroshot"]["acc1_featavg_cos"] for r in ok])
    out["zeroshot_acc5"] = med_up([r["zeroshot"]["acc5_cos"] for r in ok])
    out["ridge_feat_acc1"] = med_up([r["ridge_feat"]["acc1_cos"] for r in ok])
    out["rsa"] = med_up([r["rsa"]["r"] for r in ok])
    out["n_ok"] = len(ok)
    out["n_err"] = len(per_subject) - len(ok)
    return out


def decide(summaries, per_subject_by_variant, args):
    """**预登记判据**（写死在这里，事后不改）。"""
    dec = {}
    v0, v1 = summaries.get("v0"), summaries.get("v1")
    if not v0 or not v1:
        return {"verdict": "INCOMPLETE", "reason": "至少一个变体没有结果"}

    # --- C7 自一致性（V0 复现 vs exp93b 已发表值）---
    sc = {}
    for k, pub in PUBLISHED_V0.items():
        got = v0.get(k)
        tol = SELF_CONSISTENCY_TOL[k]
        sc[k] = {"published": pub, "got": got, "tol": tol,
                 "ok": bool(got is not None and abs(got - pub) <= tol)}
    dec["self_consistency"] = sc
    sc_ok = all(v["ok"] for v in sc.values())
    dec["self_consistency_pass"] = sc_ok

    # --- C1/C2：配对差（逐被试）+ 符号检验 ---
    def _get(rec, path):
        cur = rec
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                return None
            cur = cur[p]
        return cur

    def paired(path):
        """path 是**嵌套路径**，如 ["zeroshot", "acc1_cos"]。"""
        a = {r["subject"]: r for r in per_subject_by_variant["v0"]
             if "error" not in r}
        b = {r["subject"]: r for r in per_subject_by_variant["v1"]
             if "error" not in r}
        common = sorted(set(a) & set(b))
        diffs = []
        for s in common:
            va, vb = _get(a[s], path), _get(b[s], path)
            if va is None or vb is None:
                continue
            diffs.append(float(vb) - float(va))
        return {"n": len(diffs), "diffs": diffs,
                "median_diff": med_up(diffs),
                "n_pos": sum(1 for d in diffs if d > 0),
                "p_sign": sign_test_p(diffs)}

    dec["paired"] = {
        "zeroshot_acc1": paired(["zeroshot", "acc1_cos"]),
        "zeroshot_acc1_featavg": paired(["zeroshot", "acc1_featavg_cos"]),
        "rsa": paired(["rsa", "r"]),
    }
    c1 = dec["paired"]["zeroshot_acc1"]
    c2 = dec["paired"]["zeroshot_acc1_featavg"]
    c3 = dec["paired"]["rsa"]
    # ⚠️ 必须先确认**配对数 > 0**，否则「n_pos > n/2」会被 0>0 判成 False ⇒ 假阴性
    paired_ok = all(c["n"] > 0 for c in (c1, c2, c3))
    h1 = paired_ok and c1["n_pos"] > c1["n"] / 2 and c2["n_pos"] > c2["n"] / 2
    dec["H1_channel_identity_helps"] = bool(h1)
    dec["H1_detail"] = {
        "C1_single_trial": {"v0": v0.get("zeroshot_acc1"),
                            "v1": v1.get("zeroshot_acc1"),
                            "median_diff": c1["median_diff"],
                            "n_pos": c1["n_pos"], "n": c1["n"], "p": c1["p_sign"]},
        "C2_trial_avg": {"v0": v0.get("zeroshot_acc1_featavg"),
                         "v1": v1.get("zeroshot_acc1_featavg"),
                         "median_diff": c2["median_diff"],
                         "n_pos": c2["n_pos"], "n": c2["n"], "p": c2["p_sign"]},
        "C3_rsa": {"v0": v0.get("rsa"), "v1": v1.get("rsa"),
                   "median_diff": c3["median_diff"],
                   "n_pos": c3["n_pos"], "n": c3["n"], "p": c3["p_sign"]},
    }
    # --- R4 纪律：C1/C2/C3 方向不一致 ⇒ 不许下单句结论 ---
    if not paired_ok:
        dec["readout_consistent"] = None
        dec["R4_note"] = "⚠️ 配对数为 0 ⇒ 读出一致性**无法判定**（实现问题，不是结论）"
    else:
        signs = [np.sign(c1["median_diff"] or 0), np.sign(c2["median_diff"] or 0),
                 np.sign(c3["median_diff"] or 0)]
        dec["readout_consistent"] = bool(len(set(signs)) == 1)
        dec["R4_note"] = ("三条读出方向一致" if dec["readout_consistent"]
                          else "⚠️ 三条读出方向**不一致** ⇒ 不得下「谁更强」的单句结论")

    # --- C5/C6：位置轴的骨干无关性（H2）---
    pa = {}
    for v in ("v0", "v1"):
        recs = [r for r in per_subject_by_variant.get(v, [])
                if "error" not in r and "posaxis" in r]
        if not recs:
            pa[v] = None
            continue
        pa[v] = {
            "post_above_all_rand34": sum(
                1 for r in recs if r["posaxis"]["verdict"]["post_above_all_rand34"]),
            "ant_below_all_rand29": sum(
                1 for r in recs if r["posaxis"]["verdict"]["ant_below_all_rand29"]),
            "post_minus_ant_median": med_up(
                [r["posaxis"]["verdict"]["post_minus_ant"] for r in recs]),
            "post_acc1_median": med_up(
                [r["posaxis"]["groups"]["POST"]["acc1"] for r in recs]),
            "ant_acc1_median": med_up(
                [r["posaxis"]["groups"]["ANT"]["acc1"] for r in recs]),
            "post_rsa_median": med_up(
                [r["posaxis"]["groups"]["POST"].get("rsa") for r in recs
                 if r["posaxis"]["groups"]["POST"].get("rsa") is not None]),
            "n": len(recs),
        }
    dec["posaxis"] = pa
    if pa.get("v0") and pa.get("v1"):
        same = ((pa["v0"]["post_minus_ant_median"] or 0) > 0
                and (pa["v1"]["post_minus_ant_median"] or 0) > 0)
        dec["H2_posaxis_backbone_independent"] = bool(same)
        dec["H2_note"] = (
            "两个骨干上都 POST>ANT ⇒ 位置轴结论**不是单一骨干的产物**"
            if same else
            "⚠️ 两个骨干上位置轴结论**不一致** ⇒ A14 的模型层部分**骨干依赖**（重要发现）")
    else:
        dec["H2_posaxis_backbone_independent"] = None
        dec["H2_note"] = "位置轴数据不全"

    # --- 总判决 ---
    if not sc_ok:
        dec["verdict"] = "INVALID_SELF_CONSISTENCY"
    elif not dec["readout_consistent"]:
        dec["verdict"] = "MIXED_READOUTS"
    elif dec["H1_channel_identity_helps"] and dec["H2_posaxis_backbone_independent"]:
        dec["verdict"] = "H1_SUPPORTED_COMPONENT_FIXABLE"
    elif dec["H1_channel_identity_helps"]:
        dec["verdict"] = "H1_SUPPORTED_BUT_POSAXIS_SHIFTS"
    else:
        dec["verdict"] = "H1_NOT_SUPPORTED_TRUNK_NOT_BOTTLENECK"
    return dec


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

def self_test():
    import torch
    log("=== 自测：V0/V1 形状 + 参数量匹配 + 端到端可学性 ===")
    m0 = E93B.build_model(n_ch=63, n_times=250, out_dim=1024)
    m1 = build_model_v1(n_ch=63, n_times=250, out_dim=1024)
    n0 = sum(p.numel() for p in m0.parameters())
    n1 = sum(p.numel() for p in m1.parameters())
    log(f"  V0 参数量 {n0:,}｜V1 参数量 {n1:,}｜比值 {n1/n0:.3f}")
    assert 0.85 <= n1 / n0 <= 1.15, "参数量没对齐 ⇒ 性能差异可能来自容量"
    with torch.no_grad():
        x = torch.randn(4, 63, 250)
        z0 = m0(x)
        z1 = m1(x)
    assert z0.shape == (4, 1024) and z1.shape == (4, 1024)
    log(f"  ✓ 输出形状 {tuple(z0.shape)} / {tuple(z1.shape)}")

    # **V1 不得跨通道混合**的硬检查：
    # 改一个通道的输入 ⇒ 输出必须改变（若 V1 内部把通道混了，这条会漏检）
    with torch.no_grad():
        x2 = x.clone()
        x2[:, 5, :] = x2[:, 5, :] + 3.0
        d = (m1(x2) - m1(x)).abs().max().item()
    log(f"  V1：改通道 5 的输入 ⇒ 输出最大变化 {d:.4f}")
    assert d > 1e-4, "V1 似乎对通道 5 不敏感 ⇒ 通道身份被抹掉了？"

    # V0 的 ch_mix 是全混合 ⇒ 任何通道都影响输出（对照）
    with torch.no_grad():
        x3 = x.clone()
        x3[:, 5, :] = x3[:, 5, :] + 3.0
        d0 = (m0(x3) - m0(x)).abs().max().item()
    log(f"  V0：改通道 5 的输入 ⇒ 输出最大变化 {d0:.4f}")

    # 端到端：合成可学映射 ⇒ 两个变体都必须能把 loss 降下来
    for name, m in (("v0", m0), ("v1", m1)):
        torch.manual_seed(0)
        A = torch.randn(63, 1024) / 8.0
        xx = torch.randn(32, 63, 250)
        tgt = xx.mean(axis=2) @ A
        mm = E93B.build_model(n_ch=63, n_times=250, out_dim=1024) if name == "v0" \
            else build_model_v1(n_ch=63, n_times=250, out_dim=1024)
        opt = torch.optim.AdamW(mm.parameters(), lr=3e-3)
        l0 = None
        for i in range(80):
            loss = E93B.info_nce(torch.nn.functional.normalize(mm(xx), dim=1),
                                 torch.nn.functional.normalize(tgt, dim=1), 0.07)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if i == 0:
                l0 = float(loss)
        l1 = float(loss)
        log(f"  {name}：合成可学任务 loss {l0:.4f} → {l1:.4f}")
        assert l1 < l0 - 0.1, f"{name} 端到端学不动"
    log("=== 自测通过 ===")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument("--eval-subjects", type=int, nargs="+",
                    default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ap.add_argument("--variants", default="v0,v1")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps-per-epoch", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--align-check-images", type=int, default=1500)
    ap.add_argument("--align-check-perm", type=int, default=100)
    ap.add_argument("--ridge-train-images", type=int, default=4000)
    ap.add_argument("--min-val-acc1", type=float, default=0.008)
    ap.add_argument("--guard-epoch", type=int, default=3)
    ap.add_argument("--pa-reps", type=int, default=20,
                    help="位置轴测试用前 N 个重复（每组 200×N 次前向）")
    ap.add_argument("--pa-rand", type=int, default=3, help="每种尺寸的随机组数")
    ap.add_argument("--pa-perm", type=int, default=100, help="位置轴 RSA 的置换次数")
    ap.add_argument("--skip-posaxis", action="store_true")
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
    log(f"设备 {device}｜torch {torch.__version__}")

    fte = torch.load(E93B.FEAT_TEST, map_location="cpu", weights_only=False)
    IMG_test = fte["img_features"].numpy().astype(np.float64)
    ftr = torch.load(E93B.FEAT_TRAIN, map_location="cpu", weights_only=False)
    IMG_train = ftr["img_features"].numpy().astype(np.float64)
    log(f"图像特征 test {IMG_test.shape} | train {IMG_train.shape}")

    # ---------- 预检 P1：对齐（三段式，与 exp93b 同一套）----------
    log("[预检 P1] 对齐三段式证明")
    eeg0, paths0, labels0 = E93B.load_train_subject(args.train_subjects[0],
                                                     n_images=args.align_check_images)
    X = E73.per_rep_targets(np.asarray(eeg0, dtype=np.float16))
    del eeg0
    flat = X.reshape(-1, X.shape[-1])
    X = (X - flat.mean(axis=0)) / (flat.std(axis=0) + 1e-6)
    al = E93B.verify_alignment(X, paths0, labels0, IMG_train,
                               np.random.default_rng(SEED), args.align_check_perm)
    log(f"  ①label_match={al.get('label_match_frac')} "
        f"②nn_same_block={al.get('nn_same_block_frac', 0):.4f} "
        f"③RSA={al.get('rsa_r', 0):+.4f} (p={al.get('rsa_p', 1):.4f})")
    del X
    if not (al.get("nn_same_block_frac", 0) > 0.5 and al.get("rsa_r", 0) > 0
            and al.get("rsa_p", 1) < 0.05):
        log("✗✗ 对齐预检失败 ⇒ 中止")
        raise SystemExit(2)

    # ---------- 载入训练数据（**两个变体共用**）----------
    subjects = list(args.train_subjects)
    t0 = time.time()
    EEG, IMGIDX = [], []
    for s in subjects:
        e, _p, _l = E93B.load_train_subject(s)
        EEG.append(torch.from_numpy(e))
        IMGIDX.append(np.arange(len(e), dtype=np.int64))
        log(f"  sub-{s:02d}: {tuple(e.shape)}（累计 {time.time()-t0:.0f}s）")
    n_img, n_rep = EEG[0].shape[0], EEG[0].shape[1]
    n_ch, n_times = EEG[0].shape[2], EEG[0].shape[3]
    perm = np.random.default_rng(SEED).permutation(n_img)
    n_val = max(1, int(args.val_frac * n_img))
    val_img = np.sort(perm[:n_val])
    log(f"[数据] 每被试 {n_img} 图 × {n_rep} 重复 × {n_ch} 通道 × {n_times} 样本"
        f"｜fit {n_img-n_val} / val {len(val_img)}")

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    summaries, per_subject_by_variant, all_hist, all_best = {}, {}, {}, {}

    for variant in variants:
        log(f"\n{'='*74}\n=== 变体 {variant} ===\n{'='*74}")
        vdir = out_dir / variant
        vdir.mkdir(parents=True, exist_ok=True)
        model, best, hist = train_variant(args, IMG_train, device, EEG, IMGIDX,
                                          subjects, val_img, perm, n_img, n_rep,
                                          n_ch, n_times, variant, vdir)
        all_hist[variant] = hist
        all_best[variant] = best
        if model is None:
            log(f"✗ [{variant}] 训练中止")
            continue
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

        # reload best（按 train-split 验证集选的，未用 test）
        model, ck = reload_model(variant, vdir / f"encoder_{variant}_best.pt",
                                 device, n_ch, n_times, IMG_train.shape[1])
        log(f"[{variant}] 用 best epoch={ck['epoch']} 评测")

        per_subject = []
        for sid in args.eval_subjects:
            try:
                rec = eval_subject(model, sid, IMG_train, IMG_test, args, device)
                if not args.skip_posaxis:
                    # 位置轴分组
                    ch_names, _ = E89.load_train_chunk(sid, 1)
                    groups, _ = E89.build_groups(ch_names)
                    rec["posaxis"] = eval_posaxis(
                        model, sid, groups["POST"], groups["ANT"], n_ch,
                        IMG_test, args, device,
                        np.random.default_rng(SEED + sid))
                per_subject.append(rec)
                log(f"  sub-{sid:02d}: acc1={rec['zeroshot']['acc1_cos']:.4f} "
                    f"featavg={rec['zeroshot']['acc1_featavg_cos']:.4f} "
                    f"ridge={rec['ridge_feat']['acc1_cos']:.4f} "
                    f"RSA={rec['rsa']['r']:+.4f}")
            except Exception as e:
                per_subject.append({"subject": f"sub-{sid:02d}",
                                    "error": f"{type(e).__name__}: {e}"[:300],
                                    "trace": traceback.format_exc()[-800:]})
                log(f"  ✗ sub-{sid:02d}: {per_subject[-1]['error']}")

        per_subject_by_variant[variant] = per_subject
        summaries[variant] = summarize(per_subject)
        (vdir / f"exp94_{variant}.json").write_text(
            json.dumps({"variant": variant, "config": vars(args),
                        "train_best": best, "hist": hist,
                        "summary": summaries[variant],
                        "per_subject": per_subject},
                       indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        log(f"[{variant}] 汇总 {json.dumps(summaries[variant], ensure_ascii=False)}")

    dec = decide(summaries, per_subject_by_variant, args)
    log(f"\n{'='*74}\n=== 预登记判据 ===\n{'='*74}")
    log(json.dumps(dec, indent=2, ensure_ascii=False, default=str))

    (out_dir / "exp94_trunk_ablation.json").write_text(
        json.dumps({"config": vars(args), "align": al,
                    "summaries": summaries, "per_subject": per_subject_by_variant,
                    "train_best": all_best, "decision": dec},
                   indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"结果写入 {out_dir}/exp94_trunk_ablation.json")


if __name__ == "__main__":
    main()
