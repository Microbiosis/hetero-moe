#!/usr/bin/env python3
"""exp95_trunk_hybrid.py — **把两个变体合成一个**（V1 + 时间分辨率）并做三向交叉验证。

## 为什么要做这个（exp94 留下的一个真实混淆）

exp94 的 V0 vs V1 **同时**改了两件事：

| | 通道身份 | 时间粒度 |
|---|---|---|
| **V0**（混通道）| ❌ 丢 | ✅ 31 个时间 token |
| **V1**（保留通道）| ✅ 保留 | ❌ 每通道 31→64 压成**一个** token |

⇒ exp94 只能判「**V1 整体更好**」（C1 上 10/10、p=0.0020），
**不能**说「增益来自通道身份」⇒ 而且 **V0 也没有任何 V1 缺的优势**
（C2/C3 上两者打平）⇒ 所以「把两个拼起来」若只是「求平均」，**动机在数据上不存在**。

⇒⇒ **V2 = 把两个变体合成一个**（不是折中，是**补上被混淆的第二个因子**）：
**保留 V1 的通道身份与跨通道注意力，同时把每通道的时间表示从 1 个 token 恢复成 k 个**。
⇒ 它们的差异被压到**一个维度**上：

| | 通道身份 | 时间粒度 | Transformer 看到的 token 数 |
|---|---|---|---|
| V1 | ✅ | ❌（每通道 1 个）| 63 |
| **V2** | ✅ | ✅（每通道 k=4 个）| **252（跨通道 + 跨时间块 一起交互）** |

⇒⇒ **这一个实验同时回答两件事**：
① 「两个构造成一个」行不行（V2 vs V1）；
② **V1 的增益到底来自通道身份还是别的**（V2 若 ≈ V1 ⇒ 增益来自通道身份 ⇒ V1 就是正解；
V2 若 > V1 ⇒ 时间粒度也重要 ⇒ 合成版更好）。

## 参数量匹配（**性能差异不可能来自容量**）

V1 1,582,327 ｜ V2 ≈ 1,593,000（+1.1%）｜ V0 1,623,872 ⇒ 三者同量级。
头**完全相同**（flatten(63×64)→256→1024）⇒ V2 只多了「chunk 投影 + chunk 嵌入 + 沿时间块池化」。

## 预登记判据（写死在脚本里，事后不改）

- **C7' 自一致性（两重）**：
  ① **V1 复现** vs exp94 的 V1（逐被试）⇒ 量化**两次 Job 之间**的漂移（争用/浮点）；
  ② V1 复现 vs exp93b 已发表值（沿用 exp94 的容差）。
  ⇒ 任一不过 ⇒ 本实验结论作废。
- **H1'（合成版更好）**：V2 在 **C1（单试次 200-way）** 上配对优于 V1（符号检验 p<0.05）
  ⇒ ⇒ 「合成一个」成立 ⇒ **底座用 V2**。
- **H2'（位置轴不反向）**：V2 的模型层位置轴 **POST>全部随机34 且 ANT<全部随机29** 的被试数
  **不低于 V1**（V1 是 10/10 & 10/10）⇒ ⇒ 合成版不破坏位置轴。
- **R4**：C1/C2/C3 三条读出方向不一致 ⇒ **不得**下「V2 全面更强」的单句结论。
- **总判决**：
  - `HYBRID_BETTER`（V2 > V1 显著 且 位置轴不反向）⇒ **用 V2**；
  - `HYBRID_REDUNDANT`（V2 ≈ V1）⇒ ⇒ **剃刀砍掉 V2**，**用 V1**，
    并把「**V1 的增益归因于通道身份**」写进台账（这才是这一轮真正的新信息）；
  - `HYBRID_WORSE`（V2 < V1）⇒ 时间 token 化有害 ⇒ **用 V1**。

## 用法

    python -u research/brain_domains/exp/exp95_trunk_hybrid.py --self-test

    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp95_trunk_hybrid.py \
      --train-subjects 1 2 3 4 5 6 7 8 9 --eval-subjects 1 2 3 4 5 6 7 8 9 10 \
      --epochs 30 --steps-per-epoch 1000 --pa-reps 20 --pa-rand 3 --pa-perm 100 \
      -v hf://buckets/illusion-brain/training-data:/training:ro \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor a10g-large --timeout 180m -d --name t95-hybrid
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

for _v in ("OMP_NUM_THREADS", "OPENBLAS_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

EXP_DIR = Path(__file__).resolve().parent
_PKG_CANDIDATES = [EXP_DIR / "models", EXP_DIR.parent / "models", EXP_DIR]
PKG_DIR = next((c for c in _PKG_CANDIDATES if (c / "registry.py").exists()),
               _PKG_CANDIDATES[0])
for _p in (str(EXP_DIR), str(PKG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import exp93b_contrastive as E93B        # noqa: E402
import exp94_trunk_ablation as E94        # noqa: E402

E89 = E93B.E89
E73 = E93B.E73

OUT_DIR = Path("/domains/runs/illusionbrain-align/exp95-hybrid")
SEED = 0
CHANCE = 1.0 / 200.0

# exp94 的 V1 结果（C7' 的 Job 间漂移锚点；**只用于量级检查**）
EXP94_V1 = {"zeroshot_acc1": 0.0626875, "zeroshot_acc1_featavg": 0.215,
             "rsa": 0.11921705748210504, "zeroshot_acc5": 0.1943125}
JOB_DRIFT_TOL = {"zeroshot_acc1": 0.010, "zeroshot_acc1_featavg": 0.030,
                  "rsa": 0.020, "zeroshot_acc5": 0.025}


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# V2：保留通道身份 **+** 恢复时间粒度（每通道 k 个时间 token）
# ---------------------------------------------------------------------------

def build_model_v2(n_ch=63, n_times=250, out_dim=1024, dim=64, n_chunks=4,
                   n_layers=4, nhead=4, head_hidden=256, t_stride=8):
    """V1 + 「每通道 k 个时间 token」⇒ Transformer 同时做**跨通道**与**跨时间块**交互。

    与 V1 的**唯一**差别：Transformer 的 token 集合从 63 个（每通道 1 个）
    变成 63×k 个（每通道 k 个时间块）⇒ **单因子**。
    头与 V1 **完全相同**（沿时间块可学加权池化回 63 个 token 再 flatten）。
    """
    import torch
    from torch import nn

    class PerChannelLN(nn.Module):
        """只沿时间归一化（reshape 成 (B*C,1,T)）⇒ 绝不跨通道混合。"""

        def __init__(self, n_times):
            super().__init__()
            self.ln = nn.LayerNorm(n_times)

        def forward(self, h):
            b, c, t = h.shape
            return self.ln(h.reshape(b * c, 1, t)).reshape(b, c, t)

    class ChunkTokenEncoder(nn.Module):
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
            # 31 个降采样后时间点 → n_chunks 段（每段长度取整除，余数并入最后一段）
            self.n_chunks = n_chunks
            self.chunk_len = self.n_times // n_chunks
            self.chunk_proj = nn.Linear(self.chunk_len, dim)
            self.ch_embed = nn.Parameter(torch.zeros(1, n_ch, n_chunks, dim))
            nn.init.normal_(self.ch_embed, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=nhead, dim_feedforward=4 * dim,
                dropout=0.0, batch_first=True, norm_first=True,
                activation="gelu")
            self.tr = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.chunk_w = nn.Parameter(torch.zeros(n_chunks))
            self.head = nn.Sequential(
                nn.LayerNorm(n_ch * dim),
                nn.Linear(n_ch * dim, head_hidden), nn.GELU(),
                nn.Linear(head_hidden, out_dim))

        def forward(self, x):
            h = self.t_down(self.enc(x))            # (B, C, T')
            b, c, tp = h.shape
            k, cl = self.n_chunks, self.chunk_len
            used = k * cl
            h = h[:, :, :used]                      # 丢掉余数（31 = 4×7 + 3）
            h = h.reshape(b, c, k, cl)              # (B, C, K, Tc)
            h = self.chunk_proj(h)                  # (B, C, K, dim)
            h = h + self.ch_embed
            h = h.reshape(b, c * k, dim)            # ⇒ 跨通道 + 跨时间块 一起交互
            h = self.tr(h)
            h = h.reshape(b, c, k, dim)
            w = torch.softmax(self.chunk_w, 0).reshape(1, 1, k, 1)
            h = (h * w).sum(dim=2)                  # (B, C, dim) 沿时间块池化
            return self.head(h.flatten(1))

    return ChunkTokenEncoder()


def build_model_variant(variant, n_ch, n_times, out_dim):
    if variant == "v1":
        return E94.build_model_v1(n_ch=n_ch, n_times=n_times, out_dim=out_dim)
    if variant == "v2":
        return build_model_v2(n_ch=n_ch, n_times=n_times, out_dim=out_dim)
    raise ValueError(variant)


# ---------------------------------------------------------------------------
# 训练（**两个变体共用同一份数据、同一个采样序列** ⇒ 严格受控）
# ---------------------------------------------------------------------------

def train_variant(args, IMG_train, device, EEG, IMGIDX, subjects, val_img,
                  perm, n_img, n_rep, n_ch, n_times, variant, out_dir):
    import torch
    import torch.nn.functional as F

    model = build_model_variant(variant, n_ch, n_times,
                                IMG_train.shape[1]).to(device)
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

    step_rng = np.random.default_rng(SEED)
    hist, best = [], {"val_acc": -1.0, "epoch": -1}
    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for _ in range(args.steps_per_epoch):
            xb, tb = sample_batch(step_rng)
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
                f"< {args.min_val_acc1} ⇒ 中止该变体")
            return None, None, hist

    torch.save({"state_dict": model.state_dict(), "args": vars(args),
                "epoch": args.epochs, "n_ch": n_ch, "n_times": n_times,
                "out_dim": IMG_train.shape[1], "variant": variant},
               out_dir / f"encoder_{variant}_last.pt")
    return model, best, hist


def reload_model(variant, ckpt_path, device):
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model_variant(variant, ck["n_ch"], ck["n_times"], ck["out_dim"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


# ---------------------------------------------------------------------------
# 判据（三向：exp94 的 V1 ↔ exp95 的 V1（漂移）↔ exp95 的 V2）
# ---------------------------------------------------------------------------

def _get(rec, path):
    cur = rec
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def paired(ps_a, ps_b, path):
    a = {r["subject"]: r for r in ps_a if "error" not in r}
    b = {r["subject"]: r for r in ps_b if "error" not in r}
    common = sorted(set(a) & set(b))
    diffs = []
    for s in common:
        va, vb = _get(a[s], path), _get(b[s], path)
        if va is None or vb is None:
            continue
        diffs.append(float(vb) - float(va))
    return {"n": len(diffs), "diffs": diffs,
            "median_diff": E94.med_up(diffs),
            "n_pos": sum(1 for d in diffs if d > 0),
            "p_sign": E94.sign_test_p(diffs)}


def decide(summaries, per, args):
    dec = {}
    v1, v2 = summaries.get("v1"), summaries.get("v2")
    if not v1 or not v2:
        return {"verdict": "INCOMPLETE", "reason": "至少一个变体没有结果"}

    # --- C7' ①：V1 复现 vs exp93b 已发表值（沿用 exp94 容差）---
    sc = {}
    for k, pub in E94.PUBLISHED_V0.items():
        got = v1.get(k)
        tol = E94.SELF_CONSISTENCY_TOL[k]
        sc[k] = {"published": pub, "got": got, "tol": tol,
                 "ok": bool(got is not None and abs(got - pub) <= tol)}
    dec["self_consistency_exp93b"] = sc
    sc_ok = all(v["ok"] for v in sc.values())

    # --- C7' ②：V1 复现 vs exp94 的 V1（**Job 间漂移**）---
    drift = {}
    for k, pub in EXP94_V1.items():
        got = v1.get(k)
        tol = JOB_DRIFT_TOL[k]
        drift[k] = {"exp94": pub, "got": got, "tol": tol,
                    "ok": bool(got is not None and abs(got - pub) <= tol)}
    dec["job_drift_vs_exp94"] = drift
    drift_ok = all(v["ok"] for v in drift.values())
    dec["self_consistency_pass"] = bool(sc_ok and drift_ok)

    # --- H1'：V2 vs V1（配对）---
    p1 = paired(per["v1"], per["v2"], ["zeroshot", "acc1_cos"])
    p2 = paired(per["v1"], per["v2"], ["zeroshot", "acc1_featavg_cos"])
    p3 = paired(per["v1"], per["v2"], ["rsa", "r"])
    dec["paired_v2_minus_v1"] = {"zeroshot_acc1": p1,
                                 "zeroshot_acc1_featavg": p2, "rsa": p3}
    paired_ok = all(p["n"] > 0 for p in (p1, p2, p3))
    dec["paired_ok"] = paired_ok
    # ⚠️ **方向必须一起查**：H1' 的原文是「V2 在 C1 上**优于** V1（p<0.05）」
    #    ⇒ 只看 p 是不够的 —— p 小也可能是因为「V2 **更差**」而 9/10 反向。
    #    （2026-10-25 踩过：这一版只看 p ⇒ 把「V2 显著更差」判成了 HYBRID_BETTER。）
    h1 = (paired_ok and p1["n_pos"] > p1["n"] / 2
          and p1["p_sign"] is not None and p1["p_sign"] < 0.05)
    dec["H1_hybrid_better"] = bool(h1)
    dec["H1_detail"] = {
        "C1_single_trial": {"v1": v1.get("zeroshot_acc1"), "v2": v2.get("zeroshot_acc1"),
                            "median_diff": p1["median_diff"], "n_pos": p1["n_pos"],
                            "n": p1["n"], "p": p1["p_sign"]},
        "C2_trial_avg": {"v1": v1.get("zeroshot_acc1_featavg"),
                         "v2": v2.get("zeroshot_acc1_featavg"),
                         "median_diff": p2["median_diff"], "n_pos": p2["n_pos"],
                         "n": p2["n"], "p": p2["p_sign"]},
        "C3_rsa": {"v1": v1.get("rsa"), "v2": v2.get("rsa"),
                   "median_diff": p3["median_diff"], "n_pos": p3["n_pos"],
                   "n": p3["n"], "p": p3["p_sign"]},
    }
    if not paired_ok:
        dec["readout_consistent"] = None
        dec["R4_note"] = "⚠️ 配对数为 0 ⇒ 无法判定"
    else:
        signs = [np.sign(p["median_diff"] or 0) for p in (p1, p2, p3)]
        dec["readout_consistent"] = bool(len(set(signs)) == 1)
        dec["R4_note"] = ("三条读出方向一致" if dec["readout_consistent"]
                          else "⚠️ 三条读出方向**不一致** ⇒ 不得下「V2 全面更强」的单句结论")

    # --- H2'：位置轴不反向 ---
    pa = {}
    for v in ("v1", "v2"):
        recs = [r for r in per.get(v, []) if "error" not in r and "posaxis" in r]
        if not recs:
            pa[v] = None
            continue
        pa[v] = {
            "post_above_all_rand34": sum(
                1 for r in recs if r["posaxis"]["verdict"]["post_above_all_rand34"]),
            "ant_below_all_rand29": sum(
                1 for r in recs if r["posaxis"]["verdict"]["ant_below_all_rand29"]),
            "post_minus_ant_median": E94.med_up(
                [r["posaxis"]["verdict"]["post_minus_ant"] for r in recs]),
            "post_acc1_median": E94.med_up(
                [r["posaxis"]["groups"]["POST"]["acc1"] for r in recs]),
            "ant_acc1_median": E94.med_up(
                [r["posaxis"]["groups"]["ANT"]["acc1"] for r in recs]),
            "n": len(recs),
        }
    dec["posaxis"] = pa
    if pa.get("v1") and pa.get("v2"):
        a, b = pa["v1"], pa["v2"]
        dec["H2_posaxis_not_worse"] = bool(
            b["post_above_all_rand34"] >= a["post_above_all_rand34"]
            and b["ant_below_all_rand29"] >= a["ant_below_all_rand29"])
    else:
        dec["H2_posaxis_not_worse"] = None

    # --- 总判决 ---
    if not dec["self_consistency_pass"]:
        dec["verdict"] = "INVALID_SELF_CONSISTENCY"
    elif dec["H1_hybrid_better"] and dec["H2_posaxis_not_worse"]:
        dec["verdict"] = "HYBRID_BETTER"
    elif paired_ok and p1["n_pos"] < p1["n"] / 2 and p1["p_sign"] is not None \
            and p1["p_sign"] < 0.05:
        # V2 在 C1 上**显著更差**（多数被试 V1 更好）⇒ 合成版有害 ⇒ 砍掉它
        dec["verdict"] = "HYBRID_WORSE"
    elif p1["n_pos"] and p1["n_pos"] > p1["n"] / 2 and p1["p_sign"] is not None \
            and p1["p_sign"] >= 0.05:
        dec["verdict"] = "HYBRID_REDUNDANT"
    else:
        dec["verdict"] = "HYBRID_INCONCLUSIVE"
    return dec


# ---------------------------------------------------------------------------

def self_test():
    import torch
    log("=== 自测：V1/V2 形状 + 参数量匹配 + 通道敏感度 + 端到端可学性 ===")
    m1 = E94.build_model_v1(n_ch=63, n_times=250, out_dim=1024)
    m2 = build_model_v2(n_ch=63, n_times=250, out_dim=1024)
    n1 = sum(p.numel() for p in m1.parameters())
    n2 = sum(p.numel() for p in m2.parameters())
    log(f"  V1 参数量 {n1:,}｜V2 参数量 {n2:,}｜比值 {n2/n1:.3f}")
    assert 0.90 <= n2 / n1 <= 1.12, "参数量没对齐 ⇒ 差异可能来自容量"
    with torch.no_grad():
        x = torch.randn(4, 63, 250)
        z1 = m1(x)
        z2 = m2(x)
    assert z1.shape == (4, 1024) and z2.shape == (4, 1024)
    log(f"  ✓ 输出形状 {tuple(z1.shape)} / {tuple(z2.shape)}")

    # 通道敏感度：改通道 5 ⇒ 输出必须变（V2 不得把通道抹掉）
    with torch.no_grad():
        x2 = x.clone()
        x2[:, 5, :] = x2[:, 5, :] + 3.0
        d = (m2(x2) - m2(x)).abs().max().item()
    log(f"  V2：改通道 5 ⇒ 输出最大变化 {d:.4f}")
    assert d > 1e-4, "V2 似乎对通道 5 不敏感 ⇒ 通道身份被抹掉了？"

    # 端到端可学性
    for name, mm in (("v1", m1), ("v2", m2)):
        torch.manual_seed(0)
        A = torch.randn(63, 1024) / 8.0
        xx = torch.randn(32, 63, 250)
        tgt = xx.mean(axis=2) @ A
        m = (E94.build_model_v1(n_ch=63, n_times=250, out_dim=1024)
             if name == "v1"
             else build_model_v2(n_ch=63, n_times=250, out_dim=1024))
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
        l0 = None
        for i in range(80):
            loss = E93B.info_nce(torch.nn.functional.normalize(m(xx), dim=1),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument("--eval-subjects", type=int, nargs="+",
                    default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ap.add_argument("--variants", default="v1,v2")
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
    ap.add_argument("--pa-reps", type=int, default=20)
    ap.add_argument("--pa-rand", type=int, default=3)
    ap.add_argument("--pa-perm", type=int, default=100)
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

    # ---------- 预检 P1：对齐 ----------
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

    # ---------- 数据（两个变体共用）----------
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
    log(f"[数据] 每被试 {n_img} 图 × {n_rep} 重复 × {n_ch} 通道 × {n_times} 样本")

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    summaries, per, all_best = {}, {}, {}

    for variant in variants:
        log(f"\n{'='*74}\n=== 变体 {variant} ===\n{'='*74}")
        vdir = out_dir / variant
        vdir.mkdir(parents=True, exist_ok=True)
        model, best, hist = train_variant(args, IMG_train, device, EEG, IMGIDX,
                                          subjects, val_img, perm, n_img, n_rep,
                                          n_ch, n_times, variant, vdir)
        all_best[variant] = best
        if model is None:
            log(f"✗ [{variant}] 训练中止")
            continue
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        model, ck = reload_model(variant, vdir / f"encoder_{variant}_best.pt", device)
        log(f"[{variant}] 用 best epoch={ck['epoch']} 评测")

        ps = []
        for sid in args.eval_subjects:
            try:
                rec = E94.eval_subject(model, sid, IMG_train, IMG_test, args, device)
                if not args.skip_posaxis:
                    ch_names, _ = E89.load_train_chunk(sid, 1)
                    groups, _ = E89.build_groups(ch_names)
                    rec["posaxis"] = E94.eval_posaxis(
                        model, sid, groups["POST"], groups["ANT"], n_ch,
                        IMG_test, args, device, np.random.default_rng(SEED + sid))
                ps.append(rec)
                log(f"  sub-{sid:02d}: acc1={rec['zeroshot']['acc1_cos']:.4f} "
                    f"featavg={rec['zeroshot']['acc1_featavg_cos']:.4f} "
                    f"ridge={rec['ridge_feat']['acc1_cos']:.4f} "
                    f"RSA={rec['rsa']['r']:+.4f}")
            except Exception as e:
                ps.append({"subject": f"sub-{sid:02d}",
                           "error": f"{type(e).__name__}: {e}"[:300],
                           "trace": traceback.format_exc()[-800:]})
                log(f"  ✗ sub-{sid:02d}: {ps[-1]['error']}")

        per[variant] = ps
        summaries[variant] = E94.summarize(ps)
        (vdir / f"exp95_{variant}.json").write_text(
            json.dumps({"variant": variant, "config": vars(args),
                        "train_best": best, "hist": hist,
                        "summary": summaries[variant], "per_subject": ps},
                       indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        log(f"[{variant}] 汇总 {json.dumps(summaries[variant], ensure_ascii=False)}")

    dec = decide(summaries, per, args)
    log(f"\n{'='*74}\n=== 预登记判据 ===\n{'='*74}")
    log(json.dumps(dec, indent=2, ensure_ascii=False, default=str))

    (out_dir / "exp95_trunk_hybrid.json").write_text(
        json.dumps({"config": vars(args), "align": al,
                    "summaries": summaries, "per_subject": per,
                    "train_best": all_best, "decision": dec},
                   indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"结果写入 {out_dir}/exp95_trunk_hybrid.json")


if __name__ == "__main__":
    main()
