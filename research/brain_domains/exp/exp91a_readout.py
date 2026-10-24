#!/usr/bin/env python3
"""exp91a_readout.py — **读出强度检验**：前部「无语义」是不是只因我们这个读出太弱？

## 为什么做这个（2026-10-24 外部核对的直接后果）

外部核对 ①（`FINDINGS_TIER0_RSA.md` §二十七·一）发现：
**同一份 THINGS-EEG**，NICE-EEG（arXiv:2308.13234）用「时间+空间+频谱+语义」的编码器
拿到 **200-way 零样本 top-1 15.6%**，而我们（630-d 逐通道幅度 + ridge）只有 **1.1%**
⇒ **我们的读出比这个数据集的标准做法弱约 10–14×**
⇒ ⇒ 我们 A14 的「前部不携带图像语义」**只在我们这个读出下成立**
⇒ ⇒ 必须换**更强读出**重做位置轴的分组比较。

## 三个读出（同一份数据、同一套协议、零训练）

| 读出 | 特征 | 维度（ALL） | 补上了什么 |
|---|---|---|---|
| `amp630` | 逐通道幅度（10×100 ms 箱）—— **exp89 的原款** | 630 | （基线） |
| `bandpow` | 5 个频带（1–4 / 4–8 / 8–13 / 13–30 / 30–45 Hz）的**对数带功率** | 5×63 = 315 | **频谱**（exp89 完全没用） |
| `covpat` | 每试次**组内通道的协方差**上三角（空间模式） | 63×64/2 = 2016 | **跨通道协方差/空间模式**（exp89 完全丢掉） |

## 读法（预登记）

| 结果 | 结论 |
|---|---|
| 换强读出后，**后部仍高、前部仍≈0** | 位置轴分工**不是**读出太弱造成的 ⇒ A14/A15 站得住（去掉限定） |
| 换强读出后**前部也上来了** | 「前部无语义」**是**我们读出太弱 ⇒ A14/A15 必须长期保留限定 |

## 附带产出

逐通道的**视觉诱发幅度**（供 exp91b 扣掉全局左右增益差用）：
`AMPALL <63 个值>`，与 exp89 的 `amp_te` 同口径（峰峰值，48–500 ms）。

## 用法

    python -u research/brain_domains/exp/exp91a_readout.py --self-test

    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp91a_readout.py --subjects 1,2,3,4,5 \
      -v hf://buckets/illusion-brain/training-data:/training:ro \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor cpu-upgrade --timeout 120m -d
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
import exp73_decode200 as E73          # noqa: E402
import exp89_posaxis as E89            # noqa: E402

OUT_DIR = Path("/domains/runs/illusionbrain-align/exp91a")
FS = 250.0
GROUPS = ("ALL", "POST", "ANT", "OCC")
BANDS = ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))
SEED = 0


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 三个读出的特征提取
# ---------------------------------------------------------------------------

def amp_feats(eeg, chunk=20):
    """逐通道幅度（10×100 ms 箱）—— 与 exp89 逐字一致。"""
    return E73.per_rep_targets(eeg, chunk=chunk)


def bandpow_feats(eeg, chunk=20, bands=BANDS):
    """对数带功率，布局 **band-major**（dim = band*C + ch）。

    ⇒ 通道切片仍可用 `E89.channel_dims(ch_idx, C, n_bands)` ✓
    """
    n_img, n_rep, n_ch, T = eeg.shape
    freqs = np.fft.rfftfreq(T, d=1.0 / FS)
    masks = [(freqs >= lo) & (freqs < hi) for lo, hi in bands]
    nb = len(bands)
    out = np.empty((n_img, n_rep, n_ch * nb), dtype=np.float32)
    for s in range(0, n_img, chunk):
        x = eeg[s:s + chunk].astype(np.float32)
        F = np.abs(np.fft.rfft(x, axis=-1)) ** 2          # (n,rep,C,F)
        for b, m in enumerate(masks):
            p = F[:, :, :, m].mean(axis=-1)                # (n,rep,C)
            out[s:s + chunk, :, b * n_ch:(b + 1) * n_ch] = np.log(p + 1e-12)
    return out


def covpat_feats(eeg, ch_idx, chunk=10):
    """**组内**通道协方差的上三角（空间模式）⇒ 维度 |g|(|g|+1)/2。

    ⚠️ 协方差是**成对**的 ⇒ 组内重算（不能像维度那样切片）⇒ 每个组各算一次。
    """
    n_img, n_rep, n_ch, T = eeg.shape
    g = len(ch_idx)
    iu = np.triu_indices(g)
    out = np.empty((n_img, n_rep, len(iu[0])), dtype=np.float32)
    idx = np.asarray(ch_idx)
    for s in range(0, n_img, chunk):
        x = eeg[s:s + chunk][:, :, idx, :].astype(np.float32)   # (n,rep,g,T)
        x = x - x.mean(axis=-1, keepdims=True)
        # 逐试次的通道间协方差 C[g,s] = Σ_t x[g,t]·x[s,t]
        # ⚠️ 用**批量矩阵乘**而不是 einsum：einsum 里「出现在输入但不在输出的索引会被求和」
        #    （曾把试次轴意外求和掉，维度也跟着错）
        c = (x @ np.swapaxes(x, -1, -2)) / float(T - 1)          # (n,rep,g,g)
        out[s:s + chunk] = c[:, :, iu[0], iu[1]]
    return out


# ---------------------------------------------------------------------------
# 单被试
# ---------------------------------------------------------------------------

def run_subject(subject, IMG_test, IMG_train, args, rng):
    rec = {"subject": f"sub-{subject:02d}"}
    t0 = time.time()
    eeg_te, _ = E89.load_test_eeg(subject)
    n_img, n_rep, n_ch, T = eeg_te.shape
    ch_names, eeg_tr = E89.load_train_chunk(subject, args.train_images)
    groups, _labels = E89.build_groups(ch_names)
    log(f"  分组：" + "  ".join(f"{g}={len(groups[g])}" for g in GROUPS))

    # ---- 三个读出的特征 ----
    f_amp_te = amp_feats(eeg_te, chunk=args.chunk)
    f_amp_tr = amp_feats(eeg_tr, chunk=args.chunk)
    f_bp_te = bandpow_feats(eeg_te, chunk=args.chunk)
    f_bp_tr = bandpow_feats(eeg_tr, chunk=args.chunk)
    log(f"  特征：amp {f_amp_te.shape} | bandpow {f_bp_te.shape}")

    # 逐通道诱发幅度（exp91b 用）
    erp = E89.grand_erp(eeg_te)
    amp_te = E89.amp_map(erp)
    log("  AMPALL " + ",".join(f"{v:.4f}" for v in amp_te))

    res = {}
    # ---- 读出 1/2：可按通道切片 ----
    for tag, f_te, f_tr, n_per_ch in (("amp630", f_amp_te, f_amp_tr, 10),
                                      ("bandpow", f_bp_te, f_bp_tr, len(BANDS))):
        for g in GROUPS:
            idx = groups[g]
            if not idx:
                continue
            d = E89.channel_dims(idx, n_ch, n_per_ch)
            r, p = E89.rsa(f_te[:, :, d], IMG_test, args.n_perm, rng)
            res.setdefault(tag, {})[g] = {"r": r, "p": p, "n_ch": len(idx)}
            log(f"  RSA {tag:<8} {g:<5} n_ch={len(idx):<3} dim={len(d):<5} "
                f"r={r:+.4f} p={p:.4f}")
    # ---- 读出 3：空间模式（组内重算）----
    for g in GROUPS:
        idx = groups[g]
        if not idx:
            continue
        f_te = covpat_feats(eeg_te, idx, chunk=max(1, args.chunk // 2))
        r, p = E89.rsa(f_te, IMG_test, args.n_perm, rng)
        res.setdefault("covpat", {})[g] = {"r": r, "p": p, "n_ch": len(idx)}
        log(f"  RSA covpat   {g:<5} n_ch={len(idx):<3} "
            f"dim={f_te.shape[-1]:<5} r={r:+.4f} p={p:.4f}")
        if g in ("ALL", "POST", "ANT") and args.metric_b:
            f_tr = covpat_feats(eeg_tr, idx, chunk=max(1, args.chunk // 2))
            mu, sd = EVAL.zscore_fit(f_tr)
            rb = EVAL.metric_b_decode(EVAL.zscore_apply(f_tr, mu, sd),
                                      EVAL.zscore_apply(f_te, mu, sd),
                                      IMG_train[:args.train_images], IMG_test,
                                      rng, n_null=args.n_null)
            res.setdefault("metric_b_covpat", {})[g] = {
                "acc1": float(rb["acc1_cos"]),
                "null_acc1": float(np.median(rb["null_acc1_cos"]))}
            log(f"  MB  covpat   {g:<5} acc1={rb['acc1_cos']:.4f} "
                f"(零中位 {np.median(rb['null_acc1_cos']):.4f})")
    rec["readouts"] = res
    rec["amp_te"] = amp_te.tolist()
    rec["elapsed_s"] = round(time.time() - t0, 1)
    log(f"  用时 {rec['elapsed_s']}s")
    del eeg_te, eeg_tr, f_amp_te, f_amp_tr, f_bp_te, f_bp_tr
    return rec


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

def _synth(rng, IMG, ch, left_post, mode, d_img=16):
    """合成 EEG。

    mode='temporal'：**带符号**的幅度码（每个 100 ms 箱内恒定）⇒ 幅度读出能检出
    mode='spectral'：10 Hz 振荡，幅度 = (1+code) ⇒ **功率是平方** ⇒ 频带读出能检出
    （⚠️ 功率特征对**带符号**的幅度码是**盲**的 ⇒ 两种读出必须分别测）
    """
    n_img, n_rep, n_ch, T = IMG.shape[0], 4, 63, 250
    eeg = rng.normal(size=(n_img, n_rep, n_ch, T)).astype(np.float32) * 0.3
    W = rng.normal(size=(d_img, n_ch * 10)) * 0.0
    for c in left_post:
        for b in range(10):
            W[:, b * n_ch + c] = rng.normal(size=d_img) * 0.5
    t = np.arange(T) / FS
    for i in range(n_img):
        code = IMG[i] @ W                       # (n_ch*10,)
        if mode == "temporal":
            sig = code.reshape(10, n_ch)
            eeg[i] += np.repeat(sig.T[:, :, None], 25, axis=2).reshape(n_ch, T)[None]
        else:                                    # spectral
            sig = code.reshape(10, n_ch)
            # 每通道**一个**幅度（跨 10 个箱取均）⇒ 功率 ∝ (1+code)²
            amp = 1.0 + 0.6 * sig.T.mean(axis=1)   # (n_ch,)
            osc = np.sin(2 * np.pi * 10.0 * t)[None, :]
            eeg[i] += (amp[:, None] * osc)[None, :, :]
    return eeg


def self_test():
    log("=== 自测：三种读出（合成数据）===")
    rng = np.random.default_rng(21)
    n_img, n_ch, T, d_img = 24, 63, 250, 16
    ch = ["Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7", "F5", "F3",
          "F1", "F2", "F4", "F6", "F8", "FT9", "FT7", "FC5", "FC3", "FC1",
          "FCz", "FC2", "FC4", "FC6", "FT8", "FT10", "T7", "C5", "C3", "C1",
          "Cz", "C2", "C4", "C6", "T8", "TP9", "TP7", "CP5", "CP3", "CP1",
          "CPz", "CP2", "CP4", "CP6", "TP8", "TP10", "P7", "P5", "P3", "P1",
          "Pz", "P2", "P4", "P6", "P8", "PO7", "PO3", "POz", "PO4", "PO8",
          "O1", "Oz", "O2"]
    # 左右分组在 exp90_lateral 里（exp89 的 build_groups 只有解剖组）⇒ 惰性导入
    from exp90_lateral import build_lat_groups
    groups, _ = build_lat_groups(ch)
    assert len(groups["LEFT"]) == 28 and len(groups["RIGHT"]) == 28
    assert len(groups["MID"]) == 7
    import re

    def pre(nm):
        m = re.match(r"^(.*?)(\d+)$", nm)
        return m.group(1).upper() if m else nm.upper()

    left_post = [i for i in groups["LEFT"] if pre(ch[i]) in ("O", "PO", "P")]
    right_post = [i for i in groups["RIGHT"] if pre(ch[i]) in ("O", "PO", "P")]
    assert left_post and right_post
    IMG = rng.normal(size=(n_img, d_img))
    log(f"  植入通道（左后部 {len(left_post)} 个）: {[ch[i] for i in left_post]}")

    def lr_rsa(feats, n_per):
        r_l, _ = E89.rsa(E89.slice_group(feats, groups["LEFT"], n_ch, n_per),
                         IMG, 30, np.random.default_rng(1))
        r_r, _ = E89.rsa(E89.slice_group(feats, groups["RIGHT"], n_ch, n_per),
                         IMG, 30, np.random.default_rng(1))
        return r_l, r_r

    # ① 时间-幅度码 ⇒ 幅度读出能检出
    eeg_t = _synth(rng, IMG, ch, left_post, "temporal", d_img)
    r_l, r_r = lr_rsa(amp_feats(eeg_t, chunk=8), 10)
    log(f"  amp630（时间-幅度码）: 左 {r_l:+.4f} vs 右 {r_r:+.4f}")
    assert r_l > r_r, ("amp630", r_l, r_r)

    # ② 谱-功率码 ⇒ 频带读出能检出
    eeg_s = _synth(rng, IMG, ch, left_post, "spectral", d_img)
    r_l, r_r = lr_rsa(bandpow_feats(eeg_s, chunk=8), len(BANDS))
    log(f"  bandpow（谱-功率码）: 左 {r_l:+.4f} vs 右 {r_r:+.4f}")
    assert r_l > r_r, ("bandpow", r_l, r_r)

    # ③ 空间模式（协方差对码是**二次**的 ⇒ 与频带读出同属「平方型」）
    f_cov = covpat_feats(eeg_s, groups["LEFT"], chunk=4)
    r_l, _ = E89.rsa(f_cov, IMG, 30, np.random.default_rng(1))
    f_cov_r = covpat_feats(eeg_s, groups["RIGHT"], chunk=4)
    r_r, _ = E89.rsa(f_cov_r, IMG, 30, np.random.default_rng(1))
    log(f"  covpat（谱-功率码）: 左 {r_l:+.4f} vs 右 {r_r:+.4f}")
    assert r_l > r_r, ("covpat", r_l, r_r)
    log("✓ 三种读出都能检出植入的偏侧信号（各自对应自己的码型）")
    log("⚠️ 读出 ↔ 码型的对应关系（**这决定了「更强读出」未必对本题更敏感**）：")
    log("   · amp630（逐通道幅度）⇒ 对**带符号的时域幅度码**敏感")
    log("   · bandpow（频带功率）⇒ 功率是**平方** ⇒ 对**谱形码**敏感")
    log("   · covpat（通道协方差）⇒ 协方差对码也是**平方** ⇒ 与 bandpow 同类，"
        "外加「哪些通道共变」的空间结构")
    log("   ⇒ 真实数据里图像信息是哪种码型**未知** ⇒ 三种读出都要跑，"
        "不能预设「更强的读出一定更敏感」")
    log("=== 自测通过 ===")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--train-images", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-null", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--metric-b", type=int, default=1)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    import torch

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fte = torch.load(E89.FEAT_TEST, map_location="cpu", weights_only=False)
    IMG_test = fte["img_features"].numpy().astype(np.float64)
    ftr = torch.load(E89.FEAT_TRAIN, map_location="cpu", weights_only=False)
    IMG_train = ftr["img_features"].numpy().astype(np.float64)
    log(f"图像特征：test {IMG_test.shape} | train {IMG_train.shape}")

    per_subject = []
    for sid in [int(s) for s in args.subjects.split(",") if s.strip()]:
        log(f"\n=== sub-{sid:02d} ===")
        try:
            per_subject.append(run_subject(sid, IMG_test, IMG_train, args,
                                           np.random.default_rng(SEED + sid)))
        except Exception as e:
            rec = {"subject": f"sub-{sid:02d}",
                   "error": f"{type(e).__name__}: {e}"[:300],
                   "trace": traceback.format_exc()[-800:]}
            log(f"  ✗ {rec['error']}")
            log(rec["trace"])
            per_subject.append(rec)

    summary = {}
    for tag in ("amp630", "bandpow", "covpat"):
        for g in GROUPS:
            rs = [r["readouts"][tag][g]["r"] for r in per_subject
                  if "readouts" in r and tag in r["readouts"]
                  and g in r["readouts"][tag]]
            if not rs:
                continue
            srt = sorted(rs)
            summary.setdefault(tag, {})[g] = {
                "n": len(rs), "r_median": srt[len(srt) // 2],
                "r_all": rs}
    for g in ("ALL", "POST", "ANT"):
        v = [r["readouts"]["metric_b_covpat"][g]["acc1"] for r in per_subject
             if "readouts" in r and "metric_b_covpat" in r["readouts"]
             and g in r["readouts"]["metric_b_covpat"]]
        if v:
            summary.setdefault("metric_b_covpat", {})[g] = {
                "n": len(v), "acc1_median": sorted(v)[len(v) // 2]}
    log("\n=== 汇总 ===")
    for tag, e in summary.items():
        log(f"  {tag}: " + "  ".join(
            f"{g}={v.get('r_median', v.get('acc1_median', float('nan'))):+.4f}"
            for g, v in e.items()))
    (out_dir / "exp91a_readout.json").write_text(
        json.dumps({"config": vars(args), "per_subject": per_subject,
                    "summary": summary}, indent=2, ensure_ascii=False,
                   default=str), encoding="utf-8")
    log(f"\n结果写入 {out_dir}/exp91a_readout.json")


if __name__ == "__main__":
    main()
