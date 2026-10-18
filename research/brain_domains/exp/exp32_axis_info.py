"""验证四维数据集每个轴的信息量 —— 不能只是"有轴"，要"轴里有东西"。

## 五个轴，五种检验

| 轴 | 检验 | 若通过说明 |
|---|---|---|
| condition (200 图像) | 交叉验证解码准确率 | 图像身份可从响应区分 |
| repetition (80) | ICC / split-half | 单试次有可靠个体差异 |
| channel (63) | 电极间相关块性 | 有空间结构 |
| time (250) | ERP 的 SNR | 有锁时响应 |
| frequency (5) | 频带能量的条件 F 比 | 频带区分条件 |

## 关键设计：全部在同一份四维数据上做，不压平

- 解码用 (rep, channel, time) → 图像，逐试次输入
- ICC 用 repetition 轴的方差分解
- 块性用 (channel × channel) 相关
- ERP 用时间轴上的信噪比
- 频带用 band 数组

## 判据

  解码准确率 > 随机（1/200 = 0.5%）的 10 倍 → condition 轴有效
  ICC > 0.05 → repetition 轴有效
  块性 > 1.5 → channel 轴有效
  ERP SNR > 1 → time 轴有效
  band F 比显著 → frequency 轴有效
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

SEED = 0
FS = 250.0
N_SUBJECTS = 10


# ------------------------------------------------------------ 检验 1: 解码
def decode_condition(eeg, n_folds=2, n_rep_use=40):
    """用 (channel, time) 解码 image identity。

    eeg: (n_img, n_rep, n_ch, T)
    用最近类均值（NCM）分类器：简单、无需训练、对小样本稳健。
    """
    n_img, n_rep, n_ch, T = eeg.shape
    X = eeg[:, :, :, :].astype(np.float32)
    # 逐通道 z-score（去电极间幅度差）
    X = X.reshape(n_img * n_rep, n_ch, T)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-6
    X = (X - mu) / sd
    X = X.reshape(n_img, n_rep, n_ch * T)

    # 只用部分试次降计算量
    r = min(n_rep_use, n_rep)
    X = X[:, :r]

    # split-half: 前一半训练类均值，后一半测试
    h = r // 2
    train = X[:, :h].mean(axis=1)          # (n_img, feat)
    test = X[:, h:]                        # (n_img, r-h, feat)

    # 最近类均值（用欧氏距离）
    tr = train / (np.linalg.norm(train, axis=1, keepdims=True) + 1e-9)
    te = test / (np.linalg.norm(test, axis=2, keepdims=True) + 1e-9)
    # (n_img_test, n_rep_test, n_img_train)
    sim = np.einsum("irf,jf->irj", te, tr)
    pred = sim.argmax(axis=2)              # (n_img, n_rep_test)
    truth = np.arange(n_img)[:, None]
    acc = float((pred == truth).mean())
    return acc


# ------------------------------------------------------------ 检验 2: ICC
def icc_repetition(eeg):
    """ICC(1)：repetition 轴的方差占比。

    eeg: (n_img, n_rep, n_ch, T)
    先把 (ch, T) 压成特征，再算组间/组内方差。
    """
    n_img, n_rep = eeg.shape[:2]
    X = eeg.reshape(n_img, n_rep, -1).astype(np.float32)
    # 只取部分特征降计算量
    if X.shape[2] > 2000:
        X = X[:, :, ::max(1, X.shape[2] // 2000)]
    grand = X.mean()
    ss_between = n_rep * ((X.mean(axis=1) - grand) ** 2).sum()
    ss_within = ((X - X.mean(axis=1, keepdims=True)) ** 2).sum()
    tot = ss_between + ss_within
    return float(ss_between / tot) if tot > 1e-12 else 0.0


# ------------------------------------------------------------ 检验 3: 块性
def channel_blockiness(eeg):
    """电极相关矩阵经准对角化后的块性。"""
    n_img, n_rep, n_ch, T = eeg.shape
    X = eeg[:, :, :, :].astype(np.float32)
    # 每电极的时间序列（跨图像与试次拼接）
    M = X.transpose(2, 0, 1, 3).reshape(n_ch, -1)
    C = np.corrcoef(M)
    C = np.nan_to_num(C, nan=0.0)
    d = n_ch
    dist = np.clip(1.0 - np.abs(C), 0, 2)
    np.fill_diagonal(dist, 0.0)
    try:
        Z = linkage(squareform(dist, checks=False), method="average")
        order = leaves_list(Z)
    except Exception:
        return np.nan, np.nan
    Ca = np.abs(C)
    np.fill_diagonal(Ca, 0.0)
    Cr = Ca[np.ix_(order, order)]
    adj = float(Cr[np.arange(d - 1), np.arange(1, d)].mean())
    glob = float(Ca[np.triu_indices(d, 1)].mean())
    return adj / max(glob, 1e-9), glob


# ------------------------------------------------------------ 检验 4: ERP SNR
def erp_snr(eeg):
    """时间轴：平均响应的幅度 / 单试次噪声。"""
    n_img, n_rep, n_ch, T = eeg.shape
    X = eeg.astype(np.float32)
    # 跨图像与试次的平均 ERP
    erp = X.mean(axis=(0, 1))                      # (n_ch, T)
    # 单试次相对 ERP 的残差
    resid = X - erp[None, None]
    sig = float((erp ** 2).mean())
    noise = float((resid ** 2).mean()) / max(n_rep, 1)
    return sig / max(noise, 1e-12), float(np.abs(erp).max())


# ------------------------------------------------------------ 检验 5: 频带
def band_effect(band):
    """band: (n_img, n_rep, n_ch, n_band)。图像间 band 能量差异的 F 比。

    用单因素 ANOVA 的 F = 组间均方 / 组内均方。
    """
    n_img, n_rep, n_ch, n_band = band.shape
    X = band.astype(np.float32)                    # 条件 = 图像
    # 对每个 band：跨 ch 平均后，算图像间 vs 图像内
    Y = X.mean(axis=2)                             # (n_img, n_rep, n_band)
    grand = Y.mean(axis=(0, 1), keepdims=True)
    ss_b = n_rep * ((Y.mean(axis=1, keepdims=True) - grand) ** 2).sum(axis=(0, 1))
    ss_w = ((Y - Y.mean(axis=1, keepdims=True)) ** 2).sum(axis=(0, 1))
    df_b = n_img - 1
    df_w = n_img * (n_rep - 1)
    F = (ss_b / df_b) / (ss_w / max(df_w, 1) + 1e-12)
    return F.astype(np.float64)


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/omnibrain4d/eeg")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/out/exp32")
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 86)
    print("Phase 32 — 四维数据集：每个轴的信息量验证")
    print("=" * 86)
    print(f"  src {src}")

    rows = []
    t0 = time.time()

    for i in range(1, N_SUBJECTS + 1):
        sid = f"sub-{i:02d}"
        f_raw = src / f"{sid}.npz"
        f_band = src / f"{sid}_band.npz"
        if not f_raw.exists():
            continue

        print(f"\n  --- {sid} ---")
        eeg = np.load(f_raw)["eeg"]
        band = np.load(f_band)["band"] if f_band.exists() else None
        print(f"    eeg {eeg.shape} | band {band.shape if band is not None else None}")

        acc = decode_condition(eeg)
        chance = 1.0 / eeg.shape[0]
        print(f"    [condition] 解码准确率 {acc:.4f}  (随机 {chance:.4f}, "
              f"倍数 {acc/chance:.1f}x)")

        icc = icc_repetition(eeg)
        print(f"    [repetition] ICC = {icc:.4f}")

        blk, glob = channel_blockiness(eeg)
        print(f"    [channel] 块性 {blk:.3f} (全局相关 {glob:.4f})")

        snr, amp = erp_snr(eeg)
        print(f"    [time] ERP SNR = {snr:.3f} (峰值 {amp:.3f})")

        if band is not None:
            F = band_effect(band)
            print(f"    [frequency] F 比: " +
                  ", ".join(f"{n}={v:.1f}" for n, v in
                            zip(["δ", "θ", "α", "β", "γ"], F)))
            band_f = float(np.mean(F))
        else:
            band_f = np.nan

        rows.append({
            "subject": sid,
            "decode_acc": round(acc, 5), "chance": round(chance, 5),
            "acc_ratio": round(acc / chance, 2),
            "icc": round(icc, 5),
            "blockiness": round(blk, 4) if np.isfinite(blk) else None,
            "global_corr": round(glob, 5) if np.isfinite(glob) else None,
            "erp_snr": round(snr, 3),
            "band_F_mean": round(band_f, 2) if np.isfinite(band_f) else None,
        })
        del eeg, band

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase32.csv", index=False)
    print(f"\n  记录 {len(df)} | 耗时 {time.time()-t0:.0f}s")

    if not len(df):
        print("无数据")
        return

    print("\n" + "=" * 86)
    print("五个轴的信息量汇总（中位）")
    print("=" * 86)
    g = df.median(numeric_only=True)
    print(f"  condition  解码准确率 {g['decode_acc']:.4f} / 随机 {g['chance']:.4f} "
          f"= {g['acc_ratio']:.1f}x")
    print(f"  repetition ICC        {g['icc']:.4f}")
    print(f"  channel    块性       {g['blockiness']:.3f}")
    print(f"  time       ERP SNR    {g['erp_snr']:.2f}")
    print(f"  frequency  band F     {g['band_F_mean']:.1f}")

    # 判定
    print("\n" + "=" * 86)
    print("轴有效性判定")
    print("=" * 86)
    verdicts = {}

    verdicts["condition"] = {
        "value": round(float(g["decode_acc"]), 4), "chance": round(float(g["chance"]), 4),
        "valid": bool(g["acc_ratio"] > 10),
        "note": f"{g['acc_ratio']:.0f}x chance",
    }
    verdicts["repetition"] = {
        "value": round(float(g["icc"]), 4),
        "valid": bool(g["icc"] > 0.05),
        "note": "ICC(1) 组间方差占比",
    }
    verdicts["channel"] = {
        "value": round(float(g["blockiness"]), 3),
        "valid": bool(g["blockiness"] > 1.5),
        "note": "准对角化后相邻相关/全局相关",
    }
    verdicts["time"] = {
        "value": round(float(g["erp_snr"]), 2),
        "valid": bool(g["erp_snr"] > 1.0),
        "note": "平均响应方差/噪声方差",
    }
    verdicts["frequency"] = {
        "value": round(float(g["band_F_mean"]), 1),
        "valid": bool(g["band_F_mean"] > 1.5),
        "note": "图像间频带能量 ANOVA F 比",
    }

    for axis, v in verdicts.items():
        mark = "✅ 有效" if v["valid"] else "⚠️ 弱/无效"
        print(f"  {axis:<12} {v['value']:>10}  {mark:<10} ({v['note']})")

    n_valid = sum(1 for v in verdicts.values() if v["valid"])
    print(f"\n  {n_valid}/5 个轴携带可检测信息")

    (out / "phase32_report.json").write_text(json.dumps({
        "schema_version": "1.0",
        "dataset": "THINGS-EEG 4D",
        "shape_per_subject": [200, 80, 63, 250],
        "axes_tested": verdicts,
        "n_axes_valid": n_valid,
        "per_subject": rows,
        "method_notes": {
            "condition": "nearest class mean on split-half, chance=1/200",
            "repetition": "ICC(1) between-image variance / total",
            "channel": "quasi-diagonalization blockiness",
            "time": "ERP variance / single-trial noise variance",
            "frequency": "one-way ANOVA F across images, averaged over 5 bands",
        },
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
