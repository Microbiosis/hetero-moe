"""构建真正的四维 EEG 数据集（不压平任何一维）。

## 原始结构（已探明）

    test.pt: dict with
      eeg:    (200, 80, 63, 250)  float16
      label:  (200, 80)
      img:    (200, 80)
      text:   (200, 80)
      times:  (300,)
      ch_names: 63

四个轴：
    轴 1 = 图像条件 condition (200)
    轴 2 = 重复试次 repetition (80)      ← 这是我之前完全丢掉的
    轴 3 = 电极 space (63)
    轴 4 = 时间 time (250 @ 250Hz = 1000ms)

## 为什么这是"四维"而不是"二维压平"

前 29 个实验做的是：
    X = eeg.mean(axis=时间)     → (200*80, 63)   二维
本构建保留全部四轴，并额外派生第五轴（频率）。

## 每个轴都要有可验证的信息

    condition 轴：不同图像是否可分
    repetition 轴：split-half 可靠性
    channel 轴：电极相关结构
    time 轴：ERP 波形
    band 轴：频带能量的条件差异

若某轴无信息，如实报告"该轴是空的"。
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

SRC = Path("/td/eeg/things-eeg/Preprocessed_data_250Hz_whiten")
FS = 250.0
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
N_SUBJECTS = 10


def band_power(eeg, fs=FS):
    """eeg: (n_img, n_rep, n_ch, T) → (n_img, n_rep, n_ch, n_band) 平均功率。"""
    T = eeg.shape[-1]
    win = np.hanning(T).astype(np.float32)
    seg = eeg * win
    F = np.fft.rfft(seg, axis=-1)
    P = (np.abs(F) ** 2).astype(np.float32)
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    out = np.zeros(eeg.shape[:-1] + (len(BANDS),), dtype=np.float32)
    for i, (lo, hi) in enumerate(BANDS):
        m = (freqs >= lo) & (freqs < hi)
        if m.sum() == 0:
            continue
        out[..., i] = P[..., m].mean(axis=-1)
    return out


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/out/omnibrain4d/eeg")
    out.mkdir(parents=True, exist_ok=True)

    import torch

    print("=" * 84)
    print("构建四维 EEG 数据集（图像, 试次, 电极, 时间）+ 频率")
    print("=" * 84)
    print(f"  src {SRC}")
    print(f"  dst {out}")
    print(f"  bands {dict(zip(BAND_NAMES, BANDS))}")

    t0 = time.time()
    subjects = []
    all_meta = []
    axis_checks = {}

    for i in range(1, N_SUBJECTS + 1):
        sid = f"sub-{i:02d}"
        f = SRC / sid / "test.pt"
        if not f.exists():
            print(f"  [skip] {sid}: no test.pt")
            continue

        print(f"\n  --- {sid} ---")
        obj = torch.load(f, map_location="cpu", weights_only=False)
        eeg = obj["eeg"]
        label = obj["label"]
        img = obj["img"]
        text = obj["text"]

        print(f"    eeg {eeg.shape} {eeg.dtype} | label {label.shape}")
        n_img, n_rep, n_ch, T = eeg.shape

        # 轴检验 1: label 沿 repetition 恒定？
        lab_const = bool(np.all(label == label[:, :1]))
        print(f"    label constant over rep: {lab_const}")

        # 轴检验 2: repetition 轴 split-half 可靠性
        m1 = eeg[:, :n_rep // 2].mean(axis=1).reshape(n_img, -1).astype(np.float32)
        m2 = eeg[:, n_rep // 2:].mean(axis=1).reshape(n_img, -1).astype(np.float32)
        m1c = m1 - m1.mean(axis=0, keepdims=True)
        m2c = m2 - m2.mean(axis=0, keepdims=True)
        num = (m1c * m2c).sum(axis=1)
        den = np.sqrt((m1c ** 2).sum(axis=1) * (m2c ** 2).sum(axis=1)) + 1e-9
        rep_reliability = float(np.median(num / den))
        print(f"    rep reliability (split-half): {rep_reliability:+.4f}")

        # 存四维原始
        np.savez_compressed(out / f"{sid}.npz", eeg=eeg.astype(np.float16))

        # 派生频率轴
        bp = band_power(eeg.astype(np.float32))
        np.savez_compressed(out / f"{sid}_band.npz", band=bp)
        print(f"    band {bp.shape} -> saved")

        # 轴检验 3: band 轴条件差异
        bp_flat = bp.mean(axis=1)
        bm = bp_flat.mean(axis=1)
        band_cv = float((bm.std(axis=0) / (np.abs(bm.mean(axis=0)) + 1e-12)).mean())
        print(f"    band CV across images: {band_cv:.4f}")

        for a in range(n_img):
            for b in range(n_rep):
                all_meta.append({
                    "subject": sid, "image_idx": a, "rep": b,
                    "label": int(label[a, b]), "img": str(img[a, b]),
                    "text": str(text[a, b]),
                })

        axis_checks[sid] = {
            "shape": list(eeg.shape),
            "label_constant_over_rep": lab_const,
            "rep_reliability": round(rep_reliability, 5),
            "band_cv": round(band_cv, 5),
            "n_ch": int(n_ch), "T": int(T), "fs": FS,
        }
        subjects.append(sid)
        del obj, eeg, bp
        gc.collect()
        print(f"    done ({time.time()-t0:.0f}s)")

    meta = pd.DataFrame(all_meta)
    meta.to_parquet(out / "meta.parquet", index=False)
    print(f"\n  meta: {len(meta):,} rows, cols {list(meta.columns)}")

    info = {
        "schema_version": "1.0",
        "dataset": "THINGS-EEG 4D",
        "axes": {
            "image": {"size": 200, "meaning": "image condition (200 test images)",
                      "type": "categorical"},
            "repetition": {"size": 80, "meaning": "repeated presentation of same image",
                           "type": "trial"},
            "channel": {"size": 63, "meaning": "EEG electrodes", "type": "spatial"},
            "time": {"size": 250, "meaning": "post-stimulus window",
                     "unit": "sample @250Hz = 1000ms", "type": "temporal"},
            "frequency": {"size": len(BANDS), "meaning": "band power (derived)",
                          "bands": {n: list(b) for n, b in zip(BAND_NAMES, BANDS)},
                          "type": "spectral"},
        },
        "n_subjects": len(subjects),
        "subjects": subjects,
        "per_subject_checks": axis_checks,
        "files": {
            "raw": "{sid}.npz key=eeg shape=(200,80,63,250) float16",
            "band": "{sid}_band.npz key=band shape=(200,80,63,5) float32",
            "meta": "meta.parquet per (image, rep): label/img/text",
        },
        "why_4d": (
            "Prior 29 experiments averaged the time axis to get (n_trial, n_ch). "
            "This dataset keeps all four original axes plus a derived spectral axis."
        ),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "axis_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False),
                                        encoding="utf-8")

    print("\n" + "=" * 84)
    print("轴检验汇总")
    print("=" * 84)
    for sid, c in axis_checks.items():
        print(f"  {sid}: shape={c['shape']} "
              f"rep_rel={c['rep_reliability']:+.4f} band_cv={c['band_cv']:.4f}")

    reps = [c["rep_reliability"] for c in axis_checks.values()]
    bandcvs = [c["band_cv"] for c in axis_checks.values()]
    if reps:
        print(f"\n  rep reliability median {np.median(reps):+.4f}")
        print(f"  band CV median {np.median(bandcvs):.4f}")

    print(f"\n  done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
