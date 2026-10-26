"""构建 THINGS-EEG 4D 的 **train** 集（16540 图像 × 4 重复 × 10 被试）。

## 与 test 集的差别

| | test | train（本脚本）|
|---|---:|---:|
| 图像 | 200 | **16,540** |
| 重复 | 80 | **4** |
| 每被试试次 | 16,000 | 66,160 |
| 每被试文件 | 445 MB | ~1.94 GB |
| 10 被试总计 | 4.5 GB | **20.2 GB** |
| 试次间信度 | +0.072 | +0.005 |

**注意**：train 是「图像多、重复少」，与 test 互补。这不是错误，
是 THINGS-EEG 的原始设计。

## 输出布局

    数据集仓库 illusion-brain/things-eeg-4d 的 train/
    ├── README.md
    ├── axis_info.json
    ├── meta.parquet              (1,323,200 行)
    ├── sub-01.npz                eeg (16540, 4, 63, 250) float16
    ├── sub-01_band.npz           band (16540, 4, 63, 5)  float32
    └── ...                        ×10

    ⚠️ 这两行的形状**曾经写错**（写成 (1654, 80)）⇒ 与本文档第 1 行的
    「16540 图像 × 4 重复」自相矛盾 ⇒ 害我据此推出过「指标 B 泄漏」的**错误结论**
    （见 FINDINGS §十九 的纠错记录）。实测：文件 2,084,040,260 B
    ÷ (63×250×2) = 66,160 = **16540×4**（注意 float16 是 **2** 字节/元素）。
    ⇒ 形状以**实测**为准，不要用这份 docstring 的数字去推别的东西。

## 关键决策

1. **float16 保存**：原始是 float32（2.1 GB/被试），转 float16 后 ~3.6 GB 仍大。
   若存 float32 会是 7.2 GB/被试 = 72 GB 总计。**必须 float16。**

2. **band 单独文件**：raw 已经很大，band 另存便于只想用频带的场景。

3. **逐被试独立处理**：避免一次性载入 21 GB。每个被试载入、转换、写入、释放。

4. **原子写入**：先写 .tmp 再 rename，避免中断留下半个文件。

## 内存估算

单个被试 train.pt 约 2.1 GB（float16 存储）。
载入 + 转 float32 算 band 时峰值约 6-8 GB。Space 有 96 GB，够。

## 时间估算

- 载入 2.1 GB：~10 s
- FFT + 频带：~20 s
- 写 3.6 GB：~30 s
每被试约 60 s，10 个约 10 分钟。
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

FS = 250.0
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]

SRC = Path("/training/eeg/things-eeg/Preprocessed_data_250Hz_whiten")
OUT = Path("/domains/omnibrain4d/eeg_train")   # bucket 内部路径不变
SUBJECTS = list(range(1, 11))



def band_power(eeg, fs=FS):
    """eeg: (n_img, n_rep, n_ch, T) float32 → (..., n_band) float32"""
    T = eeg.shape[-1]
    win = np.hanning(T).astype(np.float32)
    P = np.abs(np.fft.rfft(eeg * win, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    out = np.zeros(eeg.shape[:-1] + (len(BANDS),), dtype=np.float32)
    for i, (lo, hi) in enumerate(BANDS):
        m = (freqs >= lo) & (freqs < hi)
        if m.sum():
            out[..., i] = P[..., m].mean(axis=-1)
    return out


def atomic_savez(path: Path, **arrays):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 84, flush=True)
    print("构建 THINGS-EEG 4D train 集", flush=True)
    print("=" * 84, flush=True)
    print(f"  源 {SRC}", flush=True)
    print(f"  输出 {OUT}", flush=True)

    if not SRC.exists():
        print(f"❌ 源目录不存在: {SRC}", flush=True)
        print("   检查 /training 卷是否挂载", flush=True)
        return

    t0 = time.time()
    done = []
    checks = {}
    all_meta = []

    for sid in SUBJECTS:
        sub = f"sub-{sid:02d}"
        f = SRC / sub / "train.pt"
        if not f.exists():
            print(f"  [skip] {sub}: 无 train.pt", flush=True)
            continue

        t1 = time.time()
        size_gb = f.stat().st_size / 1024 ** 3
        print(f"\n  --- {sub} ({size_gb:.2f} GB) ---", flush=True)

        import torch
        obj = torch.load(f, map_location="cpu", weights_only=False)
        eeg = obj["eeg"]
        label = obj["label"]
        img = obj["img"]
        text = obj.get("text", None)
        print(f"    eeg {eeg.shape} {eeg.dtype} | 载入 {time.time()-t1:.0f}s",
              flush=True)

        n_img, n_rep, n_ch, T = eeg.shape

        # ---- 轴检验
        lab_const = bool(np.all(label == label[:, :1]))
        print(f"    label 沿 rep 恒定: {lab_const}", flush=True)

        # 试次信度（抽样，全量太慢）
        k = min(50, n_img)
        idx = np.arange(0, n_img, max(1, n_img // k))[:k]
        e_s = eeg[idx]
        h = n_rep // 2
        m1 = e_s[:, :h].mean(axis=1).reshape(len(idx), -1).astype(np.float32)
        m2 = e_s[:, h:].mean(axis=1).reshape(len(idx), -1).astype(np.float32)
        m1 = m1 - m1.mean(0, keepdims=True)
        m2 = m2 - m2.mean(0, keepdims=True)
        num = (m1 * m2).sum(1)
        den = np.sqrt((m1 ** 2).sum(1) * (m2 ** 2).sum(1)) + 1e-9
        rep_rel = float(np.median(num / den))
        print(f"    rep 信度（{len(idx)} 抽样）: {rep_rel:+.4f}", flush=True)

        # ---- 写 raw（float16）
        t2 = time.time()
        eeg16 = eeg.astype(np.float16)
        atomic_savez(OUT / f"{sub}.npz", eeg=eeg16)
        raw_gb = (OUT / f"{sub}.npz").stat().st_size / 1024 ** 3
        print(f"    raw 已写 {raw_gb:.2f} GB ({time.time()-t2:.0f}s)", flush=True)
        del eeg16
        gc.collect()

        # ---- 算 band（需要 float32）
        t3 = time.time()
        bp = band_power(eeg.astype(np.float32))
        del eeg
        gc.collect()
        atomic_savez(OUT / f"{sub}_band.npz", band=bp)
        band_gb = (OUT / f"{sub}_band.npz").stat().st_size / 1024 ** 3
        print(f"    band 已写 {band_gb:.2f} GB ({time.time()-t3:.0f}s)", flush=True)
        del bp
        gc.collect()

        # ---- meta（抽样写入以免内存爆：1654×80=132,320 行/被试）
        for a in range(n_img):
            for b in range(n_rep):
                all_meta.append({
                    "subject": sub, "image_idx": a, "rep": b,
                    "label": int(label[a, b]), "img": str(img[a, b]),
                    "text": str(text[a, b]) if text is not None else "",
                })
        print(f"    meta +{n_img * n_rep:,} 行", flush=True)

        checks[sub] = {
            "shape": list(eeg.shape) if False else [n_img, n_rep, n_ch, T],
            "label_constant_over_rep": lab_const,
            "rep_reliability": round(rep_rel, 5),
            "raw_gb": round(raw_gb, 2),
            "band_gb": round(band_gb, 2),
        }
        done.append(sub)
        del obj, label
        gc.collect()
        print(f"    [{time.time()-t1:.0f}s] 累计 {(time.time()-t0)/60:.1f} min",
              flush=True)

    if not done:
        print("\n❌ 无成功被试", flush=True)
        return

    # ---- meta
    t4 = time.time()
    meta = pd.DataFrame(all_meta)
    meta.to_parquet(OUT / "meta.parquet", index=False)
    print(f"\n  meta 已写 {len(meta):,} 行 ({time.time()-t4:.0f}s)", flush=True)

    # ---- axis_info
    info = {
        "schema_version": "1.0",
        "dataset": "THINGS-EEG 4D (train)",
        "split": "train",
        "axes": {
            "image": {"size": int(meta.image_idx.max()) + 1,
                      "meaning": "image condition (THINGS training set)",
                      "type": "categorical"},
            "repetition": {"size": int(meta.rep.max()) + 1,
                           "meaning": "repeated presentation", "type": "trial"},
            "channel": {"size": 63, "meaning": "EEG electrodes", "type": "spatial"},
            "time": {"size": 250, "meaning": "post-stimulus 0-1000 ms",
                     "unit": "sample @250Hz", "type": "temporal"},
            "frequency": {"size": len(BANDS), "meaning": "band power (derived)",
                          "bands": {n: list(b) for n, b in zip(BAND_NAMES, BANDS)},
                          "type": "spectral"},
        },
        "n_subjects": len(done),
        "subjects": done,
        "n_rows_meta": int(len(meta)),
        "per_subject_checks": checks,
        "files": {
            "raw": "{sid}.npz key=eeg float16",
            "band": "{sid}_band.npz key=band float32",
            "meta": "meta.parquet",
        },
        "notes": [
            "float16 保存 raw（原始 float32 会是 2 倍体积）",
            "band 为绝对功率，跨频带比较须先转相对功率",
            "仅 train split；test split 见 omnibrain4d/eeg/",
        ],
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "axis_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    tot = sum(f.stat().st_size for f in OUT.glob("*.npz")) / 1024 ** 3
    print("\n" + "=" * 84, flush=True)
    print(f"完成：{len(done)} 被试 | meta {len(meta):,} 行 | npz 合计 {tot:.2f} GB",
          flush=True)
    print(f"总耗时 {(time.time()-t0)/60:.1f} 分钟", flush=True)
    for sub, c in checks.items():
        print(f"  {sub}: {c['shape']} raw={c['raw_gb']}GB "
              f"band={c['band_gb']}GB rep_rel={c['rep_reliability']:+.4f}",
              flush=True)


if __name__ == "__main__":
    main()
