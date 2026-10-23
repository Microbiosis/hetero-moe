"""domain_adapters.py — 把 THINGS-EEG 的输入域适配到各基模的期望域。

## 问题

THINGS-EEG 的 4D 数据是 `(n_img, n_rep, 63ch, 250Hz, 1000ms)`，
而 7 个基模各自的期望域完全不同：

| 基模 | 通道 | 采样率 | 窗口 |
|---|---:|---:|---:|
| cbramod | 可配 | 可配 | 可配 |
| mirepnet | 45 | 250 | 1000 |
| eegdino-small | 19 | 200 | 1000 |
| biot | 18 | 200 | 1000 |
| signal-jepa | – | 128 | – |
| labram | – | – | 3000 |
| braintokenizer | – | 256 | 512 |

## 三条处置

1. **通道**：优先用 braindecode 自带的 `Interpolated*` 模型类
   （`InterpolatedBIOT / InterpolatedLaBraM / InterpolatedSignalJEPA / InterpolatedModel`）——
   它们内部用 `make_standard_montage` + MNE 插值，正是 eegdino 模型卡推荐的路线。
   ⇒ 本模块只负责**选出通道子集**（按名字），插值交给 braindecode。
2. **重采样**：`scipy.signal.resample_poly`（带抗混叠），250 → 目标采样率。
3. **窗口**：目标窗口 ≤ 1000 ms ⇒ 裁剪；> 1000 ms（labram 的 3000）⇒ **补零**。
   R3 要求"补零 vs 改窗口两种方案都跑并标注，不静默取其一" ⇒ `window_mode` 参数化。

## 不做的事

- 不自己实现蒙太奇插值（braindecode 有）；
- 不改 EEG 的语义（只动通道集合、采样率、窗口长度）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FS_THINGS = 250.0
N_CH_THINGS = 63
T_THINGS = 250          # 1000 ms @ 250 Hz


@dataclass
class DomainSpec:
    """一个基模的期望输入域。"""
    model: str
    chans: list[str] | None = None      # 期望通道名；None = 不限（模型可配）
    sfreq: float | None = None          # 期望采样率
    n_times: int | None = None         # 期望时间点数
    window_mode: str = "pad"            # "pad"（补零）| "crop"（裁剪）| "strict"
    note: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class AdaptInfo:
    """一次适配的完整记录（要写进结果，别默默做掉）。"""
    model: str
    src_shape: tuple
    dst_shape: tuple
    src_sfreq: float
    dst_sfreq: float | None
    n_chans_in: int
    n_chans_out: int
    chans_selected: list[str] = field(default_factory=list)
    chans_missing: list[str] = field(default_factory=list)
    resample: str = "none"
    window: str = "none"
    note: str = ""

    def to_dict(self):
        return {
            "model": self.model, "src_shape": list(self.src_shape),
            "dst_shape": list(self.dst_shape), "src_sfreq": self.src_sfreq,
            "dst_sfreq": self.dst_sfreq, "n_chans_in": self.n_chans_in,
            "n_chans_out": self.n_chans_out,
            "n_chans_selected": len(self.chans_selected),
            "chans_missing": self.chans_missing,
            "resample": self.resample, "window": self.window, "note": self.note,
        }


def resample_eeg(eeg, src_sfreq, dst_sfreq):
    """(..., C, T) → 抗混叠重采样。返回新数组。"""
    from scipy.signal import resample_poly
    if dst_sfreq is None or abs(dst_sfreq - src_sfreq) < 1e-6:
        return eeg, "none"
    from math import gcd
    r = int(round(dst_sfreq))
    q = int(round(src_sfreq))
    g = gcd(q, r)
    up, down = r // g, q // g
    return resample_poly(eeg, up, down, axis=-1), f"resample_poly({up}/{down})"


def fit_window(eeg, n_target, mode="pad"):
    """把时间轴调成 n_target。返回 (eeg, 说明)。"""
    T = eeg.shape[-1]
    if n_target is None or T == n_target:
        return eeg, "none"
    if T > n_target:
        return eeg[..., :n_target], f"crop({T}→{n_target})"
    if mode == "crop":
        return eeg, f"keep({T}<{n_target},crop 模式不补零)"
    if mode == "strict":
        raise ValueError(f"窗口不足：{T} < {n_target} 且 mode=strict")
    pad = [(0, 0)] * (eeg.ndim - 1) + [(0, n_target - T)]
    return np.pad(eeg, pad, mode="constant"), f"pad_zeros({T}→{n_target})"


def select_channels(eeg, want, have):
    """按名字取通道子集。返回 (eeg_sel, selected, missing)。

    `have` 是 THINGS 的通道名列表（顺序必须与 eeg 的通道轴一致）。
    找不到的通道**不静默丢弃**：返回 missing 列表，由调用方决定用插值还是弃用该基模。
    """
    if want is None:
        return eeg, list(have) if have else [], []
    idx_map = {c: i for i, c in enumerate(have or [])}
    sel, missing = [], []
    for c in want:
        if c in idx_map:
            sel.append(idx_map[c])
        else:
            missing.append(c)
    if not sel:
        raise ValueError(f"目标通道一个都取不到（want={want[:6]}…）")
    return eeg[..., sel, :], [have[i] for i in sel], missing


def adapt(eeg, spec, have_chans=None, window_mode=None):
    """把 THINGS 4D EEG 适配到 spec 期望的域。

    eeg : (n_img, n_rep, C, T) float
    返回 (adapted, AdaptInfo)
    """
    mode = window_mode or spec.window_mode
    info = AdaptInfo(model=spec.model, src_shape=tuple(eeg.shape),
                     dst_shape=(), src_sfreq=FS_THINGS,
                     dst_sfreq=spec.sfreq, n_chans_in=eeg.shape[2],
                     n_chans_out=0, note=spec.note)

    out, ch_sel, ch_missing = select_channels(eeg, spec.chans, have_chans)
    info.chans_selected = ch_sel
    info.chans_missing = ch_missing
    info.n_chans_out = out.shape[2]

    out, rinfo = resample_eeg(out, FS_THINGS, spec.sfreq)
    info.resample = rinfo

    out, winfo = fit_window(out, spec.n_times, mode)
    info.window = winfo

    info.dst_shape = tuple(out.shape)
    return out, info


def spec_from_config(model_name, cfg, meta=None):
    """从桶内 config.json + 模型 meta 推出 DomainSpec。"""
    meta = meta or {}
    sfreq = cfg.get("sfreq") or meta.get("sfreq")
    n_times = cfg.get("n_times") or meta.get("n_times")
    chans = meta.get("ch_names") or None
    if sfreq is None and model_name in ("mirepnet",):
        sfreq = 250.0
    note = ""
    if n_times and n_times > T_THINGS:
        note = (f"期望窗口 {n_times} 点 > THINGS 的 {T_THINGS} 点 ⇒ "
                f"补零（window_mode=pad）；改窗口方案需另跑并标注")
    return DomainSpec(model=model_name, chans=chans, sfreq=sfreq,
                      n_times=n_times, note=note)
