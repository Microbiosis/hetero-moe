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
    differentials: list | None = None   # 双极导联 [(A,B)]；B=None ⇒ 平均参考
    interpolate_missing: bool = False   # 缺通道时用 MNE 插值


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

    ⚠️ **大小写不敏感**：不同来源的通道名大小写不一致
    （THINGS 用 "Fz/Cz/Pz/FCz"，mirepnet 的通道表用 "FZ/CZ/PZ/FCZ"）。
    严格匹配会把 45 个通道全判成 missing。
    先按原名精确匹配，剩下的大小写不敏感匹配。
    """
    if want is None:
        return eeg, list(have) if have else [], []
    idx_map = {c: i for i, c in enumerate(have or [])}
    lower_map = {}
    for i, c in enumerate(have or []):
        lower_map.setdefault(c.lower(), i)
    sel, missing, used = [], [], set()
    # 第一轮：精确匹配
    for c in want:
        if c in idx_map and idx_map[c] not in used:
            sel.append(idx_map[c])
            used.add(idx_map[c])
        else:
            missing.append(c)
    # 第二轮：大小写不敏感
    still = []
    for c in missing:
        i = lower_map.get(c.lower())
        if i is not None and i not in used:
            sel.append(i)
            used.add(i)
        else:
            still.append(c)
    missing = still
    if not sel:
        raise ValueError(f"目标通道一个都取不到（want={want[:6]}…）")
    return eeg[..., sel, :], [have[i] for i in sel], missing


def _montage_positions(names):
    """MNE standard_1020 里这些通道的 3D 位置。找不到的会被跳过。"""
    import mne
    mont = mne.channels.make_standard_montage("standard_1020")
    pos = mont.get_positions()["ch_pos"]
    out = {}
    for n in names:
        for cand in (n, n.upper(), n.capitalize()):
            if cand in pos:
                out[n] = np.asarray(pos[cand], dtype=np.float64)
                break
    return out


def idw_interpolate(eeg, have, want, k=6, power=3.0):
    """反距离加权（IDW）头皮插值：把 `have` 里的通道插出 `want` 里缺的通道。

    为什么自己写：MNE 的 `Raw.interpolate_channels` 在这个版本里**不存在**
    （实测 AttributeError），而且它要 pyvista 等可选依赖。
    IDW 只需 MNE 的蒙太奇位置，**确定性强、可解释**
    （"Fz 由最近的几个电极按距离倒数加权得到"）。

    `eeg` : (n, C, T)  → 返回 (n, len(want), T)
    """
    n = eeg.shape[0]
    have_l = list(have or [])
    want_l = list(want)
    lower = {c.lower(): i for i, c in enumerate(have_l)}
    have_pos = _montage_positions(have_l)

    out_cols, out_names = [], []
    for c in want_l:
        if c.lower() in lower:
            i = lower[c.lower()]
            out_cols.append(eeg[:, i, :])
            out_names.append(c)
            continue
        # 缺通道 ⇒ 用最近的 k 个已有通道做 IDW
        pos_c = _montage_positions([c]).get(c)
        if pos_c is None:
            raise ValueError(f"通道 {c} 不在 standard_1020 里，插不了")
        d = []
        for name, p in have_pos.items():
            dist = float(np.linalg.norm(pos_c - p))
            if dist > 1e-9:
                d.append((dist, name))
        if not d:
            raise ValueError(f"没有可用来插值 {c} 的通道")
        d.sort()
        d = d[:k]
        w = np.array([1.0 / (dist ** power) for dist, _ in d])
        w = w / w.sum()
        acc = np.zeros((n, eeg.shape[-1]), dtype=np.float64)
        for (dist, name), wi in zip(d, w):
            acc += wi * eeg[:, lower[name.lower()], :]
        out_cols.append(acc)
        out_names.append(c)
    return np.stack(out_cols, axis=1), out_names, [
        (d, float(wi)) for d, wi in zip(d, w)]


def interpolate_missing(eeg, want, have, sfreq=250.0, n_jobs=1):
    """把缺失的通道**插值**出来，再按 `want` 的顺序选取。

    为什么需要：THINGS-EEG 的 63 通道里**没有 Fz**（它是 F1/F2 密集前额网格），
    而 eegdino / braintokenizer / mirepnet 都要 Fz（或 FZ）⇒ 直接选会少一个通道，
    模型直接报 "expected input to have 45 channels, but got 44"。

    ⚠️ 只能插**标准电极**（standard_1020 里的名字）；
       signal-jepa / labram 的通道集里有自定义名字（FTT9h、CFC1…）⇒ 插不了，
       那两个走 braindecode 的 `Interpolated*` 包装（stable 版）。
    """
    want_l = list(want)
    have_l = list(have or [])
    lower = {c.lower() for c in have_l}
    missing = [c for c in want_l if c.lower() not in lower]
    if not missing:
        out, got, miss = select_channels(eeg, want_l, have_l)
        return out, got, miss, "无需插值"
    interp, names, _ = idw_interpolate(eeg, have_l, want_l)
    return interp, names, [], f"IDW插值({len(missing)}: {missing})"


def make_differentials(eeg, pairs, have):
    """按双极导联算差分通道。返回 (新数组, 通道名列表, 说明)。

    BIOT 的 18 个"通道"其实是 **TCP 双极导联**（braindecode 源码
    `BIOT_CHANNEL_ORDER`）⇒ 不能直接选 18 个电极，必须自己算 V(A) - V(B)。

    `pairs` 里 B 为 None ⇒ 用**平均参考**代替（THINGS 没有 A1/A2 耳垂参考）。
    这是**替代方案**，不是原始定义 —— 必须在结果里标注。
    """
    idx = {c.lower(): i for i, c in enumerate(have or [])}
    mean_ref = eeg.mean(axis=-2, keepdims=True)     # (..., 1, T) 平均参考
    cols, names, notes = [], [], []
    for a, b in pairs:
        ia = idx.get(a.lower())
        if ia is None:
            raise ValueError(f"差分通道 {a} 不在 THINGS 通道里")
        va = eeg[..., ia, :]
        if b is None:
            vb = mean_ref[..., 0, :]
            names.append(f"{a}-AVG")
            notes.append(f"{a}-avg(替代 A1/A2)")
        else:
            ib = idx.get(b.lower())
            if ib is None:
                raise ValueError(f"差分通道 {b} 不在 THINGS 通道里")
            vb = eeg[..., ib, :]
            names.append(f"{a}-{b}")
        cols.append(va - vb)
    out = np.stack(cols, axis=-2)                   # (..., n_diff, T)
    return out, names, "双极差分" + ("（含平均参考替代）" if any(b is None for _, b in pairs) else "")


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

    if getattr(spec, "interpolate_missing", False) and spec.chans:
        out, got, miss, how = interpolate_missing(eeg, spec.chans, have_chans,
                                                 FS_THINGS)
        info.chans_selected = got
        info.chans_missing = miss
        info.note = (info.note + " | " + how).strip(" |")
        info.n_chans_out = out.shape[-2]
        out, rinfo = resample_eeg(out, FS_THINGS, spec.sfreq)
        info.resample = rinfo
        out, winfo = fit_window(out, spec.n_times, mode)
        info.window = winfo
        info.dst_shape = tuple(out.shape)
        return out, info

    diffs = getattr(spec, "differentials", None)
    if diffs:
        out, names, how = make_differentials(eeg, diffs, have_chans)
        info.chans_selected = names
        info.chans_missing = []
        info.note = (info.note + " | " + how).strip(" |")
    else:
        out, ch_sel, ch_missing = select_channels(eeg, spec.chans, have_chans)
        info.chans_selected = ch_sel
        info.chans_missing = ch_missing
    info.n_chans_out = out.shape[-2]   # 通道轴是 -2，-2 之后是时间轴

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
