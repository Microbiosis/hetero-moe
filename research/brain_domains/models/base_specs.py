"""base_specs.py — 7 个基模的**实测**输入规格（2026-10-23 云端探针 v10 得出）。

## 为什么要有这张表

桶内 `config.json` 对大部分模型是**残缺**的：

| 模型 | config 里的 n_chans / n_times | 实测可用规格 |
|---|---|---|
| cbramod | null / null | (63, 250) @250 Hz |
| mirepnet | 45 / 1000 | (45, 1000) @250 Hz |
| eegdino-small | 19 / 1000（sfreq 缺） | (19, 1000) @**200** Hz |
| biot | 18 / 1000 | (18, 1000) @**200** Hz |
| labram | null / 3000 | (128, 3000) @250 Hz |
| signal-jepa | null / null | (62, 250) @128 Hz |
| braintokenizer | null / 512 | (19, 512) @256 Hz |

⇒ **不能靠 config 推**。这张表是探针「装载 + 前向都成功」试出来的，
是表评测的**权威输入规格**。

## 三个关键事实

1. **窗口一律是「补零」**：THINGS 只有 1 s（250 样本 @250 Hz），
   而多数模型要 4–12 s（1000 @250 Hz = 4 s；3000 @250 Hz = 12 s）。
   我们只有 1 s ⇒ 只能**尾部补零**。零是常数，LayerNorm 会把它压掉大部分，
   但**带位置编码的模型只在前 1/4 窗口看到信号** —— 这是本表的已知偏差，
   必须在结论里写明，不能当噪声吞掉。
2. **采样率必须重采样**：eegdino/biot 在 200 Hz、signal-jepa 在 128 Hz、
   braintokenizer 在 256 Hz，THINGS 是 250 Hz ⇒ 必须 `resample_poly`。
3. **两个模型需要通道插值**（它们的通道集不是 THINGS 的子集）：
   - `signal-jepa` 要 62 个通道（THINGS 缺 10 个）⇒ 用 braindecode 的
     `InterpolatedSignalJEPA`（用户给 63 通道，wrapper 内部插值）；
   - `labram` 要 128 个 10-10 通道（THINGS 只有 63）⇒ `InterpolatedLaBraM`。
   其余 5 个模型的通道集是 THINGS 的子集 ⇒ 直接**按名选取**（大小写不敏感）。
"""

# THINGS-EEG 的 63 个通道（云端从原始 test.pt 读出，2026-10-23）
THINGS_EEG_CHANS = [
    "Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "F2", "F4", "F6", "F8",
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "FT10",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP10",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]

# mirepnet 的 45 个通道（桶内 mirepnet_channels.json）
MIREP_CHANS = [
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
]

# signal-jepa 的 62 个通道（config.chs_info；THINGS 缺 10 个 ⇒ 需插值）
SIGNALJEPA_CHANS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "Cz", "C4", "T8", "TP9", "CP5", "CP1", "CP2", "CP6", "TP10",
    "P7", "P3", "Pz", "P4", "P8", "PO9", "O1", "Oz", "O2", "PO10",
    "FC3", "FC4", "C5", "C1", "C2", "C6", "CP3", "CPz", "CP4", "P1", "P2",
    "POz", "FT9", "FTT9h", "TTP7h", "TP7", "TPP9h", "FT10", "FTT10h",
    "TPP8h", "TP8", "TPP10h", "F9", "F10", "AF7", "AF3", "AF4", "AF8",
    "PO3", "PO4",
]

# braintokenizer 的 19 个通道（config.chs_info）—— THINGS 全有 ✓
BRAINTOK_CHANS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
]

# EEG-DINO 的 19 个通道（经典 10-20 子集，THINGS 全有 ✓）
EEGDINO_CHANS = list(BRAINTOK_CHANS)

# BIOT 的 18 个通道（经典 10-20 去掉 Fz）—— THINGS 全有 ✓
BIOT_CHANS = [c for c in BRAINTOK_CHANS if c != "Fz"]


# BIOT 的 18 个"通道"其实是 **TCP 双极导联**（braindecode 源码
# `BIOT_CHANNEL_ORDER`）：前 16 个是 TCP 双极导联，后 2 个是 SHHS 差分。
# ⇒ 不能直接选 18 个电极，必须**自己算差分** V(A) - V(B)。
# ⚠️ 差分 C3-A2 / C4-A1 要的 A1/A2（耳垂参考）**THINGS 里没有**
#    ⇒ 这两路用**平均参考**代替（明确标注为替代方案，不是原始定义）。
BIOT_DIFFS = [
    ("Fp1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"),
    ("Fp2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"),
    ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
    ("C3", None), ("C4", None),          # None ⇒ 用平均参考（A1/A2 缺失）
]


# ---------------------------------------------------------------------------
# 规格表
#   n_chans     : 模型前向要的通道数
#   n_times     : 模型前向要的时间点数
#   sfreq       : 模型前向要的采样率（= 预训练采样率）
#   chans       : 需要从 THINGS 里**选取**的通道名（None = 全部 63 个原样送入）
#   interpolate : True ⇒ 用 braindecode 的 Interpolated* 包装（模型内部插值）
#   interpolated_cls : Interpolated* 的类名
#   window_mode : 窗口怎么对齐（THINGS 只有 1 s ⇒ 一律 pad）
#   fill        : 信号占模型窗口的比例（1 s / 模型窗口秒数）
# ---------------------------------------------------------------------------
BASE_SPECS = {
    "cbramod": {
        "n_chans": 63, "n_times": 200, "sfreq": 200.0,
        "chans": None, "interpolate": False, "window_mode": "pad",
        "bd_version": "stable",
        "cls": "CBraMod", "params": 4974402,
        "repr_kwargs": {"return_features": True},
        "note": "63 通道 = THINGS 原生，零损失；**200 Hz + 200 样本 = 正好 1 个 patch**"
                "（patch_size=200；用 250 Hz 的 250 样本会触发 einops "
                "'can't divide axis of length 250 in chunks of 200'）",
    },
    "mirepnet": {
        "n_chans": 45, "n_times": 1000, "sfreq": 250.0,
        "chans": MIREP_CHANS, "interpolate": False, "window_mode": "pad",
        "interpolate_missing": True, "bd_version": "dev",
        "cls": "MIRepNet", "params": 5142786,
        "repr_kwargs": {"return_features": True},
        "note": "45 通道全在 THINGS 里（大小写不敏感）；模型要 4 s，我们只有 1 s ⇒ 补零 3 s",
    },
    "eegdino-small": {
        "n_chans": 19, "n_times": 1000, "sfreq": 200.0,
        "chans": EEGDINO_CHANS, "interpolate": False, "window_mode": "pad",
        "interpolate_missing": True, "bd_version": "stable",
        "cls": "EEGDINO", "params": 4539596,
        "repr_kwargs": {"return_encoder_output": True},
        "note": "预训练在 200 Hz ⇒ 必须重采样；模型要 5 s，我们只有 1 s ⇒ 补零 4 s",
    },
    "biot": {
        "n_chans": 18, "n_times": 1000, "sfreq": 200.0,
        "chans": None, "differentials": BIOT_DIFFS,
        "interpolate": False, "window_mode": "pad",
        "bd_version": "stable",
        "cls": "BIOT", "params": 3187202,
        "repr_kwargs": {"return_feature": True},
        "note": "18 个通道是 TCP 双极导联（不是 18 个电极）⇒ 自己算差分；"
                "C3-A2 / C4-A1 要的 A1/A2 在 THINGS 里没有 ⇒ 用平均参考代替；"
                "200 Hz 预训练 ⇒ 重采样；补零到 5 s",
    },
    "labram": {
        "excluded": True,
        "n_chans": 128, "n_times": 3000, "sfreq": 250.0,
        "chans": None, "interpolate": True, "window_mode": "pad",
        "bd_version": "stable",
        "cls": "Labram", "interpolated_cls": "InterpolatedLaBraM",
        "excluded_reason": "目标蒙太奇是 10-10（128 电极，含 CFC/CCP/T1/T2 等"
                           "自定义导联）⇒ THINGS 的 63 通道插不出；"
                           "且 InterpolatedLaBraM 只存在于 stable 版、"
                           "dev 版没有该类 ⇒ 本轮不纳入表",
        "params": 5820338,
        "repr_kwargs": {"return_features": True},
        "note": "要 128 个 10-10 通道（THINGS 只有 63）⇒ 走 InterpolatedLaBraM 插值；"
                "模型要 12 s，我们只有 1 s ⇒ 补零 11 s（**信号只占 8%**）",
    },
    "signal-jepa": {
        "n_chans": 62, "n_times": 250, "sfreq": 128.0,
        "chans": None, "interpolate": True, "window_mode": "pad",
        "bd_version": "stable",
        "cls": "SignalJEPA", "interpolated_cls": "InterpolatedSignalJEPA",
        "params": 3456852,
        "note": "要 62 个通道（THINGS 缺 10 个）⇒ InterpolatedSignalJEPA 插值；"
                "128 Hz 预训练 ⇒ 重采样",
    },
    "braintokenizer": {
        "n_chans": 19, "n_times": 512, "sfreq": 256.0,
        "chans": BRAINTOK_CHANS, "interpolate": False, "window_mode": "pad",
        "interpolate_missing": True, "bd_version": "dev",
        "cls": "BrainTokenizer", "params": 5050338,
        "note": "19 通道全在 THINGS 里；256 Hz 预训练 ⇒ 重采样；补零到 2 s。"
                "输出是**离散 token 码**（B,19,512）⇒ 池化后是每维的期望码值",
    },
}

ORDER = ["cbramod", "mirepnet", "eegdino-small", "biot",
         "labram", "signal-jepa", "braintokenizer"]

# 需要 dev 版 braindecode 才有这两个类（PyPI 稳定版 1.8.1 只有 137 个类）
BD_DEV_VERSION = "1.8.1.dev184031250"


def signal_fill(name):
    """信号占模型窗口的比例（THINGS 只有 1 s）。"""
    sp = BASE_SPECS.get(name)
    if not sp:
        return None
    window_s = sp["n_times"] / sp["sfreq"]
    return round(1.0 / window_s, 4)


def spec(name):
    """取某基模的规格（未知名返回 None）。"""
    return BASE_SPECS.get(name)


def models_for_version(version="stable"):
    """某个 braindecode 版本能跑的模型（**排除**明确标 excluded 的）。

    为什么分两版：`MIRepNet` / `BrainTokenizer` 只在 **dev 版**里有；
    而 `Interpolated*`（通道插值包装）只在 **stable 版**里有
    （dev 版 176 个类里一个 Interpolated* 都没有）⇒ 必须分两趟跑。
    """
    out = []
    for n in ORDER:
        sp = BASE_SPECS.get(n) or {}
        if sp.get("excluded"):
            continue
        if sp.get("bd_version", "stable") == version:
            out.append(n)
    return out


def excluded_models():
    """本轮不纳入表的模型 + 原因。"""
    return {n: sp.get("excluded_reason", "")
            for n, sp in BASE_SPECS.items() if sp.get("excluded")}
