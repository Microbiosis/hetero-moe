#!/usr/bin/env python3
"""exp83_determinism_probe.py — 查清 signal-jepa 的抽取为什么**不可复现**。

## 现象

阶段 D 的两轮跑同一被试（sub-01, stable）：
- **宿主行**（cbramod，原生 63 通道、不插值）：两轮**逐字相同**（+0.0102 / 0.0044 / 0.0049）；
- **v1 供体行**（signal-jepa 第 0 层）：两轮**不同**（+0.0132 → +0.0105）。

⇒ 不确定性不在评测代码（宿主行可复现），而在 **signal-jepa 这条路径**。
唯一的结构差异：**signal-jepa 用 braindecode 的 `InterpolatedSignalJEPA` 包装**
（THINGS 的 63 通道 → 它内部的 62 通道 ⇒ **要过 MNE 的通道插值**），
而 cbramod 是原生 63 通道、**不插值**。

## 本脚本做什么

同一份数据、同一模型类，分别做四组抽取并**逐元素比较**：

| 组 | 内容 | 若不同说明 |
|---|---|---|
| A | 同一 handle 抽两次（供体路径 = hook 抓 layer0） | 前向/hook 不确定 |
| B | 两个独立 handle 抽（供体路径） | 模型加载不确定 |
| C | 同一 handle 抽两次（**最终输出**路径，不用 hook） | 前向不确定（排除 hook） |
| D | 两个独立 handle 抽（最终输出路径） | 加载不确定 |

⇒ 哪一组先出现差异，就把不确定性定位到那一层。

## 用法

    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp83_determinism_probe.py \
      -v hf://buckets/illusion-brain/training-data:/training:ro \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor cpu-upgrade --timeout 30m -d
"""

from __future__ import annotations

import argparse
import os
import sys
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

import base_specs as BS            # noqa: E402
import domain_adapters as DA       # noqa: E402

TEST_4D = Path("/domains/omnibrain4d/eeg")
MODELS_DIR = Path("/training/models")
DONOR_LAYER = "transformer.encoder.layers.0.dropout1"


def log(*a):
    print(*a, flush=True)


def extract_layer(handle, adapted, layer_name):
    """供体路径：hook 抓指定中间层 → 池化。"""
    import torch
    import registry as REG
    mod = handle.model.get_submodule(layer_name)
    cap = {}

    def hook(m, inp, out):
        t = out
        if isinstance(t, (tuple, list)):
            t = t[0] if t else None
        cap["t"] = t.detach() if torch.is_tensor(t) else None

    h = mod.register_forward_hook(hook)
    try:
        with torch.no_grad():
            handle.model(torch.from_numpy(adapted))
    finally:
        h.remove()
    t = cap["t"]
    tk = REG.normalize_token_tensor(t.float())
    return REG.pool_tokens(tk, "mean_all")


def extract_final(handle, adapted):
    """最终输出路径（与基线表同口径）。"""
    import torch
    import registry as REG
    with torch.no_grad():
        out = handle.model(torch.from_numpy(adapted))
    tok, _how = REG.tokens_from_output(out, 2)
    return REG.pool_tokens(tok.float(), "mean_all")


def cmp(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return float("inf"), f"形状不同 {a.shape} vs {b.shape}"
    d = float(np.abs(a - b).max())
    return d, f"max|Δ| = {d:.3e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="signal-jepa")
    ap.add_argument("--n-img", type=int, default=4)
    ap.add_argument("--n-rep", type=int, default=4)
    ap.add_argument("--eval-mode", action="store_true",
                    help="抽取前 model.eval()（排除 Dropout 随机）")
    args = ap.parse_args()

    import torch
    import registry as REG

    REG.ensure_braindecode_version("stable")
    import braindecode.models as bdmodels

    spec = BS.BASE_SPECS.get(args.model)
    eeg = np.load(TEST_4D / "sub-01.npz")["eeg"][:args.n_img, :args.n_rep]

    def load():
        h = REG.load_base_model(args.model, models_dir=MODELS_DIR,
                                braindecode_models=bdmodels, spec=spec,
                                verbose=False)
        if args.eval_mode and h.ok:
            h.model.eval()
        return h

    hA = load()
    hB = load()
    log(f"装载：{'OK' if hA.ok else hA.error} / {'OK' if hB.ok else hB.error}")

    # ⚠️ `DA.adapt` 要的是 **DomainSpec 对象**，`BASE_SPECS[name]` 是 dict
    #    ⇒ 用 handle.spec（registry 已按实测规格建好）
    adapted, _ = DA.adapt(eeg.reshape(-1, eeg.shape[2], eeg.shape[3]), hA.spec,
                          BS.THINGS_EEG_CHANS)
    log(f"模型 {args.model} | 输入 {adapted.shape} | eval_mode={args.eval_mode}")

    # A：同一 handle，两次（供体路径）
    a1 = extract_layer(hA, adapted, DONOR_LAYER)
    a2 = extract_layer(hA, adapted, DONOR_LAYER)
    d, msg = cmp(a1, a2)
    log(f"A 同 handle × 2（供体路径/hook）：{msg}")

    # B：两个 handle（供体路径）
    b1 = extract_layer(hA, adapted, DONOR_LAYER)
    b2 = extract_layer(hB, adapted, DONOR_LAYER)
    d2, msg2 = cmp(b1, b2)
    log(f"B 两个 handle（供体路径/hook）：{msg2}")

    # C：同一 handle，两次（最终输出路径）
    c1 = extract_final(hA, adapted)
    c2 = extract_final(hA, adapted)
    d3, msg3 = cmp(c1, c2)
    log(f"C 同 handle × 2（最终输出路径）：{msg3}")

    # D：两个 handle（最终输出路径）
    d1 = extract_final(hA, adapted)
    d2 = extract_final(hB, adapted)
    d4, msg4 = cmp(d1, d2)
    log(f"D 两个 handle（最终输出路径）：{msg4}")

    # 额外：把适配后的输入也比一遍（排除适配层的不确定性）
    adapted2, _ = DA.adapt(eeg.reshape(-1, eeg.shape[2], eeg.shape[3]), hA.spec,
                           BS.THINGS_EEG_CHANS)
    de, msge = cmp(adapted, adapted2)
    log(f"E 输入适配两次：{msge}")

    log("\n" + "=" * 70)
    if args.eval_mode:
        log("（已 eval() ⇒ 若 A/C 仍不同 ⇒ 差异**不是** Dropout，而是模型内部"
            "（如 Interpolated* 包装的 MNE 插值）")
    else:
        log("⇒ 若 A/C 不同 ⇒ 前向随机（train 模式 ⇒ Dropout）；"
            "若 A/C 相同而 B/D 不同 ⇒ **模型加载/插值不确定**；"
            "若全都相同 ⇒ 差异来自别处（评测顺序 / rng）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
