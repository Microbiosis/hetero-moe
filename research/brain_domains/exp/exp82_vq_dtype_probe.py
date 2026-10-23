#!/usr/bin/env python3
"""exp82_vq_dtype_probe.py — 查清 braintokenizer 的量化输出**到底是什么**。

## 为什么必须查

exp79 出现一个**自相矛盾**的结果：
- 每级码本利用率 100%、熵 6.2/6.24 nats（健康）；
- 但「最终码分布」只有 **6 个不同取值**。

⇒ 两种可能，结论完全相反：
(a) 最终输出**是离散码号** ⇒ 残差 VQ 实际只用了 6 种码组合（信息量极低）⇒
    基线表 braintokenizer 那一行测的是「6 状态的混合比例」；
(b) 最终输出**是量化的连续向量**，而 exp79 用 `.long()` 截断了它
    ⇒ 「6 个值」是**脚本 bug**，基线表测的其实是**连续向量**。

⇒ 这直接决定 S1 的读出口径（我们的 tokenizer 若输出量化向量，
可比口径就是「量化向量池化」而不是「码号」）。

## 本脚本做什么

只跑**一个批次**，打印：
- 模型最终输出：shape / dtype / 取值范围 / 唯一值数（float 与 int 两种解释都算）；
- 每级 VectorQuantizer 的输出：shape / dtype；
- 用最近码反查：最终量化向量最接近哪个码本（判断它是不是 4 级之和）。

## 用法

    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp82_vq_dtype_probe.py \
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
import registry as REG             # noqa: E402

TEST_4D = Path("/domains/omnibrain4d/eeg")
MODELS_DIR = Path("/training/models")
MODEL = "braintokenizer"


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import torch

    REG.ensure_braindecode_version("1.8.1.dev184031250")
    import braindecode.models as bdmodels

    h = REG.load_base_model(MODEL, models_dir=MODELS_DIR, braindecode_models=bdmodels,
                            spec=BS.BASE_SPECS.get(MODEL), verbose=False)
    if not h.ok:
        log(f"✗ 装载失败：{h.error[:300]}")
        return 1

    eeg = np.load(TEST_4D / "sub-01.npz")["eeg"][:1, :args.n]
    adapted, _ = DA.adapt(eeg.reshape(-1, eeg.shape[2], eeg.shape[3]),
                          h.spec, BS.THINGS_EEG_CHANS)
    mdt = next((p.dtype for p in h.model.parameters()), None)
    xb = torch.from_numpy(np.ascontiguousarray(adapted)).to(dtype=mdt)

    # 逐级 hook
    levels = {}

    def mk(i):
        def hook(mod, inp, out):
            levels[i] = out
        return hook

    handles = []
    qlayers = []
    for nm, mod in h.model.named_modules():
        if nm.endswith("quantizer.layers"):
            qlayers = list(mod.children())
    for i, lay in enumerate(qlayers):
        handles.append(lay.register_forward_hook(mk(i)))

    with torch.no_grad():
        out = h.model(xb)
    for hh in handles:
        hh.remove()

    log(f"\n=== 模型最终输出 ===")
    log(f"  type={type(out)}")
    if isinstance(out, dict):
        for k, v in out.items():
            if torch.is_tensor(v):
                log(f"  dict[{k}]: shape={tuple(v.shape)} dtype={v.dtype} "
                    f"min={float(v.min()):.4f} max={float(v.max()):.4f}")
    elif isinstance(out, (tuple, list)):
        for i, v in enumerate(out):
            if torch.is_tensor(v):
                log(f"  tuple[{i}]: shape={tuple(v.shape)} dtype={v.dtype} "
                    f"min={float(v.min()):.4f} max={float(v.max()):.4f}")
    elif torch.is_tensor(out):
        log(f"  shape={tuple(out.shape)} dtype={out.dtype} "
            f"min={float(out.min()):.4f} max={float(out.max()):.4f}")
        u = torch.unique(out.reshape(-1, out.shape[-1]) if out.ndim == 3
                         else out.reshape(-1))
        log(f"  唯一值个数（按最后一维摊平后）= {u.numel()}")

    tok, how = REG.tokens_from_output(out, 2)
    if tok is not None:
        log(f"  tokens_from_output: shape={tuple(tok.shape)} dtype={tok.dtype} ({how})")
        if tok.ndim == 3:
            flat = tok.reshape(-1, tok.shape[-1])
            log(f"  最后一维唯一值数 = {torch.unique(flat[:, 0]).numel()}")
            log(f"  整张张量唯一值数 = {torch.unique(tok).numel()}")
            log(f"  取值范围 [{float(tok.min()):.4f}, {float(tok.max()):.4f}]")
            log(f"  是否全整数 = {bool(torch.equal(tok, tok.round()))}")

    log(f"\n=== 逐级 VectorQuantizer 输出 ===")
    for i in sorted(levels):
        o = levels[i]
        t = o
        if isinstance(t, (tuple, list)):
            log(f"  L{i}: tuple 长度 {len(t)} | 各元素：")
            for j, v in enumerate(t):
                if torch.is_tensor(v):
                    log(f"      [{j}] shape={tuple(v.shape)} dtype={v.dtype} "
                        f"min={float(v.min()):.4f} max={float(v.max()):.4f}")
                else:
                    log(f"      [{j}] {type(v)}")
        elif torch.is_tensor(t):
            log(f"  L{i}: shape={tuple(t.shape)} dtype={t.dtype} "
                f"min={float(t.min()):.4f} max={float(t.max()):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
