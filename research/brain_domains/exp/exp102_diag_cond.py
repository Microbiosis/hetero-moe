#!/usr/bin/env python3
r"""exp102_diag_cond.py — **诊断 exp102 的「条件到底有没有真的进到生成里」**。

## 为什么查这个

exp102 第一轮跑完的逐被试数字：**生成图 2-way ≈ 0.49~0.555（机会 0.5）**，
而**条件侧 2-way = 0.72~0.85** ⇒ 脑侧条件本身有概念信息，但**生成的图里看不到它**。

⇒ 关键问题不是「脑里有没有信息」，而是 **条件有没有真的进到 SD1.5 的 cross-attention 里**：
- 条件 token 是**追加**在 $77$ 个文本 token 后面的（`encoder_hidden_states`，共 $93$ 个）；
- adapter 的**末层是零初始化** ⇒ 起点 token 全 $0$；
- ⇒ 若训练只把末层权重推到 $\max|w|\approx 10^{-2}$ 量级，
  ⇒ 这 $16$ 个 token 的**幅度**相对文本 token 就是**噪声级** ⇒ cross-attention 基本看不见它
  ⇒ ⇒ **生成出来等于「无条件 SD1.5 + a photo」** ⇒ own2 ≈ 机会是**预期**，不是关于脑的结论。

## 本脚本量什么

1. `adapter_final.pt` 末层（`net[3].weight` / `net[3].bias`）的 `max|w|`、`mean|w|`；
2. 真实条件过 adapter 后，**条件 token 的范数** vs **文本 token 的范数**（同尺度可比）；
3. 条件 token 与文本 embedding 的**余弦**（方向上有没有独立信息）；
4. 附带：若把条件 token 整体放大到与文本 token 同尺度，幅度会变成什么样（只算，不生成）。

## 用法

```powershell
python research\\brain_domains\\exp\\exp102_diag_cond.py
python -u /domains/scripts/exp102_diag_cond.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

EXP_DIR = Path(__file__).resolve().parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

TEST_4D = Path("/domains/omnibrain4d/eeg")
ENCODER_CKPT = "/domains/runs/illusionbrain-align/exp93b/encoder_best.pt"
ADAPTER_CKPT = "/domains/runs/illusionbrain-align/exp102/adapter_final.pt"
N_IMG = 200


def log(*a):
    print(*a, flush=True)


def l2norm(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    import torch
    import exp102_train as T
    import exp93b_contrastive as E93B

    # ---- ① adapter 末层权重 ----
    ck = torch.load(ADAPTER_CKPT, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    out = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        out[k] = {"max_abs": float(v.abs().max()), "mean_abs": float(v.abs().mean()),
                  "shape": list(v.shape)}
        log(f"[权重] {k:22s} shape={tuple(v.shape)} max|w|={out[k]['max_abs']:.6f} "
            f"mean|w|={out[k]['mean_abs']:.6f}")

    # ---- ② 条件 token 的幅度 vs 文本 token 的幅度 ----
    adapter = T.build_adapter().to(args.device)
    adapter.load_state_dict(sd)
    adapter = adapter.eval()

    ck2 = torch.load(ENCODER_CKPT, map_location="cpu", weights_only=False)
    enc = E93B.build_model(n_ch=ck2["n_ch"], n_times=ck2["n_times"], out_dim=ck2["out_dim"])
    enc.load_state_dict(ck2["model"] if "model" in ck2 else ck2["state_dict"])
    enc = enc.to(args.device).eval()

    eeg = np.load(TEST_4D / "sub-01.npz")["eeg"]
    flat = np.asarray(eeg[:N_IMG, 0], dtype=np.float32)
    with torch.no_grad():
        z = E93B.encode_all(enc, flat, args.device, 16)
        tok = adapter(torch.from_numpy(z.astype(np.float32)).to(args.device))
    tok = tok.cpu().numpy().astype(np.float64)
    tok_norm = np.linalg.norm(tok, axis=-1)          # (200, 16)
    log("")
    log(f"[条件 token] 形状 {tok.shape}｜范数 中位 {np.median(tok_norm):.6f}"
        f"｜max {tok_norm.max():.6f}｜全零的比例 {float((tok == 0).mean()):.4f}")

    # 文本 token 的幅度（同一个尺度：CLIP text encoder 的 77×768）
    from diffusers import StableDiffusionPipeline  # noqa: F401  (仅确认 diffusers 在)
    from transformers import CLIPTextModel, CLIPTokenizer
    repo = T.SD_REPO if hasattr(T, "SD_REPO") else "stable-diffusion-v1-5/stable-diffusion-v1-5"
    tokzr = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer")
    txt = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder").eval()
    with torch.no_grad():
        ids = tokzr([T.PROMPT], padding="max_length", max_length=77, truncation=True,
                    return_tensors="pt").input_ids
        temb = txt(ids)[0].numpy().astype(np.float64)
    tnorm = np.linalg.norm(temb, axis=-1)
    log(f"[文本 token] 范数 中位 {np.median(tnorm):.4f}（77 个）")

    ratio = float(np.median(tok_norm) / max(np.median(tnorm), 1e-12))
    log(f"[幅度比] 条件 token 范数 / 文本 token 范数 = **{ratio:.6f}**")
    log(f"           ⇒ 条件占 cross-attention 输入的相对幅度约 {ratio:.2e}"
        f"（1.0 才是与文本同量级）")

    # ---- ③ 方向：条件 token 与文本 embedding 的余弦 ----
    flat_t = temb.reshape(-1, temb.shape[-1])
    tn = flat_t / np.maximum(np.linalg.norm(flat_t, axis=1, keepdims=True), 1e-12)
    cz = tok.reshape(-1, tok.shape[-1])
    cn = cz / np.maximum(np.linalg.norm(cz, axis=1, keepdims=True), 1e-12)
    cos = (cn @ tn.T).ravel()
    log(f"[方向] 条件 token 与文本 token 的余弦：均值 {cos.mean():.4f}"
        f"｜max |cos| {np.abs(cos).max():.4f}｜随机基线 ≈ 0")

    rec = {"adapter_weights": out,
           "token_norm_median": float(np.median(tok_norm)),
           "token_norm_max": float(tok_norm.max()),
           "token_zero_fraction": float((tok == 0).mean()),
           "text_token_norm_median": float(np.median(tnorm)),
           "amplitude_ratio": ratio,
           "cos_to_text_mean": float(cos.mean()),
           "cos_to_text_absmax": float(np.abs(cos).max())}
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
        log(f"[写盘] {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
