#!/usr/bin/env python3
"""exp102_diag.py — 诊断：缓存的 VAE 潜变量是不是正常的（**loss 异常小 ⇒ 潜变量可能是零**）。

要查三件事：
1. 潜变量的统计量（mean/std/min/max/零比例）⇒ 潜变量若 std≈0 ⇒ 训练是废的；
2. 用**真潜变量**与用**零潜变量**各算几步的 ε-预测损失 ⇒ 对比 ⇒ 若两者几乎一样 ⇒ 潜变量没起作用；
3. 顺便验一下 diffusers 的 `latent_dist.mode` 与 `mean` 的差异（缓存用的是 mode）。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

EXP_DIR = Path(__file__).resolve().parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/domains/runs/illusionbrain-align/exp102/cache")
    ap.add_argument("--out-dir", default="/domains/runs/illusionbrain-align/exp102")
    ap.add_argument("--steps", type=int, default=12)
    args = ap.parse_args()

    import torch
    import exp102_train as T

    cache = Path(args.cache_dir)
    lat = T.load_latents(cache)
    L = lat["latents"]
    log(f"[潜变量] shape={tuple(L.shape)} dtype={L.dtype}")
    x = L.float()
    log(f"[潜变量] 全局 mean={x.mean():.5f} std={x.std():.5f} "
        f"min={x.min():.3f} max={x.max():.3f}")
    log(f"[潜变量] |x|<1e-3 的比例={float((x.abs() < 1e-3).float().mean()):.5f}")
    per_ch = x.std(dim=(0, 2, 3))
    log(f"[潜变量] 逐通道 std: {[round(float(v), 3) for v in per_ch]}")
    log(f"[潜变量] 前 3 个通道的均值: {[round(float(v), 4) for v in x.mean(dim=(0, 2, 3))]}")

    # ---- 决定性检查：这些潜变量是不是**全都一样**（= 灰占位图）----
    # ⚠️ 2026-10-27 的坑：旧版缓存把 8270 张图全读成了灰占位图，而**静默**跑完了
    #    ⇒ 训练「看起来正常」，其实全废 ⇒ ⇒ 这里要有**能一眼看出全同**的检查。
    rows = x[:8].reshape(8, -1)
    d01 = float(torch.linalg.norm(rows[0] - rows[1]))
    d02 = float(torch.linalg.norm(rows[0] - rows[2]))
    dmax = float(torch.linalg.norm(rows[0] - rows[-1]))
    log(f"[占位检查] 前 8 行的两两 L2 距离: d(0,1)={d01:.6f} d(0,2)={d02:.6f} d(0,7)={dmax:.6f}")
    n_identical = 1
    for i in range(1, 8):
        if float(torch.linalg.norm(rows[0] - rows[i])) == 0.0:
            n_identical += 1
    log(f"[占位检查] 前 8 行里与第 0 行**完全相同**的有 {n_identical}/8"
        f" ⇒ {'⚠️⚠️ 全是同一个占位图（缓存坏了）' if n_identical >= 8 else '✓ 各行不同（不是占位图）'}")
    # 跨行差异的量级（真图 ⇒ 距离应是潜变量尺度的百分之几）
    scale = float(torch.linalg.norm(rows[0]))
    log(f"[占位检查] 第 0 行的 L2 范数={scale:.4f}；行间相对差="
        f"{(d02/scale if scale else float('nan')):.4f}")

    # ---- 2) 真潜变量 vs 零潜变量的 ε-预测损失 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = T.load_sd_component("unet", dtype=torch.float32, device=device).eval()
    for p in unet.parameters():
        p.requires_grad_(False)
    txt = T.load_sd_component("text_encoder", dtype=torch.float32, device=device).eval()
    tok = T.load_sd_component("tokenizer")
    sched = T.load_sd_component("scheduler")
    ac = sched.alphas_cumprod.to(device=device, dtype=torch.float32)
    ids = tok([T.PROMPT], padding="max_length", max_length=77, truncation=True,
              return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        text_emb = txt(ids)[0]

    rng = np.random.default_rng(0)
    losses_real, losses_zero = [], []
    for i in range(args.steps):
        idx = int(rng.integers(0, L.shape[0]))
        x0 = L[idx:idx + 1].to(device=device, dtype=torch.float32)
        noise = torch.randn_like(x0)
        t = torch.randint(0, 1000, (1,), device=device)
        a = ac[t].view(-1, 1, 1, 1)
        for tag, base, acc in (("real", x0, losses_real), ("zero", torch.zeros_like(x0), losses_zero)):
            xt = a.sqrt() * base + (1.0 - a).sqrt() * noise
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                enc = text_emb.expand(1, -1, -1)
                pred = unet(xt, t, encoder_hidden_states=enc).sample
            acc.append(float(torch.nn.functional.mse_loss(pred.float(), noise.float())))
    log(f"[损失] 真潜变量 ε-MSE 均值={np.mean(losses_real):.6f}（{args.steps} 步）")
    log(f"[损失] 零潜变量 ε-MSE 均值={np.mean(losses_zero):.6f}")
    log(f"[损失] 两者比 = {np.mean(losses_zero)/max(np.mean(losses_real), 1e-12):.2f}×"
        "（≈1 ⇒ 潜变量对预测几乎没影响 ⇒ 潜变量有问题）")

    # ---- 3) diffusers 里 mode 与 mean 的差别（缓存用 mode）----
    try:
        from PIL import Image
        import exp102_cache as C
        vae = T.load_sd_component("vae", dtype=torch.float16, device=device).eval()
        paths, _ = C.master_paths(tuple(T.TRAIN_SUBJECTS))
        p = C.IMAGE_SET / paths[0]
        im = Image.open(p).convert("RGB").resize((512, 512), Image.BICUBIC)
        arr = np.asarray(im, dtype=np.float32) / 127.5 - 1.0
        xx = torch.from_numpy(arr).permute(2, 0, 1)[None].to(device=device, dtype=torch.float16)
        with torch.no_grad():
            dist = vae.encode(xx).latent_dist
            mode = dist.mode() if callable(dist.mode) else dist.mode
            mean = dist.mean if not callable(getattr(dist, "mean", None)) else dist.mean()
            samp = dist.sample()
        for name, v in (("mode", mode), ("mean", mean), ("sample", samp)):
            log(f"[VAE] {name}: shape={tuple(v.shape)} mean={float(v.float().mean()):.4f} "
                f"std={float(v.float().std()):.4f} max={float(v.float().abs().max()):.3f}")
        log(f"[VAE] mode 与 mean 的最大绝对差 = "
            f"{float((mode.float()-mean.float()).abs().max()):.6f}")
    except Exception as e:                                       # noqa: BLE001
        log(f"[VAE] 检查失败：{type(e).__name__}: {e}")

    log("DIAG DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
