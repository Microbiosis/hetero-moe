#!/usr/bin/env python3
"""exp102_probe_gen.py — ②③ 条件生成线的**环境探针**（**只探路，不出结论**）。

## 为什么先探路（而不是直接写训练）

②③ 要用「**连续条件 + 冻结生成先验**」（`MODEL_FORM.md` §7.3.1 的两支先例：ATM / ControlNet 式）。
⇒ ⇒ 但**桶里没有扩散先验**（`training-data/models/` 只有 EEG 编码器：biot / labram / eegdino /
mirepnet / cbramod / braintokenizer；THINGS-EEG 的**图像像素**在 `eeg/things-eeg/Image_set/`）
⇒ ⇒ ⇒ **先花几分钟问清楚三件事**，再决定 exp102 怎么设计：

| 探什么 | 为什么决定设计 |
|---|---|
| **能不能装 `diffusers` / `transformers`** | 决定用不用 HF 的标准扩散管线 |
| **能不能从 hub 下到模型**（网络 + 凭证） | 决定「冻结先验」是否可行 |
| **GPU 型号/显存 + 磁盘余量** | 决定 SD1.5（~4 GB）能不能放、能不能训 adapter |
| **THINGS-EEG 图像像素能不能读**（数量/尺寸/许可） | 图像侧数据是生成训练的输入 |
| **tiny 扩散管线能否端到端跑通** | 把「下载→管线→GPU→解码」这条路一次验证掉 |

⚠️ **本脚本不产生任何科学结论**（不测条件质量、不出 FID）⇒ 它的产出只用来定 exp102 的设计。
⇒ ⇒ **探针的结果必须**「哪些通了、哪些没通、卡在哪」**如实报**（不通也是结论）。

## 用法

```powershell
python -u exp102_probe_gen.py --out /domains/runs/illusionbrain-align/exp102_probe
```
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path


def log(*a):
    print(*a, flush=True)


def step(name, fn):
    """探针的一步 ⇒ 失败不抛出，只记状态（**不通也是结论**）。"""
    t0 = __import__("time").time()
    try:
        v = fn()
        log(f"[OK ] {name}: {v}")
        return {"name": name, "ok": True, "info": str(v)[:400],
                "sec": round(__import__("time").time() - t0, 1)}
    except Exception as e:
        log(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:200]}")
        return {"name": name, "ok": False, "err": f"{type(e).__name__}: {str(e)[:300]}",
                "trace": traceback.format_exc()[-400:],
                "sec": round(__import__("time").time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/domains/runs/illusionbrain-align/exp102_probe")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"steps": []}

    def add(name, fn):
        r = step(name, fn)
        res["steps"].append(r)
        return r

    # ① 硬件
    def hw():
        import torch
        info = {"torch": torch.__version__, "cuda": torch.cuda.is_available()}
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_GB"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        return info
    res["hardware"] = add("硬件（torch/CUDA/GPU/显存）", hw)

    # ② 磁盘
    def disk():
        u = shutil.disk_usage("/domains")
        return {"free_GB": round(u.free / 1e9, 1), "total_GB": round(u.total / 1e9, 1)}
    res["disk"] = add("磁盘（/domains 余量，SD1.5 约需 4–5 GB）", disk)

    # ③ 能不能装 diffusers
    def install():
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                            "--no-cache-dir", "diffusers", "transformers", "accelerate",
                            "safetensors", "Pillow"],
                           capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            raise RuntimeError((r.stderr or r.stdout)[-300:])
        import diffusers, transformers
        return {"diffusers": diffusers.__version__, "transformers": transformers.__version__}
    res["pip"] = add("pip install diffusers/transformers/accelerate/Pillow", install)

    # ④ 能不能连 hub 下模型
    def hub():
        from huggingface_hub import hf_hub_download
        p = hf_hub_download("hf-internal-testing/tiny-stable-diffusion-torch",
                            "model_index.json", cache_dir="/domains/hf_cache")
        return p
    res["hub"] = add("从 hub 下载 tiny 扩散管线（model_index.json）", hub)

    # ⑤ tiny 管线端到端（下载→管线→GPU→解码）
    def tiny_pipe():
        import torch
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-torch", cache_dir="/domains/hf_cache",
            torch_dtype=torch.float32, safety_checker=None)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(dev)
        g = torch.Generator(device=dev).manual_seed(0)
        out = pipe("a photo of a cat", num_inference_steps=2, guidance_scale=7.5,
                   generator=g, height=64, width=64)
        img = out.images[0]
        return {"device": dev, "size": list(img.size), "mode": img.mode}
    res["tiny_pipeline"] = add("tiny 扩散管线端到端（生成一张 64×64）", tiny_pipe)

    # ⑥ THINGS-EEG 图像像素：能不能读、数量、尺寸
    def images():
        from PIL import Image
        root = Path("/training/eeg/things-eeg/Image_set")
        cands = sorted(root.glob("*")) if root.exists() else []
        info = {"root_exists": root.exists(), "top_entries": [p.name for p in cands[:6]]}
        jpgs = []
        for d in cands:
            if d.is_dir():
                jpgs = sorted(d.glob("*.jpg"))
                break
        info["n_jpg_in_first_dir"] = len(jpgs)
        if jpgs:
            im = Image.open(jpgs[0])
            info["first"] = {"name": jpgs[0].name, "size": list(im.size), "mode": im.mode}
        return info
    res["images"] = add("THINGS-EEG 图像像素（Image_set）可读性", images)

    # ⑦ VAE 能否把图像编成 latent（生成训练的前置）
    def vae():
        from PIL import Image
        import torch
        from diffusers import AutoencoderKL
        # ⚠️ THINGS-EEG 的图像是**两层**：`Image_set/{split}_images/{概念}/{文件}.jpg`
        jpgs = sorted(Path("/training/eeg/things-eeg/Image_set").glob("*/*/*.jpg"))
        if not jpgs:
            raise RuntimeError("没找到 jpg（glob 深度 */*/*.jpg）")
        import numpy as np
        info = {"n_jpg_total": len(jpgs),
                "splits": sorted({p.parts[-3] for p in jpgs})}
        im = Image.open(jpgs[0]).convert("RGB")
        x = torch.from_numpy(np.array(im)).float().permute(2, 0, 1)[None] / 127.5 - 1
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse",
                                           cache_dir="/domains/hf_cache")
        with torch.no_grad():
            z = vae.encode(x).latent_dist.mean
        info.update({"first": jpgs[0].name, "px_size": list(im.size),
                     "latent_shape": list(z.shape)})
        return info
    res["vae"] = add("SD-VAE 编码 + THINGS-EEG 图像清单", vae)

    (out / "exp102_probe.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log("\n=== 探针小结 ===")
    for r in res["steps"]:
        log(f"  {'OK ' if r['ok'] else 'FAIL'} {r['name']} ({r['sec']}s)")
    log(f"硬件 {res.get('hardware', {}).get('info', '')}")
    log(f"磁盘 {res.get('disk', {}).get('info', '')}")
    log(f"结果写入 {out}/exp102_probe.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
