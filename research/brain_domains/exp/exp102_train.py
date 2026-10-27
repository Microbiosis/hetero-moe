#!/usr/bin/env python3
"""exp102_train.py — exp102（②③ 条件生成线：**连续条件 + 冻结 SD1.5**）的**训练**一步，
同时是本轮的**共用组件库**（`exp102_cache.py` / `exp102_gen.py` 都从这里 import）。

## 这一步在做什么

训**唯一一个可训练的部件**：**adapter**（EEG 条件 → SD 的 cross-attention 条件 token）。
SD1.5 的 **UNet / VAE / text encoder 全冻结**（§四十四·二）⇒ 这一步**不学扩散模型本身**，
只学「**怎么把 1024 维脑侧条件摆成 SD 听得懂的条件 token**」。

## 为什么这么写（口径全部来自 §四十四·五，**一字不改**）

| 项 | 定死成什么 | 在代码里对应哪一行 |
|---|---|---|
| 底模 | **SD1.5**（UNet/VAE/text encoder 全冻结） | `load_sd_component()` |
| 条件 token | EEG(1024) → MLP → **K=16 个 768 维 token**，**追加**在文本 embedding 之后 ⇒ `encoder_hidden_states` 长 **77+16** | `build_adapter()` + `enc = cat([text, tokens])` |
| 训练 | **ε-prediction MSE**；**只训 adapter**；AdamW；lr **1e-4**；batch **8**；**梯度检查点** | `train()` |
| 分辨率 / 采样 | 512×512；**DDIM 30 步**；**guidance scale = 1.0**（**不做 CFG**） | `exp102_gen.py` |
| 训练数据 | train split：每个概念的**全部 10 张训练照片** × 它们的**全部可用 EEG rep** × **10 个 train 被试** | `exp102_cache.py` |

- **末层零初始化**：初始时 16 个 token 全为 0 ⇒ 起点是「**没有脑侧条件**」的 SD1.5
  ⇒ 训练是**从无条件往有条件推**，而不是从随机 token 噪声开始。
- **只训 adapter**：UNet 的参数 `requires_grad_(False)`，但梯度**必须**穿过 UNet 回传到 token
  ⇒ 这正是「冻结生成先验 + 只训条件支」的含义（梯度检查点只是为了省激活显存）。
- **按墙钟时间收尾**（`--max-minutes`）：平台 Job 的硬上限是 60 min ⇒ 脚本在**epoch 边界**
  干净地停下来并落盘 checkpoint + 损失曲线 ⇒ 下一段 Job 用 `--resume` 接着训
  ⇒ **超时不丢中间产物**（§四十四 的实现约定 + 平台硬约束）。

## 本文件同时是 exp102 的共用组件库

- `exp102_cache.py` 从这里取 SD 仓库候选（要用 SD1.5 **自带**的 VAE ⇒ 潜变量与 UNet 同一个先验）。
- `exp102_gen.py` 从这里取 UNet / VAE / text encoder / scheduler / adapter 定义 / prompt。
  ⇒ **一条事实只写一处**。

## 用法

```powershell
# 本地自测（需要 torch；不碰 diffusers）
python -u research/brain_domains/exp/exp102_train.py --self-test

# 云端（GPU，a10g-large）
python -u /domains/scripts/exp102_train.py --max-minutes 50 --batch 8 --lr 1e-4
# 续训（读 adapter_last.pt，损失曲线接着写）
python -u /domains/scripts/exp102_train.py --resume --max-minutes 50 --batch 8 --lr 1e-4
```
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

EXP_DIR = Path(__file__).resolve().parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

# ===========================================================================
# §四十四·五 定死的口径（**一字不改**）
# ===========================================================================
SD_REPO_CANDIDATES = ("stable-diffusion-v1-5/stable-diffusion-v1-5",
                      "runwayml/stable-diffusion-v1-5",
                      "sd-legacy/stable-diffusion-v1-5")
PROMPT = "a photo"          # **固定**提示词：不含任何概念信息 ⇒ 语义只能来自 EEG 条件
COND_DIM = 1024             # exp93b 冻结 encoder 的输出维度
TOKEN_DIM = 768             # SD1.5 cross-attn 的 token 维度
N_TOKENS = 16               # §四十四·五：K = 16
ADAPTER_HIDDEN = 768
TRAIN_SUBJECTS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
HF_CACHE = "/domains/hf_cache"
DEFAULT_OUT = Path("/domains/runs/illusionbrain-align/exp102")


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# SD1.5 部件（**三个候选仓库按顺序试**；全失败就报错，不静默换底模）
# ---------------------------------------------------------------------------
def ensure_deps():
    """**Job 镜像里没有 diffusers / transformers**（探针那次是自己 pip 装的）
    ⇒ 正式脚本必须自己装（幂等：已装就跳过 ⇒ 不重复耗时间）。

    ⚠️ 这一步是 exp102_cache 的 vae step 第一次失败的原因（`No module named 'diffusers'`）。
    """
    import importlib
    import subprocess

    need = []
    for mod, pkg in (("diffusers", "diffusers"), ("transformers", "transformers"),
                     ("accelerate", "accelerate"), ("safetensors", "safetensors"),
                     ("PIL", "Pillow")):
        try:
            importlib.import_module(mod)
        except Exception:                                     # noqa: BLE001
            need.append(pkg)
    if not need:
        return {"installed": []}
    log(f"[依赖] 缺 {need} ⇒ pip install（镜像里没有 diffusion 全家桶）")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                        "--no-cache-dir", *need],
                       capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError("pip install 失败: " + (r.stderr or r.stdout)[-500:])
    return {"installed": need}


def load_sd_component(kind, dtype=None, device=None):
    """取 SD1.5 的某个部件（unet / vae / text_encoder / tokenizer / scheduler）。

    ⚠️ 优先下 **fp16 权重**（体积小一半、下载快；fp16 的数值能精确放进 fp32），
    失败再退回完整权重 ⇒ 平台上的下载时间是硬约束。
    """
    import torch

    ensure_deps()
    last = None
    for repo in SD_REPO_CANDIDATES:
        for variant in ("fp16", None):
            try:
                obj = _sd_from_pretrained(kind, repo, dtype, variant)
                log(f"  ✓ SD1.5[{kind}] ← {repo}"
                    f"{'（fp16 权重）' if variant else ''}")
                if device is not None:
                    obj = obj.to(device)
                return obj
            except Exception as e:                       # noqa: BLE001
                last = e
                log(f"  ⚠ [{kind}] {repo} variant={variant} 不可用："
                    f"{type(e).__name__}: {str(e)[:140]}")
    raise RuntimeError(f"SD1.5 的 {kind} 取不到（候选仓库全失败）：{last}")


def _sd_from_pretrained(kind, repo, dtype, variant):
    import torch

    kw = {"cache_dir": HF_CACHE}
    if dtype is not None:
        kw["torch_dtype"] = dtype
    if variant is not None:
        kw["variant"] = variant
    if kind == "unet":
        from diffusers import UNet2DConditionModel
        return UNet2DConditionModel.from_pretrained(repo, subfolder="unet", **kw)
    if kind == "vae":
        from diffusers import AutoencoderKL
        return AutoencoderKL.from_pretrained(repo, subfolder="vae", **kw)
    if kind == "text_encoder":
        from transformers import CLIPTextModel
        return CLIPTextModel.from_pretrained(repo, subfolder="text_encoder", **kw)
    if kind == "tokenizer":
        from transformers import CLIPTokenizer
        return CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer", cache_dir=HF_CACHE)
    if kind == "scheduler":
        from diffusers import DDIMScheduler
        return DDIMScheduler.from_pretrained(repo, subfolder="scheduler", cache_dir=HF_CACHE)
    raise ValueError(f"未知的 SD 部件：{kind}")


# ---------------------------------------------------------------------------
# adapter（**本轮唯一可训练的部件**）
# ---------------------------------------------------------------------------
def build_adapter(cond_dim=COND_DIM, token_dim=TOKEN_DIM, n_tokens=N_TOKENS,
                  hidden=ADAPTER_HIDDEN):
    """EEG(1024) → **K=16 个 768 维 token**（§四十四·五）。

    结构：LayerNorm → Linear(1024→768) → GELU → Linear(768→K*768) → reshape (B, K, 768)
    ⇒ 参数量 ≈ **10.2M**（§四十四·五：「几 M 参数（≈10M）」）。
    ⚠️ **末层零初始化** ⇒ 初始 token 全为 0 ⇒ 起点是「无脑侧条件」的 SD1.5。
    """
    import torch
    from torch import nn

    class CondAdapter(nn.Module):
        """条件 token 支：把 1024 维脑侧条件投成 SD cross-attn 听得懂的 K 个 token。"""

        def __init__(self):
            super().__init__()
            self.n_tokens = n_tokens
            self.token_dim = token_dim
            self.net = nn.Sequential(
                nn.LayerNorm(cond_dim),
                nn.Linear(cond_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, n_tokens * token_dim),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        def forward(self, z):
            return self.net(z).view(z.shape[0], self.n_tokens, self.token_dim)

    return CondAdapter()


def enable_ckpt(unet):
    """开梯度检查点（§四十四·五）。`use_reentrant=False` ⇒ 不依赖输入是否 require_grad。"""
    import torch

    def _ckpt(fn, *a, **kw):
        return torch.utils.checkpoint.checkpoint(fn, *a, use_reentrant=False, **kw)

    try:
        unet.enable_gradient_checkpointing(gradient_checkpointing_func=_ckpt)
    except TypeError:
        unet.enable_gradient_checkpointing()
    return unet


# ---------------------------------------------------------------------------
# 缓存读取（exp102_cache.py 的产物）
# ---------------------------------------------------------------------------
def load_latents(cache_dir):
    """→ {"latents": (n_img, 4, 64, 64) fp16 张量, "paths": [...]}（行号 = EEG 行号）"""
    import torch

    files = sorted(Path(cache_dir).glob("vae_latents_*.pt"))
    if not files:
        raise RuntimeError(f"{cache_dir} 里没有 vae_latents_*.pt ⇒ 先跑 exp102_cache.py --step vae")
    idx_all, lat_all, paths = [], [], []
    seen = set()
    for f in files:
        obj = torch.load(f, map_location="cpu", weights_only=False)
        # ⚠️ `idx` 可能是 **list**（缓存脚本就是 list 存的）也可能是张量 ⇒ 两种都能读
        idxs = obj["idx"]
        idxs = idxs.tolist() if hasattr(idxs, "tolist") else list(idxs)
        for i, l, p in zip(idxs, obj["latents"], obj["paths"]):
            i = int(i)
            if i in seen:      # 续跑可能留下重叠分片 ⇒ 去重（保留先到的）
                continue
            seen.add(i)
            idx_all.append(i)
            lat_all.append(l)
            paths.append(str(p))
    order = np.argsort(np.asarray(idx_all))
    idx_all = [idx_all[k] for k in order]
    lat_all = [lat_all[k] for k in order]
    paths = [paths[k] for k in order]
    if idx_all != list(range(len(idx_all))):
        raise RuntimeError(f"VAE 潜变量行号不连续（共 {len(idx_all)} 行）⇒ 缓存坏了")
    return {"latents": torch.stack(lat_all), "paths": paths}


def load_conditions(cache_dir, subjects=TRAIN_SUBJECTS):
    """→ {被试号: (n_img, n_reps, 1024) fp16 张量}（冻结 exp93b encoder 的输出）"""
    import torch

    out = {}
    for sid in subjects:
        f = Path(cache_dir) / f"eeg_cond_sub{sid:02d}.pt"
        if not f.exists():
            raise RuntimeError(f"缺 {f} ⇒ 先跑 exp102_cache.py --step eeg")
        obj = torch.load(f, map_location="cpu", weights_only=False)
        out[sid] = obj["cond"]
    return out


def loss_curve_summary(curve, frac=0.1):
    """**C0 的第一半**：末 10% epoch 的均值 < 首 10% epoch 的均值（**纯函数**）。"""
    vals = [float(e["mean_loss"]) for e in (curve or []) if e.get("mean_loss") is not None]
    n = len(vals)
    if n == 0:
        return {"n_epochs": 0, "k_epochs": 0, "first10_mean": None, "last10_mean": None,
                "decreased": False}
    k = max(1, int(round(frac * n)))
    first = float(np.mean(vals[:k]))
    last = float(np.mean(vals[-k:]))
    return {"n_epochs": n, "k_epochs": k, "first10_mean": first, "last10_mean": last,
            "decreased": bool(last < first)}


def _write_json(path, payload):
    """落盘（**先写临时文件再改名** ⇒ 写一半被杀不会留下半个 JSON）。"""
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def train(args):
    import torch
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("exp102 训练必须有 GPU（SD1.5 512×512 的 UNet 反传在 CPU 上不可行）")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    log(f"[数据] 读 VAE 潜变量（{cache_dir}/vae_latents_*.pt）")
    lat = load_latents(cache_dir)
    n_img = lat["latents"].shape[0]
    log(f"  ✓ {n_img} 张训练图 × latent {tuple(lat['latents'].shape[1:])}（fp16，SD1.5 自带 VAE）")
    log("[数据] 读 EEG 条件（**冻结** exp93b encoder 的 1024 维输出）")
    conds = load_conditions(cache_dir, tuple(args.train_subjects))
    first = conds[args.train_subjects[0]]
    n_reps = int(first.shape[1])
    log(f"  ✓ {len(conds)} 个被试 × {n_img} 图 × {n_reps} 试次")

    # ---- 模型（**只有 adapter 可训**）----
    unet = load_sd_component("unet", dtype=torch.float32, device=device)
    unet.eval()
    for p in unet.parameters():
        p.requires_grad_(False)
    enable_ckpt(unet)
    text_encoder = load_sd_component("text_encoder", dtype=torch.float32, device=device).eval()
    for p in text_encoder.parameters():
        p.requires_grad_(False)
    tokenizer = load_sd_component("tokenizer")
    sched = load_sd_component("scheduler")
    ac = sched.alphas_cumprod.to(device=device, dtype=torch.float32)
    n_train_ts = int(sched.config.num_train_timesteps)

    ids = tokenizer([PROMPT], padding="max_length",
                    max_length=tokenizer.model_max_length, truncation=True,
                    return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        text_emb = text_encoder(ids)[0]                 # (1, 77, 768)
    log(f"[条件] 固定提示词 {PROMPT!r} ⇒ 文本 embedding {tuple(text_emb.shape)}"
        f"（与 {N_TOKENS} 个条件 token 拼接 ⇒ {text_emb.shape[1] + N_TOKENS}）")

    adapter = build_adapter().to(device)
    nparam = sum(p.numel() for p in adapter.parameters())
    log(f"[模型] adapter 参数量 {nparam:,}（§四十四·五：几 M / ≈10M）｜设备 {device}")

    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.01)

    # ---- 续训 ----
    curve_path = out_dir / "train_curve.json"
    curve = []
    if curve_path.exists():
        try:
            curve = json.loads(curve_path.read_text(encoding="utf-8")).get("epochs", [])
        except Exception:
            curve = []
    ckpt_path = out_dir / "adapter_last.pt"
    step = 0
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        adapter.load_state_dict(ck["state_dict"])
        if ck.get("optimizer"):
            opt.load_state_dict(ck["optimizer"])
        step = int(ck.get("step", 0))
        log(f"[续训] 从 step {step} 接着训（损失曲线已有 {len(curve)} 个 epoch）")
    elif not args.resume:
        log("[训练] 从头开始（新的一次训练）")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    def save_ckpt(path, cur_step):
        torch.save({"state_dict": adapter.state_dict(),
                    "optimizer": opt.state_dict(),
                    "step": int(cur_step),
                    "config": vars(args),
                    "cond_dim": COND_DIM, "token_dim": TOKEN_DIM, "n_tokens": N_TOKENS,
                    "prompt": PROMPT}, path)

    t_start = time.time()
    losses = []
    stop_reason = "完成步数"
    while step < args.steps:
        idx = rng.integers(0, n_img, size=args.batch)
        subs = rng.integers(0, len(args.train_subjects), size=args.batch)
        rps = rng.integers(0, n_reps, size=args.batch)
        x0 = lat["latents"][torch.from_numpy(idx)].to(device=device, dtype=torch.float32)
        zs = torch.stack([conds[args.train_subjects[int(subs[b])]][int(idx[b]), int(rps[b])]
                          for b in range(args.batch)]).to(device=device, dtype=torch.float32)
        noise = torch.randn(x0.shape, device=device, dtype=torch.float32)
        t = torch.randint(0, n_train_ts, (args.batch,), device=device)
        a = ac[t].view(-1, 1, 1, 1)
        x_t = a.sqrt() * x0 + (1.0 - a).sqrt() * noise
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens = adapter(zs)                                   # (B, 16, 768)
            enc = torch.cat([text_emb.expand(args.batch, -1, -1), tokens], dim=1)
            pred = unet(x_t, t, encoder_hidden_states=enc).sample
        loss = F.mse_loss(pred.float(), noise.float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
        step += 1

        if step % args.steps_per_epoch == 0:
            mean_loss = float(np.mean(losses))
            losses = []
            ep = step // args.steps_per_epoch
            curve.append({"epoch": ep, "step": step, "mean_loss": mean_loss,
                          "elapsed_s": round(time.time() - t_start, 1)})
            _write_json(curve_path, {"epochs": curve, "config": vars(args)})
            print(f"===EPOCH=== {json.dumps(curve[-1], ensure_ascii=False)}", flush=True)
            log(f"  epoch {ep:>3} | step {step} | loss {mean_loss:.5f} | "
                f"{time.time() - t_start:.0f}s")
            if step % args.save_every == 0:
                save_ckpt(out_dir / f"adapter_step{step}.pt", step)
            save_ckpt(ckpt_path, step)
            if args.max_minutes and (time.time() - t_start) / 60.0 >= args.max_minutes:
                stop_reason = f"墙钟到 {args.max_minutes} min（epoch 边界收尾）"
                break

    save_ckpt(ckpt_path, step)
    save_ckpt(out_dir / "adapter_final.pt", step)
    summ = loss_curve_summary(curve)
    log(f"[结束] {stop_reason}｜共 {step} 步｜{len(curve)} 个 epoch")
    log(f"[C0-损失] 首 10% 均值 {summ['first10_mean']}｜末 10% 均值 {summ['last10_mean']}"
        f"｜下降：{summ['decreased']}")
    log(f"结果写入 {out_dir}/adapter_last.pt 与 {curve_path}")
    return {"steps": step, "epochs": len(curve), "loss_summary": summ,
            "adapter_params": nparam, "stop_reason": stop_reason}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100000, help="步数上限（很少用满）")
    ap.add_argument("--max-minutes", type=float, default=50.0,
                    help="墙钟上限（分钟）⇒ 到点后在 epoch 边界干净收尾（平台上限 60 min）")
    ap.add_argument("--steps-per-epoch", type=int, default=250)
    ap.add_argument("--batch", type=int, default=8)          # §四十四·五
    ap.add_argument("--lr", type=float, default=1e-4)         # §四十四·五
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--train-subjects", type=int, nargs="+", default=list(TRAIN_SUBJECTS))
    ap.add_argument("--cache-dir", default=str(DEFAULT_OUT / "cache"))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    res = train(args)
    print("===TRAINRESULT===" + json.dumps(res, ensure_ascii=False, default=str), flush=True)
    return 0


def self_test():
    """自测：adapter 的形状 / 参数量 / 零初始化 / 损失曲线摘要。"""
    import torch

    ok, bad = 0, []

    def chk(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            bad.append(name)

    m = build_adapter()
    nparam = sum(p.numel() for p in m.parameters())
    z = torch.randn(5, COND_DIM)
    with torch.no_grad():
        out = m(z)
    chk("adapter 输出形状 (B, 16, 768)", tuple(out.shape) == (5, N_TOKENS, TOKEN_DIM))
    chk("adapter 参数量 ≈10M（§四十四·五：几 M / ≈10M）",
        8e6 <= nparam <= 13e6)
    chk("末层零初始化 ⇒ 初始 token 全为 0（起点 = 无条件）",
        bool(torch.all(out == 0)))
    # 梯度能回到第一层（零初始化后第一层梯度为 0 是正常的，但形状要对）
    out2 = m(torch.randn(2, COND_DIM))
    out2.sum().backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    chk("反传有梯度（至少末层有）", len(grads) >= 1)

    c = [{"mean_loss": 1.0 - 0.01 * i} for i in range(20)]
    s = loss_curve_summary(c)
    chk("损失曲线摘要：末 10% < 首 10% ⇒ decreased=True", s["decreased"] is True)
    c2 = [{"mean_loss": 1.0 + 0.01 * i} for i in range(20)]
    s2 = loss_curve_summary(c2)
    chk("损失曲线摘要：上升 ⇒ decreased=False", s2["decreased"] is False)
    s3 = loss_curve_summary([])
    chk("损失曲线摘要：空曲线不崩", s3["n_epochs"] == 0 and s3["decreased"] is False)

    print(f"\n{'✓ 全部通过' if not bad else '✗ 有失败'}：{ok} 项检查"
          f"{'，失败：' + ', '.join(bad) if bad else ''}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
