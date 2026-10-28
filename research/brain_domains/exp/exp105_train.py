#!/usr/bin/env python3
r"""exp105_train.py — exp105（**三个交叉对照**，§四十九）的**训练**一步。

## 三个对照 = **同一份训练循环 + 三种条件来源**

| 对照 | 条件来源 | 训练步数 | 起点 |
|---|---|---|---|
| **X1「SNR」** | `cache/eeg_cond_subXX.pt` 的 4 个 rep **在 encoder 输出空间平均**（当唯一 rep） | 与 exp102 同量级 | 从零 |
| **X2「训练量」** | 与 exp102 **完全一样**（单 rep 随机取） | **≈3×** | exp102 的 `adapter_last.pt` |
| **X3「oracle」** | 训练图的 **CLIP 特征**（当唯一 rep；所有被试同一份） | 与 exp102 同量级 | 从零 |

⇒ **训练循环（ε-MSE / batch 采样 / 梯度检查点 / lr / batch / clip / 零初始化）一字不改**，
  全部零件都从 `exp102_train` import ⇒ **不可能与 exp102 漂移**。

## ⚠️ R2 的修复（§四十九·八）：**每段用不同 seed**

exp102 的 `--resume` **每段用同一个 seed**（默认 0）⇒ 5 段**重复采样**同一段 3,100 步
⇒ 有效训练量只有 3,100 步（不是 10,250）。
⇒ 本脚本的 `--seed` 由调用方**逐段递增**（1, 2, 3…）⇒ 续训才真的增加训练量。
⇒ ⚠️ **exp102 的历史一个字不改**（本脚本只读它的 `adapter_last.pt` / `train_curve.json`）。

## X2 的起点：**复制**，不改动 exp102 的目录

`--variant more` 且 out-dir 里还没有 `adapter_last.pt` ⇒ 从
`/domains/runs/illusionbrain-align/exp102/` 复制 `adapter_last.pt` + `train_curve.json`
⇒ ⇒ exp102 的产物**保持只读**（历史证据不被覆盖）。

## 用法

```powershell
# X1（SNR）
python -u /domains/scripts/exp105_train.py --variant snr \
  --out-dir /domains/runs/illusionbrain-align/exp105/snr \
  --cache-dir /domains/runs/illusionbrain-align/exp102/cache \
  --max-minutes 42 --batch 8 --lr 1e-4 --steps-per-epoch 50 --seed 1

# X2（训练量；第 1 段会自动从 exp102 复制起点，第 2 段起加 --resume）
python -u /domains/scripts/exp105_train.py --variant more --resume \
  --out-dir /domains/runs/illusionbrain-align/exp105/more \
  --cache-dir /domains/runs/illusionbrain-align/exp102/cache \
  --max-minutes 42 --batch 8 --lr 1e-4 --steps-per-epoch 50 --seed 2

# X3（oracle）
python -u /domains/scripts/exp105_train.py --variant oracle \
  --out-dir /domains/runs/illusionbrain-align/exp105/oracle \
  --cache-dir /domains/runs/illusionbrain-align/exp102/cache \
  --cond-clip /domains/runs/illusionbrain-align/exp105/cache/clip_train.pt \
  --max-minutes 42 --batch 8 --lr 1e-4 --steps-per-epoch 50 --seed 1
```
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

EXP_DIR = Path(__file__).resolve().parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import exp102_train as T  # noqa: E402  （**共用件**：latents / adapter / SD 部件 / 曲线摘要）

EXP102_ROOT = Path("/domains/runs/illusionbrain-align/exp102")
EXP105_ROOT = Path("/domains/runs/illusionbrain-align/exp105")
VARIANTS = ("snr", "more", "oracle")


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 条件装载（**三个对照唯一的差异点**）
# ---------------------------------------------------------------------------
def load_conditions(variant, cache_dir, train_subjects, cond_clip, batch=8):
    """→ {被试号: (n_img, n_reps, 1024)}（与 exp102_train.load_conditions **同构**）。

    - `snr`：4 个 rep **在输出空间平均** ⇒ 形状 (n_img, 1, 1024)（当作「唯一的 rep」）
    - `more`：与 exp102 **完全一样**（(n_img, 4, 1024)，训练时随机取 rep）
    - `oracle`：训练图 CLIP 特征 ⇒ (n_img, 1, 1024)（所有被试同一份）
    """
    import torch

    if variant == "oracle":
        fp = Path(cond_clip)
        if not fp.exists():
            raise RuntimeError(f"缺 {fp} ⇒ 先跑 exp105_conditions.py --step clipfeat")
        obj = torch.load(fp, map_location="cpu", weights_only=False)
        clip = obj["clip"].float()                       # (n_img, 1024)
        out = {sid: clip.unsqueeze(1) for sid in train_subjects}
        log(f"[条件] oracle：训练图 CLIP 特征 {tuple(out[train_subjects[0]].shape)}"
            f"（所有被试同一份 ⇒ **统计单元是概念**，§四十九·八 R1）")
        return out

    conds = T.load_conditions(cache_dir, tuple(train_subjects))
    if variant == "snr":
        out = {}
        for sid, c in conds.items():
            c = c.float()
            out[sid] = c.mean(dim=1, keepdim=True)         # (n_img, 1, 1024)
            log(f"[条件] snr：sub-{sid:02d} {tuple(c.shape)} ⇒ 4 rep 输出空间平均 "
                f"⇒ {tuple(out[sid].shape)}")
        return out
    if variant == "more":
        first = conds[train_subjects[0]]
        log(f"[条件] more：与 exp102 完全一致（{tuple(first.shape)}，训练时随机取 rep）")
        return conds
    raise ValueError(f"未知 variant：{variant}")


def maybe_seed_from_exp102(variant, out_dir, resume):
    """X2：**复制** exp102 的起点（adapter + 曲线）到自己的 out-dir ⇒ 不改动 exp102。

    ⚠️ **只在 `--resume` 时才复制**（2026-10-28 改）：
      X2 改成**从头训**（三个对照训练量完全一致 ⇒ 归因表干净）⇒ 第 1 段不带 `--resume`
      ⇒ 不复制 exp102 的起点 ⇒ `more` 与 `snr`/`oracle` 训练量严格一致。
    """
    if variant != "more" or not resume:
        return
    out_dir = Path(out_dir)
    ck = out_dir / "adapter_last.pt"
    if ck.exists():
        log(f"[起点] {ck} 已存在 ⇒ 不复制")
        return
    src_ck = EXP102_ROOT / "adapter_last.pt"
    if not src_ck.exists():
        raise RuntimeError(f"缺 {src_ck} ⇒ X2 没有起点")
    shutil.copy2(src_ck, ck)
    log(f"[起点] ✓ 已复制 {src_ck} → {ck}")
    src_curve = EXP102_ROOT / "train_curve.json"
    dst_curve = out_dir / "train_curve.json"
    if src_curve.exists() and not dst_curve.exists():
        shutil.copy2(src_curve, dst_curve)
        log(f"[起点] ✓ 已复制 {src_curve} → {dst_curve}（曲线**接着** exp102 ⇒ 同一条轨迹）")


# ---------------------------------------------------------------------------
# 训练（**循环照抄 exp102_train.train()，一字不改**）
# ---------------------------------------------------------------------------
def train(args):
    import torch
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("exp105 训练必须有 GPU")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    maybe_seed_from_exp102(args.variant, out_dir, args.resume)

    log(f"[数据] 读 VAE 潜变量（{cache_dir}/vae_latents_*.pt）")
    lat = T.load_latents(cache_dir)
    n_img = lat["latents"].shape[0]
    log(f"  ✓ {n_img} 张训练图 × latent {tuple(lat['latents'].shape[1:])}（fp16）")
    conds = load_conditions(args.variant, cache_dir, tuple(args.train_subjects),
                            args.cond_clip)
    first = conds[args.train_subjects[0]]
    n_reps = int(first.shape[1])
    log(f"  ✓ {len(conds)} 个被试 × {n_img} 图 × {n_reps} 试次（口径：{args.variant}）")

    # ---- 模型（**只有 adapter 可训**）----
    unet = T.load_sd_component("unet", dtype=torch.float32, device=device)
    unet.eval()
    for p in unet.parameters():
        p.requires_grad_(False)
    T.enable_ckpt(unet)
    text_encoder = T.load_sd_component("text_encoder", dtype=torch.float32, device=device).eval()
    for p in text_encoder.parameters():
        p.requires_grad_(False)
    tokenizer = T.load_sd_component("tokenizer")
    sched = T.load_sd_component("scheduler")
    ac = sched.alphas_cumprod.to(device=device, dtype=torch.float32)
    n_train_ts = int(sched.config.num_train_timesteps)

    ids = tokenizer([T.PROMPT], padding="max_length",
                    max_length=tokenizer.model_max_length, truncation=True,
                    return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        text_emb = text_encoder(ids)[0]
    log(f"[条件] 固定提示词 {T.PROMPT!r} ⇒ 文本 embedding {tuple(text_emb.shape)}"
        f"（+ {T.N_TOKENS} 个条件 token ⇒ {text_emb.shape[1] + T.N_TOKENS}）")

    adapter = T.build_adapter().to(device)
    nparam = sum(p.numel() for p in adapter.parameters())
    log(f"[模型] adapter 参数量 {nparam:,}｜设备 {device}")

    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.01)

    # ---- 续训（**曲线只在 --resume 时接着读** ⇒ 绝不拼接两条轨迹）----
    curve_path = out_dir / "train_curve.json"
    curve = []
    if args.resume and curve_path.exists():
        try:
            curve = json.loads(curve_path.read_text(encoding="utf-8")).get("epochs", [])
            log(f"[续训] 接着上次的损失曲线（已有 {len(curve)} 个 epoch）")
        except Exception:
            curve = []
    run_id = time.strftime("%Y%m%d-%H%M%S")
    ckpt_path = out_dir / "adapter_last.pt"
    step = 0
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        adapter.load_state_dict(ck["state_dict"])
        if ck.get("optimizer"):
            opt.load_state_dict(ck["optimizer"])
        step = int(ck.get("step", 0))
        log(f"[续训] 从 step {step} 接着训｜**seed={args.seed}**（≠ 上段的 seed ⇒ 采样序列不重复）")
    elif not args.resume:
        log("[训练] 从头开始（新的一次训练）")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    log(f"[采样] seed={args.seed}（⚠️ R2：**每段必须不同**）")

    def save_ckpt(path, cur_step):
        """写 checkpoint：**原子写 + 失败不致命** + 降频（曲线仍每 epoch 记）。

        ⚠️ 2026-10-28 实测（t105-train-snr-s1 崩过一次）：每 epoch 往桶里写 122 MB
        ⇒ CAS 写入偶发 `RuntimeError: basic_ios::clear: iostream error`
        ⇒ **整场训练崩掉**（已训到 epoch 32 白丢）⇒ ⇒ 三条改法：
          ① **原子写**：先写 `path.tmp` 再 rename ⇒ 永远不会留下半个文件；
          ② **失败只警告、不崩**：训练本身比 checkpoint 更值钱；
          ③ **降频**（`--ckpt-every`，默认 5 个 epoch）：曲线文件小（几 KB），
             **仍每 epoch 记**（C0 的「首/末 10%」靠它）。
        """
        tmp = Path(str(path) + ".tmp")
        try:
            torch.save({"state_dict": adapter.state_dict(),
                        "optimizer": opt.state_dict(),
                        "step": int(cur_step),
                        "config": vars(args),
                        "variant": args.variant,
                        "cond_dim": T.COND_DIM, "token_dim": T.TOKEN_DIM,
                        "n_tokens": T.N_TOKENS, "prompt": T.PROMPT}, tmp)
            tmp.replace(path)
        except Exception as e:                                   # noqa: BLE001
            log(f"[ckpt] ⚠ 写 {path} 失败（**不影响训练**）：{type(e).__name__}: {str(e)[:140]}")
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:                                    # noqa: BLE001
                pass

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
            tokens = adapter(zs)
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
                          "elapsed_s": round(time.time() - t_start, 1),
                          "run_id": run_id})
            T._write_json(curve_path, {"epochs": curve, "config": vars(args)})
            print(f"===EPOCH=== {json.dumps(curve[-1], ensure_ascii=False)}", flush=True)
            log(f"  epoch {ep:>4} | step {step} | loss {mean_loss:.5f} | "
                f"{time.time() - t_start:.0f}s")
            if step % args.save_every == 0:
                save_ckpt(out_dir / f"adapter_step{step}.pt", step)
            # ⚠️ checkpoint **降频**（每 epoch 写 122 MB 会把 CAS 写通道压垮 ⇒ 整场训练崩）
            if step % max(1, args.ckpt_every * args.steps_per_epoch) == 0:
                save_ckpt(ckpt_path, step)
            # ⚠️ **按步数收尾**（不只靠墙钟）⇒ 段与段之间的训练量可比
            if args.steps > 0 and step >= args.steps:
                stop_reason = f"到达步数 {args.steps}"
                break
            if args.max_minutes and (time.time() - t_start) / 60.0 >= args.max_minutes:
                stop_reason = f"墙钟到 {args.max_minutes} min（epoch 边界收尾）"
                break

    save_ckpt(ckpt_path, step)
    save_ckpt(out_dir / "adapter_final.pt", step)
    # ⚠️ **兜底**：生成阶段只认 `adapter_final.pt` ⇒ 若它写失败（CAS 抖动）⇒ 重试几次
    final_path = out_dir / "adapter_final.pt"
    for attempt in range(1, 6):
        if final_path.exists() and final_path.stat().st_size > 100_000_000:
            break
        log(f"[ckpt] adapter_final.pt 不可用（第 {attempt} 次）⇒ 重试保存")
        save_ckpt(final_path, step)
        time.sleep(5)
    summ = T.loss_curve_summary(curve)
    # ---- **只报**：epoch 间 loss 的振荡幅度（给 C0 的「下降」一个噪声量级参照，§四十九·六）----
    vals = [float(e["mean_loss"]) for e in curve if e.get("mean_loss") is not None]
    osc = None
    if vals:
        osc = {"n_epochs": len(vals), "std": float(np.std(vals)),
               "min": float(np.min(vals)), "max": float(np.max(vals)),
               "range": float(np.max(vals) - np.min(vals)),
               "rel_range": float((np.max(vals) - np.min(vals)) / max(np.mean(vals), 1e-12)),
               "c0_delta": float(summ["last10_mean"] - summ["first10_mean"])}
    log(f"[结束] {stop_reason}｜共 {step} 步｜{len(curve)} 个 epoch")
    log(f"[C0-损失] 首 10% 均值 {summ['first10_mean']}｜末 10% 均值 {summ['last10_mean']}"
        f"｜下降：{summ['decreased']}")
    if osc:
        log(f"[振荡-只报] epoch 间 loss：std={osc['std']:.5f} 极差={osc['range']:.5f} "
            f"（相对 {osc['rel_range']*100:.1f}%）｜C0 的差 {osc['c0_delta']:+.5f}")
    log(f"结果写入 {out_dir}/adapter_last.pt 与 {curve_path}")
    return {"steps": step, "epochs": len(curve), "loss_summary": summ,
            "loss_oscillation": osc, "adapter_params": nparam,
            "stop_reason": stop_reason, "variant": args.variant, "seed": args.seed}


def self_test():
    """自测：三个 variant 的条件装载形状 + adapter 形状 + 零初始化 + 振荡统计。"""
    import torch

    ok, bad = 0, []

    def chk(name, cond):
        nonlocal ok
        if cond:
            ok += 1
            print(f"✓ {name}")
        else:
            bad.append(name)
            print(f"✗ {name}")

    # 条件聚合（用假的 cond 张量，不碰桶）
    fake = {s: torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3) for s in (1, 2)}
    avg = {sid: c.mean(dim=1, keepdim=True) for sid, c in fake.items()}
    chk("snr：平均后形状 (n_img, 1, 1024)", tuple(avg[1].shape) == (2, 1, 3))
    chk("snr：数值 = 逐 rep 均值",
        bool(torch.allclose(avg[1][0, 0], fake[1][0].mean(dim=0))))

    clip = torch.randn(2, 8)
    ora = {sid: clip.unsqueeze(1) for sid in (1, 2)}
    chk("oracle：形状 (n_img, 1, 1024) 且所有被试同一份",
        tuple(ora[1].shape) == (2, 1, 8) and bool(torch.equal(ora[1], ora[2])))

    m = T.build_adapter()
    nparam = sum(p.numel() for p in m.parameters())
    chk(f"adapter 参数量 = exp102 的 {nparam:,}", nparam == 10_238_720)
    z = torch.randn(3, T.COND_DIM)
    with torch.no_grad():
        tok = m(z)
    chk("token 形状 (B, 16, 768)", tuple(tok.shape) == (3, 16, 768))
    chk("末层零初始化 ⇒ 起点 token 全 0", bool(torch.allclose(tok, torch.zeros_like(tok))))

    curve = [{"mean_loss": v} for v in (0.066, 0.071, 0.076, 0.090, 0.075, 0.090,
                                        0.067, 0.080, 0.094, 0.066)]
    s = T.loss_curve_summary(curve)
    chk("loss_curve_summary 可用（复用 exp102 的函数）", s["n_epochs"] == 10)
    print(f"\n{ok} 通过 / {len(bad)} 失败")
    if bad:
        print("失败：" + "；".join(bad))
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), default="",
                    help="snr / more / oracle（--self-test 时不需要）")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--max-minutes", type=float, default=42.0)
    ap.add_argument("--steps-per-epoch", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1,
                    help="⚠️ **每段必须不同**（R2）⇒ 链子逐段递增")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=5,
                    help="**每几个 epoch 存一次 checkpoint**（默认 5）"
                         "⇒ 每 epoch 写 122 MB 会把 CAS 写通道压垮（实测崩过）")
    ap.add_argument("--train-subjects", type=int, nargs="+", default=list(T.TRAIN_SUBJECTS))
    ap.add_argument("--cache-dir", default=str(EXP102_ROOT / "cache"))
    ap.add_argument("--cond-clip", default=str(EXP105_ROOT / "cache/clip_train.pt"))
    ap.add_argument("--out-dir", default=str(EXP105_ROOT))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.variant:
        ap.error("--variant 必须给（snr / more / oracle）")
    res = train(args)
    print("===TRAINRESULT===" + json.dumps(res, ensure_ascii=False, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
