#!/usr/bin/env python3
"""exp102_cache.py — exp102（②③ 条件生成线：**连续条件 + 冻结 SD1.5**）的**数据缓存**一步。

## 这一步在做什么

把 exp102 训练/生成所需的**两样重资产**预先算好、落盘到 `/domains`：

1. **EEG 条件**（`--step eeg`）：用**冻结**的 exp93b encoder（`encoder_best.pt`）把
   THINGS-EEG **train split** 的 EEG 编成 **1024 维条件向量**（每被试 × 每图 × 每试次一条）。
2. **VAE 潜变量**（`--step vae`）：用 **SD1.5 自带的** VAE 把 train split 的 **16,540 张**照片
   编成 `(4, 64, 64)` 潜变量（缩放系数用 SD1.5 的 `scaling_factor` = 0.18215）。

## 为什么这么写（算力与内存的账）

- **只训 adapter**（§四十四·五）⇒ 训练步里**不能**现编 VAE 潜变量：
  512×512 的一次 VAE 编码 ≈ 一次 UNet 前向的 1/5 成本 ⇒ 现编等于白烧 ~20% 算力
  ⇒ **必须预缓存**。
- ⚠️ **潜变量必须用 SD1.5 自带的 VAE**：底模被换掉 ⇒ 潜变量分布就变了 ⇒ 训练目标对不上。
  （探针里试过的 `sd-vae-ft-mse` **只是探测用**，本轮**不用**。）
- EEG 原文件是**每被试 41.7 GB**（16540×80×63×250 fp16）⇒ **不能**整份读进内存
  ⇒ `torch.load(mmap=True)` + **逐试次切片**，峰值内存只有 1 个试次（~0.5 GB）。
- **必须能分片并行**（平台同时只跑 2 个 Job）⇒ `--shard i --n-shards N`：
  - `--step eeg`：按**被试**分片（每个被试的读盘量最大 ⇒ 按被试切最均衡）；
  - `--step vae`：按**图像行号**分片（VAE 编码成本均匀 ⇒ 按行号切最均衡）。

## 行号约定（**一条事实只写一处**）

VAE 潜变量的**行号 = EEG 的行号 = train.pt 的行号 = 「概念升序 × 每概念 10 张」**。
行 ↔ 照片的对应**以 `train.pt` 的 `img` 字段为准**（它直接编码了这个配对），
文件系统只用来**定位文件**。

## 用法

```powershell
# 云端（GPU；eeg 与 vae 各分 2 片 ⇒ 4 个 Job，同时最多 2 个）
python -u /domains/scripts/exp102_cache.py --step eeg --shard 0 --n-shards 2 \
  --out-dir /domains/runs/illusionbrain-align/exp102
python -u /domains/scripts/exp102_cache.py --step vae --shard 0 --n-shards 2 \
  --out-dir /domains/runs/illusionbrain-align/exp102
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

TRAIN_PT = Path("/training/eeg/things-eeg/Preprocessed_data_250Hz_whiten")
IMAGE_SET = Path("/training/eeg/things-eeg/Image_set")
ENCODER_CKPT = "/domains/runs/illusionbrain-align/exp93b/encoder_best.pt"
DEFAULT_OUT = Path("/domains/runs/illusionbrain-align/exp102")
TRAIN_SUBJECTS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 读取 train.pt（**mmap**：41.7 GB 的文件不能整份读进内存）
# ---------------------------------------------------------------------------
def load_train_pt(subject):
    import torch

    fp = TRAIN_PT / f"sub-{subject:02d}" / "train.pt"
    if not fp.exists():
        raise FileNotFoundError(f"缺 {fp}")
    try:
        return torch.load(fp, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:                       # 老版 torch 没有 mmap 参数
        return torch.load(fp, map_location="cpu", weights_only=False)


def _as_2d_strings(cand, n_img):
    """把 train.pt 的 'img' / 'label' 字段统一成 (n_img, k) 的字符串/整数数组。"""
    import torch

    if cand is None:
        return None
    if isinstance(cand, torch.Tensor):
        a = cand.numpy()
    else:
        a = np.asarray(cand)
    return a


def img_paths_of(obj, n_img):
    """train.pt 的 'img' 字段 → 每行一条图像相对路径（**列 0**；兼容 (n,k)/(n,) 两种排法）。"""
    a = _as_2d_strings(obj.get("img") if isinstance(obj, dict) else None, n_img)
    if a is None:
        return None
    if a.ndim >= 2:
        a = a[:, 0]
    flat = [str(x) for x in np.asarray(a).reshape(-1)]
    if len(flat) < n_img:
        return None
    return flat[:n_img]


def concept_of_path(p):
    """'train_images/00001_aardvark/aardvark_01b.jpg' → 0（概念号减一）。"""
    try:
        return int(p.split("/")[1][:5]) - 1
    except Exception:
        return -1


def resolve_image_index():
    """**自发现** Image_set 里的真实文件 ⇒ (概念目录, 文件名) → 绝对路径。

    ⚠️⚠️ **为什么要自发现（这是一个真坑，2026-10-27）**：
    `train.pt` 的 `img` 字段写的是 **`train_images/...`**，而 `Image_set` 下真实的 split 目录
    是 **`training_images/`**（探针实测顶层 = `.DS_Store` / `test_images` / `training_images`）
    ⇒ ⇒ 直接拼路径 ⇒ **100% 读不到** ⇒ 而旧版代码**静默用灰占位图顶替** ⇒
    ⇒ ⇒ **潜变量全是灰图的** ⇒ 训练看着「正常跑完」，其实全废
    ⇒ ⇒ ⇒ **两条硬规矩**：① 路径**不信任**上游字符串 ⇒ 自己 rglob 建索引；
    ② **缺图即报错**（不许占位图顶替）⇒ 让问题当场炸，而不是变成一场静默的假实验。

    ⇒ 索引键 = (概念目录名, 文件名) ⇒ 对前缀差异免疫。
    """
    idx = {}
    by_dir = {}
    for p in IMAGE_SET.rglob("*.jpg"):
        idx[(p.parent.name, p.name)] = p
        by_dir.setdefault(p.parent.name, []).append(p)
    for p in IMAGE_SET.rglob("*.png"):
        idx.setdefault((p.parent.name, p.name), p)
        by_dir.setdefault(p.parent.name, []).append(p)
    return idx, by_dir


def resolve_paths(paths, idx):
    """把 train.pt 的相对路径解析成**真实绝对路径**；解析不到 ⇒ 直接抛（不许占位）。"""
    out = []
    missing = []
    for rel in paths:
        parts = str(rel).split("/")
        cand = idx.get((parts[-2], parts[-1]))
        if cand is None:
            # 退路：只按文件名找（同一张图在别处）
            hits = [v for k, v in idx.items() if k[1] == parts[-1]]
            cand = hits[0] if len(hits) == 1 else None
        if cand is None:
            missing.append(str(rel))
        else:
            out.append(cand)
    if missing:
        raise RuntimeError(
            f"**{len(missing)}/{len(paths)} 张图解析不到真实文件**"
            f"（例：{missing[:3]}）⇒ 拒绝用占位图顶替"
            f"（旧版就是在这里静默把 8270 张全变成灰图 ⇒ 训练全废）")
    return out


def master_paths(subjects=TRAIN_SUBJECTS):
    """→ (paths list, source str)：**行号 = EEG 行号**的图像路径表。"""
    for sid in subjects:
        obj = load_train_pt(sid)
        n_img = int(obj["eeg"].shape[0])
        paths = img_paths_of(obj, n_img)
        if paths:
            return paths, f"train.pt sub-{sid:02d}.img"
    # 兜底：文件系统（⚠️ 此时**行内**的照片顺序未必与 EEG 行序一致 ⇒ 只算兜底）
    jpgs = sorted(p for p in IMAGE_SET.rglob("*.jpg") if "train" in str(p))
    rel = [str(p.relative_to(IMAGE_SET)) for p in jpgs]
    return rel, "文件系统（兜底：行内顺序未验证）"


def pick_reps(n_rep, k):
    """从 n_rep 个试次里**均匀取 k 个**（确定性 ⇒ 分片可复现）。"""
    k = max(1, min(int(k), int(n_rep)))
    if k == 1:
        return [0]
    return sorted(set(int(x) for x in np.linspace(0, n_rep - 1, k).round()))


def _shard_of(items, shard, n_shards):
    """按 round-robin 分片（负载均衡 ⇒ 每个分片的总量差不多）。"""
    return [it for i, it in enumerate(items) if i % int(n_shards) == int(shard)]


# ---------------------------------------------------------------------------
# 步骤 ①：EEG 条件
# ---------------------------------------------------------------------------
def eeg_step(args):
    import torch

    import exp93b_contrastive as E93B

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.encoder_ckpt, map_location="cpu", weights_only=False)
    model = E93B.build_model(n_ch=ck["n_ch"], n_times=ck["n_times"], out_dim=ck["out_dim"])
    model.load_state_dict(ck["state_dict"])
    model = model.to(device).eval()
    log(f"[编码器] exp93b encoder_best.pt（epoch={ck.get('epoch')}，**冻结**）")

    subjects = _shard_of(list(args.subjects), args.shard, args.n_shards)
    log(f"[分片] shard {args.shard}/{args.n_shards} ⇒ 被试 {subjects}")
    t0 = time.time()
    for sid in subjects:
        fp = cache_dir / f"eeg_cond_sub{sid:02d}.pt"
        if fp.exists() and not args.overwrite:
            log(f"  ✓ sub-{sid:02d} 已有缓存 ⇒ 跳过")
            continue
        obj = load_train_pt(sid)
        eeg = obj["eeg"]
        n_img, n_rep = int(eeg.shape[0]), int(eeg.shape[1])
        reps = pick_reps(n_rep, args.n_reps)
        conds = []
        for r in reps:
            rows = np.asarray(eeg[:, r])                    # (n_img, 63, 250) fp16
            z = E93B.encode_all(model, rows, device, args.batch).astype(np.float16)
            conds.append(z)
            del rows
        arr = np.stack(conds, axis=1)                       # (n_img, n_reps, 1024)
        torch.save({"cond": torch.from_numpy(arr), "reps": reps, "subject": sid,
                    "n_img": n_img, "n_rep_total": n_rep,
                    "encoder": "exp93b encoder_best.pt（冻结）"},
                   fp)
        log(f"  ✓ sub-{sid:02d}: {n_img} 图 × {len(reps)} 试次 {reps} "
            f"⇒ {tuple(arr.shape)}（{time.time()-t0:.0f}s）")
    return 0


# ---------------------------------------------------------------------------
# 步骤 ②：VAE 潜变量
# ---------------------------------------------------------------------------
def vae_step(args):
    import torch
    from PIL import Image

    import exp102_train as E102T

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    paths, source = master_paths(tuple(args.subjects))
    n_img = len(paths)
    log(f"[路径] {n_img} 张训练图（来源：{source}）")
    # 行内顺序自检：概念号必须是 i//10（与 exp93b 的 P1 预检同一条）
    cons = np.array([concept_of_path(p) for p in paths])
    ok_block = bool(np.array_equal(cons, np.arange(n_img) // 10))
    log(f"  行序自检：概念号 == i//10 ⇒ {ok_block}"
        f"（否则说明路径表与 EEG 行序对不上 ⇒ 停）")
    if not ok_block:
        raise RuntimeError("图像路径表与 EEG 行序对不上（概念块顺序不对）⇒ 不产出缓存")

    lo = n_img * args.shard // args.n_shards
    hi = n_img * (args.shard + 1) // args.n_shards
    sel = list(range(lo, hi))
    log(f"[分片] shard {args.shard}/{args.n_shards} ⇒ 行 [{lo}, {hi})（{len(sel)} 张）")

    vae = E102T.load_sd_component("vae", dtype=torch.float16, device=device).eval()
    try:
        vae.enable_slicing()      # 512×512 的 VAE encoder 很吃显存 ⇒ 切片更稳
    except Exception:                                          # noqa: BLE001
        pass
    scale = float(vae.config.scaling_factor)
    log(f"[VAE] SD1.5 自带 VAE｜scaling_factor={scale}｜{args.res}×{args.res}"
        f"｜batch={args.vae_batch}")

    # ---- **自发现**真实文件路径（⚠️ train.pt 的 'train_images/' 前缀 ≠ 真实的 'training_images/'）----
    idx, _by_dir = resolve_image_index()
    log(f"[路径] 自发现索引：{len(idx)} 个图片文件")
    abs_paths = resolve_paths(paths, idx)        # 解析不到 ⇒ 直接抛（**不许占位图**）
    log(f"[路径] ✓ 全部 {len(abs_paths)} 张都解析到真实文件（缺一张就停）")

    lat_all = []
    idx_all = []
    miss = 0
    t0 = time.time()

    # ---- 分段落盘（⚠️ 2026-10-27：平台会中途取消长 Job ⇒ 必须**中间落盘 + 可续跑**）----
    cache_dir = out_dir / "cache"
    part_files = sorted(cache_dir.glob(f"vae_latents_{args.shard:02d}_p*.pt"))
    done_idx = set()
    for pf in part_files:
        try:
            obj = torch.load(pf, map_location="cpu", weights_only=False)
            done_idx.update(int(i) for i in obj["idx"])
            log(f"  [续跑] 已有分片 {pf.name}：{len(obj['idx'])} 行")
        except Exception as e:                                   # noqa: BLE001
            log(f"  [续跑] 分片 {pf.name} 读不出（{type(e).__name__}）⇒ 忽略")
    if done_idx:
        log(f"  [续跑] 已完成 {len(done_idx)} 张 ⇒ 跳过它们")

    def flush_part(idx_chunk, lat_chunk, start_idx):
        fp = cache_dir / f"vae_latents_{args.shard:02d}_p{int(start_idx):06d}.pt"
        torch.save({"idx": list(idx_chunk), "latents": torch.cat(lat_chunk, 0),
                    "paths": [str(abs_paths[i]) for i in idx_chunk],
                    "resolution": args.res, "scaling_factor": scale}, fp)
        log(f"  ✓ 写分片 {fp.name}：{len(idx_chunk)} 行")

    part_k = len(part_files)
    for b in range(0, len(sel), args.vae_batch):
        chunk = [i for i in sel[b:b + args.vae_batch] if i not in done_idx]
        if not chunk:
            continue
        imgs = []
        for i in chunk:
            p = abs_paths[i]
            try:
                im = Image.open(p).convert("RGB")
            except Exception as e:                               # noqa: BLE001
                # ⚠️ **不许**用灰占位图顶替：旧版就是在这里静默把 8270 张全变成灰图
                raise RuntimeError(f"读图失败 {p}: {type(e).__name__}: {e}")
            im = im.resize((args.res, args.res), Image.BICUBIC)
            arr = np.asarray(im, dtype=np.float32) / 127.5 - 1.0
            imgs.append(torch.from_numpy(arr).permute(2, 0, 1))
        x = torch.stack(imgs).to(device=device, dtype=torch.float16)
        with torch.no_grad():
            # ⚠️ diffusers 0.42 里 `DiagonalGaussianDistribution.mode` 是**方法**
            #    （早期版本是属性）⇒ 写成「能调就调」⇒ 跨版本都稳。
            #    用 mode（均值）而不是 sample ⇒ **潜变量是确定的** ⇒ 同一张图永远同一个 latent。
            dist = vae.encode(x).latent_dist
            mode = dist.mode() if callable(dist.mode) else dist.mode
            lat = mode * scale
        lat_all.append(lat.cpu().half())
        idx_all.append(chunk)
        n_done = sum(len(c) for c in idx_all)
        if n_done % args.flush_every < args.vae_batch:
            # 按**起始行号**命名分片 ⇒ 续跑时不会与旧分片重名 ⇒ 不会产生重复行
            flush_part([i for c in idx_all for i in c], lat_all, idx_all[0][0])
            lat_all, idx_all = [], []
            part_k += 1
        if (b // args.vae_batch) % 20 == 0:
            log(f"  {n_done}/{len(sel)}｜{time.time()-t0:.0f}s")
    if idx_all:
        flush_part([i for c in idx_all for i in c], lat_all, idx_all[0][0])
    log(f"  ✓ 本分片完成｜缺图 {miss} 张｜{time.time()-t0:.0f}s")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["eeg", "vae"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--subjects", type=int, nargs="+", default=list(TRAIN_SUBJECTS))
    ap.add_argument("--n-reps", type=int, default=4,
                    help="每个被试每个概念缓存几个试次（§四十四·五：全部可用 rep；"
                         "受读盘时间限制，取均匀的 k 个）")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--vae-batch", type=int, default=4,
                    help="VAE 编码的批大小（512×512 很吃显存 ⇒ 别用 --batch 那个大值）")
    ap.add_argument("--flush-every", type=int, default=1500,
                    help="每编码多少张就落一个分片（平台会中途取消 Job ⇒ 必须中间落盘）")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--encoder-ckpt", default=ENCODER_CKPT)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.step == "eeg":
        return eeg_step(args)
    return vae_step(args)


if __name__ == "__main__":
    sys.exit(main())
