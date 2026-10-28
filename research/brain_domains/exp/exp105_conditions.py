#!/usr/bin/env python3
r"""exp105_conditions.py — exp105（**三个交叉对照**，§四十九）的**前置闸门 + 共用件**。

## 这一步在做什么

两个 step，都很便宜，**都在 GPU 上跑**（每个 < 30 min）：

1. `--step testcond`：算**测试侧**的三个条件口径，并直接算出**X1 的前置闸门**：
   - `rep0`：只编码第 0 个试次（= exp102 的条件口径）
   - `avg4`：编码缓存里那 4 个试次（rep 号**直接读缓存的 `reps` 字段** ⇒ 与训练侧必然一致）
     再在 **encoder 输出空间**求平均（§四十九·七·3）
   - `avg80`：全部 80 个试次的平均（**只报**：给出「SNR 能提到多高」的天花板量级）
   - 每个口径都算 **条件侧 2-way**（对 `gen2/true_clip.npy`，同一个 CLIP ⇒ 与 §四十八 同口径）
   ⇒ **闸门**：X1 的 `avg4` 条件侧 2-way 中位必须**显著高于** `rep0` 的 0.7650
     （不满足 ⇒ X1 记无效，**不是**「SNR 不重要」）。

2. `--step clipfeat`：算 **16540 张训练图的 CLIP 特征**（X3 的 oracle 条件）
   - **必须用生成侧那一个 CLIP**（`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`）
   - ⇒ 不许用 `/training/.../ViT-H-14_features_test.pt` 那套缓存特征
     （**特征空间不一致 ⇒ 2-way 静默失效** ⇒ §四十九·七·2）

## 复用（一条事实只写一处）

- 冻结 encoder 的装载口径 ⇒ 抄 `exp102_gen.py`（`build_model` + `load_state_dict`，兼容 `model`/`state_dict` 两种键）
- 条件编码 ⇒ `exp93b_contrastive.encode_all`
- CLIP 装载与编码 ⇒ `exp102_gen.load_clip` / `exp102_gen.clip_encode`（**同一个 CLIP、同一个预处理**）
- 真图路径自发现 ⇒ `exp102_gen.find_test_image_paths`
- 2-way / 候选集 ⇒ `exp102_gen.two_way_acc` / `build_candidate_structure`

## 用法

```powershell
python -u /domains/scripts/exp105_conditions.py --step testcond \
  --out-dir /domains/runs/illusionbrain-align/exp105
python -u /domains/scripts/exp105_conditions.py --step clipfeat \
  --out-dir /domains/runs/illusionbrain-align/exp105
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

EXP102_ROOT = Path("/domains/runs/illusionbrain-align/exp102")
EXP105_ROOT = Path("/domains/runs/illusionbrain-align/exp105")
TEST_4D = Path("/domains/omnibrain4d/eeg")
ENCODER_CKPT = "/domains/runs/illusionbrain-align/exp93b/encoder_best.pt"
TRUE_CLIP = EXP102_ROOT / "gen2" / "true_clip.npy"
TEST_SUBJECTS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
TRAIN_SUBJECTS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
N_IMG = 200


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 冻结 encoder（装载口径抄 exp102_gen.py）
# ---------------------------------------------------------------------------
def load_encoder(device):
    import torch

    import exp93b_contrastive as E93B

    ck = torch.load(ENCODER_CKPT, map_location="cpu", weights_only=False)
    model = E93B.build_model(n_ch=ck["n_ch"], n_times=ck["n_times"], out_dim=ck["out_dim"])
    model.load_state_dict(ck["model"] if "model" in ck else ck["state_dict"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log(f"[encoder] exp93b encoder_best.pt 已冻结加载（n_ch={ck['n_ch']}, "
        f"n_times={ck['n_times']}, out_dim={ck['out_dim']}）")
    return model


def encode_rows(model, rows, device, batch):
    """rows: (n, 63, 250) ⇒ (n, 1024) float32。"""
    import exp93b_contrastive as E93B

    z = E93B.encode_all(model, np.asarray(rows, dtype=np.float32), device, batch)
    return np.asarray(z, dtype=np.float32)


# ---------------------------------------------------------------------------
# 真图 CLIP（同一个 CLIP ⇒ 与生成侧同空间）
# ---------------------------------------------------------------------------
def get_true_clip(device, out_dir):
    """优先复用 exp102 的 `true_clip.npy`；没有就现算一遍（口径完全一致）。"""
    import exp102_gen as G

    if TRUE_CLIP.exists():
        arr = np.load(TRUE_CLIP).astype(np.float64)
        log(f"[真图 CLIP] 复用 {TRUE_CLIP} ⇒ {arr.shape}")
        return arr
    from PIL import Image

    log("[真图 CLIP] 没有 true_clip.npy ⇒ 现算（口径与 exp102_gen 完全一致）")
    model, proc = G.load_clip(device)
    paths = G.find_test_image_paths()
    if len(paths) != N_IMG:
        raise RuntimeError(f"test 图自发现得到 {len(paths)} 张（期望 {N_IMG}）⇒ 停")
    imgs = [Image.open(p).convert("RGB").resize((G.RES, G.RES), Image.BICUBIC) for p in paths]
    arr = G.clip_encode(model, proc, imgs, device)
    np.save(out_dir / "true_clip.npy", arr)
    return arr


# ---------------------------------------------------------------------------
# step ①：测试侧条件 + X1 的闸门
# ---------------------------------------------------------------------------
def step_testcond(args):
    import torch

    import exp102_gen as G

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    cond_dir = out_dir / "cond"
    cond_dir.mkdir(parents=True, exist_ok=True)

    model = load_encoder(device)
    true_clip = get_true_clip(device, out_dir)
    group, neg = G.build_candidate_structure(N_IMG)

    records = []
    for sid in TEST_SUBJECTS:
        # ---- rep 号：**直接读训练缓存里的 `reps` 字段** ⇒ 两边必然一致 ----
        cpath = EXP102_ROOT / "cache" / f"eeg_cond_sub{sid:02d}.pt"
        if not cpath.exists():
            log(f"[跳过] 缺训练条件缓存 {cpath} ⇒ 被试 {sid} 的 rep 号取不到")
            continue
        cobj = torch.load(cpath, map_location="cpu", weights_only=False)
        reps = [int(r) for r in cobj["reps"]]
        log(f"[sub-{sid:02d}] 缓存里的 rep 号 = {reps}")

        eeg = np.load(TEST_4D / f"sub-{sid:02d}.npz")["eeg"]
        n_rep = int(eeg.shape[1])
        log(f"[sub-{sid:02d}] test EEG {eeg.shape}（{n_rep} 个试次）")
        if max(reps) >= n_rep:
            raise RuntimeError(f"缓存的 rep 号 {reps} 超出 test EEG 的 {n_rep} 个试次 ⇒ 停")

        conds = {}
        # rep0（= exp102 的口径）
        conds["rep0"] = encode_rows(model, eeg[:N_IMG, 0], device, args.batch)
        # avg4（X1 的口径）：4 个 rep **在输出空间**求平均
        z4 = [encode_rows(model, eeg[:N_IMG, r], device, args.batch) for r in reps]
        conds["avg4"] = np.mean(np.stack(z4, axis=0), axis=0)
        # avg80（只报）：全部试次平均 ⇒ SNR 天花板
        if args.with_avg80:
            z80 = [encode_rows(model, eeg[:N_IMG, r], device, args.batch)
                   for r in range(n_rep)]
            conds["avg80"] = np.mean(np.stack(z80, axis=0), axis=0)

        rec = {"subject": f"sub-{sid:02d}", "reps": reps, "n_rep_test": n_rep,
               "metrics": {}}
        for k, v in conds.items():
            cz = G.l2norm(v.astype(np.float64))
            acc2 = G.two_way_acc(cz, true_clip[:len(cz)], neg[:len(cz)])
            sim = cz @ true_clip[:len(cz)].T
            acc200 = float((sim.argmax(axis=1) == np.arange(len(cz))).mean())
            rec["metrics"][k] = {"two_way": acc2, "acc200": acc200}
            log(f"  [sub-{sid:02d}] {k:>6}: 2-way={acc2:.4f}  200-way={acc200:.4f}")
            torch.save({"cond": torch.from_numpy(v.astype(np.float32)),
                        "kind": k, "reps": reps, "subject": sid},
                       cond_dir / f"test_cond_{k}_sub{sid:02d}.pt")
        records.append(rec)

    # ---- 闸门：X1 的 avg4 必须显著高于 rep0 ----
    import exp101_condition as E101

    gate = {"n_subjects": len(records)}
    for k in ("rep0", "avg4", "avg80"):
        vals = [r["metrics"][k]["two_way"] for r in records if k in r["metrics"]]
        if vals:
            gate[k] = {"two_way_median": E101.med_up(vals), "n": len(vals),
                       "per_subject": {r["subject"]: r["metrics"][k]["two_way"]
                                        for r in records if k in r["metrics"]}}
    if "avg4" in gate and "rep0" in gate:
        diffs = []
        for r in records:
            if "avg4" in r["metrics"] and "rep0" in r["metrics"]:
                diffs.append(r["metrics"]["avg4"]["two_way"] - r["metrics"]["rep0"]["two_way"])
        v = E101.paired_test(diffs, "avg4_minus_rep0", {})
        gate["x1_gate"] = {
            "diff": v,
            "pass": bool(v.get("n_eff", 0) >= 8 and v.get("p_sign", 1.0) <= 0.05
                         and (v.get("median") or 0) > 0),
            "rule": "X1 有效 ⇔ avg4 的条件侧 2-way 显著高于 rep0（中位 > 0，p ≤ 0.05，n ≥ 8）",
        }
        log(f"[闸门] X1：avg4 − rep0 ⇒ {json.dumps(gate['x1_gate']['diff'], ensure_ascii=False)}"
            f"｜通过={gate['x1_gate']['pass']}")
    gate["records"] = records
    (out_dir / "cond_gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"[闸门] 写入 {out_dir / 'cond_gate.json'}")
    return 0


# ---------------------------------------------------------------------------
# step ②：训练图 CLIP 特征（X3 的 oracle 条件）
# ---------------------------------------------------------------------------
def step_clipfeat(args):
    import torch
    from PIL import Image

    import exp102_cache as C
    import exp102_gen as G

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = cache_dir / "clip_train.pt"
    if fp.exists() and not args.overwrite:
        log(f"[跳过] 已有 {fp}")
        return 0

    # ---- 路径自发现（**不许占位图**；口径与 exp102_cache 完全一致）----
    paths, source = C.master_paths(TRAIN_SUBJECTS)
    n_img = len(paths)
    cons = np.array([C.concept_of_path(p) for p in paths])
    ok_block = bool(np.array_equal(cons, np.arange(n_img) // 10))
    log(f"[路径] {n_img} 张训练图（来源：{source}）｜行序自检（概念号 == i//10）⇒ {ok_block}")
    if not ok_block:
        raise RuntimeError("图像路径表与 EEG 行序对不上 ⇒ 不产出缓存")
    idx, _by_dir = C.resolve_image_index()
    abs_paths = C.resolve_paths(paths, idx)
    log(f"[路径] ✓ 全部 {len(abs_paths)} 张都解析到真实文件")

    model, proc = G.load_clip(device)
    feats = []
    t0 = time.time()
    for b in range(0, len(abs_paths), args.batch):
        chunk = abs_paths[b:b + args.batch]
        imgs = []
        for p in chunk:
            im = Image.open(p).convert("RGB").resize((G.RES, G.RES), Image.BICUBIC)
            imgs.append(im)
        feats.append(G.clip_encode(model, proc, imgs, device))
        if (b // args.batch) % 20 == 0:
            log(f"  {b + len(chunk)}/{len(abs_paths)}｜{time.time() - t0:.0f}s")
    arr = np.concatenate(feats, axis=0).astype(np.float32)
    if arr.shape != (n_img, 1024):
        raise RuntimeError(f"CLIP 特征形状 {arr.shape} ≠ ({n_img}, 1024) ⇒ 停")
    torch.save({"clip": torch.from_numpy(arr.astype(np.float16)),
                "paths": [str(p) for p in abs_paths],
                "clip_repo": G.CLIP_CANDIDATES[0]}, fp)
    log(f"[CLIP] ✓ {n_img} × 1024 已落盘 {fp}（{time.time() - t0:.0f}s）")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["testcond", "clipfeat"], required=True)
    ap.add_argument("--out-dir", default=str(EXP105_ROOT))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--with-avg80", action="store_true",
                    help="也算 80 试次平均（只报：SNR 天花板）；贵，默认不开")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.step == "testcond":
        return step_testcond(args)
    return step_clipfeat(args)


if __name__ == "__main__":
    sys.exit(main())
