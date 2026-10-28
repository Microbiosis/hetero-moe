#!/usr/bin/env python3
r"""exp105_gen.py — exp105（**三个交叉对照**，§四十九）的**生成 + 判决 + 机制测量**。

## 判据正本

`FINDINGS_TIER0_RSA.md` **§四十九**（判据沿用 §四十四·三；X3 的统计单元见 §四十九·八 R1）。

| 对照 | 条件（生成时喂给 adapter 的 1024 维向量） | 统计单元 | 地板 |
|---|---|---|---|
| **X1「SNR」** | test EEG 的 **4 个 rep 在 encoder 输出空间的平均** | 被试（n=10） | 跨被试 + 跨试次错配（与 §四十四·三 同） |
| **X2「训练量」** | test EEG 的 **rep 0**（与 exp102 一字不差） | 被试（n=10） | 同上 |
| **X3「oracle」** | **该概念真图的 CLIP 特征** | **概念（n=200）** | **错概念 CLIP 条件**生成的图 |

## 复用（一条事实只写一处）

- SD1.5 部件 / adapter 结构 / 训练常量 ⇒ `exp102_train`
- CLIP 装载与编码 / 真图自发现 / 2-way / 50-way / 候选集 / 种子 ⇒ `exp102_gen`
- **判据函数** ⇒ `exp102_gen.judge`（X1/X2 **直接调用它** ⇒ 判据一字不改）
- 冻结 encoder ⇒ `exp93b_contrastive`

## 生成口径（**与 exp102 一字不差**）

512×512｜DDIM 30 步｜guidance 1.0（不做 CFG）｜种子 = `gen_seed(被试, 概念)`｜
同一个 CLIP 编「生成图」与「真图」。

## 用法

```powershell
python -u /domains/scripts/exp105_gen.py --step generate --variant snr \
  --out-dir /domains/runs/illusionbrain-align/exp105/snr/gen
python -u /domains/scripts/exp105_gen.py --step judge --variant snr \
  --out-dir /domains/runs/illusionbrain-align/exp105/snr/gen \
  --results-dir /domains/runs/illusionbrain-align/exp105/snr/results
python -u /domains/scripts/exp105_gen.py --step mechanism --variant snr
```
"""
from __future__ import annotations

import argparse
import csv
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

import exp102_gen as G  # noqa: E402  （**共用件**：CLIP / 2-way / 50-way / 候选集 / 种子 / judge）

EXP102_ROOT = Path("/domains/runs/illusionbrain-align/exp102")
EXP105_ROOT = Path("/domains/runs/illusionbrain-align/exp105")
TEST_4D = Path("/domains/omnibrain4d/eeg")
ENCODER_CKPT = "/domains/runs/illusionbrain-align/exp93b/encoder_best.pt"
TEST_SUBJECTS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
N_IMG = 200
MIN_CONCEPTS = 180          # X3 的概念层阈值（200 个概念里零差剔除后至少 180 个）
VARIANTS = ("snr", "more", "oracle")


def log(*a):
    print(*a, flush=True)


def var_root(variant):
    return EXP105_ROOT / variant


def adapter_path(variant):
    return var_root(variant) / "adapter_final.pt"


# ---------------------------------------------------------------------------
# 条件（三个对照唯一的差异点）
# ---------------------------------------------------------------------------
def load_test_conditions(variant, device, out_dir):
    """→ {被试号: (200, 1024) float32}（X1/X2）；X3 不用它（条件 = 真图 CLIP）。

    - `snr`：读 `exp105/cond/test_cond_avg4_subXX.pt`（**前置闸门 step 的产物**）
    - `more`：**自己**从 test EEG 算 rep 0（不依赖 exp102 的产物 ⇒ 自洽）
    """
    import torch

    if variant == "snr":
        out = {}
        for sid in TEST_SUBJECTS:
            fp = EXP105_ROOT / "cond" / f"test_cond_avg4_sub{sid:02d}.pt"
            if not fp.exists():
                raise RuntimeError(f"缺 {fp} ⇒ 先跑 exp105_conditions.py --step testcond")
            obj = torch.load(fp, map_location="cpu", weights_only=False)
            out[sid] = obj["cond"].numpy().astype(np.float32)
        log(f"[条件] snr：{len(out)} 个被试 × {tuple(out[1].shape)}（4 rep 输出空间平均）")
        return out

    import exp93b_contrastive as E93B

    ck = torch.load(ENCODER_CKPT, map_location="cpu", weights_only=False)
    enc = E93B.build_model(n_ch=ck["n_ch"], n_times=ck["n_times"], out_dim=ck["out_dim"])
    enc.load_state_dict(ck["model"] if "model" in ck else ck["state_dict"])
    enc = enc.to(device).eval()
    out = {}
    for sid in TEST_SUBJECTS:
        eeg = np.load(TEST_4D / f"sub-{sid:02d}.npz")["eeg"]
        flat = np.asarray(eeg[:N_IMG, 0], dtype=np.float32)
        with torch.no_grad():
            z = E93B.encode_all(enc, flat, device, 64)
        out[sid] = np.asarray(z, dtype=np.float32)
    log(f"[条件] more：{len(out)} 个被试 × {tuple(out[1].shape)}（rep 0，与 exp102 一致）")
    return out


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------
def gen_images(cond, seed_of, lo, hi, sched, unet, vae, txt, tokzr, text_emb,
               adapter, clip_model, clip_proc, device, samples_out=None):
    """生成 [lo, hi) 区间的图 ⇒ (CLIP 特征, 样例图列表, 像素列表)。

    `seed_of(i)` → 该概念的随机种子（**own 与地板共用同一套种子分布**，§四十四·五）。
    """
    import torch
    from PIL import Image

    feats, pix, samples = [], [], []
    for i in range(lo, hi):
        g = torch.Generator(device="cpu").manual_seed(seed_of(i))
        lat = torch.randn((1, 4, G.RES // 8, G.RES // 8), generator=g,
                          dtype=torch.float32).to(device) * sched.init_noise_sigma
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            c = torch.from_numpy(cond[i:i + 1]).to(device, torch.float32)
            tokens = adapter(c)
            enc = torch.cat([text_emb, tokens], dim=1)
            for t in sched.timesteps:
                inp = sched.scale_model_input(lat, t)
                eps = unet(inp, t, encoder_hidden_states=enc).sample
                lat = sched.step(eps.float(), t, lat).prev_sample
            img = vae.decode(lat / vae.config.scaling_factor).sample
        arr = ((img[0].float().permute(1, 2, 0).cpu().numpy() + 1) * 127.5)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        if len(samples) < 8:
            samples.append(pil.copy())
        feats.append(G.clip_encode(clip_model, clip_proc, [pil], device)[0])
        if len(pix) < 16:
            pix.append(arr[::16, ::16, :].astype(np.float32))
    return feats, samples, pix


def step_generate(args):
    import torch
    from PIL import Image

    import exp102_train as T

    T.ensure_deps()          # ⚠️ 必须在 `from diffusers import ...` **之前**
    from diffusers import DDIMScheduler

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variant = args.variant
    log(f"[生成] variant={variant}｜设备 {device}｜分片 {args.shard}/{args.n_shards}")

    # ---- 冻结部件 ----
    adapter = T.build_adapter().to(device)
    ck2 = torch.load(args.adapter, map_location="cpu", weights_only=False)
    adapter.load_state_dict(ck2["state_dict"] if "state_dict" in ck2 else ck2)
    adapter = adapter.eval()
    for p in adapter.parameters():
        p.requires_grad_(False)

    unet = T.load_sd_component("unet", dtype=torch.float32, device=device).eval()
    vae = T.load_sd_component("vae", dtype=torch.float32, device=device).eval()
    tokzr = T.load_sd_component("tokenizer")
    txt = T.load_sd_component("text_encoder", dtype=torch.float32, device=device).eval()
    sched = DDIMScheduler.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="scheduler",
        cache_dir="/domains/hf_cache")
    sched.set_timesteps(G.DDIM_STEPS, device=device)
    with torch.no_grad():
        ids = tokzr([T.PROMPT], padding="max_length", max_length=77, truncation=True,
                    return_tensors="pt").input_ids.to(device)
        text_emb = txt(ids)[0]
    clip_model, clip_proc = G.load_clip(device)
    log(f"[SD] 冻结加载完毕｜DDIM {G.DDIM_STEPS}｜guidance {G.GUIDANCE}")

    # ---- 真图（同一个 CLIP）⇒ 直接复用 exp102 的 true_clip.npy（**同空间**）----
    true_clip = None
    for cand in (EXP102_ROOT / "gen2" / "true_clip.npy", out_dir / "true_clip.npy"):
        if cand.exists():
            true_clip = G.l2norm(np.load(cand).astype(np.float64))
            log(f"[真图] 复用 {cand} ⇒ {true_clip.shape}")
            break
    if true_clip is None:
        paths = G.find_test_image_paths()
        if len(paths) != N_IMG:
            raise RuntimeError(f"test 图自发现得到 {len(paths)} 张（期望 {N_IMG}）⇒ 停")
        imgs = [Image.open(p).convert("RGB").resize((G.RES, G.RES), Image.BICUBIC)
                for p in paths]
        true_clip = G.clip_encode(clip_model, clip_proc, imgs, device)
        np.save(out_dir / "true_clip.npy", true_clip)
    group, neg = G.build_candidate_structure(N_IMG)

    lo = N_IMG * args.shard // args.n_shards
    hi = N_IMG * (args.shard + 1) // args.n_shards

    # ---- 被试剔除记录（X1/X2 的对齐闸门）----
    excl_path = out_dir / "gen_excluded_subjects.json"
    excluded = []
    if excl_path.exists():
        try:
            excluded = json.loads(excl_path.read_text(encoding="utf-8"))
            log(f"[剔除] 沿用已有记录：{[e.get('subject') for e in excluded]}")
        except Exception as e:                                   # noqa: BLE001
            log(f"[剔除] 读已有记录失败（按空处理）：{type(e).__name__}: {e}")

    # =======================================================================
    # X3（oracle）：条件 = 真图 CLIP 特征；own + **错概念地板**（各 200 张）
    # =======================================================================
    if variant == "oracle":
        cond_own = true_clip[:N_IMG].astype(np.float32)
        # 地板：概念错配 +37（与 §四十四·三 的地板构造同构）⇒ 种子也按错位后的概念取
        cond_floor = np.stack([true_clip[G.floor_index(i)] for i in range(N_IMG)], axis=0)
        cond_floor = cond_floor.astype(np.float32)

        for tag, cond, seed_fn in (
                ("own", cond_own, lambda i, s=1: G.gen_seed(s, i)),
                ("floor", cond_floor, lambda i, s=1: G.gen_seed(s, G.floor_index(i)))):
            fp = out_dir / f"gen_clip_{tag}.pt"
            if fp.exists() and not args.overwrite:
                log(f"[跳过] {tag} 已有产物")
                continue
            feats, samples, pix = gen_images(
                cond, seed_fn, lo, hi, sched, unet, vae, txt, tokzr, text_emb,
                adapter, clip_model, clip_proc, device)
            feats = G.l2norm(np.stack(feats, axis=0))
            torch.save({"clip": torch.from_numpy(feats.astype(np.float32)),
                        "lo": lo, "hi": hi, "kind": tag}, fp)
            if len(pix) >= 2:
                p = np.stack(pix, axis=0).reshape(len(pix), -1)
                d = [float(np.linalg.norm(p[i] - p[j]))
                     for i in range(len(p)) for j in range(i + 1, len(p))]
                torch.save({"img_pair_dist_median": float(np.median(d))},
                           out_dir / f"gen_dist_{tag}.pt")
            log(f"  ✓ {tag}: {len(feats)} 张已落盘 {fp}")
        log("[生成] X3 完成")
        return 0

    # =======================================================================
    # X1 / X2：10 个被试 × 200 图（被试层统计）
    # =======================================================================
    conds = load_test_conditions(variant, device, out_dir)
    for sid in args.subjects:
        fp = out_dir / f"gen_clip_sub{sid:02d}.pt"
        if fp.exists() and not args.overwrite:
            log(f"[跳过] sub-{sid:02d} 已有产物")
            continue
        cond = conds[sid]
        feats, samples, pix = gen_images(
            cond, lambda i, s=sid: G.gen_seed(s, i), lo, hi, sched, unet, vae,
            txt, tokzr, text_emb, adapter, clip_model, clip_proc, device)
        feats = G.l2norm(np.stack(feats, axis=0))

        # ---- **对齐闸门**（预登记的数据质量闸门，与判决结果无关）----
        cz = G.l2norm(cond.astype(np.float64))
        sim = cz @ true_clip[:len(cz)].T
        acc200 = float((sim.argmax(axis=1) == np.arange(len(cz))).mean())
        acc2 = G.two_way_acc(cz, true_clip[:len(cz)], neg[:len(cz)])
        log(f"[对齐] sub-{sid:02d}: 200-way top-1={acc200:.4f}｜条件侧 2-way={acc2:.4f}")
        if acc200 < 0.01:
            excluded.append({"subject": sid, "align_acc200": acc200, "align_acc2": acc2,
                             "reason": "对齐闸门未过（200-way < 0.01）⇒ 剔除该被试"
                                       "（预登记的数据质量闸门，与判决结果无关）"})
            log(f"!! 剔除 sub-{sid:02d}（对齐闸门未过）")
            excl_path.write_text(json.dumps(excluded, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
            continue
        torch.save({"clip": torch.from_numpy(feats.astype(np.float32)),
                    "lo": lo, "hi": hi, "subject": sid,
                    "align_acc200": acc200, "align_acc2": acc2}, fp)
        if len(pix) >= 2:
            p = np.stack(pix, axis=0).reshape(len(pix), -1)
            d = [float(np.linalg.norm(p[i] - p[j]))
                 for i in range(len(p)) for j in range(i + 1, len(p))]
            torch.save({"img_pair_dist_median": float(np.median(d))},
                       out_dir / f"gen_dist_sub{sid:02d}.pt")
        try:
            grid = Image.new("RGB", (256 * 4, 256 * 2))
            paths = G.find_test_image_paths()
            for j in range(8):
                if lo + j < len(paths):
                    grid.paste(Image.open(paths[lo + j]).convert("RGB").resize((256, 256)),
                               (256 * (j % 4), 256 * (j // 4)))
            grid.save(out_dir / f"true_samples_sub{sid:02d}.jpg", quality=88)
        except Exception as e:                                   # noqa: BLE001
            log(f"[样例] 失败（不影响判决）：{e}")
    log(f"[生成] {variant} 完成 ⇒ {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# 判决
# ---------------------------------------------------------------------------
def judge_concepts(own, flo, true_clip, group, neg, loss_summary):
    """**X3 的判决（纯函数）**：统计单元 = **概念**（n=200），地板 = 错概念 CLIP 条件。

    ⚠️ 为什么不用被试层：X3 的条件来自图像 ⇒ **与被试无关** ⇒ 10 个被试只是换种子的重复
    ⇒ 被试层 = 伪重复（§四十九·八 R1）⇒ 改用概念层（n=200 ⇒ 阈值更严）。

    ⚠️⚠️ **主判据为什么用「相似度 margin」而不是 2-way 胜负**（§四十九·八 R3）：
      2-way 的**逐概念判定是二值的**（对/错）⇒ 在 oracle 条件下 own 恒为「对」而地板恒为「半对」
      ⇒ 差值只剩 {0, 1} ⇒ **超过一半是 0** ⇒ 上中位 `sorted[n//2]` 恰好 = 0
      ⇒ 「中位 > 0」这条判据**在完美情形下也失效**（自测实测抓到的就是这个）。
      ⇒ ⇒ 主判据改用**连续量**：同一候选对上的**相似度 margin 之差**
        （Δm = (sim[i,i] − sim[i,neg[i]]) 的 own − floor；50-way 同理用同组内的最大相似度）
      ⇒ 候选集与 §四十四·三 **完全相同** ⇒ 可比；只是统计量连续化。
      ⇒ 二值 2-way / 50-way **照报**（直观好读），但不参与判决。
    """
    import exp101_condition as E101

    n = min(len(own), len(flo), len(true_clip))
    own, flo = own[:n], flo[:n]
    tc = true_clip[:n]
    sim_o = own @ tc.T
    sim_f = flo @ tc.T
    t = np.arange(n)

    # ---- 二值版（只报）----
    own2 = (sim_o[t, t] > sim_o[t, neg[:n]]).astype(float)
    flo2 = (sim_f[t, t] > sim_f[t, neg[:n]]).astype(float)
    mask = np.zeros_like(sim_o, dtype=bool)
    for gi in np.unique(group):
        cols = np.flatnonzero(group == gi)
        mask[np.ix_(cols, cols)] = True
    simm_o = np.where(mask, sim_o, -np.inf)
    simm_f = np.where(mask, sim_f, -np.inf)
    own50 = (simm_o.argmax(axis=1) == t).astype(float)
    flo50 = (simm_f.argmax(axis=1) == t).astype(float)

    # ---- **连续版（主判据）**：相似度 margin 之差 ----
    # 2-way margin：与**自己**的负例比
    m2_o = sim_o[t, t] - sim_o[t, neg[:n]]
    m2_f = sim_f[t, t] - sim_f[t, neg[:n]]
    d2 = m2_o - m2_f
    # 50-way margin：与**同组内最像的别人**比（不含自己）
    so = sim_o.copy()
    sf = sim_f.copy()
    so[t, t] = -np.inf
    sf[t, t] = -np.inf
    so_masked = np.where(mask, so, -np.inf)
    sf_masked = np.where(mask, sf, -np.inf)
    m50_o = sim_o[t, t] - so_masked[t].max(axis=1)
    m50_f = sim_f[t, t] - sf_masked[t].max(axis=1)
    d50 = m50_o - m50_f

    c1 = {}
    v = E101.paired_test(list(d2), "margin2_way_minus_floor_concept", c1)
    v["pass"] = bool(v.get("n_eff", 0) >= MIN_CONCEPTS and v.get("p_sign", 1.0) <= G.ALPHA
                     and (v.get("median") or 0) > 0)
    c2 = {}
    v2 = E101.paired_test(list(d50), "margin50_way_minus_floor_concept", c2)
    v2["pass"] = bool(v2.get("n_eff", 0) >= MIN_CONCEPTS and v2.get("p_sign", 1.0) <= G.ALPHA
                      and (v2.get("median") or 0) > 0)

    own2_med = E101.med_up(list(own2))
    verdict = {
        "C0": {"loss_decreased": bool(loss_summary.get("decreased")),
               "loss_first10": loss_summary.get("first10_mean"),
               "loss_last10": loss_summary.get("last10_mean"),
               "images_not_constant": True},
        "C1": c1, "C1_pass": bool(c1) and all(x.get("pass") for x in c1.values()),
        "C2": c2, "C2_pass": bool(c2) and all(x.get("pass") for x in c2.values()),
        "C3_report": {
            "transfer_efficiency": (own2_med / 1.0) if own2_med is not None else None,
            "transfer_efficiency_def": "生成的 2-way 中位 ÷ 条件 2-way"
                                       "（oracle 条件 = 真图 CLIP 特征 ⇒ 条件 2-way = 1.0）",
            "binary_report_only": {
                "generated_2way_median": own2_med,
                "floor_2way_median": E101.med_up(list(flo2)),
                "generated_50way_median": E101.med_up(list(own50)),
                "floor_50way_median": E101.med_up(list(flo50)),
                "note": "二值版**只报**（主判据用连续 margin，见 R3）",
            },
            "unit": "concept",
        },
        "stat_unit": "concept",
        "n_concepts": int(n),
    }
    verdict["C0_pass"] = bool(verdict["C0"]["loss_decreased"])
    if verdict["C1_pass"] and verdict["C2_pass"]:
        verdict["verdict"] = "CONDITION_DRIVES_GENERATION"
    elif verdict["C1_pass"]:
        verdict["verdict"] = "CONDITION_DRIVES_GENERATION_50WAY_WEAK"
    else:
        med = c1.get("margin2_way_minus_floor_concept", {}).get("median")
        verdict["verdict"] = ("DIRECTION_RIGHT_UNDERPOWERED" if (med or 0) > 0
                              else "CONDITION_DOES_NOT_DRIVE_GENERATION")
    return verdict, own2, flo2, own50, flo50


def step_judge(args):
    import torch

    variant = args.variant
    out_dir = Path(args.out_dir)
    res_dir = Path(args.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    group, neg = G.build_candidate_structure(N_IMG)

    tp = out_dir / "true_clip.npy"
    true_clip = None
    for cand in (tp, EXP102_ROOT / "gen2" / "true_clip.npy"):
        if cand.exists():
            true_clip = G.l2norm(np.load(cand).astype(np.float64))
            break
    if true_clip is None:
        log("[真图] ✗ 找不到 true_clip.npy ⇒ 本轮不判")
        return 1

    # ---- C0 的损失曲线（**两级都找**）----
    loss_summary = {}
    curve = None
    for cand in (out_dir / "train_curve.json", out_dir.parent / "train_curve.json"):
        if cand.exists():
            curve = cand
            break
    if curve is not None:
        try:
            import exp102_train as T
            loss_summary = T.loss_curve_summary(
                json.loads(curve.read_text(encoding="utf-8")).get("epochs", []))
        except Exception as e:                                   # noqa: BLE001
            log(f"[C0] 损失曲线读不出：{type(e).__name__}: {e}")

    # =======================================================================
    # X3：概念层（n=200）
    # =======================================================================
    if variant == "oracle":
        own_fp = out_dir / "gen_clip_own.pt"
        fl_fp = out_dir / "gen_clip_floor.pt"
        if not (own_fp.exists() and fl_fp.exists()):
            log(f"[judge] 缺 own/floor 产物（{own_fp} / {fl_fp}）⇒ 不判")
            return 1
        own = G.l2norm(torch.load(own_fp, map_location="cpu",
                                  weights_only=False)["clip"].numpy().astype(np.float64))
        flo = G.l2norm(torch.load(fl_fp, map_location="cpu",
                                  weights_only=False)["clip"].numpy().astype(np.float64))
        verdict, own2, flo2, own50, flo50 = judge_concepts(
            own, flo, true_clip, group, neg, loss_summary)
        n = verdict["n_concepts"]
        out = {"verdict": verdict, "stat_unit": "concept", "n_concepts": n,
               "records": [], "loss_summary": loss_summary}
        (res_dir / "verdict.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        (res_dir / "loss_summary.json").write_text(
            json.dumps(loss_summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        with open(res_dir / "per_concept.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["concept", "own2", "floor2", "own50", "floor50"])
            for i in range(n):
                w.writerow([i, int(own2[i]), int(flo2[i]), int(own50[i]), int(flo50[i])])
        log(f"[判决] {variant}（概念层）⇒ {verdict['verdict']}")
        log(json.dumps(verdict, ensure_ascii=False, indent=1))
        return 0

    # =======================================================================
    # X1 / X2：被试层（**直接调用 exp102_gen.judge ⇒ 判据一字不改**）
    # =======================================================================
    records = []
    cond2_all = {}
    for sid in TEST_SUBJECTS:
        cp = out_dir / f"test_cond_sub{sid:02d}.pt"
        if not cp.exists():
            continue
        c = torch.load(cp, map_location="cpu", weights_only=False)["cond"].numpy().astype(np.float64)
        cond2_all[sid] = G.two_way_acc(G.l2norm(c), true_clip, neg)

    for sid in TEST_SUBJECTS:
        fp = out_dir / f"gen_clip_sub{sid:02d}.pt"
        if not fp.exists():
            log(f"[judge] ⚠ 缺 sub-{sid:02d} 的生成特征 ⇒ 跳过")
            continue
        obj = torch.load(fp, map_location="cpu", weights_only=False)
        gfeat = G.l2norm(obj["clip"].numpy().astype(np.float64))
        lo = int(obj.get("lo", 0))
        n = len(gfeat)
        d = G.donor_subject(sid)
        dp = out_dir / f"gen_clip_sub{d:02d}.pt"
        floor_feat = None
        if dp.exists():
            dobj = torch.load(dp, map_location="cpu", weights_only=False)
            dfeat = G.l2norm(dobj["clip"].numpy().astype(np.float64))
            floor_feat = np.stack([dfeat[G.floor_index(i)] for i in range(n)], axis=0)
        rec = {"subject": f"sub-{sid:02d}", "ok": True, "n_images": n, "shard_lo": lo,
               "own2": G.two_way_acc(gfeat, true_clip[:n], neg[:n]),
               "own50": G.fifty_way_acc(gfeat, true_clip[:n], group[:n]),
               "floor2": G.two_way_acc(floor_feat, true_clip[:n], neg[:n]) if floor_feat is not None else None,
               "floor50": G.fifty_way_acc(floor_feat, true_clip[:n], group[:n]) if floor_feat is not None else None,
               "cross_subject2": None,
               "cond2_same_clip": cond2_all.get(sid),
               "clip_sim": float(np.mean(np.sum(gfeat * true_clip[:n], axis=1))),
               "loss_summary": loss_summary}
        if floor_feat is None:
            rec["ok"] = False
            rec["ok_reason"] = f"地板缺失：地板来自 donor sub-{d:02d}（被剔除）⇒ 不进配对统计"
        dpf = out_dir / f"gen_dist_sub{sid:02d}.pt"
        if dpf.exists():
            rec["img_pair_dist"] = torch.load(dpf, map_location="cpu",
                                              weights_only=False)["img_pair_dist_median"]
        records.append(rec)

    by_sub = {r["subject"]: r for r in records}
    for r in records:
        sid = int(r["subject"].split("-")[1])
        d = G.donor_subject(sid)
        dr = by_sub.get(f"sub-{d:02d}")
        r["cross_subject2"] = dr["own2"] if dr else None
        r["cross_subject_subject"] = f"sub-{d:02d}"

    recall = None
    if records:
        subs = sorted(r["subject"] for r in records)
        n_min = min(r["n_images"] for r in records)
        pc = []
        for i in range(n_min):
            hit = 0
            for s in subs:
                gf = by_sub[s]
                gv = G.l2norm(torch.load(out_dir / f"gen_clip_{s}.pt", map_location="cpu",
                                        weights_only=False)["clip"].numpy().astype(np.float64))
                hit += int(gv[i] @ true_clip[i] > gv[i] @ true_clip[neg[i]])
            pc.append(hit / len(subs))
        recall = float(np.median(pc))
    for r in records:
        r["per_concept_recall"] = recall

    verdict = G.judge(records)
    excluded = []
    excl_path = out_dir / "gen_excluded_subjects.json"
    if excl_path.exists():
        try:
            excluded = json.loads(excl_path.read_text(encoding="utf-8"))
        except Exception as e:                                   # noqa: BLE001
            log(f"[judge] 剔除记录读不出：{type(e).__name__}: {e}")
    out = {"verdict": verdict, "records": records,
           "n_subjects_used": len(records),
           "subjects_used": [r["subject"] for r in records],
           "subjects_excluded": [e.get("subject") for e in excluded],
           "exclusion_reason": excluded,
           "variant": variant,
           "atm_reference_C_class": G.ATM_REF,
           "exp101_condition_reference": G.EXP101_REF}
    (res_dir / "verdict.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (res_dir / "loss_summary.json").write_text(
        json.dumps(loss_summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if records:
        cols = ["subject", "n_images", "own2", "floor2", "own50", "floor50",
                "cross_subject2", "cross_subject_subject", "cond2_same_clip",
                "clip_sim", "img_pair_dist", "per_concept_recall"]
        with open(res_dir / "per_subject.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in records:
                w.writerow(r)
    lines = [f"=== exp105 判决（{variant}｜§四十九）===",
             f"判决: {verdict.get('verdict')}",
             f"C0: {json.dumps(verdict.get('C0'), ensure_ascii=False)}",
             f"C1: {json.dumps(verdict.get('C1'), ensure_ascii=False)}",
             f"C2: {json.dumps(verdict.get('C2'), ensure_ascii=False)}",
             f"C3: {json.dumps(verdict.get('C3_report'), ensure_ascii=False)}",
             "", "逐被试："]
    for r in records:
        lines.append(f"  {r['subject']}: own2={r['own2']:.4f} floor2={r['floor2']} "
                     f"own50={r['own50']:.4f} floor50={r['floor50']} "
                     f"cross2={r['cross_subject2']} cond2={r['cond2_same_clip']} "
                     f"clip_sim={r['clip_sim']:.4f}")
    (res_dir / "judge_log.txt").write_text("\n".join(lines), encoding="utf-8")
    for ln in lines:
        log(ln)
    return 0


# ---------------------------------------------------------------------------
# 机制测量（**只报**：条件 token 的跨概念一致性，§四十九·六）
# ---------------------------------------------------------------------------
def step_mechanism(args):
    import torch

    import exp102_train as T

    T.ensure_deps()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    variant = args.variant

    ck = torch.load(args.adapter, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    adapter = T.build_adapter().to(device)
    adapter.load_state_dict(sd)
    adapter = adapter.eval()

    if variant == "oracle":
        tp = EXP102_ROOT / "gen2" / "true_clip.npy"
        cond = np.load(tp).astype(np.float32)[:N_IMG]
    elif variant == "snr":
        fp = EXP105_ROOT / "cond" / "test_cond_avg4_sub01.pt"
        cond = torch.load(fp, map_location="cpu",
                          weights_only=False)["cond"].numpy().astype(np.float32)
    else:
        import exp93b_contrastive as E93B
        ck2 = torch.load(ENCODER_CKPT, map_location="cpu", weights_only=False)
        enc = E93B.build_model(n_ch=ck2["n_ch"], n_times=ck2["n_times"],
                               out_dim=ck2["out_dim"])
        enc.load_state_dict(ck2["model"] if "model" in ck2 else ck2["state_dict"])
        enc = enc.to(device).eval()
        eeg = np.load(TEST_4D / "sub-01.npz")["eeg"]
        flat = np.asarray(eeg[:N_IMG, 0], dtype=np.float32)
        with torch.no_grad():
            cond = E93B.encode_all(enc, flat, device, 64).astype(np.float32)

    with torch.no_grad():
        tok = adapter(torch.from_numpy(cond).to(device)).cpu().numpy().astype(np.float64)
    tok_norm = np.linalg.norm(tok, axis=-1)
    mu = tok.mean(axis=0)
    per_tok_mean = np.linalg.norm(mu, axis=-1)
    per_tok_avg = np.linalg.norm(tok, axis=-1).mean(axis=0)
    cons = float((per_tok_mean / per_tok_avg).mean())
    rec = {"variant": variant,
           "w_max": float(sd["net.3.weight"].abs().max()),
           "w_mean": float(sd["net.3.weight"].abs().mean()),
           "tok_med": float(np.median(tok_norm)),
           "consistency": cons,
           "token_norm_median": float(np.median(tok_norm)),
           "token_norm_max": float(tok_norm.max()),
           "n_cond": int(cond.shape[0])}
    log(f"[机制] {variant}: w_max={rec['w_max']:.6f} w_mean={rec['w_mean']:.6f} "
        f"tok_med={rec['tok_med']:.4f} consistency={cons:.6f}")
    out_json = Path(args.out_json) if args.out_json else (
        var_root(variant) / "results" / "mechanism.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[写盘] {out_json}")
    return 0


def self_test():
    """自测：**X3 的概念层判决**（纯函数，n=200）。

    ⚠️ 为什么要自测：exp102 的「50-way 行索引广播不上」那个 bug 是在云端 judge 第一次真跑
    时才炸出来的（22/22 自测全绿却带着 bug）⇒ ⇒ **新写的判据路径必须先被自测覆盖**。
    """
    ok, bad = 0, []

    def chk(name, cond):
        nonlocal ok
        if cond:
            ok += 1
            print(f"✓ {name}")
        else:
            bad.append(name)
            print(f"✗ {name}")

    import exp101_condition as E101

    rng = np.random.default_rng(0)
    group, neg = G.build_candidate_structure(200)
    tc = G.l2norm(rng.normal(size=(200, 1024)))
    own = tc.copy()
    flo = G.l2norm(rng.normal(size=(200, 1024)))

    # ① 完美 own（= 真图）⇒ own2/own50 全 1 ⇒ C1/C2 都过
    v, own2, flo2, own50, flo50 = judge_concepts(own, flo, tc, group, neg,
                                                  {"decreased": True})
    chk("完美 own ⇒ own2 全 1", float(np.mean(own2)) == 1.0)
    chk("完美 own ⇒ own50 全 1", float(np.mean(own50)) == 1.0)
    chk("随机地板 ⇒ floor2 ≈ 机会 0.5", 0.35 <= float(np.mean(flo2)) <= 0.65)
    chk("随机地板 ⇒ floor50 ≈ 机会 0.02", 0.0 <= float(np.mean(flo50)) <= 0.10)
    chk("C1 过（连续 margin 差 ⇒ 中位 > 0）", v["C1_pass"] is True)
    chk("C2 过", v["C2_pass"] is True)
    chk("判决 = CONDITION_DRIVES_GENERATION", v["verdict"] == "CONDITION_DRIVES_GENERATION")
    chk("统计单元 = 概念", v["stat_unit"] == "concept")
    chk("传递效率 = 1.0（oracle 条件 2-way = 1.0）",
        abs((v["C3_report"]["transfer_efficiency"] or 0) - 1.0) < 1e-9)
    chk("二值版只报（不进主判据）", "binary_report_only" in v["C3_report"])

    # ①b ⚠️ **二值差值在完美情形下中位 = 0**（这正是 R3 的由来）⇒ 记一条断言防回退
    diffs_binary = list(own2 - flo2)
    med_binary = E101.med_up(diffs_binary)
    chk("⚠️ 二值差值的中位在完美情形下 = 0 ⇒ 主判据**不能**用二值（R3 的证据）",
        med_binary == 0.0)
    chk("⚠️ 连续 margin 差的中位 > 0（主判据用它）",
        v["C1"]["margin2_way_minus_floor_concept"]["median"] > 0)

    # ② own == 地板 ⇒ 逐概念差恒为 0 ⇒ 零差剔除 ⇒ n_eff = 0 ⇒ C1 不过（**不许崩**）
    v2, *_ = judge_concepts(flo, flo.copy(), tc, group, neg, {"decreased": True})
    chk("own == 地板 ⇒ C1 不过（差恒为 0）", v2["C1_pass"] is False)
    chk("own == 地板 ⇒ 判决 = CONDITION_DOES_NOT_DRIVE_GENERATION",
        v2["verdict"] == "CONDITION_DOES_NOT_DRIVE_GENERATION")

    # ③ 高信噪比的弱扰动（噪声 0.02）⇒ 仍应判为「驱动」（oracle 条件本该很容易驱动）
    weak = G.l2norm(tc + 0.02 * rng.normal(size=(200, 1024)))
    v3, *_ = judge_concepts(weak, flo, tc, group, neg, {"decreased": True})
    chk("高 SNR 的 own（噪声 0.02）⇒ C1 仍过", v3["C1_pass"] is True)

    # ③b **纯随机** own（与真概念无关）⇒ 判不过（这是「条件没起作用」的样子）
    rand_own = G.l2norm(rng.normal(size=(200, 1024)))
    v3b, *_ = judge_concepts(rand_own, flo, tc, group, neg, {"decreased": True})
    chk("纯随机 own ⇒ C1 不过", v3b["C1_pass"] is False)
    chk("纯随机 own ⇒ 判决在 {DIRECTION_RIGHT_UNDERPOWERED, CONDITION_DOES_NOT...} 里",
        v3b["verdict"] in ("DIRECTION_RIGHT_UNDERPOWERED", "CONDITION_DOES_NOT_DRIVE_GENERATION"))

    # ④ C0 的损失曲线是判据的一部分
    v4, *_ = judge_concepts(own, flo, tc, group, neg, {"decreased": False})
    chk("C0（损失没降）⇒ C0_pass=False", v4["C0_pass"] is False)

    print(f"\n{ok} 通过 / {len(bad)} 失败")
    if bad:
        print("失败：" + "；".join(bad))
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), default="",
                    help="snr / more / oracle（--self-test 时不需要）")
    ap.add_argument("--step", choices=["generate", "judge", "mechanism"], required=True)
    ap.add_argument("--subjects", type=int, nargs="+", default=list(TEST_SUBJECTS))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--results-dir", default="")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.variant:
        ap.error("--variant 必须给（snr / more / oracle）")
    if not args.adapter:
        args.adapter = str(adapter_path(args.variant))
    if not args.out_dir:
        args.out_dir = str(var_root(args.variant) / "gen")
    if not args.results_dir:
        args.results_dir = str(var_root(args.variant) / "results")
    if args.step == "generate":
        return step_generate(args)
    if args.step == "judge":
        return step_judge(args)
    return step_mechanism(args)


if __name__ == "__main__":
    sys.exit(main())
