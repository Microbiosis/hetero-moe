#!/usr/bin/env python3
r"""exp102_gen.py — exp102（②③ 脑→图像条件生成）的**判据脚本**：生成 + 评测 + 判决。

## 判据正本

`FINDINGS_TIER0_RSA.md` **§四十四**（§四十四·三 的表格 = 判据，§四十四·五 = 实现约定，
§四十四·六 = 出结果前的修订留痕）。**本脚本实现它，一字不改**。

| # | 判据 | 阈值 |
|---|---|---|
| C0 | 管线自检：训练损失下降（末 10% epoch 均值 < 首 10%）**且** 生成的图不是常数（图间成对距离中位 > 0）| 不过 ⇒ 全实验作废 |
| C1（主判据）| 逐被试 2-way（生成图↔真图 vs 生成图↔另一个候选）显著高于**地板**（**跨被试 + 跨试次**错配）| ≥8 被试、p ≤ 0.05、中位 > 0（配对符号检验）|
| C1b（只报）| 被试特异性：地板换成「另一个被试、**同一概念**的 EEG」（C1 原口径）| 只报 |
| C2 | 50-way top-1 显著高于地板（地板口径同 C1）| ≥8 被试、p ≤ 0.05、中位 > 0 |
| C3（只报）| ① 传递效率 ② CLIP 相似度 ③ ATM 并排（C 类参照）④ 每概念召回 | — |

## 三个关键设计（都在 §四十四·五 里定死了）

1. **地板生成零额外算力**：被试 s 的目标 i 的地板图 = `own(s+1, (i+37)%200)`
   （donor 被试、另一个概念）⇒ 它**就是**被试 s+1 自己那批生成图 ⇒ **不用多生成一张**。
2. **不做 CFG**（guidance scale = 1.0）⇒ 不需要 null 条件分支 ⇒ 算力减半。
3. **同一个 CLIP 同时编「生成图」与「真图」** ⇒ 两边同空间；条件侧 2-way 也用同一个
   CLIP 重算 ⇒ **传递效率自洽**；exp101 的已发表数字只作**并列参照**。

## 三个 step

- `--step generate`（GPU，可分片）：test EEG → 冻结 encoder → 条件 → SD1.5 出图 →
  CLIP 编码 → 落盘（生成特征 + test 条件 + 真图特征 + 样例 PNG + C0 的图像间距离）。
- `--step judge`（CPU）：读各被试产物 ⇒ **纯函数**判决 ⇒ 逐被试原始记录 + judge_log。
- `--self-test`：合成数据覆盖「通过 / 不通过 / 方向对但检验力不足 / 中位 ≤ 0」四种情形。

## 用法

```powershell
# 本地（自测）
python research\brain_domains\exp\exp102_gen.py --self-test

# 云端
python -u /domains/scripts/exp102_gen.py --step generate --subjects 1 2 3 4 5 \
  -v hf://buckets/illusion-brain/training-data:/training:ro \
  -v hf://buckets/illusion-brain/brain-domains:/domains --flavor a10g-large --timeout 60m
python -u /domains/scripts/exp102_gen.py --step judge --out-dir /domains/runs/.../exp102
```
"""
from __future__ import annotations

import argparse
import csv
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

import exp101_condition as E101  # noqa: E402  （med_up / sign_test_nonzero / 候选集）

# ===========================================================================
# §四十四·五 定死的口径
# ===========================================================================
TEST_SUBJECTS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
TEST_4D = Path("/domains/omnibrain4d/eeg")
IMAGE_SET = Path("/training/eeg/things-eeg/Image_set")
FEAT_TEST = "/training/eeg/eeg-image-decode/ViT-H-14_features_test.pt"
ENCODER_CKPT = "/domains/runs/illusionbrain-align/exp93b/encoder_best.pt"
DEFAULT_OUT = Path("/domains/runs/illusionbrain-align/exp102")

RES = 512                 # §四十四·五：512×512
DDIM_STEPS = 30           # §四十四·五：DDIM 30 步
GUIDANCE = 1.0            # §四十四·五：guidance scale = 1.0（不做 CFG）
SEED_BASE = 20261027      # 生成种子基准（§四十四·五：种子由 (donor 被试, 概念号) 决定）

# 地板：donor 被试 = s+1（环），概念偏移 = +37
FLOOR_DONOR_OFFSET = 1
FLOOR_CONCEPT_OFFSET = 37

MIN_SUBJECTS = 8          # §四十四·三
ALPHA = 0.05              # §四十四·三
CHANCE_2 = 0.5
CHANCE_50 = 1.0 / 50.0

# C 类参照（别人发表的系统 ⇒ 只作量级参照，不是同协议对标）
ATM_REF = {"two_way_pct": 83.1, "fifty_way_top1_pct": 28.4, "fid": 24.6}
# exp101（我们的条件侧）已发表数字 ⇒ 并列参照
EXP101_REF = {"two_way_pct_single_trial": 78.09, "two_way_pct_trial_avg": 92.00,
              "fifty_way_pct_single_trial": 14.83, "fifty_way_pct_trial_avg": 42.00,
              "two_hundred_way_pct_single_trial": 5.89}

# CLIP（**必须 1024 维** ⇒ 冻结 encoder 的输出维度才接得上）
CLIP_CANDIDATES = ("laion/CLIP-ViT-H-14-laion2B-s32B-b79K",)


def log(*a):
    print(*a, flush=True)


# ===========================================================================
# 工具
# ===========================================================================
def l2norm(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-12)


def gen_seed(donor_subject, concept):
    """§四十四·五：种子由 (donor 被试, 概念号) 决定 ⇒ own 与地板共用同一套种子分布。"""
    return int((SEED_BASE + int(donor_subject) * 100003 + int(concept) * 7919) % (2 ** 31 - 1))


def floor_index(concept, n_img=200):
    """地板图在 donor 那批里的行号（跨试次错配：概念 ≠ 当前目标）。"""
    return int((int(concept) + FLOOR_CONCEPT_OFFSET) % n_img)


def donor_subject(subject, n_subjects=len(TEST_SUBJECTS)):
    """C1b / 地板用的 donor 被试（环）。"""
    i = TEST_SUBJECTS.index(subject)
    return TEST_SUBJECTS[(i + FLOOR_DONOR_OFFSET) % n_subjects]


def two_way_acc(gen_feat, true_feat, neg):
    """逐行 2-way：生成图 ↔ 真图 vs 生成图 ↔ 负例。gen_feat/true_feat 已 L2 归一。"""
    sim = gen_feat @ true_feat.T
    t = np.arange(len(gen_feat))
    return float((sim[t, t] > sim[t, neg]).mean())


def fifty_way_acc(gen_feat, true_feat, group):
    """50-way top-1：只在**同组**的 50 个候选里比。"""
    sim = gen_feat @ true_feat.T
    mask = np.zeros_like(sim, dtype=bool)
    for gi in np.unique(group):
        cols = np.flatnonzero(group == gi)
        rows = np.flatnonzero(group == np.arange(len(group))[group == gi])
        mask[np.ix_(rows, cols)] = True
    simm = np.where(mask, sim, -np.inf)
    return float((simm.argmax(axis=1) == np.arange(len(sim))).mean())


def build_candidate_structure(n_img=200):
    """沿用 exp101 的候选集（同一 seed ⇒ 可比）。"""
    group, neg = E101.make_candidate_structure(n_img, n_groups=4, seed=0)
    return np.asarray(group), np.asarray(neg)


# ===========================================================================
# 判决（**纯函数**：输入逐被试原始记录 ⇒ 输出判决）
# ===========================================================================
def judge(records, min_subjects=MIN_SUBJECTS, alpha=ALPHA):
    """records: [{subject, own2, floor2, own50, floor50, cross_subject2, clip_sim,
                  img_pair_dist, cond2_same_clip}, ...]"""
    out = {}
    ok = [r for r in records if r.get("ok", True)]
    if not ok:
        return {"verdict": "INSUFFICIENT", "why": "没有可用被试"}

    # ---- C0 管线自检 ----
    loss = ok[0].get("loss_summary") or {}
    dists = [r.get("img_pair_dist") for r in ok if r.get("img_pair_dist") is not None]
    med_dist = E101.med_up(dists) if dists else float("nan")
    c0 = {"loss_decreased": bool(loss.get("decreased")),
          "loss_first10": loss.get("first10_mean"),
          "loss_last10": loss.get("last10_mean"),
          "img_pair_dist_median": med_dist,
          "images_not_constant": bool(dists and med_dist > 0)}
    out["C0"] = c0
    out["C0_pass"] = bool(c0["loss_decreased"] and c0["images_not_constant"])
    if not out["C0_pass"]:
        out["verdict"] = "C0_FAIL"
        out["why"] = ("训练损失没降" if not c0["loss_decreased"] else "") + \
                     ("生成的图是常数" if not c0["images_not_constant"] else "")
        return out

    # ---- C1（主判据）：own2 − floor2 ----
    c1 = {}
    diffs = [r["own2"] - r["floor2"] for r in ok if "own2" in r and "floor2" in r]
    if diffs:
        v = E101.paired_test(diffs, "two_way_minus_floor", c1)
        v["pass"] = bool(v["n_eff"] >= min_subjects and v["p_sign"] <= alpha
                         and (v.get("median") or 0) > 0)
    out["C1"] = c1
    out["C1_pass"] = bool(c1) and all(x.get("pass") for x in c1.values())

    # ---- C1b（只报）：被试特异性 = own2(s) − own2(donor 同概念) ----
    c1b = {}
    diffs_b = [r["own2"] - r["cross_subject2"] for r in ok
               if "own2" in r and "cross_subject2" in r]
    if diffs_b:
        E101.paired_test(diffs_b, "two_way_minus_cross_subject", c1b)
    out["C1b_report"] = c1b
    out["C1b_mean_diff"] = float(np.mean(diffs_b)) if diffs_b else None

    # ---- C2：50-way ----
    c2 = {}
    diffs2 = [r["own50"] - r["floor50"] for r in ok if "own50" in r and "floor50" in r]
    if diffs2:
        v = E101.paired_test(diffs2, "fifty_way_minus_floor", c2)
        v["pass"] = bool(v["n_eff"] >= min_subjects and v["p_sign"] <= alpha
                         and (v.get("median") or 0) > 0)
    out["C2"] = c2
    out["C2_pass"] = bool(c2) and all(x.get("pass") for x in c2.values())

    # ---- C3（只报）----
    own2_med = E101.med_up([r["own2"] for r in ok if "own2" in r])
    floor2_med = E101.med_up([r["floor2"] for r in ok if "floor2" in r])
    cond2_med = E101.med_up([r["cond2_same_clip"] for r in ok
                             if r.get("cond2_same_clip") is not None])
    eff = (own2_med / cond2_med) if (cond2_med and cond2_med > 0) else None
    out["C3_report"] = {
        "transfer_efficiency": eff,
        "transfer_efficiency_def": "生成的 2-way 中位 ÷ 条件的 2-way 中位（**同一个 CLIP** 下的口径）",
        "generated_2way_median": own2_med,
        "condition_2way_median_same_clip": cond2_med,
        "floor_2way_median": floor2_med,
        "clip_similarity_mean": float(np.mean([r["clip_sim"] for r in ok if "clip_sim" in r]))
        if any("clip_sim" in r for r in ok) else None,
        "per_concept_recall_median": float(np.median(
            [r["per_concept_recall"] for r in ok if r.get("per_concept_recall") is not None]))
        if any(r.get("per_concept_recall") is not None for r in ok) else None,
        "atm_reference_C_class": ATM_REF,
        "exp101_condition_reference": EXP101_REF,
    }

    # ---- 总判决 ----
    if out["C1_pass"] and out["C2_pass"]:
        out["verdict"] = "CONDITION_DRIVES_GENERATION"
    elif out["C1_pass"]:
        out["verdict"] = "CONDITION_DRIVES_GENERATION_50WAY_WEAK"
    else:
        med = c1.get("two_way_minus_floor", {}).get("median")
        if med is not None and med > 0:
            out["verdict"] = "DIRECTION_RIGHT_UNDERPOWERED"
        else:
            out["verdict"] = "CONDITION_DOES_NOT_DRIVE_GENERATION"
        out["interpretation"] = (
            "生成 2-way > 地板但不显著 ⇒ 方向对、检验力不足" if (med or 0) > 0
            else "生成 2-way ≤ 地板 ⇒ 条件驱动不了生成")
    return out


# ===========================================================================
# 生成 step
# ===========================================================================
def find_test_image_paths():
    """发现 test split 的 200 张图（**自发现 schema，不写死路径**）。

    ⇒ 目录结构：`Image_set/test_images/<概念号_概念名>/<图名>.jpg`，每个概念目录
      恰好 1 张 ⇒ 按概念号排序即与特征行序一致（与 exp93b 的行号约定一致）。
    """
    root = IMAGE_SET / "test_images"
    if not root.exists():
        root = IMAGE_SET
    per_concept = {}
    for p in sorted(root.rglob("*.jpg")) + sorted(root.rglob("*.png")):
        try:
            con = int(p.parent.name.split("_")[0])
        except Exception:
            continue
        per_concept.setdefault(con, []).append(p)
    paths = []
    for con in sorted(per_concept):
        paths.append(sorted(per_concept[con])[0])
    return paths


def load_clip(device):
    """加载 ViT-H-14（**1024 维**）。失败就报错，不静默换模型。"""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    import exp102_train as T
    T.ensure_deps()
    last = None
    for repo in CLIP_CANDIDATES:
        try:
            model = CLIPModel.from_pretrained(repo, cache_dir="/domains/hf_cache")
            proc = CLIPProcessor.from_pretrained(repo, cache_dir="/domains/hf_cache")
            model = model.to(device).eval()
            log(f"[CLIP] ✓ {repo}")
            return model, proc
        except Exception as e:                                   # noqa: BLE001
            last = e
            log(f"[CLIP] ⚠ {repo} 不可用：{type(e).__name__}: {str(e)[:140]}")
    raise RuntimeError(f"CLIP 取不到（候选全失败）：{last}")


def clip_encode(model, proc, images, device, batch=16):
    import torch

    feats = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            chunk = images[i:i + batch]
            inp = proc(images=chunk, return_tensors="pt").to(device)
            out = model.get_image_features(**inp)
            feats.append(out.float().cpu().numpy())
    return l2norm(np.concatenate(feats, axis=0).astype(np.float64))


def step_generate(args):
    import torch

    import exp102_train as T
    T.ensure_deps()          # ⚠️ 必须在 `from diffusers import ...` **之前**（镜像里没有 diffusion 全家桶）
    from diffusers import DDIMScheduler
    from PIL import Image

    import exp93b_contrastive as E93B

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"[生成] 设备 {device}｜分片 {args.shard}/{args.n_shards}｜被试 {args.subjects}")

    # ---- 冻结的脑侧 encoder（exp93b）----
    ck = torch.load(args.encoder_ckpt, map_location="cpu", weights_only=False)
    encoder = E93B.build_model(n_ch=ck["n_ch"], n_times=ck["n_times"], out_dim=ck["out_dim"])
    encoder.load_state_dict(ck["model"] if "model" in ck else ck["state_dict"])
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    log("[encoder] exp93b encoder_best.pt 已冻结加载")

    # ---- adapter ----
    adapter = T.build_adapter().to(device)
    ck2 = torch.load(args.adapter, map_location="cpu", weights_only=False)
    adapter.load_state_dict(ck2["state_dict"] if "state_dict" in ck2 else ck2)
    adapter = adapter.eval()
    for p in adapter.parameters():
        p.requires_grad_(False)
    log(f"[adapter] {args.adapter} 已加载")

    # ---- SD1.5（**权重用 fp32** + autocast bf16 ⇒ 与训练脚本同一套，避免 fp16/fp32 混用崩）----
    unet = T.load_sd_component("unet", dtype=torch.float32, device=device).eval()
    vae = T.load_sd_component("vae", dtype=torch.float32, device=device).eval()
    tok = T.load_sd_component("tokenizer")
    txt = T.load_sd_component("text_encoder", dtype=torch.float32, device=device).eval()
    sched = DDIMScheduler.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="scheduler",
        cache_dir="/domains/hf_cache")
    sched.set_timesteps(DDIM_STEPS, device=device)
    log(f"[SD] UNet/VAE/text-encoder 全冻结｜DDIM {DDIM_STEPS} 步｜guidance {GUIDANCE}")

    # ---- 文本 embedding（固定提示词）----
    with torch.no_grad():
        ids = tok([T.PROMPT], padding="max_length", max_length=77, truncation=True,
                  return_tensors="pt").input_ids.to(device)
        text_emb = txt(ids)[0]                                 # (1, 77, 768) fp32

    clip_model, clip_proc = load_clip(device)

    # ---- 真图（test split）：路径自发现 + CLIP 编码 ----
    true_paths = find_test_image_paths()
    log(f"[真图] 自发现 {len(true_paths)} 张 test 图")
    if len(true_paths) != 200:
        # ⚠️ 宁可失败，也不要静默换真图特征空间（生成侧用本地 CLIP，
        #    真图侧若用另一套特征 ⇒ 两个空间不可比 ⇒ 判决无效）。
        raise RuntimeError(
            f"test split 的图自发现得到 {len(true_paths)} 张（期望 200）⇒ "
            f"不能用缓存特征当真图侧（特征空间会不一致）⇒ 先查 Image_set 布局")
    imgs = [Image.open(p).convert("RGB").resize((RES, RES), Image.BICUBIC)
            for p in true_paths]
    true_clip = clip_encode(clip_model, clip_proc, imgs, device)
    np.save(out_dir / "true_clip.npy", true_clip)
    with open(out_dir / "true_clip_paths.json", "w", encoding="utf-8") as f:
        json.dump([str(p) for p in true_paths], f, ensure_ascii=False, indent=1)

    # ---- 逐被试：条件 → 生成 → CLIP ----
    n_img = 200
    lo = n_img * args.shard // args.n_shards
    hi = n_img * (args.shard + 1) // args.n_shards
    log(f"[生成] 概念区间 [{lo}, {hi})")

    for sid in args.subjects:
        fp = out_dir / f"gen_clip_sub{sid:02d}.pt"
        if fp.exists() and not args.overwrite:
            log(f"[跳过] sub-{sid:02d} 已有产物")
            continue

        # 条件：test EEG（取 rep 0 ⇒ 每被试每概念 1 个试次）
        eeg = np.load(TEST_4D / f"sub-{sid:02d}.npz")["eeg"]
        log(f"[sub-{sid:02d}] test EEG {eeg.shape}")
        flat = np.asarray(eeg[:n_img, 0], dtype=np.float32)
        with torch.no_grad():
            z = E93B.encode_all(encoder, flat, device, args.batch)
        cond = z.astype(np.float32)
        torch.save({"cond": torch.from_numpy(cond)}, out_dir / f"test_cond_sub{sid:02d}.pt")

        feats = []
        pix = []
        samples = []
        for i in range(lo, hi):
            g = torch.Generator(device="cpu").manual_seed(gen_seed(sid, i))
            lat = torch.randn((1, 4, RES // 8, RES // 8), generator=g,
                              dtype=torch.float32).to(device) * sched.init_noise_sigma
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                c = torch.from_numpy(cond[i:i + 1]).to(device, torch.float32)
                tokens = adapter(c)                              # (1, 16, 768) fp32
                enc = torch.cat([text_emb, tokens], dim=1)      # (1, 77+16, 768)
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
            feats.append(clip_encode(clip_model, clip_proc, [pil], device)[0])
            if len(pix) < 16:
                pix.append(arr[::16, ::16, :].astype(np.float32))
            if (i - lo) % 25 == 0:
                log(f"  [sub-{sid:02d}] {i}/{hi}")
        # 样例拼图（**给人看的**：生成图 + 对应真图，各 8 张）
        try:
            if samples:
                grid = Image.new("RGB", (256 * 4, 256 * 4), (255, 255, 255))
                for j, sm in enumerate(samples[:8]):
                    grid.paste(sm.resize((256, 256)), (256 * (j % 4), 256 * (j // 4)))
                for j, sm in enumerate(samples[:8]):
                    ti = lo + j
                    if ti < len(true_paths):
                        grid.paste(Image.open(true_paths[ti]).convert("RGB").resize((256, 256)),
                                   (256 * (j % 4), 256 * 2 + 256 * (j // 4)))
                grid.save(out_dir / f"gen_samples_sub{sid:02d}.jpg", quality=88)
        except Exception as e:                                   # noqa: BLE001
            log(f"[样例] 生成图拼图失败（不影响判决）：{e}")
        feats = l2norm(np.stack(feats, axis=0))
        # ---- **对齐闸门**：真图顺序若与 EEG 行序对不上 ⇒ 2-way 全无意义 ⇒ 当场停 ----
        # 判据：冻结 encoder 的 200-way top-1 必须显著高于随机（1/200 = 0.005）
        # ⇒ 参照 exp101 的同口径数字 0.0589（管线零漂移）。
        g_all, neg_all = build_candidate_structure(200)
        cz = l2norm(cond.astype(np.float64))
        sim = cz @ true_clip[:len(cz)].T
        acc200 = float((sim.argmax(axis=1) == np.arange(len(cz))).mean())
        acc2 = two_way_acc(cz, true_clip[:len(cz)], neg_all[:len(cz)])
        log(f"[对齐] sub-{sid:02d}: 200-way top-1={acc200:.4f}（随机 0.0050｜"
            f"exp101 参照 0.0589）｜条件侧 2-way={acc2:.4f}")
        if acc200 < 0.01:
            raise RuntimeError(
                f"对齐闸门未过：200-way={acc200:.4f} ≈ 随机 ⇒ **真图顺序与 EEG 行序对不上** "
                f"⇒ 2-way/50-way 全无意义 ⇒ 停（先查 test_images 的概念顺序）")
        torch.save({"clip": torch.from_numpy(feats.astype(np.float32)),
                    "lo": lo, "hi": hi, "subject": sid,
                    "align_acc200": acc200, "align_acc2": acc2},
                   fp)
        # C0 的「图不是常数」：前 16 张的像素两两距离中位
        if len(pix) >= 2:
            p = np.stack(pix, axis=0).reshape(len(pix), -1)
            d = [float(np.linalg.norm(p[i] - p[j]))
                 for i in range(len(p)) for j in range(i + 1, len(p))]
            torch.save({"img_pair_dist_median": float(np.median(d))},
                       out_dir / f"gen_dist_sub{sid:02d}.pt")
        # 样例图（给人看的，8 张拼一张）
        try:
            from PIL import Image as I2
            grid = I2.new("RGB", (256 * 4, 256 * 2))
            for j in range(8):
                grid.paste(Image.open(true_paths[lo + j]).convert("RGB").resize((256, 256)),
                           (256 * (j % 4), 256 * (j // 4)))
            grid.save(out_dir / f"true_samples_sub{sid:02d}.jpg", quality=88)
        except Exception as e:                                   # noqa: BLE001
            log(f"[样例] 真图拼图失败（不影响判决）：{e}")

    log(f"[生成] 完成 ⇒ {out_dir}")
    return 0


# ===========================================================================
# judge step
# ===========================================================================
def step_judge(args):
    import torch

    out_dir = Path(args.out_dir)
    records = []
    group, neg = build_candidate_structure(200)

    true_clip = None
    tp = out_dir / "true_clip.npy"
    if tp.exists():
        true_clip = l2norm(np.load(tp).astype(np.float64))
    else:
        # ⚠️ **不许**在这里退回缓存的 ViT-H-14 特征：生成侧用的是本地 CLIP，
        #    两套特征若不同源 ⇒ 2-way 的两个比较项不在同一空间 ⇒ 判决无效。
        log("[真图] ✗ 缺 true_clip.npy ⇒ 生成 step 没跑成功（或真图自发现失败）"
            "⇒ 不用缓存特征顶替（特征空间会不一致）⇒ 本轮不判")
        return 1

    # 条件侧 2-way（**同一个 CLIP** 下的口径 ⇒ 传递效率自洽）
    cond2_cache = out_dir / "cond2_same_clip.json"
    cond2_all = {}
    for sid in TEST_SUBJECTS:
        cp = out_dir / f"test_cond_sub{sid:02d}.pt"
        if not cp.exists():
            continue
        c = torch.load(cp, map_location="cpu", weights_only=False)["cond"].numpy().astype(np.float64)
        cond2_all[sid] = two_way_acc(l2norm(c), true_clip, neg)

    loss_summary = {}
    curve = out_dir / "train_curve.json"
    if curve.exists():
        try:
            import exp102_train as T
            loss_summary = T.loss_curve_summary(json.loads(curve.read_text(encoding="utf-8")))
        except Exception as e:                                   # noqa: BLE001
            log(f"[C0] 损失曲线读不出：{e}")

    gen_feats = {}
    for sid in TEST_SUBJECTS:
        fp = out_dir / f"gen_clip_sub{sid:02d}.pt"
        if not fp.exists():
            log(f"[judge] ⚠ 缺 sub-{sid:02d} 的生成特征 ⇒ 跳过")
            continue
        obj = torch.load(fp, map_location="cpu", weights_only=False)
        g = l2norm(obj["clip"].numpy().astype(np.float64))
        gen_feats[f"sub-{sid:02d}"] = g
        lo = int(obj.get("lo", 0))
        n = len(g)

        # 地板：donor 被试 = s+1，概念 = (i+37)%200 ⇒ **零额外生成**
        d = donor_subject(sid)
        dp = out_dir / f"gen_clip_sub{d:02d}.pt"
        floor_feat = None
        if dp.exists():
            dobj = torch.load(dp, map_location="cpu", weights_only=False)
            dfeat = l2norm(dobj["clip"].numpy().astype(np.float64))
            floor_feat = np.stack([dfeat[floor_index(i)] for i in range(n)], axis=0)

        own2 = two_way_acc(g, true_clip[:n], neg[:n])
        own50 = fifty_way_acc(g, true_clip[:n], group[:n])
        rec = {"subject": f"sub-{sid:02d}", "ok": True, "n_images": n,
               "shard_lo": lo,
               "own2": own2, "own50": own50,
               "floor2": two_way_acc(floor_feat, true_clip[:n], neg[:n]) if floor_feat is not None else None,
               "floor50": fifty_way_acc(floor_feat, true_clip[:n], group[:n]) if floor_feat is not None else None,
               "cross_subject2": None,     # 下面用 donor 的 own2 覆盖（C1b 的地板口径）
               "cond2_same_clip": cond2_all.get(sid),
               "clip_sim": float(np.mean(np.sum(g * true_clip[:n], axis=1))),
               "loss_summary": loss_summary}
        dp_obj = out_dir / f"gen_dist_sub{sid:02d}.pt"
        if dp_obj.exists():
            rec["img_pair_dist"] = torch.load(dp_obj, map_location="cpu",
                                              weights_only=False)["img_pair_dist_median"]
        records.append(rec)

    # C1b 的「跨被试同概念」地板 = donor 被试**自己的** 2-way（同一概念配对）
    by_sub = {r["subject"]: r for r in records}
    for r in records:
        sid = int(r["subject"].split("-")[1])
        d = donor_subject(sid)
        dr = by_sub.get(f"sub-{d:02d}")
        r["cross_subject2"] = dr["own2"] if dr else None
        r["cross_subject_subject"] = f"sub-{d:02d}"

    # 每概念召回（只报）：**每个被试自己的**生成图对每个概念，
    # 「更靠近真图而不是负例」的比例 ⇒ 再对概念取中位。
    recall = None
    if gen_feats:
        subs = sorted(gen_feats.keys())
        n_min = min(len(gen_feats[s]) for s in subs)
        per_concept = []
        for i in range(n_min):
            hit = 0
            for s in subs:
                g = gen_feats[s]
                s_true = g[i] @ true_clip[i]
                s_neg = g[i] @ true_clip[neg[i]]
                hit += int(s_true > s_neg)
            per_concept.append(hit / len(subs))
        if per_concept:
            recall = float(np.median(per_concept))
    for r in records:
        r["per_concept_recall"] = recall

    verdict = judge(records)
    out = {"verdict": verdict, "records": records,
           "atm_reference_C_class": ATM_REF,
           "exp101_condition_reference": EXP101_REF}

    # 落盘：逐被试原始记录（**离线可重算**）+ 判决
    res_dir = Path(args.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    if records:
        cols = ["subject", "n_images", "own2", "floor2", "own50", "floor50",
                "cross_subject2", "cross_subject_subject", "cond2_same_clip",
                "clip_sim", "img_pair_dist", "per_concept_recall"]
        with open(res_dir / "per_subject.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in records:
                w.writerow(r)
    (res_dir / "verdict.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    # C0 的「损失下降」是**整轮量**（不在逐被试表里）⇒ 单独落盘 ⇒ 离线重算要读它
    (res_dir / "loss_summary.json").write_text(
        json.dumps(loss_summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    lines = []
    lines.append("=== exp102 判据（§四十四）===")
    lines.append(f"判决: {verdict.get('verdict')}")
    lines.append(f"C0: {json.dumps(verdict.get('C0'), ensure_ascii=False)}")
    lines.append(f"C1: {json.dumps(verdict.get('C1'), ensure_ascii=False)}")
    lines.append(f"C1b: {json.dumps(verdict.get('C1b_report'), ensure_ascii=False)}")
    lines.append(f"C2: {json.dumps(verdict.get('C2'), ensure_ascii=False)}")
    lines.append(f"C3: {json.dumps(verdict.get('C3_report'), ensure_ascii=False)}")
    lines.append("")
    lines.append("逐被试：")
    for r in records:
        lines.append(
            f"  {r['subject']}: own2={r['own2']:.4f} floor2={r['floor2']} "
            f"own50={r['own50']:.4f} floor50={r['floor50']} "
            f"cross2={r['cross_subject2']} cond2={r['cond2_same_clip']} "
            f"clip_sim={r['clip_sim']:.4f}")
    (res_dir / "judge_log.txt").write_text("\n".join(lines), encoding="utf-8")
    for ln in lines:
        log(ln)
    log(f"结果写入 {res_dir}")
    return 0


# ===========================================================================
# 自测
# ===========================================================================
def _fake_records(n=10, own=0.80, floor=0.52, own50=0.30, floor50=0.02,
                  cross=0.79, cond=0.78, dist=50.0, loss_ok=True):
    recs = []
    for i in range(n):
        recs.append({"subject": f"sub-{i+1:02d}", "ok": True, "n_images": 200,
                     "own2": own, "floor2": floor, "own50": own50, "floor50": floor50,
                     "cross_subject2": cross, "cond2_same_clip": cond,
                     "clip_sim": 0.2, "img_pair_dist": dist,
                     "per_concept_recall": 0.8,
                     "loss_summary": {"decreased": loss_ok, "first10_mean": 1.0,
                                      "last10_mean": 0.5 if loss_ok else 1.2}})
    return recs


def self_test():
    fail = 0

    def chk(name, cond):
        nonlocal fail
        if cond:
            print(f"✓ {name}")
        else:
            print(f"✗ {name}")
            fail += 1

    # ① 判决：通过的情形
    v = judge(_fake_records())
    chk("C0 通过（损失降 + 图不是常数）", v.get("C0_pass") is True)
    chk("C1 通过（own2 显著高于地板）", v.get("C1_pass") is True)
    chk("C2 通过（own50 显著高于地板）", v.get("C2_pass") is True)
    chk("总判决 = CONDITION_DRIVES_GENERATION",
        v.get("verdict") == "CONDITION_DRIVES_GENERATION")
    chk("传递效率 ≈ 0.80/0.78", abs((v["C3_report"]["transfer_efficiency"] or 0) - 0.80 / 0.78) < 1e-6)

    # ② 判决：生成 ≤ 地板 ⇒ 「条件驱动不了生成」
    v2 = judge(_fake_records(own=0.50, floor=0.52))
    chk("own2 ≤ 地板 ⇒ C1 不过",
        v2.get("C1_pass") is False)
    chk("own2 ≤ 地板 ⇒ 判决 = CONDITION_DOES_NOT_DRIVE_GENERATION",
        v2.get("verdict") == "CONDITION_DOES_NOT_DRIVE_GENERATION")

    # ③ 判决：方向对但检验力不足（差为正、但被试数不够）
    v3 = judge(_fake_records(n=5, own=0.80, floor=0.52))
    chk("被试数 < 8 ⇒ C1 不过", v3.get("C1_pass") is False)
    chk("被试数 < 8 ⇒ 判决 = DIRECTION_RIGHT_UNDERPOWERED",
        v3.get("verdict") == "DIRECTION_RIGHT_UNDERPOWERED")

    # ④ C0 不过 ⇒ 全实验作废
    v4 = judge(_fake_records(loss_ok=False))
    chk("损失没降 ⇒ C0 不过", v4.get("C0_pass") is False)
    chk("损失没降 ⇒ 判决 = C0_FAIL", v4.get("verdict") == "C0_FAIL")
    v5 = judge(_fake_records(dist=0.0))
    chk("图是常数 ⇒ C0 不过", v5.get("C0_pass") is False)

    # ⑤ 统计口径：中位 = 上中位（sorted[n//2]）
    chk("上中位口径", E101.med_up([1.0, 2.0, 3.0, 4.0]) == 3.0)
    n, pos, neg, p, zero = E101.sign_test_nonzero([0.1] * 10)
    chk("符号检验：10/10 正 ⇒ p ≤ 0.002（双侧精确）",
        abs(p - 2.0 / 1024.0) < 1e-12 and pos == 10 and zero == 0)

    # ⑥ 地板口径：跨被试 + 跨试次（概念必须错开）
    chk("地板概念确实错开（+37 mod 200）", floor_index(0) == 37 and floor_index(190) == 27)
    chk("donor 被试是环（sub-10 → sub-01）",
        donor_subject(10) == 1 and donor_subject(1) == 2)

    # ⑦ 种子：own 与地板共用同一套分布（同 (donor, 概念) ⇒ 同种子）
    chk("种子由 (donor, 概念) 决定 ⇒ 可复现",
        gen_seed(3, 17) == gen_seed(3, 17) and gen_seed(3, 17) != gen_seed(4, 17))

    # ⑧ 候选集沿用 exp101（同一 seed ⇒ 可比）
    g1, n1 = build_candidate_structure(200)
    chk("候选集 = 200 张、4 组、每组 50",
        len(g1) == 200 and len(set(g1.tolist())) == 4)
    chk("2-way 负例不自指", all(n1[i] != i for i in range(200)))

    # ⑨ 2-way / 50-way 的计算：造可手算的数据
    t = np.eye(4)                      # 真图 = 单位基
    negs = np.array([1, 2, 3, 0])
    a_ok = two_way_acc(np.eye(4), t, negs)
    chk("2-way 手算一致（4/4：生成图与真图逐一对应）", abs(a_ok - 1.0) < 1e-9)
    g_bad = np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    a_bad = two_way_acc(g_bad, t, negs)
    chk("2-way 手算一致（3/4：第一行的生成图指向了别的候选）",
        abs(a_bad - 0.75) < 1e-9)

    print(f"\n{'全部通过' if fail == 0 else f'{fail} 项失败'}")
    return 1 if fail else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["generate", "judge"], default="generate")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--results-dir", default="/domains/runs/illusionbrain-align/exp102/results")
    ap.add_argument("--subjects", type=int, nargs="+", default=list(TEST_SUBJECTS))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--adapter", default=str(DEFAULT_OUT / "adapter_final.pt"))
    ap.add_argument("--encoder-ckpt", default=ENCODER_CKPT)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if args.step == "generate":
        return step_generate(args)
    return step_judge(args)


if __name__ == "__main__":
    sys.exit(main())
