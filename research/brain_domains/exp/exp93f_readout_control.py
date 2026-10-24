#!/usr/bin/env python3
"""exp93f_readout_control.py — **预检**：位置特异性是被「训练目标」洗掉的，还是被「读出」洗掉的？

## 为什么必须先做这个预检

§二十九·五 由 exp93c 得到一个很强的断言：

> 「**语义监督会把位置特异性洗掉**」（S1b 的剂量斜率只有自监督 S1a 的约 1/5）

而这条断言**正是 exp90 选型的前提**（前部原语若用「猕猴 M1/PMd/DMFC 运动目标」这种
**外部功能监督**来训，会不会继承同样的风险？）。

⇒ **但 exp93c 的比较里藏着一个混淆**：

| 模型 | 训练目标 | 读出 |
|---|---|---|
| S1a | 掩码波形重建（**自监督**） | `recon`（重建波形） |
| S1b | 加语义监督 | `onehot_mean`（**离散码**） |

⇒ **读出类型与训练目标同时变了** ⇒ 「斜率差 5 倍」到底归因于哪一个，**目前分不开**。

## 本脚本怎么拆

**M1（排除读出混淆）**：S1a 与 S1b **各算 6 种读出**（`recon` / `onehot_mean` /
`continuous` / `emb_mean` / `random_codebook` / `code_ids`），
在**同一读出**下比较剂量斜率。
- 若同一读出下 S1b 仍远平于 S1a ⇒ 混淆排除 ⇒ **目标**是原因 ⇒ 猕猴目标有实质风险；
- 若同一读出下两者相近 ⇒ 差异来自**读出** ⇒ §二十九·五 那条断言**必须限定/改写**。

**M2（换一种语义监督再验一次）**：把 **exp93b 的对比学习 encoder**（InfoNCE→CLIP 图像特征，
另一种语义监督、连续读出）也测一条剂量曲线。
- 若它**陡** ⇒ 连对比学习都不洗掉位置特异性 ⇒ 更支持「读出混淆」；
- 若它**平** ⇒ 语义/对比监督确实洗 ⇒ 对猕猴目标**必须**在真数据上先预检。

**代价**：`readouts_for` **一次前向返回全部读出** ⇒ M1 几乎不增加算力。

## 指标

- 每个 (模型, 读出, k) 的 RSA 中位（k 个后部通道，总数固定 34，抽 `--n-draw` 次）；
- **斜率** = r(k=34) − r(k=5)（**越大 ⇒ 越依赖后部**）；
- **归一化斜率** = (r(k=34) − r(k=5)) / (r(ALL) − r(k=5))；
- **单调性**：相邻 k 的差值有多少为正。

## 用法

    python -u research/brain_domains/exp/exp93f_readout_control.py --self-test

    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp93f_readout_control.py --subjects 1,2,3,4,5 \
      -v hf://buckets/illusion-brain/training-data:/training:ro \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor cpu-upgrade --timeout 180m -d --name t93f-s1-5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
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

import exp89_posaxis as E89        # noqa: E402
import exp81_tokenizer_eval as E81  # noqa: E402
from exp93c_dose_response import dose_groups, mask_channels  # noqa: E402

CKPT_S1A = "/domains/runs/illusionbrain-align/exp80/k1024/tokenizer_sub-01.pt"
CKPT_S1B = "/domains/runs/illusionbrain-align/exp86v2/ffn-plain/tokenizer_assembled.pt"
CKPT_C93B = "/domains/runs/illusionbrain-align/exp93b/encoder_best.pt"
OUT_DIR = Path("/domains/runs/illusionbrain-align/exp93f")

SEED = 0
K_LEVELS = (5, 11, 17, 23, 29, 34)
READOUTS = ("recon", "onehot_mean", "continuous", "emb_mean",
            "random_codebook", "code_ids")


def log(*a):
    print(*a, flush=True)


def model_readouts(model, rnd_cb, eeg, n_ch, want, batch_size=64):
    """一次前向拿**全部**读出（exp81.readouts_for 本来就全给）。"""
    import torch
    n_img, n_rep = eeg.shape[0], eeg.shape[1]
    flat = eeg.reshape(n_img * n_rep, n_ch, eeg.shape[3])
    acc = {k: [] for k in want}
    for s in range(0, flat.shape[0], batch_size):
        x = torch.from_numpy(np.ascontiguousarray(flat[s:s + batch_size])).float()
        o = E81.readouts_for(model, x, rnd_cb)
        for k in want:
            acc[k].append(o[k].cpu().numpy())
    return {k: np.concatenate(v, 0).reshape(n_img, n_rep, -1) for k, v in acc.items()}


def self_test():
    log("=== 自测：剂量组 + 一次前向拿全部读出 ===")
    rng = np.random.default_rng(0)
    post = list(range(34))
    ant = list(range(34, 63))
    for k in K_LEVELS:
        gs = dose_groups(post, ant, k, 2, rng)
        for g in gs:
            assert len(g) == 34 and len(set(g)) == 34
            assert sum(1 for c in g if c in post) == k
    log("  ✓ 剂量组构造正确")
    # 读出集合齐全（用假模型不行 ⇒ 只检查键名常量与 exp81 的一致）
    assert set(READOUTS) <= {"recon", "onehot_mean", "continuous", "emb_mean",
                             "random_codebook", "code_ids"}
    log("  ✓ 读出集合与 exp81.readouts_for 一致")
    log("=== 自测通过 ===")


def run_subject(subject, IMG_test, args, rng):
    import torch
    rec = {"subject": f"sub-{subject:02d}"}
    eeg_te, _ = E89.load_test_eeg(subject)
    n_img, n_rep, n_ch, T = eeg_te.shape
    ch_names, _ = E89.load_train_chunk(subject, 1)
    groups, _ = E89.build_groups(ch_names)
    post, ant = groups["POST"], groups["ANT"]
    rec["n_post"], rec["n_ant"] = len(post), len(ant)

    # ---- 参考点：ALL 通道（每个模型/读出各一个）----
    ref = {}
    for tag, ckpt, want in (("s1a", CKPT_S1A, READOUTS),
                            ("s1b", CKPT_S1B, READOUTS)):
        if not Path(ckpt).exists():
            log(f"  ✗ 缺 checkpoint {ckpt}")
            continue
        model, _ck, rnd_cb = E81.load_model(ckpt)
        f = model_readouts(model, rnd_cb, eeg_te, n_ch, want, args.batch_size)
        for ro in want:
            r, _p = E89.rsa(f[ro], IMG_test, args.n_perm, rng)
            ref.setdefault(tag, {})[ro] = r
        log(f"  [参考 ALL] {tag}: " + "  ".join(
            f"{ro}={ref[tag][ro]:+.4f}" for ro in want))
        del model

    # ---- 剂量曲线 ----
    dose = {}
    for tag, ckpt, want in (("s1a", CKPT_S1A, READOUTS),
                            ("s1b", CKPT_S1B, READOUTS)):
        if not Path(ckpt).exists():
            continue
        model, _ck, rnd_cb = E81.load_model(ckpt)
        for k in K_LEVELS:
            rs = {ro: [] for ro in want}
            for g in dose_groups(post, ant, k, args.n_draw, rng):
                f = model_readouts(model, rnd_cb, mask_channels(eeg_te, g), n_ch,
                                   want, args.batch_size)
                for ro in want:
                    r, _p = E89.rsa(f[ro], IMG_test, args.n_perm_null, rng)
                    rs[ro].append(r)
            for ro in want:
                dose.setdefault(tag, {}).setdefault(ro, {})[str(k)] = {
                    "r_median": float(np.median(rs[ro])),
                    "rs": [float(x) for x in rs[ro]]}
            log(f"  [{tag}] k={k:<3} " + "  ".join(
                f"{ro}={np.median(rs[ro]):+.4f}" for ro in want))
        del model

    # ---- exp93b 的对比学习 encoder（另一种语义监督、连续读出）----
    if Path(CKPT_C93B).exists():
        import exp93b_contrastive as E93B
        ck = torch.load(CKPT_C93B, map_location="cpu", weights_only=False)
        model = E93B.build_model(n_ch=ck["n_ch"], n_times=ck["n_times"],
                                  out_dim=ck["out_dim"])
        model.load_state_dict(ck["state_dict"])
        model.eval()

        def enc(eeg):
            n_i, n_r = eeg.shape[0], eeg.shape[1]
            flat = eeg.reshape(n_i * n_r, n_ch, eeg.shape[3])
            outs = []
            with torch.no_grad():
                for s in range(0, flat.shape[0], args.batch_size):
                    x = torch.from_numpy(
                        np.ascontiguousarray(flat[s:s + args.batch_size])).float()
                    outs.append(model(x).numpy())
            return np.concatenate(outs, 0).reshape(n_i, n_r, -1)

        z_all = enc(eeg_te)
        r_all, _p = E89.rsa(z_all, IMG_test, args.n_perm, rng)
        ref.setdefault("c93b", {})["zeroshot_proj"] = r_all
        log(f"  [参考 ALL] c93b: zeroshot_proj={r_all:+.4f}")
        for k in K_LEVELS:
            rs = []
            for g in dose_groups(post, ant, k, args.n_draw, rng):
                z = enc(mask_channels(eeg_te, g))
                r, _p = E89.rsa(z, IMG_test, args.n_perm_null, rng)
                rs.append(r)
            dose.setdefault("c93b", {}).setdefault("zeroshot_proj", {})[str(k)] = {
                "r_median": float(np.median(rs)),
                "rs": [float(x) for x in rs]}
            log(f"  [c93b] k={k:<3} zeroshot_proj={np.median(rs):+.4f}")
        del model
    else:
        log(f"  ⚠ 缺 {CKPT_C93B} ⇒ 跳过对比学习 encoder 的剂量曲线")

    rec["ref_all"] = ref
    rec["dose"] = dose
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-perm-null", type=int, default=50)
    ap.add_argument("--n-draw", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    import torch
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    IMG_test = torch.load(E89.FEAT_TEST, map_location="cpu",
                          weights_only=False)["img_features"].numpy().astype(np.float64)

    per_subject = []
    for sid in [int(s) for s in args.subjects.split(",") if s.strip()]:
        t0 = time.time()
        log(f"\n=== sub-{sid:02d} ===")
        try:
            rec = run_subject(sid, IMG_test, args, np.random.default_rng(SEED + sid))
            per_subject.append(rec)
        except Exception as e:
            rec = {"subject": f"sub-{sid:02d}",
                   "error": f"{type(e).__name__}: {e}"[:300],
                   "trace": traceback.format_exc()[-800:]}
            log(f"  ✗ {rec['error']}")
            log(rec["trace"])
            per_subject.append(rec)
        log(f"  用时 {time.time()-t0:.0f}s")

    n_err = sum(1 for r in per_subject if "error" in r)
    log(f"\n完成：{len(per_subject)-n_err}/{len(per_subject)} 成功"
        + (f"，**{n_err} 个报错**" if n_err else ""))
    if per_subject and n_err == len(per_subject):
        log("✗✗ 全部被试都报错 ⇒ 结果不可用")

    (out_dir / "exp93f_readout_control.json").write_text(
        json.dumps({"config": vars(args), "per_subject": per_subject},
                   indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"结果写入 {out_dir}/exp93f_readout_control.json")


if __name__ == "__main__":
    main()
