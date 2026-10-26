#!/usr/bin/env python3
"""exp93c_dose_response.py — **位置轴的剂量-反应**（模型层 + 特征层）。

## 为什么需要这个实验

exp92-A 补的「模型层地板」（同尺寸随机通道组）给出了一个**出乎意料**的结果：

- **前部（ANT）低于**全部同尺寸随机组：**5/5**（S1a、S1b 都成立）✅
- **后部（POST）高于**全部同尺寸随机组：只有 **3/5**（S1a）、**1/5**（S1b）❌

⇒ 地板**不是内容匹配的**：一个「随机 34 通道组」平均已经含 ~18 个后部通道
（34/63 × 34），它天然携带大量后部信号 ⇒ 用它当「后部特有性」的地板，
**测不出后部特有性，只测得出「信号需要足够多的通道」**。

⇒ 正确的对照不是「随机同样多通道」，而是「**同样多通道、但后部通道数不同**」：
固定通道总数（= 34 = POST 的规模），只改其中**后部通道的个数 k**，
看 RSA 是否随 k 单调上升。

⇒ 这同时检验 D25：**位置轴是梯度，不是两块**。

## 设计

| 量 | 取值 |
|---|---|
| 通道组总数 | 固定 **34**（= POST 规模 ⇒ 通道数带来的信息量被匹配掉）|
| 后部通道数 k | 5 / 11 / 17 / 23 / 29 / 34（k≥5：前部只有 29 个通道）|
| 每个 k 的随机抽样 | 3 次（换哪几个后部、哪几个前部）⇒ 取中位 |
| 读出 | 模型层 S1a `recon`、S1b `onehot_mean`；特征层 630-d |
| 被试 | 10 个 |

参照点：ALL（63）、POST（34 全后部）、ANT（29 全前部）。

## 判据（预先写死）

1. **单调性**：模型层 RSA 随 k 单调上升（相邻 k 差值多数为正）；
2. **斜率**：模型层「k 从 5 到 34」的上升幅度 **显著大于**随机抽取 k 个后部通道时的上升幅度
   （否则只是「通道越多信号越多」）；
3. **ANT 端**：k=5（几乎纯前部）应接近 ANT 的水平（≈0）。

## 用法

    python -u research/brain_domains/exp/exp93c_dose_response.py --self-test

    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp93c_dose_response.py --subjects 1,2,3,4,5 \
      -v hf://buckets/illusion-brain/training-data:/training:ro \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor cpu-upgrade --timeout 180m -d --name t93c-dose-s1-5
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

import exp73_decode200 as E73   # noqa: E402
import exp89_posaxis as E89      # noqa: E402
import exp81_tokenizer_eval as E81  # noqa: E402

CKPT_S1A = "/domains/runs/illusionbrain-align/exp80/k1024/tokenizer_sub-01.pt"
CKPT_S1B = "/domains/runs/illusionbrain-align/exp86v2/ffn-plain/tokenizer_assembled.pt"
OUT_DIR = Path("/domains/runs/illusionbrain-align/exp93c")
SEED = 0
GROUP_SIZE = 34
K_LEVELS = (5, 11, 17, 23, 29, 34)


def log(*a):
    print(*a, flush=True)


def mask_channels(eeg, keep_idx):
    x = eeg.copy()
    m = np.ones(x.shape[2], dtype=bool)
    m[np.asarray(keep_idx)] = False
    x[:, :, m, :] = 0.0
    return x


def dose_groups(post, ant, k, n_draw, rng, size=GROUP_SIZE):
    """固定**总数**为 size 的通道组，其中**恰好 k 个后部**通道。"""
    if k > len(post) or (size - k) > len(ant):
        return []
    if k == len(post) and size == len(post):
        return [list(post)]                      # k=34 就是 POST 本身（确定性）
    out = []
    for _ in range(n_draw):
        p = rng.choice(post, size=k, replace=False)
        a = rng.choice(ant, size=size - k, replace=False)
        out.append(list(p) + list(a))
    return out


def model_readouts(model, rnd_cb, eeg, n_ch, want, batch_size=64):
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
    log("=== 自测：剂量组的构造 ===")
    rng = np.random.default_rng(0)
    post = list(range(34))
    ant = list(range(34, 63))
    for k in K_LEVELS:
        gs = dose_groups(post, ant, k, 3, rng)
        assert gs, f"k={k} 造不出组"
        for g in gs:
            assert len(g) == GROUP_SIZE, (k, len(g))
            assert len(set(g)) == GROUP_SIZE, f"k={k} 有重复通道"
            assert sum(1 for c in g if c in post) == k, f"k={k} 后部通道数不对"
        assert len({tuple(sorted(g)) for g in gs}) == (1 if k == len(post) else 3)
    log("  ✓ 每个 k：总数固定 34、后部恰好 k、无重复")

    # k=5 应当几乎不含后部信息 ⇒ 其通道应几乎全来自 ant
    g = dose_groups(post, ant, 5, 1, rng)[0]
    assert sum(1 for c in g if c in post) == 5
    log("  ✓ k=5：5 个后部 + 29 个前部")
    log("=== 自测通过 ===")


def run_subject(subject, IMG_test, args, rng):
    rec = {"subject": f"sub-{subject:02d}"}
    eeg_te, _ = E89.load_test_eeg(subject)
    n_img, n_rep, n_ch, T = eeg_te.shape
    ch_names, _ = E89.load_train_chunk(subject, 1)
    groups, _ = E89.build_groups(ch_names)
    post, ant = groups["POST"], groups["ANT"]
    rec["n_post"], rec["n_ant"], rec["n_all"] = len(post), len(ant), n_ch

    # ---------- 特征层（630-d，零成本）----------
    feat = {}
    for tag, idx in (("ALL", list(range(n_ch))),
                     ("POST", post), ("ANT", ant)):
        f = E73.per_rep_targets(mask_channels(eeg_te, idx))
        r, p = E89.rsa(f, IMG_test, args.n_perm, rng)
        feat[tag] = {"r": r, "p": p, "n_ch": len(idx)}
        log(f"  [特征] {tag:<4} n_ch={len(idx):<3} r={r:+.4f} p={p:.4f}")
    dose_feat = {}
    for k in K_LEVELS:
        rs = []
        for g in dose_groups(post, ant, k, args.n_draw, rng):
            f = E73.per_rep_targets(mask_channels(eeg_te, g))
            r, _p = E89.rsa(f, IMG_test, args.n_perm_null, rng)
            rs.append(r)
        dose_feat[str(k)] = {"r_median": float(np.median(rs)),
                             "rs": [float(x) for x in rs]}
        log(f"  [特征] k={k:<3} 中位 {np.median(rs):+.4f}（{len(rs)} 次抽样）")
    rec["feature"] = feat
    rec["feature_dose"] = dose_feat

    # ---------- 模型层 ----------
    for tag, ckpt, want in (("s1a", CKPT_S1A, ("recon",)),
                            ("s1b", CKPT_S1B, ("onehot_mean",))):
        if not Path(ckpt).exists():
            log(f"  ✗ 缺 checkpoint {ckpt}")
            continue
        model, _ck, rnd_cb = E81.load_model(ckpt)
        dose = {}
        for k in K_LEVELS:
            rs = []
            for g in dose_groups(post, ant, k, args.n_draw, rng):
                f = model_readouts(model, rnd_cb, mask_channels(eeg_te, g), n_ch,
                                   want, args.batch_size)
                r, _p = E89.rsa(f[want[0]], IMG_test, args.n_perm_null, rng)
                rs.append(r)
            dose[str(k)] = {"r_median": float(np.median(rs)),
                            "rs": [float(x) for x in rs]}
            log(f"  [{tag}] k={k:<3} 中位 {np.median(rs):+.4f}（{len(rs)} 次抽样）")
        rec.setdefault("model_dose", {})[tag] = dose
        del model
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-perm-null", type=int, default=50)
    ap.add_argument("--n-draw", type=int, default=3)
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
    fte = torch.load(E89.FEAT_TEST, map_location="cpu", weights_only=False)
    IMG_test = fte["img_features"].numpy().astype(np.float64)

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

    (out_dir / "exp93c_dose_response.json").write_text(
        json.dumps({"config": vars(args), "per_subject": per_subject},
                   indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"结果写入 {out_dir}/exp93c_dose_response.json")


if __name__ == "__main__":
    main()
