#!/usr/bin/env python3
"""exp75_layerwise_rsa.py — 基模**逐层**的图像语义承载曲线（阶段 C 的前置）。

## 要回答的问题

「7 基模脑对齐表」已经测出：基模的冻结表征在 RSA 上有弱但真实的图像语义
（signal-jepa +0.0260 / 9-10 被试显著），在**线性可解码**上全员随机。

但那是**最后一层**的表征。基模内部是 L 层 transformer：
图像语义是在**哪一层**被装进去的？是所有层都有，还是集中在中间/最后？

⇒ 这个答案直接决定阶段 D「跨模型换块」该换哪一层。

## 做法（零训练，一次前向）

1. 用 forward hook 收集模型内部**每个 3 维中间输出**（按调用序）⇒ 逐层 token；
2. 每层按**预先写死**的口径池化（mean_all，含全局 token）；
3. 与 ViT-H-14 图像特征做 RSA（**复用 exp65 的 rdm_full / spearman / rsa_with_null /
   p_from_null**，归一化链：跨图 demean → 逐维 z-score → rdm_full → 秩相关）；
4. 每层给 r、p、以及该层"是否显著"。

## 为什么口径要写死

- 池化规则（mean_all）**预先定死**，不按结果挑；
- RSA 的下三角相关 + 200 次置换，与 exp65 完全同口径 ⇒ 可直接比；
- 每层维度不同**没关系**：RSA 只看 RDM 的序关系，不受维度影响。

## 成本

test split 200 图 × R 次重复（默认 10）= 2000 试次/模型 ⇒
零训练、单次前向，cpu-upgrade 上全部基模约 4–6 分钟。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

EXP_DIR = Path(__file__).resolve().parent
_PKG_CANDIDATES = [EXP_DIR / "models", EXP_DIR.parent / "models", EXP_DIR]
PKG_DIR = next((c for c in _PKG_CANDIDATES if (c / "registry.py").exists()),
               _PKG_CANDIDATES[0])
for _p in (str(EXP_DIR), str(PKG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import warnings                                        # noqa: E402
warnings.filterwarnings('ignore', message='.*standard_1020.*')
warnings.filterwarnings('ignore', message='.*Montage name.*')
import numpy as np                                    # noqa: E402
import exp65_tier0_rsa_fix as E65                     # noqa: E402
import base_specs as BS                               # noqa: E402
import domain_adapters as DA                          # noqa: E402

BLOCKISH = re.compile(r"(layers?\.|blocks?\.|encoder\.layer|transformer\.)", re.I)
MODELS_DIR = Path("/training/models")
TEST_4D = Path("/domains/omnibrain4d/eeg")
FEAT_TEST = "/training/eeg/eeg-image-decode/ViT-H-14_features_test.pt"
OUT_DIR = Path("/domains/runs/illusionbrain-align/layerwise")
N_PERM = 200
SEED = 0


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 逐层中间表征收集
# ---------------------------------------------------------------------------
def collect_ordered(model, x):
    """收集模型内部所有 3 维中间输出（按调用序）。

    返回 [(order, module_name, tensor(B,N,D))]，同一模块保留**最后一次**调用
    的输出（多次调用的模块取最终态），排序按首次调用序。
    """
    import torch
    rec = []
    counter = {"i": 0}
    handles = []

    def mk(nm):
        def hook(mod, inp, out):
            t = out
            if isinstance(t, (tuple, list)):
                t = t[0] if len(t) else None
            # ⚠️ 要收 3 维 **和 4 维**：CBraMod 的 block 输出是 (B, rows, cols, D)
            #    的二维网格（4 维），只收 3 维会漏掉所有 block、只剩注意力子模块。
            if torch.is_tensor(t) and t.ndim in (3, 4) and t.shape[1] > 1:
                rec.append((counter["i"], nm, t.detach()))
                counter["i"] += 1
        return hook

    for nm, mod in model.named_modules():
        if nm:
            handles.append(mod.register_forward_hook(mk(nm)))
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()
    # 同一模块取最后一次；顺序用首次调用号
    first, last = {}, {}
    for i, nm, t in rec:
        first.setdefault(nm, i)
        last[nm] = (i, t)
    out = [(first[nm], nm, t) for nm, (i, t) in last.items()]
    out.sort(key=lambda x: x[0])
    return out


def layerwise_features(handle, eeg, have_chans, batch_size=32, device="cpu",
                       pool="mean_all", log_every=None):
    """对 (n_img, n_rep, C, T) 的 EEG 逐层抽特征。

    返回 {module_name: (n_img, n_rep, D)}
    """
    import torch
    import registry as REG
    n_img, n_rep = eeg.shape[0], eeg.shape[1]
    flat = eeg.reshape(n_img * n_rep, eeg.shape[2], eeg.shape[3])
    acc = {}          # module -> list of (B, D)
    shapes = {}
    dropped = []
    order_of = {}      # module -> 首次调用序号（排序用，不能用模块名）
    for s in range(0, flat.shape[0], batch_size):
        sub = flat[s:s + batch_size]
        adapted, info = DA.adapt(sub, handle.spec, have_chans)
        # ⚠️ dtype 必须跟随模型参数：mirepnet 的权重是 fp16，
        #    而域适配（IDW 插值 / resample_poly）可能给 float32/float64
        mdt = next((q.dtype for q in handle.model.parameters()), None)
        if mdt is None or not torch.is_floating_point(torch.empty(0, dtype=mdt)):
            mdt = torch.float32
        xb = torch.from_numpy(np.ascontiguousarray(adapted)).to(device=device, dtype=mdt)
        with torch.no_grad():
            recs = collect_ordered(handle.model, xb)
        for order, nm, t in recs:
            order_of.setdefault(nm, order)
            tk = REG.normalize_token_tensor(t.float())      # 4 维网格 → (B,N,D)
            f = REG.pool_tokens(tk, pool)
            # ⚠️ **按输入批大小截断**：有些模型（braintokenizer 的 SEANet 编码器）
            #    内部把批次补齐到固定大小（最后一批 16 → 输出 32），
            #    不截断就会多出 16 行 ⇒ 整层被误判为"行数不匹配"而丢掉。
            f = f[:xb.shape[0]]
            acc.setdefault(nm, []).append(f.cpu().numpy())
            shapes[nm] = (tk.shape[1], tk.shape[2])
        # 模型**最终输出**也作为一条曲线（"__final__"）——
        # 有些模型（braintokenizer）内部层名不含 "layers."，会被 block 过滤掉，
        # 但它的最终输出正是脑对齐表里用的那个表征。
        try:
            with torch.no_grad():
                o = handle.model(xb)
            tk, _how = REG.tokens_from_output(o, 2)
            if tk is not None:
                f = REG.pool_tokens(tk.float(), pool)[:xb.shape[0]]
                acc.setdefault("__final__", []).append(f.cpu().numpy())
                shapes["__final__"] = (tk.shape[1], tk.shape[2])
                order_of.setdefault("__final__", 10 ** 6)   # 排在最后
        except Exception:
            pass
    out = {}
    want = n_img * n_rep
    for nm, lst in acc.items():
        X = np.concatenate(lst, axis=0)
        if X.shape[0] != want:
            # ⚠️ 某些模块在个别批次上多/少产出（braintokenizer 的量化器就是）
            #    ⇒ 行数对不上就无法 reshape ⇒ 直接剔除并记账，不能硬 reshape。
            dropped.append({"module": nm, "got_rows": int(X.shape[0]),
                            "want_rows": int(want)})
            continue
        out[nm] = X.reshape(n_img, n_rep, -1)
    # 按**调用序**排（模块名是字典序：layers.10 会排到 layers.1 前面）
    out = dict(sorted(out.items(), key=lambda kv: order_of.get(kv[0], 9999)))
    # 只保留 **block 层**（"逐 block"要看的就是这些）；
    # 其余是 patch embed / conv / 量化器内部，不是 block。
    keep = {nm: X for nm, X in out.items() if BLOCKISH.search(nm)}
    dropped.extend({"module": nm, "reason": "非 block 层"}
                   for nm in out if nm not in keep)
    out = keep
    return out, shapes, dropped


# ---------------------------------------------------------------------------
# RSA（复用 exp65 口径）
# ---------------------------------------------------------------------------
def rsa_of(trial_feats, img_feats, n_perm=N_PERM, rng=None):
    """(n_img, n_rep, D) → RSA。归一化链与 exp65 的 demean 变体同构。"""
    n_img = trial_feats.shape[0]
    X = trial_feats.mean(axis=1).astype(np.float64)      # 试次平均
    X = X - X.mean(axis=0, keepdims=True)                # 跨图去共同成分
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-12
    X = (X - mu) / sd                                     # 逐维 z-score
    R_eeg = E65.rdm_full(X)
    R_img = E65.rdm_full(img_feats.astype(np.float64))
    r, null = E65.rsa_with_null(R_eeg, R_img, n_perm, rng)
    return {"r": float(r), "p": float(E65.p_from_null(r, null)),
            "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "n_perm": int(n_perm)}


# ---------------------------------------------------------------------------
def run_model(name, handle, eeg, img_feats, rng, args):
    feats, shapes, dropped = layerwise_features(handle, eeg, args.chans,
                                              args.batch_size, args.device,
                                              args.pool)
    if dropped:
        log(f"    （剔除 {len(dropped)} 个行数不匹配的层："
            f"{[d['module'] for d in dropped][:4]}）")
    rows = []
    for order, (nm, X) in enumerate(sorted(feats.items())):
        try:
            res = rsa_of(X, img_feats, args.n_perm, rng)
        except Exception as e:
            res = {"r": None, "p": None, "error": f"{type(e).__name__}: {e}"[:150]}
        n_tok, d = shapes.get(nm, (None, None))
        rows.append({"order": order, "module": nm, "n_tokens": n_tok, "dim": d,
                     **res})
    for dr in dropped:
        rows.append({"order": 999, "module": dr["module"], "n_tokens": None,
                     "dim": None, "r": None, "p": None,
                     "dropped": True, "got_rows": dr.get("got_rows"),
                     "reason": dr.get("reason", "行数不匹配")})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--bd-version", default="stable")
    ap.add_argument("--reps", type=int, default=10, help="每图用几次重复")
    ap.add_argument("--pool", default="mean_all")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--chans", default="",
                    help="逗号分隔的输入通道名（默认用 THINGS 的 63 个）")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    import torch
    import registry as REG

    REG.ensure_braindecode_version(args.bd_version)
    import braindecode.models as bdmodels

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    fte = torch.load(FEAT_TEST, map_location="cpu", weights_only=False)
    IMG_test = fte["img_features"].numpy().astype(np.float64)

    eeg = np.load(TEST_4D / "sub-01.npz")["eeg"][:, :args.reps]
    log(f"真实数据：{eeg.shape}（200 图 × {args.reps} 重复）")
    chans = (args.chans.split(",") if args.chans else BS.THINGS_EEG_CHANS)
    args.chans = chans

    names = (BS.models_for_version("stable" if args.bd_version == "stable" else "dev")
             if args.models == "all" else
             [x.strip() for x in args.models.split(",") if x.strip()])
    log(f"本趟模型：{names}")

    report = {"bd_version": args.bd_version, "reps": args.reps,
              "pool": args.pool, "n_perm": args.n_perm, "models": {}}
    for nm in names:
        t0 = time.time()
        sp = BS.BASE_SPECS.get(nm)
        h = REG.load_base_model(nm, models_dir=MODELS_DIR,
                                braindecode_models=bdmodels, spec=sp, verbose=False)
        if not h.ok:
            log(f"  ✗ {nm}: {h.error[:160]}")
            report["models"][nm] = {"ok": False, "error": h.error[:300]}
            continue
        try:
            rows = run_model(nm, h, eeg, IMG_test, rng, args)
        except Exception as e:
            log(f"  ✗ {nm}: {type(e).__name__}: {e}")
            report["models"][nm] = {"ok": False,
                                    "error": f"{type(e).__name__}: {e}"[:300],
                                    "trace": traceback.format_exc()[-600:]}
            continue
        report["models"][nm] = {"ok": True, "n_params": h.n_params, "rows": rows}
        best = max((r for r in rows if r.get("r") is not None),
                   key=lambda r: r["r"], default=None)
        sig = sum(1 for r in rows if (r.get("p") or 1) < 0.05)
        bt = (f"最强 {best['module']} r={best['r']:+.4f} p={best['p']:.3f}"
              if best else "无可用层")
        log(f"  ✓ {nm}: {len(rows)} 层 | RSA 显著层 {sig} | {bt}"
            f" | {time.time()-t0:.0f}s")
        for r in rows:
            if r.get("r") is not None:
                log(f"      [{r['order']:>2}] {r['module'][:52]:<52} "
                    f"tok={r['n_tokens']:<4} d={r['dim']:<5} "
                    f"r={r['r']:+.4f} p={r['p']:.3f}")

    (out_dir / f"layerwise_{'stable' if args.bd_version=='stable' else 'dev'}.json"
     ).write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                  encoding="utf-8")
    log(f"\n结果写入 {out_dir}")


if __name__ == "__main__":
    main()
