#!/usr/bin/env python3
"""exp78_pooling_axis.py — 量化一个口径缺陷：**CBraMod 的池化轴反了**。

## 缺陷是什么

`registry.pool_tokens` 按 (B, N, D) 处理，**沿 dim=1（token 轴）**取均值。
但 **CBraMod 的最终输出是 (B, D, N)**（通道在前：(B, 200, 63)）⇒
对它来说，dim=1 是**通道轴** ⇒ 实际做的是「每个空间位置把 200 维特征压成一个标量」，
得到 **63 维**（每 token 一个数），而不是「63 个 token 的 200 维特征取均值」。

⇒ 基线表里 `cbramod` 那一行（+0.0105，7/10 显著）与其它 6 个模型
（沿 token 轴、得到 D 维）**口径不一致** ⇒ 跨行比大小不严谨。

## 本脚本做什么

同一批真实数据、同一批试次，**两种池化各算一次指标 A（RSA）**：
- `as_is`：现状（沿 dim=1 均值）；
- `layout_aware`：先用 **token 化入口的输出形状** (N_tok, D) 判定最终输出的轴序，
  若是 (B, D, N) 就转成 (B, N, D)，再沿 token 轴均值。

⚠️ 判定**不用「shape[1] > shape[2]」这种脆弱启发式**，而是用
**token 化入口的实测输出形状**做基准（`_find_tokenizer2` 与 exp76 同一套）。

## 用法

    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp78_pooling_axis.py \
      -v hf://buckets/illusion-brain/training-data:/training:ro \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor cpu-upgrade --timeout 60m -d --models cbramod,signal-jepa --subjects 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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

import base_specs as BS            # noqa: E402
import domain_adapters as DA       # noqa: E402
import eval_brain_alignment as EVAL   # noqa: E402

TEST_4D = Path("/domains/omnibrain4d/eeg")
FEAT_TEST = "/training/eeg/eeg-image-decode/ViT-H-14_features_test.pt"
MODELS_DIR = Path("/training/models")
OUT_DIR = Path("/domains/runs/illusionbrain-align/pooling")
SEED = 0


def log(*a):
    print(*a, flush=True)


def _find_tokenizer2(model, x):
    """找 token 化入口（接受 3 维与 4 维输出）—— 与 exp76 同一套，但**排除 Rearrange**。

    ⚠️ 必须排除 einops 的 `Rearrange`：它只是改轴序的纯布局算子，
    **不是** token 化入口，而且 CBraMod 把它**调用了两次**
    （patch embed 之后转成 token、结束前再转回去）⇒
    ① 它的单独调用会把 3 维输入重排成假形状（实测 (B,200,63)）；
    ② hook 抓到的是**最后一次**调用（转回去之后）⇒ 基准形状会认反。
    """
    import torch
    cands = []

    def hook(nm):
        def _h(mod, inp, o):
            t = o
            if isinstance(t, (tuple, list)):
                t = t[0] if t else None
            if torch.is_tensor(t) and t.ndim in (3, 4) and t.shape[1] > 1:
                cands.append(nm)
        return _h

    handles = [m.register_forward_hook(hook(nm)) for nm, m in model.named_modules() if nm]
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()
    cand_set = set(cands)
    for nm, mod in model.named_modules():
        if nm not in cand_set:
            continue
        if "rearrange" in type(mod).__name__.lower():
            continue                      # 纯布局算子，不是 token 化入口
        try:
            with torch.no_grad():
                out = mod(x)
        except Exception:
            continue
        t = out[0] if isinstance(out, (tuple, list)) else out
        if torch.is_tensor(t) and t.ndim in (3, 4) and t.shape[1] > 1:
            return nm, mod
    return None, None


# exp76 结构勘查**实测**的 (N_tok, D)（来源：research/brain_domains/results/structure/*.json）
# ⚠️ 不用运行时去猜 token 化入口：CBraMod 的 patch embed 前向里给 4 维 (B,200,7,9)，
#    而 `rearrange` 被调用两次、单独调用还会给出假形状 ⇒ 运行时识别极脆（实测连踩三次）。
#    这里用**已测好的表**，可追溯到 exp76 的勘查记录。
REF_ND = {
    "cbramod": (63, 200),
    "eegdino-small": (95, 200),
    "biot": (162, 256),
    "signal-jepa": (62, 64),
    "mirepnet": (61, 256),
    "braintokenizer": (19, 512),
}


def token_ref_shape(model, x, name=None):
    """token 数与 token 维 (N_tok, D)。

    优先用实测表（REF_ND）；查不到再退回运行时识别（并打印告警）。
    """
    if name and name in REF_ND:
        return REF_ND[name]
    import torch
    import registry as REG
    nm, mod = _find_tokenizer2(model, x)
    if mod is None:
        return None
    cap = {}

    def hook(m, inp, out):
        t = out
        if isinstance(t, (tuple, list)):
            t = t[0] if t else None
        if "t" not in cap and torch.is_tensor(t):
            cap["t"] = t.detach()

    h = mod.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(x)
    finally:
        h.remove()
    t = cap.get("t")
    if t is None:
        return None
    t = REG.normalize_token_tensor(t.float())
    return tuple(t.shape[1:])


def extract_both(handle, eeg, have_chans, args):
    """同一批数据，抽两份特征：as_is 与 layout_aware。"""
    import torch
    import registry as REG
    n_img, n_rep = eeg.shape[0], eeg.shape[1]
    flat = eeg.reshape(n_img * n_rep, eeg.shape[2], eeg.shape[3])
    mdt = next((p.dtype for p in handle.model.parameters()), None)
    if mdt is None or not torch.is_floating_point(torch.empty(0, dtype=mdt)):
        mdt = torch.float32

    # 基准形状（token 数, 维数）
    dummy = np.zeros((1, eeg.shape[2], eeg.shape[3]), dtype=np.float32)
    adapted, _ = DA.adapt(dummy, handle.spec, have_chans)
    ref = token_ref_shape(handle.model,
                          torch.from_numpy(np.ascontiguousarray(adapted)).to(dtype=mdt),
                          name=handle.name)
    log(f"    token 数/维基准 (N_tok, D) = {ref}")

    outs = {"as_is": [], "layout_aware": []}
    flipped = None
    for s in range(0, flat.shape[0], args.chunk):
        sub = flat[s:s + args.chunk]
        adapted, _ = DA.adapt(sub, handle.spec, have_chans)
        xb = torch.from_numpy(np.ascontiguousarray(adapted)).to(device=args.device,
                                                                  dtype=mdt)
        with torch.no_grad():
            out = handle.model(xb)
        tok, how = REG.tokens_from_output(out, 2)
        if tok is None:
            raise RuntimeError(f"抽不到 token（{how}）")
        tok_raw = tok.float()
        tok_layout = tok_raw
        # 判定轴序：最终输出 (B, a, b)；
        # 若 (a, b) == (D, N_tok)（而不是 (N_tok, D)）⇒ 是 (B, D, N) ⇒ 需要转置
        if ref is not None and tok_raw.ndim == 3:
            a, b = int(tok_raw.shape[1]), int(tok_raw.shape[2])
            D, N = int(ref[1]), int(ref[0])
            flipped = (a == D and b == N and a != b)
            if flipped:
                tok_layout = tok_raw.transpose(1, 2)
        else:
            flipped = False
        # ⚠️ 两份特征必须来自**不同**的张量（as_is 用未转置的），
        #    否则转置与否算出来完全一样 ⇒ 实验等于没做。
        outs["as_is"].append(
            REG.pool_tokens(tok_raw, "mean_all")[:xb.shape[0]].cpu().numpy())
        outs["layout_aware"].append(
            REG.pool_tokens(tok_layout, "mean_all")[:xb.shape[0]].cpu().numpy())
    if flipped is None:
        flipped = False
    res = {}
    for k, v in outs.items():
        X = np.concatenate(v, axis=0)
        res[k] = X.reshape(n_img, n_rep, -1)
    res["flipped"] = flipped
    res["ref"] = ref
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="cbramod,signal-jepa")
    ap.add_argument("--subjects", default="1")
    ap.add_argument("--pool", default="mean_all")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    import torch
    import registry as REG

    REG.ensure_braindecode_version("stable")
    import braindecode.models as bdmodels

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fte = torch.load(FEAT_TEST, map_location="cpu", weights_only=False)
    IMG_test = fte["img_features"].numpy().astype(np.float64)

    names = [s.strip() for s in args.models.split(",") if s.strip()]
    sids = [int(s) for s in args.subjects.split(",") if s.strip()]
    report = {"models": {}, "n_perm": args.n_perm, "pool": args.pool}

    for nm in names:
        h = REG.load_base_model(nm, models_dir=MODELS_DIR, braindecode_models=bdmodels,
                                spec=BS.BASE_SPECS.get(nm), verbose=False)
        if not h.ok:
            log(f"✗ {nm}: {h.error[:200]}")
            report["models"][nm] = {"ok": False, "error": h.error[:300]}
            continue
        report["models"][nm] = {"ok": True, "subjects": {}}
        for sid in sids:
            eeg = np.load(TEST_4D / f"sub-{sid:02d}.npz")["eeg"]
            log(f"\n=== {nm} sub-{sid:02d} test {eeg.shape} ===")
            try:
                feats = extract_both(h, eeg, BS.THINGS_EEG_CHANS, args)
            except Exception as e:
                log(f"  ✗ 抽取失败：{type(e).__name__}: {e}")
                report["models"][nm]["subjects"][f"sub-{sid:02d}"] = {
                    "ok": False, "error": f"{type(e).__name__}: {e}"[:300],
                    "trace": traceback.format_exc()[-500:]}
                continue
            rec = {"flipped": bool(feats["flipped"]), "ref": list(feats["ref"]) if feats["ref"] else None}
            for key in ("as_is", "layout_aware"):
                X = feats[key]
                rng = np.random.default_rng(SEED)
                a = EVAL.metric_a_rsa(X, IMG_test, args.n_perm, rng)
                rec[key] = {"dim": int(X.shape[-1]), "r_img": a["r_img"],
                            "p_img": a["p_img"]}
                log(f"  {key:<14} dim={X.shape[-1]:<5} A r={a['r_img']:+.4f} "
                    f"p={a['p_img']:.3f}")
            if rec["flipped"]:
                log("  ⇒ 该模型的最终输出是 (B, D, N)（通道在前）⇒ 现状沿通道轴池化")
            report["models"][nm]["subjects"][f"sub-{sid:02d}"] = rec
            del eeg, feats

    (out_dir / "pooling_axis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"\n结果写入 {out_dir}/pooling_axis.json")


if __name__ == "__main__":
    main()
