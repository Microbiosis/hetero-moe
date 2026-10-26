#!/usr/bin/env python3
"""exp79_vq_readout_calib.py — S1 前置：把「量化码到底怎么读」这件事钉死。

## 为什么必须先做这一步

基线表里 `braintokenizer` 那一行是 **+0.0155（8/10 被试显著）**，
但那一行**没有说清读的是什么**：

- `pool_tokens(mean_all)` 作用在最终输出 (B, 19, 512) 上 ⇒ 沿 **19 个通道位置**取均值
  ⇒ 得到 (B, 512)，**每个数是 19 个码号的算术平均**。
  ⚠️ 码号的**数值本身是任意的**（码本顺序没有语义）⇒ 512 个「码号均值」是什么，
  此前**没人验证过**。
- braintokenizer 用的是 **ResidualVectorQuantizer**（`quantizer.layers` 多级残差）
  ⇒ 同一个位置其实有**多级码**，而表里那一行把它们混在了一起（码号直接相加？）。

⇒ S1（自训 EEG 离散 tokenizer）要拿 braintokenizer 当**对照基线**，
就必须先知道：**它的语义到底来自哪一级残差、哪种读出**。
否则我们自训出来的码本会拿一个**没定义清楚的指标**去比。

## 本脚本做什么（零训练，一次前向）

对 braintokenizer 的量化输出，同时算几种读出的指标 A（RSA）：

| 读出 | 构造 | 维度 |
|---|---|---|
| ① `code_mean` | 沿 19 个通道位置取**码号**的均值（= 基线表那一行） | 512 |
| ② `embed_mean` | 码号查 `k_proj` 码本 → 沿位置均值 → 沿时间均值 | d（码本维） |
| ③ `level_k` | **逐级**残差码，各自做读出 ① | 512 × L |
| ④ `hist` | 512 个时间步各自的**码号频率直方图** | K |

外加**码本利用率**：用到的码比例 / 频率熵 / 前若干码的占比。

## 用法

    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp79_vq_readout_calib.py \
      -v hf://buckets/illusion-brain/training-data:/training:ro \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor cpu-upgrade --timeout 90m -d --subjects 1,2,3
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

import base_specs as BS            # noqa: E402
import domain_adapters as DA       # noqa: E402
import eval_brain_alignment as EVAL   # noqa: E402

TEST_4D = Path("/domains/omnibrain4d/eeg")
FEAT_TEST = "/training/eeg/eeg-image-decode/ViT-H-14_features_test.pt"
MODELS_DIR = Path("/training/models")
OUT_DIR = Path("/domains/runs/illusionbrain-align/vq-calib")
SEED = 0
MODEL = "braintokenizer"


def log(*a):
    print(*a, flush=True)


def find_codebooks(model):
    """逐级找码本 + 残差级数。

    braintokenizer 用的是 **ResidualVectorQuantizer**（`quantizer.layers` 多级残差，
    实测 **4 级**），每级一个 `VectorQuantizer`，码本在它的 `_codebook` 里
    （`EMACodebook`，是 **buffer** 不是 Parameter）：
        `_codebook.embed`       (512, 256)  ← 码本
        `_codebook.cluster_size` (512,)     ← 每个码的累计使用量 ⇒ 利用率的一手数据
        `_codebook.embed_avg`   (512, 256)

    ⇒ 模型里唯一的 `Embedding` 是 `sensor_embed.sensor_embedding_layer` (3, 256)，
      **不是**码本 ⇒ 不能拿它当码本用（实测踩过）。
    """
    import torch
    quant_layers = []
    for nm, mod in model.named_modules():
        if nm.endswith("quantizer.layers"):
            quant_layers = list(mod.children())
    per_level = []
    for i, lay in enumerate(quant_layers):
        cb = getattr(lay, "_codebook", None)
        emb = getattr(cb, "embed", None) if cb is not None else None
        cs = getattr(cb, "cluster_size", None) if cb is not None else None
        per_level.append((i, f"quantizer.layers.{i}._codebook.embed",
                          emb.detach().clone() if emb is not None else None,
                          cs.detach().clone() if cs is not None else None))
    return per_level, quant_layers


def run_subject(sid, IMG_test, handle, args):
    """一个被试：前向 + 各种读出 + 指标 A。"""
    import torch
    import registry as REG

    eeg = np.load(TEST_4D / f"sub-{sid:02d}.npz")["eeg"]
    n_img, n_rep = eeg.shape[0], eeg.shape[1]
    flat = eeg.reshape(n_img * n_rep, eeg.shape[2], eeg.shape[3])

    mdt = next((p.dtype for p in handle.model.parameters()), None)
    if mdt is None or not torch.is_floating_point(torch.empty(0, dtype=mdt)):
        mdt = torch.float32

    per_level_cb, quant_layers = find_codebooks(handle.model)
    log(f"  残差 VQ 级数：{len(quant_layers)} | 各级码本："
        f"{[(i, n, (tuple(w.shape) if w is not None else None)) for i, n, w, _c in per_level_cb]}")

    # hook：量化器输出（每级的码）
    levels = []

    def mk_hook(idx):
        def hook(mod, inp, out):
            t = out
            if isinstance(t, (tuple, list)):
                t = t[0] if t else None
            if torch.is_tensor(t):
                levels.append((idx, t.detach()))
        return hook

    handles = []
    for i, lay in enumerate(quant_layers):
        handles.append(lay.register_forward_hook(mk_hook(i)))
    # ⚠️ 量化器本身**不 hook**：`final` 那一行必须用**模型的最终输出**
    #    （`tokens_from_output(model(x))`）⇒ 与基线表那一行**逐字同口径**。
    #    （模型的 forward 返回的就是码号，结构勘查实测 final_shape = (1,19,512)）

    acc = {"code_mean": [], "embed_mean": [], "hist": [], "final": []}
    per_level = {}
    code_counter = None
    for s in range(0, flat.shape[0], args.chunk):
        sub = flat[s:s + args.chunk]
        adapted, _ = DA.adapt(sub, handle.spec, BS.THINGS_EEG_CHANS)
        xb = torch.from_numpy(np.ascontiguousarray(adapted)).to(
            device=args.device, dtype=mdt)
        levels.clear()
        with torch.no_grad():
            out = handle.model(xb)
        nb = xb.shape[0]

        def pool_codes(c, cb_w):
            """c: (B, 19, 512) 码号 → 各种读出。"""
            out = {"code_mean": c.float().mean(dim=1)}             # (B, 512)
            if cb_w is not None:
                idx = c.long().clamp(0, cb_w.shape[0] - 1)
                emb = cb_w.to(idx.device)[idx]                    # (B,19,512,d)
                out["embed_mean"] = emb.mean(dim=1).mean(dim=1)   # (B, d)
            return out

        ft = None
        import registry as _REG
        ft, _how = _REG.tokens_from_output(out, 2)
        if ft is not None and ft.ndim == 3:
            c = ft.long()
            # 最终输出是**残差量化器逐级相加**后的结果 ⇒ 读出 ① 用「最终码」
            acc["final"].append(pool_codes(c, None)["code_mean"].cpu().numpy())
            flat_codes = c.reshape(-1).cpu().numpy()
            K = int(flat_codes.max()) + 1 if flat_codes.size else 0
            if code_counter is None:
                code_counter = np.zeros(max(K, 1), dtype=np.int64)
            for v, c_ in zip(*np.unique(flat_codes, return_counts=True)):
                if v < len(code_counter):
                    code_counter[v] += int(c_)
        # 逐级：每一级用自己的码本
        for idx, t in levels:
            if t.ndim != 3:
                continue
            cbw = None
            for li, _nm, w, _cs in per_level_cb:
                if li == idx:
                    cbw = w
                    break
            if t.is_floating_point():
                # 输出是**量化向量** ⇒ 用最近码反查索引
                # （‖t-e‖² = ‖t‖²-2t·e+‖e‖²，避开 (BT,K,D) 的巨大张量）
                if cbw is None:
                    continue
                tt = t.reshape(-1, t.shape[-1]).float()
                e = cbw.to(tt.device).float()
                d = (tt.pow(2).sum(1, keepdim=True) - 2 * tt @ e.t()
                     + e.pow(2).sum(1))
                c = d.argmin(1).view(t.shape[0], t.shape[1])
            else:
                c = t.long()
            o = pool_codes(c, cbw)
            per_level.setdefault(idx, []).append(o["code_mean"].cpu().numpy())
            if "embed_mean" in o:
                per_level.setdefault((idx, "emb"), []).append(
                    o["embed_mean"].cpu().numpy())

    for h in handles:
        h.remove()

    res = {"subject": f"sub-{sid:02d}"}
    rng = np.random.default_rng(SEED)
    for key, chunks in acc.items():
        if not chunks:
            continue
        X = np.concatenate(chunks, axis=0).reshape(n_img, n_rep, -1)
        a = EVAL.metric_a_rsa(X, IMG_test, args.n_perm, rng)
        res[key] = {"dim": int(X.shape[-1]), "r_img": a["r_img"], "p_img": a["p_img"]}
        log(f"  {key:<12} dim={X.shape[-1]:<5} A r={a['r_img']:+.4f} p={a['p_img']:.3f}")
    for idx in sorted({k for k in per_level if isinstance(k, int)}):
        X = np.concatenate(per_level[idx], axis=0).reshape(n_img, n_rep, -1)
        a = EVAL.metric_a_rsa(X, IMG_test, args.n_perm, rng)
        res[f"level{idx}"] = {"dim": int(X.shape[-1]), "r_img": a["r_img"],
                              "p_img": a["p_img"]}
        log(f"  level{idx:<7} dim={X.shape[-1]:<5} A r={a['r_img']:+.4f} "
            f"p={a['p_img']:.3f}")
    for idx in sorted({k[0] for k in per_level if isinstance(k, tuple)}):
        chunks = per_level.get((idx, "emb"))
        if not chunks:
            continue
        X = np.concatenate(chunks, axis=0).reshape(n_img, n_rep, -1)
        a = EVAL.metric_a_rsa(X, IMG_test, args.n_perm, rng)
        res[f"level{idx}_emb"] = {"dim": int(X.shape[-1]), "r_img": a["r_img"],
                                  "p_img": a["p_img"]}
        log(f"  level{idx}_emb dim={X.shape[-1]:<5} A r={a['r_img']:+.4f} "
            f"p={a['p_img']:.3f}")
    if code_counter is not None and code_counter.sum() > 0:
        cnt = code_counter[code_counter > 0]
        p = cnt / cnt.sum()
        ent = float(-(p * np.log(p)).sum())
        res["codebook"] = {
            "n_codes_total": int(len(code_counter)),
            "n_codes_used": int(len(cnt)),
            "utilization": float(len(cnt) / max(len(code_counter), 1)),
            "entropy_nats": ent,
            "entropy_max_nats": float(np.log(max(len(cnt), 1))),
            "top5_share": float(np.sort(p)[::-1][:5].sum()),
        }
        log(f"  最终码分布：用了 {res['codebook']['n_codes_used']}/"
            f"{res['codebook']['n_codes_total']} 个码 "
            f"({res['codebook']['utilization']:.1%}) | 熵 {ent:.2f} nats "
            f"(上限 {res['codebook']['entropy_max_nats']:.2f}) | "
            f"前 5 码占 {res['codebook']['top5_share']:.1%}")
    # 各级码本的利用率（`cluster_size` 是一手数据，不用从码号频率反推）
    for li, nm, w, cs in per_level_cb:
        if cs is None:
            continue
        c = cs.detach().cpu().numpy().astype(np.float64)
        used = int((c > 0).sum())
        tot = max(c.sum(), 1e-9)
        p = c[c > 0] / tot
        ent = float(-(p * np.log(p)).sum())
        res[f"codebook_L{li}"] = {
            "n_codes": int(len(c)), "used": used,
            "utilization": used / max(len(c), 1),
            "entropy_nats": ent,
            "entropy_max_nats": float(np.log(max(used, 2))),
            "top5_share": float(np.sort(p)[::-1][:5].sum()),
        }
        log(f"  L{li} 码本：{used}/{len(c)} 个码被用 "
            f"({res[f'codebook_L{li}']['utilization']:.1%}) | 熵 {ent:.2f} nats "
            f"(上限 {res[f'codebook_L{li}']['entropy_max_nats']:.2f}) | "
            f"前 5 码占 {res[f'codebook_L{li}']['top5_share']:.1%}")
    del eeg
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1,2,3")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    import torch
    import registry as REG

    REG.ensure_braindecode_version("1.8.1.dev184031250")
    import braindecode.models as bdmodels

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fte = torch.load(FEAT_TEST, map_location="cpu", weights_only=False)
    IMG_test = fte["img_features"].numpy().astype(np.float64)

    h = REG.load_base_model(MODEL, models_dir=MODELS_DIR, braindecode_models=bdmodels,
                            spec=BS.BASE_SPECS.get(MODEL), verbose=False)
    if not h.ok:
        log(f"✗ {MODEL} 装载失败：{h.error[:300]}")
        raise SystemExit(1)
    log(f"✓ {MODEL} 装载成功（{h.n_params:,} 参数）")

    sids = [int(s) for s in args.subjects.split(",") if s.strip()]
    report = {"model": MODEL, "subjects": {}}
    for sid in sids:
        t0 = time.time()
        log(f"\n=== sub-{sid:02d} ===")
        try:
            rec = run_subject(sid, IMG_test, h, args)
        except Exception as e:
            rec = {"subject": f"sub-{sid:02d}", "error": f"{type(e).__name__}: {e}"[:300],
                   "trace": traceback.format_exc()[-600:]}
            log(f"  ✗ {rec['error']}")
        rec["elapsed_s"] = round(time.time() - t0, 1)
        report["subjects"][f"sub-{sid:02d}"] = rec

    (out_dir / "vq_readout_calib.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"\n结果写入 {out_dir}/vq_readout_calib.json")


if __name__ == "__main__":
    main()
