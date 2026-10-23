#!/usr/bin/env python3
"""exp86b_extract_donors.py — 把各基模的**组件**抽出来存进桶（组装的前置）。

## 为什么单独一步

组装时要混装**不同模型的**组件，而它们分属两个 braindecode 版本：
- **stable**：`signal-jepa`（注意力供体）、`biot`、`eegdino-small`
- **dev**：`mirepnet`（exp75 里层内语义最强的 FFN 就在这儿）

⇒ 同一进程装不了两个版本（`Interpolated*` 只在 stable、`MIRepNet`/`BrainTokenizer` 只在 dev）
⇒ 拆成「**按版本各抽一次 → 组件落桶**」，训练时**只读组件、完全不碰 braindecode**。

## 抽什么

| 组件 | 怎么取 | 落盘 |
|---|---|---|
| 注意力 | donor 模块的 `state_dict()`（`nn.MultiheadAttention`） | `attn_state`（逐键原样） |
| FFN | 在 donor 模块里找**一对互相转置的 Linear** (h,d)/(d,h) ⇒ 权重 + 偏置 | `ffn`（h, d, w1, b1, w2, b2） |

⇒ 抽出后**形状不变**（原样保存）⇒ 组装时能搬的搬、不能搬的插学习投影。

## 用法

    # stable 版供体
    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp86b_extract_donors.py \
      -v hf://buckets/illusion-brain/training-data:/training:ro \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor cpu-upgrade --timeout 30m -d --bd-version stable \
      --donors signal-jepa,biot,eegdino-small

    # dev 版供体
    ... --bd-version dev --donors mirepnet
"""

from __future__ import annotations

import argparse
import json
import os
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

import base_specs as BS            # noqa: E402

MODELS_DIR = Path("/training/models")
OUT_DIR = Path("/domains/runs/illusionbrain-align/donors")

# 供体 → 组件路径
ATTN_PATHS = {
    "signal-jepa": "transformer.encoder.layers.0.self_attn",
    "mirepnet": "transformer.0.0.fn.1",     # 备选（自定义 MHA，键名不同，慎用）
}
FFN_PATHS = {
    "mirepnet": "transformer.0.1.fn",
    "biot": "encoder.transformer.layers.layers.0.1.fn",
    "eegdino-small": "encoder_layers.0.mlp",
}


def log(*a):
    print(*a, flush=True)


def extract_attn(model, path):
    mod = model.get_submodule(path)
    sd = {k: v.detach().clone() for k, v in mod.state_dict().items()}
    shapes = {k: list(v.shape) for k, v in sd.items()}
    return {"state": sd, "shapes": shapes, "path": path}


def extract_ffn(model, path):
    mod = model.get_submodule(path)
    sd = mod.state_dict()
    mats = [(k, v) for k, v in sd.items()
            if v.ndim == 2 and k.endswith("weight")]
    for i, (ka, a) in enumerate(mats):
        for kb, b in mats[i + 1:]:
            if a.shape[0] == b.shape[1] and b.shape[0] == a.shape[1] \
                    and a.shape[0] != a.shape[1]:
                ba = sd.get(ka.replace("weight", "bias"))
                bb = sd.get(kb.replace("weight", "bias"))
                if ba is None or bb is None:
                    continue
                return {"h": int(a.shape[0]), "d": int(a.shape[1]),
                        "w1": a.detach().clone(), "b1": ba.detach().clone(),
                        "w2": b.detach().clone(), "b2": bb.detach().clone(),
                        "src": f"{path}:{ka}/{kb}",
                        "shapes": {ka: list(a.shape), kb: list(b.shape)}}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bd-version", default="stable",
                    help="stable | dev（决定能用哪些供体）")
    ap.add_argument("--donors", default="")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    import torch
    import registry as REG

    REG.ensure_braindecode_version(
        "stable" if args.bd_version == "stable" else BS.BD_DEV_VERSION)
    import braindecode.models as bdmodels

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = ([s.strip() for s in args.donors.split(",") if s.strip()]
             if args.donors else
             sorted(set(list(ATTN_PATHS) + list(FFN_PATHS))))
    log(f"braindecode {args.bd_version} | 供体 {names}")

    for nm in names:
        t0 = time.time()
        sp = BS.BASE_SPECS.get(nm)
        if sp and sp.get("bd_version", "stable") != args.bd_version:
            log(f"  跳过 {nm}：它属于 {sp.get('bd_version')} 版")
            continue
        h = REG.load_base_model(nm, models_dir=MODELS_DIR, braindecode_models=bdmodels,
                                spec=sp, verbose=False)
        if not h.ok:
            log(f"  ✗ {nm} 装载失败：{h.error[:200]}")
            continue
        payload = {"model": nm, "bd_version": args.bd_version}
        if nm in ATTN_PATHS:
            try:
                payload["attn"] = extract_attn(h.model, ATTN_PATHS[nm])
                log(f"  ✓ {nm} 注意力 {ATTN_PATHS[nm]} "
                    f"{payload['attn']['shapes']}")
            except Exception as e:
                log(f"  ✗ {nm} 注意力抽取失败：{e}")
        if nm in FFN_PATHS:
            try:
                f = extract_ffn(h.model, FFN_PATHS[nm])
                if f is None:
                    log(f"  ✗ {nm} 的 {FFN_PATHS[nm]} 里没找到 FFN 的 Linear 对")
                else:
                    payload["ffn"] = f
                    log(f"  ✓ {nm} FFN {f['src']}：{f['d']}→{f['h']}→{f['d']}")
            except Exception as e:
                log(f"  ✗ {nm} FFN 抽取失败：{e}")
        if len(payload) == 2:
            log(f"  · {nm} 没有可抽的组件，跳过")
            continue
        fp = out_dir / f"{nm}.pt"
        torch.save(payload, fp)
        log(f"  → {fp}（{fp.stat().st_size/1e6:.2f} MB，{time.time()-t0:.0f}s）")

    idx = out_dir / "index.json"
    rows = []
    for fp in sorted(out_dir.glob("*.pt")):
        d = torch.load(fp, map_location="cpu", weights_only=False)
        rows.append({"file": fp.name, "model": d.get("model"),
                     "has_attn": "attn" in d, "has_ffn": "ffn" in d,
                     "attn_shapes": (d.get("attn") or {}).get("shapes"),
                     "ffn_src": (d.get("ffn") or {}).get("src"),
                     "ffn_d": (d.get("ffn") or {}).get("d"),
                     "ffn_h": (d.get("ffn") or {}).get("h")})
    idx.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\n索引写入 {idx}")
    for r in rows:
        log(f"  {r['file']:<22} attn={r['has_attn']} ffn={r['has_ffn']} "
            f"{r['ffn_src'] or ''}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:
        log(f"✗ {type(e).__name__}: {e}")
        log(traceback.format_exc()[-1500:])
        raise SystemExit(1)
