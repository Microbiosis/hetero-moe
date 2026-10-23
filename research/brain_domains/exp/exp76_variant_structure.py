#!/usr/bin/env python3
"""exp76_variant_structure.py — 阶段 D 的**结构勘查**（零训练，前置）。

## 为什么必须先跑这一步

教训 #24：**先验证结构是否存在，再构建结构**。
本项目的组装器（`assembler.py`）走的是「拆成零件 → 重建整条链」的路子，
但它有一条致命限制已经写在代码注释里：

> CBraMod 的 Criss-Cross 是**卷积**实现，我们的 `CrissCross2D` 是**注意力**实现
> —— 结构不同、权重**不可搬**，只能标 `arch-only`。

⇒ 如果拿 cbramod 当宿主去「重建整条链」，产出的其实是一个
**随机初始化的 transformer**（只有 patch embed 是真的）⇒ 跑 RSA 必然接近 0，
测到的是「随机初始化的 Transformer 没有脑对齐」，不是「重组有没有用」。

所以本脚本先把**真实结构**取出来，让「能不能嫁接」变成一个**本地可判定**的问题：

1. 每个模型的 block 列表（名字 / mixer 类 / FFN 类）；
2. 每个 block 的 mixer / FFN **逐键权重形状**；
3. 每个 block 的**输出 token 形状** (B, N, D) ⇒ 深度与 token 维；
4. patch embed（token 化入口）的输出形状。

拿到这些之后，嫁接分三种结论，每种都有明确处置：

| 结论 | 含义 | 处置 |
|---|---|---|
| **形状兼容** | 供体与宿主的对应权重**逐键同形** | 真嫁接：把供体权重搬进宿主槽位（零训练、逐键可核） |
| **需投影** | 维度不同但结构同族 | 零训练下**不做**（学习投影是随机初始化 ⇒ 白搬）⇒ 记为负结果 |
| **结构不同** | 供体是 conv 注意力、宿主是 MHA 之类 | 不可搬 ⇒ 走**表征级拼接**（D1 路线） |

## 输出

- 日志：每个模型一行 `###MODEL### {json}`（便于从 `hf jobs logs` 解析）；
- 文件：`/domains/runs/illusionbrain-align/structure/<version>.json`。

## 用法

云端：

    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp76_variant_structure.py \
      -v hf://buckets/illusion-brain/training-data:/training:ro \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor cpu-upgrade --timeout 30m -d --bd-version stable

本地自检（合成数据，不碰网络）：

    python exp76_variant_structure.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
# models 包的位置：本地 = brain_domains/models；云端 = /domains/scripts/models
_PKG_CANDIDATES = [EXP_DIR / "models", EXP_DIR.parent / "models", EXP_DIR]
PKG_DIR = next((c for c in _PKG_CANDIDATES if (c / "registry.py").exists()),
               _PKG_CANDIDATES[0])
for _p in (str(EXP_DIR), str(PKG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import base_specs as BS            # noqa: E402
import domain_adapters as DA       # noqa: E402

MODELS_DIR = Path("/training/models")
OUT_ROOT = Path("/domains/runs/illusionbrain-align/structure")

# dev 版 braindecode 的完整版本串。
# ⚠️ `ensure_braindecode_version` 收的是**完整版本串**，不是 "dev" 这个标签 ——
# 传 "dev" 会让 pip 去装 `braindecode[hub]==dev`（不存在）⇒ 直接 RuntimeError。
BD_DEV_VERSION = "1.8.1.dev184031250"


def _bd_install_target(v):
    """把版本标签翻译成 `ensure_braindecode_version` 要的**完整版本串**。"""
    v = (v or "stable").strip()
    if v == "stable":
        return "stable"
    if v == "dev":
        return BD_DEV_VERSION
    return v


def _ver_tag(v):
    """模型选择用的版本标签（规格表里是 "stable" / "dev"）。"""
    return "stable" if (v or "stable").strip() == "stable" else "dev"


def log(*a):
    print(*a, flush=True)


def _as_tokens(t):
    """把各种 block 输出统一成 (B, N, D) 的 tensor（拿不到就 None）。"""
    import torch
    if t is None:
        return None
    if hasattr(t, "tokens"):          # TokenBatch
        t = t.tokens
    if isinstance(t, (tuple, list)):
        t = t[0] if t else None
    if t is None:
        return None
    if not torch.is_tensor(t):
        return None
    if t.ndim == 3:
        return t
    if t.ndim == 2:
        return t.unsqueeze(1)
    if t.ndim == 4:
        # ⚠️ 4 维统一交给 registry 的 `normalize_token_tensor`：
        #    它的约定是**通道在前** (B, D, H, W) ⇒ 折成 (B, H*W, D)。
        #    （CBraMod 的 block 输出就是 (B, D, H, W)；exp75 用的也是这个函数
        #      ⇒ 这里必须与它同口径，不能自己按通道在后折。）
        import registry as REG
        return REG.normalize_token_tensor(t)
    return None


def _shapes(mod):
    """模块的权重形状表 {键: [shape]}（只取浮点张量，避免 BN 的 int buffer）。"""
    import torch
    out = {}
    for k, v in mod.state_dict().items():
        if torch.is_tensor(v) and v.is_floating_point():
            out[k] = list(v.shape)
    return out


def _cls(mod):
    return type(mod).__name__


def _find_tokenizer2(model, x):
    """找 token 化入口（patch embed），**接受 3 维与 4 维输出**。

    为什么不能直接用 assembler 的 `find_tokenizer`：它只收 `t.ndim == 3`，
    而 **CBraMod 的 patch embed 输出是 4 维**（B, rows, cols, D）的二维网格
    ⇒ 那个函数在 cbramod 上**必然返回 None**（实测 d_model=None）。
    这里把 4 维也认下来，并按同样的「谁能独立吃原始 EEG」判据筛。
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
        try:
            with torch.no_grad():
                out = mod(x)
        except Exception:
            continue
        t = out[0] if isinstance(out, (tuple, list)) else out
        if torch.is_tensor(t) and t.ndim in (3, 4) and t.shape[1] > 1:
            return nm, mod
    return None, None


LAYERISH = re.compile(r"(layers?\.\d+|blocks?\.\d+|encoder\.layer\d+|transformer\.\d+)", re.I)


def _layer_detail(model):
    """每个「像层」的模块：类名 + 直接子模块的类名与权重形状。

    这是**嫁接可行性**的关键数据：要知道宿主的槽位是
    「卷积注意力」还是「MultiheadAttention」、FFN 的 Linear 形状是多少。
    """
    import torch
    out = []
    for nm, mod in model.named_modules():
        if not LAYERISH.search(nm):
            continue
        kids = []
        for cn, cm in mod.named_children():
            sh = _shapes(cm)
            kids.append({"name": cn, "cls": type(cm).__name__,
                         "n_params": int(sum(p.numel() for p in cm.parameters())),
                         "shapes": sh})
        out.append({"name": nm, "cls": type(mod).__name__,
                    "n_params": int(sum(p.numel() for p in mod.parameters())),
                    "children": kids})
    return out


def _module_tree(model):
    """完整模块树 [名字, 类名]（嫁接要精确定位路径，光有 block 列表不够）。"""
    return [[nm, type(m).__name__] for nm, m in model.named_modules() if nm]


def probe_model(name, handle, have_chans=None):
    """勘查一个基模：block 结构 + token 形状 + 权重形状 + 模块树。"""
    import torch
    import registry as REG
    import assembler as ASM

    spec = handle.spec
    out = {"model": name, "ok": False}

    # --- 1. 造一个真实形状的输入（走正式的域适配路径）---
    dummy = np.zeros((1, 63, 250), dtype=np.float32)
    try:
        adapted, info = DA.adapt(dummy, spec, have_chans or BS.THINGS_EEG_CHANS)
    except Exception as e:
        out["error"] = f"域适配失败：{type(e).__name__}: {e}"
        return out
    out["adapted_shape"] = list(adapted.shape)
    out["adapt"] = info.to_dict()

    mdt = next((p.dtype for p in handle.model.parameters()), None)
    x = torch.from_numpy(np.ascontiguousarray(adapted)).to(dtype=mdt or torch.float32)

    # --- 2. 找 block 与 token 化入口 ---
    blocks = ASM.find_block_modules(handle.model)
    out["n_blocks"] = len(blocks)
    out["block_names"] = [b["name"] for b in blocks]

    tok_name, tok_mod = _find_tokenizer2(handle.model, x)
    out["tokenizer"] = {"name": tok_name, "cls": _cls(tok_mod) if tok_mod else None}

    # --- 3. 挂 hook 收每个 block / token 化入口 的输出形状 ---
    got = {}

    def mk(nm):
        def hook(mod, inp, o):
            t = _as_tokens(o)
            if t is not None:
                got[nm] = list(t.shape)
        return hook

    handles = []
    for b in blocks:
        handles.append(b["module"].register_forward_hook(mk(b["name"])))
    if tok_mod is not None:
        handles.append(tok_mod.register_forward_hook(mk("__tokenizer__")))

    try:
        with torch.no_grad():
            handle.model(x)
    except Exception as e:
        out["error"] = f"前向失败：{type(e).__name__}: {e}"
        out["trace"] = traceback.format_exc()[-600:]
        return out
    finally:
        for h in handles:
            h.remove()

    # 最终输出（表里那一行用的表征）
    try:
        with torch.no_grad():
            o = handle.model(x)
        ft, _how = REG.tokens_from_output(o, 2)
        if ft is not None:
            got["__final__"] = list(ft.shape)
    except Exception:
        pass

    out["tokenizer"]["out_shape"] = got.get("__tokenizer__")
    out["final_shape"] = got.get("__final__")
    out["d_model"] = (out["tokenizer"]["out_shape"] or [None, None, None])[-1]
    out["n_tokens"] = (out["tokenizer"]["out_shape"] or [None, None])[-2]
    out["final_dim"] = (out["final_shape"] or [None, None, None])[-1]

    # --- 4. 逐 block 的 mixer / FFN 结构与权重形状 ---
    binfo = []
    for i, b in enumerate(blocks):
        attn_name, attn_mod = b["attn"]
        ffn_name, ffn_mod = b["ffn"]
        binfo.append({
            "index": i,
            "name": b["name"],
            "mixer": {
                "path": f"{b['name']}.{attn_name}",
                "cls": _cls(attn_mod),
                "is_mha": isinstance(attn_mod, torch.nn.MultiheadAttention),
                "shapes": _shapes(attn_mod),
            },
            "ffn": {
                "path": f"{b['name']}.{ffn_name}",
                "cls": _cls(ffn_mod),
                "shapes": _shapes(ffn_mod),
            },
            "out_shape": got.get(b["name"]),
        })
    out["blocks"] = binfo

    # --- 5. 「像层」的模块明细 + 完整模块树 ---
    out["layer_detail"] = _layer_detail(handle.model)
    out["module_tree"] = _module_tree(handle.model)

    # --- 6. 布局推断 ---
    lay = REG.infer_layout(out["n_tokens"] or 0, handle.cfg, handle.meta, handle.model)
    out["layout"] = {"layout": list(lay["layout"]) if lay["layout"] else None,
                     "n_global": lay["n_global"],
                     "confidence": lay["confidence"], "reason": lay["reason"]}
    out["ok"] = True
    return out


def run(args):
    import registry as REG
    REG.ensure_braindecode_version(_bd_install_target(args.bd_version))
    try:
        import braindecode.models as bdmodels
    except Exception as e:
        log(f"✗ braindecode 不可用：{e!r}")
        return 1

    if args.models in ("all", "auto"):
        names = BS.models_for_version(_ver_tag(args.bd_version))
    else:
        names = [s.strip() for s in args.models.split(",") if s.strip()]
    log(f"本趟（braindecode {args.bd_version}）模型：{names}")

    results = []
    for nm in names:
        sp = BS.BASE_SPECS.get(nm)
        h = REG.load_base_model(nm, models_dir=MODELS_DIR, braindecode_models=bdmodels,
                                spec=sp, verbose=False)
        if not h.ok:
            rec = {"model": nm, "ok": False, "error": h.error}
            results.append(rec)
            log(f"###MODEL### {json.dumps(rec, ensure_ascii=False)}")
            continue
        try:
            rec = probe_model(nm, h)
        except Exception as e:
            rec = {"model": nm, "ok": False,
                   "error": f"{type(e).__name__}: {e}",
                   "trace": traceback.format_exc()[-600:]}
        rec["n_params"] = h.n_params
        rec["class"] = type(h.model).__name__
        rec["license"] = h.attribution.get("license")
        results.append(rec)
        log(f"###MODEL### {json.dumps(rec, ensure_ascii=False)}")

        if rec.get("ok"):
            log(f"✓ {nm:<16} {rec.get('class')} "
                f"blocks={rec.get('n_blocks')} "
                f"d_model={rec.get('d_model')} n_tokens={rec.get('n_tokens')} "
                f"layout={rec.get('layout', {}).get('layout')} "
                f"({rec.get('layout', {}).get('confidence')})")
        else:
            log(f"✗ {nm:<16} {rec.get('error')}"[:250])

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"bd_version": args.bd_version, "models": results},
                                indent=2, ensure_ascii=False, default=str),
                     encoding="utf-8")
        log(f"结构报告写入 {p}")
    return 0 if all(r.get("ok") for r in results) else 1


# ---------------------------------------------------------------------------
# 自测（合成模型，不碰网络）
# ---------------------------------------------------------------------------
def self_test():
    import torch
    from torch import nn
    import interfaces as IF
    import registry as REG

    log("=== 自测：结构勘查链路 ===")

    class PatchEmbed(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.proj = nn.Linear(9 * 25, d)

        def forward(self, x):
            B, C, T = x.shape
            p = x.unfold(-1, 25, 25).unfold(1, 9, 9)      # (B, cols, rows, 9, 25)
            p = p.permute(0, 2, 1, 3, 4).reshape(B, -1, 9 * 25)
            return self.proj(p)

    class Blk(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
            self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
            self.n1 = nn.LayerNorm(d)

        def forward(self, x):
            h = self.n1(x)
            a, _ = self.attn(h, h, h, need_weights=False)
            return x + a + self.ffn(h)

    class Fake(nn.Module):
        def __init__(self, d=32, L=2):
            super().__init__()
            self.patch = PatchEmbed(d)
            self.blocks = nn.ModuleList([Blk(d) for _ in range(L)])

        def forward(self, x):
            t = self.patch(x)
            for b in self.blocks:
                t = b(t)
            return {"embedding": t}

    class NS:
        pass
    ns = NS()
    ns.CBraMod = Fake

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "cbramod"
        d.mkdir()
        (d / "config.json").write_text(json.dumps(
            {"n_chans": 63, "n_times": 250, "patch_size": [9, 25], "d_patch": 32}),
            encoding="utf-8")
        (d / "ATTRIBUTION.txt").write_text("Source: synthetic\nLicense: MIT\n",
                                           encoding="utf-8")
        h = REG.load_base_model("cbramod", models_dir=Path(td), braindecode_models=ns)
        assert h.ok, h.error
        rec = probe_model("cbramod", h, BS.THINGS_EEG_CHANS)
    assert rec["ok"], rec.get("error")
    assert rec["n_blocks"] == 2, rec
    assert rec["d_model"] == 32, rec
    assert rec["blocks"][0]["mixer"]["is_mha"] is True
    assert rec["blocks"][0]["out_shape"][-1] == 32, rec["blocks"][0]
    log(f"  ✓ 勘查：blocks={rec['n_blocks']} d_model={rec['d_model']} "
        f"n_tokens={rec['n_tokens']}")
    log(f"  ✓ mixer 权重键：{list(rec['blocks'][0]['mixer']['shapes'])}")
    log("=== 自测通过 ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--bd-version", default="stable",
                    help="stable | dev（1.8.1.dev184031250）")
    ap.add_argument("--out", default="")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.out:
        args.out = str(OUT_ROOT / f"{args.bd_version}.json")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
