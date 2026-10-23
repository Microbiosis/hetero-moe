#!/usr/bin/env python3
"""probe_base_models.py v2 — 阶段 A′ 的门（第二轮探测）。

## v1 的结论

- ✅ **braindecode 1.8.1 装上了**（`pip install braindecode[hub]` 在 Job 里可行，PyPI 可达）
  ⇒ **R1 的阻塞解除**。
- ❌ v1 的类查找写错了（`getattr(module.__dict__, name)` 对 mappingproxy 无效），
  所以 7 个模型全部误报"找不到类"。实际上类都在。
- 💡 braindecode 自带 `InterpolatedBIOT / InterpolatedLaBraM / InterpolatedSignalJEPA /
  InterpolatedModel / InterpolatedBENDR / InterpolatedEEGPT` —— **通道不匹配容忍版**
  正是 eegdino 模型卡推荐的路线 ⇒ R2 有现成解。

## v2 要回答

1. 7 个基模能否**装上并跑通前向**（用 `from_pretrained(本地目录)`，不依赖 hub 网络）；
2. 每个基模的**期望通道名 / 采样率 / 窗口 / token 形状 / 层清单**；
3. **前向吞吐**（ms/试次）——用来算 CPU 优先能不能跑完整个表；
4. **THINGS-EEG 那 63 个通道叫什么**（原始 .pt 里读，云端读，不往本地拉）。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

MODELS_DIR = Path("/training/models")
THINGS_PT = Path("/training/eeg/things-eeg/Preprocessed_data_250Hz_whiten/sub-01/test.pt")
OUT_DIR = Path("/domains/runs/base-probe")

# 桶内目录名 → braindecode 类名候选
CLASS_CANDIDATES = {
    "cbramod": ["CBraMod"],
    "mirepnet": ["MIRepNet", "MIRepnet"],
    "eegdino-small": ["EEGDINO"],
    "biot": ["BIOT"],
    "labram": ["LaBraM", "LabRAM"],
    "signal-jepa": ["SignalJEPA", "SignalJepa"],
    "braintokenizer": ["BrainTokenizer", "BrainTok"],
}
ORDER = ["cbramod", "mirepnet", "eegdino-small", "biot",
         "labram", "signal-jepa", "braintokenizer"]


def log(*a):
    print(*a, flush=True)


def sh(cmd, timeout=900):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def ensure_braindecode():
    info = {}
    try:
        import braindecode
        info["version"] = getattr(braindecode, "__version__", "?")
        info["installed_before"] = True
        return info, True
    except Exception:
        pass
    rc, out = sh(f"{sys.executable} -m pip install --quiet 'braindecode[hub]'")
    info["pip_rc"] = rc
    try:
        import braindecode
        info["version"] = getattr(braindecode, "__version__", "?")
        info["ok"] = True
        return info, True
    except Exception as e:
        info["ok"] = False
        info["err"] = repr(e)[:300]
        log(f"✗ braindecode 安装失败：{e!r}")
        return info, False


def read_json(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def read_attribution(name):
    p = MODELS_DIR / name / "ATTRIBUTION.txt"
    if not p.exists():
        return {}
    d = {}
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("Source:"):
            d["source"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("license:"):
            d["license"] = line.split(":", 1)[1].strip()
    return d


def layer_inventory(model, max_items=120):
    inv = []
    for name, mod in model.named_modules():
        params = [{"p": pn, "shape": list(p.shape)}
                  for pn, p in mod.named_parameters(recurse=False)]
        entry = {"name": name, "type": type(mod).__name__}
        if params:
            entry["params"] = params[:6]
        for attr in ("nhead", "num_heads", "n_layer", "dim_feedforward", "d_model",
                     "num_layers", "dropout", "activation", "final_dim", "embed_dim",
                     "d_token", "n_chans", "n_times", "patch_size", "n_global_tokens"):
            v = getattr(mod, attr, None)
            if isinstance(v, (int, float, str, bool)):
                entry[attr] = v
        if params or len(entry) > 2:
            inv.append(entry)
        if len(inv) >= max_items:
            inv.append({"_truncated": True})
            break
    return inv


def _safe_getattr(obj, attr):
    """braindecode 的部分属性（如 sfreq）是**会抛异常**的 property，
    直接 getattr 会把整个探测跑挂 ⇒ 统一包起来。"""
    try:
        return getattr(obj, attr, None)
    except Exception:
        return None


def model_meta(model):
    out = {}
    mi = _safe_getattr(model, "model_info")
    if isinstance(mi, dict):
        out["model_info"] = {k: (list(v) if isinstance(v, (list, tuple)) else v)
                             for k, v in mi.items()}
    for attr in ("ch_names", "sfreq", "n_chans", "n_times", "input_window_seconds",
                 "embedding_dim", "final_dim"):
        v = _safe_getattr(model, attr)
        if v is not None:
            out[attr] = list(v) if isinstance(v, (list, tuple)) else v
    return out


def probe_one(name, M):
    rec = {"name": name, "ok": False}
    rec["config"] = read_json(MODELS_DIR / name / "config.json")
    rec["attribution"] = read_attribution(name)
    d = MODELS_DIR / name
    if not d.exists():
        rec["error"] = "桶内目录不存在"
        return rec

    cls = None
    for cand in CLASS_CANDIDATES.get(name, []):
        if hasattr(M, cand):
            cls = getattr(M, cand)
            rec["class"] = cand
            break
    if cls is None:
        rec["error"] = f"braindecode 里没有候选类 {CLASS_CANDIDATES.get(name)}"
        return rec

    # --- 路径 A：from_pretrained(本地目录)（自动处理 config→ctor 与权重）---
    # config 里 n_outputs 常为 null（如 cbramod）⇒ 补一个占位头（后面只要 token 表征）
    try:
        t0 = time.time()
        try:
            model = cls.from_pretrained(str(d), n_outputs=2)
            rec["n_outputs_patched"] = 2
        except TypeError:
            model = cls.from_pretrained(str(d))
        rec["load_path"] = "from_pretrained(local)"
        rec["load_s"] = round(time.time() - t0, 1)
        rec["ok"] = True
    except Exception as e:
        rec["from_pretrained_error"] = f"{type(e).__name__}: {e}"[:400]
        # --- 路径 B：手工 ctor + 手工权重 ---
        try:
            import inspect
            import torch
            sig = inspect.signature(cls.__init__)
            avail = set(sig.parameters) - {"self"}
            cfg = rec["config"]
            kwargs = {k: v for k, v in cfg.items() if k in avail}
            if "n_outputs" in avail and kwargs.get("n_outputs") is None:
                kwargs["n_outputs"] = 2      # 只要 token 表征，头是占位
            rec["ctor_kwargs"] = {k: (list(v) if isinstance(v, (list, tuple)) else v)
                                  for k, v in kwargs.items()}
            rec["ctor_ignored"] = sorted(k for k in cfg if k not in avail)
            model = cls(**kwargs)
            rec["load_path"] = "manual"
            sd = None
            st, bt = d / "model.safetensors", d / "pytorch_model.bin"
            if st.exists():
                try:
                    from safetensors.torch import load_file
                    sd = load_file(str(st))
                except Exception:
                    sd = None
            if sd is None and bt.exists():
                obj = torch.load(str(bt), map_location="cpu", weights_only=False)
                sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
            if sd is not None:
                clean = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
                miss, unexp = model.load_state_dict(clean, strict=False)
                rec["weights_report"] = {"missing": list(miss)[:10], "unexpected": list(unexp)[:10]}
            rec["ok"] = True
        except Exception as e2:
            rec["error"] = f"from_pretrained 失败({rec['from_pretrained_error']})；手工路径也失败：{e2!r}"
            rec["trace"] = traceback.format_exc()[-1200:]
            return rec

    rec["n_params"] = int(sum(p.numel() for p in model.parameters()))
    rec["meta"] = model_meta(model)
    rec["layers"] = layer_inventory(model)

    # --- 前向 + 吞吐 ---
    try:
        import torch
        cfg = rec["config"]
        n_ch = cfg.get("n_chans") or (rec["meta"].get("n_chans") or None)
        n_t = cfg.get("n_times") or (rec["meta"].get("n_times") or None)
        sf = cfg.get("sfreq") or rec["meta"].get("sfreq")
        # 配置没给就用模型的期望
        if not n_ch:
            ch = rec["meta"].get("ch_names")
            n_ch = len(ch) if ch else 32
        if not n_t:
            n_t = 1000
        model.eval()
        bs = 8
        x = torch.zeros(bs, int(n_ch), int(n_t))
        with torch.no_grad():
            model(x)          # 预热
            t0 = time.time()
            y = model(x)
            dt = time.time() - t0
        rec["forward"] = {
            "input": [bs, int(n_ch), int(n_t)], "sfreq": sf,
            "out_shape": list(y.shape) if hasattr(y, "shape") else str(type(y)),
            "ms_per_trial": round(dt / bs * 1000, 2),
        }
        log(f"  ✓ {name}: 参数 {rec['n_params']:,} | 前向 {rec['forward']['out_shape']} "
            f"| {rec['forward']['ms_per_trial']} ms/试次 | 载入 {rec['load_path']}")
    except Exception as e:
        rec["forward_error"] = f"{type(e).__name__}: {e}"[:400]
        log(f"  ✓ {name} 已装载但前向失败：{e}")
    return rec


def probe_things_channels():
    """云端读原始 .pt，取 THINGS-EEG 的通道名（不往本地拉数据）。"""
    out = {"path": str(THINGS_PT)}
    if not THINGS_PT.exists():
        out["error"] = "原始 .pt 不存在"
        return out
    try:
        import torch
        d = torch.load(THINGS_PT, map_location="cpu", weights_only=False)
        out["keys"] = [str(k) for k in d.keys()]
        for k in d.keys():
            v = d[k]
            if hasattr(v, "shape"):
                out[f"shape_{k}"] = list(v.shape)
            elif isinstance(v, (list, tuple)) and len(v) and isinstance(v[0], str):
                out[f"names_{k}"] = list(v)[:80]
        # 通道名可能藏在 montage / ch_names / info 里
        for k in d.keys():
            if isinstance(d[k], dict):
                out[f"dict_keys_{k}"] = list(d[k].keys())[:30]
    except Exception as e:
        out["error"] = repr(e)[:300]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default="")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    install, ok = ensure_braindecode()
    log(f"braindecode: {install}")
    report = {"install": install, "models": {}}
    if ok:
        import braindecode.models as M
        allc = sorted(k for k in dir(M) if not k.startswith("_"))
        report["all_classes"] = allc
        log(f"braindecode.models 类数：{len(allc)}")
        for name in ORDER:
            if args.only and name not in args.only.split(","):
                continue
            log(f"\n===== 探测 {name} =====")
            try:
                rec = probe_one(name, M)
            except Exception as e:      # 单个模型崩了不能带倒整轮
                rec = {"name": name, "ok": False,
                       "error": f"探测过程异常：{type(e).__name__}: {e}"[:400],
                       "trace": traceback.format_exc()[-1500:]}
                log(f"  ✗ {rec['error']}")
            report["models"][name] = rec
            if not rec.get("ok"):
                log(f"  ✗ {str(rec.get('error'))[:300]}")
            (OUT_DIR / "probe.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8")
    else:
        report["fatal"] = "braindecode 不可用"

    log("\n===== THINGS-EEG 通道名 =====")
    report["things_channels"] = probe_things_channels()
    log(json.dumps(report["things_channels"], ensure_ascii=False)[:1200])
    (OUT_DIR / "probe.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    ok_models = [n for n, r in report["models"].items() if r.get("ok")]
    log("\n" + "=" * 70)
    log(f"装载成功 {len(ok_models)}/{len(report['models'])} → {ok_models}")
    log(f"耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
