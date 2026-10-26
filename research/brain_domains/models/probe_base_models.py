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
# ⚠️ 类名随版本变过：1.8.1 里是 `Labram`（不是 `LaBraM`）；
#    `MIRepNet` / `BrainTokenizer` / `BrainOmni` 只在 **dev 版**里有
#    （PyPI 有 1.8.1.dev* 序列）⇒ 见 ensure_braindecode 的 --pre 回退。
CLASS_CANDIDATES = {
    "cbramod": ["CBraMod"],
    "mirepnet": ["MIRepNet", "MIRepnet"],
    "eegdino-small": ["EEGDINO"],
    "biot": ["BIOT"],
    "labram": ["Labram", "LaBraM", "LabRAM"],
    "signal-jepa": ["SignalJEPA", "SignalJepa", "SignalJEPA_PreLocal"],
    "braintokenizer": ["BrainTokenizer", "BrainTok"],
}
ORDER = ["cbramod", "mirepnet", "eegdino-small", "biot",
         "labram", "signal-jepa", "braintokenizer"]

# 输入规格候选（config 里 n_chans/n_times 常为 null ⇒ 只能试）
# 优先用 config，其次用模型自己的 chs_info 条数，最后试常见的。
SPEC_FALLBACK_CHANS = [63, 64, 62, 45, 19, 32, 128, 23]
SPEC_FALLBACK_TIMES = [250, 1000, 512, 256, 128, 3000, 200, 500, 100]

# PyPI 上 braindecode 的 dev 版（按上传日期从新到旧）。
# ⚠️ `pip install --pre --upgrade` **不会**升级到 dev 版（实测装完仍是 1.8.1），
#    必须点名版本号。
BD_DEV_VERSIONS = [
    "1.8.1.dev184031250",   # 2026-10-16
    "1.8.1.dev183670874",   # 2026-10-16
    "1.8.1.dev184307392",   # 2026-10-11
    "1.8.1.dev184305395",   # 2026-10-11
]


def log(*a):
    print(*a, flush=True)


def sh(cmd, timeout=900):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def probe_classes_subprocess():
    """在**全新进程**里查 braindecode 的版本与可用类。

    ⚠️ 不可用 `importlib.reload` 原地刷新：pip `--force-reinstall` 换掉了磁盘上的
    文件，但 `braindecode` / `braindecode.models` 已在内存里 ⇒ reload 拿到的是
    **陈旧模块**（实测：装完 dev 版后 7 个类全部"消失"，比不装还糟）。
    只有开新进程才是干净的。
    """
    import subprocess as _sp
    code = (
        "import json, braindecode, braindecode.models as M\n"
        f"cand = {CLASS_CANDIDATES!r}\n"
        "print('@@JSON@@' + json.dumps({"
        "'version': getattr(braindecode, '__version__', '?'),"
        "'have': {n: [x for x in cs if hasattr(M, x)] for n, cs in cand.items()}"
        "}))\n"
    )
    try:
        r = _sp.run([sys.executable, "-c", code], capture_output=True,
                    text=True, timeout=240)
    except Exception as e:
        return {"version": "?", "have": {}, "err": repr(e)[:200]}
    out = r.stdout or ""
    i = out.find("@@JSON@@")
    if i < 0:
        return {"version": "?", "have": {},
                "err": (r.stderr or out)[-200:]}
    try:
        return json.loads(out[i + 8:].strip())
    except Exception as e:
        return {"version": "?", "have": {}, "err": repr(e)[:200]}


def ensure_braindecode():
    """装 braindecode（缺类时试 dev 版 / GitHub master）。

    为什么可能要非稳定版：`MIRepNet` / `BrainTokenizer` / `BrainOmni`
    在 PyPI 稳定版 1.8.1 里**不存在**（137 个类里没有它们），
    而桶里 braintokenizer 的 config 写着 braindecode_version="1.8.1" ——
    说明转换时用的就是"版本串仍是 1.8.1"的构建（1.8.1 发布之后、1.8.2 之前的 master），
    所以候选里必须有 **GitHub master**。

    ⚠️ `pip install --pre --upgrade` **不会**升级到 dev 版（实测装完仍是 1.8.1），
       必须点名版本号。
    ⚠️ 缺类**不中断**整体探测 —— 能装几个是几个，缺的要如实记账。
    """
    info = {}

    def installed():
        try:
            import braindecode
            return getattr(braindecode, "__version__", "?")
        except Exception:
            return None

    if installed() is None:
        rc, out = sh(f"{sys.executable} -m pip install --quiet 'braindecode[hub]'")
        info["pip_rc"] = rc
    probe = probe_classes_subprocess()
    info["version"] = probe.get("version") or installed()
    have = {n: v for n, v in (probe.get("have") or {}).items() if v}
    missing = [n for n in ORDER if not have.get(n)]
    info["missing_classes"] = missing
    info["ok"] = bool(have)
    if not missing:
        log(f"  braindecode {info['version']}：7 个类齐全")
        return info, True

    log(f"  稳定版 {info['version']} 缺类：{missing} ⇒ 试 dev 版 / master")
    candidates = [f"braindecode[hub]=={v}" for v in BD_DEV_VERSIONS]
    candidates.append("braindecode[hub] @ git+https://github.com/braindecode/braindecode.git")
    for spec in candidates:
        # ⚠️ `--no-deps`：依赖（torch/mne/…）在装稳定版时已经就位了。
        #    加 `--force-reinstall` 而不加 `--no-deps` ⇒ 每次都把 torch（~2 GB）
        #    重下一遍，一个 dev 版要好几分钟。
        rc, out = sh(f'{sys.executable} -m pip install --quiet --upgrade '
                     f'--no-deps --force-reinstall "{spec}"', timeout=900)
        p2 = probe_classes_subprocess()
        have2 = {n: v for n, v in (p2.get("have") or {}).items() if v}
        m2 = [n for n in ORDER if not have2.get(n)]
        log(f"  试 {spec.split('==')[-1][:44]}：rc={rc} 版本={p2.get('version')} "
            f"仍缺={m2}")
        info.setdefault("install_tries", []).append(
            {"spec": spec, "rc": rc, "version": p2.get("version"),
             "missing": m2, "stderr": (out or "")[-200:]})
        if len(m2) < len(missing):
            info["version"] = p2.get("version")
            info["missing_classes"] = m2
            missing = m2
            if not m2:
                break
    if missing:
        log(f"  ⇒ 仍缺：{missing}（这些基模本轮不参与表）")
    return info, True


def load_state_dict_of(d: Path):
    """读桶内权重（safetensors 优先，其次 pytorch_model.bin）。"""
    import torch
    st, bt = d / "model.safetensors", d / "pytorch_model.bin"
    if st.exists():
        try:
            from safetensors.torch import load_file
            return load_file(str(st))
        except Exception:
            pass
    if bt.exists():
        obj = torch.load(str(bt), map_location="cpu", weights_only=False)
        return obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    return None


def shape_summary(sd, n=40):
    """权重键 + 形状（前 n 个）—— 用来反推输入规格。"""
    if not sd:
        return []
    out = []
    for i, (k, v) in enumerate(sd.items()):
        if i >= n:
            break
        out.append(f"{k}:{list(v.shape) if hasattr(v, 'shape') else '?'}")
    return out


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
    # braindecode 模型对象上的 chs_info（mne 风格 dict 列表）——**期望通道名**的权威来源
    ci = _safe_getattr(model, "chs_info")
    if isinstance(ci, (list, tuple)) and ci:
        names = []
        for c in ci:
            if isinstance(c, dict):
                names.append(c.get("ch_name") or c.get("name"))
            else:
                names.append(str(c))
        names = [n for n in names if n]
        if names:
            out["chs_info_names"] = names
    return out


def _count_params(model):
    """统计参数量，跳过未初始化的（LazyModule）参数。

    ⚠️ braindecode 的一些模型用 `torch.nn.LazyLinear`：
    权重在**第一次前向**之前是 UninitializedParameter，
    直接 `p.numel()` 会抛 ValueError（"Attempted to use an uninitialized parameter"）。
    所以参数量必须在**前向之后**统计，或者跳过未初始化的参数。
    """
    n, lazy = 0, 0
    for p in model.parameters():
        try:
            n += p.numel()
        except Exception:
            lazy += 1
    return n, lazy


def forward_test(model, n_ch, n_t, bs=8):
    """跑一次前向。返回 (ok, out_shape 或错误)。

    ⚠️ **判据是「前向成功」，不是「装载成功」**：
    signal-jepa 的 `from_pretrained` 会成功（打印 "Loading weights from local
    directory"），但用错的时间轴一前向就炸 ⇒ 只看装载会把坏模型当成好的。
    """
    import torch
    try:
        model.eval()
        x = torch.zeros(int(bs), int(n_ch), int(n_t))
        with torch.no_grad():
            y = model(x)
        shape = list(y.shape) if hasattr(y, "shape") else str(type(y))
        return True, shape
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"[:220]


# 各基模的**预训练采样率**（来自模型卡 / braindecode 自己的警告）。
# ⚠️ eegdino 在 200 Hz 预训练，传 250 会触发 braindecode 的警告
#    "EEG-DINO was pretrained at 200 Hz" ⇒ 候选里必须把 200 排在前面。
MODEL_SFREQ_HINT = {
    "eegdino-small": 200.0,
    "biot": 200.0,
    "signal-jepa": 128.0,
    "braintokenizer": 256.0,
    "labram": 250.0,
    "mirepnet": 250.0,
    "cbramod": 250.0,
}
SFREQ_CANDIDATES = [250.0, 200.0, 128.0, 256.0, 1000.0]


def spec_candidates(cfg, name):
    """(n_chans, n_times, sfreq) 候选序列（先 config/提示，再常见值）。"""
    out = []

    def add(c, t, s):
        if (isinstance(c, int) and c > 0 and isinstance(t, int) and t > 0
                and s):
            if (c, t, s) not in out:
                out.append((c, t, s))

    nci = len(cfg["chs_info"]) if cfg.get("chs_info") else None
    hint = MODEL_SFREQ_HINT.get(name)
    sfs = []
    for s in ([cfg.get("sfreq")] + ([hint] if hint else []) + SFREQ_CANDIDATES):
        if isinstance(s, (int, float)) and s not in sfs:
            sfs.append(float(s))
    chans = []
    for c in [cfg.get("n_chans"), nci] + SPEC_FALLBACK_CHANS:
        if isinstance(c, int) and c > 0 and c not in chans:
            chans.append(c)
    times = []
    for t in [cfg.get("n_times")] + SPEC_FALLBACK_TIMES + [4352, 272, 1280]:
        if isinstance(t, int) and t > 0 and t not in times:
            times.append(t)
    # config 自己的组合最先试（它最可能是对的）
    add(cfg.get("n_chans"), cfg.get("n_times"), cfg.get("sfreq"))
    add(nci, cfg.get("n_times"), hint)
    for s in sfs:
        for c in chans:
            for t in times:
                add(c, t, s)
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

    cfg = rec["config"]
    cands = spec_candidates(cfg, name)
    rec["spec_candidates_tried"] = []
    chs_info = cfg.get("chs_info")

    # --- 逐个规格试：**装载 + 前向都成功**才算数 ---
    model, chosen, errs = None, None, []
    for (nc, nt, sf) in cands:
        kw = {"n_outputs": 2, "n_chans": nc, "n_times": nt, "sfreq": sf}
        # 有些模型要的是 chs_info 而不是 n_chans（braintokenizer: "chs_info not specified"）
        if chs_info and nc == len(chs_info):
            kw["chs_info"] = chs_info
        try:
            m = cls.from_pretrained(str(d), **kw)
        except TypeError:
            try:
                m = cls.from_pretrained(str(d), n_outputs=2)
            except Exception as e:
                errs.append(f"({nc},{nt},{sf}) from_pretrained: {type(e).__name__}: {e}"[:190])
                continue
        except Exception as e:
            errs.append(f"({nc},{nt},{sf}) from_pretrained: {type(e).__name__}: {e}"[:190])
            continue
        ok, info = forward_test(m, nc, nt)
        rec["spec_candidates_tried"].append({"n_chans": nc, "n_times": nt, "sfreq": sf,
                                             "forward_ok": ok, "info": str(info)[:120]})
        if ok:
            model, chosen = m, (nc, nt, sf)
            rec["load_path"] = (f"from_pretrained(n_chans={nc},n_times={nt},sfreq={sf})")
            rec["spec_search"] = {"n_chans": nc, "n_times": nt, "sfreq": sf}
            break
        errs.append(f"({nc},{nt},{sf}) 前向失败: {info}")
    if model is None:
        rec["error"] = ("所有规格都失败：" + " | ".join(errs[:6]))[:700]
        rec["trace"] = traceback.format_exc()[-1000:]
        return rec

    nc, nt, sf = chosen
    rec["ok"] = True
    rec["meta"] = model_meta(model)
    # 模型**期望的通道名**（表评测的 domain adapter 要用）
    exp_ch = (rec["meta"].get("chs_info_names")
              or rec["meta"].get("ch_names")
              or (rec["meta"].get("model_info") or {}).get("ch_names"))
    if not exp_ch and chs_info:
        exp_ch = [c.get("ch_name") for c in chs_info if isinstance(c, dict)]
    rec["expected_chans"] = list(exp_ch) if exp_ch else None
    rec["expected_domain"] = {"n_chans": nc, "n_times": nt, "sfreq": sf,
                              "n_expected_chans": len(exp_ch) if exp_ch else None}

    # --- 前向 + 吞吐 ---
    try:
        import torch
        fwd = {"input_ch": int(nc), "input_times": int(nt), "sfreq": sf,
               "batches": {}}
        for bs in (8, 32):
            x = torch.zeros(bs, int(nc), int(nt))
            try:
                with torch.no_grad():
                    model(x)                       # 预热（同时初始化 LazyModule）
                    ts = []
                    for _ in range(3):
                        t0 = time.time()
                        y = model(x)
                        ts.append(time.time() - t0)
                    ms = sorted(ts)[len(ts) // 2] / bs * 1000
                fwd["batches"][str(bs)] = {
                    "ms_per_trial": round(ms, 3),
                    "runs_s": [round(t, 4) for t in ts],
                }
                if bs == 8:
                    fwd["out_shape"] = (list(y.shape) if hasattr(y, "shape")
                                        else str(type(y)))
            except Exception as e:
                fwd["batches"][str(bs)] = {"error": f"{type(e).__name__}: {e}"[:200]}
        rec["forward"] = fwd
    except Exception as e:
        rec["forward_error"] = f"{type(e).__name__}: {e}"[:400]

    # 参数量与层清单放在**前向之后**：
    # braindecode 的一些模型用 LazyLinear，第一次前向前参数是未初始化的，
    # 直接 numel() 会抛 "Attempted to use an uninitialized parameter"。
    n_par, n_lazy = _count_params(model)
    rec["n_params"] = n_par
    rec["n_lazy_params"] = n_lazy
    rec["layers"] = layer_inventory(model)

    b8 = (rec.get("forward") or {}).get("batches", {}).get("8", {})
    b32 = (rec.get("forward") or {}).get("batches", {}).get("32", {})
    ms8 = b8.get("ms_per_trial")
    ms32 = b32.get("ms_per_trial")
    rec["ms_per_trial"] = ms32 or ms8
    log(f"  ✓ {name}: 参数 {rec['n_params']:,}"
        + (f"（未初始化 {n_lazy}）" if n_lazy else "")
        + f" | 输入 ({nc},{nt})@{sf}"
        + f" | 前向 {(rec.get('forward') or {}).get('out_shape')}"
        + f" | bs8 {ms8} ms/试次 | bs32 {ms32} ms/试次 | {rec['load_path']}")
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
    ap.add_argument("--flavor", type=str, default="unknown",
                    help="当前档位名（写进报告，便于按档位汇总吞吐）")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    out_dir = Path(args.out) if args.out else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    install, ok = ensure_braindecode()
    log(f"braindecode: {install}")
    report = {"install": install, "flavor": args.flavor, "models": {}}
    if ok:
        import braindecode.models as M
        allc = sorted(k for k in dir(M) if not k.startswith("_"))
        report["all_classes"] = allc
        log(f"braindecode.models 类数：{len(allc)}")
        # 把类名全打出来：候选名对不上时，靠这份清单改名（比猜快）
        log("ALL_CLASSES: " + ", ".join(allc))
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
                if rec.get("from_pretrained_errors"):
                    for e in rec["from_pretrained_errors"]:
                        log(f"     · {e}")
                log(f"     · config: {json.dumps(rec.get('config', {}), ensure_ascii=False)[:600]}")
            (out_dir / "probe.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8")
    else:
        report["fatal"] = "braindecode 不可用"

    log("\n===== THINGS-EEG 通道名 =====")
    report["things_channels"] = probe_things_channels()
    log(json.dumps(report["things_channels"], ensure_ascii=False)[:1200])
    (out_dir / "probe.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    ok_models = [n for n, r in report["models"].items() if r.get("ok")]
    log("\n" + "=" * 70)
    log(f"装载成功 {len(ok_models)}/{len(report['models'])} → {ok_models}")
    log(f"耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
