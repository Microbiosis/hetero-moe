"""registry.py — 基模的「装载 → 出 token → 定布局 → 池化」统一入口。

## 为什么要有这一层

7 个基模的**接口各不相同**：

| 基模 | 期望输入 | 前向返回 | token 形态 |
|---|---|---|---|
| cbramod | (B, C, T) 可配 | dict / tensor | (B, N, D)，N 是**二维网格**展平 |
| mirepnet | (B, 45, 1000) @250Hz | tensor | (B, N, D) 或 (B, D) |
| eegdino-small | (B, 19, 1000) @200Hz | dict | 含 "embedding" / "last_hidden_state" |
| biot | (B, 18, 1000) @200Hz | tensor | (B, N, D)，含 cls token |
| labram | (B, C, 3000) | tensor | 层级输出 |
| signal-jepa | (B, C, T) @128Hz | dict | 预测器输出 |
| braintokenizer | (B, C, 512) @256Hz | token ids / tensor | 量化码 |

要「逐 block 拆开重组」，必须有**一个共同的抽取口径**，否则各模型的
表征没法放在同一张表里比较。这一层负责把它统一成：

    EEG (n, C, T) ──adapt──▶ (n, C', T') ──forward──▶ tokens (n, N, D)
                                              └─pool──▶ 特征 (n, D)

## 三条硬规则

1. **池化规则预先写死**：默认对**全部 token 取均值**（含全局 token）。
   改规则必须整表重算 —— 口径混用比数字错误更糟。
2. **布局必须带来源**：`(rows, cols)` 若是从 config 推出来的就是权威，
   若是猜的，必须标 `layout_confidence="guessed"`；**猜测布局不得用于二维 mixer**
   （CBraMod 的 Criss-Cross 一旦轴搞反，注意力就作用在错误的轴上）。
3. **抽不到 token 不许硬凑**：直接返回 `ok=False` 并记原因，
   宁可表里少一行，也不要拿一个来源不明的向量冒充表征。
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:                                    # 作为包导入：models.registry
    from .domain_adapters import DomainSpec, spec_from_config
except ImportError:                     # 作为脚本/平铺导入：registry
    from domain_adapters import DomainSpec, spec_from_config

# 桶内目录名 → braindecode 类名候选（与 probe_base_models.py 一致）
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

# 模型返回 dict 时，按这个顺序找 token 字段
TOKEN_KEY_PRIORITY = (
    "embedding", "embeddings", "last_hidden_state", "hidden_states",
    "features", "representations", "tokens", "sequence_output", "patch_tokens",
)


def _safe_getattr(obj, attr, default=None):
    """braindecode 部分属性（如 sfreq）是会抛异常的 property。"""
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


def read_json(p: Path) -> dict:
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def read_attribution(models_dir: Path, name: str) -> dict:
    p = models_dir / name / "ATTRIBUTION.txt"
    out = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s.lower().startswith("license:"):
            out["license"] = s.split(":", 1)[1].strip()
        elif s.lower().startswith("source:"):
            out["source"] = s.split(":", 1)[1].strip()
    return out


@dataclass
class BaseModelHandle:
    """一个已装载的基模。"""
    name: str
    model: Any = None
    cfg: dict = field(default_factory=dict)
    attribution: dict = field(default_factory=dict)
    spec: DomainSpec | None = None
    meta: dict = field(default_factory=dict)
    n_params: int = 0
    n_outputs: int = 2
    load_path: str = ""
    class_name: str = ""
    excluded: bool = False
    ok: bool = False
    error: str = ""

    def to_dict(self):
        return {
            "name": self.name, "ok": self.ok, "error": self.error,
            "n_params": self.n_params, "load_path": self.load_path,
            "attribution": self.attribution,
            "domain": ({"chans": self.spec.chans, "sfreq": self.spec.sfreq,
                        "n_times": self.spec.n_times,
                        "window_mode": self.spec.window_mode,
                        "note": self.spec.note} if self.spec else None),
            "meta": self.meta,
        }


# ---------------------------------------------------------------------------
# 装载
# ---------------------------------------------------------------------------
def _probe_version_subprocess():
    """在全新进程里查磁盘上的 braindecode 版本。"""
    import subprocess
    import sys
    try:
        r = subprocess.run(
            [sys.executable, "-c",
             "import braindecode;print(braindecode.__version__)"],
            capture_output=True, text=True, timeout=180)
        return (r.stdout or "").strip().splitlines()[-1] if r.stdout.strip() else None
    except Exception:
        return None


BD_STABLE_VERSION = "1.8.1"
BD_DEV_VERSION = "1.8.1.dev184031250"


def ensure_braindecode_version(target="stable", quiet=False):
    """保证**磁盘上**的 braindecode 是目标版本；装完**重入脚本**。

    target = "stable" | "1.8.1.dev184031250"（或任意版本串）

    ⚠️ 为什么必须重入：pip 换的是磁盘上的文件，而当前进程里 `braindecode` /
    `braindecode.models` 早已 import 过 ⇒ 内存里还是旧版
    （实测：装完 dev 版仍报"没有候选类 MIRepNet"）。
    `importlib.reload` 也不可靠（实测会把 7 个类全弄丢）。
    ⇒ 唯一干净的做法是**重入**（os.execv）。
    """
    import os
    import subprocess
    import sys

    want = BD_STABLE_VERSION if target == "stable" else target
    cur = _probe_version_subprocess()
    if cur == want:
        if not quiet:
            print(f"  braindecode {cur} ✓", flush=True)
        return cur
    def _pip(target_spec, no_deps=False):
        a = [sys.executable, "-m", "pip", "install", "-q", "--upgrade"]
        if no_deps:
            a += ["--no-deps", "--force-reinstall"]
        a.append(target_spec)
        return subprocess.run(a, capture_output=True, text=True,
                              timeout=1800).returncode

    if want != BD_STABLE_VERSION and cur is None:
        # ⚠️ dev 版**没装依赖**（--no-deps）⇒ braindecode 连 import 都失败。
        #    先装一次 stable 把依赖拉齐，再换 dev 版。
        rc0 = _pip("braindecode[hub]")
        if not quiet:
            print(f"  pip install braindecode[hub]（取依赖）rc={rc0}", flush=True)
        cur = _probe_version_subprocess()
        if cur is None:
            raise RuntimeError("连稳定版 braindecode 都装不上")
    spec = ("braindecode[hub]" if want == BD_STABLE_VERSION
            else f"braindecode[hub]=={want}")
    args = [sys.executable, "-m", "pip", "install", "-q", "--upgrade"]
    if want != BD_STABLE_VERSION:
        # dev 版：依赖在装稳定版时已就位，别把 torch（~2 GB）重下一遍
        args += ["--no-deps", "--force-reinstall"]
    args.append(spec)
    rc = subprocess.run(args, capture_output=True, text=True, timeout=1800).returncode
    new = _probe_version_subprocess()
    if not quiet:
        print(f"  pip install {spec} rc={rc} → {new}", flush=True)
    if new != want:
        raise RuntimeError(f"装不上 braindecode {want}（现在 {new}）")
    if not os.environ.get("ILLUSIONBRAIN_BD_REEXEC"):
        os.environ["ILLUSIONBRAIN_BD_REEXEC"] = "1"
        print("  ↻ 重入脚本以加载目标版本的 braindecode", flush=True)
        sys.stdout.flush()
        os.execv(sys.executable, [sys.executable] + sys.argv)
    return new


def things_chs_info(sfreq=250.0):
    """THINGS-EEG 的 63 通道（mne 格式，带位置）。

    位置是**必须的** —— braindecode 的 `Interpolated*` 包装靠它做 spline 插值。
    THINGS 的通道名（Fp1/AFz/FT9/…）全在 MNE 的 standard_1020 里。
    """
    import mne
    try:
        from .base_specs import THINGS_EEG_CHANS
    except ImportError:
        from base_specs import THINGS_EEG_CHANS
    info = mne.create_info(THINGS_EEG_CHANS, sfreq, "eeg")
    info.set_montage("standard_1020")
    return info["chs"]


def load_base_model(name, models_dir="/training/models", braindecode_models=None,
                    n_outputs=2, verbose=True, spec=None):
    """装载一个基模。返回 BaseModelHandle。

    装载策略（与 probe_base_models.py v10 一致，避免两套口径）：
      路径 A：`cls.from_pretrained(本地目录)` —— 首选；
      路径 B：手工 ctor + 手工权重（strict=False）—— 兜底。

    `n_outputs`：config 里常是 null（如 cbramod）⇒ 补一个占位头。
    我们只要 **token 表征**，头是什么无所谓，但缺了它模型构造会直接失败。

    `spec`（来自 base_specs.py）：**实测的输入规格**，比 config 权威
    （config 里大量 n_chans/n_times 是 null）。给了它就按它装载：
      · n_chans / n_times / sfreq 显式传给 from_pretrained；
      · `interpolate=True` ⇒ 改用 braindecode 的 `Interpolated*` 包装，
        用户给 THINGS 的 63 个通道，包装内部插值到骨干的规范通道集。
    """
    h = BaseModelHandle(name=name)
    h.load_s = 0.0
    if spec and spec.get("excluded"):
        h.excluded = True
        h.error = spec.get("excluded_reason", "本轮不纳入")
        return h
    d = Path(models_dir) / name
    h.cfg = read_json(d / "config.json")
    h.attribution = read_attribution(Path(models_dir), name)
    if not d.exists():
        h.error = f"桶内目录不存在：{d}"
        return h

    if braindecode_models is None:
        try:
            import braindecode.models as braindecode_models  # noqa: PLW0127
        except Exception as e:
            h.error = f"braindecode 不可用：{e!r}"
            return h

    # --- 选类：普通类 or Interpolated* 包装 ---
    cls = None
    if spec:
        want = spec.get("interpolated_cls") if spec.get("interpolate") \
            else spec.get("cls")
        if want and hasattr(braindecode_models, want):
            cls = getattr(braindecode_models, want)
            h.class_name = want
    if cls is None:
        for cand in CLASS_CANDIDATES.get(name, []):
            if hasattr(braindecode_models, cand):
                cls = getattr(braindecode_models, cand)
                h.class_name = cand
                break
    if cls is None:
        h.error = (f"braindecode 里没有候选类 "
                   f"{spec.get('cls') if spec else CLASS_CANDIDATES.get(name, [])}")
        return h

    # --- from_pretrained 的参数 ---
    kwargs = {"n_outputs": n_outputs}
    if spec:
        kwargs["n_chans"] = spec["n_chans"]
        kwargs["n_times"] = spec["n_times"]
        kwargs["sfreq"] = spec["sfreq"]
        if spec.get("interpolate"):
            kwargs["chs_info"] = things_chs_info(spec["sfreq"])

    # 表评测要的是**表征**而不是 logits ⇒ 打开模型的"返回表征"开关。
    # ⚠️ 各模型参数名不同（return_encoder_output / return_features /
    #    return_feature），认错了会直接构造失败 ⇒ 从**构造签名里探测**，
    #    并准备多套候选参数集逐个试。
    REPR_CANDIDATES = ("return_encoder_output", "return_features", "return_feature")
    repr_sets = []
    if spec:
        try:
            import inspect
            avail = set(inspect.signature(cls.__init__).parameters)
            for c in REPR_CANDIDATES:
                if c in avail:
                    repr_sets.append({c: True})
                    break
        except Exception:
            pass
        if spec.get("repr_kwargs"):
            repr_sets.append(dict(spec["repr_kwargs"]))
    repr_sets.append({})

    cand_kwargs = []
    for rk in repr_sets:
        kw = dict(kwargs)
        kw.update(rk)
        cand_kwargs.append(kw)
    # 插值模式参数在某些版本里不存在 ⇒ 再准备一套不带它的
    if spec and spec.get("interpolate"):
        kw = dict(kwargs)
        for rk in repr_sets:
            k2 = dict(kw)
            k2.update(rk)
            k2["interpolation_mode"] = "always"
            cand_kwargs.append(k2)
            cand_kwargs.append(dict(k2))

    t0 = time.time()
    errs = []
    for kw in cand_kwargs:
        try:
            h.model = cls.from_pretrained(str(d), **kw)
            h.load_path = f"from_pretrained({','.join(sorted(kw))})"
            h.load_s = round(time.time() - t0, 2)
            h.ok = True
            break
        except Exception as e:
            errs.append(f"{type(e).__name__}: {e}"[:180])
    if not h.ok:
        h.error = "from_pretrained 失败：" + " | ".join(errs[:3])[:400]
    if not h.ok:
        try:
            import inspect
            import torch
            sig = inspect.signature(cls.__init__)
            avail = set(sig.parameters) - {"self"}
            kw = {k: v for k, v in h.cfg.items() if k in avail}
            for k, v in kwargs.items():
                if k in avail:
                    kw[k] = v
            if "n_outputs" in avail and kw.get("n_outputs") is None:
                kw["n_outputs"] = n_outputs
            h.model = cls(**kw)
            h.load_path = "manual"
            h.load_s = round(time.time() - t0, 2)
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
                clean = {(k[7:] if k.startswith("module.") else k): v
                         for k, v in sd.items()}
                miss, unexp = h.model.load_state_dict(clean, strict=False)
                h.meta["weights_missing"] = list(miss)[:10]
                h.meta["weights_unexpected"] = list(unexp)[:10]
            h.ok = True
        except Exception as e2:
            h.error = (h.error + f"；手工路径也失败：{type(e2).__name__}: {e2}")[:500]
            h.meta["trace"] = traceback.format_exc()[-1200:]
            return h

    h.n_params = int(sum(p.numel() for p in h.model.parameters()))
    h.n_outputs = int(n_outputs)
    h.meta.update(_collect_meta(h.model))
    h.spec = spec_from_config(name, h.cfg, h.meta)
    if spec:
        # 以实测规格覆盖（config 里大量是 null）
        h.spec = DomainSpec(
            model=name,
            chans=spec.get("chans"),
            sfreq=spec.get("sfreq"),
            n_times=spec.get("n_times"),
            window_mode=spec.get("window_mode", "pad"),
            note=spec.get("note", ""),
            differentials=spec.get("differentials"),
            interpolate_missing=bool(spec.get("interpolate_missing")))
        h.meta["base_spec"] = {k: v for k, v in spec.items() if k != "chans"}
    if verbose:
        if h.ok:
            print(f"  ✓ {name}: {h.n_params:,} 参数 | {h.class_name} | "
                  f"{h.load_path} ({h.load_s}s)", flush=True)
            if h.load_path == "manual":
                # 走了手工路径 ⇒ 权重是 strict=False 装的，**必须核对缺了哪些**
                miss = h.meta.get("weights_missing") or []
                unexp = h.meta.get("weights_unexpected") or []
                print(f"     ⚠ 手工加载：缺 {len(miss)} 个 / 多 {len(unexp)} 个"
                      + (f"；缺的前几个 {miss[:4]}" if miss else ""), flush=True)
        else:
            print(f"  ✗ {name}: {h.error[:400]}", flush=True)
            if h.meta.get("weights_missing"):
                print(f"     缺权重: {h.meta['weights_missing'][:6]}", flush=True)
    return h


def _collect_meta(model) -> dict:
    out = {}
    mi = _safe_getattr(model, "model_info")
    if isinstance(mi, dict):
        out["model_info"] = {k: (list(v) if isinstance(v, (list, tuple)) else v)
                             for k, v in mi.items()}
    for attr in ("ch_names", "sfreq", "n_chans", "n_times", "input_window_seconds",
                 "embedding_dim", "final_dim", "embed_dim", "d_model", "n_layer"):
        v = _safe_getattr(model, attr)
        if v is not None:
            out[attr] = list(v) if isinstance(v, (list, tuple)) else v
    return out


# ---------------------------------------------------------------------------
# token 抽取
# ---------------------------------------------------------------------------
def normalize_token_tensor(t):
    """把各种形状的输出统一成 (B, N, D)。

    - (B, N, D)  直接用
    - (B, C, H, W) ⇒ 展平成 (B, H*W, C)（二维特征图 ⇒ token 序列）
    - (B, D)     ⇒ 视作 1 个 token
    """
    import torch
    if t.ndim == 3:
        return t
    if t.ndim == 4:
        b, c, h, w = t.shape
        return t.reshape(b, c, h * w).transpose(1, 2).contiguous()
    if t.ndim == 2:
        return t.unsqueeze(1)
    raise ValueError(f"不支持的 token 张量维度 {t.ndim}")


def tokens_from_output(out, n_outputs=None):
    """从模型返回值里找 token 张量。返回 (tokens or None, how)。

    - dict：按 TOKEN_KEY_PRIORITY 取第一个 (B, N, D)
    - tuple/list：优先取**不是分类 logits**的那个（BIOT 的 return_feature 返回
      (特征, logits) 的元组，取错元素就会把 logits 当表征）
    - tensor 3 维：直接用
    - tensor 4 维：(B, C, H, W) ⇒ 展平成 (B, H*W, C)
    - tensor 2 维：(B, D) ⇒ 视作 1 个 token（但若 D == n_outputs 就是 logits）
    """
    import torch

    def _is_logits(t):
        return (n_outputs is not None and t.ndim >= 2
                and int(t.shape[-1]) == int(n_outputs))
    if isinstance(out, dict):
        for k in TOKEN_KEY_PRIORITY:
            if k in out:
                v = out[k]
                if torch.is_tensor(v) and v.ndim in (2, 3, 4):
                    return normalize_token_tensor(v), f"dict[{k}]"
        # 退而求其次：dict 里第一个 3 维张量
        for k, v in out.items():
            if torch.is_tensor(v) and v.ndim == 3:
                return normalize_token_tensor(v), f"dict-scan[{k}]"
        return None, "dict 里没有 token 字段"
    if isinstance(out, (tuple, list)):
        cands = [(i, v) for i, v in enumerate(out)
                 if torch.is_tensor(v) and v.ndim in (2, 3)]
        # 先排除 logits 形状的
        cands = [(i, v) for i, v in cands if not _is_logits(v)] or cands
        for i, v in cands:
            if v.ndim == 3:
                return normalize_token_tensor(v), f"tuple[{i}]"
        for i, v in cands:
            if v.ndim == 2:
                return normalize_token_tensor(v), f"tuple[{i}]2d"
        return None, "tuple 里没有 2/3 维张量"
    if torch.is_tensor(out):
        if out.ndim in (2, 3, 4):
            if _is_logits(out):
                return None, "输出是分类 logits，不是表征"
            return normalize_token_tensor(out), f"tensor{out.ndim}d"
        return None, f"tensor{out.ndim}d 不支持"
    return None, f"未知输出类型 {type(out).__name__}"


def _norm_tensor(t):
    """统一成 (B, N, D) 的 float32。"""
    import torch
    if t.dtype not in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
        t = t.float()
    return t


def collect_layer_tokens(model, x, depth1_only=True):
    """用 forward hook 收集中间 token。返回 {模块名: (B, N, D)}。

    `depth1_only=True` 时只挂根模块的**直接子模块**（如 `encoder`、`blocks`）——
    它们的输出通常就是最终表征；挂更深的模块会拿到注意力内部的中间量，
    语义不清。
    """
    import torch
    out = {}
    handles = []
    root_children = {name for name, _ in model.named_children()}

    def mk_hook(nm):
        def hook(mod, inp, o):
            t = o
            if isinstance(t, (tuple, list)):
                t = t[0] if t else None
            if torch.is_tensor(t) and t.ndim == 3:
                out[nm] = t.detach()
        return hook

    for nm, mod in model.named_modules():
        if nm == "":
            continue
        head = nm.split(".")[0]
        if depth1_only and head not in root_children:
            continue
        if nm in root_children or not depth1_only:
            handles.append(mod.register_forward_hook(mk_hook(nm)))
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()
    return out


def infer_layout(n_tokens, cfg=None, meta=None, model=None):
    """推出 token 的空间布局。

    返回 {"layout": (rows, cols) | None, "n_global": int,
          "rows": int|None, "cols": int|None, "confidence": str, "reason": str}

    `rows` = 时间轴分块数，`cols` = 通道轴分块数（EEG 惯例：行=时间）。
    confidence： "config"（权威）| "module"（从模型属性推的）|
                "guessed"（因数分解猜的，**不得用于二维 mixer**）| "none"
    """
    cfg = cfg or {}
    meta = meta or {}
    res = {"layout": None, "n_global": 0, "rows": None, "cols": None,
           "confidence": "none", "reason": ""}

    n_chans = cfg.get("n_chans") or meta.get("n_chans")
    if n_chans is None:
        ch = meta.get("ch_names") or (meta.get("model_info") or {}).get("ch_names")
        if ch:
            n_chans = len(ch)
    n_times = cfg.get("n_times") or meta.get("n_times")

    patch = _find_patch(cfg, meta, model)
    if patch and n_chans and n_times:
        pc, pt = patch
        if pc > 0 and pt > 0 and n_chans % pc == 0 and n_times % pt == 0:
            cols = n_chans // pc
            rows = n_times // pt
            if rows * cols <= n_tokens:
                res.update(layout=(rows, cols), rows=rows, cols=cols,
                           n_global=int(n_tokens - rows * cols),
                           confidence="config",
                           reason=f"patch=(ch{pc},t{pt}) ⇒ {rows}×{cols} 网格"
                                  f"（+{n_tokens - rows * cols} 全局 token）")
                return res

    # 因数分解猜：取最接近方形的分解
    best = None
    for r in range(1, int(n_tokens ** 0.5) + 1):
        if n_tokens % r == 0:
            c = n_tokens // r
            if best is None or abs(r - c) < abs(best[0] - best[1]):
                best = (r, c)
    if best:
        res.update(layout=(best[0], best[1]), rows=best[0], cols=best[1],
                   n_global=0, confidence="guessed",
                   reason=f"按 {n_tokens} 因数分解猜 {best[0]}×{best[1]}"
                          f"（**不得用于二维 mixer**）")
    else:
        res["reason"] = "无法确定布局"
    return res


def _find_patch(cfg, meta, model):
    """推出 (patch_ch, patch_time)。找不到返回 None。

    EEG 惯例 patch_dims=(通道, 时间)，但不同实现顺序不一 ⇒ 用
    「能否整除 n_chans / n_times」来定序，不靠猜。
    """
    cands = []
    for src in (cfg, meta):
        for key in ("patch_dims", "patch_size", "patch_length", "patch_shape"):
            v = src.get(key)
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                try:
                    cands.append((int(v[0]), int(v[1])))
                except Exception:
                    pass
        ke = src.get("kernel_size")
        if isinstance(ke, (list, tuple)) and len(ke) >= 2:
            try:
                cands.append((int(ke[0]), int(ke[1])))
            except Exception:
                pass
    if model is not None:
        try:
            for _, mod in model.named_modules():
                for attr in ("kernel_size", "patch_size", "patch_dims"):
                    v = _safe_getattr(mod, attr)
                    if isinstance(v, (list, tuple)) and len(v) >= 2:
                        try:
                            cands.append((int(v[0]), int(v[1])))
                        except Exception:
                            pass
        except Exception:
            pass
    return cands[0] if cands else None


# ---------------------------------------------------------------------------
# 池化与编码
# ---------------------------------------------------------------------------
POOL_MODES = ("mean_all", "mean_spatial", "cls", "last")


def pool_tokens(tokens, mode="mean_all", layout=None, n_global=0):
    """池化 token → (B, D)。

    - mean_all：对**全部** token 取均值（含全局 token）—— 默认口径
    - mean_spatial：只对空间 token 取均值（排除全局 token）
    - cls：取第一个 token
    - last：取最后一个 token
    """
    import torch
    if tokens.ndim != 3:
        raise ValueError(f"pool 需要 (B, N, D)，收到 {tuple(tokens.shape)}")
    if mode == "mean_all":
        return tokens.mean(dim=1)
    if mode == "cls":
        return tokens[:, 0]
    if mode == "last":
        return tokens[:, -1]
    if mode == "mean_spatial":
        n_spatial = tokens.shape[1] - int(n_global or 0)
        if n_spatial <= 0:
            return tokens.mean(dim=1)
        return tokens[:, :n_spatial].mean(dim=1)
    raise ValueError(f"未知池化口径 {mode}")


def encode_trials(handle, eeg, batch_size=32, device="cpu", pool="mean_all",
                  progress=None):
    """把 (n, C, T) 的 EEG 跑过基模，得到 (n, D) 的池化特征。

    返回 {"ok": bool, "features": np.ndarray|None, "info": dict}
    """
    import torch
    if not handle.ok or handle.model is None:
        return {"ok": False, "features": None,
                "info": {"error": handle.error or "模型未装载"}}
    eeg = np.asarray(eeg, dtype=np.float32)
    if eeg.ndim != 3:
        return {"ok": False, "features": None,
                "info": {"error": f"期望 (n, C, T)，收到 {eeg.shape}"}}
    model = handle.model
    model.eval()
    try:
        model.to(device)
    except Exception:
        device = "cpu"

    feats = []
    info = {"pool": pool, "device": device, "n_trials": int(eeg.shape[0]),
            "layout": None, "layout_confidence": "none", "n_global": 0,
            "token_how": None}
    n = eeg.shape[0]
    # ⚠️ 输入 dtype 必须**跟随模型参数**：
    #   · 域适配（IDW 插值 / resample_poly / pad）可能把 dtype 提到 float64，
    #     而权重是 float32 ⇒ "Input type (double) and bias type (float) should be the same"；
    #   · mirepnet 的权重本身是 **fp16** ⇒ 喂 float32 反而报
    #     "Input type (c10::Half) and bias type (float) should be the same"。
    mdt = next((p.dtype for p in model.parameters()), None)
    if mdt is None or not torch.is_floating_point(torch.empty(0, dtype=mdt)):
        mdt = torch.float32

    for s in range(0, n, batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(eeg[s:s + batch_size]))
        xb = xb.to(device=device, dtype=mdt)
        try:
            with torch.no_grad():
                out = model(xb)
            n_out = int(getattr(handle, "n_outputs", 2) or 2)
            tok, how = tokens_from_output(out, n_out)
            # ⚠️ 抽到的可能是**分类 logits**（维度 == n_outputs）而不是表征
            #    （实测 BIOT 开了 return_feature 仍返回 (B,2)）⇒ 改用 hook
            #    抓根子模块的中间输出（那是编码器表征）。
            if tok is not None and tok.shape[-1] == n_out and tok.shape[1] <= 2:
                lay = collect_layer_tokens(model, xb)
                cand = [v for v in lay.values() if v.shape[-1] != n_out]
                if cand:
                    tok = cand[-1]
                    how = f"hook(避开 logits):{list(lay.keys())[-1]}"
            if tok is None:
                # 直接输出里没有 token ⇒ 退回 hook 抓根子模块输出
                lay = collect_layer_tokens(model, xb)
                if not lay:
                    return {"ok": False, "features": None,
                            "info": {**info, "error": f"抽不到 token：{how}"}}
                tok = list(lay.values())[-1]
                how = f"hook:{list(lay.keys())[-1]}"
            tok = tok.float()
            if info["token_how"] is None:
                info["token_how"] = how
                lay_info = infer_layout(tok.shape[1], handle.cfg, handle.meta, model)
                info["layout"] = list(lay_info["layout"]) if lay_info["layout"] else None
                info["layout_confidence"] = lay_info["confidence"]
                info["n_global"] = lay_info["n_global"]
                info["layout_reason"] = lay_info["reason"]
            f = pool_tokens(tok, pool, info["layout"], info["n_global"])
            feats.append(f.detach().cpu().numpy())
        except Exception as e:
            return {"ok": False, "features": None,
                    "info": {**info, "error": f"前向失败：{type(e).__name__}: {e}"[:300],
                             "trace": traceback.format_exc()[-800:]}}
        if progress:
            progress(min(s + batch_size, n), n)
    if not feats:
        return {"ok": False, "features": None, "info": {**info, "error": "没有产出特征"}}
    X = np.concatenate(feats, axis=0)
    info["n_features"] = int(X.shape[1])
    return {"ok": True, "features": X, "info": info}
