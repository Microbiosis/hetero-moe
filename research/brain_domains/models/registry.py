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
    load_path: str = ""
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
def load_base_model(name, models_dir="/training/models", braindecode_models=None,
                    n_outputs=2, verbose=True):
    """装载一个基模。返回 BaseModelHandle。

    装载策略（与 probe_base_models.py v3 完全一致，避免两套口径）：
      路径 A：`cls.from_pretrained(本地目录)` —— 首选，自动处理 config→ctor；
      路径 B：手工 ctor + 手工权重（strict=False）—— 兜底。

    `n_outputs`：config 里常是 null（如 cbramod）⇒ 补一个占位头。
    我们只要 **token 表征**，头是什么无所谓，但缺了它模型构造会直接失败。
    """
    h = BaseModelHandle(name=name)
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

    cls = None
    for cand in CLASS_CANDIDATES.get(name, []):
        if hasattr(braindecode_models, cand):
            cls = getattr(braindecode_models, cand)
            break
    if cls is None:
        h.error = f"braindecode 里没有候选类 {CLASS_CANDIDATES.get(name, [])}"
        return h

    t0 = time.time()
    try:
        try:
            h.model = cls.from_pretrained(str(d), n_outputs=n_outputs)
            h.load_path = "from_pretrained(local)+n_outputs"
        except TypeError:
            h.model = cls.from_pretrained(str(d))
            h.load_path = "from_pretrained(local)"
    except Exception as e:
        h.load_path = "manual"
        h.error = f"from_pretrained 失败：{type(e).__name__}: {e}"[:300]
        try:
            import inspect
            import torch
            sig = inspect.signature(cls.__init__)
            avail = set(sig.parameters) - {"self"}
            kwargs = {k: v for k, v in h.cfg.items() if k in avail}
            if "n_outputs" in avail and kwargs.get("n_outputs") is None:
                kwargs["n_outputs"] = n_outputs
            h.model = cls(**kwargs)
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
        except Exception as e2:
            h.error = (h.error + f"；手工路径也失败：{type(e2).__name__}: {e2}")[:500]
            h.meta["trace"] = traceback.format_exc()[-1200:]
            return h

    h.ok = True
    h.load_s = round(time.time() - t0, 2)
    h.n_params = int(sum(p.numel() for p in h.model.parameters()))
    h.meta.update(_collect_meta(h.model))
    h.spec = spec_from_config(name, h.cfg, h.meta)
    if verbose:
        print(f"  ✓ {name}: {h.n_params:,} 参数 | 载入 {h.load_path} ({h.load_s}s)",
              flush=True)
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


def tokens_from_output(out):
    """从模型返回值里找 token 张量。返回 (tokens or None, how)。

    - dict：按 TOKEN_KEY_PRIORITY 取第一个 (B, N, D)
    - tuple/list：取第一个 3 维张量
    - tensor 3 维：直接用
    - tensor 4 维：(B, C, H, W) ⇒ 展平成 (B, H*W, C)
    - tensor 2 维：(B, D) ⇒ 视作 1 个 token
    """
    import torch
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
        for i, v in enumerate(out):
            if torch.is_tensor(v) and v.ndim == 3:
                return normalize_token_tensor(v), f"tuple[{i}]"
        return None, "tuple 里没有 3 维张量"
    if torch.is_tensor(out):
        if out.ndim in (2, 3, 4):
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
    for s in range(0, n, batch_size):
        xb = torch.from_numpy(eeg[s:s + batch_size])
        xb = xb.to(device)
        try:
            with torch.no_grad():
                out = model(xb)
            tok, how = tokens_from_output(out)
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
