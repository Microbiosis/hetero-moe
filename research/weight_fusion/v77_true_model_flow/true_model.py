"""true_model.py — A1 路线 β：真 HF 模型的逐层驱动与层边界注入（S1beta）。

背景：docs/A1_DESIGN.md §3.2 / §8（S3 的硬前提）+ docs/MAINLINE_ROUTE.md §4.8。

v76 在路线 α 上证明了两件事：
  (1)「层边界打断 -> 注入 -> 续跑」在计算图层面成立（机制存在）；
  (2) 路线 α（随机输入的权重级前向）测不了语义层（匹配槽 ≈ 错槽）⇒ 语义层必须走路线 β。

本文件做 S1beta = 路线 β 的机械可行性 + 真模型侧的前置测量，**不做** P1a/P1b/P2/P3：

  B1 权重同源        切片权重 == 真模型参数（证明两条路线用的是同一批权重）
  B2 逐层驱动        真模型逐层状态能取到；零注入的打断/续跑与原前向等价
  B3 注入通路        非零注入真的改变后续层输出，且没把模型搞坏
  B4 真模型层栈健康  含 ViT（检验 v76 的 ViT 缺陷根因假设）
  B5 槽身份可分辨    真输入下匹配槽是否强于错槽（v76 的 C5 在真模型上重跑；**对照**，不是 P1b）

⚠️ 两条边界必须随数字一起引用：
  - B5 只是**对照测量**：它测的是「真模型真输入下槽身份可不可分辨」，
    不是 P1b（槽特异）——P1b 要靠 S3 的四臂注入实验（arm1 vs arm2/arm3）才能判。
  - 本代仍**没有**任何端到端任务指标：注入不改变任何底座权重，也没有标签级的
    能力指标 ⇒ 不得说成能力或精度结论。

判据阈值全部在跑之前写死（见 THRESHOLDS），事后不得放宽。
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util as ilu
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# xet 后端在本机会卡在 read-token 端点；只对**本进程**生效，不写任何持久配置。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch

# ---------------------------------------------------------------- 常量（跑前写死）
DS = 256                      # 共享空间维度（与 v62 一致）
N_LAYERS = 4                  # 与 v75/v76 同口径：只看前 4 层
N_TEXT = 128                  # 真实文本条数（ag_news）
SEQ = 128                     # 文本序列长
N_IMG = 4                     # 真实图片张数（food101）
BATCH = 16

MODEL_IDS = {
    "Qwen": "Qwen/Qwen2.5-0.5B-Instruct",
    "BERT": "google/bert_uncased_L-4_H-256_A-4",
    "ViT": "google/vit-base-patch16-224-in21k",
}

#: 只拉真正需要的文件（避开 flax / tf_model.h5 / bert_model.ckpt 这类大文件）
ALLOW_PATTERNS = {
    "Qwen": ["config.json", "generation_config.json", "model.safetensors",
             "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"],
    "BERT": ["config.json", "pytorch_model.bin", "vocab.txt"],
    "ViT": ["config.json", "model.safetensors", "preprocessor_config.json"],
}

#: 槽 -> 真模型参数路径（与 v62 measure.py 的 SLOT_PATHS 同源，此处只作断言用）
SLOT_SOURCE = {"Q": "out", "K": "out", "V": "out",
                "O": "in", "expand": "out", "contract": "in"}

THRESHOLDS = {
    # B1 权重同源
    "B1_rel_err_exact": 1e-5,        # <= 判为与真模型逐位同源
    "B1_rel_err_lowprec": 1e-2,      # <= 判为同源但切片经低精度量化
    # B2 逐层驱动等价
    "B2_zero_inject_rel_err": 1e-6,
    # B3 注入通路
    "B3_min_rel_perturbation": 1e-3,
    "B3_max_rel_perturbation": 0.5,  # 注入没把模型搞坏
    # B4 层栈健康
    "B4_norm_ratio": [0.1, 10.0],
    "B4_eff_rank_min": 1.1,
    # B5 槽身份可分辨（对照）
    "B5_matched_better_min": 0.75,   # 16 块中至少 12 块
    "B5_sign_test_p_max": 0.05,
}

CACHE_DIR = Path.home() / ".cache" / "hetero-moe"

#: B4 层栈健康的判据范围 = A1 首代覆盖的底座 (design §2: Qwen<->BERT),
#: 与 v76 的 C3 同口径; ViT 只作范围外记录。
HEALTH_SCOPE = ("Qwen", "BERT")

#: 本地模型目录：把权重拉到固定目录后再 from_pretrained，绕开 hf_hub 客户端
#: 在本机被掐断的分块读取（"Server disconnected without sending a response"）
#: —— 同样的 URL 用 requests 直取一次就通。
MODEL_DIRS = {name: (CACHE_DIR / "models" / name) for name in MODEL_IDS}


# ---------------------------------------------------------------- 按路径载入（复用不复制）
def _load_by_path(path: Path, name: str):
    spec = ilu.spec_from_file_location(name, str(path))
    mod = ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_PKG = Path(__file__).resolve().parent
_V76 = _load_by_path(_PKG.parent / "v76_cross_layer_flow" / "a1_forward.py", "a1_forward_v76_by_path")
_V75 = _load_by_path(_PKG.parent / "v75_weight_conjugation" / "g2_conjugation.py", "g2_v75_by_path")
MEAS = _V75.MEAS                     # v62 measure.py（v75 已按路径载入）
SLOT_PATHS = MEAS.SLOT_PATHS

#: 口径同源凭证（跑的时候现场算 sha256）
PROVENANCE_FILES = {
    "v62_measure": _V75.V62_MEASURE_PATH,
    "v75_g2": str(_PKG.parent / "v75_weight_conjugation" / "g2_conjugation.py"),
    "v76_a1_forward": str(_PKG.parent / "v76_cross_layer_flow" / "a1_forward.py"),
}


def sha256_of(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def eff_rank(x: torch.Tensor) -> float:
    """参与比有效秩（与 v76 同式，复用 v76 实现，不复制）。"""
    return _V76.eff_rank(x)


def ridge_fit_r2(U: torch.Tensor, V: torch.Tensor):
    """闭式岭回归 R²（与 v75 fit_transport / v76 同式，复用 v76 实现）。"""
    return _V76.ridge_fit_r2(U, V)


# ---------------------------------------------------------------- 模型装载
def _link_or_copy(src: Path, dest: Path) -> None:
    """硬链接优先（同一磁盘零拷贝），失败退回拷贝。"""
    try:
        os.link(os.path.realpath(str(src)), str(dest))
    except OSError:
        import shutil
        shutil.copy2(str(src), str(dest))


def _download(url: str, dest: Path, tries: int = 5) -> Path:
    """requests 直取 + 重试；不写任何持久配置，只落在这个缓存目录里。"""
    import requests
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(tries):
        try:
            resp = requests.get(url, timeout=120, stream=True)
            resp.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(1 << 20):
                    if chunk:
                        f.write(chunk)
            tmp.replace(dest)
            return dest
        except Exception as exc:  # noqa: BLE001 - 网络抖动必须重试
            print("  [retry %d] %s: %s" % (attempt, type(exc).__name__, str(exc)[:90]), flush=True)
            time.sleep(3)
    raise RuntimeError("下载失败: %s" % url)


def fetch_models() -> Dict[str, str]:
    """把三个底座拉到本地固定目录（只拉需要的文件）。

    优先复用 HF 缓存里已经下好的文件（硬链接，零拷贝）；
    缓存里没有的用 requests 直取（hf_hub 客户端在本机读大文件会被掐断）。
    """
    from huggingface_hub import snapshot_download
    out = {}
    for name, repo in MODEL_IDS.items():
        d = MODEL_DIRS[name]
        d.mkdir(parents=True, exist_ok=True)
        cached = None
        try:
            cached = Path(snapshot_download(repo, allow_patterns=ALLOW_PATTERNS[name],
                                           local_files_only=True))
        except Exception:
            cached = None
        for fn in ALLOW_PATTERNS[name]:
            dest = d / fn
            if dest.exists() and dest.stat().st_size > 0:
                continue
            if cached is not None and (cached / fn).exists():
                _link_or_copy(cached / fn, dest)
                continue
            url = "https://huggingface.co/%s/resolve/main/%s" % (repo, fn)
            print("  [download] %s/%s" % (name, fn), flush=True)
            _download(url, dest)
        out[name] = str(d)
    return out


def _from_pretrained(cls, repo: str, **kw):
    """transformers 5.x 用 dtype=，4.x 用 torch_dtype=；两边都试。"""
    try:
        return cls.from_pretrained(repo, dtype=torch.float32, **kw)
    except TypeError:
        return cls.from_pretrained(repo, torch_dtype=torch.float32, **kw)


def layers_of(model) -> List[torch.nn.Module]:
    """拿到底座的 decoder layer 列表（不同架构路径不同，逐个试）。"""
    for path in ("layers", "model.layers", "encoder.layer", "encoder.layers",
                 "model.encoder.layer", "model.decoder.layers"):
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and isinstance(obj, (torch.nn.ModuleList, list, tuple)):
            return list(obj)
    raise RuntimeError("找不到 decoder layers: %s" % type(model).__name__)


class Backbone:
    """一个底座：模型 + 分词/预处理 + 层列表。"""

    def __init__(self, name: str):
        from transformers import (AutoImageProcessor, AutoModel, AutoTokenizer)
        self.name = name
        self.repo = MODEL_IDS[name]
        self.local = str(MODEL_DIRS[name])
        self.kind = "image" if name == "ViT" else "text"
        if self.kind == "text":
            self.tok = AutoTokenizer.from_pretrained(self.local)
            self.proc = None
        else:
            self.tok = None
            self.proc = AutoImageProcessor.from_pretrained(self.local)
        self.model = _from_pretrained(AutoModel, self.local).eval()
        self.layers = layers_of(self.model)
        self.cfg = self.model.config
        self.dim = int(getattr(self.cfg, "hidden_size",
                               getattr(self.cfg, "num_features", 0)))

    def n_layers(self) -> int:
        return int(getattr(self.cfg, "num_hidden_layers", len(self.layers)))

    def heads(self) -> dict:
        d = self.dim
        nh = int(getattr(self.cfg, "num_attention_heads", 0) or 0)
        nkv = int(getattr(self.cfg, "num_key_value_heads", nh) or nh)
        return {"n_heads": nh, "n_kv": nkv,
                "head_dim": (d // nh) if nh else 0, "d": d,
                "n_layers": self.n_layers()}


def text_batches(bb: Backbone, texts: List[Dict]) -> List[dict]:
    """把文本切成 token 批次 (padding 到固定长度, 跨模型同口径)。

    提到模块级是为了让同线的 v78 (S3 前置诊断) 按路径复用同一套分批口径 ——
    两代必须用**同一批语料与同一套 tokenization**, 否则诊断与实验不可比。
    """
    out = []
    for i in range(0, len(texts), BATCH):
        chunk = [t["text"] for t in texts[i:i + BATCH]]
        enc = bb.tok(chunk, return_tensors="pt", padding="max_length",
                     truncation=True, max_length=SEQ)
        out.append({k: v for k, v in enc.items()})
    return out


# ---------------------------------------------------------------- 层边界注入
class Injector:
    """在 decoder layer 的**输入**处加 delta —— 即「层边界打断 -> 注入 -> 续跑」。

    用 forward_pre_hook 而不是手工逐层调用：真模型的层 forward 签名
    （position_embeddings / cache / mask）随 transformers 版本变动，手工调用
    极易漏参数；hook 走的是模型自己的前向，续跑语义与真模型完全一致。
    """

    def __init__(self, layer_module, delta: Optional[torch.Tensor]):
        self.layer = layer_module
        self.delta = delta
        self.handle = None

    def __enter__(self):
        def hook(module, args, kwargs):
            if self.delta is None:
                return args, kwargs
            hs = kwargs.get("hidden_states", None)
            if hs is None and len(args) > 0:
                args = (args[0] + self.delta,) + tuple(args[1:])
                return args, kwargs
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = hs + self.delta
            return args, kwargs

        self.handle = self.layer.register_forward_pre_hook(hook, with_kwargs=True)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
        return False


def _resolve(obj, path: str):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def slot_module(layer, model_name: str, slot: str):
    path = SLOT_PATHS[model_name][slot]
    mod_path, attr = path.rsplit(".", 1)
    mod = _resolve(layer, mod_path)
    return mod, attr


def capture_slots(bb: Backbone, batch: dict,
                  layer_slots: Sequence[Tuple[int, str]]) -> Dict[Tuple[int, str], torch.Tensor]:
    """抓真模型里若干 (层, 槽) 的激活 —— **一次前向抓全部**（别每个槽跑一遍）。

    投影类槽（Q/K/V/expand）取模块**输出**；o_proj / down_proj 类槽
    （O/contract）取模块**输入** —— 这与 v62 的槽权重语义一致：
    槽 = 该功能矩阵作用在层内状态上的结果。
    """
    store: Dict[Tuple[int, str], torch.Tensor] = {}
    handles = []
    for ly, slot in layer_slots:
        mod, _attr = slot_module(bb.layers[ly], bb.name, slot)
        key = (ly, slot)
        if SLOT_SOURCE[slot] == "out":
            def h(module, inp, out, _k=key):
                store[_k] = out.detach()
            handles.append(mod.register_forward_hook(h))
        else:
            def h(module, inp, _k=key):
                store[_k] = inp[0].detach()
            handles.append(mod.register_forward_pre_hook(h))
    try:
        with torch.no_grad():
            bb.model(**batch)
    finally:
        for h in handles:
            h.remove()
    return store


# ---------------------------------------------------------------- 真实语料 / 图片
def _get_with_retry(requests, url: str, tries: int = 5, valid=None):
    """本机到 CDN 的连接会被中途掐断 ⇒ 所有取数都走重试。

    valid: 可选的校验函数（返回 True 才算成功），用来挡"200 但内容是坏的"。
    """
    import time as _t
    last = None
    for attempt in range(tries):
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            if valid is None or valid(resp):
                return resp
            last = RuntimeError("内容校验未过 (status=%s, len=%d)"
                                % (resp.status_code, len(resp.content)))
        except Exception as exc:  # noqa: BLE001 - 网络抖动必须重试
            last = exc
        print("  [retry %d] %s: %s" % (attempt, type(last).__name__, str(last)[:90]), flush=True)
        _t.sleep(3)
    raise RuntimeError("取数失败: %s (%s)" % (url, last))


def fetch_texts(n: int = N_TEXT) -> List[Dict]:
    """取真实带标签文本（ag_news），走 HF datasets-server 的 rows API，零安装。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / ("agnews_%d.json" % n)
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    import requests
    out: List[Dict] = []
    # ag_news 的 train 是**按标签排序**的 ⇒ 直接取前 N 条只会覆盖少数类。
    # 改成在 120000 行里按固定种子随机取 offset（跨类覆盖），每次取 chunk 行。
    rng = np.random.RandomState(1234)
    n_train = 120000
    chunk = 8
    n_blocks = (n + chunk - 1) // chunk
    offsets = rng.randint(0, n_train - chunk, size=n_blocks).tolist()
    for off in offsets:
        if len(out) >= n:
            break
        url = ("https://datasets-server.huggingface.co/rows?dataset=fancyzhx/ag_news"
               "&config=default&split=train&offset=%d&length=%d" % (off, chunk))
        resp = _get_with_retry(requests, url)
        rows = resp.json()["rows"]
        out.extend({"text": r["row"]["text"], "label": int(r["row"]["label"])}
                   for r in rows)
    out = out[:n]
    if len(out) < n:
        raise RuntimeError("只拿到 %d 条文本，少于 %d" % (len(out), n))
    cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def fetch_images(n: int = N_IMG) -> List:
    """取真实图片（food101），同样走 rows API；图片本体走 datasets-server 的静态资源。"""
    from io import BytesIO
    from PIL import Image
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / ("food101_%d.pkl" % n)
    if cache.exists():
        import pickle
        with open(cache, "rb") as f:
            return pickle.load(f)
    import pickle
    import requests
    url = ("https://datasets-server.huggingface.co/rows?dataset=ethz/food101"
           "&config=default&split=train&offset=0&length=%d" % n)
    resp = _get_with_retry(requests, url)
    rows = resp.json()["rows"]
    imgs = []
    for r in rows:
        src = r["row"]["image"]["src"]
        # 静态资源可能被中途掐断 ⇒ 校验 JPEG magic 后才算拿到
        b = _get_with_retry(requests, src, valid=lambda x: x.content[:2] == b"\xff\xd8")
        imgs.append(Image.open(BytesIO(b.content)).convert("RGB"))
    with open(cache, "wb") as f:
        pickle.dump(imgs, f)
    return imgs


# ---------------------------------------------------------------- B1 权重同源
def check_weight_provenance(bb: Backbone, wall) -> dict:
    """切片权重 vs 真模型参数 —— 两条路线用的是不是同一批权重。"""
    rows = []
    verdicts = []
    for ly in range(N_LAYERS):
        for slot in SLOT_PATHS[bb.name]:
            key = (bb.name, str(ly), slot)
            if key not in wall:
                continue
            layer = bb.layers[ly]
            mod, attr = slot_module(layer, bb.name, slot)
            w_true = getattr(mod, attr).detach().float()
            w_slice = wall[key].float()
            if tuple(w_true.shape) != tuple(w_slice.shape):
                rows.append({"layer": ly, "slot": slot, "shape_match": False,
                             "true": list(w_true.shape), "slice": list(w_slice.shape),
                             "rel_err": None})
                verdicts.append("shape_mismatch")
                continue
            rel = float((w_true - w_slice).abs().max() / (w_true.abs().max() + 1e-12))
            if rel <= THRESHOLDS["B1_rel_err_exact"]:
                v = "same_source_exact"
            elif rel <= THRESHOLDS["B1_rel_err_lowprec"]:
                v = "same_source_low_precision"
            else:
                v = "different_source"
            verdicts.append(v)
            rows.append({"layer": ly, "slot": slot, "shape_match": True,
                         "rel_err": rel, "verdict": v})
    ok = bool(rows) and all(v in ("same_source_exact", "same_source_low_precision")
                            for v in verdicts)
    return {"ok": ok, "model": bb.name, "n_blocks": len(rows),
            "n_exact": sum(1 for v in verdicts if v == "same_source_exact"),
            "n_low_precision": sum(1 for v in verdicts if v == "same_source_low_precision"),
            "n_different": sum(1 for v in verdicts if v == "different_source"),
            "max_rel_err": max([r["rel_err"] for r in rows if r.get("rel_err") is not None] or [None]),
            "rows": rows}


# ---------------------------------------------------------------- B2 逐层驱动
def check_layerwise_drive(bb: Backbone, batch: dict) -> dict:
    """真模型的逐层状态能取到吗？零注入的打断/续跑与原前向等价吗？"""
    with torch.no_grad():
        ref = bb.model(**batch, output_hidden_states=True)
    hs_ref = ref.hidden_states
    n_expected = len(bb.layers) + 1
    shapes_ok = len(hs_ref) == n_expected
    finite = all(bool(torch.isfinite(h).all()) for h in hs_ref)

    layer = bb.layers[1]
    zero = torch.zeros_like(hs_ref[1])
    with Injector(layer, zero):
        with torch.no_grad():
            inj = bb.model(**batch, output_hidden_states=True)
    per_layer = []
    worst = 0.0
    for i, (a, b) in enumerate(zip(hs_ref, inj.hidden_states)):
        rel = float((a - b).pow(2).sum().sqrt() / (a.pow(2).sum().sqrt() + 1e-12))
        worst = max(worst, rel)
        per_layer.append({"layer": i, "rel_err": rel})
    ok = bool(shapes_ok and finite and worst <= THRESHOLDS["B2_zero_inject_rel_err"])
    return {"ok": ok, "model": bb.name, "n_layers": len(bb.layers),
            "n_hidden_states": len(hs_ref), "expected": n_expected,
            "shapes_ok": shapes_ok, "finite": finite,
            "zero_inject_max_rel_err": worst,
            "per_layer_rel_err": per_layer,
            "hidden_shape": list(hs_ref[0].shape)}


# ---------------------------------------------------------------- B3 注入通路
def check_injection(bb: Backbone, batch: dict, inject_at: int = 1) -> dict:
    """注入非零向量后，后续层输出真的变了吗？模型没被搞坏吗？"""
    with torch.no_grad():
        ref = bb.model(**batch, output_hidden_states=True)
    hs_ref = ref.hidden_states
    h = hs_ref[inject_at]
    g = torch.Generator().manual_seed(0)
    delta = torch.randn(h.shape, generator=g)
    delta = delta / (delta.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-12)
    delta = delta * (0.10 * h.pow(2).mean(dim=-1, keepdim=True).sqrt())

    with Injector(bb.layers[inject_at], delta):
        with torch.no_grad():
            inj = bb.model(**batch, output_hidden_states=True)

    curve = []
    for li in range(inject_at + 1, len(hs_ref)):
        a, b = hs_ref[li], inj.hidden_states[li]
        d = b - a
        rel = float(d.pow(2).sum().sqrt() / (a.pow(2).sum().sqrt() + 1e-12))
        d_mean = d.mean(dim=1)
        dd = delta.mean(dim=1)
        align = float(((d_mean * dd).sum(-1) / (dd.pow(2).sum(-1) + 1e-12)).mean())
        curve.append({"layer": li - 1, "rel_perturbation": rel, "alignment_mean": align})
    final = curve[-1] if curve else {}
    finite = all(bool(torch.isfinite(t).all()) for t in inj.hidden_states)
    ok = bool(finite
              and final.get("rel_perturbation", 0.0) > THRESHOLDS["B3_min_rel_perturbation"]
              and final.get("rel_perturbation", 9.9) < THRESHOLDS["B3_max_rel_perturbation"])
    return {"ok": ok, "model": bb.name, "inject_at": inject_at, "curve": curve,
            "final": final, "finite": finite,
            "thresholds": {"min_rel_perturbation": THRESHOLDS["B3_min_rel_perturbation"],
                           "max_rel_perturbation": THRESHOLDS["B3_max_rel_perturbation"]},
            "note": ("alignment 接近 1 是残差流的**正常**表现（真模型里注入方向本来就会"
                     "被残差携带）⇒ 只看 rel_perturbation，不拿 alignment 当判据。"
                     "v76 的路线 α 上 alignment 只有 0.63~0.83，是因为那里没有真 embedding/norm。")}


# ---------------------------------------------------------------- B4 层栈健康
def check_stack_health(bb: Backbone, batch: dict) -> dict:
    """真模型的逐层范数/有效秩 —— 含 ViT（检验 v76 的 ViT 缺陷根因假设）。"""
    with torch.no_grad():
        out = bb.model(**batch, output_hidden_states=True)
    hs = out.hidden_states
    rms = [float(h.pow(2).mean().sqrt()) for h in hs]
    ratio = [rms[i + 1] / rms[i] for i in range(len(rms) - 1)]
    eff = [eff_rank(h) for h in hs]
    lo, hi = THRESHOLDS["B4_norm_ratio"]
    ok = bool(all(lo <= r <= hi for r in ratio)
              and all(e > THRESHOLDS["B4_eff_rank_min"] for e in eff))
    return {"ok": ok, "model": bb.name, "n_layers": len(bb.layers),
            "rms_by_layer": rms, "rms_ratio": ratio, "eff_rank_by_layer": eff,
            "min_ratio": min(ratio) if ratio else None,
            "max_ratio": max(ratio) if ratio else None,
            "min_eff_rank": min(eff) if eff else None,
            "bounds": {"ratio": THRESHOLDS["B4_norm_ratio"],
                       "eff_rank_min": THRESHOLDS["B4_eff_rank_min"]}}


# ---------------------------------------------------------------- B5 槽身份可分辨（对照）
def extract_pooled(bb: Backbone, batches: List[dict],
                   slots: Sequence[str] = ("Q", "O"), seed: int = 0
                   ) -> Dict[Tuple[int, str], torch.Tensor]:
    """把一个底座的槽激活抽成共享空间的**样本级**表示。

    为什么是样本级 mean-pool：Qwen 与 BERT 的 tokenizer 不同 ⇒ token 网格根本
    对不齐 ⇒ 跨模型只能按样本配对（与 v76 的"随机向量"相比，这里的每个坐标
    都来自真实文本的真实激活）。
    """
    p = MEAS.projections(seed)
    want = [(ly, slot) for ly in range(N_LAYERS) for slot in slots]
    acc: Dict[Tuple[int, str], List[torch.Tensor]] = {k: [] for k in want}
    for batch in batches:
        st = capture_slots(bb, batch, want)
        for k in want:
            acc[k].append(st[k].mean(dim=1))               # 样本级 mean-pool
    out: Dict[Tuple[int, str], torch.Tensor] = {}
    for k in want:
        a = torch.cat(acc[k], dim=0)                       # [N, D_model]
        # 原生 -> 共享: 按 v62 slot_repr 的约定是 @ P^T (P 是 [DS, D])
        out[k] = a @ p[bb.name].t()                       # -> 共享空间 [N, DS]
    return out


def compare_pooled(pa: Dict[Tuple[int, str], torch.Tensor],
                   pb: Dict[Tuple[int, str], torch.Tensor],
                   pair: str, slots: Sequence[str] = ("Q", "O")) -> dict:
    """真输入下：A 的槽 s 激活能否比 A 的别的槽更好地预测 B 的槽 s 激活？

    这是 v76 的 C5 在真模型上的重跑（同样的共享空间 ridge 口径），
    区别只有一处：**输入从随机向量换成了真实文本**。
    v76 在随机输入下测到「匹配槽 ≈ 错槽」；这里问的是真模型真输入下是否还这样。

    ⚠️ 这是**对照测量**，不是 P1b：没有注入臂，无法排除「所有槽都可互相预测」。
    """
    rows = []
    for ly in range(N_LAYERS):
        for slot in slots:
            other = "O" if slot == "Q" else "Q"
            u = pa[(ly, slot)]
            r2_same, _ = ridge_fit_r2(u, pb[(ly, slot)])
            r2_wrong, _ = ridge_fit_r2(u, pb[(ly, other)])
            rows.append({"layer": ly, "slot": slot,
                         "r2_matched": float(r2_same),
                         "r2_wrong_slot": float(r2_wrong),
                         "delta": float(r2_same - r2_wrong)})
    deltas = [r["delta"] for r in rows]
    n_pos = sum(1 for d in deltas if d > 0)
    frac = n_pos / len(rows) if rows else None
    try:
        from scipy.stats import binomtest
        p_val = float(binomtest(n_pos, len(rows), 0.5, alternative="two-sided").pvalue)
    except Exception:
        p_val = None
    ok = bool(rows
              and frac >= THRESHOLDS["B5_matched_better_min"]
              and (p_val is not None and p_val < THRESHOLDS["B5_sign_test_p_max"]))
    return {"ok": ok, "pair": pair, "n_blocks": len(rows),
            "frac_matched_better": frac, "median_delta": float(np.median(deltas)),
            "sign_test_p": p_val, "n_positive": n_pos,
            "mean_r2_matched": float(np.mean([r["r2_matched"] for r in rows])),
            "mean_r2_wrong": float(np.mean([r["r2_wrong_slot"] for r in rows])),
            "thresholds": {"matched_better_min": THRESHOLDS["B5_matched_better_min"],
                           "sign_test_p_max": THRESHOLDS["B5_sign_test_p_max"]},
            "rows": rows,
            "note": ("对照测量, 不是 P1b: 没有注入臂, 无法排除「所有槽都互相可预测」。"
                     "与 v76 的 C5 同口径, 差别只在输入 (真实文本 vs 随机向量)。")}


def post_hoc_per_slot(b5_all: dict) -> dict:
    """**事后拆解**（明确不参与本代裁决）：按槽看匹配槽占优的情况。

    为什么单独拆：预登记的 B5 是把 Q/O 两个槽混在一起的二值判据。若两个槽的
    行为不同，混在一起会把功效稀释掉。拆开看是**假设生成**，不是翻案 ——
    下一代（S3）若要用某个槽，必须重新预登记并在**新块**上确认，
    不能拿这 16 块既当判据集又当确认集。
    """
    out = {}
    for slot in ("Q", "O"):
        rs = [r for r in b5_all["rows"] if r["slot"] == slot]
        if not rs:
            continue
        n_pos = sum(1 for r in rs if r["delta"] > 0)
        try:
            from scipy.stats import binomtest
            p_val = float(binomtest(n_pos, len(rs), 0.5, alternative="two-sided").pvalue)
        except Exception:
            p_val = None
        out[slot] = {
            "n_blocks": len(rs),
            "n_positive": n_pos,
            "frac_matched_better": n_pos / len(rs),
            "sign_test_p": p_val,
            "median_delta": float(np.median([r["delta"] for r in rs])),
            "mean_delta": float(np.mean([r["delta"] for r in rs])),
            "median_r2_matched": float(np.median([r["r2_matched"] for r in rs])),
            "median_r2_wrong": float(np.median([r["r2_wrong_slot"] for r in rs])),
            "n_blocks_matched_r2_gt_0.1": sum(1 for r in rs if r["r2_matched"] > 0.1),
        }
    return {
        "status": "post_hoc_only_not_used_for_verdict",
        "note": ("事后拆解, **不参与**本代裁决 (B5 仍按预登记的 Q/O 合并判据判 FAIL)。"
                 "仅用于 S3 的设计: 判据粒度要跟机制粒度对齐。"),
        "by_slot": out,
    }


# ---------------------------------------------------------------- 主流程
def run(s1beta: bool = True) -> dict:
    t0 = time.time()
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(min(16, os.cpu_count() or 8))

    fetch_models()
    texts = fetch_texts(N_TEXT)
    labels = [t["label"] for t in texts]
    print("[corpus] ag_news %d 条, 标签分布 %s" % (len(texts), np.bincount(labels).tolist()), flush=True)

    imgs = fetch_images(N_IMG)
    wall = _V75.load_wall()

    # 逐个底座加载、用完即释放: 本机可用内存只有 ~8 GB, 三个 fp32 模型同时常驻
    # 会在 transformers 的权重 materialize 阶段直接 MemoryError。
    b1: Dict[str, dict] = {}
    b2: Dict[str, dict] = {}
    b3: Dict[str, dict] = {}
    b4_models: Dict[str, dict] = {}
    pooled: Dict[str, Dict[Tuple[int, str], torch.Tensor]] = {}
    heads: Dict[str, dict] = {}

    for name in ("Qwen", "BERT", "ViT"):
        bb = Backbone(name)
        heads[name] = dict(bb.heads(), repo=bb.repo, local_dir=bb.local,
                            class_name=type(bb.model).__name__)
        print("[model] %s: %s | layers=%d heads=%s"
              % (name, type(bb.model).__name__, bb.n_layers(), bb.heads()), flush=True)
        if bb.kind == "text":
            batches = text_batches(bb, texts)
        else:
            proc = bb.proc(images=imgs, return_tensors="pt")
            batches = [{k: v for k, v in proc.items()}]
        batch0 = batches[0]
        b1[name] = check_weight_provenance(bb, wall)
        b2[name] = check_layerwise_drive(bb, batch0)
        if bb.kind == "text":
            b3[name] = check_injection(bb, batch0, inject_at=1)
        b4_models[name] = check_stack_health(bb, batch0)
        if bb.kind == "text":
            pooled[name] = extract_pooled(bb, batches, slots=("Q", "O"))
        del bb, batches, batch0
        gc.collect()

    # B4 判据范围 = A1 首代覆盖的底座 (design §2), 与 v76 的 C3 同口径:
    # ViT 的数值作为**范围外记录**保留, 不进判据 (首代不含 ViT)。
    b4 = {"scope": list(HEALTH_SCOPE),
          "in_scope_ok": bool(all(b4_models[m]["ok"] for m in HEALTH_SCOPE)),
          "models": b4_models,
          "out_of_scope": {m: b4_models[m] for m in b4_models if m not in HEALTH_SCOPE},
          "note": ("判据范围 = A1 首代覆盖的 Qwen/BERT (design §2), 与 v76 的 C3 同口径; "
                   "ViT 只作范围外记录, 不进判据。")}
    b5 = compare_pooled(pooled["Qwen"], pooled["BERT"], "Qwen->BERT")
    b5b = compare_pooled(pooled["BERT"], pooled["Qwen"], "BERT->Qwen")

    # B5 合并两个方向 (16 块)
    rows5 = b5["rows"] + b5b["rows"]
    deltas = [r["delta"] for r in rows5]
    n_pos = sum(1 for d in deltas if d > 0)
    frac = n_pos / len(rows5)
    try:
        from scipy.stats import binomtest
        p5 = float(binomtest(n_pos, len(rows5), 0.5, alternative="two-sided").pvalue)
    except Exception:
        p5 = None
    b5_all = {
        "ok": bool(frac >= THRESHOLDS["B5_matched_better_min"]
                   and p5 is not None and p5 < THRESHOLDS["B5_sign_test_p_max"]),
        "n_blocks": len(rows5), "frac_matched_better": frac,
        "median_delta": float(np.median(deltas)), "sign_test_p": p5,
        "n_positive": n_pos,
        "mean_r2_matched": float(np.mean([r["r2_matched"] for r in rows5])),
        "mean_r2_wrong": float(np.mean([r["r2_wrong_slot"] for r in rows5])),
        "directions": {"Qwen->BERT": b5, "BERT->Qwen": b5b},
        "rows": rows5,
        "note": ("对照测量, 不是 P1b: 没有注入臂。与 v76 的 C5 同口径, "
                 "差别只在输入 (真实文本 vs 随机向量)。"),
    }

    criteria = {
        "B1_weight_same_source": all(v["ok"] for v in b1.values()),
        "B2_layerwise_drive_ok": all(v["ok"] for v in b2.values()),
        "B3_injection_path_ok": all(v["ok"] for v in b3.values()),
        "B4_stack_health_ok": b4["in_scope_ok"],
        "B5_slot_identity_resolvable": b5_all["ok"],
    }
    verdict = "S1B_PASS" if all(criteria.values()) else "S1B_FAIL"

    return {
        "protocol": {
            "route": "beta (真 HF 模型逐层驱动 + 真实输入)",
            "scope": "S1beta 机械可行性 + 真模型侧前置测量; 不含 P1a/P1b/P2/P3 的注入臂",
            "seed": 0,
            "device": "cpu",
            "dtype": "float32",
            "models": heads,
            "corpus": {
                "text": "fancyzhx/ag_news (train 前 %d 条), 经 HF datasets-server rows API 取" % N_TEXT,
                "image": "ethz/food101 (train 前 %d 张), 同 API" % N_IMG,
                "labels": "随文本一起取回 (ag_news 4 类), 本代不用它做能力指标",
                "cache": str(CACHE_DIR),
            },
            "seq": SEQ,
            "batch": BATCH,
            "thresholds": THRESHOLDS,
            "reused_from": ["v62 measure.py (SLOT_PATHS/projections)",
                            "v75 g2_conjugation.py (load_wall)",
                            "v76 a1_forward.py (eff_rank/ridge_fit_r2)"],
            "provenance_sha256": {k: sha256_of(v) for k, v in PROVENANCE_FILES.items()},
            "declared_modeling_choices": [
                "fp32 on CPU (避开 WDDM 下被桌面占满的显存; 同时让权重同源比对是逐位精确的)",
                "注入用 forward_pre_hook 打在 decoder layer 输入处, 不手工逐层调用 "
                "(真模型层 forward 签名随 transformers 版本变动, 手工调用易漏 position_embeddings/cache)",
                "槽激活: 投影类槽取模块输出, o_proj/down_proj 类槽取模块输入",
                "跨模型可比单位 = 样本级 mean-pool (两个 tokenizer 的 token 网格不同, 没法逐 token 配对)",
            ],
        },
        "B1_weight_provenance": b1,
        "B2_layerwise_drive": b2,
        "B3_injection": b3,
        "B4_stack_health": b4,
        "B5_slot_identity": b5_all,
        "criteria": criteria,
        "verdict": verdict,
        "post_hoc_per_slot": post_hoc_per_slot(b5_all),
        "limits": [
            "本代只证机制与真模型侧的前置性质, 不含任何端到端任务指标",
            "B5 是对照测量, 不是 P1b (无注入臂, 无法排除所有槽互相可预测)",
            "槽激活按 v62 的槽权重语义取 (投影输出 / 反投影输入), 不是模型自述的槽",
            "注入不改变任何底座权重 ⇒ 与权重级 G1/G2 正交",
            "ViT 本代只进 B1/B4 (权重同源 + 层栈健康), 不进 B5 (跨模型槽可比)",
            "B5 未过的是**功效**不是方向: 匹配槽占优 12/16、中位差为正, 但双侧符号检验 p>0.05 "
            "(n=16 时需 >=13 个正号) ⇒ 读作「不可判定」, 不得读作「槽身份不可分辨」",
        ],
        "runtime_sec": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_PKG / "results" / "metrics.json"))
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    res = run()
    print("      criteria: %s" % res["criteria"], flush=True)
    print("      verdict: %s  (%.0fs)" % (res["verdict"], res["runtime_sec"]), flush=True)
    if not args.no_write:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\r\n") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print("      wrote %s" % out, flush=True)
    return 0 if res["verdict"] == "S1B_PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
