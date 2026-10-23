#!/usr/bin/env python3
"""eval_brain_alignment.py — 「7 基模脑对齐表」评测脚本（D2 交付物）。

## 这张表要回答什么

把 7 个 EEG 基模的**冻结表征**放进 THINGS-EEG 输入域，然后用**两个口径不同的
脑对齐指标**同时打分：

| 指标 | 口径 | 复用的既有实现 | 要回答的问题 |
|---|---|---|---|
| **A：RSA** | 冻结表征 ↔ ViT-H-14 图像表征的 RDM 秩相关 | `exp65.rdm_full / spearman / rsa_with_null / p_from_null` | 表征里有没有**图像语义结构** |
| **B：ridge 200 选 1** | 冻结表征上闭式 ridge → test split 200 选 1 | `exp73.per_rep_targets / fit_and_decode` | 表征里的信息**能不能读出**图像 |

两个指标**同表并列、不给主次**（D3）：A 只看结构、B 只看可解码性，
一个模型可能 A 高 B 低，反之亦然 —— 只报一个会把结论带偏。

## 三条口径纪律（混用比算错更糟）

1. **基线行同协议**：表里 "630-d 原始特征" 那行，用**同一个 train 图像数**
   重算，不是直接抄 exp73 的 0.0309。否则基线与模型行不可比。
2. **池化规则预先写死**：默认 `mean_all`（全部 token 取均值，含全局 token）。
3. **指标 B 复用 exp73 的 `fit_and_decode` 本体**（试次级 Gram `A0=G_full/n_rep`）。
   ⚠️ exp73 的方向审计结论：**图级降维在解码方向不成立**
   （图级 Gram 漏掉图内散布 Δ），所以这里必须是试次级。

## 用法

云端（桶已挂载）：

    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/eval_brain_alignment.py \
      -v hf://buckets/illusion-brain/training-data:/training:ro \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor cpu-upgrade --timeout 90m -d

本地自检（合成数据，不碰网络）：

    python eval_brain_alignment.py --self-test
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
PKG_DIR = EXP_DIR.parent / "models"
for _p in (str(EXP_DIR), str(PKG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---- 复用既有实现（口径必须逐字一致，不重写）----
import exp65_tier0_rsa_fix as E65          # noqa: E402
import exp73_decode200 as E73              # noqa: E402
import domain_adapters as DA               # noqa: E402

# ---- 桶内路径 ----
TRAIN_4D = Path("/domains/omnibrain4d/eeg_train")   # (16540, 4, 63, 250)
TEST_4D = Path("/domains/omnibrain4d/eeg")           # (200, 80, 63, 250)
FEAT_TEST = "/training/eeg/eeg-image-decode/ViT-H-14_features_test.pt"
FEAT_TRAIN = "/training/eeg/eeg-image-decode/ViT-H-14_features_train.pt"
MODELS_DIR = Path("/training/models")
OUT_DIR = Path("/domains/runs/illusionbrain-align")

ALPHAS = E73.ALPHAS          # [0.001 … 1000]，与 exp73 同
N_NULL = 8
N_PERM_RSA = 200
SEED = 0
BASELINE_ROW = "630-d 原始特征"


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 指标 A：RSA（结构对齐）
# ---------------------------------------------------------------------------
def metric_a_rsa(trial_feats, img_feats, n_perm=N_PERM_RSA, rng=None):
    """冻结表征 ↔ 图像表征的 RSA。

    trial_feats : (n_img, n_rep, D)  逐试次特征
    img_feats   : (n_img, D_img)     图像特征（行与 trial_feats 的图像一一对应）

    归一化链（与 exp65 的 "demean" 变体同构）：
        试次平均 → 跨图去共同成分 → 逐维 z-score → rdm_full → RSA

    对 (C, T) 的 EEG，exp65 的 demean + 逐通道 z-score 就是**逐维跨图**操作；
    这里 D 维特征做同样的逐维操作，结构一致。
    """
    n_img = trial_feats.shape[0]
    if n_img != img_feats.shape[0]:
        raise ValueError(f"图像数不一致：{n_img} vs {img_feats.shape[0]}")
    X = trial_feats.mean(axis=1).astype(np.float64)      # 试次平均
    X = X - X.mean(axis=0, keepdims=True)                 # 去共同成分
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-12
    X = (X - mu) / sd                                     # 逐维 z-score
    R_eeg = E65.rdm_full(X)
    R_img = E65.rdm_full(img_feats.astype(np.float64))
    r, null = E65.rsa_with_null(R_eeg, R_img, n_perm, rng)
    return {
        "r_img": float(r),
        "p_img": float(E65.p_from_null(r, null)),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std()),
        "null_p95": float(np.percentile(np.abs(null), 95)),
        "percentile": float((null < r).mean()),
        "n_perm": int(n_perm),
        "n_img": int(n_img),
    }


# ---------------------------------------------------------------------------
# 指标 B：闭式 ridge → test split 200 选 1（可解码性）
# ---------------------------------------------------------------------------
def metric_b_decode(Xrep, Xte, IMG_train, IMG_test, rng, n_null=N_NULL,
                    cols=None):
    """冻结表征上的闭式 ridge → 200 选 1。

    **直接调用 exp73 的 `fit_and_decode`**（试次级 Gram `A0 = G_full/n_rep`），
    不重写、不改口径。exp73 的方向审计已证明图级 Gram 在解码方向不成立。

    Xrep : (n_img_tr, n_rep_tr, D)  训练试次特征（已 z-score）
    Xte  : (n_img_te, n_rep_te, D)  测试试次特征（已 z-score）
    """
    res = E73.fit_and_decode(Xrep, Xte, IMG_train, IMG_test, cols,
                             ALPHAS, rng, n_null=n_null)
    return res


def zscore_fit(X):
    flat = X.reshape(-1, X.shape[-1])
    return flat.mean(axis=0), flat.std(axis=0) + 1e-6


def zscore_apply(X, mu, sd):
    return (X - mu) / sd


# ---------------------------------------------------------------------------
# 特征抽取：基模 → 逐试次特征
# ---------------------------------------------------------------------------
def model_trial_features(handle, eeg, have_chans=None, batch_size=32,
                         device="cpu", pool="mean_all", log_every=None):
    """把 (n_img, n_rep, C, T) 的 EEG 适配到基模域并跑前向。

    返回 {"ok", "features": (n_img, n_rep, D) | None, "info"}
    """
    import registry as REG
    n_img, n_rep = eeg.shape[0], eeg.shape[1]
    flat = eeg.reshape(n_img * n_rep, eeg.shape[2], eeg.shape[3])
    adapted, info = DA.adapt(flat, handle.spec, have_chans)
    res = REG.encode_trials(handle, adapted, batch_size=batch_size,
                             device=device, pool=pool)
    if not res["ok"]:
        return {"ok": False, "features": None, "info": {**res["info"],
                                                        "adapt": info.to_dict()}}
    X = res["features"].reshape(n_img, n_rep, -1)
    return {"ok": True, "features": X,
            "info": {**res["info"], "adapt": info.to_dict()}}


def baseline_630_features(eeg):
    """630-d 原始 EEG 特征（exp73 的 `per_rep_targets`，逐字复用）。"""
    return E73.per_rep_targets(eeg)


# ---------------------------------------------------------------------------
# 单被试
# ---------------------------------------------------------------------------
def run_subject(sid, IMG_train, IMG_test, handles, args, rng):
    t0 = time.time()
    tr_fp = TRAIN_4D / f"sub-{sid:02d}.npz"
    te_fp = TEST_4D / f"sub-{sid:02d}.npz"
    if not tr_fp.exists() or not te_fp.exists():
        return None

    n_train_images = args.train_images or None
    need_train = any(h.ok for h in handles.values()) or args.baseline

    # --- 测试集（指标 A + 指标 B 都要）---
    eeg_te = np.load(te_fp)["eeg"]
    log(f"  sub-{sid:02d} test {eeg_te.shape}")
    have_chans = args.have_chans or None

    out = {"subject": f"sub-{sid:02d}", "rows": [], "n_train_images": n_train_images}

    # --- 630-d 基线行 ---
    if args.baseline:
        Xte_630 = baseline_630_features(eeg_te)
        eeg_tr = np.load(tr_fp)["eeg"] if need_train else None
        if eeg_tr is not None:
            Xrep_630 = baseline_630_features(eeg_tr if n_train_images is None
                                             else eeg_tr[:n_train_images])
            del eeg_tr
        else:
            Xrep_630 = None
        row = {"name": BASELINE_ROW, "kind": "baseline_eeg", "n_params": 0,
               "features": None}
        if Xrep_630 is not None:
            mu, sd = zscore_fit(Xrep_630)
            rb = metric_b_decode(zscore_apply(Xrep_630, mu, sd),
                                 zscore_apply(Xte_630, mu, sd),
                                 IMG_train[:n_train_images] if n_train_images else IMG_train,
                                 IMG_test, rng, n_null=args.n_null)
            row["metric_b"] = rb
            row["metric_a"] = metric_a_rsa(Xte_630, IMG_test, args.n_perm, rng)
            log(f"    [基线 630-d] A r={row['metric_a']['r_img']:+.4f} "
                f"p={row['metric_a']['p_img']:.3f} | "
                f"B acc1={rb['acc1_cos']:.4f} (零中位 {np.median(rb['null_acc1_cos']):.4f})")
        out["rows"].append(row)

    # --- 各基模 ---
    for name, h in handles.items():
        if not h.ok:
            out["rows"].append({"name": name, "kind": "base_model", "ok": False,
                                "error": h.error})
            continue
        r = {"name": name, "kind": "base_model", "ok": True,
             "n_params": h.n_params, "load_path": h.load_path,
             "license": h.attribution.get("license")}
        try:
            fte = model_trial_features(h, eeg_te, have_chans, args.batch_size,
                                       args.device, args.pool)
            if not fte["ok"]:
                r["ok"] = False
                r["error"] = fte["info"].get("error")
                r["info"] = fte["info"]
                out["rows"].append(r)
                log(f"    [{name}] ✗ {r['error']}")
                continue
            Xte = fte["features"]
            r["info"] = fte["info"]
            r["metric_a"] = metric_a_rsa(Xte, IMG_test, args.n_perm, rng)

            if need_train:
                eeg_tr = np.load(tr_fp)["eeg"]
                sub = eeg_tr if n_train_images is None else eeg_tr[:n_train_images]
                ftr = model_trial_features(h, sub, have_chans, args.batch_size,
                                           args.device, args.pool)
                del eeg_tr, sub
                if not ftr["ok"]:
                    r["ok"] = False
                    r["error"] = f"train 抽取失败：{ftr['info'].get('error')}"
                    out["rows"].append(r)
                    log(f"    [{name}] ✗ {r['error']}")
                    continue
                Xrep = ftr["features"]
                mu, sd = zscore_fit(Xrep)
                IMG_tr = IMG_train if n_train_images is None else IMG_train[:n_train_images]
                rb = metric_b_decode(zscore_apply(Xrep, mu, sd),
                                     zscore_apply(Xte, mu, sd),
                                     IMG_tr, IMG_test, rng, n_null=args.n_null)
                r["metric_b"] = rb
                del Xrep
            log(f"    [{name}] A r={r['metric_a']['r_img']:+.4f} "
                f"p={r['metric_a']['p_img']:.3f} | "
                + (f"B acc1={rb['acc1_cos']:.4f} α={rb['alpha']:g} "
                   f"(零中位 {np.median(rb['null_acc1_cos']):.4f})"
                   if "metric_b" in r else "B 跳过"))
        except Exception as e:
            r["ok"] = False
            r["error"] = f"{type(e).__name__}: {e}"[:300]
            r["trace"] = traceback.format_exc()[-800:]
            log(f"    [{name}] ✗ {r['error']}")
        out["rows"].append(r)

    del eeg_te
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


# ---------------------------------------------------------------------------
# 汇总与出表
# ---------------------------------------------------------------------------
def summarize(per_subject):
    """按行名聚合（跨被试中位 + 极值 + 显著计数）。"""
    agg = {}
    for rec in per_subject:
        for row in rec.get("rows", []):
            key = row["name"]
            a = agg.setdefault(key, {"name": key, "kind": row.get("kind"),
                                     "n_subjects": 0, "n_ok": 0,
                                     "A_r": [], "A_p": [], "B_acc1": [],
                                     "B_null": [], "B_acc5": [],
                                     "B_featavg": [], "B_predavg": [],
                                     "n_params": row.get("n_params"),
                                     "errors": []})
            a["n_subjects"] += 1
            if not row.get("ok", True):
                a["errors"].append(row.get("error"))
                continue
            a["n_ok"] += 1
            if "metric_a" in row:
                a["A_r"].append(row["metric_a"]["r_img"])
                a["A_p"].append(row["metric_a"]["p_img"])
            if "metric_b" in row:
                a["B_acc1"].append(row["metric_b"]["acc1_cos"])
                a["B_null"].append(float(np.median(row["metric_b"]["null_acc1_cos"])))
                a["B_acc5"].append(row["metric_b"].get("acc5_cos"))
                a["B_featavg"].append(row["metric_b"].get("acc1_featavg_cos"))
                a["B_predavg"].append(row["metric_b"].get("acc1_predavg_cos"))

    def med(v):
        v = [x for x in v if x is not None]
        return float(np.median(v)) if v else None

    for a in agg.values():
        a["A_r_median"] = med(a["A_r"])
        a["A_p_median"] = med(a["A_p"])
        a["A_p_min"] = float(np.min(a["A_p"])) if a["A_p"] else None
        a["A_n_significant"] = int(sum(1 for p in a["A_p"] if p is not None and p < 0.05))
        a["B_acc1_median"] = med(a["B_acc1"])
        a["B_acc1_min"] = float(np.min(a["B_acc1"])) if a["B_acc1"] else None
        a["B_acc1_max"] = float(np.max(a["B_acc1"])) if a["B_acc1"] else None
        a["B_acc5_median"] = med(a["B_acc5"])
        a["B_null_median"] = med(a["B_null"])
        a["B_featavg_median"] = med(a["B_featavg"])
        a["B_predavg_median"] = med(a["B_predavg"])
        a["B_ratio_vs_random"] = (a["B_acc1_median"] / 0.005
                                  if a["B_acc1_median"] else None)
        a["B_n_above_null"] = int(sum(
            1 for r, n in zip(a["B_acc1"], a["B_null"]) if r > n))
    return agg


def render_table(agg, order=None):
    """出 markdown 表（两个指标并列，不分主次）。"""
    names = order or list(agg.keys())
    lines = []
    lines.append("| 模型 | 指标A r_img (中位) | 指标A p (中位) | A 显著被试 | "
                 "指标B acc1 (中位) | B 零中位 | B ×随机 | B top5 | B 特征平均 | "
                 "B 预测平均 | 参数 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for n in names:
        a = agg.get(n)
        if not a:
            continue
        def f(v, fmt="{:.4f}"):
            return fmt.format(v) if v is not None else "—"
        npar = f"{a['n_params']:,}" if a.get("n_params") else "—"
        lines.append(
            f"| {n} | {f(a['A_r_median'], '{:+.4f}')} | "
            f"{f(a['A_p_median'], '{:.3f}')} | {a['A_n_significant']}/{a['n_ok']} | "
            f"{f(a['B_acc1_median'])} | {f(a['B_null_median'])} | "
            f"{f(a['B_ratio_vs_random'], '{:.1f}')} | {f(a['B_acc5_median'])} | "
            f"{f(a['B_featavg_median'])} | {f(a['B_predavg_median'])} | {npar} |")
    return "\n".join(lines)


def merge_write(per_subject, path):
    """按被试合并增量落盘（补跑不覆盖已有分段）。"""
    merged = {}
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            merged = {r["subject"]: r for r in old.get("per_subject", [])}
        except Exception:
            merged = {}
    for r in per_subject:
        merged[r["subject"]] = r
    ordered = [merged[k] for k in sorted(merged)]
    path.write_text(json.dumps({"per_subject": ordered}, indent=2,
                                ensure_ascii=False), encoding="utf-8")
    return ordered


# ---------------------------------------------------------------------------
# 自测（合成数据，完整走一遍真实代码路径）
# ---------------------------------------------------------------------------
def _fake_braindecode_module():
    """构造一个假的 braindecode.models 命名空间 + 一个 2D 网格假模型。

    目的不是"证明模型对"，而是把 **registry 的装载路径 → domain 适配 → 前向 →
    池化 → 指标 A → 指标 B → 出表** 这条链在本地跑通，
    避免把一整轮云端算力押在一条会崩的链上。
    """
    import torch
    from torch import nn
    import interfaces as IF

    class FakeGridModel(nn.Module):
        """(B, C, T) → tokens (B, rows*cols+1, d_patch)，带 1 个 cls token。"""

        def __init__(self, n_chans=63, n_times=250, patch_ch=9, patch_t=25,
                     d_patch=64, n_layers=2):
            super().__init__()
            self.n_chans, self.n_times = n_chans, n_times
            self.patch_ch, self.patch_t = patch_ch, patch_t
            self.rows, self.cols = n_times // patch_t, n_chans // patch_ch
            self.patch = nn.Linear(patch_ch * patch_t, d_patch)
            self.cls = nn.Parameter(torch.zeros(1, 1, d_patch))
            self.blocks = nn.ModuleList([
                IF.Block(IF.SelfAttention1D(d_patch, 4), IF.FFN(d_patch, 2 * d_patch),
                         d_patch) for _ in range(n_layers)])

        @classmethod
        def from_pretrained(cls, path, n_outputs=None):
            cfg = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
            import inspect
            avail = set(inspect.signature(cls.__init__).parameters) - {"self"}
            return cls(**{k: v for k, v in cfg.items() if k in avail})

        def forward(self, x):
            B, C, T = x.shape
            p = x.unfold(-1, self.patch_t, self.patch_t) \
                 .unfold(1, self.patch_ch, self.patch_ch)      # (B, cols, rows, pc, pt)
            p = p.permute(0, 2, 1, 3, 4).reshape(B, self.rows * self.cols, -1)
            tok = self.patch(p)
            cls = self.cls.expand(B, 1, -1)
            tb = IF.TokenBatch(torch.cat([cls, tok], dim=1), (self.rows, self.cols))
            for blk in self.blocks:
                tb = blk(tb)
            return {"embedding": tb.tokens}

    ns = type("FakeBD", (), {})()
    ns.CBraMod = FakeGridModel
    return ns


def _synthetic_bundle(rng, n_img_tr=180, n_img_te=60, n_rep_tr=4, n_rep_te=8,
                      d_img=32, d_eeg=63 * 250, seed=7):
    """造一份**有信号**的合成数据：IMG → EEG（线性）→ 基模 → 表征。

    这样：指标 A 应当 > 0，指标 B 应当显著高于随机。
    """
    import torch
    IMG_tr = rng.normal(size=(n_img_tr, d_img))
    IMG_te = rng.normal(size=(n_img_te, d_img))
    W = rng.normal(size=(d_img, d_eeg)) * 0.05
    noise = 0.25

    def make(IMG, n_rep):
        eeg = IMG @ W + noise * rng.normal(size=(IMG.shape[0], d_eeg))
        eeg = eeg.reshape(IMG.shape[0], 63, 250)
        eeg = np.repeat(eeg[:, None, :, :], n_rep, axis=1)
        return eeg.astype(np.float32)

    return {"IMG_train": IMG_tr, "IMG_test": IMG_te,
            "eeg_train": make(IMG_tr, n_rep_tr), "eeg_test": make(IMG_te, n_rep_te)}


def self_test():
    log("=== 自测：合成数据走完整链路 ===")
    rng = np.random.default_rng(11)
    bd = _fake_braindecode_module()

    # --- (1) registry 装载（from_pretrained 本地目录）---
    import tempfile
    import registry as REG
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "cbramod"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({
            "n_chans": 63, "n_times": 250, "patch_size": [9, 25],
            "d_patch": 64, "n_layers": 2}), encoding="utf-8")
        (d / "ATTRIBUTION.txt").write_text("Source: synthetic\nLicense: MIT\n",
                                           encoding="utf-8")
        h = REG.load_base_model("cbramod", models_dir=Path(td),
                                braindecode_models=bd)
        assert h.ok, f"装载失败：{h.error}"
        assert h.n_params > 0
        assert h.attribution.get("license") == "MIT"
        log(f"  ✓ registry 装载：{h.n_params:,} 参数，载入路径 {h.load_path}")

        # --- (2) 布局推断（必须是 config 权威，不能是猜的）---
        lay = REG.infer_layout(71, h.cfg, h.meta, h.model)
        assert lay["confidence"] == "config", f"布局来源不对：{lay}"
        assert lay["layout"] == (10, 7), lay
        assert lay["n_global"] == 1, lay
        log(f"  ✓ 布局推断：{lay['layout']}（{lay['confidence']}）"
            f" 全局 token {lay['n_global']} —— {lay['reason']}")

        # --- (3) domain 适配 + 前向 + 池化 ---
        B = _synthetic_bundle(rng)
        r = model_trial_features(h, B["eeg_test"], batch_size=16, pool="mean_all")
        assert r["ok"], f"抽取失败：{r['info'].get('error')}"
        X = r["features"]
        assert X.shape[:2] == (B["eeg_test"].shape[0], B["eeg_test"].shape[1]), X.shape
        assert r["info"]["layout"] == [10, 7]
        log(f"  ✓ 抽取：{B['eeg_test'].shape[:2]} → 特征 {X.shape}"
            f"（token 方式 {r['info']['token_how']}）")

        # --- (4) 指标 A：应有正 RSA ---
        a = metric_a_rsa(X, B["IMG_test"], n_perm=100, rng=rng)
        log(f"  ✓ 指标A：r={a['r_img']:+.4f} p={a['p_img']:.3f}")
        assert a["r_img"] > 0.0, "有信号的合成数据上 RSA 应为正"

        # --- (5) 指标 B：应显著高于随机 ---
        rtr = model_trial_features(h, B["eeg_train"], batch_size=16)
        assert rtr["ok"], rtr["info"].get("error")
        Xtr = rtr["features"]
        mu, sd = zscore_fit(Xtr)
        b = metric_b_decode(zscore_apply(Xtr, mu, sd),
                             zscore_apply(X, mu, sd),
                             B["IMG_train"], B["IMG_test"], rng, n_null=4)
        null_med = float(np.median(b["null_acc1_cos"]))
        log(f"  ✓ 指标B：acc1_cos={b['acc1_cos']:.4f}（零中位 {null_med:.4f}，"
            f"α={b['alpha']:g}）")
        assert b["acc1_cos"] > 3 * 0.005, "有信号时 200 选 1 应远高于随机"

        # --- (6) 630-d 基线行也走同一条路（口径一致性）---
        X630_te = baseline_630_features(B["eeg_test"])
        X630_tr = baseline_630_features(B["eeg_train"])
        assert X630_te.shape[-1] == 630, X630_te.shape
        mu6, sd6 = zscore_fit(X630_tr)
        b6 = metric_b_decode(zscore_apply(X630_tr, mu6, sd6),
                             zscore_apply(X630_te, mu6, sd6),
                             B["IMG_train"], B["IMG_test"], rng, n_null=4)
        a6 = metric_a_rsa(X630_te, B["IMG_test"], n_perm=100, rng=rng)
        log(f"  ✓ 630-d 基线行：A r={a6['r_img']:+.4f} B acc1={b6['acc1_cos']:.4f}"
            f"（零中位 {np.median(b6['null_acc1_cos']):.4f}）")
        assert b6["acc1_cos"] > 3 * 0.005, "630-d 基线在有信号数据上也应高于随机"

        # --- (7) 出表 ---
        agg = summarize([{"subject": "sub-00", "rows": [
            {"name": BASELINE_ROW, "kind": "baseline_eeg", "ok": True,
             "n_params": 0, "metric_a": a6, "metric_b": b6},
            {"name": "cbramod", "kind": "base_model", "ok": True,
             "n_params": h.n_params, "metric_a": a, "metric_b": b}]}])
        md = render_table(agg, order=[BASELINE_ROW, "cbramod"])
        assert "630-d 原始特征" in md and "cbramod" in md
        log("\n" + md)

    log("=== 自测通过 ===")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all",
                    help="逗号分隔的基模名，默认 all")
    ap.add_argument("--only-subjects", default="")
    ap.add_argument("--train-images", type=int, default=2000,
                    help="指标 B 的训练图像数（0 = 全量 16540）")
    ap.add_argument("--pool", default="mean_all")
    ap.add_argument("--n-perm", type=int, default=N_PERM_RSA)
    ap.add_argument("--n-null", type=int, default=N_NULL)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-baseline", dest="baseline", action="store_false")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    args.train_images = args.train_images or None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    import torch
    import registry as REG

    ftr = torch.load(FEAT_TRAIN, map_location="cpu", weights_only=False)
    IMG_train = ftr["img_features"].numpy().astype(np.float64)
    fte = torch.load(FEAT_TEST, map_location="cpu", weights_only=False)
    IMG_test = fte["img_features"].numpy().astype(np.float64)
    log(f"ViT-H-14 特征：train {IMG_train.shape} / test {IMG_test.shape}")

    names = (REG.ORDER if args.models == "all"
             else [s.strip() for s in args.models.split(",") if s.strip()])
    handles = {}
    try:
        import braindecode.models as bdmodels
    except Exception as e:
        log(f"✗ braindecode 不可用：{e!r}")
        bdmodels = None
    for nm in names:
        handles[nm] = REG.load_base_model(nm, models_dir=MODELS_DIR,
                                          braindecode_models=bdmodels)
    log(f"装载成功：{[n for n, h in handles.items() if h.ok]}")

    sids = ([int(s) for s in args.only_subjects.split(",") if s.strip()]
            if args.only_subjects else list(range(1, 11)))
    rng = np.random.default_rng(SEED)

    per_subject = []
    for sid in sids:
        log(f"\n===== sub-{sid:02d} =====")
        rec = run_subject(sid, IMG_train, IMG_test, handles, args, rng)
        if rec is None:
            log(f"  sub-{sid:02d}: 数据缺失，跳过")
            continue
        per_subject.append(rec)
        # ⚠️ **每个被试单独落一个文件**。
        # cpu-upgrade 上跑整表要十几小时 ⇒ 实际是「10 个被试 = 10 个并行 Job」，
        # 多个 Job 同时读写同一个合并文件会互相覆盖 ⇒ 必须按被试分文件，
        # 最后再本地合并。
        (out_dir / f"alignment_sub-{sid:02d}.json").write_text(
            json.dumps({"per_subject": [rec]}, indent=2, ensure_ascii=False,
                       default=str), encoding="utf-8")
        if len(sids) > 1:
            per_subject = merge_write(per_subject, out_dir / "alignment_per_subject.json")

    agg = summarize(per_subject)
    order = ([BASELINE_ROW] if args.baseline else []) + \
            [n for n in names if n in agg]
    md = render_table(agg, order=order)
    log("\n" + "=" * 100)
    log("7 基模脑对齐表（指标 A：RSA 结构对齐；指标 B：ridge 200 选 1 可解码性）")
    log("=" * 100)
    log(md)
    log(f"\n训练图像数：{args.train_images or '全量'} | 池化：{args.pool} | "
        f"置换 {args.n_perm} 次 | 零分布 {args.n_null} 次 | 耗时 {time.time()-t0:.0f}s")

    (out_dir / "alignment_table.md").write_text(md, encoding="utf-8")
    (out_dir / "alignment_full.json").write_text(json.dumps(
        {"schema_version": "1.0", "script": "eval_brain_alignment",
         "config": vars(args), "summary": agg, "per_subject": per_subject,
         "elapsed_s": round(time.time() - t0, 1)},
        indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"结果写入 {out_dir}")


if __name__ == "__main__":
    main()
