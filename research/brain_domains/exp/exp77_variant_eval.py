#!/usr/bin/env python3
"""exp77_variant_eval.py — 阶段 D：跨模型重组变体的脑对齐评测（**零件资格赛**）。

## 这轮要回答什么

「7 基模脑对齐表」和 exp75 的逐层曲线已经把两件事钉死了：

1. **基模的冻结表征装得下图像语义，却不以线性可读的形式保留它**
   （指标 B 全员随机水平 1.0–1.2×，630-d 原始特征 2.2×）；
2. **语义在模型内部的位置因模型而异**（cbramod 铺满 12 层 / signal-jepa 只在
   第 0 层且深层显著为负 / mirepnet 藏在 FFN / braintokenizer 的语义是
   **量化**造出来的）。

⇒ 阶段 D 问的是：**这些脑部件有没有资格进联合训练骨架？**
判据 = 把供体部件接进宿主后，**脑对齐不掉**（甚至更好）。

## 变体口径

| 行 | 构造 | 作用 |
|---|---|---|
| `cbramod` | 宿主（真实模型，未改） | 宿主基线；**同时是跨轮复现检查**（应复现基线表的 cbramod 行） |
| `v1_sjepa_l0` | cbramod ⊕ signal-jepa 第 0 层 | 供体贡献 |
| `v1_sjepa_l0_shuf` | cbramod ⊕ **打乱**的 signal-jepa 第 0 层 | **维度膨胀的零信息基线** |
| `v2_mirepnet_ffn` | cbramod ⊕ mirepnet 的 FFN | 供体贡献 |
| `v2_mirepnet_ffn_shuf` | 同上（打乱） | 对照 |
| `v3_braintok_code` | cbramod ⊕ braintokenizer 的量化码 | 供体贡献 |
| `v3_braintok_code_shuf` | 同上（打乱） | 对照 |

⚠️ **为什么必须带 `_shuf` 对照**：拼接让特征维度变大 ⇒
RSA 的 RDM 混合与 ridge 的参数增多都会让分数**天然变好**，
那不是供体的功劳。对照行维度完全相同、边际统计不变，
只是供体的**图像对应关系被摧毁**（训练/测试各自独立打乱）。
⇒ **变体 − 对照** 才是「供体贡献」；单看变体高低没有意义。

## 供体层特征的口径

用 exp75 的同一套抽取与池化（`normalize_token_tensor` → `mean_all`），
但**只挂目标模块的 hook**（exp75 挂全部模块是为了画曲线；
正式跑表要 24000 试次，全模块收集要 3 GB 内存 ⇒ 扛不住）。

## 用法

云端（桶已挂载）：

    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" \
      python -u /domains/scripts/exp77_variant_eval.py \
      -v hf://buckets/illusion-brain/training-data:/training:ro \
      -v hf://buckets/illusion-brain/brain-domains:/domains \
      --flavor cpu-upgrade --timeout 90m -d \
      --bd-version stable --only-subjects 1 \
      --out-dir /domains/runs/illusionbrain-align/variants

本地自检（合成数据，不碰网络）：

    python exp77_variant_eval.py --self-test
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
import eval_brain_alignment as EVAL   # noqa: E402  ← 指标与抽取全部复用，不重写
import variants as V               # noqa: E402

TRAIN_4D = Path("/domains/omnibrain4d/eeg_train")
TEST_4D = Path("/domains/omnibrain4d/eeg")
FEAT_TEST = "/training/eeg/eeg-image-decode/ViT-H-14_features_test.pt"
FEAT_TRAIN = "/training/eeg/eeg-image-decode/ViT-H-14_features_train.pt"
MODELS_DIR = Path("/training/models")
OUT_DIR = Path("/domains/runs/illusionbrain-align/variants")

SEED = 0
CHUNK = 2000          # 分块适配：一次 24000 试次会吃掉 3 GB/Job（见 EVAL 的注释）


def log(*a):
    print(*a, flush=True)


def _ver_tag(v):
    return "stable" if (v or "stable").strip() == "stable" else "dev"


def _bd_install_target(v):
    v = (v or "stable").strip()
    if v == "stable":
        return "stable"
    if v == "dev":
        return BS.BD_DEV_VERSION
    return v


# ---------------------------------------------------------------------------
# 供体层特征：定点 hook（只抓一个模块）
# ---------------------------------------------------------------------------
def extract_module_features(handle, eeg, module_path, have_chans, args):
    """抓指定模块的输出 → (n_img, n_rep, D)。

    口径与 exp75 逐层曲线**逐字一致**：
    `normalize_token_tensor` → `pool_tokens(mean_all)` → 按输入批大小截断。
    差别只有一个：exp75 挂全部模块（画曲线用），这里只挂目标模块
    （24000 试次下全模块收集要 3 GB 内存）。
    """
    import torch
    import registry as REG

    n_img, n_rep = eeg.shape[0], eeg.shape[1]
    flat = eeg.reshape(n_img * n_rep, eeg.shape[2], eeg.shape[3])
    try:
        mod = handle.model.get_submodule(module_path)
    except Exception as e:
        raise RuntimeError(f"供体模型里没有模块 {module_path}（{e}）") from e

    outs = []
    mdt = next((p.dtype for p in handle.model.parameters()), None)
    if mdt is None or not torch.is_floating_point(torch.empty(0, dtype=mdt)):
        mdt = torch.float32
    for s in range(0, flat.shape[0], args.chunk):
        sub = flat[s:s + args.chunk]
        adapted, _info = DA.adapt(sub, handle.spec, have_chans)
        xb = torch.from_numpy(np.ascontiguousarray(adapted)).to(
            device=args.device, dtype=mdt)
        cap = {}

        def hook(m, inp, out):
            t = out
            if isinstance(t, (tuple, list)):
                t = t[0] if len(t) else None
            cap["t"] = t.detach() if torch.is_tensor(t) else None

        h = mod.register_forward_hook(hook)
        try:
            with torch.no_grad():
                handle.model(xb)
        finally:
            h.remove()
        t = cap.get("t")
        if t is None:
            raise RuntimeError(f"模块 {module_path} 前向时没有输出张量")
        tk = REG.normalize_token_tensor(t.float())
        f = REG.pool_tokens(tk, args.pool)[:xb.shape[0]]
        outs.append(f.cpu().numpy())
    X = np.concatenate(outs, axis=0)
    if X.shape[0] != n_img * n_rep:
        raise RuntimeError(f"行数对不上：{X.shape[0]} vs {n_img * n_rep}")
    return X.reshape(n_img, n_rep, -1)


def donor_features(spec, handle, eeg, have_chans, args):
    """供体的 (n_img, n_rep, D) 特征。

    kind="final" ⇒ 用模型的**最终输出**（与基线表那一行同口径）；
    kind="layer" ⇒ 抓指定中间模块。
    """
    if spec.kind == "final":
        r = EVAL.model_trial_features(handle, eeg, have_chans, args.batch_size,
                                      args.device, args.pool)
        if not r["ok"]:
            raise RuntimeError(r["info"].get("error") or "最终输出抽取失败")
        return r["features"]
    return extract_module_features(handle, eeg, spec.layer, have_chans, args)


# ---------------------------------------------------------------------------
# 单被试
# ---------------------------------------------------------------------------
def _metrics(Xte, Xrep, IMG_train, IMG_test, n_train_images, args, rng):
    a = EVAL.metric_a_rsa(Xte, IMG_test, args.n_perm, rng)
    mu, sd = EVAL.zscore_fit(Xrep)
    rb = EVAL.metric_b_decode(
        EVAL.zscore_apply(Xrep, mu, sd), EVAL.zscore_apply(Xte, mu, sd),
        IMG_train[:n_train_images] if n_train_images else IMG_train,
        IMG_test, rng, n_null=args.n_null)
    return a, rb


def run_subject(sid, IMG_train, IMG_test, handles, host_name, donors, args, rng):
    t0 = time.time()
    tr_fp = TRAIN_4D / f"sub-{sid:02d}.npz"
    te_fp = TEST_4D / f"sub-{sid:02d}.npz"
    if not tr_fp.exists() or not te_fp.exists():
        return None
    have_chans = BS.THINGS_EEG_CHANS
    n_train_images = args.train_images or None

    eeg_te = np.load(te_fp)["eeg"]
    eeg_tr = np.load(tr_fp)["eeg"]
    if n_train_images:
        # ⚠️ **必须 copy**：切片是**视图**，会一直吊着整个 (16540,4,63,250) 数组
        #    （float32 ≈ 4.2 GB/Job）⇒ 10 个并发就是 42 GB ⇒ 抢内存把 Job 一起拖死
        #    （上一轮基线表就踩过这个坑）。copy 之后只剩 2000 张的 ≈ 500 MB。
        eeg_tr = eeg_tr[:n_train_images].copy()
    log(f"  sub-{sid:02d} test {eeg_te.shape} train {eeg_tr.shape}")

    out = {"subject": f"sub-{sid:02d}", "rows": [], "n_train_images": n_train_images}

    h = handles.get(host_name)
    if h is None or not h.ok:
        out["rows"].append({"name": host_name, "kind": "host", "ok": False,
                             "error": (h.error if h else "宿主未装载")})
        return out

    # --- 宿主特征（只算一次，各变体共用）---
    fte = EVAL.model_trial_features(h, eeg_te, have_chans, args.batch_size,
                                    args.device, args.pool)
    if not fte["ok"]:
        out["rows"].append({"name": host_name, "kind": "host", "ok": False,
                             "error": fte["info"].get("error")})
        return out
    H_te = fte["features"]
    ftr = EVAL.model_trial_features(h, eeg_tr, have_chans, args.batch_size,
                                    args.device, args.pool)
    if not ftr["ok"]:
        out["rows"].append({"name": host_name, "kind": "host", "ok": False,
                             "error": f"train 抽取失败：{ftr['info'].get('error')}"})
        return out
    H_tr = ftr["features"]
    # ⚠️ eeg_tr **不能在这里删**：供体还要用它抽 train 特征
    #    （上一版在这里 del 了，供体行直接 UnboundLocalError ⇒ 10 个 Job 全废）

    row = {"name": host_name, "kind": "host", "ok": True, "n_params": h.n_params,
           "dim": int(H_te.shape[-1])}
    try:
        a, rb = _metrics(H_te, H_tr, IMG_train, IMG_test, n_train_images, args, rng)
        row["metric_a"] = a
        row["metric_b"] = rb
        log(f"    [{host_name}（宿主）] A r={a['r_img']:+.4f} p={a['p_img']:.3f} | "
            f"B acc1={rb['acc1_cos']:.4f} (零中位 {np.median(rb['null_acc1_cos']):.4f})")
    except Exception as e:
        row["ok"] = False
        row["error"] = f"{type(e).__name__}: {e}"[:300]
        log(f"    [{host_name}] ✗ {row['error']}")
    out["rows"].append(row)

    # --- 各供体 ---
    for spec in donors:
        dh = handles.get(spec.donor)
        if dh is None or not dh.ok:
            for nm in (spec.vid, spec.vid + "_shuf"):
                out["rows"].append({"name": nm, "kind": "variant", "ok": False,
                                     "error": (dh.error if dh else "供体未装载"),
                                     "donor": spec.donor})
            continue
        try:
            D_te = donor_features(spec, dh, eeg_te, have_chans, args)
            D_tr = donor_features(spec, dh, eeg_tr, have_chans, args)
        except Exception as e:
            for nm in (spec.vid, spec.vid + "_shuf"):
                out["rows"].append({"name": nm, "kind": "variant", "ok": False,
                                     "error": f"{type(e).__name__}: {e}"[:300],
                                     "donor": spec.donor})
            log(f"    [{spec.vid}] ✗ 供体抽取失败：{type(e).__name__}: {e}"[:200])
            continue

        n_par = h.n_params + dh.n_params
        for suffix, Dte, Dtr in (
                ("", D_te, D_tr),
                ("_shuf", V.shuffle_images(D_te, rng),
                 V.shuffle_images(D_tr, rng))):
            nm = spec.vid + suffix
            r = {"name": nm, "kind": "control_shuffle" if suffix else "variant",
                 "ok": True, "n_params": n_par, "donor": spec.donor,
                 "donor_layer": spec.layer,
                 "dim": int(H_te.shape[-1] + Dte.shape[-1])}
            try:
                Xte = V.concat_feats(H_te, Dte)
                Xtr = V.concat_feats(H_tr, Dtr)
                a, rb = _metrics(Xte, Xtr, IMG_train, IMG_test, n_train_images,
                                 args, rng)
                r["metric_a"] = a
                r["metric_b"] = rb
                log(f"    [{nm}] A r={a['r_img']:+.4f} p={a['p_img']:.3f} | "
                    f"B acc1={rb['acc1_cos']:.4f} "
                    f"(零中位 {np.median(rb['null_acc1_cos']):.4f}) d={r['dim']}")
                del Xte, Xtr
            except Exception as e:
                r["ok"] = False
                r["error"] = f"{type(e).__name__}: {e}"[:300]
                log(f"    [{nm}] ✗ {r['error']}")
            out["rows"].append(r)
        del D_te, D_tr

    del eeg_te, eeg_tr, H_te, H_tr
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


# ---------------------------------------------------------------------------
# 自测（合成数据，走真实代码路径）
# ---------------------------------------------------------------------------
def self_test():
    import torch
    from torch import nn
    import interfaces as IF
    import registry as REG

    log("=== 自测：变体链路（合成数据）===")
    rng = np.random.default_rng(3)

    class Patch(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.proj = nn.Linear(9 * 25, d)

        def forward(self, x):
            B, C, T = x.shape
            p = x.unfold(-1, 25, 25).unfold(1, 9, 9)
            p = p.permute(0, 2, 1, 3, 4).reshape(B, -1, 225)
            return self.proj(p)

    class HostModel(nn.Module):
        def __init__(self, d=32):
            super().__init__()
            self.patch = Patch(d)
            self.blk = IF.Block(IF.SelfAttention1D(d, 4), IF.FFN(d, 2 * d), d)

        @classmethod
        def from_pretrained(cls, path, n_outputs=None):
            return cls()

        def forward(self, x):
            tb = IF.TokenBatch(self.patch(x))
            tb = self.blk(tb)
            return {"embedding": tb.tokens}

    class DonorModel(nn.Module):
        """有个叫 transformer.layers.0.fn 的模块（命中 exp75 的 BLOCKISH 规则）。"""

        def __init__(self, d=16):
            super().__init__()
            self.patch = Patch(d)
            self.transformer = nn.Module()
            self.transformer.layers = nn.ModuleList([nn.Module()])
            self.transformer.layers[0].fn = nn.Linear(d, d)

        @classmethod
        def from_pretrained(cls, path, n_outputs=None):
            return cls()

        def forward(self, x):
            t = self.patch(x)
            t = self.transformer.layers[0].fn(t)
            return {"embedding": t}

    class NS:
        pass
    ns = NS()
    ns.CBraMod = HostModel
    ns.SignalJEPA = DonorModel

    # --- 合成数据：IMG → EEG → 两种表征 ---
    n_img_tr, n_img_te, n_rep_tr, n_rep_te, d_img = 180, 60, 4, 8, 32
    IMG_tr = rng.normal(size=(n_img_tr, d_img))
    IMG_te = rng.normal(size=(n_img_te, d_img))
    W = rng.normal(size=(d_img, 63 * 250)) * 0.05

    def make(IMG, n_rep):
        e = (IMG @ W + 0.25 * rng.normal(size=(IMG.shape[0], 63 * 250)))
        e = e.reshape(IMG.shape[0], 63, 250)
        return np.repeat(e[:, None], n_rep, axis=1).astype(np.float32)

    eeg_tr, eeg_te = make(IMG_tr, n_rep_tr), make(IMG_te, n_rep_te)

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        for nm, cls in (("cbramod", HostModel), ("signal-jepa", DonorModel)):
            d = Path(td) / nm
            d.mkdir()
            (d / "config.json").write_text(json.dumps(
                {"n_chans": 63, "n_times": 250, "patch_size": [9, 25]}),
                encoding="utf-8")
            (d / "ATTRIBUTION.txt").write_text("Source: synthetic\nLicense: MIT\n",
                                               encoding="utf-8")
        handles = {}
        # ⚠️ 本地自测**不能**用 base_specs 的真规格：signal-jepa 的
        #    interpolate=True 会走 `things_chs_info` ⇒ 需要 mne（本地没装）。
        #    假模型不需要通道插值，用一份自造的最小规格即可。
        fake_specs = {
            "cbramod": {"n_chans": 63, "n_times": 250, "sfreq": 200.0,
                        "chans": None, "window_mode": "pad", "interpolate": False,
                        "cls": "CBraMod"},
            "signal-jepa": {"n_chans": 63, "n_times": 250, "sfreq": 200.0,
                            "chans": None, "window_mode": "pad",
                            "interpolate": False, "cls": "SignalJEPA"},
        }
        for nm in ("cbramod", "signal-jepa"):
            handles[nm] = REG.load_base_model(
                nm, models_dir=Path(td), braindecode_models=ns,
                spec=fake_specs[nm])
            assert handles[nm].ok, handles[nm].error

        spec = V.DonorSpec(vid="v1_test", donor="signal-jepa",
                            layer="transformer.layers.0.fn")
        args = argparse.Namespace(train_images=n_img_tr, n_perm=50, n_null=4,
                                  batch_size=16, chunk=2000, device="cpu",
                                  pool="mean_all", out_dir=".")
        D_te = donor_features(spec, handles["signal-jepa"], eeg_te,
                              BS.THINGS_EEG_CHANS, args)
        D_tr = donor_features(spec, handles["signal-jepa"], eeg_tr,
                              BS.THINGS_EEG_CHANS, args)
        log(f"  ✓ 供体层特征：{D_te.shape} / {D_tr.shape}")
        assert D_te.shape[:2] == (n_img_te, n_rep_te), D_te.shape

        H_te = EVAL.model_trial_features(handles["cbramod"], eeg_te,
                                          BS.THINGS_EEG_CHANS, 16, "cpu",
                                          "mean_all")["features"]
        H_tr = EVAL.model_trial_features(handles["cbramod"], eeg_tr,
                                          BS.THINGS_EEG_CHANS, 16, "cpu",
                                          "mean_all")["features"]
        X = V.concat_feats(H_te, D_te)
        assert X.shape[-1] == H_te.shape[-1] + D_te.shape[-1]
        log(f"  ✓ 拼接：宿主 {H_te.shape[-1]}d + 供体 {D_te.shape[-1]}d "
            f"⇒ {X.shape[-1]}d")

        # --- shuffle 对照必须**真的**打乱图像对应关系 ---
        S = V.shuffle_images(D_te, np.random.default_rng(0))
        assert S.shape == D_te.shape
        # 逐图看：打乱后同一位置的供体特征不应等于原特征（除自环）
        same = np.allclose(S[0], D_te[0])
        assert not same, "shuffle 没有生效"
        # 供体的图像信息确实被摧毁：打乱后与图像特征的 RSA 应当掉到随机附近
        a_real = EVAL.metric_a_rsa(D_te, IMG_te, 50, np.random.default_rng(1))
        a_shuf = EVAL.metric_a_rsa(S, IMG_te, 50, np.random.default_rng(1))
        log(f"  ✓ 对照有效性：供体自身 A r={a_real['r_img']:+.4f} → "
            f"打乱后 r={a_shuf['r_img']:+.4f}")
        assert a_shuf["r_img"] < a_real["r_img"], "打乱后 RSA 应当下降"

        # --- 端到端：宿主行 + 变体行 + 对照行 ---
        a_h = EVAL.metric_a_rsa(H_te, IMG_te, 50, rng)
        a_v = EVAL.metric_a_rsa(V.concat_feats(H_te, D_te), IMG_te, 50, rng)
        a_s = EVAL.metric_a_rsa(V.concat_feats(H_te, S), IMG_te, 50, rng)
        log(f"  ✓ 三行：宿主 A r={a_h['r_img']:+.4f} | 变体 A r={a_v['r_img']:+.4f}"
            f" | 对照 A r={a_s['r_img']:+.4f}")
        assert all(x["r_img"] == x["r_img"] for x in (a_h, a_v, a_s))   # 非 NaN

    log("=== 自测通过 ===")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="",
                    help="逗号分隔的模型（默认：宿主 + 该版本能装的供体）")
    ap.add_argument("--host", default=V.HOST)
    ap.add_argument("--donors", default="",
                    help="逗号分隔的供体模型名（默认全部）")
    ap.add_argument("--only-subjects", default="")
    ap.add_argument("--train-images", type=int, default=2000)
    ap.add_argument("--pool", default="mean_all")
    ap.add_argument("--n-perm", type=int, default=EVAL.N_PERM_RSA)
    ap.add_argument("--n-null", type=int, default=EVAL.N_NULL)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--bd-version", default="stable")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    args.train_images = args.train_images or None

    import torch
    import registry as REG

    REG.ensure_braindecode_version(_bd_install_target(args.bd_version))
    try:
        import braindecode.models as bdmodels
    except Exception as e:
        log(f"✗ braindecode 不可用：{e!r}")
        raise SystemExit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ftr = torch.load(FEAT_TRAIN, map_location="cpu", weights_only=False)
    IMG_train = ftr["img_features"].numpy().astype(np.float64)
    fte = torch.load(FEAT_TEST, map_location="cpu", weights_only=False)
    IMG_test = fte["img_features"].numpy().astype(np.float64)
    log(f"ViT-H-14 特征：train {IMG_train.shape} / test {IMG_test.shape}")

    # 供体清单：本趟能装哪些供体
    all_donor_models = ([d.strip() for d in args.donors.split(",") if d.strip()]
                        if args.donors else
                        sorted({d.donor for d in V.DONORS}))
    donors = V.donors_for_trip(all_donor_models)

    if args.models:
        names = [s.strip() for s in args.models.split(",") if s.strip()]
    else:
        names = [args.host] + [d.donor for d in donors]
    names = list(dict.fromkeys(names))
    log(f"本趟（braindecode {args.bd_version}）模型：{names}")
    log(f"供体：{[d.vid + '←' + d.donor + '/' + d.layer for d in donors]}")

    handles = {}
    for nm in names:
        handles[nm] = REG.load_base_model(nm, models_dir=MODELS_DIR,
                                          braindecode_models=bdmodels,
                                          spec=BS.BASE_SPECS.get(nm))
        if not handles[nm].ok:
            log(f"  ✗ {nm} 装载失败：{handles[nm].error[:200]}")
    log(f"装载成功：{[n for n, h in handles.items() if h.ok]}")

    sids = ([int(s) for s in args.only_subjects.split(",") if s.strip()]
            if args.only_subjects else list(range(1, 11)))
    rng = np.random.default_rng(SEED)

    per_subject = []
    for sid in sids:
        log(f"\n===== sub-{sid:02d} =====")
        rec = run_subject(sid, IMG_train, IMG_test, handles, args.host, donors,
                          args, rng)
        if rec is None:
            log(f"  sub-{sid:02d}: 数据缺失，跳过")
            continue
        per_subject.append(rec)
        (out_dir / f"variant_sub-{sid:02d}.json").write_text(
            json.dumps({"per_subject": [rec]}, indent=2, ensure_ascii=False,
                       default=str), encoding="utf-8")

    agg = EVAL.summarize(per_subject)
    order = [args.host] + [d.vid for d in donors] + [d.vid + "_shuf" for d in donors]
    order = [n for n in order if n in agg]
    md = EVAL.render_table(agg, order=order)
    log("\n" + "=" * 100)
    log("阶段 D 变体表（宿主 cbramod + 供体层拼接；_shuf = 维度膨胀对照）")
    log("=" * 100)
    log(md)

    (out_dir / f"variant_table_{_ver_tag(args.bd_version)}.md").write_text(
        md, encoding="utf-8")
    (out_dir / f"variant_full_{_ver_tag(args.bd_version)}.json").write_text(
        json.dumps({"schema_version": "1.0", "script": "exp77_variant_eval",
                    "config": vars(args), "summary": agg,
                    "per_subject": per_subject}, indent=2, ensure_ascii=False,
                   default=str), encoding="utf-8")
    log(f"结果写入 {out_dir}")


if __name__ == "__main__":
    main()
