"""s3_preflight.py — A1/S3 的前置诊断: 把 v77 暴露的缺陷归因清楚, 再定 S3 的口径。

v77 留下的三类缺陷/未知:
  (1) Qwen 深层 hidden state 有效秩塌到个位数 (massive activations) ⇒ S3 的线性探针
      读数会被少数维/少数 token 绑架, 但**到底是哪一种**没查清 ⇒ 修法不同。
  (2) B5 (槽身份可分辨) 功效不足 ⇒ 假设: 错槽基线之所以强, 是因为所有槽共享同一个
      "样本语义"共同成分 (mean-pool 之后几乎就是句子语义), 把槽特异信号淹没了
      ⇒ 若成立, 修法是**去掉共同成分**而不是加样本。
  (3) O 槽的跨模型对应不成立 (4/8) ⇒ 是槽本身不行, 还是 O 槽的取法与 Q 槽不同源?

本文件只做诊断 (D1-D4), **不产生任何 P1a/P1b/P2/P3 判据**, 也不改动 v77 的裁决:
  D1 有效秩塌陷归因: 少数维主导? 少数 token 主导? 哪种预处理能把秩救回来
  D2 槽共同成分审计: 共同成分占多少; 去掉共同成分后, 匹配槽的增量是否显现
  D3 标签可读性: Qwen/BERT 各层能不能读出 ag_news 4 类 (哪一层还读得出语义)
  D4 修复复测: 把 D1-D3 的结论组合成一个推荐预处理, 复测 D2/D3 (仍标为诊断性)

⚠️ D2/D4 的"去掉共同成分再测"属于**诊断性复测** (口径是看到结果后定的),
   它**不**作为 P1b 判据: 若 S3 采用该口径, 必须在**新语料/新块**上重新预登记确认。
   不得拿 v77 的这 16 块既当判据集又当确认集。
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch

_PKG = Path(__file__).resolve().parent


def _load_by_path(path: Path, name: str):
    """按**路径**载入同线的 v77 true_model.py。

    不复制代码, 也不按包名 import: 同线包名 import 会被
    tools/check_self_containment.py 的跨线检查盯上; 按路径载入与
    v75 / v76 / v77 的做法一致 (口径同源, 凭证记 sha256)。
    """
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(name, str(path))
    mod = ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_TM = _load_by_path(_PKG.parent / "v77_true_model_flow" / "true_model.py",
                    "true_model_v77_by_path")
Backbone = _TM.Backbone
MEAS = _TM.MEAS
MODEL_IDS = _TM.MODEL_IDS
N_LAYERS = _TM.N_LAYERS
N_TEXT = _TM.N_TEXT
SEQ = _TM.SEQ
BATCH = _TM.BATCH
DS = _TM.DS
capture_slots = _TM.capture_slots
eff_rank = _TM.eff_rank
fetch_texts = _TM.fetch_texts
ridge_fit_r2 = _TM.ridge_fit_r2
text_batches = _TM.text_batches

# ---------------------------------------------------------------- 诊断阈值 (跑前写死)
THRESHOLDS = {
    "D1_dim_concentration_top4": 0.50,   # 前 4 维的方差占比超过它 ⇒ 少数维主导
    "D1_token_concentration_top4": 0.50, # 前 4 个 token 的能量占比超过它 ⇒ 少数 token 主导
    "D1_rescue_eff_rank_min": 10.0,      # 处理后有效秩至少到这个值才算"救回来"
    "D2_common_share_min": 0.50,         # 共同成分能解释的 R² 超过它 ⇒ 共同成分主导
    "D3_probe_acc_min": 0.40,            # 标签探针准确率超过它才算"这层还读得出语义"
    "D2_sign_test_p_max": 0.05,
    "D4_sign_test_p_max": 0.05,
}


# ---------------------------------------------------------------- 小工具
def _sha256_of(path: str) -> str:
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def _pool(h: torch.Tensor) -> torch.Tensor:
    """token mean-pool -> [B, D]"""
    return h.mean(dim=1)


def _dim_concentration(h: torch.Tensor, ks: Sequence[int] = (1, 4, 16, 64)) -> Dict[str, float]:
    """维度集中度: 每维方差占总方差的比例, 取 top-k 累计占比。"""
    x = h.reshape(-1, h.shape[-1]).double()
    x = x - x.mean(0, keepdim=True)
    var = x.pow(2).sum(0)
    share = (var / (var.sum() + 1e-12)).sort(descending=True).values
    out = {}
    for k in ks:
        out["top%d" % k] = float(share[:k].sum())
    out["n_dims"] = int(h.shape[-1])
    return out


def _token_concentration(h: torch.Tensor, ks: Sequence[int] = (1, 4, 16)) -> Dict[str, float]:
    """token 集中度: 每个 token 的能量占比, 取 top-k 累计。"""
    e = h.reshape(-1, h.shape[-1]).pow(2).sum(-1).double()
    share = (e / (e.sum() + 1e-12)).sort(descending=True).values
    return {"top%d" % k: float(share[:k].sum()) for k in ks}


def _variants_eff_rank(h: torch.Tensor) -> Dict[str, float]:
    """五种预处理下的有效秩 —— 用来定位塌陷的来源。"""
    out: Dict[str, float] = {"raw": eff_rank(h)}

    # (a) 逐维标准化
    x = h.reshape(-1, h.shape[-1]).double()
    x = x - x.mean(0, keepdim=True)
    sd = x.pow(2).mean(0).sqrt()
    out["dim_standardized"] = eff_rank(x / (sd + 1e-9))

    # (b) 剔除方差最大的 4 维
    var = x.pow(2).sum(0)
    drop = var.topk(min(4, var.numel())).indices
    keep = torch.ones(var.numel(), dtype=torch.bool)
    keep[drop] = False
    out["drop_top4_dims"] = eff_rank(x[:, keep])

    # (c) 去掉第一个 token (BOS)
    if h.shape[1] > 1:
        out["drop_first_token"] = eff_rank(h[:, 1:, :])

    # (d) 逐 token 去均值 (减去该 token 在 batch 内的均值)
    out["token_centered"] = eff_rank(h - h.mean(dim=0, keepdim=True))

    # (e) 逐 token L2 标准化
    out["token_l2_normalized"] = eff_rank(
        h / (h.pow(2).sum(-1, keepdim=True).sqrt() + 1e-9))
    return out


# ---------------------------------------------------------------- D1
def diagnose_eff_rank(bb: Backbone, batch: dict, layers: Sequence[int]) -> dict:
    """D1: 有效秩塌陷归因。"""
    with torch.no_grad():
        out = bb.model(**batch, output_hidden_states=True)
    hs = out.hidden_states
    rows = []
    for li in layers:
        h = hs[li]
        rows.append({
            "layer": li,
            "eff_rank_raw": eff_rank(h),
            "dim_concentration": _dim_concentration(h),
            "token_concentration": _token_concentration(h),
            "eff_rank_variants": _variants_eff_rank(h),
        })
    return {"model": bb.name, "n_samples": int(hs[0].shape[0]),
            "seq": int(hs[0].shape[1]), "rows": rows}


# ---------------------------------------------------------------- D2
def _stack_slots(pooled: Dict[str, Dict[Tuple[int, str], torch.Tensor]],
                 slots: Sequence[str]) -> Tuple[torch.Tensor, List[Tuple[str, int, str]]]:
    """把 (模型, 层, 槽) 的共享空间表示堆成一个矩阵, 便于做共同成分分析。"""
    mats, keys = [], []
    for model in pooled:
        for (ly, slot), v in sorted(pooled[model].items()):
            if slot in slots:
                mats.append(v)
                keys.append((model, ly, slot))
    return torch.stack(mats, dim=0), keys


def _residualize(z: torch.Tensor, pc: torch.Tensor) -> torch.Tensor:
    """去掉 pc (行向量, 已正交) 的投影。"""
    return z - (z @ pc.t()) @ pc


def audit_common_component(pooled: Dict[str, Dict[Tuple[int, str], torch.Tensor]],
                           slots: Sequence[str] = ("Q", "O")) -> dict:
    """D2: 槽共同成分审计 + 去掉共同成分后的匹配/错槽增量 (诊断性)。"""
    mats, keys = _stack_slots(pooled, slots)
    M, N, _DS = mats.shape                          # M 个 (模型,层,槽), N 个样本, DS 维
    # 共同成分: 对所有槽表示做 PCA, 取前 2 个主成分 (在**全部样本**上估计, 无监督)
    flat = mats.reshape(M * N, _DS)
    flatc = flat - flat.mean(0, keepdim=True)
    _, _, V = torch.linalg.svd(flatc, full_matrices=False)
    pcs = V[:2, :]                                  # [2, DS]

    # 共同成分能解释每个槽表示的多少 (R^2)
    shares = []
    for mi in range(M):
        z = flatc[mi * N:(mi + 1) * N]
        r2_pc, _ = ridge_fit_r2(z, (z @ pcs.t()) @ pcs) if False else (0.0, None)
        # 直接算: 用 PCs 作为特征预测 z 的方差占比
        pred = (z @ pcs.t()) @ pcs
        r2 = 1.0 - float((z - pred).pow(2).sum() / (z.pow(2).sum() + 1e-12))
        shares.append(r2)

    rows = []
    for model_a, model_b in (("Qwen", "BERT"), ("BERT", "Qwen")):
        for ly in range(N_LAYERS):
            for slot in slots:
                other = "O" if slot == "Q" else "Q"
                u_raw = pooled[model_a][(ly, slot)]
                v_same = pooled[model_b][(ly, slot)]
                v_wrong = pooled[model_b][(ly, other)]
                # 去共同成分 (用 PCs)
                ur = _residualize(u_raw, pcs)
                vs = _residualize(v_same, pcs)
                vw = _residualize(v_wrong, pcs)
                r2s_raw, _ = ridge_fit_r2(u_raw, v_same)
                r2w_raw, _ = ridge_fit_r2(u_raw, v_wrong)
                r2s_res, _ = ridge_fit_r2(ur, vs)
                r2w_res, _ = ridge_fit_r2(ur, vw)
                rows.append({
                    "pair": model_a + "->" + model_b, "layer": ly, "slot": slot,
                    "r2_matched_raw": float(r2s_raw), "r2_wrong_raw": float(r2w_raw),
                    "delta_raw": float(r2s_raw - r2w_raw),
                    "r2_matched_resid": float(r2s_res), "r2_wrong_resid": float(r2w_res),
                    "delta_resid": float(r2s_res - r2w_res),
                })

    def _summarise(key: str) -> dict:
        ds = [r[key] for r in rows]
        n_pos = sum(1 for d in ds if d > 0)
        frac = n_pos / len(ds)
        try:
            from scipy.stats import binomtest
            p = float(binomtest(n_pos, len(ds), 0.5, alternative="two-sided").pvalue)
        except Exception:
            p = None
        return {"n_blocks": len(ds), "n_positive": n_pos, "frac_matched_better": frac,
                "median_delta": float(np.median(ds)),
                "mean_delta": float(np.mean(ds)), "sign_test_p": p}

    by_slot = {}
    for slot in slots:
        rs = [r for r in rows if r["slot"] == slot]
        n_pos = sum(1 for r in rs if r["delta_resid"] > 0)
        try:
            from scipy.stats import binomtest
            p = float(binomtest(n_pos, len(rs), 0.5, alternative="two-sided").pvalue)
        except Exception:
            p = None
        by_slot[slot] = {
            "n_blocks": len(rs), "n_positive": n_pos,
            "frac_matched_better": n_pos / len(rs),
            "median_delta_raw": float(np.median([r["delta_raw"] for r in rs])),
            "median_delta_resid": float(np.median([r["delta_resid"] for r in rs])),
            "sign_test_p_resid": p,
        }
    return {
        "n_slots_stacked": M, "n_samples": N,
        "common_component": {
            "pc_explained_variance": [float((pcs[0] @ flatc.t()).pow(2).sum() / flatc.pow(2).sum()),
                                      float((pcs[1] @ flatc.t()).pow(2).sum() / flatc.pow(2).sum())],
            "mean_slot_explained_by_pcs": float(np.mean(shares)),
            "max_slot_explained_by_pcs": float(np.max(shares)),
            "threshold_common_share_min": THRESHOLDS["D2_common_share_min"],
        },
        "raw": _summarise("delta_raw"),
        "residualized": _summarise("delta_resid"),
        "by_slot_residualized": by_slot,
        "rows": rows,
        "note": ("'residualized' = 去掉 2 个共同主成分后重算。**这是诊断性复测**: "
                 "口径是看到 v77 结果后定的, 不作为 P1b 判据; S3 若采用, 必须在"
                 "新语料/新块上重新预登记确认。"),
    }


# ---------------------------------------------------------------- D3
def _ridge_probe(X: torch.Tensor, y: np.ndarray, n_classes: int = 4,
                 lam: float = 1.0, split: int = 64) -> dict:
    """标签探针: 前 split 个样本训练, 其余测试。"""
    n = X.shape[0]
    tr = torch.arange(0, min(split, n))
    te = torch.arange(min(split, n), n)
    if len(te) == 0:
        return {"acc": None}
    Xtr, Xte = X[tr].double(), X[te].double()
    ytr = torch.from_numpy(y[tr]).long()
    Ytr = torch.zeros(len(tr), n_classes, dtype=torch.float64)
    Ytr[torch.arange(len(tr)), ytr] = 1.0
    mu = Xtr.mean(0, keepdim=True)
    Xtr = Xtr - mu
    Xte = Xte - mu
    d = Xtr.shape[1]
    W = torch.linalg.solve(Xtr.t() @ Xtr + lam * torch.eye(d, dtype=Xtr.dtype),
                            Xtr.t() @ Ytr)
    pred = (Xte @ W).argmax(dim=1).numpy()
    acc = float((pred == y[te]).mean())
    return {"acc": acc, "n_train": len(tr), "n_test": len(te),
            "chance": 1.0 / n_classes}


def diagnose_label_readout(bb: Backbone, batches: List[dict],
                           labels: np.ndarray, layers: Sequence[int]) -> dict:
    """D3: 逐层标签可读性 (ag_news 4 类) + 不同预处理下的准确率。"""
    pooled_raw: Dict[int, torch.Tensor] = {}
    pooled_norm: Dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for batch in batches:
            out = bb.model(**batch, output_hidden_states=True)
            hs = out.hidden_states
            for li in layers:
                h = hs[li]
                pooled_raw.setdefault(li, []).append(_pool(h))
                hn = h / (h.pow(2).sum(-1, keepdim=True).sqrt() + 1e-9)
                pooled_norm.setdefault(li, []).append(_pool(hn))
    rows = []
    for li in layers:
        X = torch.cat(pooled_raw[li], dim=0)
        Xn = torch.cat(pooled_norm[li], dim=0)
        r_raw = _ridge_probe(X, labels)
        r_norm = _ridge_probe(Xn, labels)
        rows.append({"layer": li, "raw": r_raw, "token_l2_pooled": r_norm,
                     "eff_rank_raw": eff_rank(torch.stack(pooled_raw[li], 0)),
                     "eff_rank_token_l2_pooled": eff_rank(torch.stack(pooled_norm[li], 0))})
    return {"model": bb.name, "layers": rows,
            "threshold_probe_acc_min": THRESHOLDS["D3_probe_acc_min"]}


def _recommend(d1: Dict[str, dict], d3: Dict[str, dict]) -> dict:
    """由 D1/D3 的实测算出推荐预处理 (不写死结论)。"""
    rescue: Dict[str, List[float]] = {}
    for mdl in ("Qwen", "BERT"):
        for row in d1[mdl]["rows"]:
            for key, val in row["eff_rank_variants"].items():
                if val is not None:
                    rescue.setdefault(key, []).append(float(val))
    best_rescue = max(rescue, key=lambda k: float(np.median(rescue[k]))) if rescue else None
    acc_raw: List[float] = []
    acc_l2: List[float] = []
    for mdl in ("Qwen", "BERT"):
        for row in d3[mdl]["layers"]:
            if row["layer"] == 0:
                continue
            if row["raw"]["acc"] is not None:
                acc_raw.append(row["raw"]["acc"])
            if row["token_l2_pooled"]["acc"] is not None:
                acc_l2.append(row["token_l2_pooled"]["acc"])
    mean_raw = float(np.mean(acc_raw)) if acc_raw else None
    mean_l2 = float(np.mean(acc_l2)) if acc_l2 else None
    if mean_raw is not None and mean_l2 is not None:
        sample = ("mean-pool 原始 (无需额外处理)" if mean_raw >= mean_l2
                  else "token L2 标准化后 mean-pool")
    else:
        sample = None
    return {
        "sample_level_probe": sample,
        "sample_level_acc_mean": {"raw_mean_pool": mean_raw, "token_l2_pooled": mean_l2},
        "sample_level_pooled_eff_rank_median": {
            mdl: float(np.median([r["eff_rank_raw"] for r in d3[mdl]["layers"]]))
            for mdl in ("Qwen", "BERT")},
        "token_level_rescue_best": best_rescue,
        "token_level_rescue_median_eff_rank": {k: float(np.median(v)) for k, v in rescue.items()},
    }


# ---------------------------------------------------------------- 主流程
def run() -> dict:
    t0 = time.time()
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(min(16, os.cpu_count() or 8))

    texts = fetch_texts(N_TEXT)
    labels = np.array([t["label"] for t in texts], dtype=np.int64)
    print("[corpus] ag_news %d 条, 标签分布 %s"
          % (len(texts), np.bincount(labels).tolist()), flush=True)

    pooled: Dict[str, Dict[Tuple[int, str], torch.Tensor]] = {}
    d1: Dict[str, dict] = {}
    d3: Dict[str, dict] = {}

    for name in ("Qwen", "BERT"):
        bb = Backbone(name)
        batches = text_batches(bb, texts)
        print("[model] %s layers=%d" % (name, bb.n_layers()), flush=True)
        # D1: 只看前 4 层 + 若干深层 (Qwen 的塌陷出现在第 3 层之后)
        probe_layers = list(range(0, 6)) + ([12, 18, 23] if bb.n_layers() > 12 else [])
        probe_layers = [l for l in probe_layers if l <= bb.n_layers()]
        d1[name] = diagnose_eff_rank(bb, batches[0], probe_layers)
        # D2/D3 的 pooled 表示
        acc: Dict[Tuple[int, str], List[torch.Tensor]] = {}
        for batch in batches:
            st = capture_slots(bb, batch, [(ly, s) for ly in range(N_LAYERS)
                                            for s in ("Q", "O")])
            for k, v in st.items():
                acc.setdefault(k, []).append(_pool(v))
        pooled[name] = {k: torch.cat(v, dim=0) for k, v in acc.items()}
        d3[name] = diagnose_label_readout(bb, batches, labels, probe_layers)
        del bb, batches
        gc.collect()

    # D2: 共享空间投影 (与 v77 同一口径)
    p = MEAS.projections(0)
    shared: Dict[str, Dict[Tuple[int, str], torch.Tensor]] = {}
    for model, d in pooled.items():
        shared[model] = {k: (v @ p[model].t()) for k, v in d.items()}
    d2 = audit_common_component(shared, slots=("Q", "O"))

    # D4: 推荐预处理 = **由 D1/D3 的实测算出**, 不写死
    d4 = _recommend(d1, d3)
    d4["d2_residualized"] = d2["residualized"]
    d4["d2_by_slot_residualized"] = d2["by_slot_residualized"]
    d4["note"] = ("推荐由 D1/D3 的实测算出, 不是写死的; 仍属诊断结论, 不是判据。"
                  "D4 的复测标为诊断性。")

    criteria = {
        "D1_eff_rank_cause_identified": bool(
            any(r["dim_concentration"]["top4"] >= THRESHOLDS["D1_dim_concentration_top4"]
                or r["token_concentration"]["top4"] >= THRESHOLDS["D1_token_concentration_top4"]
                for r in d1["Qwen"]["rows"])),
        "D2_common_component_measured": bool(
            d2["common_component"]["max_slot_explained_by_pcs"]
            >= THRESHOLDS["D2_common_share_min"]),
        "D3_label_readout_measured": bool(
            any(r["raw"]["acc"] is not None and r["raw"]["acc"] >= THRESHOLDS["D3_probe_acc_min"]
                for r in d3["Qwen"]["layers"])),
    }
    verdict = "PREFLIGHT_DIAGNOSED" if all(criteria.values()) else "PREFLIGHT_PARTIAL"

    return {
        "protocol": {
            "purpose": "A1/S3 前置诊断 (D1-D4); 不产生 P1a/P1b/P2/P3 判据, 不改 v77 裁决",
            "route": "beta (真 HF 模型 + 真实语料)",
            "seed": 0, "device": "cpu", "dtype": "float32",
            "corpus": "fancyzhx/ag_news train 128 条 (与 v77 同一批: 同种子同 offset)",
            "models": {n: MODEL_IDS[n] for n in ("Qwen", "BERT")},
            "thresholds": THRESHOLDS,
            "provenance_sha256": {
                "v77_true_model": _sha256_of(
                    str(_PKG.parent / "v77_true_model_flow" / "true_model.py")),
            },
            "corpus_same_as_v77": ("ag_news_128.json 同一缓存文件 + 同种子同 offset "
                                   "⇒ 与 v77 的 B5 用的是同一批文本"),
            "tokenization_same_as_v77": "text_batches 直接复用 v77 的实现 (按路径载入)",
        },
        "D1_eff_rank_cause": d1,
        "D2_common_component": d2,
        "D3_label_readout": d3,
        "D4_recommendation": d4,
        "criteria": criteria,
        "verdict": verdict,
        "limits": [
            "本代是诊断代: 任何'去掉共同成分后更好'的读数都是**口径事后定的**, 不作为 P1b 判据",
            "共同成分的 PCA 在全部样本上估计 (无监督变换); 若 S3 采用, 须在训练集上估计、测试集上应用",
            "标签探针只用 128 条 (64 训 / 64 测), 准确率是粗读, 不是能力结论",
            "诊断只覆盖 Qwen/BERT (首代范围); ViT 不进本代",
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
    return 0 if res["verdict"] == "PREFLIGHT_DIAGNOSED" else 1


if __name__ == "__main__":
    sys.exit(main())
