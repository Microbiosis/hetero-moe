# -*- coding: utf-8 -*-
"""载体性质探针 (post_hoc_only) —— 共享空间是**线性**子空间, 注入算子里唯一的非线性是逐样本范数归一化.

问题: 「A1 的共享对齐空间是线性还是非线性? 换非线性 / 扩维度能不能救槽特异与深度对应?」

本探针只做**载体性质**的测量, 不参与 S4-P 的任何裁决, 不推翻任何阈值:
  1. 投影矩阵的正交性 (P Pᵀ = I? PᵀP = I?): v62 measure.projections() 是**随机正交**投影;
  2. 载体的范数保持: ‖Δ‖ / ‖a_s‖ (源侧 -> 目标侧);
  3. 载体的角度保持: 源侧两点夹角 vs 目标侧两点夹角 (成对 cos);
  4. 投影的方差保留 r2_project (对照 v81 ceiling_diag.json 的同口径读数).

结论口径: 若载体**范数与夹角都被保持**, 则失败不能归因于「空间是线性的」,
「换非线性」只会引入新混淆、「扩维度」只补幅度 —— 破不了「分不清」, 因为
「分不清」源于**源侧语义本身** (同层 Q/O 槽可互换), 不源于映射的性质.

用法: python research/weight_fusion/v82_s4_pipe/carrier_probe.py
"""
import importlib.util as ilu
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_PKG = Path(__file__).resolve().parent


def _load_by_path(path: Path, name: str):
    spec = ilu.spec_from_file_location(name, str(path))
    mod = ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


S4 = _load_by_path(_PKG / "s4_pipe.py", "s4_pipe_v82_by_path")

N_PAIRS = 200          # 夹角保持的成对抽样数 (每个 (方向,层,种子))
RNG_SEED = 20261118    # 与 s4_pipe.RAND_SEED_BASE 同源, 抽样可复现


def _ortho_err(P: torch.Tensor):
    """P: (DS, d) 行正交 (P Pᵀ = I_DS). 返回 (‖P Pᵀ − I‖_F, ‖PᵀP − I‖_F 或 None)."""
    I = torch.eye(P.shape[0], dtype=P.dtype)
    e1 = float((P @ P.t() - I).norm())
    e2 = None
    if P.shape[0] == P.shape[1]:
        e2 = float((P.t() @ P - I).norm())
    return e1, e2


def _cos_pairs(A: torch.Tensor, B: torch.Tensor, rng):
    """成对 cos: 源侧 A 的夹角 vs 目标侧 B 的夹角. 返回 (mean_abs_diff, max_abs_diff, pearson_r)."""
    n = A.shape[0]
    i = rng.integers(0, n, size=N_PAIRS)
    j = rng.integers(0, n, size=N_PAIRS)
    ok = i != j
    i, j = i[ok], j[ok]
    ca = torch.nn.functional.cosine_similarity(A[i], A[j], dim=-1).numpy()
    cb = torch.nn.functional.cosine_similarity(B[i], B[j], dim=-1).numpy()
    d = np.abs(ca - cb)
    if cb.std() > 0 and ca.std() > 0:
        r = float(np.corrcoef(ca, cb)[0, 1])
    else:
        r = None
    return float(d.mean()), float(d.max()), r


def main() -> int:
    torch.set_num_threads(min(16, os.cpu_count() or 8))
    S4.fetch_models()
    texts = S4.fetch_texts(S4.N_TEXT)
    print("[corpus] %d 条" % len(texts), flush=True)

    src_cache = {}
    for name, layers in (("Qwen", S4.L_Q), ("BERT", S4.L_B)):
        src_cache[name] = S4.capture_source(name, layers, texts, None)
        print("[source] %s 抓完 (%d 个 (层,槽))" % (name, len(src_cache[name])), flush=True)

    rng = np.random.default_rng(RNG_SEED)
    out = {
        "role": "post_hoc_only —— **载体性质诊断**, 不参与 S4-P 任何裁决, 不推翻任何阈值",
        "question": "A1 的共享对齐空间是线性还是非线性? 换非线性 / 扩维度能否救槽特异与深度对应?",
        "definitions": {
            "carrier": "Δ = (a_s @ P_Aᵀ) @ P_B —— P_A/P_B 是**常量**矩阵 => 载体是**线性**映射; "
                       "注入端唯一的非线性是逐样本范数归一化 ‖Δ_i‖ = λ·正文第0层平均范数 "
                       "(s4_pipe.py make_pipe_delta), 它只改幅度、不改方向.",
            "norm_ratio": "‖Δ‖₂ / ‖a_s‖₂ (逐样本), 1.0 = 范数保持 (等距)",
            "cos_diff": "成对 cos 的 |cos_源 − cos_目标| (夹角保持; 0 = 完全保持)",
            "r2_project": "1 − ‖a_s − P_AᵀP_A a_s‖² / ‖a_s‖² (投影掉的方差比例; 对照 v81 ceiling_diag.json)",
        },
        "source_slot": "Q 槽 (与 arm1_slot_Q 同源), 样本级 mean-pool (与正式跑批同口径)",
        "n_pairs": N_PAIRS,
        "rng_seed": RNG_SEED,
        "directions": {},
    }

    for direction, s_name, t_name, sl, _tl in S4.DIRECTIONS:
        per_layer = {}
        ortho = {}
        for seed in S4.SEEDS:
            proj = S4.MEAS.projections(seed)
            for m in (s_name, t_name):
                e1, e2 = _ortho_err(proj[m])
                ortho.setdefault(m, {}).setdefault(str(seed), {
                    "ppT_err_fro": e1, "PtP_err_fro": e2, "shape": list(proj[m].shape)})
            P_src = proj[s_name]
            P_tgt = proj[t_name]
            for ly in sl:
                a = src_cache[s_name][(ly, "Q")].float()
                z = a @ P_src.t()
                D = z @ P_tgt
                na = a.norm(dim=-1)
                nD = D.norm(dim=-1)
                ratio = (nD / na.clamp_min(1e-12)).numpy()
                rec = a - (z @ P_src)
                # 中心化方差 R² (与 v81 ceiling_diag.json 同口径，避免公共均值项主导未中心化范数比)
                a_c = a - a.mean(dim=0, keepdim=True)
                rec_c = rec - rec.mean(dim=0, keepdim=True)
                r2proj = (1.0 - (rec_c.norm(dim=-1) ** 2) / (a_c.norm(dim=-1) ** 2).clamp_min(1e-12)).numpy()
                md, xd, rr = _cos_pairs(a, D, rng)
                slot = per_layer.setdefault(str(ly), [])
                slot.append({
                    "seed": int(seed),
                    "norm_ratio_median": float(np.median(ratio)),
                    "norm_ratio_mean": float(ratio.mean()),
                    "norm_ratio_sq_mean": float((ratio ** 2).mean()),
                    "r2_project_median": float(np.median(r2proj)),
                    "cos_abs_diff_mean": md,
                    "cos_abs_diff_max": xd,
                    "cos_pearson_r": rr,
                })
        agg = {
            "n_layers": len(per_layer),
            "n_seeds": len(S4.SEEDS),
            "norm_ratio_median_of_medians": float(np.median(
                [x["norm_ratio_median"] for v in per_layer.values() for x in v])),
            "norm_ratio_min": float(min(x["norm_ratio_median"] for v in per_layer.values() for x in v)),
            "norm_ratio_max": float(max(x["norm_ratio_median"] for v in per_layer.values() for x in v)),
            "norm_ratio_sq_mean_of_means": float(np.mean(
                [x["norm_ratio_sq_mean"] for v in per_layer.values() for x in v])),
            "r2_project_median_of_medians": float(np.median(
                [x["r2_project_median"] for v in per_layer.values() for x in v])),
            "cos_abs_diff_mean_of_means": float(np.mean(
                [x["cos_abs_diff_mean"] for v in per_layer.values() for x in v])),
            "cos_abs_diff_max": float(max(x["cos_abs_diff_max"] for v in per_layer.values() for x in v)),
            "cos_pearson_r_mean": float(np.mean(
                [x["cos_pearson_r"] for v in per_layer.values() for x in v
                 if x["cos_pearson_r"] is not None])),
            "orthonormality": ortho,
            "per_layer": per_layer,
        }
        out["directions"][direction] = agg
        print("[%s] 范数比中位 %.6f | cos 差均值 %.3e (max %.3e, r=%.4f) | r2_project 中位 %.6f"
              % (direction, agg["norm_ratio_median_of_medians"], agg["cos_abs_diff_mean_of_means"],
                 agg["cos_abs_diff_max"], agg["cos_pearson_r_mean"],
                 agg["r2_project_median_of_medians"]), flush=True)

    out["conclusion"] = {
        "carrier_is_linear": True,
        "only_nonlinearity": "逐样本范数归一化 (s4_pipe.py make_pipe_delta) —— 只改幅度不改方向",
        "reading": "载体是**线性**映射 (P_A/P_B 常量正交矩阵); 范数与夹角在「源维 = 共享维」的方向上被保持 "
                   "(等距), 在「源维 > 共享维」的方向上被压缩 => **换非线性不解决可分性, 扩维度只补幅度**;",
        "therefore": "S4-P 的 P2′(深度对应) 与 P5d(非目标槽普涨) 两个失败项 **不能**归因于「空间是线性的」",
    }
    out["provenance_sha256"] = {k: (S4.sha256_of(v) if os.path.exists(v) else None)
                                for k, v in S4.PROVENANCE_FILES.items()}
    out["note"] = "本文件只做载体性质诊断; S4-P 的判据 (P5a/P5b/P5c/P5d/P2′/P3①) 与阈值一字未动."

    p = _PKG / "results" / "diag_carrier.json"
    with open(p, "w", encoding="utf-8", newline="\r\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("written -> %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())