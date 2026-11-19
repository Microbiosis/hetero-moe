"""s2_lambda_calib.py - S2-1: 标定注入强度 lambda (S3 开跑前的重新预登记).

背景: v80 (S2) 的机械闸门查出 C4「没把 B 搞坏」被证伪 —— lambda=0.1 (块 Frobenius
口径) 在 BERT->Qwen 的 j in {3,6,12} 上把最终输出改 55-80%, 超出协议 §3 P3① 的 0.5 上限。
根因是 Qwen 残差流在第 3 层量级爆炸 (v77 B4 实测 RMS 0.277 -> 2.092): 落在爆炸层的
注入其绝对幅度被放大 117 倍。

本代**只做一件事**: 在写死的选择规则下, 标定出一个既「读出层可见」又「不把 B 搞坏」的
lambda, 供 S3 重新预登记使用。**不产生任何 P1a/P1b/P2/P3 判据**, 不改动 v80 的裁决。

标定协议 (全部跑前写死, 见 THRESHOLDS / LAMBDA_CANDIDATES):
  - 候选 lambda: LAMBDA_CANDIDATES (块 Frobenius 口径, 即 v80 的实现口径)
  - 标定块: **全部 84 块** (文本 64 + ViT 20, 与 S3 同网格), 每块只跑 arm1
    (arm2/arm3 的内容无关性已由 v80 实测覆盖: 同块三臂 final 扰动比值 < 1.6)
  - 选择规则: 取满足
      (R1) 84 块上 max final 扰动 <= CALIB_R1_MAX_FINAL
            (协议 P3① 上限 0.5 的 60%, 留 40% 余量给样本量漂移与强度->放大率的非线性)
      (R2) 84 块上 min readout 扰动 >= CALIB_R2_MIN_READOUT
            (协议 P3① 下界 1e-3, 不改)
    的**最大**候选 lambda; 若无一满足 => 裁决 CALIB_NO_FEASIBLE_LAMBDA (报回, 不硬选)
  - 样本量: 16 (与 S2 闸门同); S3 用 128 条 => R1 的余量即为此留
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util as ilu
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch


# ---------------------------------------------------------------- 常量 (跑前写死)
LAMBDA_CANDIDATES = (0.01, 0.02, 0.03, 0.04, 0.05)

THRESHOLDS = {
    "CALIB_R1_MAX_FINAL": 0.3,
    "CALIB_R2_MIN_READOUT": 1e-3,
    "CALIB_ARM": "arm1_slot_Q",
    "P3_1_upper_ref": 0.5,
    "P3_1_lower_ref": 1e-3,
}

VERDICT_SELECTED = "CALIB_LAMBDA_SELECTED"
VERDICT_NONE = "CALIB_NO_FEASIBLE_LAMBDA"


def _load_by_path(path: Path, name: str):
    spec = ilu.spec_from_file_location(name, str(path))
    mod = ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_PKG = Path(__file__).resolve().parent
_S2 = _load_by_path(_PKG / "s2_operator.py", "s2_operator_v80_by_path")

Backbone = _S2.Backbone
Injector = _S2.Injector
capture_layers = _S2.capture_layers
capture_slots = _S2.capture_slots
make_batch = _S2.make_batch
build_delta = _S2.build_delta
rel_l2 = _S2.rel_l2
fetch_models = _S2.fetch_models
fetch_texts = _S2.fetch_texts
fetch_images = _S2.fetch_images
MEAS = _S2.MEAS
DIRECTIONS = _S2.DIRECTIONS
SOURCE_SLOTS = _S2.SOURCE_SLOTS
sha256_of = _S2.sha256_of
CACHE_DIR = _S2.CACHE_DIR

PROVENANCE_FILES = {
    "v80_s2_operator": str(_PKG / "s2_operator.py"),
    "v80_s2_lambda_calib": str(_PKG / "s2_lambda_calib.py"),
}


def run() -> dict:
    t0 = time.time()
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(min(16, os.cpu_count() or 8))

    fetch_models()
    texts = fetch_texts(_S2.N_TEXT)
    imgs = fetch_images(_S2.N_IMG)
    proj = MEAS.projections(_S2.PROJ_SEED)

    src_cache: Dict[str, Dict[Tuple[int, str], torch.Tensor]] = {}
    for name, layers in (("Qwen", _S2.L_Q), ("BERT", _S2.L_B), ("ViT", _S2.L_V)):
        bb = Backbone(name)
        batch = make_batch(bb, name, texts, imgs)
        want = [(ly, slot) for ly in layers for slot in SOURCE_SLOTS]
        st = capture_slots(bb, batch, want)
        src_cache[name] = {k: st[k].mean(dim=1).clone() for k in want}
        del bb, batch, st
        gc.collect()
        print("[source] %s 完成" % name, flush=True)

    rows: List[dict] = []
    for dname, s_name, t_name, s_layers, t_layers in DIRECTIONS:
        bb = Backbone(t_name)
        batch = make_batch(bb, t_name, texts, imgs)
        n_layers = len(bb.layers)
        last = n_layers - 1
        ref = capture_layers(bb, batch,
                             out_layers=list(range(n_layers)),
                             in_layers=list(t_layers))
        p_a = proj[s_name]
        p_b = proj[t_name]
        for j in t_layers:
            h_ref = ref[("in", j)]
            for ly in s_layers:
                if s_name == t_name and ly == j:
                    continue
                z = src_cache[s_name][(ly, "Q")] @ p_a.t()
                for lam in LAMBDA_CANDIDATES:
                    delta, _lam = build_delta(z, p_b, h_ref, lam)
                    with Injector(bb.layers[j], delta):
                        out = capture_layers(bb, batch, out_layers=sorted({j, last}))
                    o_ref = ref[("out", j)]
                    o_inj = out[("out", j)]
                    f_ref = ref[("out", last)]
                    f_inj = out[("out", last)]
                    rows.append({
                        "direction": dname,
                        "source_layer": int(ly),
                        "target_layer": int(j),
                        "lambda": lam,
                        "rel_perturb_readout": round(rel_l2(o_ref, o_inj), 9),
                        "rel_perturb_final": round(rel_l2(f_ref, f_inj), 9),
                        "finite": bool(torch.isfinite(o_inj).all() and torch.isfinite(f_inj).all()),
                    })
                    del out, delta
            print("[calib] %s 目标层 %d 完成" % (dname, j), flush=True)
        del bb, batch, ref
        gc.collect()

    # ---------------------------------------------------------------- 选择规则 (写死)
    per_lambda = {}
    for lam in LAMBDA_CANDIDATES:
        sub = [r for r in rows if r["lambda"] == lam]
        max_final = max(r["rel_perturb_final"] for r in sub)
        min_readout = min(r["rel_perturb_readout"] for r in sub)
        r1 = max_final <= THRESHOLDS["CALIB_R1_MAX_FINAL"]
        r2 = min_readout >= THRESHOLDS["CALIB_R2_MIN_READOUT"]
        per_lambda["%.4g" % lam] = {
            "lambda": lam,
            "n_blocks": len(sub),
            "max_rel_perturb_final": round(max_final, 9),
            "min_rel_perturb_readout": round(min_readout, 9),
            "median_rel_perturb_readout": round(
                float(np.median([r["rel_perturb_readout"] for r in sub])), 9),
            "median_rel_perturb_final": round(
                float(np.median([r["rel_perturb_final"] for r in sub])), 9),
            "R1_final_within_cap": bool(r1),
            "R2_readout_above_floor": bool(r2),
            "feasible": bool(r1 and r2),
            "all_finite": all(r["finite"] for r in sub),
        }

    feasible = [per_lambda[k]["lambda"] for k in per_lambda if per_lambda[k]["feasible"]]
    chosen = max(feasible) if feasible else None
    verdict = VERDICT_SELECTED if chosen is not None else VERDICT_NONE

    return {
        "protocol": {
            "step": "A1 S2-1 (注入强度 lambda 的标定)",
            "scope": ("只标定 lambda; **不产生**任何 P1a/P1b/P2/P3 判据, 不改动 v80 的裁决; "
                      "标定结果供 S3 重新预登记使用"),
            "candidates": list(LAMBDA_CANDIDATES),
            "selection_rule": ("取满足 (R1) 84 块上 max final <= 0.3 且 "
                               "(R2) 84 块上 min readout >= 1e-3 的**最大**候选; "
                               "无一满足 => CALIB_NO_FEASIBLE_LAMBDA"),
            "thresholds": THRESHOLDS,
            "calib_blocks": "全部 84 块 (文本 64 + ViT 20), 每块只跑 arm1",
            "calib_arm": THRESHOLDS["CALIB_ARM"],
            "calib_arm_note": ("arm2/arm3 由 v80 的内容无关性结论覆盖 (同块三臂 final 扰动 "
                               "比值 < 1.6) ⇒ 标定只需 arm1"),
            "sample_size": _S2.BATCH,
            "sample_size_note": ("标定用 16 条 (与 S2 闸门同); S3 用 128 条 ⇒ R1 的 40% 余量 "
                                 "即为样本量漂移预留"),
            "discipline": [
                "候选集与选择规则跑前写死 (源码常量), 事后不得放宽",
                "选**最大**可行候选 ⇒ 注入信号尽量强 (探针功效优先)",
                "标定结果只用于 S3 的重新预登记, 不改写 v80 的任何裁决",
            ],
            "provenance_sha256": {k: sha256_of(v) for k, v in PROVENANCE_FILES.items()},
        },
        "per_lambda": per_lambda,
        "feasible_lambdas": sorted(feasible),
        "chosen_lambda": chosen,
        "verdict": verdict,
        "rows": rows,
        "runtime_sec": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_PKG / "results" / "lambda_calibration.json"))
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    res = run()
    for k, v in res["per_lambda"].items():
        print("  lambda=%-7s max_final=%.4f min_readout=%.6f  R1=%s R2=%s feasible=%s"
              % (k, v["max_rel_perturb_final"], v["min_rel_perturb_readout"],
                 v["R1_final_within_cap"], v["R2_readout_above_floor"], v["feasible"]), flush=True)
    print("      chosen_lambda: %s" % res["chosen_lambda"], flush=True)
    print("      verdict: %s  (%.0fs)" % (res["verdict"], res["runtime_sec"]), flush=True)
    if not args.no_write:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\r\n") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print("      wrote %s" % out, flush=True)
    return 0 if res["verdict"] == VERDICT_SELECTED else 1


if __name__ == "__main__":
    sys.exit(main())
