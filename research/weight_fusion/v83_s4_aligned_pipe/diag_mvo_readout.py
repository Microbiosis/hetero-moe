# -*- coding: utf-8 -*-
"""diag_mvo_readout.py — v83 post_hoc_only 诊断 (读出端双向槽特异 + Procrustes 等距性 + Qwen GQA M_vo 谱瓶颈).

⚠️ 本脚本产出的 results/diag_mvo_readout.json 明确标注 post_hoc_only:
   **不参与** S4-AP 的任何主判据裁决, **不推翻**任何跑前写死的阈值。

回答三个机理与解读问题:
  1. 正文读出端的**双向槽特异性** (对照 v82 readout_diag.py 与 v81 P1b):
     v82 中正文侧槽特异 arm1−arm2 在 Qwen->BERT 方向未过 (+0.000788, p = 0.1324);
     本代引入真 O 槽输出 + 训练集同层正交解耦后, Qwen->BERT 与 BERT->Qwen 两个方向的
     正文侧内容特异 (arm1−arm3) 与槽特异 (arm1−arm2) 是否**双向均显著成立**?
  2. 逐样本管道连通与通道寻址 (对照 v82 diag_pipe.json):
     v82 中逐样本管道注意力差符号检验 p = 1.0、逐样本通道寻址 flip 差 p = 0.787;
     本代引入第 0 层闭式 Q-K 接收场 + 训练集正交 Procrustes 对齐后, 逐样本层面是否显著?
  3. 为什么单点第 0 层写入下 P2′ (sl == tl) 仍受制约:
     测量 Qwen (24 层, GQA 14Q/2KV) 与 BERT (4 层, MHA) 各层 Value-Output 传输算子
     M_vo^(l) 的奇异值谱与 Frobenius 范数, 证明 Qwen 第 0 层存在 rank=128 的 GQA 秩瓶颈
     且 Frobenius 范数仅 0.2862 (比 BERT L0 小 43 倍、比 Qwen L1-L23 小 11-65 倍)。
"""
from __future__ import annotations

import collections
import importlib.util as ilu
import json
import math
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


S4AP = _load_by_path(_PKG / "s4_aligned_pipe.py", "s4_aligned_pipe_v83_by_path")


def _perm(pairs, n_perm: int = 10000, seed: int = 0):
    """块级配对置换检验 (整块同翻号 + Phipson–Smyth (cnt+1)/(n_perm+1) 平滑)。"""
    if not pairs:
        return None
    byb = collections.defaultdict(list)
    for b, a, c in pairs:
        byb[b].append(a - c)
    D = np.array(list(byb.values()), dtype=float)
    obs = float(np.median(D, axis=1).mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, D.shape[0], 1))
    st = np.median(D[None, :, :] * signs, axis=2).mean(axis=1)
    cnt = int((st >= obs - 1e-15).sum())
    return float((cnt + 1) / (n_perm + 1))


def _stat(pairs):
    d = np.array([x - y for _b, x, y in pairs], dtype=float)
    byb = collections.defaultdict(list)
    for b, x, y in pairs:
        byb[b].append(x - y)
    bm = np.array([np.median(v) for v in byb.values()], dtype=float)
    return {
        "n_obs": int(len(pairs)),
        "n_blocks": int(len(byb)),
        "median_obs_delta": float(np.median(d)),
        "n_positive_obs": int((d > 0).sum()),
        "median_block_delta": float(np.median(bm)),
        "mean_block_delta": float(bm.mean()),
    }


def _sign_p_normal_approx(n_pos: int, n_tot: int) -> float:
    if n_tot <= 0:
        return 1.0
    z = (n_pos - 0.5 * n_tot) / math.sqrt(0.25 * n_tot)
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


@torch.no_grad()
def _measure_mvo_spectra() -> dict:
    out = {"Qwen": {}, "BERT": {}}
    bb_q = S4AP.load_target("Qwen", eager=True)
    for ly in S4AP.L_Q:
        l = bb_q.layers[ly]
        wv = l.self_attn.v_proj.weight.detach().float()
        wo = l.self_attn.o_proj.weight.detach().float()
        ln_w = l.input_layernorm.weight.detach().float()
        wv_rep = wv.view(2, 1, 64, 896).expand(2, 7, 64, 896).reshape(896, 896)
        M_vo = (wo @ wv_rep) * ln_w.unsqueeze(0)
        s = torch.linalg.svdvals(M_vo)
        out["Qwen"][str(ly)] = {
            "rank_gt_1e5": int((s > 1e-5).sum()),
            "top1_sv": float(s[0]),
            "mean_top128_sv": float(s[:128].mean()),
            "sv_128": float(s[128]),
            "frobenius_norm": float(M_vo.norm()),
        }
    del bb_q

    bb_b = S4AP.load_target("BERT", eager=True)
    for ly in S4AP.L_B:
        l = bb_b.layers[ly]
        wv = l.attention.self.value.weight.detach().float()
        wo = l.attention.output.dense.weight.detach().float()
        ln_w = getattr(getattr(l.attention.output, "LayerNorm", None), "weight", None)
        M_vo = (ln_w.detach().float().unsqueeze(1) * (wo @ wv)) if ln_w is not None else (wo @ wv)
        s = torch.linalg.svdvals(M_vo)
        out["BERT"][str(ly)] = {
            "rank_gt_1e5": int((s > 1e-5).sum()),
            "top1_sv": float(s[0]),
            "mean_sv": float(s.mean()),
            "frobenius_norm": float(M_vo.norm()),
        }
    del bb_b
    return out


def main() -> int:
    res_dir = _PKG / "results"
    b_path = res_dir / "blocks.jsonl"
    a_path = res_dir / "attention.jsonl"
    if not b_path.exists() or not a_path.exists():
        print("[skip]缺少 blocks.jsonl 或 attention.jsonl", flush=True)
        return 1

    rows = [json.loads(l) for l in b_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    att_rows = [json.loads(l) for l in a_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    bl = collections.defaultdict(dict)
    for r in rows:
        bl[(r["direction"], int(r["source_layer"]), int(r["target_layer"]), int(r["seed"]))][r["arm"]] = r

    out = {
        "role": "post_hoc_only —— **读出端双向槽特异 + 逐样本寻址 + GQA M_vo 谱瓶颈诊断**, 不参与 S4-AP 任何裁决, 不推翻任何阈值",
        "readout_specificity": {},
        "per_sample_pipe_attention": {},
        "mvo_spectral_bottleneck": {},
    }

    defs = [
        ("arm1_minus_arm0_presence", "arm1_slot_Q", "arm0_pipe_zero", 11),
        ("arm1_minus_arm3_content_specificity", "arm1_slot_Q", "arm3_random_dir", 12),
        ("arm1_minus_arm2_slot_specificity", "arm1_slot_Q", "arm2_wrong_slot_O", 13),
        ("arm3_minus_arm0_random_noise", "arm3_random_dir", "arm0_pipe_zero", 14),
    ]
    for name, a, c, sd in defs:
        pairs = [(k[:3], v[a]["r2_q"], v[c]["r2_q"]) for k, v in bl.items() if a in v and c in v]
        s = _stat(pairs)
        s["permutation_p"] = _perm(pairs, seed=sd)
        s["n_perm"] = 10000
        bydir = {}
        for d in ("Qwen->BERT", "BERT->Qwen"):
            pp = [(k[:3], v[a]["r2_q"], v[c]["r2_q"]) for k, v in bl.items() if k[0] == d and a in v and c in v]
            sd2 = _stat(pp)
            sd2["permutation_p"] = _perm(pp, seed=sd)
            bydir[d] = sd2
        s["by_direction"] = bydir
        out["readout_specificity"][name] = s

    # 逐样本管道注意力与通道寻址 (对照 v82 diag_pipe.json)
    abl = collections.defaultdict(dict)
    for r in att_rows:
        abl[(r["direction"], int(r["source_layer"]), int(r["seed"]))][r["arm"]] = r

    # 严格仅在留出集 test_idx (64..127) 上评估逐样本管道注意力与通道寻址，避免混入 Procrustes 拟合集 (0..63)
    test_start = int(getattr(S4AP, "TEXT_SPLIT_TRAIN", 64))
    ps_10, ps_13, ps_flip = [], [], []
    blk_10, blk_13, blk_flip = [], [], []
    for k, v in sorted(abl.items()):
        if not all(a in v for a in ("arm0_pipe_zero", "arm1_slot_Q", "arm2_wrong_slot_O", "arm3_random_dir")):
            continue
        p0 = (np.array(v["arm0_pipe_zero"]["per_sample_ch0"]) + np.array(v["arm0_pipe_zero"]["per_sample_ch1"]))[test_start:]
        p1 = (np.array(v["arm1_slot_Q"]["per_sample_ch0"]) + np.array(v["arm1_slot_Q"]["per_sample_ch1"]))[test_start:]
        p3 = (np.array(v["arm3_random_dir"]["per_sample_ch0"]) + np.array(v["arm3_random_dir"]["per_sample_ch1"]))[test_start:]
        fl1 = (np.array(v["arm1_slot_Q"]["per_sample_ch0"]) - np.array(v["arm1_slot_Q"]["per_sample_ch1"]))[test_start:]
        fl2 = (np.array(v["arm2_wrong_slot_O"]["per_sample_ch0"]) - np.array(v["arm2_wrong_slot_O"]["per_sample_ch1"]))[test_start:]
        ps_10.extend((p1 - p0).tolist())
        ps_13.extend((p1 - p3).tolist())
        ps_flip.extend((fl1 - fl2).tolist())
        blk_10.append((k[:2], float(np.median(p1)), float(np.median(p0))))
        blk_13.append((k[:2], float(np.median(p1)), float(np.median(p3))))
        blk_flip.append((k[:2], float(np.median(fl1)), float(np.median(fl2))))

    for tag, arr, blk_pairs, sd in (
        ("arm1_minus_arm0_per_sample", ps_10, blk_10, 21),
        ("arm1_minus_arm3_per_sample", ps_13, blk_13, 22),
        ("flip_arm1_minus_arm2_per_sample", ps_flip, blk_flip, 23),
    ):
        a_np = np.array(arr, dtype=float)
        npos = int((a_np > 0).sum())
        ntot = int(a_np.size)
        out["per_sample_pipe_attention"][tag] = {
            "eval_scope": f"heldout_test_only[{test_start}:]",
            "n_sample_obs": ntot,
            "n_positive": npos,
            "positive_fraction": round(npos / max(1, ntot), 6),
            "median_delta": float(np.median(a_np)),
            "block_permutation_p": _perm(blk_pairs, seed=sd),
            "sign_test_p_naive_iid": _sign_p_normal_approx(npos, ntot),
        }

    out["mvo_spectral_bottleneck"] = _measure_mvo_spectra()
    q0_fro = out["mvo_spectral_bottleneck"]["Qwen"]["0"]["frobenius_norm"]
    b0_fro = out["mvo_spectral_bottleneck"]["BERT"]["0"]["frobenius_norm"]
    q23_fro = out["mvo_spectral_bottleneck"]["Qwen"]["23"]["frobenius_norm"]
    out["mvo_spectral_bottleneck"]["reading"] = (
        f"Qwen L0 的 M_vo 秩仅 128 (GQA 2 KV 头 × 64) 且 Frobenius 范数仅 {q0_fro:.4f} "
        f"(比 BERT L0 的 {b0_fro:.4f} 小 {b0_fro / q0_fro:.1f} 倍, 比 Qwen L23 的 {q23_fro:.4f} 小 "
        f"{q23_fro / q0_fro:.1f} 倍) ⇒ 单点第 0 层管道注入在 BERT->Qwen 方向的第 0 层读出 (tl=0) "
        f"必然受 40-65 倍带宽节流, 信息需经 L1+ 放大后才完全展开。"
    )

    dst = res_dir / "diag_mvo_readout.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[diag] 已写出 %s" % dst, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
