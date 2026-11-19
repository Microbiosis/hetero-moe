"""s2_0_align.py — S2-0: 定死 align[s|A→B] 的来源 (真模型激活上的成对对齐拟合与核验)。

问题 (S3 的头号工程风险, 见 docs/S3_PROTOCOL.md §2.3):
  S3 的注入算子需要一个 256×256 的跨模型对齐映射:
      z~ = align[s|A→B] · z      (z = 源模型第 l 层槽 s 激活的共享空间表示)
      Δ_B = P_Bᵀ · z~            (映回目标模型的隐空间)
  但 align 的**来源**有两个候选, 各自的适用性未知:
    (甲) v75 的成对对齐矩阵 —— 拟合在**合成切片权重 + 随机输入**上 (不是真模型激活)
    (乙) 在**真模型激活**上现拟合 (同一闭式公式族: ridge / procrustes / low-rank)

本代要回答的问题 (可证伪):
  Q1 存在这样的 align 吗? —— 拟合后, 同槽跨模型是否比异槽更相似 (槽特异性)?
  Q2 它比「不对齐」强吗? —— 否则注入只是加了个无意义的旋转
  Q3 结论对 λ 稳健吗? —— 否则是某个超参的偶然
  Q4 (甲) 能不能直接用? —— 若 (甲) 不过而 (乙) 过, 则来源只能是 (乙)

判据 (跑前写死, 事后不得放宽):
  A1 槽特异性: matched CKA > wrong-slot CKA, 留出集上双侧符号检验 p < 0.05
  A2 优于不对齐: matched CKA > no-align CKA, 同上
  A3 λ 稳健: ridge 在 λ ∈ {0.1, 1.0, 10.0} 三个取值上 A1/A2 均成立
  A4 决策: 在通过 A1∧A2 的候选里取「留出集 matched−noalign 中位数」最大者;
          并列时取更简单的 (none < lowrank16 < procrustes < ridge < v75_synth);
          全不过 ⇒ 机器裁决 ALIGN_SOURCE_NOT_FOUND (负结果, S3 的注入算子必须重新设计)

三条纪律 (与 v75/v77/v78 同款, 一条都不能破):
  1. 拟合/评估**块级分离**: 评估块 (l, j) 的对齐只用 l'≠l 且 j'≠j 的块拟合
     (leave-block-out ⇒ 源层与目标层都与拟合集不相交, 防循环论证)
  2. 样本**留出**: 前 64 条训练拟合, 后 64 条评估 (按标签分层切)
  3. 阈值跑前写死; 事后拆解只能标 post_hoc_only, 不得用来翻案

本代**不产生**任何 P1a/P1b/P2/P3 判据, 也**不改动** v77/v78 的裁决。
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

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch

_PKG = Path(__file__).resolve().parent


def _load_by_path(path: Path, name: str):
    """按**路径**载入同线模块 (同 v75/v76/v77/v78 的做法: 同线包名 import 会被
    tools/check_self_containment.py 判成跨线引用)。"""
    spec = ilu.spec_from_file_location(name, str(path))
    mod = ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_PRE = _load_by_path(_PKG.parent / "v78_s3_preflight" / "s3_preflight.py", "v78_pre_by_path")
_G2 = _load_by_path(_PKG.parent / "v75_weight_conjugation" / "g2_conjugation.py",
                    "v75_g2_by_path")

Backbone = _PRE.Backbone
capture_slots = _PRE.capture_slots
eff_rank = _PRE.eff_rank
text_batches = _PRE.text_batches
fetch_texts = _PRE.fetch_texts
MEAS = _PRE.MEAS

fit_transport = _G2.fit_transport
load_wall = _G2.load_wall
native_units = _G2.native_units
shared_ops = _G2.shared_ops
cka = MEAS.cka
projections = MEAS.projections

# ---------------------------------------------------------------- 判据阈值 (跑前写死)
THRESHOLDS = {
    "sign_p_max": 0.05,          # 双侧符号检验显著性
    "min_blocks": 32,            # 功效闸门: 评估块少于它不开跑
    "ridge_lam": 1.0,            # 主 λ (沿用 v75 的 RIDGE_LAM)
    "ridge_lam_robust": [0.1, 10.0],   # A3 稳健性检查的两个额外 λ
    "lowrank_k": 16,             # 低秩子空间维数
    "n_train": 64,               # 训练样本数 (前 64 条, 按标签分层)
    "n_test": 64,                # 留出样本数 (后 64 条)
    "layers": [0, 1, 2, 3],      # 评估网格的层 (两个底座的前 4 层: 切片只覆盖 0-3)
    "slots": ["Q", "O"],         # Q = 主注入槽; O = 错槽对照
    "dirs": [["Qwen", "BERT"], ["BERT", "Qwen"]],
}

CANDIDATES = ("none", "lowrank16", "procrustes", "ridge", "v75_synth")
#: 决策时的「更简单者优先」次序 (同 CANDIDATES 的顺序)
SIMPLICITY_ORDER = ("none", "lowrank16", "procrustes", "ridge", "v75_synth")


# ---------------------------------------------------------------- 小工具
def _sha256_of(path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _sign_test_two_sided(vals: Sequence[float]) -> dict:
    """精确双侧符号检验: p = min(1, 2·P(X >= max(pos, n - pos)))。"""
    vals = [float(v) for v in vals]
    n = len(vals)
    if n == 0:
        return {"n_blocks": 0, "n_positive": 0, "sign_p_two_sided": None,
                "median": None, "mean": None}
    pos = sum(1 for v in vals if v > 0)
    k_hi = max(pos, n - pos)
    p1 = sum(math.comb(n, k) for k in range(k_hi, n + 1)) / (2 ** n)
    return {"n_blocks": n, "n_positive": pos,
            "sign_p_two_sided": float(min(1.0, 2.0 * p1)),
            "median": float(np.median(vals)), "mean": float(np.mean(vals))}


def _center(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(0, keepdim=True)


def _top_pc(X: torch.Tensor) -> torch.Tensor:
    """中心化矩阵的 top-1 右奇异向量 (共同成分方向)。"""
    Xc = _center(X)
    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    return Vh[0]


def _remove_pc(X: torch.Tensor, pc: torch.Tensor) -> torch.Tensor:
    """去掉给定方向 (pc 已在 X 的中心化空间里估计)。"""
    return X - (X @ pc).unsqueeze(1) * pc.unsqueeze(0)


def _fit_ridge(U: torch.Tensor, V: torch.Tensor, lam: float) -> torch.Tensor:
    """闭式岭回归: 求 M 使 U·M ≈ V (行向量约定)。与 v75.fit_transport 同式。"""
    d = U.shape[1]
    G = U.t() @ U + lam * torch.eye(d, dtype=U.dtype)
    return torch.linalg.solve(G, U.t() @ V)


def _fit_procrustes(U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """正交 Procrustes: 在正交约束下求 min ||U·M − V||。"""
    P = U.t() @ V
    Uu, _, Vv = torch.linalg.svd(P)
    return Uu @ Vv


def _fit_lowrank(U: torch.Tensor, V: torch.Tensor, lam: float, k: int) -> torch.Tensor:
    """低秩子空间对齐: 在 U 的 top-k PCA 子空间里做岭回归, 再映回全空间。"""
    Uc = _center(U)
    _, _, Vh = torch.linalg.svd(Uc, full_matrices=False)
    P = Vh[:k].t()                       # [d, k]
    Up = Uc @ P
    A = _fit_ridge(Up, V, lam)
    return P @ A                         # [d, d], 作用: u → u·P·A


def _v75_synthetic_outputs(nat, proj62: Dict[str, torch.Tensor],
                             slot: str) -> Dict[Tuple[str, str], torch.Tensor]:
    """复刻 v75 的合成侧输入: 切片算子 + 随机共享输入 + 中心化输出。

    与 v75 的差别只有一处 (且必须如此): **正交帧换成 v62 的 projections(0)**。
    帧是规范选择 (随机正交基), 方法不变; 换了帧才能让所有候选在**同一帧**里比,
    否则 CKA 跨候选不可比 (CKA 只在正交换下不变, 而两套随机帧之间不是正交换)。
    """
    g = torch.Generator().manual_seed(_G2._seed_of("X", 0))
    X = torch.randn(_G2.N_REP, _G2.DS, generator=g)
    sh = shared_ops(nat, proj62)
    out: Dict[Tuple[str, str], torch.Tensor] = {}
    for (m, slotname, ly), M in sh.items():
        if slotname == slot and ly in [str(x) for x in THRESHOLDS["layers"]]:
            out[(m, ly)] = _G2.outputs(M, X)
    return out


# ---------------------------------------------------------------- 激活抓取
def capture_shared_layers(model_name: str, texts: List[dict],
                          layers: Sequence[int], slots: Sequence[str],
                          proj: Dict[str, torch.Tensor]) -> Dict[Tuple[int, str], torch.Tensor]:
    """跑真模型, 抓 (层, 槽) 激活并 mean-pool, 再投到共享空间 → [(N, 256)]。"""
    bb = Backbone(model_name)
    batches = text_batches(bb, texts)
    acc: Dict[Tuple[int, str], List[torch.Tensor]] = {}
    want = [(ly, s) for ly in layers for s in slots]
    for batch in batches:
        st = capture_slots(bb, batch, want)
        for k, v in st.items():
            acc.setdefault(k, []).append(v.mean(dim=1))     # token mean-pool
    out = {k: torch.cat(v, dim=0) for k, v in acc.items()}
    del bb, batches
    gc.collect()
    q = proj[model_name]
    return {k: (v @ q.t()) for k, v in out.items()}          # 共享空间 (行向量约定)


def stratified_split(labels: np.ndarray, n_train: int) -> Tuple[np.ndarray, np.ndarray]:
    """按标签分层的确定性切分: 每类前一半进训练, 后一半进留出。"""
    tr, te = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        half = len(idx) // 2
        tr.extend(idx[:half].tolist())
        te.extend(idx[half:].tolist())
    tr_a = np.array(sorted(tr))
    te_a = np.array(sorted(te))
    if len(tr_a) > n_train or len(te_a) > n_train:
        # 分层切分给出的数量可能与目标略有出入 (各类样本数不等): 只截断到目标上限,
        # 实际数量记进快照的 protocol.split, 不静默改口径。
        tr_a = tr_a[:n_train]
        te_a = te_a[:n_train]
    return tr_a, te_a


# ---------------------------------------------------------------- 候选对齐矩阵
def build_v75_synth(nat, proj62: Dict[str, torch.Tensor],
                    slot: str) -> Dict[Tuple[str, str, int], torch.Tensor]:
    """候选 (甲): v75 的成对对齐**流程** (切片 + 随机输入 + 闭式 ridge + cyclic LOO)。

    v75 的方案是 cyclic leave-one-layer-out: 测第 l 层 ⇒ 在第 (l+1)%4 层拟合;
    本代沿用该方案, 且沿用 v75 的闭式 fit_transport 与同一份切片、同一随机输入种子。
    """
    out = _v75_synthetic_outputs(nat, proj62, slot)
    maps: Dict[Tuple[str, str, int], torch.Tensor] = {}
    for ly in THRESHOLDS["layers"]:
        fit_layer = str((ly + 1) % 4)
        for (m1, m2) in (("Qwen", "BERT"), ("BERT", "Qwen")):
            U = out[(m1, fit_layer)]
            V = out[(m2, fit_layer)]
            # fit_transport 返回 A 使 U @ A.t() ≈ V；下游按行向量约定算 uA_test @ M，故此处取转置
            maps[(m1, m2, ly)] = fit_transport(U, V).t().contiguous()
    return maps


def fit_candidates(U: torch.Tensor, V: torch.Tensor,
                   v75_maps: Optional[Dict[Tuple[str, str, int], torch.Tensor]],
                   key: Tuple[str, str, int]) -> Dict[str, torch.Tensor]:
    """在给定的 (源层, 目标层) 评估块上拟合各候选。"""
    Uc, Vc = _center(U), _center(V)
    maps: Dict[str, torch.Tensor] = {"none": torch.eye(U.shape[1], dtype=U.dtype)}
    maps["lowrank16"] = _fit_lowrank(Uc, Vc, THRESHOLDS["ridge_lam"], THRESHOLDS["lowrank_k"])
    maps["procrustes"] = _fit_procrustes(Uc, Vc)
    maps["ridge"] = _fit_ridge(Uc, Vc, THRESHOLDS["ridge_lam"])
    if v75_maps is not None and key in v75_maps:
        maps["v75_synth"] = v75_maps[key]
    return maps


# ---------------------------------------------------------------- 主流程
def run() -> dict:
    t0 = time.time()
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(min(16, os.cpu_count() or 8))

    layers = THRESHOLDS["layers"]
    slots = THRESHOLDS["slots"]
    dirs = THRESHOLDS["dirs"]

    texts = fetch_texts(128)
    labels = np.array([t["label"] for t in texts], dtype=np.int64)
    print("[corpus] ag_news %d 条, 标签分布 %s" % (len(texts), np.bincount(labels).tolist()),
          flush=True)

    proj62 = projections(0)
    tr_idx, te_idx = stratified_split(labels, THRESHOLDS["n_train"])
    print("[split] train=%d test=%d (按标签分层)" % (len(tr_idx), len(te_idx)), flush=True)

    shared: Dict[str, Dict[Tuple[int, str], torch.Tensor]] = {}
    for name in ("Qwen", "BERT"):
        shared[name] = capture_shared_layers(name, texts, layers, slots, proj62)
        print("[model] %s 槽激活已抓 (%d 层 × %d 槽)" % (name, len(layers), len(slots)),
              flush=True)

    # 候选 (甲): v75 的成对对齐矩阵 (合成切片 + 随机输入)
    print("[v75] 载入切片并构造成对对齐矩阵 …", flush=True)
    nat = native_units(load_wall())
    v75_maps: Dict[str, Dict[Tuple[str, str, int], torch.Tensor]] = {}
    for s in slots:
        v75_maps[s] = build_v75_synth(nat, proj62, s)
    del nat
    gc.collect()

    rows: List[dict] = []
    for slot in slots:
        other = "O" if slot == "Q" else "Q"
        for (m1, m2) in dirs:
            UA = shared[m1]
            VB = shared[m2]
            for l in layers:
                for j in layers:
                    # leave-block-out: 源层 l'≠l 且 目标层 j'≠j 的块才进拟合集
                    Us, Vs = [], []
                    for lp in layers:
                        if lp == l:
                            continue
                        for jp in layers:
                            if jp == j:
                                continue
                            Us.append(UA[(lp, slot)][tr_idx])
                            Vs.append(VB[(jp, slot)][tr_idx])
                    U_fit = torch.cat(Us, dim=0)
                    V_fit = torch.cat(Vs, dim=0)
                    maps = fit_candidates(U_fit, V_fit, v75_maps.get(slot), (m1, m2, l))

                    uA_test = UA[(l, slot)][te_idx]
                    v_match = VB[(j, slot)][te_idx]
                    v_wrong = VB[(j, other)][te_idx]
                    for cand, M in maps.items():
                        rows.append({
                            "slot": slot, "dir": m1 + "->" + m2,
                            "src_layer": l, "tgt_layer": j, "candidate": cand,
                            "n_fit_pairs": int(U_fit.shape[0]),
                            "cka_matched": float(cka(uA_test @ M, v_match)),
                            "cka_wrong_slot": float(cka(uA_test @ M, v_wrong)),
                            "cka_noalign": float(cka(uA_test, v_match)),
                        })
            print("[fit] %s 槽 %s->%s 完成" % (slot, m1, m2), flush=True)

    # ---------------- 汇总与判据 ----------------
    by_cand: Dict[str, dict] = {}
    for cand in CANDIDATES:
        per_slot = {}
        for slot in slots:
            rs = [r for r in rows if r["candidate"] == cand and r["slot"] == slot]
            if not rs:
                continue
            d_spec = [r["cka_matched"] - r["cka_wrong_slot"] for r in rs]
            d_base = [r["cka_matched"] - r["cka_noalign"] for r in rs]
            t_spec = _sign_test_two_sided(d_spec)
            t_base = _sign_test_two_sided(d_base)
            per_slot[slot] = {
                "n_blocks": len(rs),
                "A1_slot_specificity": t_spec,
                "A2_beats_noalign": t_base,
                "mean_cka_matched": float(np.mean([r["cka_matched"] for r in rs])),
                "mean_cka_wrong_slot": float(np.mean([r["cka_wrong_slot"] for r in rs])),
                "mean_cka_noalign": float(np.mean([r["cka_noalign"] for r in rs])),
                "A1_ok": bool(t_spec["median"] is not None and t_spec["median"] > 0
                              and t_spec["sign_p_two_sided"] is not None
                              and t_spec["sign_p_two_sided"] < THRESHOLDS["sign_p_max"]),
                "A2_ok": bool(t_base["median"] is not None and t_base["median"] > 0
                              and t_base["sign_p_two_sided"] is not None
                              and t_base["sign_p_two_sided"] < THRESHOLDS["sign_p_max"]),
            }
        by_cand[cand] = per_slot

    # A3: ridge 的 λ 稳健性 (只在 λ 上重拟合 ridge, 其余不变)
    lam_robust = {}
    for lam in THRESHOLDS["ridge_lam_robust"]:
        lam_rows = []
        for slot in slots:
            other = "O" if slot == "Q" else "Q"
            for (m1, m2) in dirs:
                UA, VB = shared[m1], shared[m2]
                for l in layers:
                    for j in layers:
                        Us, Vs = [], []
                        for lp in layers:
                            if lp == l:
                                continue
                            for jp in layers:
                                if jp == j:
                                    continue
                                Us.append(UA[(lp, slot)][tr_idx])
                                Vs.append(VB[(jp, slot)][tr_idx])
                        U_fit = torch.cat(Us, 0)
                        V_fit = torch.cat(Vs, 0)
                        M = _fit_ridge(_center(U_fit), _center(V_fit), lam)
                        uA_test = UA[(l, slot)][te_idx]
                        lam_rows.append({
                            "slot": slot, "dir": m1 + "->" + m2, "src_layer": l, "tgt_layer": j,
                            "cka_matched": float(cka(uA_test @ M, VB[(j, slot)][te_idx])),
                            "cka_wrong_slot": float(cka(uA_test @ M, VB[(j, other)][te_idx])),
                            "cka_noalign": float(cka(uA_test, VB[(j, slot)][te_idx])),
                        })
        t_spec = _sign_test_two_sided([r["cka_matched"] - r["cka_wrong_slot"] for r in lam_rows])
        t_base = _sign_test_two_sided([r["cka_matched"] - r["cka_noalign"] for r in lam_rows])
        lam_robust[str(lam)] = {
            "A1_slot_specificity": t_spec, "A2_beats_noalign": t_base,
            "A1_ok": bool(t_spec["median"] and t_spec["median"] > 0
                          and t_spec["sign_p_two_sided"] < THRESHOLDS["sign_p_max"]),
            "A2_ok": bool(t_base["median"] and t_base["median"] > 0
                          and t_base["sign_p_two_sided"] < THRESHOLDS["sign_p_max"]),
        }

    # A4: 决策 (Q 槽为主判据: S3 的 arm1 需要 Q 槽映射; O 槽映射是错槽对照臂)
    n_blocks_q = by_cand.get("ridge", {}).get("Q", {}).get("n_blocks", 0)
    eligible = [c for c in CANDIDATES
                if by_cand.get(c, {}).get("Q", {}).get("A1_ok")
                and by_cand.get(c, {}).get("Q", {}).get("A2_ok")]
    chosen = None
    if eligible:
        chosen = max(eligible, key=lambda c: (by_cand[c]["Q"]["A2_beats_noalign"]["median"],
                                               -SIMPLICITY_ORDER.index(c)))
    criteria = {
        "power_gate_blocks_ok": bool(n_blocks_q >= THRESHOLDS["min_blocks"]),
        "A1_some_candidate_has_slot_specificity": bool(
            any(by_cand.get(c, {}).get("Q", {}).get("A1_ok") for c in CANDIDATES)),
        "A2_some_candidate_beats_noalign": bool(
            any(by_cand.get(c, {}).get("Q", {}).get("A2_ok") for c in CANDIDATES)),
        "A3_ridge_lambda_robust": bool(
            all(v["A1_ok"] and v["A2_ok"] for v in lam_robust.values())),
        "A4_source_decided_uniquely": bool(chosen is not None and len(eligible) >= 1),
    }
    verdict = "ALIGN_SOURCE_DECIDED" if all(criteria.values()) else "ALIGN_SOURCE_NOT_FOUND"

    # ---------------- post_hoc 诊断 (不参与判据, 只为解释负结果) ----------------
    post_hoc: Dict[str, dict] = {
        "status": "post_hoc_only_not_used_for_verdict",
        "note": ("本段**不参与**本代裁决 (判据已在上面写死并算完)。它只回答两个问题: "
                 "(1) 若对齐矩阵按块单独拟合 (用该块自己的 63 条训练样本), 能不能超过不对齐基线?"
                 " —— 若能, 说明负结果来自『跨层池化的映射太粗 / 数据不够』; "
                 "(2) 池化拟合在训练集上是否本来就高 —— 若是, 说明是过拟合而不是没对应关系。"),
        "block_specific": {},
        "in_sample_vs_heldout": {},
    }
    for cand in ("ridge", "procrustes", "lowrank16"):
        ph_rows = []
        for slot in slots:
            other = "O" if slot == "Q" else "Q"
            for (m1, m2) in dirs:
                UA, VB = shared[m1], shared[m2]
                for l in layers:
                    for j in layers:
                        U_fit = UA[(l, slot)][tr_idx]
                        V_fit = VB[(j, slot)][tr_idx]
                        Uc, Vc = _center(U_fit), _center(V_fit)
                        if cand == "ridge":
                            M = _fit_ridge(Uc, Vc, THRESHOLDS["ridge_lam"])
                        elif cand == "procrustes":
                            M = _fit_procrustes(Uc, Vc)
                        else:
                            M = _fit_lowrank(Uc, Vc, THRESHOLDS["ridge_lam"],
                                             THRESHOLDS["lowrank_k"])
                        uA_test = UA[(l, slot)][te_idx]
                        ph_rows.append({
                            "cka_matched": float(cka(uA_test @ M, VB[(j, slot)][te_idx])),
                            "cka_wrong_slot": float(cka(uA_test @ M, VB[(j, other)][te_idx])),
                            "cka_noalign": float(cka(uA_test, VB[(j, slot)][te_idx])),
                        })
        post_hoc["block_specific"][cand] = {
            "A1_slot_specificity": _sign_test_two_sided(
                [r["cka_matched"] - r["cka_wrong_slot"] for r in ph_rows]),
            "A2_beats_noalign": _sign_test_two_sided(
                [r["cka_matched"] - r["cka_noalign"] for r in ph_rows]),
            "mean_cka_matched": float(np.mean([r["cka_matched"] for r in ph_rows])),
            "mean_cka_wrong_slot": float(np.mean([r["cka_wrong_slot"] for r in ph_rows])),
            "mean_cka_noalign": float(np.mean([r["cka_noalign"] for r in ph_rows])),
        }
    # 池化 ridge: 训练集内 vs 留出集
    tr_rows, te_rows = [], []
    for slot in slots:
        other = "O" if slot == "Q" else "Q"
        for (m1, m2) in dirs:
            UA, VB = shared[m1], shared[m2]
            for l in layers:
                for j in layers:
                    Us, Vs = [], []
                    for lp in layers:
                        if lp == l:
                            continue
                        for jp in layers:
                            if jp == j:
                                continue
                            Us.append(UA[(lp, slot)][tr_idx])
                            Vs.append(VB[(jp, slot)][tr_idx])
                    M = _fit_ridge(_center(torch.cat(Us, 0)), _center(torch.cat(Vs, 0)),
                                   THRESHOLDS["ridge_lam"])
                    uA_tr = UA[(l, slot)][tr_idx]
                    uA_te = UA[(l, slot)][te_idx]
                    tr_rows.append(float(cka(uA_tr @ M, VB[(j, slot)][tr_idx])))
                    te_rows.append(float(cka(uA_te @ M, VB[(j, slot)][te_idx])))
    post_hoc["in_sample_vs_heldout"] = {
        "ridge_pooled_train_cka_median": float(np.median(tr_rows)),
        "ridge_pooled_test_cka_median": float(np.median(te_rows)),
        "noalign_test_cka_median": float(np.median(
            [r["cka_noalign"] for r in rows if r["candidate"] == "ridge"])),
    }

    # (3) 共同成分诊断: 池化表示的有效秩; 去掉 top1 PC 后, 块级映射还剩多少对应关系
    post_hoc["pooled_eff_rank"] = {
        "%s|L%d|%s" % (m, l, s): float(eff_rank(shared[m][(l, s)]))
        for m in ("Qwen", "BERT") for l in layers for s in slots
    }
    res_rows = []
    for slot in slots:
        other = "O" if slot == "Q" else "Q"
        for (m1, m2) in dirs:
            UA, VB = shared[m1], shared[m2]
            for l in layers:
                for j in layers:
                    uA_tr = UA[(l, slot)][tr_idx]
                    uA_te = UA[(l, slot)][te_idx]
                    v_tr = VB[(j, slot)][tr_idx]
                    v_te = VB[(j, slot)][te_idx]
                    vo_te = VB[(j, other)][te_idx]
                    pa, pb = _top_pc(uA_tr), _top_pc(v_tr)
                    uA_r = _remove_pc(uA_te, pa)
                    v_r = _remove_pc(v_te, pb)
                    vo_r = _remove_pc(vo_te, pb)
                    U_fit = _remove_pc(uA_tr, pa)
                    V_fit = _remove_pc(v_tr, pb)
                    M = _fit_ridge(_center(U_fit), _center(V_fit), THRESHOLDS["ridge_lam"])
                    res_rows.append({
                        "cka_matched": float(cka(uA_r @ M, v_r)),
                        "cka_wrong_slot": float(cka(uA_r @ M, vo_r)),
                        "cka_noalign": float(cka(uA_r, v_r)),
                    })
    post_hoc["residualized_top1pc_blockspecific_ridge"] = {
        "A1_slot_specificity": _sign_test_two_sided(
            [r["cka_matched"] - r["cka_wrong_slot"] for r in res_rows]),
        "A2_beats_noalign": _sign_test_two_sided(
            [r["cka_matched"] - r["cka_noalign"] for r in res_rows]),
        "mean_cka_matched": float(np.mean([r["cka_matched"] for r in res_rows])),
        "mean_cka_wrong_slot": float(np.mean([r["cka_wrong_slot"] for r in res_rows])),
        "mean_cka_noalign": float(np.mean([r["cka_noalign"] for r in res_rows])),
    }

    # ---------------- 产物: 块级对齐矩阵 (给 S3 用) ----------------
    # ⚠️ **这不是本代的"被选中来源"** (本代裁决是 ALIGN_SOURCE_NOT_FOUND, 没有来源被选中)。
    # 这是按 **post_hoc 诊断** (块级映射在留出集上确实超过不对齐基线) 产出的工程件:
    #   - 块级 = 每个 (方向, 源层 l, 目标层 j, 槽) 各一个映射, 只用该块的**训练样本**拟合;
    #   - 族 = ridge (post_hoc 的中位增量最大: ridge +0.0677 > lowrank16 +0.0395 > procrustes 0.0000);
    #   - S3 必须在自己的留出集上**重新验证**它 (P1a/P1b 本身就是留出集上的判据)。
    prod: Dict[str, torch.Tensor] = {}
    for slot in slots:
        for (m1, m2) in dirs:
            UA, VB = shared[m1], shared[m2]
            for l in layers:
                for j in layers:
                    U_fit = UA[(l, slot)][tr_idx]
                    V_fit = VB[(j, slot)][tr_idx]
                    prod["%s|%s->%s|L%d->L%d" % (slot, m1, m2, l, j)] = _fit_ridge(
                        _center(U_fit), _center(V_fit), THRESHOLDS["ridge_lam"])
    maps_path = _PKG / "results" / "align_maps.pt"
    maps_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"chosen_source": chosen,
                "artifact_kind": "block_specific_train_fit_ridge (post_hoc 依据; 非本代裁决产物)",
                "maps": prod}, maps_path)
    maps_sha = _sha256_of(maps_path)

    return {
        "protocol": {
            "purpose": "S2-0: 定死 align[s|A->B] 的来源 (真模型激活 vs v75 合成拟合)",
            "route": "beta (真 HF 模型 + 真实语料)",
            "seed": 0, "device": "cpu", "dtype": "float32",
            "corpus": "fancyzhx/ag_news train 128 条 (与 v77/v78 同一缓存 agnews_128.json)",
            "models": {"Qwen": "Qwen/Qwen2.5-0.5B-Instruct",
                       "BERT": "google/bert_uncased_L-4_H-256_A-4"},
            "thresholds": THRESHOLDS,
            "candidates": list(CANDIDATES),
            "discipline": [
                "leave-block-out: 评估块 (l,j) 的对齐只用 l'!=l 且 j'!=j 的块拟合",
                "样本留出: 前 64 条训练拟合 / 后 64 条评估 (按标签分层)",
                "槽激活按 v62 槽语义取 (投影类取模块输出 / O 槽取模块输入), 与 v77/v78 同口径",
                "共享空间用 v62 的 projections(0) 正交投影; v75 的矩阵按 S = Q62·Q75ᵀ 换算到同一帧",
            ],
            "provenance_sha256": {
                "v78_preflight": _sha256_of(_PKG.parent / "v78_s3_preflight" / "s3_preflight.py"),
                "v75_g2": _sha256_of(_PKG.parent / "v75_weight_conjugation" / "g2_conjugation.py"),
                "v62_measure": _sha256_of(str(_G2.V62_MEASURE_PATH)),
            },
            "split": {"train_idx": tr_idx.tolist(), "test_idx": te_idx.tolist()},
        },
        "by_candidate": by_cand,
        "lambda_robustness": lam_robust,
        "criteria": criteria,
        "verdict": verdict,
        "chosen_source": chosen,
        "eligible_candidates": eligible,
        "align_maps_file": "results/align_maps.pt",
        "align_maps_sha256": maps_sha,
        "n_align_maps": len(prod),
        "rows": rows,
        "post_hoc": post_hoc,
        "limits": [
            "本代是 S2-0 的**来源核验**: 只回答 align 的来源, 不产生任何 P1a/P1b/P2/P3 判据",
            "评估网格只覆盖两个底座的前 4 层 (切片只覆盖 0-3) ⇒ S3 在更深层的应用属外推",
            "对齐只能在**样本级**拟合: 两个模型的 token 网格不同, 无法逐 token 配对",
            "本代不改动 v77/v78 的裁决; S3 仍未开跑, A1 整体仍未判定",
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
    print("      chosen_source: %s" % res["chosen_source"], flush=True)
    if not args.no_write:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\r\n") as f:
            json.dump(res, f, ensure_ascii=False, indent=2, default=str)
        print("      wrote %s" % out, flush=True)
    return 0 if res["verdict"] == "ALIGN_SOURCE_DECIDED" else 1


if __name__ == "__main__":
    sys.exit(main())
