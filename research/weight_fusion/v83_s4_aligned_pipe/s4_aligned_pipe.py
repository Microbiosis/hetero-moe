"""s4_aligned_pipe.py — v83 S4-AP: Q-K 接收场正交对齐 + 源侧同层正交解耦管道.

上游: v82_s4_pipe (S4-P: 裁决 S4PIPE_FAIL —— P5b/P5c/P3① 通过, P5a/P5d/P2′ 未过).

本代针对 v82 查清的三处算法缺陷做预登记修复 (全部在训练集 train_idx 内闭式求解, 零测试集泄漏):
  1. 载体正交对齐 (治 P5a 管道不连通):
     v82 的 Δ_B = (a_s @ P_Aᵀ) @ P_B 使用未对齐的独立随机正交矩阵 P_A, P_B, 导致注入方向与
     目标模型第 0 层注意力 Query-Key 接收锥正交 (P5a 失败: arm1−arm3 = −0.000983, p = 0.9053)。
     本代用纯前向闭式算子计算目标模型第 0 层注意力对管道位的 Q-K 接收场方向 g_0 (含 Qwen RoPE
     正向/逆向旋转 + GQA 14->2 组平均 + RMSNorm, 及 BERT MHA 拉回), 在训练集 train_idx (前 64 条)
     上求解正交 Procrustes 矩阵 Ω_s = argmax_{ΩᵀΩ=I} Tr(Ωᵀ Z_{s,tr}ᵀ Y_{0,tr}) in O(256)。
     由于 Ω_s 严格正交, 载体 z -> z @ Ω_s @ P_B 仍是严格等距映射 (范数与成对夹角 100% 保持)。
  2. 真 O 槽输出 + 同层正交解耦 (治 P5d 非目标槽普涨 & Qwen->BERT 槽混淆):
     v77-v82 的 SLOT_SOURCE["O"] = "in" 抓取的是 o_proj 的输入而非输出, 且样本级 mean-pool 使
     同层 Q/O 共享句子主成分 (R²(a_Q->a_O) ≈ 0.45), 导致注入 Q 必然抬高 O (P5d 失败)。
     本代改为抓取 o_proj 真输出 (SLOT_SOURCE_V83["O"] = "out"), 并在训练集 train_idx 上对同层
     (a_Q, a_O) 做闭式正交残差化 (a_{Q⊥O}, a_{O⊥Q}), 剥离同层共享句子成分。
  3. 判据与统计口径与 v82 完全一致 (P5a / P5b / P5c / P5d / P2′ / P3①), 阈值一条不放宽。

用法:
  python s4_aligned_pipe.py --calib      # S4AP-1: lambda 标定 (写死候选集 + 写死选择规则)
  python s4_aligned_pipe.py              # 正式跑 (5 种子 x 64 块 x 4 臂 = 1280 行)
  python s4_aligned_pipe.py --smoke      # 减网格冒烟 (不产出裁决)
  python s4_aligned_pipe.py --aggregate  # 从 checkpoint 重算判据并写 metrics.json
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
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch

# ================================================================ 常量 (跑前写死)
DS = 256
SEQ = 128
K_PIPE = 8
N_CHANNELS = 2
CH_SIZE = K_PIPE // N_CHANNELS      # 4: 每通道 4 个管道位
TXT = SEQ - K_PIPE                   # 120: 正文长度
N_TEXT = 128
N_VIT = 64
BATCH = 32

L_Q = (0, 1, 2, 3, 6, 12, 18, 23)
L_B = (0, 1, 2, 3)
L_V = (0, 3, 6, 9, 11)

INJECT_LAYER = 0
ATT_LAYER = 0

SOURCE_SLOTS = ("Q", "O")
#: 修 v77-v82 将 O 槽取为 o_proj 输入 ("in") 的缺陷: Q 与 O 均取模块真实输出 ("out")
SLOT_SOURCE_V83 = {"Q": "out", "O": "out"}
ORTHO_RIDGE_LAM = 10.0

ARMS = ("arm0_pipe_zero", "arm1_slot_Q", "arm2_wrong_slot_O", "arm3_random_dir")
INJECT_ARMS = ("arm1_slot_Q", "arm2_wrong_slot_O", "arm3_random_dir")
ARM_SLOT = {"arm1_slot_Q": "Q", "arm2_wrong_slot_O": "O"}
ARM_CHANNEL = {"arm1_slot_Q": 0, "arm2_wrong_slot_O": 1, "arm3_random_dir": 0}

DIRECTIONS = (
    ("Qwen->BERT", "Qwen", "BERT", L_Q, L_B),
    ("BERT->Qwen", "BERT", "Qwen", L_B, L_Q),
)

WITHDRAWN_DIRECTIONS = (
    ("ViT->ViT", "ViT", "ViT", L_V, L_V),
)

SEEDS = (0, 1, 2, 3, 4)
N_PERM = 10000
ALPHA = 0.01
LAMBDA_GRID = (0.25, 0.5, 1.0, 2.0)
CALIB_MAX_FINAL = 0.3
CALIB_MIN_GROUPS = 8
CALIB_N_GROUPS = 12
PROBE_LAM_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
PROBE_CV_FOLDS = 5
PROBE_PILOT_BLOCKS = 3
TEXT_SPLIT_TRAIN = 64
VIT_SPLIT_TRAIN = 32
DROP_FIRST_TOKEN = True
RAND_SEED_BASE = 20261118

THRESHOLDS = {
    "n_text_blocks": 64,
    "n_vit_blocks": 0,
    "n_attention_blocks": 12,
    "alpha": ALPHA,
    "n_perm": N_PERM,
    "n_seeds": len(SEEDS),
    "p3_min_rel_perturbation": 1e-3,
    "p3_max_rel_perturbation": 0.5,
    "p5d_selectivity_p_min": 0.05,
    "p2_require_same_ge_cross": True,
    "calib_max_final": CALIB_MAX_FINAL,
    "lambda_grid": list(LAMBDA_GRID),
    "ortho_ridge_lam": ORTHO_RIDGE_LAM,
    "slot_source_o": SLOT_SOURCE_V83["O"],
    "carrier_alignment": "train_split_orthogonal_procrustes_to_layer0_qk_receptive_field",
    "k_pipe": K_PIPE,
    "n_channels": N_CHANNELS,
    "inject_layer": INJECT_LAYER,
    "att_layer": ATT_LAYER,
    "seq": SEQ,
    "txt": TXT,
}


def _load_by_path(path: Path, name: str):
    spec = ilu.spec_from_file_location(name, str(path))
    mod = ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_PKG = Path(__file__).resolve().parent
_S4 = _load_by_path(_PKG.parent / "v82_s4_pipe" / "s4_pipe.py", "s4_pipe_v82_by_path")
_S3 = _S4._S3
_V77 = _S4._V77

MEAS = _S4.MEAS
Backbone = _S4.Backbone
fetch_models = _S4.fetch_models
fetch_texts = _S4.fetch_texts
fetch_vit_images = _S4.fetch_vit_images
make_batches = _S4.make_batches
perm_paired = _S4.perm_paired
perm_group_stratified = _S4.perm_group_stratified
sign_test = _S4.sign_test
sha256_of = _S4.sha256_of
CACHE_DIR = _S4.CACHE_DIR

load_target = _S4.load_target
build_layout = _S4.build_layout
query_hi = _S4.query_hi
capture_layer_input = _S4.capture_layer_input
body_norms = _S4.body_norms
install_pipe = _S4.install_pipe
make_pipe_delta = _S4.make_pipe_delta
rel_final_body = _S4.rel_final_body
split_idx = _S4.split_idx
att_record = _S4.att_record
evaluate_criteria = _S4.evaluate_criteria

PROVENANCE_FILES = {
    "v62_measure": _V77.PROVENANCE_FILES["v62_measure"],
    "v75_g2_conjugation": str(_PKG.parent / "v75_weight_conjugation" / "g2_conjugation.py"),
    "v77_true_model": str(_PKG.parent / "v77_true_model_flow" / "true_model.py"),
    "v81_s3_flow": str(_PKG.parent / "v81_s3_semantic" / "s3_flow.py"),
    "v82_s4_pipe": str(_PKG.parent / "v82_s4_pipe" / "s4_pipe.py"),
    "v83_s4_aligned_pipe": str(_PKG / "s4_aligned_pipe.py"),
}


def _seed_of(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


# ================================================================ 机制 1: 真 O 槽输出 + 训练集内同层正交解耦
@torch.no_grad()
def capture_slots_true_out(bb: Backbone, batch: dict,
                           layer_slots: Sequence[Tuple[int, str]]) -> Dict[Tuple[int, str], torch.Tensor]:
    """抓真模型里若干 (层, 槽) 的真实模块**输出** (修 v77-v82 对 O 槽取输入而非输出的缺陷)。"""
    store: Dict[Tuple[int, str], torch.Tensor] = {}
    handles = []
    for ly, slot in layer_slots:
        mod, _attr = _V77.slot_module(bb.layers[ly], bb.name, slot)
        key = (ly, slot)
        def h(module, inp, out, _k=key):
            o = out[0] if isinstance(out, tuple) else out
            store[_k] = o.detach()
        handles.append(mod.register_forward_hook(h))
    try:
        bb.model(**batch)
    finally:
        for hnd in handles:
            hnd.remove()
    return store


@torch.no_grad()
def orthogonalize_slot_pair(a_q: torch.Tensor, a_o: torch.Tensor,
                            train_idx: np.ndarray,
                            lam: float = ORTHO_RIDGE_LAM) -> Tuple[torch.Tensor, torch.Tensor]:
    """在**训练集 train_idx** 上做同层槽正交解耦 (Gram-Schmidt / Ridge 残差化, 测试集全程不参与拟合):

        a_{Q⊥O} = (a_Q - μ_{Q,tr}) - (a_O - μ_{O,tr}) W_{O->Q, tr}
        a_{O⊥Q} = (a_O - μ_{O,tr}) - (a_Q - μ_{Q,tr}) W_{Q->O, tr}

    作用: 剥离同层 Q 槽与 O 槽共享的句子级词袋主成分, 提取纯槽特异激活分量。
    """
    tr = torch.as_tensor(train_idx, dtype=torch.long)
    mu_q = a_q[tr].mean(0, keepdim=True)
    mu_o = a_o[tr].mean(0, keepdim=True)
    qc = a_q - mu_q
    oc = a_o - mu_o
    d = a_q.shape[1]
    eye = torch.eye(d, dtype=a_q.dtype)

    G_o = oc[tr].t() @ oc[tr] + float(lam) * eye
    W_o2q = torch.linalg.solve(G_o, oc[tr].t() @ qc[tr])
    q_perp = qc - oc @ W_o2q

    G_q = qc[tr].t() @ qc[tr] + float(lam) * eye
    W_q2o = torch.linalg.solve(G_q, qc[tr].t() @ oc[tr])
    o_perp = oc - qc @ W_q2o
    return q_perp, o_perp


def capture_source_orthogonal(name: str, layers: Sequence[int], texts, imgs,
                              train_idx: np.ndarray) -> Dict[Tuple[int, str], torch.Tensor]:
    """抓源模型 (层, 槽) 的真模块输出 mean-pool 激活, 并在 train_idx 上做同层 Q/O 正交解耦。"""
    bb = Backbone(name)
    batches = make_batches(bb, name, texts, imgs)
    want = [(ly, slot) for ly in layers for slot in SOURCE_SLOTS]
    acc: Dict[Tuple[int, str], List[torch.Tensor]] = {k: [] for k in want}
    for b in batches:
        st = capture_slots_true_out(bb, b, want)
        for k in want:
            acc[k].append(st[k].mean(dim=1).float())
    raw = {k: torch.cat(v, 0) for k, v in acc.items()}
    del bb, batches, acc
    gc.collect()

    out: Dict[Tuple[int, str], torch.Tensor] = {}
    for ly in layers:
        qp, op = orthogonalize_slot_pair(raw[(ly, "Q")], raw[(ly, "O")], train_idx, ORTHO_RIDGE_LAM)
        out[(ly, "Q")] = qp
        out[(ly, "O")] = op
    return out


# ================================================================ 机制 2: 闭式 Q-K 接收场 + 训练集正交 Procrustes
@torch.no_grad()
def target_qk_receptive_field(bb: Backbone, h0: torch.Tensor, layout: dict) -> torch.Tensor:
    """纯前向闭式计算目标模型第 0 层注意力对管道位的 Q-K 接收场方向 [N, D_B] (无任何反向传播):

    - BERT (绝对位置编码, MHA):
        q_bar_i = mean_{t in body_i} (h0_{i,t} W_Q^T + b_Q) -> 拉回 K 空间: g_{0,i} = q_bar_i W_K
    - Qwen (RoPE + GQA 14 Q 头 / 2 KV 头 + RMSNorm):
        1) normed = input_layernorm(h0)
        2) q = normed W_Q^T + b_Q, 施加第 0 层真实 rotary_emb 的正向旋转 RoPE(t)
        3) 在正文有效位置求均值后, 施加管道位 p=0..K-1 的逆旋转 RoPE(-p)
        4) 按 GQA 组 (每 7 个 Q 头归 1 个 KV 头) 求均值, 经 W_K 与 RMSNorm 权重拉回隐空间 [N, 896]。
    """
    layer0 = bb.layers[0]
    n, seq, d = h0.shape
    out = torch.zeros(n, d, dtype=torch.float32)

    if bb.name == "BERT":
        sa = layer0.attention.self
        wq = sa.query.weight.detach().float()
        bq = sa.query.bias.detach().float() if sa.query.bias is not None else 0.0
        wk = sa.key.weight.detach().float()
        for i in range(n):
            lo = layout["body_start"]
            hi = query_hi(layout, i)
            hb = h0[i, lo:hi].float()
            q_bar = (hb @ wq.t() + bq).mean(dim=0)
            out[i] = q_bar @ wk
        return out

    ln = layer0.input_layernorm
    sa = layer0.self_attn
    wq = sa.q_proj.weight.detach().float()
    bq = sa.q_proj.bias.detach().float() if sa.q_proj.bias is not None else 0.0
    wk = sa.k_proj.weight.detach().float()
    ln_w = ln.weight.detach().float()

    n_heads = int(bb.cfg.num_attention_heads)
    n_kv = int(bb.cfg.num_key_value_heads)
    head_dim = d // n_heads
    group = n_heads // n_kv

    normed = ln(h0).float()
    q_all = (normed @ wq.t() + bq).view(n, seq, n_heads, head_dim).transpose(1, 2)

    pos_ids = torch.arange(seq, device=h0.device).unsqueeze(0)
    rotary = getattr(bb.model, "rotary_emb", getattr(sa, "rotary_emb", None))
    cos, sin = rotary(normed, pos_ids)
    cos = cos.float().unsqueeze(1)
    sin = sin.float().unsqueeze(1)

    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : head_dim // 2]
        x2 = x[..., head_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_rot = q_all * cos + _rotate_half(q_all) * sin
    cos_p = cos[:, :, :K_PIPE, :].mean(dim=2)
    sin_p = sin[:, :, :K_PIPE, :].mean(dim=2)

    for i in range(n):
        lo = layout["body_start"]
        hi = query_hi(layout, i)
        qb = q_rot[i : i + 1, :, lo:hi, :].mean(dim=2)
        k_head_q = qb * cos_p - _rotate_half(qb) * sin_p
        k_kv = k_head_q.view(n_kv, group, head_dim).mean(dim=1).reshape(-1)
        out[i] = (k_kv @ wk) * ln_w
    return out


@torch.no_grad()
def fit_orthogonal_procrustes(Z_tr: torch.Tensor, Y_tr: torch.Tensor) -> torch.Tensor:
    """在训练集 train_idx 上求解正交 Procrustes: Ω = argmax_{ΩᵀΩ=I} Tr(Ωᵀ Z_trᵀ Y_tr) = U Vᵀ."""
    M = Z_tr.float().t() @ Y_tr.float()
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    return (U @ Vh).to(Z_tr.dtype)


# ================================================================ 前向与快速等价探针
def run_forward(bb, layout: dict, delta_pipe, want_att: bool,
                read_layers: Sequence[int], att_layers: Sequence[int]):
    """按 BATCH=32 跑完全部批次, 返回 (reads, fins, attn)。"""
    n = layout["n"]
    reads = {j: [] for j in read_layers}
    fins = []
    attn = {j: ([], []) for j in att_layers}
    for s in range(0, n, BATCH):
        e = min(s + BATCH, n)
        hnd = None
        if delta_pipe is not None:
            hnd = install_pipe(bb.layers[INJECT_LAYER], delta_pipe[s:e])
        try:
            with torch.no_grad():
                out = bb.model(input_ids=layout["ids"][s:e],
                               attention_mask=layout["mask"][s:e],
                               output_attentions=want_att, output_hidden_states=True)
        finally:
            if hnd is not None:
                hnd.remove()
        for j in read_layers:
            reads[j].append(out.hidden_states[j + 1][:, layout["body_start"]:].float())
        fins.append(out.hidden_states[-1].float())
        if want_att:
            for j in att_layers:
                a = out.attentions[j]
                if a is None:
                    raise RuntimeError("attentions[%d] 为 None —— 模型没走 eager 实现" % j)
                for i in range(e - s):
                    lo = layout["body_start"]
                    hi = query_hi(layout, s + i)
                    attn[j][0].append(float(a[i, :, lo:hi, 0:CH_SIZE].mean()))
                    attn[j][1].append(float(a[i, :, lo:hi, CH_SIZE:K_PIPE].mean()))
    reads = {j: torch.cat(v, 0) for j, v in reads.items()}
    fins = torch.cat(fins, 0)
    attn = {j: (torch.tensor(v[0]), torch.tensor(v[1])) for j, v in attn.items()}
    return reads, fins, attn


def _gather_rows(readout: torch.Tensor, mask: torch.Tensor, idx: np.ndarray):
    r = readout[idx]
    m = mask[idx].bool()
    return r[m], m.sum(1).long()


def precompute_masked_probe(readout: torch.Tensor, mask: torch.Tensor,
                            lam: float, train_idx: np.ndarray, test_idx: np.ndarray) -> dict:
    """预计算读出矩阵的 Gram 求解器与样本内 token 聚合算子 (与 v82 masked_r2_multi 逐位等价, 提速 50 倍)。"""
    D = readout.shape[-1]
    Xtr, ctr = _gather_rows(readout, mask, train_idx)
    Xte, cte = _gather_rows(readout, mask, test_idx)
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    G = Xtr.t() @ Xtr + float(lam) * torch.eye(D, dtype=Xtr.dtype)
    seg_tr = torch.repeat_interleave(torch.arange(len(ctr)), ctr)
    Xtr_sum = torch.zeros(len(ctr), D, dtype=Xtr.dtype)
    Xtr_sum.index_add_(0, seg_tr, Xtr)
    K_mat = torch.linalg.solve(G, Xtr_sum.t())
    P_te = Xte @ K_mat
    P_te_G = P_te.t() @ P_te
    seg_te = torch.repeat_interleave(torch.arange(len(cte)), cte)
    P_te_sum = torch.zeros(len(cte), len(ctr), dtype=Xtr.dtype)
    P_te_sum.index_add_(0, seg_te, P_te)
    cf_tr = ctr.to(Xtr.dtype).unsqueeze(1)
    cf_te = cte.to(Xtr.dtype).unsqueeze(1)
    return {
        "P_te_G": P_te_G,
        "P_te_sum": P_te_sum,
        "cf_tr": cf_tr,
        "cf_te": cf_te,
        "n_tok_tr": float(ctr.sum()),
        "train_idx": train_idx,
        "test_idx": test_idx,
    }


def eval_precomputed_probe(cache: dict, targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
    P_te_G = cache["P_te_G"]
    P_te_sum = cache["P_te_sum"]
    cf_tr = cache["cf_tr"]
    cf_te = cache["cf_te"]
    n_tok_tr = cache["n_tok_tr"]
    train_idx = cache["train_idx"]
    test_idx = cache["test_idx"]
    out: Dict[str, float] = {}
    for name, tgt in targets.items():
        yt_s = tgt[train_idx]
        ye_s = tgt[test_idx]
        ym = (yt_s * cf_tr).sum(0, keepdim=True) / n_tok_tr
        Yc = yt_s - ym
        Ytc = ye_s - ym
        term1 = ((P_te_G @ Yc) * Yc).sum()
        term2 = ((P_te_sum @ Yc) * Ytc).sum() * 2.0
        term3 = (cf_te * Ytc.pow(2)).sum()
        ss_res = float(term1 - term2 + term3)
        ss_tot = float(term3)
        out[name] = 1.0 - ss_res / (ss_tot + 1e-12)
    return out


def masked_r2_multi(readout: torch.Tensor, mask: torch.Tensor,
                    targets: Dict[str, torch.Tensor], lam: float,
                    train_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, float]:
    """token 级 masked ridge 探针 (只读正文真实 token, 留出集 R²)。"""
    cache = precompute_masked_probe(readout, mask, lam, train_idx, test_idx)
    return eval_precomputed_probe(cache, targets)


def masked_lambda_scores(readout: torch.Tensor, mask: torch.Tensor,
                          tgt: torch.Tensor, train_idx: np.ndarray) -> Dict[float, float]:
    """训练集内 5 折 CV 选 ridge lambda (测试集全程不参与)。"""
    folds = np.arange(len(train_idx)) % PROBE_CV_FOLDS
    scores: Dict[float, List[float]] = {float(g): [] for g in PROBE_LAM_GRID}
    for f in range(PROBE_CV_FOLDS):
        tr_pos = np.where(folds != f)[0]
        te_pos = np.where(folds == f)[0]
        if tr_pos.size == 0 or te_pos.size == 0:
            continue
        sub_tr = train_idx[tr_pos]
        sub_te = train_idx[te_pos]
        for g in PROBE_LAM_GRID:
            r2 = masked_r2_multi(readout, mask, {"t": tgt}, float(g), sub_tr, sub_te)["t"]
            scores[float(g)].append(r2)
    return {g: (float(np.mean(v)) if v else float("nan")) for g, v in scores.items()}


def pooled_r2_multi(readout: torch.Tensor, mask: torch.Tensor,
                    targets: Dict[str, torch.Tensor], lam: float,
                    train_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, float]:
    return _S4.pooled_r2_multi(readout, mask, targets, lam, train_idx, test_idx)


def pilot_probe_lambda(bb, layout: dict, blocks, s_name: str, t_name: str,
                       src_cache, proj, y0_shared: torch.Tensor, train_idx: np.ndarray,
                       lam_pipe: float, bn: torch.Tensor, dtype) -> dict:
    """预登记 pilot 块 (seed 0) + 训练样本内 5 折 CV 选 token 级探针的 ridge lambda。"""
    p_a, p_b = proj[s_name], proj[t_name]
    scores: Dict[float, List[float]] = {float(g): [] for g in PROBE_LAM_GRID}
    for ly, j in blocks:
        zq_raw = src_cache[s_name][(ly, "Q")] @ p_a.t()
        omega_q = fit_orthogonal_procrustes(zq_raw[train_idx], y0_shared[train_idx])
        zq = zq_raw @ omega_q
        delta = make_pipe_delta(zq, lam_pipe, bn, p_b, ARM_CHANNEL["arm1_slot_Q"],
                                layout["n_seq"], dtype)
        reads, _, _ = run_forward(bb, layout, delta, False, [j], [])
        sc = masked_lambda_scores(reads[j], layout["body_mask"],
                                  src_cache[s_name][(ly, "Q")], train_idx)
        for g, v in sc.items():
            if not np.isnan(v):
                scores[g].append(v)
        del reads, delta
    best_l, best_s = float(PROBE_LAM_GRID[0]), -1e30
    for g, vs in scores.items():
        if not vs:
            continue
        m = float(np.mean(vs))
        if m > best_s:
            best_l, best_s = g, m
    return {"probe_lambda": best_l, "cv_r2": best_s,
            "cv_scores": {("%.4g" % g): float(np.mean(v)) for g, v in sorted(scores.items()) if v}}


# ================================================================ 标定与正式跑批
def evaluate_calibration(cand: Dict[str, dict], arm0: Dict[str, dict]):
    """按写死的三条规则评估每个候选 λ (纯函数, 可用合成数据单测)。"""
    results = {}
    feasible_lams = []
    for lam in LAMBDA_GRID:
        key = "%.4g" % lam
        blocks = []
        max_final = 0.0
        for dname, s_name, t_name, s_layers, t_layers in DIRECTIONS:
            a0 = arm0[dname]
            for ly in s_layers:
                a1 = cand[key][dname][str(ly)]["arm1_slot_Q"]
                a2 = cand[key][dname][str(ly)]["arm2_wrong_slot_O"]
                c2 = (a1["attn_pipe_mean"] - a0["attn_pipe_mean"]) > 0
                c3 = (a1["flip_mean"] - a2["flip_mean"]) > 0
                max_final = max(max_final, a1["rel"], a2["rel"])
                blocks.append({
                    "direction": dname, "source_layer": int(ly),
                    "attn_pipe_arm1": a1["attn_pipe_mean"],
                    "attn_pipe_arm0": a0["attn_pipe_mean"],
                    "delta_attn_pipe": a1["attn_pipe_mean"] - a0["attn_pipe_mean"],
                    "flip_arm1": a1["flip_mean"], "flip_arm2": a2["flip_mean"],
                    "delta_flip": a1["flip_mean"] - a2["flip_mean"],
                    "rel_final_arm1": a1["rel"], "rel_final_arm2": a2["rel"],
                    "gating": bool(dname != "ViT->ViT"),
                    "cond2_pipe_alive": bool(c2), "cond3_addressed": bool(c3)})
        cond1 = max_final <= CALIB_MAX_FINAL
        gate = [b for b in blocks if b["gating"]]
        n2 = sum(1 for b in gate if b["cond2_pipe_alive"])
        n3 = sum(1 for b in gate if b["cond3_addressed"])
        ok = bool(cond1 and len(gate) == CALIB_N_GROUPS
                  and n2 >= CALIB_MIN_GROUPS and n3 >= CALIB_MIN_GROUPS)
        if ok:
            feasible_lams.append(float(lam))
        results[key] = {
            "lambda": float(lam), "feasible": ok,
            "cond1_max_final_ok": bool(cond1),
            "cond2_all_pipe_alive": bool(n2 >= CALIB_MIN_GROUPS),
            "cond3_all_addressed": bool(n3 >= CALIB_MIN_GROUPS),
            "n_gating_blocks": len(gate),
            "min_required_groups": CALIB_MIN_GROUPS,
            "n_cond2_pass": n2, "n_cond3_pass": n3,
            "max_final": max_final,
            "blocks": blocks}
    selected = max(feasible_lams) if feasible_lams else None
    return results, feasible_lams


def run_calib() -> dict:
    """S4AP-1: 在 12 个文本 (方向, 源层) 组上标定注入强度 λ (seed 0)。"""
    t0 = time.time()
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(min(16, os.cpu_count() or 8))

    fetch_models()
    texts = fetch_texts(N_TEXT)
    train_idx, _ = split_idx(False, None)
    proj = MEAS.projections(0)

    src_cache: Dict[str, Dict[Tuple[int, str], torch.Tensor]] = {}
    for name, layers in (("Qwen", L_Q), ("BERT", L_B)):
        src_cache[name] = capture_source_orthogonal(name, layers, texts, None, train_idx)

    res_dir = _PKG / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    rec_path = res_dir / "calib_records.jsonl"
    rec_f = rec_path.open("w", encoding="utf-8")

    cand: Dict[str, dict] = {"%.4g" % lam: {} for lam in LAMBDA_GRID}
    arm0: Dict[str, dict] = {}

    for dname, s_name, t_name, s_layers, _t_layers in DIRECTIONS:
        bb = load_target(t_name, eager=True)
        layout = build_layout(bb, False, texts, None)
        h0 = capture_layer_input(bb, layout, INJECT_LAYER)
        bn = body_norms(h0, layout)
        g0 = target_qk_receptive_field(bb, h0, layout)
        dtype = next(bb.model.parameters()).dtype
        p_a, p_b = proj[s_name], proj[t_name]
        y0_shared = g0 @ p_b.t()

        _, fins0, attn0 = run_forward(bb, layout, None, True, [], [ATT_LAYER])
        c0_0, c1_0 = attn0[ATT_LAYER]
        pipe0 = float((c0_0 + c1_0).mean())
        flip0 = float((c0_0 - c1_0).mean())
        arm0[dname] = {"attn_pipe_mean": pipe0, "flip_mean": flip0}

        for lam in LAMBDA_GRID:
            key = "%.4g" % lam
            cand[key][dname] = {}
            for ly in s_layers:
                zq_raw = src_cache[s_name][(ly, "Q")] @ p_a.t()
                zo_raw = src_cache[s_name][(ly, "O")] @ p_a.t()
                omega_q = fit_orthogonal_procrustes(zq_raw[train_idx], y0_shared[train_idx])
                omega_o = fit_orthogonal_procrustes(zo_raw[train_idx], y0_shared[train_idx])
                zq = zq_raw @ omega_q
                zo = zo_raw @ omega_o
                cand[key][dname][str(ly)] = {}
                for arm, zz in (("arm1_slot_Q", zq), ("arm2_wrong_slot_O", zo)):
                    delta = make_pipe_delta(zz, float(lam), bn, p_b, ARM_CHANNEL[arm],
                                            layout["n_seq"], dtype)
                    _, fins, attn = run_forward(bb, layout, delta, True, [], [ATT_LAYER])
                    rel = rel_final_body(fins, fins0, layout)
                    c0, c1 = attn[ATT_LAYER]
                    entry = {
                        "attn_pipe_mean": float((c0 + c1).mean()),
                        "flip_mean": float((c0 - c1).mean()),
                        "rel": float(rel),
                    }
                    cand[key][dname][str(ly)][arm] = entry
                    rec_f.write(json.dumps({
                        "lambda": float(lam), "direction": dname,
                        "source_layer": int(ly), "arm": arm, **entry
                    }, ensure_ascii=False) + "\n")
                    rec_f.flush()
        del bb, layout, h0, fins0
        gc.collect()

    rec_f.close()
    results, feasible = evaluate_calibration(cand, arm0)
    selected = max(feasible) if feasible else None
    verdict = "S4AP_LAMBDA_SELECTED" if selected is not None else "CALIB_NO_FEASIBLE_LAMBDA"
    return {
        "verdict": verdict,
        "selected_lambda": selected,
        "feasible_lambdas": feasible,
        "lambda_grid": list(LAMBDA_GRID),
        "calib_max_final": CALIB_MAX_FINAL,
        "min_required_groups": CALIB_MIN_GROUPS,
        "n_gating_groups": CALIB_N_GROUPS,
        "rule": (
            "候选 λ ∈ {0.25, 0.5, 1.0, 2.0}; 同时满足 ① 全部块 max final 相对扰动 ≤ 0.3 "
            "② 主判据集 (12 个文本组) 里 ≥ 8/12 组 attn_pipe(arm1) − attn_pipe(arm0) > 0 "
            "③ 主判据集里 ≥ 8/12 组 flip(arm1) − flip(arm2) > 0 的候选里取**最大**的那个; "
            "无一满足 ⇒ CALIB_NO_FEASIBLE_LAMBDA (报回, 不硬选)"
        ),
        "arm0_baseline": arm0,
        "candidates": results,
        "runtime_sec": round(time.time() - t0, 1),
    }


def _protocol_texts() -> dict:
    return {
        "scope": (
            "S4-AP (Q-K 接收场正交对齐 + 源侧同层正交解耦管道): P5a / P5b / P5c / P5d / P2_prime / P3_1; "
            "本代不训练任何底座权重, 不产生任何能力指标, 不产生 A1 的总判定。"
        ),
        "discipline": [
            "阈值全部写进源码常量 (THRESHOLDS / SEEDS / N_PERM / ALPHA / LAMBDA_GRID), 跑前写死, 事后不得放宽",
            "正交 Procrustes 矩阵 Ω_s 与同层 Q/O 正交残差化严格在训练样本 train_idx (前 64 条) 内闭式求解, 测试样本全程不参与",
            "Ω_s 满足 Ω_sᵀ Ω_s = I_256, 载体仍为严格线性等距映射 (范数与成对夹角 100% 保持)",
            "判据一律给 块数 / 正号数 / 中位增量 / p, 蒙特卡洛置换 10^4 次",
            "探针为 masked token 级 ridge (只读正文有效 token, 去首 token), R² 一律在留出集上算",
            "本代不训练任何底座权重, 不产生任何能力指标",
        ],
        "limits": [
            "管道内容写在第 0 层 => 单次注入经目标各层传播, 第 0 层在 Qwen 中受 GQA Value-Output 算子 M_vo (rank=128, fro=0.2862) 52 倍衰减制约",
            "正文被截断到 120 位 (SEQ-K_PIPE), 与 v82 同口径",
            "ViT 平行臂沿用 v82 跑前撤出决定, 本代只覆盖文本两个方向 (BERT<->Qwen)",
            "共享空间维度仍为 256 维 (沿用 v62/v80-v82)",
            "本代不产生 A1 的总判定",
        ],
    }


def run_formal(lam: float, smoke: bool = False, fresh: bool = False,
               only: Optional[Dict[str, set]] = None,
               shard: Optional[str] = None) -> dict:
    t0 = time.time()
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(min(16, os.cpu_count() or 8))

    fetch_models()
    texts = fetch_texts(N_TEXT)
    labels_text = np.array([int(t["label"]) for t in texts], dtype=int)
    train_idx, test_idx = split_idx(False, None)

    seeds = (0, 1) if smoke else SEEDS
    proj_cache = {s: MEAS.projections(s) for s in seeds}

    src_cache: Dict[str, Dict[Tuple[int, str], torch.Tensor]] = {}
    for name, layers in (("Qwen", L_Q), ("BERT", L_B)):
        src_cache[name] = capture_source_orthogonal(name, layers, texts, None, train_idx)
        print("[source] %s 正交解耦槽激活就绪 (%d 个 (层,槽))" % (name, len(src_cache[name])), flush=True)

    res_dir = _PKG / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    suf = ("_" + shard) if shard else ""
    b_name = "blocks_smoke.jsonl" if smoke else f"blocks{suf}.jsonl"
    a_name = "attention_smoke.jsonl" if smoke else f"attention{suf}.jsonl"
    b_path = res_dir / b_name
    a_path = res_dir / a_name
    pilot_cache_path = res_dir / "part_pilot.json"
    pilot_cache = json.loads(pilot_cache_path.read_text(encoding="utf-8")) if pilot_cache_path.exists() else {}

    done: Dict[Tuple, dict] = {}
    att_done: Dict[Tuple, dict] = {}
    if not fresh:
        if b_path.exists():
            for line in b_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                k = (r["direction"], int(r["source_layer"]), int(r["target_layer"]), r["arm"], int(r["seed"]))
                done[k] = r
        if a_path.exists():
            for line in a_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                k = (r["direction"], int(r["source_layer"]), r["arm"], int(r["seed"]))
                att_done[k] = r

    ck = b_path.open("w" if fresh else "a", encoding="utf-8")
    ack = a_path.open("w" if fresh else "a", encoding="utf-8")

    rows: List[dict] = []
    att_recs: List[dict] = []
    lam_info: Dict[str, dict] = {}
    layout_info: Dict[str, dict] = {}

    for dname, s_name, t_name, s_layers, t_layers in DIRECTIONS:
        if only is not None and dname not in only:
            continue
        s_run = s_layers[:2] if smoke else s_layers
        t_run = t_layers[:2] if smoke else t_layers
        if only is not None and only.get(dname):
            s_run = tuple(ly for ly in s_run if ly in only[dname])

        bb = load_target(t_name, eager=True)
        layout = build_layout(bb, False, texts, None)
        h0 = capture_layer_input(bb, layout, INJECT_LAYER)
        bn = body_norms(h0, layout)
        g0 = target_qk_receptive_field(bb, h0, layout)
        dtype = next(bb.model.parameters()).dtype

        layout_info[dname] = {
            "n_seq": layout["n_seq"], "k_pipe": K_PIPE, "n_body": layout["n_body"],
            "n_truncated": int((layout["take"] == TXT).sum()) if layout["take"] is not None else 0,
            "body_norm_median": float(bn.median()),
        }

        if not smoke and dname in pilot_cache.get("probe_lambda_by_direction", {}):
            lam_info[dname] = pilot_cache["probe_lambda_by_direction"][dname]
            probe_lam = float(lam_info[dname]["probe_lambda"])
            print("[pilot-cached] %s: probe_lambda=%g (CV R2=%.4f)"
                  % (dname, probe_lam, lam_info[dname]["cv_r2"]), flush=True)
        else:
            pilot_blocks = _S3._pilot_blocks(s_layers, t_layers, s_name, t_name, PROBE_PILOT_BLOCKS)
            y0_seed0 = g0 @ proj_cache[0][t_name].t()
            pl = pilot_probe_lambda(bb, layout, pilot_blocks, s_name, t_name,
                                    src_cache, proj_cache[0], y0_seed0, train_idx,
                                    lam, bn, dtype)
            probe_lam = float(pl["probe_lambda"])
            lam_info[dname] = {
                "probe_lambda": probe_lam, "cv_r2": pl["cv_r2"], "cv_scores": pl["cv_scores"],
                "grid": list(PROBE_LAM_GRID), "n_folds": PROBE_CV_FOLDS,
                "pilot_blocks": [[int(a), int(b)] for a, b in pilot_blocks], "pilot_seed": 0,
            }
            print("[pilot] %s: probe_lambda=%g (CV R2=%.4f)" % (dname, probe_lam, pl["cv_r2"]), flush=True)
            if not smoke and not shard:
                pilot_cache.setdefault("probe_lambda_by_direction", {})[dname] = lam_info[dname]
                pilot_cache.setdefault("layout_by_direction", {})[dname] = layout_info[dname]
                pilot_cache_path.write_text(json.dumps(pilot_cache, ensure_ascii=False, indent=2), encoding="utf-8")

        reads0, fins0, attn0 = run_forward(bb, layout, None, True, t_run, [ATT_LAYER])
        c0_0, c1_0 = attn0[ATT_LAYER]
        probe_cache0 = {j: precompute_masked_probe(reads0[j], layout["body_mask"],
                                                   probe_lam, train_idx, test_idx)
                        for j in t_run}

        for ly in s_run:
            for seed in seeds:
                akey = (dname, int(ly), "arm0_pipe_zero", int(seed))
                if akey not in att_done:
                    rec = att_record(dname, ly, "arm0_pipe_zero", seed, c0_0, c1_0)
                    rec["rel"] = 0.0
                    att_recs.append(rec)
                    att_done[akey] = rec
                    ack.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    ack.flush()
                else:
                    att_recs.append(att_done[akey])

        for ly in s_run:
            tgt_q = src_cache[s_name][(ly, "Q")]
            tgt_o = src_cache[s_name][(ly, "O")]
            for j in t_run:
                r2 = eval_precomputed_probe(probe_cache0[j], {"q": tgt_q, "o": tgt_o})
                r2p = pooled_r2_multi(reads0[j], layout["body_mask"],
                                      {"q": tgt_q, "o": tgt_o}, probe_lam,
                                      train_idx, test_idx)
                for seed in seeds:
                    key = (dname, int(ly), int(j), "arm0_pipe_zero", int(seed))
                    if key in done:
                        rows.append(done[key])
                        continue
                    row = {
                        "direction": dname, "source_model": s_name, "target_model": t_name,
                        "source_layer": int(ly), "target_layer": int(j),
                        "arm": "arm0_pipe_zero", "seed": int(seed),
                        "r2_q": r2["q"], "r2_o": r2["o"],
                        "r2_q_pooled_aux": r2p["q"], "r2_o_pooled_aux": r2p["o"],
                        "rel_perturb_final": 0.0, "probe_lam": float(probe_lam),
                        "inject_lambda": float(lam),
                    }
                    rows.append(row)
                    done[key] = row
                    ck.write(json.dumps(row, ensure_ascii=False) + "\n")
                    ck.flush()

        for seed in seeds:
            p_a = proj_cache[seed][s_name]
            p_b = proj_cache[seed][t_name]
            y0_shared = g0 @ p_b.t()
            for ly in s_run:
                zq_raw = src_cache[s_name][(ly, "Q")] @ p_a.t()
                zo_raw = src_cache[s_name][(ly, "O")] @ p_a.t()
                omega_q = fit_orthogonal_procrustes(zq_raw[train_idx], y0_shared[train_idx])
                omega_o = fit_orthogonal_procrustes(zo_raw[train_idx], y0_shared[train_idx])
                zq = zq_raw @ omega_q
                zo = zo_raw @ omega_o

                g = torch.Generator().manual_seed(_seed_of("arm3", seed, dname, ly))
                u = torch.randn(zq.shape, generator=g)
                u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)

                tgt_q = src_cache[s_name][(ly, "Q")]
                tgt_o = src_cache[s_name][(ly, "O")]

                for arm, zz in (("arm1_slot_Q", zq),
                                ("arm2_wrong_slot_O", zo),
                                ("arm3_random_dir", u)):
                    akey = (dname, int(ly), arm, int(seed))
                    if akey in att_done and all(
                        (dname, int(ly), int(j), arm, int(seed)) in done for j in t_run
                    ):
                        att_recs.append(att_done[akey])
                        for j in t_run:
                            rows.append(done[(dname, int(ly), int(j), arm, int(seed))])
                        continue

                    delta = make_pipe_delta(zz, lam, bn, p_b, ARM_CHANNEL[arm],
                                            layout["n_seq"], dtype)
                    reads, fins, attn = run_forward(bb, layout, delta, True,
                                                    t_run, [ATT_LAYER])
                    rel = rel_final_body(fins, fins0, layout)
                    rec = att_record(dname, ly, arm, seed,
                                     attn[ATT_LAYER][0], attn[ATT_LAYER][1])
                    rec["rel"] = rel
                    att_recs.append(rec)
                    att_done[akey] = rec
                    ack.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    ack.flush()

                    for j in t_run:
                        pc = precompute_masked_probe(reads[j], layout["body_mask"],
                                                     probe_lam, train_idx, test_idx)
                        r2 = eval_precomputed_probe(pc, {"q": tgt_q, "o": tgt_o})
                        r2p = pooled_r2_multi(reads[j], layout["body_mask"],
                                              {"q": tgt_q, "o": tgt_o}, probe_lam,
                                              train_idx, test_idx)
                        key = (dname, int(ly), int(j), arm, int(seed))
                        if key in done:
                            rows.append(done[key])
                            continue
                        row = {
                            "direction": dname, "source_model": s_name, "target_model": t_name,
                            "source_layer": int(ly), "target_layer": int(j),
                            "arm": arm, "seed": int(seed),
                            "r2_q": r2["q"], "r2_o": r2["o"],
                            "r2_q_pooled_aux": r2p["q"], "r2_o_pooled_aux": r2p["o"],
                            "rel_perturb_final": rel, "probe_lam": float(probe_lam),
                            "inject_lambda": float(lam),
                        }
                        rows.append(row)
                        done[key] = row
                        ck.write(json.dumps(row, ensure_ascii=False) + "\n")
                        ck.flush()
                    del delta, reads, fins
                print("[block] %s ly=%d seed=%d 完成 (累计 %d 行, %.0fs)"
                      % (dname, ly, seed, len(rows), time.time() - t0), flush=True)
        del bb, layout, h0, reads0, fins0
        gc.collect()

    ck.close()
    ack.close()
    crit = evaluate_criteria(rows, att_recs)
    # 将 verdict 前缀对齐本代 S4AP_READY / S4AP_FAIL (同时保留 passed 与 failed_criteria)
    passed = bool(crit["verdict"]["passed"])
    crit["verdict"]["verdict"] = "S4AP_READY" if passed else "S4AP_FAIL"

    prov = {k: (sha256_of(v) if os.path.exists(v) else None)
            for k, v in PROVENANCE_FILES.items()}
    return {
        "verdict": crit["verdict"]["verdict"],
        "verdict_detail": crit["verdict"],
        "criteria": {k: v for k, v in crit.items() if k != "verdict"},
        "protocol": {
            "thresholds": THRESHOLDS,
            "constants": {
                "DS": DS, "SEQ": SEQ, "K_PIPE": K_PIPE, "TXT": TXT,
                "N_CHANNELS": N_CHANNELS, "CH_SIZE": CH_SIZE,
                "N_TEXT": N_TEXT, "N_VIT": N_VIT, "BATCH": BATCH,
                "L_Q": list(L_Q), "L_B": list(L_B), "L_V": list(L_V),
                "INJECT_LAYER": INJECT_LAYER, "ATT_LAYER": ATT_LAYER,
                "SEEDS": list(seeds), "N_PERM": N_PERM, "ALPHA": ALPHA,
                "LAMBDA_INJECT": float(lam),
                "ORTHO_RIDGE_LAM": ORTHO_RIDGE_LAM,
                "PROBE_LAM_GRID": list(PROBE_LAM_GRID),
                "PROBE_CV_FOLDS": PROBE_CV_FOLDS,
                "TEXT_SPLIT_TRAIN": TEXT_SPLIT_TRAIN, "VIT_SPLIT_TRAIN": VIT_SPLIT_TRAIN,
                "DROP_FIRST_TOKEN": DROP_FIRST_TOKEN,
            },
            "probe_lambda_by_direction": lam_info,
            "layout_by_direction": layout_info,
            "corpus": {
                "text": "ag_news train 128 条 (缓存 agnews_128.json)",
                "text_label_dist": np.bincount(labels_text, minlength=4).tolist(),
            },
            "provenance_sha256": prov,
            "seed": 0, "device": "cpu", "dtype": "float32",
            "smoke": bool(smoke),
            "batch_selection": ({k: sorted(v) for k, v in only.items()} if only else "all"),
            **_protocol_texts(),
        },
        "blocks": rows,
        "attention_summary": [
            {k: v for k, v in r.items() if not k.startswith("per_sample")}
            for r in att_recs
        ],
        "runtime_sec": round(time.time() - t0, 1),
    }


def aggregate() -> dict:
    """从 results/blocks.jsonl + results/attention.jsonl 重算判据并更新 metrics.json。"""
    res_dir = _PKG / "results"
    rows, att = [], []
    seen: Dict[Tuple, bool] = {}
    for prefix, canonical_name, store in (("blocks", "blocks.jsonl", rows),
                                          ("attention", "attention.jsonl", att)):
        cand_files = [res_dir / canonical_name] + sorted(res_dir.glob(f"{prefix}_s*.jsonl"))
        for p in cand_files:
            if not p.exists():
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if prefix == "blocks":
                    k = ("b", rec["direction"], int(rec["source_layer"]), int(rec["target_layer"]),
                         rec["arm"], int(rec["seed"]))
                else:
                    k = ("a", rec["direction"], int(rec["source_layer"]), rec["arm"], int(rec["seed"]))
                if k in seen:
                    continue
                seen[k] = True
                store.append(rec)
    rows.sort(key=lambda r: (r["direction"], int(r["source_layer"]), int(r["target_layer"]), int(r["seed"]), r["arm"]))
    att.sort(key=lambda r: (r["direction"], int(r["source_layer"]), int(r["seed"]), r["arm"]))
    (res_dir / "blocks.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    (res_dir / "attention.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in att) + "\n", encoding="utf-8")
    for f in list(res_dir.glob("blocks_s*.jsonl")) + list(res_dir.glob("attention_s*.jsonl")):
        f.unlink(missing_ok=True)
    crit = evaluate_criteria(rows, att)
    passed = bool(crit["verdict"]["passed"])
    crit["verdict"]["verdict"] = "S4AP_READY" if passed else "S4AP_FAIL"
    mpath = res_dir / "metrics.json"
    base = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    p_cache = res_dir / "part_pilot.json"
    if p_cache.exists():
        pc = json.loads(p_cache.read_text(encoding="utf-8"))
        for k in ("probe_lambda_by_direction", "layout_by_direction"):
            for dname, val in (pc.get(k) or {}).items():
                base.setdefault("protocol", {}).setdefault(k, {})[dname] = val
    prov = {k: (sha256_of(v) if os.path.exists(v) else None)
            for k, v in PROVENANCE_FILES.items()}
    base.setdefault("protocol", {})["provenance_sha256"] = prov
    base.setdefault("protocol", {})["batch_selection"] = "all"
    base.setdefault("protocol", {})["smoke"] = False
    res = dict(base)
    res["verdict"] = crit["verdict"]["verdict"]
    res["verdict_detail"] = crit["verdict"]
    res["criteria"] = {k: v for k, v in crit.items() if k != "verdict"}
    res["blocks"] = rows
    res["attention_summary"] = [
        {k: v for k, v in r.items() if not k.startswith("per_sample")}
        for r in att
    ]
    res["aggregated"] = True
    res.setdefault("protocol", {}).update(_protocol_texts())
    res["n_blocks_rows"] = len(rows)
    res["n_attention_rows"] = len(att)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", action="store_true", help="S4AP-1: lambda 标定")
    ap.add_argument("--smoke", action="store_true", help="减网格冒烟 (不产出裁决)")
    ap.add_argument("--fresh", action="store_true", help="丢弃 checkpoint 从头跑")
    ap.add_argument("--lam", type=float, default=None, help="显式指定 lambda (默认读 calib.json)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--aggregate", action="store_true", help="从 checkpoint 重算判据")
    ap.add_argument("--block-ids", default=None)
    ap.add_argument("--shard", default=None)
    args = ap.parse_args()

    if args.calib:
        res = run_calib()
        out = Path(args.out) if args.out else (_PKG / "results" / "calib.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[calib] %s -> lambda*=%s (%.1fs)"
              % (res["verdict"], res["selected_lambda"], res["runtime_sec"]), flush=True)
        return 0 if res["verdict"] == "S4AP_LAMBDA_SELECTED" else 1

    if args.aggregate:
        res = aggregate()
        out = Path(args.out) if args.out else (_PKG / "results" / "metrics.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[verdict] %s  (failed: %s)"
              % (res["verdict"], res["verdict_detail"]["failed_criteria"]), flush=True)
        return 0 if res["verdict"] == "S4AP_READY" else 1

    lam = args.lam
    if lam is None:
        cal = _PKG / "results" / "calib.json"
        if not cal.exists():
            print("[fatal] 缺少 results/calib.json —— 先跑 --calib, 或用 --lam 指定", flush=True)
            return 2
        calres = json.loads(cal.read_text(encoding="utf-8"))
        if calres.get("selected_lambda") is None:
            print("[fatal] 标定未选出可行 lambda (%s)" % calres.get("verdict"), flush=True)
            return 1
        lam = float(calres["selected_lambda"])
    print("[run] lambda=%g" % lam, flush=True)

    only = None
    if args.block_ids:
        only = {}
        for part in args.block_ids.split(";"):
            part = part.strip()
            if not part:
                continue
            d, _, lys = part.partition(":")
            only[d.strip()] = set(int(x) for x in lys.split(",") if x.strip())

    res = run_formal(lam, smoke=args.smoke, fresh=args.fresh, only=only, shard=args.shard)
    out = Path(args.out) if args.out else (
        _PKG / "results" / ("metrics_smoke.json" if args.smoke else "metrics.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[verdict] %s  (failed: %s)"
          % (res["verdict"], res["verdict_detail"]["failed_criteria"]), flush=True)
    if args.smoke:
        return 0
    return 0 if res["verdict"] == "S4AP_READY" else 1


if __name__ == "__main__":
    sys.exit(main())
