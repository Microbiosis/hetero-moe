"""s4_pipe.py — v82 S4-P: 管道 (注意力位置旁路 + 按专属通道编号) 的正式实现.

机制 (SPEC.md §2, 不新增自由度):
  序列布局   [ 管道 K=8 位 | 正文 (SEQ-K) 位 ]      <- 管道必须在正文之前 (因果解码器)
  管道位置   attention_mask = 1 (模型把它们当真 token)
  写入点     **目标模型第 0 层的输入** hidden_states (forward_pre_hook)  <- D2 硬约束
  写入内容   Δ_B = (a_s @ P_Aᵀ) @ P_B                <- 共享空间载体, 不学任何映射
  写入范围   **只写管道位置**, 正文位置一律零增量
  强度       ||Δ_i||₂ = λ · (该样本正文 token 在第 0 层的 hidden 平均范数)
  通道编号   槽 → 通道: Q 槽 → 通道 0 (位 0..3); O 槽 → 通道 1 (位 4..7)

四臂:
  arm0_pipe_zero      管道开, 内容为 0   (布局与其它臂完全相同 => 干净配对基线)
  arm1_slot_Q         源第 l 层 Q 槽激活 -> 通道 0
  arm2_wrong_slot_O   源同层 O 槽激活   -> 通道 1
  arm3_random_dir     共享空间内随机方向 -> 通道 0 (与 arm1 同通道 => 纯内容对照)

所有阈值/口径都是**源码常量** (跑前写死), 事后不得放宽:
  THRESHOLDS / SEEDS / N_PERM / ALPHA / LAMBDA_GRID / CALIB_* / INJECT_LAYER / ATT_LAYER
  以及 SPEC.md §3 的六项判据 (P5a / P5b / P5c / P5d / P2′ / P3①).

用法:
  python s4_pipe.py --calib     # S4P-1: lambda 标定 (写死候选集 + 写死选择规则)
  python s4_pipe.py             # 正式跑 (5 种子)
  python s4_pipe.py --smoke     # 减网格冒烟 (不产出裁决)
"""
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
BATCH = 16

L_Q = (0, 1, 2, 3, 6, 12, 18, 23)
L_B = (0, 1, 2, 3)
L_V = (0, 3, 6, 9, 11)

#: **写入点固定在第 0 层** (SPEC §1.2 的 D2 硬约束: 写深了内容被 pad 位的固有表示稀释)
INJECT_LAYER = 0
#: **管道注意力在第 0 层测量** —— 管道内容被写入的那一层才是「模型读不读管道」的机制点;
#: 深层的注意力已不再对内容敏感 (pipe_probe3 实测), 不作机制读数。
ATT_LAYER = 0

SOURCE_SLOTS = ("Q", "O")
ARMS = ("arm0_pipe_zero", "arm1_slot_Q", "arm2_wrong_slot_O", "arm3_random_dir")
INJECT_ARMS = ("arm1_slot_Q", "arm2_wrong_slot_O", "arm3_random_dir")
ARM_SLOT = {"arm1_slot_Q": "Q", "arm2_wrong_slot_O": "O"}
ARM_CHANNEL = {"arm1_slot_Q": 0, "arm2_wrong_slot_O": 1, "arm3_random_dir": 0}

#: 本代的块网格 (主判据集 = 文本两个方向)。
#: ⚠️ ViT 平行臂已于跑前**整臂撤出** (SPEC §0.1 修订 9): ViT 没有「空位」可放管道,
#: 管道位必然携带图像内容 ⇒ 第一轮标定实测: 末层扰动 40 条记录里 20 条越守门上限 0.3
#: (各 λ 的块级 max = 0.5455 / 0.5673 / 0.5664 / 0.5651), 且**通道寻址完全失效**
#: (arm1−arm2 的管道注意力差 5/5 块为负) ⇒ 该臂测的不是同一个机制。
#: 撤出前的实测证据留档在 results/calib_records.jsonl (direction == "ViT->ViT" 的 40 行)。
DIRECTIONS = (
    ("Qwen->BERT", "Qwen", "BERT", L_Q, L_B),
    ("BERT->Qwen", "BERT", "Qwen", L_B, L_Q),
)

#: 已撤出的方向 (只留记录, 不参与任何跑批与裁决)
WITHDRAWN_DIRECTIONS = (
    ("ViT->ViT", "ViT", "ViT", L_V, L_V),
)

SEEDS = (0, 1, 2, 3, 4)
N_PERM = 10000
ALPHA = 0.01
LAMBDA_GRID = (0.5, 1.0, 2.0, 3.0)
CALIB_MAX_FINAL = 0.3
#: 修订 9: ②③ 的门控从「全部 12 组都为正」改为「主判据集里 **≥ 8/12** 组为正」
#: (用户 2026-11-18 拍板)。理由: 正式判据 P5a/P5b 是**块级聚合置换检验**, 不要求
#: 逐组全过; 实测每个 λ 都有 1–3 个源层失灵 (源槽经 256 维共享空间投影后与目标方向
#: 近乎正交) ⇒ 用逐组门槛卡 λ 比正式判据本身还严。
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
_S3 = _load_by_path(_PKG.parent / "v81_s3_semantic" / "s3_flow.py", "s3_flow_v81_by_path")
_V77 = _S3._V77

MEAS = _S3.MEAS
Backbone = _S3.Backbone
capture_slots = _S3.capture_slots
fetch_models = _S3.fetch_models
fetch_texts = _S3.fetch_texts
fetch_vit_images = _S3.fetch_vit_images
make_batches = _S3.make_batches
perm_paired = _S3.perm_paired
perm_group_stratified = _S3.perm_group_stratified
sign_test = _S3.sign_test
sha256_of = _S3.sha256_of
CACHE_DIR = _S3.CACHE_DIR

PROVENANCE_FILES = {
    "v62_measure": _V77.PROVENANCE_FILES["v62_measure"],
    "v75_g2_conjugation": str(_PKG.parent / "v75_weight_conjugation" / "g2_conjugation.py"),
    "v77_true_model": str(_PKG.parent / "v77_true_model_flow" / "true_model.py"),
    "v81_s3_flow": str(_PKG.parent / "v81_s3_semantic" / "s3_flow.py"),
    "v82_s4_pipe": str(_PKG / "s4_pipe.py"),
}


def _seed_of(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


# ================================================================ 目标模型与布局
def load_target(name: str, eager: bool = True):
    """目标底座。注意力权重必须 eager (默认 sdpa 取不到)。"""
    bb = Backbone(name)
    if eager:
        ok = False
        try:
            bb.model.set_attn_implementation("eager")
            ok = (getattr(bb.model.config, "_attn_implementation", None) == "eager")
        except Exception:
            ok = False
        if not ok:
            from transformers import AutoModel
            bb.model = AutoModel.from_pretrained(
                bb.local, attn_implementation="eager").eval()
            bb.layers = _V77.layers_of(bb.model)
    return bb


def build_layout(bb, is_vit: bool, texts, imgs) -> dict:
    """布局 = [管道 K 位 | 正文位]。返回布局字典 (文本臂正文长度逐样本不同)。"""
    if is_vit:
        px = bb.proc(images=imgs, return_tensors="pt")["pixel_values"]
        n_patch = bb.model.config.image_size // bb.model.config.patch_size
        n_seq = n_patch * n_patch + 1          # +CLS
        # 管道 = 位置 0..7 (CLS + 7 个 patch); 正文 = 位置 8..n_seq-1
        body_len = n_seq - K_PIPE
        body_mask = torch.ones(px.shape[0], body_len, dtype=torch.float32)
        if DROP_FIRST_TOKEN:
            body_mask[:, 0] = 0.0
        return {"kind": "image", "n": px.shape[0], "n_seq": n_seq,
                "px": px, "body_start": K_PIPE, "body_mask": body_mask,
                "take": None, "n_body": body_len}
    enc = bb.tok([t["text"] for t in texts], padding="max_length", truncation=True,
                 max_length=SEQ, return_tensors="pt")
    ids, am = enc["input_ids"], enc["attention_mask"]
    lens = am.sum(1)
    take = torch.minimum(lens, torch.tensor(TXT))
    pad_id = bb.tok.pad_token_id if bb.tok.pad_token_id is not None else bb.tok.eos_token_id
    new_ids = torch.full_like(ids, pad_id)
    new_mask = torch.zeros_like(am)
    new_mask[:, :K_PIPE] = 1
    body_mask = torch.zeros((ids.shape[0], TXT), dtype=torch.float32)
    for i in range(ids.shape[0]):
        L = int(take[i])
        new_ids[i, K_PIPE:K_PIPE + L] = ids[i, :L]
        new_mask[i, K_PIPE:K_PIPE + L] = 1
        body_mask[i, :L] = 1.0
    if DROP_FIRST_TOKEN:
        body_mask[:, 0] = 0.0
    return {"kind": "text", "n": ids.shape[0], "n_seq": SEQ,
            "ids": new_ids, "mask": new_mask, "take": take,
            "body_start": K_PIPE, "body_mask": body_mask, "n_body": TXT}


def query_hi(layout: dict, i: int) -> int:
    if layout["take"] is None:
        return layout["n_seq"]
    return K_PIPE + int(layout["take"][i])


def capture_layer_input(bb, layout: dict, layer_idx: int):
    cap = {}

    def hook(module, args, kwargs):
        hs = kwargs.get("hidden_states", None)
        if hs is None and len(args) > 0:
            hs = args[0]
        cap["h"] = hs.detach()

    hnd = bb.layers[layer_idx].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        forward_raw(bb, layout, None, False)
    finally:
        hnd.remove()
    return cap["h"]


def body_norms(h0: torch.Tensor, layout: dict) -> torch.Tensor:
    """逐样本: 正文 token 在该层的 hidden 平均范数 (强度归一化基准)。"""
    n = h0.shape[0]
    out = torch.zeros(n, dtype=torch.float32)
    for i in range(n):
        lo = layout["body_start"]
        hi = query_hi(layout, i)
        seg = h0[i, lo:hi].reshape(-1, h0.shape[-1]).float()
        out[i] = seg.norm(dim=-1).mean() if seg.numel() else torch.tensor(0.0)
    return out


def install_pipe(layer, delta_full):
    """在写入层的**输入**处, **只**往管道位置加 delta (其余位置零增量)。"""
    def hook(module, args, kwargs):
        hs = kwargs.get("hidden_states", None)
        if hs is None and len(args) > 0:
            args = (args[0] + delta_full,) + tuple(args[1:])
            return args, kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = hs + delta_full
        return args, kwargs
    return layer.register_forward_pre_hook(hook, with_kwargs=True)


def make_pipe_delta(vec: torch.Tensor, lam: float, bn: torch.Tensor,
                    P_target: torch.Tensor, channel: int,
                    n_seq: int, dtype) -> torch.Tensor:
    """vec [N,256] 共享空间向量 -> 目标空间; 逐样本归一化到 λ·正文平均范数;
    只写到指定通道的 CH_SIZE 个管道位。"""
    raw = (vec @ P_target).float()
    rn = raw.norm(dim=-1)
    scale = torch.where(rn > 1e-9, lam * bn / rn.clamp_min(1e-12), torch.zeros_like(bn))
    delta = raw * scale[:, None]
    full = torch.zeros(vec.shape[0], n_seq, delta.shape[-1], dtype=dtype)
    if channel is None:
        return full
    base = channel * CH_SIZE
    for p in range(base, base + CH_SIZE):
        full[:, p, :] = delta
    return full


def forward_raw(bb, layout: dict, delta_pipe, want_att: bool):
    with torch.no_grad():
        if layout["kind"] == "text":
            return bb.model(input_ids=layout["ids"], attention_mask=layout["mask"],
                            output_attentions=want_att, output_hidden_states=True)
        return bb.model(pixel_values=layout["px"],
                        output_attentions=want_att, output_hidden_states=True)


def run_forward(bb, layout: dict, delta_pipe, want_att: bool,
                read_layers: Sequence[int], att_layers: Sequence[int]):
    """跑完全部批次, 返回 (reads, fins, attn)。

    reads[j] = 读出层 j (hidden_states[j+1]) 的**正文位置** [N, n_body, D]
    fins     = 末层输出 [N, n_seq, D]
    attn[j]  = (管道通道 0 平均注意力, 管道通道 1 平均注意力), 逐样本
    """
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
                if layout["kind"] == "text":
                    out = bb.model(input_ids=layout["ids"][s:e],
                                   attention_mask=layout["mask"][s:e],
                                   output_attentions=want_att, output_hidden_states=True)
                else:
                    out = bb.model(pixel_values=layout["px"][s:e],
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


def rel_final_body(fin: torch.Tensor, ref_fin: torch.Tensor, layout: dict) -> float:
    """正文位置的末层相对扰动 (逐样本, 取均值)。"""
    rr = []
    for i in range(layout["n"]):
        lo = layout["body_start"]
        hi = query_hi(layout, i)
        a = fin[i, lo:hi].float()
        b = ref_fin[i, lo:hi].float()
        rr.append(float((a - b).norm() / b.norm().clamp_min(1e-9)))
    return float(np.mean(rr))


# ================================================================ 探针 (token 级, 带 mask)
def _gather_rows(readout: torch.Tensor, mask: torch.Tensor, idx: np.ndarray):
    r = readout[idx]
    m = mask[idx].bool()
    return r[m], m.sum(1).long()


def _gather_targets(tgt: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    return tgt.repeat_interleave(counts, dim=0)


def masked_r2_multi(readout: torch.Tensor, mask: torch.Tensor,
                    targets: Dict[str, torch.Tensor], lam: float,
                    train_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, float]:
    """token 级 ridge 探针, **只读正文里的真实 token** (pad 位不参与)。

    与 v81 的 _ridge_r2_multi 同式 (逐维标准化用训练行统计量 / 留出集 R² /
    SS_tot 围绕训练集 y 均值), 差别只有一处: 本代按 **正文有效 token** 掩码聚集,
    因为管道方案里 pad 位是「会注意到管道但不该算正文」的位置。
    """
    D = readout.shape[-1]
    Xtr, ctr = _gather_rows(readout, mask, train_idx)
    Xte, cte = _gather_rows(readout, mask, test_idx)
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    G0 = Xtr.t() @ Xtr
    eye = torch.eye(D, dtype=Xtr.dtype)
    out: Dict[str, float] = {}
    for name, tgt in targets.items():
        Ytr = _gather_targets(tgt[train_idx], ctr)
        Yte = _gather_targets(tgt[test_idx], cte)
        ym = Ytr.mean(0, keepdim=True)
        Ytr = Ytr - ym
        Yte = Yte - ym
        A = torch.linalg.solve(G0 + lam * eye, Xtr.t() @ Ytr)
        pred = Xte @ A
        ss_res = float((pred - Yte).pow(2).sum())
        ss_tot = float(Yte.pow(2).sum())
        out[name] = 1.0 - ss_res / (ss_tot + 1e-12)
    return out


def masked_lambda_scores(readout: torch.Tensor, mask: torch.Tensor,
                          tgt: torch.Tensor, train_idx: np.ndarray) -> Dict[float, float]:
    """每个候选 lambda 的**训练集内** 5 折 CV 平均 R² (测试样本全程不参与)。"""
    folds = np.arange(len(train_idx)) % PROBE_CV_FOLDS
    scores: Dict[float, List[float]] = {float(g): [] for g in PROBE_LAM_GRID}
    for f in range(PROBE_CV_FOLDS):
        tr_pos = np.where(folds != f)[0]
        te_pos = np.where(folds == f)[0]
        if tr_pos.size == 0 or te_pos.size == 0:
            continue
        Xa, ca = _gather_rows(readout, mask, train_idx[tr_pos])
        Xb, cb = _gather_rows(readout, mask, train_idx[te_pos])
        mu = Xa.mean(0, keepdim=True)
        sd = Xa.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
        Xa = (Xa - mu) / sd
        Xb = (Xb - mu) / sd
        Ya = _gather_targets(tgt[train_idx[tr_pos]], ca)
        Yb = _gather_targets(tgt[train_idx[te_pos]], cb)
        ym = Ya.mean(0, keepdim=True)
        Ya = Ya - ym
        Yb = Yb - ym
        G0 = Xa.t() @ Xa
        d = Xa.shape[1]
        for g in PROBE_LAM_GRID:
            A = torch.linalg.solve(G0 + float(g) * torch.eye(d, dtype=Xa.dtype), Xa.t() @ Ya)
            pred = Xb @ A
            ssr = float((pred - Yb).pow(2).sum())
            sst = float(Yb.pow(2).sum())
            scores[float(g)].append(1.0 - ssr / (sst + 1e-12))
    return {g: (float(np.mean(v)) if v else float("nan")) for g, v in scores.items()}


def pooled_r2_multi(readout: torch.Tensor, mask: torch.Tensor,
                    targets: Dict[str, torch.Tensor], lam: float,
                    train_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, float]:
    """预登记**辅助读数**: 样本级 mean-pool (只对正文有效 token 求均值) 上的 ridge R²。
    用途只有一个 —— 区分「信息没流过去」与「token 级探针读不出来」; 不参与任何裁决。"""
    n = readout.shape[0]
    D = readout.shape[-1]
    m = mask.unsqueeze(-1)
    pooled = (readout * m).sum(1) / m.sum(1).clamp_min(1.0)
    Xtr, Xte = pooled[train_idx], pooled[test_idx]
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    G0 = Xtr.t() @ Xtr
    eye = torch.eye(D, dtype=Xtr.dtype)
    out: Dict[str, float] = {}
    for name, tgt in targets.items():
        Ytr, Yte = tgt[train_idx], tgt[test_idx]
        ym = Ytr.mean(0, keepdim=True)
        Ytr = Ytr - ym
        Yte = Yte - ym
        A = torch.linalg.solve(G0 + lam * eye, Xtr.t() @ Ytr)
        pred = Xte @ A
        ss_res = float((pred - Yte).pow(2).sum())
        ss_tot = float(Yte.pow(2).sum())
        out[name] = 1.0 - ss_res / (ss_tot + 1e-12)
    return out


# ================================================================ 语料 / 源侧 / 划分
def capture_source(name: str, layers, texts, imgs) -> Dict[Tuple[int, str], torch.Tensor]:
    """抓源模型若干 (层, 槽) 的**样本级 mean-pool** 激活 (原生维; 与种子无关, 抓一次)。"""
    bb = Backbone(name)
    batches = make_batches(bb, name, texts, imgs)
    want = [(ly, slot) for ly in layers for slot in SOURCE_SLOTS]
    acc: Dict[Tuple[int, str], List[torch.Tensor]] = {k: [] for k in want}
    for b in batches:
        st = capture_slots(bb, b, want)
        for k in want:
            acc[k].append(st[k].mean(dim=1))
    src = {k: torch.cat(v, 0) for k, v in acc.items()}
    del bb, batches, acc
    gc.collect()
    return src


def split_idx(is_vit: bool, vit_meta) -> Tuple[np.ndarray, np.ndarray]:
    """文本: 前 64 训 / 后 64 测 (同 v81)。
    ViT: **按类别分层** —— 每类的 8 张 train-half 训、8 张 test-half 测,
    两侧都覆盖 4 个类别 (修 v81 §4.6 记录的「train/test 类别不相交」缺陷)。"""
    if not is_vit:
        return (np.arange(TEXT_SPLIT_TRAIN), np.arange(TEXT_SPLIT_TRAIN, N_TEXT))
    items = vit_meta["items"]
    tr = np.array([i for i, m in enumerate(items) if m["split"] == "train"], dtype=int)
    te = np.array([i for i, m in enumerate(items) if m["split"] != "train"], dtype=int)
    return tr, te


def att_record(direction: str, ly: int, arm: str, seed: int,
               c0: torch.Tensor, c1: torch.Tensor) -> dict:
    pipe = (c0 + c1)
    flip = (c0 - c1)
    return {
        "direction": direction, "source_layer": int(ly), "arm": arm, "seed": int(seed),
        "attn_pipe_med": float(pipe.median()), "attn_pipe_mean": float(pipe.mean()),
        "attn_ch0_med": float(c0.median()), "attn_ch1_med": float(c1.median()),
        "flip_med": float(flip.median()), "flip_mean": float(flip.mean()),
        "n_samples": int(pipe.numel()),
        "per_sample_ch0": [float(x) for x in c0],
        "per_sample_ch1": [float(x) for x in c1],
    }


# ================================================================ pilot (探针 lambda)
def pilot_probe_lambda(bb, layout: dict, blocks, s_name: str, t_name: str,
                       src_cache, proj, train_idx: np.ndarray,
                       lam_pipe: float, bn: torch.Tensor, dtype) -> dict:
    """预登记 pilot 块 (seed 0) + **训练样本内** 5 折 CV 选 token 级探针的 ridge lambda。"""
    p_a, p_b = proj[s_name], proj[t_name]
    scores: Dict[float, List[float]] = {float(g): [] for g in PROBE_LAM_GRID}
    for ly, j in blocks:
        zq = src_cache[s_name][(ly, "Q")] @ p_a.t()
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


# ================================================================ 判据聚合
def _blocks_of(rows: List[dict]):
    out: Dict[Tuple[str, int, int], Dict[str, Dict[int, dict]]] = {}
    for r in rows:
        key = (r["direction"], int(r["source_layer"]), int(r["target_layer"]))
        out.setdefault(key, {}).setdefault(r["arm"], {})[int(r["seed"])] = r
    return out


def _att_blocks(recs: List[dict]):
    out: Dict[Tuple[str, int], Dict[str, Dict[int, dict]]] = {}
    for r in recs:
        key = (r["direction"], int(r["source_layer"]))
        out.setdefault(key, {}).setdefault(r["arm"], {})[int(r["seed"])] = r
    return out


def _median(vals) -> Optional[float]:
    v = [float(x) for x in vals if x is not None]
    return float(np.median(v)) if v else None


def paired_obs(bl, arm_a: str, arm_b: str, field: str, keys):
    obs, bids = [], []
    for bi, key in enumerate(keys):
        aa = bl[key].get(arm_a, {})
        bb = bl[key].get(arm_b, {})
        for s in sorted(set(aa) & set(bb)):
            va, vb = aa[s].get(field), bb[s].get(field)
            if va is None or vb is None:
                continue
            obs.append((va, vb))
            bids.append(bi)
    return obs, bids


def _arm_stat(bl, key, arm, field, seeds=None):
    aa = bl[key].get(arm, {})
    if seeds is not None:
        aa = {s: v for s, v in aa.items() if s in seeds}
    return _median([v.get(field) for v in aa.values()])


def evaluate_criteria(rows: List[dict], att_recs: List[dict]) -> dict:
    bl = _blocks_of(rows)
    abl = _att_blocks(att_recs)
    text_keys = sorted(k for k in bl if k[0] != "ViT->ViT")
    vit_keys = sorted(k for k in bl if k[0] == "ViT->ViT")
    text_att = sorted(k for k in abl if k[0] != "ViT->ViT")
    vit_att = sorted(k for k in abl if k[0] == "ViT->ViT")

    # ---- P5a 管道连通: arm1 vs arm0 的管道注意力中位之差 (臂间配对置换)
    p5a_obs, p5a_bids = paired_obs(abl, "arm1_slot_Q", "arm0_pipe_zero",
                                    "attn_pipe_med", text_att)
    p5a_perm = perm_paired(p5a_obs, p5a_bids, N_PERM, seed=2001)
    p5a_deltas = [a - b for a, b in p5a_obs]
    p5a_median = _median(p5a_deltas)
    # 附加 (同一判据内): arm1 vs arm3 —— 「随便什么内容都能被读」必须不成立
    p5a3_obs, p5a3_bids = paired_obs(abl, "arm1_slot_Q", "arm3_random_dir",
                                     "attn_pipe_med", text_att)
    p5a3_perm = perm_paired(p5a3_obs, p5a3_bids, N_PERM, seed=2002)
    p5a3_deltas = [a - b for a, b in p5a3_obs]
    p5a3_median = _median(p5a3_deltas)
    p5a_pass = bool(p5a_median is not None and p5a_median > 0
                    and p5a_perm["permutation_p"] is not None
                    and p5a_perm["permutation_p"] < ALPHA
                    and p5a3_median is not None and p5a3_median > 0
                    and p5a3_perm["permutation_p"] is not None
                    and p5a3_perm["permutation_p"] < ALPHA)

    # ---- P5b 管道可分 (槽): 通道寻址统计量 flip 的 arm1 与 arm2 之差
    p5b_obs, p5b_bids = paired_obs(abl, "arm1_slot_Q", "arm2_wrong_slot_O",
                                    "flip_med", text_att)
    p5b_perm = perm_paired(p5b_obs, p5b_bids, N_PERM, seed=2003)
    p5b_deltas = [a - b for a, b in p5b_obs]
    p5b_median = _median(p5b_deltas)
    p5b_pass = bool(p5b_median is not None and p5b_median > 0
                    and p5b_perm["permutation_p"] is not None
                    and p5b_perm["permutation_p"] < ALPHA)

    # ---- P5c 贯通 (正文可读): arm1 vs arm0 的 token 级 r2_q 增量
    p5c_obs, p5c_bids = paired_obs(bl, "arm1_slot_Q", "arm0_pipe_zero", "r2_q", text_keys)
    p5c_perm = perm_paired(p5c_obs, p5c_bids, N_PERM, seed=2004)
    p5c_deltas = [a - b for a, b in p5c_obs]
    p5c_median = _median(p5c_deltas)
    p5c_pass = bool(p5c_median is not None and p5c_median > 0
                    and p5c_perm["permutation_p"] is not None
                    and p5c_perm["permutation_p"] < ALPHA)

    # ---- P5d 选择性: 非目标槽 (O) 的可读性增量不普涨
    p5d_obs, p5d_bids = paired_obs(bl, "arm1_slot_Q", "arm0_pipe_zero", "r2_o", text_keys)
    p5d_perm = perm_paired(p5d_obs, p5d_bids, N_PERM, seed=2005)
    p5d_deltas = [a - b for a, b in p5d_obs]
    p5d_median = _median(p5d_deltas)
    p5d_pass = bool(p5d_median is not None
                    and (p5d_median <= 0
                         or p5d_perm["permutation_p"] is None
                         or p5d_perm["permutation_p"] >= THRESHOLDS["p5d_selectivity_p_min"]))

    # ---- P2′ 深度语义: 同层块 vs 跨层块 (块内分层置换)
    same_mask = np.array([1 if k[1] == k[2] else 0 for k in text_keys], dtype=int)
    p2_vals = np.array([(_arm_stat(bl, k, "arm1_slot_Q", "r2_q") or 0.0)
                        - (_arm_stat(bl, k, "arm0_pipe_zero", "r2_q") or 0.0)
                        for k in text_keys], dtype=float)
    p2_perm = perm_group_stratified(p2_vals, same_mask, N_PERM, seed=2006)
    p2_same = _median(p2_vals[same_mask == 1].tolist())
    p2_cross = _median(p2_vals[same_mask == 0].tolist())
    p2_pass = bool(p2_same is not None and p2_cross is not None
                   and p2_same >= p2_cross
                   and p2_perm["permutation_p"] is not None
                   and p2_perm["permutation_p"] < ALPHA)

    # ---- P3① 守门: 注入臂块的末层相对扰动区间
    p3_rows = [r for r in rows if r["arm"] in INJECT_ARMS]
    bad = [r for r in p3_rows
           if not (THRESHOLDS["p3_min_rel_perturbation"] < r["rel_perturb_final"]
                   <= THRESHOLDS["p3_max_rel_perturbation"])]
    fins = [r["rel_perturb_final"] for r in p3_rows]
    p3_1 = {"n_rows": len(p3_rows), "n_violations": len(bad),
            "min_rel_perturbation": min(fins) if fins else None,
            "max_rel_perturbation": max(fins) if fins else None,
            "passed": len(bad) == 0,
            "violations": [{"direction": r["direction"], "source_layer": r["source_layer"],
                            "target_layer": r["target_layer"], "arm": r["arm"],
                            "seed": r["seed"], "rel_perturb_final": r["rel_perturb_final"]}
                           for r in bad[:20]]}

    passed = bool(p5a_pass and p5b_pass and p5c_pass and p5d_pass
                  and p2_pass and p3_1["passed"])
    failed = [n for n, ok in (("P5a", p5a_pass), ("P5b", p5b_pass), ("P5c", p5c_pass),
                              ("P5d", p5d_pass), ("P2_prime", p2_pass),
                              ("P3_1", p3_1["passed"])) if not ok]

    # ---- ViT 平行臂 (单独报告, 不参与主裁决)
    vit = _parallel_vit(bl, abl, vit_keys, vit_att)

    # ---- 辅助读数 (预登记, 不参与裁决): 样本级 mean-pool 探针
    aux = {}
    for arm in ARMS:
        vals = []
        for k in text_keys:
            v = _arm_stat(bl, k, arm, "r2_q_pooled_aux")
            if v is not None:
                vals.append(v)
        aux[arm] = {"median_r2_q_pooled_aux": _median(vals)}

    return {
        "P5a": {"passed": p5a_pass, "n_blocks": len(text_att), "n_obs": p5a_perm["n_obs"],
                "n_positive": sum(1 for x in p5a_deltas if x > 0),
                "median_delta": p5a_median,
                "mean_delta": float(np.mean(p5a_deltas)) if p5a_deltas else None,
                "permutation_p": p5a_perm["permutation_p"], "n_perm": p5a_perm.get("n_perm", N_PERM),
                "alpha": ALPHA,
                "content_specific_vs_random": {
                    "n_obs": p5a3_perm["n_obs"], "median_delta": p5a3_median,
                    "n_positive": sum(1 for x in p5a3_deltas if x > 0),
                    "permutation_p": p5a3_perm["permutation_p"], "n_perm": p5a3_perm.get("n_perm", N_PERM)},
                "rule": "arm1−arm0 > 0 且 p<α, **且** arm1−arm3 > 0 且 p<α"},
        "P5b": {"passed": p5b_pass, "n_blocks": len(text_att), "n_obs": p5b_perm["n_obs"],
                "n_positive": sum(1 for x in p5b_deltas if x > 0),
                "median_delta": p5b_median,
                "mean_delta": float(np.mean(p5b_deltas)) if p5b_deltas else None,
                "permutation_p": p5b_perm["permutation_p"], "n_perm": p5b_perm.get("n_perm", N_PERM),
                "alpha": ALPHA,
                "rule": "flip(arm1)−flip(arm2) > 0 且 p<α (flip = 通道0注意力 − 通道1注意力)"},
        "P5c": {"passed": p5c_pass, "n_blocks": len(text_keys), "n_obs": p5c_perm["n_obs"],
                "n_positive": sum(1 for x in p5c_deltas if x > 0),
                "median_delta": p5c_median,
                "mean_delta": float(np.mean(p5c_deltas)) if p5c_deltas else None,
                "permutation_p": p5c_perm["permutation_p"], "n_perm": p5c_perm.get("n_perm", N_PERM),
                "alpha": ALPHA},
        "P5d": {"passed": p5d_pass, "n_blocks": len(text_keys), "n_obs": p5d_perm["n_obs"],
                "n_positive": sum(1 for x in p5d_deltas if x > 0),
                "median_delta": p5d_median,
                "permutation_p": p5d_perm["permutation_p"], "n_perm": p5d_perm.get("n_perm", N_PERM),
                "alpha": ALPHA,
                "rule": "median<=0 或 p>=%.2f" % THRESHOLDS["p5d_selectivity_p_min"],
                "passed_flag": p5d_pass},
        "P2_prime": {"passed": p2_pass, "n_same_layer": int((same_mask == 1).sum()),
                     "n_cross_layer": int((same_mask == 0).sum()),
                     "median_same_layer": p2_same, "median_cross_layer": p2_cross,
                     "permutation_p": p2_perm["permutation_p"], "n_perm": p2_perm.get("n_perm", N_PERM),
                     "alpha": ALPHA},
        "P3_1": p3_1,
        "vit_parallel": vit,
        "aux_pooled": aux,
        "verdict": {"verdict": "S4PIPE_READY" if passed else "S4PIPE_FAIL",
                    "passed": passed, "failed_criteria": failed,
                    "note": "主判据集 = 64 个文本块 (P5a/P5b 的注意力块 = 12 个 (方向,源层) 组); "
                            "ViT 臂已于跑前整臂撤出 (SPEC §0.1 修订 9), 本代只跑文本 64 块。"},
    }


def _parallel_vit(bl, abl, vit_keys, vit_att) -> dict:
    """ViT 平行臂: 本代已于跑前整臂撤出 ⇒ 这里只写撤出记录 (无数据)。"""
    out = {}
    for name, arm_a, arm_b, field in (("attn_pipe", "arm1_slot_Q", "arm0_pipe_zero", "attn_pipe_med"),
                                      ("flip", "arm1_slot_Q", "arm2_wrong_slot_O", "flip_med")):
        obs, bids = paired_obs(abl, arm_a, arm_b, field, vit_att)
        if not obs:
            continue
        pr = perm_paired(obs, bids, N_PERM, seed=2007)
        ds = [a - b for a, b in obs]
        out[name] = {"n_blocks": len(vit_att), "n_obs": pr["n_obs"],
                     "median_delta": _median(ds),
                     "n_positive": sum(1 for x in ds if x > 0),
                     "permutation_p": pr["permutation_p"], "n_perm": pr.get("n_perm", N_PERM)}
    for name, field in (("r2_q", "r2_q"), ("r2_o", "r2_o")):
        obs, bids = paired_obs(bl, "arm1_slot_Q", "arm0_pipe_zero", field, vit_keys)
        if not obs:
            continue
        pr = perm_paired(obs, bids, N_PERM, seed=2008)
        ds = [a - b for a, b in obs]
        out[name] = {"n_blocks": len(vit_keys), "n_obs": pr["n_obs"],
                     "median_delta": _median(ds),
                     "n_positive": sum(1 for x in ds if x > 0),
                     "permutation_p": pr["permutation_p"], "n_perm": pr.get("n_perm", N_PERM)}
    out["role"] = ("ViT 平行臂已于**跑前整臂撤出** (SPEC §0.1 修订 9, 用户 2026-11-18 拍板): "
                    "ViT 没有「空位」可放管道, 管道位必然携带图像内容; 第一轮标定实测 末层扰动 "
                    "40 条记录里 20 条越守门上限 0.3, 且**通道寻址完全失效** (arm1−arm2 的管道"
                    "注意力差 5/5 块为负) ⇒ 该臂测的不是同一个机制。撤出前的实测证据留档在 "
                    "results/calib_records.jsonl (direction == \"ViT->ViT\" 的 40 行)。")
    out["withdrawn"] = True
    return out


def evaluate_calibration(cand: Dict[str, dict], arm0: Dict[str, dict]):
    """按写死的三条规则评估每个候选 λ (纯函数, 可用合成数据单测)。

    ① 全部块 max final ≤ CALIB_MAX_FINAL (安全闸门, 覆盖全部块);
    ② 主判据集 (文本 12 组) 里 **≥ 8 组** attn_pipe(arm1) − attn_pipe(arm0) > 0;
    ③ 主判据集里 **≥ 8 组** flip(arm1) − flip(arm2) > 0;
    三条全满足才算可行; 取可行候选里最大的那个 (修订 9: ②③ 是「≥ 8/12 组」,
    不是逐组全过 —— 正式判据 P5a/P5b 本就是块级聚合置换检验)。
    """
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
        # 门控作用在**主判据集** (文本 12 组): ②③ 要求 **≥ CALIB_MIN_GROUPS 组**为正
        # (修订 9, 用户拍板); 安全闸门 ① 覆盖全部块。
        gate = [b for b in blocks if b["gating"]]
        n2 = sum(1 for b in gate if b["cond2_pipe_alive"])
        n3 = sum(1 for b in gate if b["cond3_addressed"])
        ok = bool(cond1 and len(gate) == CALIB_N_GROUPS
                  and n2 >= CALIB_MIN_GROUPS and n3 >= CALIB_MIN_GROUPS)
        if ok:
            feasible_lams.append(lam)
        results[key] = {
            "lambda": lam, "feasible": ok,
            "cond1_max_final_ok": bool(cond1), "max_final": max_final,
            "n_blocks": len(blocks),
            "n_fail_cond2": sum(1 for b in gate if not b["cond2_pipe_alive"]),
            "n_fail_cond3": sum(1 for b in gate if not b["cond3_addressed"]),
            "n_gating_blocks": len(gate),
            "blocks": blocks}
    return results, feasible_lams


# ================================================================ S4P-1: lambda 标定
def run_calib() -> dict:
    """写死的候选集 {0.5,1,2,3} + 写死的选择规则 (SPEC §3.1)。

    规则 (三条全满足才算可行):
      ① 全部块 max final ≤ 0.3            (没把目标模型搞坏)
      ② 全部 (方向,源层) 组: attn_pipe(arm1) − attn_pipe(arm0) > 0   (管道通水)
      ③ 全部 (方向,源层) 组: flip(arm1) − flip(arm2) > 0              (按通道寻址)
    取满足条件里**最大**的候选; 无一满足 ⇒ CALIB_NO_FEASIBLE_LAMBDA (报回, 不硬选)。
    """
    t0 = time.time()
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(min(16, os.cpu_count() or 8))

    fetch_models()
    texts = fetch_texts(N_TEXT)
    imgs, vit_meta = fetch_vit_images()
    proj = MEAS.projections(0)
    src_cache = {}
    for name, layers in (("Qwen", L_Q), ("BERT", L_B)):
        src_cache[name] = capture_source(name, layers, texts, None)
        print("[source] %s 抓完" % name, flush=True)

    cand: Dict[str, dict] = {}
    arm0: Dict[str, dict] = {}
    # 标定也做断点续跑: 每算完一个 (λ, 源层, 臂) 就落一行, 重跑同一命令会自动复用
    ck_dir = _PKG / "results"
    ck_dir.mkdir(parents=True, exist_ok=True)
    ck_path = ck_dir / "calib_records.jsonl"
    ck = ck_path.open("a", encoding="utf-8")
    done_calib: Dict[Tuple, dict] = {}
    if ck_path.exists():
        with ck_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                done_calib[(r["direction"], "%.4g" % r["lam"], int(r["source_layer"]),
                            r["arm"])] = r
        print("[calib-resume] 已有 %d 条标定记录, 直接复用" % len(done_calib), flush=True)
    for dname, s_name, t_name, s_layers, t_layers in DIRECTIONS:
        for lam in LAMBDA_GRID:
            for ly in s_layers:
                for arm in ("arm1_slot_Q", "arm2_wrong_slot_O"):
                    k = (dname, "%.4g" % lam, int(ly), arm)
                    if k in done_calib:
                        cand.setdefault("%.4g" % lam, {}).setdefault(dname, {}).setdefault(
                            str(ly), {})[arm] = done_calib[k]
    for dname, s_name, t_name, s_layers, t_layers in DIRECTIONS:
        is_vit = (t_name == "ViT")
        bb = load_target(t_name, eager=True)
        layout = build_layout(bb, is_vit, texts, imgs)
        h0 = capture_layer_input(bb, layout, INJECT_LAYER)
        bn = body_norms(h0, layout)
        dtype = h0.dtype
        p_a, p_b = proj[s_name], proj[t_name]
        _, fins0, attn0 = run_forward(bb, layout, None, True, [], [ATT_LAYER])
        rec0 = att_record(dname, 0, "arm0_pipe_zero", 0,
                          attn0[ATT_LAYER][0], attn0[ATT_LAYER][1])
        arm0[dname] = rec0
        for lam in LAMBDA_GRID:
            for ly in s_layers:
                for arm in ("arm1_slot_Q", "arm2_wrong_slot_O"):
                    k = (dname, "%.4g" % lam, int(ly), arm)
                    if k in done_calib:
                        continue
                    z = src_cache[s_name][(ly, ARM_SLOT[arm])] @ p_a.t()
                    delta = make_pipe_delta(z, lam, bn, p_b, ARM_CHANNEL[arm],
                                            layout["n_seq"], dtype)
                    _, fins, attn = run_forward(bb, layout, delta, True, [], [ATT_LAYER])
                    rec = att_record(dname, ly, arm, 0,
                                     attn[ATT_LAYER][0], attn[ATT_LAYER][1])
                    rec["rel"] = rel_final_body(fins, fins0, layout)
                    rec["lam"] = float(lam)
                    cand.setdefault("%.4g" % lam, {}).setdefault(dname, {}).setdefault(
                        str(ly), {})[arm] = rec
                    done_calib[k] = rec
                    ck.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    ck.flush()
                    del delta, fins
                print("[calib] %s lam=%g ly=%d 完成 (%.0fs)"
                      % (dname, lam, ly, time.time() - t0), flush=True)
        del bb, layout, h0
        gc.collect()
    ck.close()

    results, feasible_lams = evaluate_calibration(cand, arm0)
    selected = max(feasible_lams) if feasible_lams else None
    return {
        "verdict": "S4PIPE_LAMBDA_SELECTED" if selected is not None
                   else "CALIB_NO_FEASIBLE_LAMBDA",
        "selected_lambda": selected,
        "lambda_grid": list(LAMBDA_GRID),
        "rule": ("① 全部块 max final ≤ %.1f; ② 主判据集 (文本 %d 组) 里 "
                 "**≥ %d 组** attn_pipe(arm1) − attn_pipe(arm0) > 0; "
                 "③ 主判据集里 **≥ %d 组** flip(arm1) − flip(arm2) > 0; "
                 "取可行候选里的最大者" % (CALIB_MAX_FINAL, CALIB_N_GROUPS,
                                         CALIB_MIN_GROUPS, CALIB_MIN_GROUPS)),
        "thresholds": THRESHOLDS,
        "attention_layer": ATT_LAYER,
        "inject_layer": INJECT_LAYER,
        "candidates": results,
        "runtime_sec": round(time.time() - t0, 1),
    }


# ================================================================ 正式跑批
def _protocol_texts() -> dict:
    # 协议层的**静态文本** (scope / discipline / limits) —— 单一来源。
    # run_formal 与 aggregate 共用: aggregate 的 base 可能是**旧批次**写的 metrics.json,
    # 静态文本若留在 base 里就会过期 ⇒ 汇总时必须用本函数刷新。
    return {
        'scope': ('S4-P 管道: P5a/P5b/P5c/P5d/P2′/P3①; ViT 臂已于跑前整臂撤出 '
                  '(SPEC §0.1 修订 9), 本代只跑文本 64 块; 不训练任何底座权重, '
                  '不产生任何能力指标'),
        'discipline': [
            '阈值全部写进源码常量 (静态可扫), 跑前写死, 事后不得放宽',
            '判据一律给 块数 / 正号数 / 中位增量 / p',
            '置换 10^4 次 (最小可达 p ≈ 1e-4), 报告须同时给 p 与置换次数',
            '探针 lambda 由训练样本内 5 折 CV 在写死网格上选出, 测试样本全程不参与',
            'R² 一律在留出集上算, 且只读正文里的真实 token (pad 位不参与)',
            '本代不训练任何底座权重, 不产生任何能力指标'],
        'limits': [
            '管道内容写在第 0 层 ⇒ 与 v81 的「层边界注入」口径不同, 不可直接比大小',
            '正文被截断到 SEQ−K ⇒ 与 v81 的 128 token 正文口径不同',
            'arm0 口径不同 (管道开、内容为 0 vs v81 的管道不可见)',
            '注意力在第 0 层测量 (写入层), 深层的注意力已不对内容敏感 (pipe_probe3 实测)',
            '共享空间仍是 256 维 ⇒ 源侧信息上限不变 (BERT 无损、Qwen 29%、ViT 33%)',
            'ViT 臂已于跑前整臂撤出 (SPEC §0.1 修订 9), 本代只跑文本 64 块',
            '本代不训练任何底座权重, 不产生任何能力指标, 不产生 A1 的总判定']}


def run_formal(lam: float, smoke: bool = False, fresh: bool = False,
               only: Optional[dict] = None) -> dict:
    """only: {方向: set(源层)} —— **分批跑批**用 (每批的 pilot 仍用**全量**源层算,
    保证跨批次的探针 λ 口径一致); None = 全跑。"""
    t0 = time.time()
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(min(16, os.cpu_count() or 8))

    seeds = (0,) if smoke else SEEDS
    max_blocks = 1 if smoke else None
    n_done_blocks = 0

    fetch_models()
    texts = fetch_texts(N_TEXT)
    labels_text = np.array([int(t["label"]) for t in texts])
    imgs, vit_meta = fetch_vit_images()
    labels_vit = np.array([int(m["label"]) for m in vit_meta["items"]])
    proj_cache = {s: MEAS.projections(s) for s in seeds}
    print("[corpus] ag_news %d 条 / ViT %d 张" % (len(texts), len(imgs)), flush=True)

    src_cache = {}
    for name, layers in (("Qwen", L_Q), ("BERT", L_B)):
        src_cache[name] = capture_source(name, layers, texts, None)
        print("[source] %s: %d 个 (层,槽) 组合" % (name, len(src_cache[name])), flush=True)

    rows: List[dict] = []
    att_recs: List[dict] = []
    lam_info: Dict[str, dict] = {}
    layout_info: Dict[str, dict] = {}
    res_dir = _PKG / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    ckpt = res_dir / ("blocks_smoke.jsonl" if smoke else "blocks.jsonl")
    att_ckpt = res_dir / ("attention_smoke.jsonl" if smoke else "attention.jsonl")
    done: Dict[Tuple, dict] = {}
    att_done: Dict[Tuple, dict] = {}
    for path, store, keyf in ((ckpt, done, "blocks"), (att_ckpt, att_done, "att")):
        if fresh and path.exists():
            path.unlink()
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if keyf == "blocks":
                        store[(r["direction"], int(r["source_layer"]), int(r["target_layer"]),
                               r["arm"], int(r["seed"]))] = r
                    else:
                        store[(r["direction"], int(r["source_layer"]), r["arm"],
                               int(r["seed"]))] = r
            print("[resume] %s 已有 %d 条" % (path.name, len(store)), flush=True)
    ck = ckpt.open("a", encoding="utf-8")
    ack = att_ckpt.open("a", encoding="utf-8")

    for dname, s_name, t_name, s_layers, t_layers in DIRECTIONS:
        s_layers_all = s_layers       # pilot 用**全量**源层 (跨批次口径一致)
        if only:
            s_layers = tuple(l for l in s_layers if l in set(only.get(dname, ())))
            if not s_layers:
                continue
        is_vit = (t_name == "ViT")
        n_done_blocks = 0        # 冒烟模式按方向各跑 1 个块
        train_idx, test_idx = split_idx(is_vit, vit_meta)
        bb = load_target(t_name, eager=True)
        layout = build_layout(bb, is_vit, texts, imgs)
        h0 = capture_layer_input(bb, layout, INJECT_LAYER)
        bn = body_norms(h0, layout)
        dtype = h0.dtype
        layout_info[dname] = {
            "n_samples": int(layout["n"]),
            "pipe_positions": K_PIPE,
            "body_positions": int(layout["n_body"]),
            "n_truncated_to_txt": (int((layout["take"] == TXT).sum()) if not is_vit else 0),
            "query_hi_rule": ("K_PIPE + take[i]" if not is_vit else "n_seq"),
        }

        pilot_blocks = _S3._pilot_blocks(s_layers_all, t_layers, s_name, t_name)
        pl = pilot_probe_lambda(bb, layout, pilot_blocks, s_name, t_name, src_cache,
                                proj_cache[seeds[0]], train_idx, lam, bn, dtype)
        pl.update({"pilot_blocks": [[int(a), int(b)] for a, b in pilot_blocks],
                   "grid": list(PROBE_LAM_GRID), "n_folds": PROBE_CV_FOLDS,
                   "pilot_seed": int(seeds[0])})
        lam_info[dname] = pl
        probe_lam = pl["probe_lambda"]
        print("[pilot] %s: probe-lambda=%g (CV R2=%.4f)"
              % (dname, probe_lam, pl["cv_r2"]), flush=True)

        # arm0: 管道开、内容为 0 (delta 全零 ⇒ 不装 hook) —— 布局与其它臂完全相同
        reads0, fins0, attn0 = run_forward(bb, layout, None, True, t_layers, [ATT_LAYER])
        base_rec0 = att_record(dname, 0, "arm0_pipe_zero", 0,
                                attn0[ATT_LAYER][0], attn0[ATT_LAYER][1])
        # arm0 的管道内容恒为 0 ⇒ 注意力与种子无关, 但**逐种子各记一份**:
        # 臂间配对置换要靠「同块同样多的观测」成立, 只记种子 0 会把 arm1 的另外
        # 4 个种子在配对时丢掉。
        for ly in s_layers:
            for seed in seeds:
                key = (dname, int(ly), "arm0_pipe_zero", int(seed))
                if key not in att_done:
                    rec = dict(base_rec0)
                    rec["source_layer"] = int(ly)
                    rec["seed"] = int(seed)
                    att_recs.append(rec)
                    att_done[key] = rec
                    ack.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    ack.flush()
                else:
                    att_recs.append(att_done[key])

        for ly in s_layers:
            for j in t_layers:
                if s_name == t_name and ly == j:
                    continue
                tgt_q = src_cache[s_name][(ly, "Q")]
                tgt_o = src_cache[s_name][(ly, "O")]
                r2 = masked_r2_multi(reads0[j], layout["body_mask"],
                                     {"q": tgt_q, "o": tgt_o}, probe_lam,
                                     train_idx, test_idx)
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
                        "inject_lambda": float(lam)}
                    rows.append(row)
                    done[key] = row
                    ck.write(json.dumps(row, ensure_ascii=False) + "\n")
                    ck.flush()

        for seed in seeds:
            p_a = proj_cache[seed][s_name]
            p_b = proj_cache[seed][t_name]
            for ly in s_layers:
                if max_blocks and n_done_blocks >= max_blocks:
                    break
                zq = src_cache[s_name][(ly, "Q")] @ p_a.t()
                zo = src_cache[s_name][(ly, "O")] @ p_a.t()
                g = torch.Generator().manual_seed(_seed_of("arm3", seed, dname, ly))
                u = torch.randn(zq.shape, generator=g)
                u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                for arm, zz in (("arm1_slot_Q", zq),
                                ("arm2_wrong_slot_O", zo),
                                ("arm3_random_dir", u)):
                    akey = (dname, int(ly), arm, int(seed))
                    bkey = (dname, int(ly), int(0), arm, int(seed))
                    if akey in att_done and all(
                            (dname, int(ly), int(j), arm, int(seed)) in done
                            for j in t_layers if not (s_name == t_name and ly == j)):
                        att_recs.append(att_done[akey])
                        for j in t_layers:
                            if s_name == t_name and ly == j:
                                continue
                            rows.append(done[(dname, int(ly), int(j), arm, int(seed))])
                        continue
                    delta = make_pipe_delta(zz, lam, bn, p_b, ARM_CHANNEL[arm],
                                            layout["n_seq"], dtype)
                    reads, fins, attn = run_forward(bb, layout, delta, True,
                                                   t_layers, [ATT_LAYER])
                    rel = rel_final_body(fins, fins0, layout)
                    rec = att_record(dname, int(ly), arm, int(seed),
                                     attn[ATT_LAYER][0], attn[ATT_LAYER][1])
                    rec["rel"] = rel
                    att_recs.append(rec)
                    att_done[akey] = rec
                    ack.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    ack.flush()
                    for j in t_layers:
                        if s_name == t_name and ly == j:
                            continue
                        tgt_q = src_cache[s_name][(ly, "Q")]
                        tgt_o = src_cache[s_name][(ly, "O")]
                        r2 = masked_r2_multi(reads[j], layout["body_mask"],
                                             {"q": tgt_q, "o": tgt_o}, probe_lam,
                                             train_idx, test_idx)
                        r2p = pooled_r2_multi(reads[j], layout["body_mask"],
                                              {"q": tgt_q, "o": tgt_o}, probe_lam,
                                              train_idx, test_idx)
                        key = (dname, int(ly), int(j), arm, int(seed))
                        if key in done:
                            rows.append(done[key])
                            continue
                        row = {
                            "direction": dname, "source_model": s_name,
                            "target_model": t_name, "source_layer": int(ly),
                            "target_layer": int(j), "arm": arm, "seed": int(seed),
                            "r2_q": r2["q"], "r2_o": r2["o"],
                            "r2_q_pooled_aux": r2p["q"], "r2_o_pooled_aux": r2p["o"],
                            "rel_perturb_final": rel, "probe_lam": float(probe_lam),
                            "inject_lambda": float(lam)}
                        rows.append(row)
                        done[key] = row
                        ck.write(json.dumps(row, ensure_ascii=False) + "\n")
                        ck.flush()
                    del delta, reads, fins
                n_done_blocks += 1
                print("[block] %s ly=%d seed=%d 完成 (累计 %d 行, %.0fs)"
                      % (dname, ly, seed, len(rows), time.time() - t0), flush=True)
        del bb, layout, h0, reads0, fins0
        gc.collect()

    ck.close()
    ack.close()
    crit = evaluate_criteria(rows, att_recs)
    prov = {k: (sha256_of(v) if os.path.exists(v) else None)
            for k, v in PROVENANCE_FILES.items()}
    res = {
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
                "PROBE_LAM_GRID": list(PROBE_LAM_GRID),
                "PROBE_CV_FOLDS": PROBE_CV_FOLDS,
                "TEXT_SPLIT_TRAIN": TEXT_SPLIT_TRAIN, "VIT_SPLIT_TRAIN": VIT_SPLIT_TRAIN,
                "DROP_FIRST_TOKEN": DROP_FIRST_TOKEN},
            "probe_lambda_by_direction": lam_info,
            "layout_by_direction": layout_info,
            "corpus": {
                "text": "ag_news train 128 条 (缓存 agnews_128.json)",
                "text_label_dist": np.bincount(labels_text, minlength=4).tolist(),
                "vit": "food101 4 类 x 16 张 (类别均衡)",
                "vit_classes": vit_meta["classes"],
                "vit_split": "按类别分层: 每类 8 张 train-half / 8 张 test-half "
                             "(修 v81 §4.6 的 train/test 类别不相交缺陷)",
                "vit_label_dist": np.bincount(labels_vit, minlength=4).tolist()},
            "provenance_sha256": prov,
            "seed": 0, "device": "cpu", "dtype": "float32",
            "smoke": bool(smoke),
            "batch_selection": ({k: sorted(v) for k, v in only.items()} if only else "all"),
            **_protocol_texts(),
        },
        "blocks": rows,
        "attention_summary": [
            {k: v for k, v in r.items() if not k.startswith("per_sample")}
            for r in att_recs],
        "runtime_sec": round(time.time() - t0, 1),
    }
    return res



def aggregate() -> dict:
    '''**只汇总**: 从 results/blocks.jsonl + results/attention.jsonl 重算判据, 不跑模型。

    分批跑批 (--block-ids) 时每批各自写一次 metrics.json, 各自只含**启动时**读到的
    checkpoint => 任何单批的 metrics.json 都不完整。判据函数 (evaluate_criteria) 与
    正式跑批**完全相同**, 所以汇总 = 读全量 checkpoint 重跑一次判据。
    '''
    res_dir = _PKG / 'results'
    rows = []
    att = []
    seen: Dict[Tuple, bool] = {}
    for name, store in (('blocks.jsonl', rows), ('attention.jsonl', att)):
        p = res_dir / name
        if not p.exists():
            continue
        with p.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if name == 'blocks.jsonl':
                    k = (rec['direction'], int(rec['source_layer']),
                         int(rec['target_layer']), rec['arm'], int(rec['seed']))
                else:
                    k = (rec['direction'], int(rec['source_layer']),
                         rec['arm'], int(rec['seed']))
                if k in seen:
                    continue        # 断点续跑可能重写同一 key => 只留第一条
                seen[k] = True
                store.append(rec)
    crit = evaluate_criteria(rows, att)
    mpath = res_dir / 'metrics.json'
    base = json.loads(mpath.read_text(encoding='utf-8')) if mpath.exists() else {}
    # 合并分批跑批留下的 **protocol 片段** (results/part_*.json):
    # 每批只带自己那个方向的 pilot (探针 lambda / CV R2) 与布局信息, 汇总时补齐。
    for extra in sorted(res_dir.glob('part_*.json')):
        try:
            frag = json.loads(extra.read_text(encoding='utf-8'))
        except Exception:
            continue
        fproto = frag.get('protocol') or {}
        for key in ('probe_lambda_by_direction', 'layout_by_direction'):
            for dname, val in (fproto.get(key) or {}).items():
                base.setdefault('protocol', {}).setdefault(key, {}).setdefault(dname, val)
    res = dict(base)
    res['verdict'] = crit['verdict']['verdict']
    res['verdict_detail'] = crit['verdict']
    res['criteria'] = {k: v for k, v in crit.items() if k != 'verdict'}
    res['blocks'] = rows
    res['attention_summary'] = [
        {k: v for k, v in r.items() if not k.startswith('per_sample')}
        for r in att]
    res['aggregated'] = True
    res['aggregate_note'] = ('由 --aggregate 从 results/blocks.jsonl + results/attention.jsonl '
                             '重算 (分批跑批后汇总; 判据函数与正式跑批完全相同)')
    res.setdefault('protocol', {}).update(_protocol_texts())
    res['n_blocks_rows'] = len(rows)
    res['n_attention_rows'] = len(att)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", action="store_true", help="S4P-1: lambda 标定")
    ap.add_argument("--smoke", action="store_true", help="减网格冒烟 (不产出裁决)")
    ap.add_argument("--fresh", action="store_true", help="丢弃 checkpoint 从头跑")
    ap.add_argument("--lam", type=float, default=None, help="显式指定 lambda (默认读 calib.json)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--aggregate", action="store_true",
                    help="只汇总: 从 checkpoint 重算判据并写 metrics.json (不跑模型)")
    ap.add_argument("--block-ids", default=None,
                    help="分批跑批: 'Qwen->BERT:0,1,2,3;BERT->Qwen:0' (只跑指定方向/源层)")
    args = ap.parse_args()

    if args.calib:
        res = run_calib()
        out = Path(args.out) if args.out else (_PKG / "results" / "calib.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[calib] %s -> lambda*=%s"
              % (res["verdict"], res["selected_lambda"]), flush=True)
        return 0 if res["verdict"] == "S4PIPE_LAMBDA_SELECTED" else 1

    if args.aggregate:
        res = aggregate()
        out = Path(args.out) if args.out else (_PKG / 'results' / 'metrics.json')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
        print('[aggregate] %d 行 blocks / %d 行 attention -> %s'
              % (res.get('n_blocks_rows', 0), res.get('n_attention_rows', 0), out), flush=True)
        print('[verdict] %s  (failed: %s)'
              % (res['verdict'], res['verdict_detail']['failed_criteria']), flush=True)
        return 0 if res['verdict'] == 'S4PIPE_READY' else 1

    lam = args.lam
    if lam is None:
        cal = _PKG / "results" / "calib.json"
        if not cal.exists():
            print("[fatal] 缺少 results/calib.json —— 先跑 --calib, 或用 --lam 指定",
                  flush=True)
            return 2
        calres = json.loads(cal.read_text(encoding="utf-8"))
        if calres.get("selected_lambda") is None:
            print("[fatal] 标定未选出可行 lambda (%s) —— 报回, 不硬选"
                  % calres.get("verdict"), flush=True)
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
        print("[run] 分批: %s" % {k: sorted(v) for k, v in only.items()}, flush=True)
    res = run_formal(lam, smoke=args.smoke, fresh=args.fresh, only=only)
    out = Path(args.out) if args.out else (
        _PKG / "results" / ("metrics_smoke.json" if args.smoke else "metrics.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[verdict] %s  (failed: %s)"
          % (res["verdict"], res["verdict_detail"]["failed_criteria"]), flush=True)
    if args.smoke:
        return 0
    return 0 if res["verdict"] == "S4PIPE_READY" else 1


if __name__ == "__main__":
    sys.exit(main())
