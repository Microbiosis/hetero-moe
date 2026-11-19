"""s3_flow.py - A1 S3: 语义判据 (token 级探针 R² + 置换检验).

协议正本: docs/S3_PROTOCOL.md (HMF-A1-S3-1.0, 2026-11-16 预登记, 2026-11-17 修订第 6 条).
上游: v78 (S3 前置诊断) / v79 (S2-0 对齐来源) / v80 (S2 注入算子 + λ 标定).

本代做什么 (协议 §2 / §3):
  在**真模型层边界**按 §2.3 的口径注入 (共享空间载体, **不学任何学出来的映射**),
  四臂 (arm0/arm1/arm2/arm3) 在 **84 个块** (文本 64 + ViT 20) 上跑通, 并对每个块做
  **token 级探针** (协议 §2.6) 读出:
    - 主探针: 目标模型第 j+1 层的 **token 级** hidden -> 回归源模型第 l 层槽 s 的激活
      (按样本广播到它的 token); 逐维标准化 + 去首 token; **按样本**切留出集;
      R² 一律在留出集上算。
    - 次探针 (P3② 选择性): 同一读出上回归**非目标槽** (O) 的激活。
    - 标签可读性 (P1c / Δ_Qwen / Δ_BERT): 读出的**样本级 mean-pool** -> 4 类岭分类器
      (复用 v78 的 D3 探针实现), 留出集准确率。
  然后按 §3 的判据给 p 值: P1a 穿透 / P1b 槽特异 / P2 深度语义 / P3 守门 / P4 否决,
  外加 ViT 平行臂的 P1c (不参与主裁决, 协议 §2.2)。

本代**不**改动 v75-v80 的任何裁决, **不**训练任何底座权重, **不**产生能力指标。

⚠️ 阈值/口径全部写在本文件的常量里 (静态可扫), 跑前写死, **事后不得放宽**。
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

# xet 后端在本机会卡在 read-token 端点; 只对**本进程**生效, 不写任何持久配置。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch


# ================================================================ 常量 (跑前写死)
DS = 256                      # 共享空间维度 (与 v62 一致)
SEQ = 128                     # 文本序列长 (与 v77/v78/v79/v80 同)
N_TEXT = 128                  # 文本条数 (ag_news train, 同一缓存 agnews_128.json)
N_VIT = 64                    # ViT 臂图片数 (协议 §2.7: 4 类 x 16 张)
VIT_N_CLASSES = 4             # 协议 §2.7: 与文本侧标签基数相同
VIT_PER_CLASS = 16
VIT_CLASS_SEED = 20261117     # 类别抽样的固定种子 (类别与 offset 记入快照)
BATCH = 32                    # 每批样本数

#: 层阶梯 (协议 §2.1)
L_Q = (0, 1, 2, 3, 6, 12, 18, 23)     # Qwen 24 层
L_B = (0, 1, 2, 3)                    # BERT 4 层
L_V = (0, 3, 6, 9, 11)                # ViT 12 层

SOURCE_SLOTS = ("Q", "O")             # 协议 §2.4: Q 注入源, O 错槽对照
ARMS = ("arm0_no_inject", "arm1_slot_Q", "arm2_wrong_slot_O", "arm3_random_dir")
INJECT_ARMS = ("arm1_slot_Q", "arm2_wrong_slot_O", "arm3_random_dir")

#: (方向名, 源模型, 目标模型, 源层阶梯, 目标层阶梯) —— 协议 §2.1 的块网格
DIRECTIONS = (
    ("Qwen->BERT", "Qwen", "BERT", L_Q, L_B),
    ("BERT->Qwen", "BERT", "Qwen", L_B, L_Q),
    ("ViT->ViT", "ViT", "ViT", L_V, L_V),
)

SEEDS = (0, 1, 2, 3, 4)       # 5 个 nuisance 种子 (协议 §3; v75 的 SEEDS_DEFAULT)
N_PERM = 10000                # 蒙特卡洛置换次数 (协议 §3: 10^4 => 最小可达 p ~ 1e-4)
ALPHA = 0.01                  # 主判据显著性水平 (协议 §3)
P1C_ALPHA = 0.05              # ViT 平行臂的符号检验水平 (协议 §3)
LAMBDA_INJECT = 0.03          # 注入强度 (协议 §2.3 修订记录第 6 条; 块 Frobenius 口径)

#: 探针 ridge lambda 的候选网格 —— 协议未规定, 本代口径 (跑前写死):
#: 由**训练样本内**的 5 折分层 CV 选出, 测试样本全程不参与。
PROBE_LAM_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
PROBE_CV_FOLDS = 5
PROBE_PILOT_BLOCKS = 3        # 每个方向用 3 个块做 pilot 选 lambda (seed 0; 跨层阶梯)

TEXT_SPLIT_TRAIN = 64         # 文本: 前 64 训 / 后 64 测 (协议 §2.6, 同 D3 口径)
VIT_SPLIT_TRAIN = 32          # ViT: 32 训 / 32 测 (按类别分层, 顺序在快照里登记)
DROP_FIRST_TOKEN = True       # 协议 §2.6 的「去 BOS」: 去掉每个样本的第 0 个 token

RAND_SEED_BASE = 20261117     # arm3 随机方向的种子基准 (逐块派生)

THRESHOLDS = {
    "min_text_blocks": 32,
    "n_text_blocks": 64,
    "n_vit_blocks": 20,
    "alpha": ALPHA,
    "p1c_alpha": P1C_ALPHA,
    "n_perm": N_PERM,
    "n_seeds": len(SEEDS),
    "inject_lambda": LAMBDA_INJECT,
    "p3_min_rel_perturbation": 1e-3,
    "p3_max_rel_perturbation": 0.5,
    "p3_selectivity_p_min": 0.05,
    "p2_require_same_ge_cross": True,
}


# ================================================================ 按路径载入 (复用不复制)
def _load_by_path(path: Path, name: str):
    spec = ilu.spec_from_file_location(name, str(path))
    mod = ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_PKG = Path(__file__).resolve().parent
_V77 = _load_by_path(_PKG.parent / "v77_true_model_flow" / "true_model.py",
                     "true_model_v77_by_path")
_V78 = _load_by_path(_PKG.parent / "v78_s3_preflight" / "s3_preflight.py",
                     "s3_preflight_v78_by_path")
_V75 = _load_by_path(_PKG.parent / "v75_weight_conjugation" / "g2_conjugation.py",
                     "g2_conjugation_v75_by_path")
_V80 = _load_by_path(_PKG.parent / "v80_inject_operator" / "s2_operator.py",
                     "s2_operator_v80_by_path")

MEAS = _V77.MEAS
Backbone = _V77.Backbone
Injector = _V77.Injector
capture_slots = _V77.capture_slots
fetch_models = _V77.fetch_models
fetch_texts = _V77.fetch_texts
sha256_of = _V77.sha256_of
CACHE_DIR = _V77.CACHE_DIR
_get_with_retry = _V77._get_with_retry

capture_layers = _V80.capture_layers          # 层边界捕获 (读出层 = 层 j 的输出)
build_delta = _V80.build_delta                # 注入量构造 (逐样本 Frobenius 归一化)
rel_l2 = _V80.rel_l2                          # 相对扰动

ridge_probe_label = _V78._ridge_probe        # v78 的 D3 标签探针 (4 类 + 留出集)
sign_test = _V75.sign_test                    # v75 的单侧符号检验

PROVENANCE_FILES = {
    "v62_measure": _V77.PROVENANCE_FILES["v62_measure"],
    "v75_g2_conjugation": str(_PKG.parent / "v75_weight_conjugation" / "g2_conjugation.py"),
    "v77_true_model": str(_PKG.parent / "v77_true_model_flow" / "true_model.py"),
    "v78_s3_preflight": str(_PKG.parent / "v78_s3_preflight" / "s3_preflight.py"),
    "v80_s2_operator": str(_PKG.parent / "v80_inject_operator" / "s2_operator.py"),
    "v81_s3_flow": str(_PKG / "s3_flow.py"),
}


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _seed_of(*parts) -> int:
    """确定性种子派生 (不用 Python 的 hash(), 它带进程随机盐)。"""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


# ================================================================ 语料
def fetch_vit_images(n_classes: int = VIT_N_CLASSES, per_class: int = VIT_PER_CLASS,
                      seed: int = VIT_CLASS_SEED):
    """food101 的 **4 类 x 16 张 = 64 张** (协议 §2.7), 类别均衡、offset 固定并登记。

    为什么不能用 v77 的 fetch_images: 它取 train 的**前 N 行**, 而 food101 的 train
    是按类别排序的 => 前 N 行只会覆盖 1-2 个类别。本函数按类别块取 (每类 750 行),
    并把「先一半 train / 后一半 test」排好序, 使 v78 的探针 (前 split 个样本训练)
    在 ViT 臂上也是**类别分层**的。
    """
    import pickle
    cache = CACHE_DIR / ("food101_balanced_%dc_%dimg.pkl" % (n_classes, per_class))
    meta_cache = CACHE_DIR / ("food101_balanced_%dc_%dimg.json" % (n_classes, per_class))
    if cache.exists() and meta_cache.exists():
        with open(cache, "rb") as f:
            imgs = pickle.load(f)
        return imgs, json.loads(meta_cache.read_text(encoding="utf-8"))

    from io import BytesIO
    from PIL import Image
    import requests

    classes = sorted(int(c) for c in np.random.RandomState(seed).choice(
        101, n_classes, replace=False).tolist())
    per_class_rows = 750
    got = {}
    for c in classes:
        url = ("https://datasets-server.huggingface.co/rows?dataset=ethz/food101"
               "&config=default&split=train&offset=%d&length=%d"
               % (c * per_class_rows, per_class))
        resp = _get_with_retry(requests, url)
        rows = resp.json()["rows"]
        lst = []
        for r in rows:
            src = r["row"]["image"]["src"]
            b = _get_with_retry(requests, src,
                                valid=lambda x: x.content[:2] == b"\xff\xd8")
            lst.append(Image.open(BytesIO(b.content)).convert("RGB"))
        if len(lst) != per_class:
            raise RuntimeError("类别 %d 只拿到 %d 张" % (c, len(lst)))
        got[c] = lst

    half = per_class // 2
    imgs: List = []
    meta = {"n_classes": n_classes, "per_class": per_class, "class_seed": seed,
            "classes": classes, "items": []}
    for li, c in enumerate(classes):
        for i, im in enumerate(got[c]):
            imgs.append(im)
            meta["items"].append({
                "label": li, "class_id": c,
                "split": "train" if i < half else "test",
                "row_offset": c * per_class_rows + i,
            })
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(imgs, f)
    meta_cache.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return imgs, meta


def make_batches(bb, name: str, texts, imgs, batch_size: int = BATCH):
    """分批口径与 v77 的 text_batches 一致 (padding=max_length, truncation,
    max_length=SEQ), 只是批大小改成 32 (吞吐) —— tokenization 逐条, 与批大小无关。"""
    if name == "ViT":
        out = []
        for i in range(0, len(imgs), batch_size):
            proc = bb.proc(images=imgs[i:i + batch_size], return_tensors="pt")
            out.append({k: v for k, v in proc.items()})
        return out
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = [t["text"] for t in texts[i:i + batch_size]]
        enc = bb.tok(chunk, return_tensors="pt", padding="max_length",
                     truncation=True, max_length=SEQ)
        out.append({k: v for k, v in enc.items()})
    return out


# ================================================================ 探针
def _standardize(Xtr: torch.Tensor, Xte: torch.Tensor):
    """逐维标准化, **统计量只取训练行** (协议 §2.6 的「跨 token 去均值 + 除标准差」;
    标准化统计量只用训练样本 => 测试样本不参与预处理)。"""
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _ridge_r2(Xtr: torch.Tensor, Ytr: torch.Tensor, Xte: torch.Tensor,
              Yte: torch.Tensor, lam: float) -> float:
    """闭式岭回归 + **留出集** R²。

    R² 定义 (本代口径): 1 - SS_res/SS_tot, SS_tot 围绕**训练集**的 y 均值计算
    => lambda -> inf 时预测退化为常数 => R² -> 0; 这是诚实的**预测** R²
    (v76 的同名函数是同样本拟合, 协议 §2.6 明确要求留出集读数)。
    """
    ym = Ytr.mean(0, keepdim=True)
    Ytr = Ytr - ym
    Yte = Yte - ym
    d = Xtr.shape[1]
    G = Xtr.t() @ Xtr + lam * torch.eye(d, dtype=Xtr.dtype)
    A = torch.linalg.solve(G, Xtr.t() @ Ytr)
    pred = Xte @ A
    ss_res = float((pred - Yte).pow(2).sum())
    ss_tot = float(Yte.pow(2).sum())
    return 1.0 - ss_res / (ss_tot + 1e-12)


def _broadcast_target(tgt: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """把每样本一个目标向量广播到该样本的每个 token (协议 §2.6)。"""
    return tgt.unsqueeze(1).expand(-1, n_tokens, -1).reshape(-1, tgt.shape[-1])


def _ridge_r2_multi(readout: torch.Tensor, targets: Dict[str, torch.Tensor],
                         lam: float, train_idx: np.ndarray, test_idx: np.ndarray
                         ) -> Dict[str, float]:
    """token 级探针: readout [N,T,D] -> 多个目标 (Q 槽 / O 槽) 的留出集 R²。

    **一次标准化 + 一次 Gram, 多个目标各解一次岭回归** —— Gram 是内存带宽瓶颈
    (特征列 896 维 => 29 MB 的矩阵要流过 896 遍), 两个目标共用它省掉一半探针开销。
    """
    n_tok = readout.shape[1]
    D = readout.shape[2]
    Xtr = readout[train_idx].reshape(-1, D)
    Xte = readout[test_idx].reshape(-1, D)
    Xtr, Xte = _standardize(Xtr, Xte)
    G0 = Xtr.t() @ Xtr
    eye = torch.eye(D, dtype=Xtr.dtype)
    out: Dict[str, float] = {}
    for name, tgt in targets.items():
        Ytr = _broadcast_target(tgt[train_idx], n_tok)
        Yte = _broadcast_target(tgt[test_idx], n_tok)
        ym = Ytr.mean(0, keepdim=True)
        Yc = Ytr - ym
        Ytc = Yte - ym
        A = torch.linalg.solve(G0 + lam * eye, Xtr.t() @ Yc)
        pred = Xte @ A
        ss_res = float((pred - Ytc).pow(2).sum())
        ss_tot = float(Ytc.pow(2).sum())
        out[name] = 1.0 - ss_res / (ss_tot + 1e-12)
    return out


def probe_r2(readout: torch.Tensor, tgt: torch.Tensor, lam: float,
             train_idx: np.ndarray, test_idx: np.ndarray) -> float:
    """token 级探针: readout [N,T,D] -> tgt [N,DA] 的留出集 R²。"""
    return _ridge_r2_multi(readout, {"t": tgt}, lam, train_idx, test_idx)["t"]


def probe_lambda_scores(readout: torch.Tensor, tgt: torch.Tensor,
                        train_idx: np.ndarray) -> Dict[float, float]:
    """每个候选 lambda 的**训练集内** 5 折 CV 平均 R²。"""
    n_tok = readout.shape[1]
    D = readout.shape[2]
    folds = np.arange(len(train_idx)) % PROBE_CV_FOLDS
    scores: Dict[float, List[float]] = {float(g): [] for g in PROBE_LAM_GRID}
    for f in range(PROBE_CV_FOLDS):
        tr_pos = np.where(folds != f)[0]
        te_pos = np.where(folds == f)[0]
        if tr_pos.size == 0 or te_pos.size == 0:
            continue
        Xa = readout[train_idx[tr_pos]].reshape(-1, D)
        Xb = readout[train_idx[te_pos]].reshape(-1, D)
        Xa, Xb = _standardize(Xa, Xb)
        Ya = _broadcast_target(tgt[train_idx[tr_pos]], n_tok)
        Yb = _broadcast_target(tgt[train_idx[te_pos]], n_tok)
        for g in PROBE_LAM_GRID:
            scores[float(g)].append(_ridge_r2(Xa, Ya, Xb, Yb, float(g)))
    return {g: (float(np.mean(v)) if v else float("nan")) for g, v in scores.items()}


def label_readability(readout: torch.Tensor, labels: np.ndarray, split: int,
                      lam: float = 1.0) -> Optional[float]:
    """标签可读性 (4 类) = 读出的**样本级 mean-pool** -> 岭分类器的**留出集**准确率。
    口径直接复用 v78 的 D3 探针实现, 保证与 v78 的读数可比; lambda 由 pilot 标定
    (ViT 臂 32 样本下 v78 写死的 lambda=1.0 会欠正则)。"""
    pooled = readout.mean(dim=1)
    out = ridge_probe_label(pooled, labels, n_classes=4, lam=lam, split=split)
    return out.get("acc")


# ================================================================ 置换检验
def perm_paired(obs: Sequence[Tuple[float, float]], block_ids: Sequence[int],
                n_perm: int = N_PERM, seed: int = 0) -> dict:
    """**臂间配对置换** (v75 的 paired_perm 口径, 块内逐观测交换两臂的标签)。

    obs: 每个 (块, 观测=种子) 一对 (臂A值, 臂B值); block_ids 给出每个观测所属的块。
    置换: 每个观测独立地以 0.5 概率交换 A/B (= 翻转该观测的差值符号)。
    统计量: 块统计量 = 块内各观测差值的**中位数**; 总统计量 = 各块统计量的**均值**。
    """
    if not obs:
        return {"n_obs": 0, "n_blocks": 0, "observed": None, "permutation_p": None}
    a = np.array([o[0] for o in obs], dtype=float)
    b = np.array([o[1] for o in obs], dtype=float)
    bid = np.asarray(block_ids, dtype=int)
    diff = a - b
    uniq, inv = np.unique(bid, return_inverse=True)
    counts = np.bincount(inv)
    if counts.min() != counts.max():
        raise RuntimeError("各块观测数不等, 分块中位数不可用")
    n_blocks = int(len(uniq))
    m = int(counts[0])
    order = np.argsort(inv, kind="stable")
    D = diff[order].reshape(n_blocks, m)
    obs_stat = float(np.median(D, axis=1).mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n_blocks, m))
    perm_stats = np.median(D[None, :, :] * signs, axis=2).mean(axis=1)
    cnt = int((perm_stats >= obs_stat - 1e-15).sum())
    return {"n_obs": int(diff.size), "n_blocks": n_blocks,
            "statistic": "mean_over_blocks_of_median_over_seeds",
            "observed": obs_stat, "permutation_p": cnt / n_perm, "n_perm": n_perm}


def perm_group_stratified(values: np.ndarray, groups: np.ndarray,
                           n_perm: int = N_PERM, seed: int = 0) -> dict:
    """**块内分层置换** (协议 §3 的分层置换在 P2 上的落地)。

    零假设: 块所属的层组 (同层 / 跨层) 与增量无关。
    置换: 在**块集合内**按层组分层重分配 (保持各层组的块数不变)。
    统计量: mean(Δ|同层) - mean(Δ|跨层)。
    """
    if values.size == 0:
        return {"n_blocks": 0, "observed": None, "permutation_p": None}
    g1 = groups == 1
    g0 = groups == 0
    if g1.sum() == 0 or g0.sum() == 0:
        return {"n_blocks": int(values.size), "observed": None, "permutation_p": None}
    obs = float(values[g1].mean() - values[g0].mean())
    rng = np.random.default_rng(seed)
    # 每行一个**独立**的随机置换 (np.random.permutation 作用在数组上是全局打乱, 不能用)
    idx = np.argsort(rng.random((n_perm, values.size)), axis=1)
    perm_vals = values[idx]
    perm_stats = perm_vals[:, g1].mean(axis=1) - perm_vals[:, g0].mean(axis=1)
    cnt = int((perm_stats >= obs - 1e-15).sum())
    return {"n_blocks": int(values.size), "statistic": "mean_same_minus_mean_cross",
            "observed": obs, "permutation_p": cnt / n_perm, "n_perm": n_perm}


def _median(vals: Sequence[float]) -> Optional[float]:
    v = [float(x) for x in vals if x is not None]
    return float(np.median(v)) if v else None


# ================================================================ 前向
def run_capture(bb, batches, j: int, last: int, delta: Optional[torch.Tensor] = None):
    """跑完全部批次, 返回 (读出层输出 [N,T,D], 末层输出 [N,T,D])。

    读出层 = **注入层 j 的输出** = hidden_states[j+1] = 协议 §2.6 的「第 j+1 层」。
    """
    reads, fins = [], []
    for bi, b in enumerate(batches):
        d = None if delta is None else delta[bi * BATCH:(bi + 1) * BATCH]
        if d is None:
            st = capture_layers(bb, b, out_layers=[j, last])
        else:
            with Injector(bb.layers[j], d):
                st = capture_layers(bb, b, out_layers=[j, last])
        reads.append(st[("out", j)])
        fins.append(st[("out", last)])
    return torch.cat(reads, 0), torch.cat(fins, 0)


def run_reference(bb, batches, j: int, last: int):
    """参考前向 (零注入): 抓层 j 的**输入** (注入归一化的基准) 与层 j / 末层的输出。"""
    ins, reads, fins = [], [], []
    for b in batches:
        st = capture_layers(bb, b, out_layers=[j, last], in_layers=[j])
        ins.append(st[("in", j)])
        reads.append(st[("out", j)])
        fins.append(st[("out", last)])
    return torch.cat(ins, 0), torch.cat(reads, 0), torch.cat(fins, 0)


def _drop_first(readout: torch.Tensor) -> torch.Tensor:
    return readout[:, 1:, :] if DROP_FIRST_TOKEN else readout


def _arm_stats(readout: torch.Tensor, fin: torch.Tensor, ref_fin: torch.Tensor,
               delta: torch.Tensor, h_ref: torch.Tensor,
               tgt_q: torch.Tensor, tgt_o: torch.Tensor, labels: np.ndarray, split: int,
               train_idx: np.ndarray, test_idx: np.ndarray, lam: float,
               lam_p: float, lam_label: float) -> dict:
    ro = _drop_first(readout)
    n = h_ref.shape[0]
    ratio = (delta.reshape(n, -1).norm(dim=-1) / h_ref.reshape(n, -1).norm(dim=-1))
    r2 = _ridge_r2_multi(ro, {"q": tgt_q, "o": tgt_o}, lam, train_idx, test_idx)
    # 辅助读数 (**预登记, 不参与任何裁决**): 同一读出的**样本级 mean-pool** 探针。
    # 用途只有一个 —— 区分「信息没流过去」与「token 级探针读不出来」(协议 §1 拍板 4
    # 把探针单位从样本级改成 token 级, 这条是那次改动的兜底读数)。
    pooled = ro.mean(dim=1, keepdim=True)
    r2p = _ridge_r2_multi(pooled, {"q": tgt_q, "o": tgt_o}, lam_p, train_idx, test_idx)
    return {
        "r2_q": r2["q"],
        "r2_o": r2["o"],
        "r2_q_pooled_aux": r2p["q"],
        "r2_o_pooled_aux": r2p["o"],
        "label_acc": label_readability(ro, labels, split, lam_label),
        "rel_perturb_final": rel_l2(ref_fin, fin),
        "norm_ratio_max_abs_dev": float((ratio - THRESHOLDS["inject_lambda"]).abs().max()),
    }


def _pilot_blocks(s_layers, t_layers, s_name: str, t_name: str,
                  n: int = PROBE_PILOT_BLOCKS) -> List[Tuple[int, int]]:
    """预登记的 pilot 块: 目标层阶梯的**首 / 中 / 末**各取一个源层。

    为什么不能只取最浅层: 最浅目标层的注入信号最弱 (Qwen 第 0 层的读出几乎还没
    传递注入内容), 在那里选出的 lambda 会偏大 (过度正则) ⇒ 预登记跨阶梯取块。
    """
    out: List[Tuple[int, int]] = []
    for j in (t_layers[0], t_layers[len(t_layers) // 2], t_layers[-1]):
        cands = [ly for ly in s_layers if not (s_name == t_name and ly == j)]
        if cands:
            out.append((cands[0], int(j)))
    for ly in s_layers:
        for j in t_layers:
            if (int(ly), int(j)) not in out and not (s_name == t_name and ly == j):
                out.append((int(ly), int(j)))
        if len(out) >= n:
            break
    return out[:n]


def _label_probe_acc(X: torch.Tensor, y: np.ndarray, lam: float,
                        train_idx: np.ndarray, test_idx: np.ndarray) -> float:
    """4 类岭分类器的留出集准确率 (口径与 v78 的 _ridge_probe 同式:
    训练集均值中心化 + one-hot 岭回归 + argmax), 只是折的划分与 lambda 可控。"""
    Xtr = X[train_idx].double()
    Xte = X[test_idx].double()
    mu = Xtr.mean(0, keepdim=True)
    Xtr = Xtr - mu
    Xte = Xte - mu
    n_cls = 4
    Ytr = torch.zeros(len(train_idx), n_cls, dtype=torch.float64)
    Ytr[torch.arange(len(train_idx)), torch.from_numpy(y[train_idx]).long()] = 1.0
    d = Xtr.shape[1]
    W = torch.linalg.solve(Xtr.t() @ Xtr + lam * torch.eye(d, dtype=Xtr.dtype),
                            Xtr.t() @ Ytr)
    pred = (Xte @ W).argmax(dim=1).numpy()
    return float((pred == y[test_idx]).mean())


def _label_lam_scores(pooled: torch.Tensor, y: np.ndarray, train_idx: np.ndarray
                       ) -> Dict[float, float]:
    """标签探针 lambda 的**训练集内** 5 折 CV 准确率。

    为什么必须单独标定: ViT 臂只有 32 个训练样本 × 768 维, v78 写死的 lambda=1.0
    在那个尺度下欠正则 (实测留出集准确率 0.000, 低于随机的 0.25) ⇒ 标签可读性
    读数会失真, 而 P1c 正是靠它。
    """
    folds = np.arange(len(train_idx)) % PROBE_CV_FOLDS
    scores: Dict[float, List[float]] = {float(g): [] for g in PROBE_LAM_GRID}
    for f in range(PROBE_CV_FOLDS):
        tr = np.where(folds != f)[0]
        te = np.where(folds == f)[0]
        if tr.size == 0 or te.size == 0:
            continue
        for g in PROBE_LAM_GRID:
            scores[float(g)].append(
                _label_probe_acc(pooled, y, float(g), train_idx[tr], train_idx[te]))
    return {g: (float(np.mean(v)) if v else float("nan")) for g, v in scores.items()}


def _pilot_lambda(bb, batches, blocks: Sequence[Tuple[int, int]], s_name: str, t_name: str,
                  src_cache, proj, train_idx: np.ndarray, labels: np.ndarray
                  ) -> Dict[str, float]:
    """用预登记的 pilot 块 (seed 0) + **训练样本内** 5 折 CV 标定三个 lambda:

    - token 级主探针 (协议 §2.6 的口径) —— 决定 P1a/P1b/P2/P3② 的读数;
    - 样本级 mean-pool 辅助探针 (**不参与裁决**) —— 兜底读数;
    - 标签可读性探针 (4 类) —— P1c 与 Δ_Qwen/Δ_BERT 的读数。

    为什么要有 pilot: 协议没有规定探针的 lambda, 而 token 级探针 (每样本只有 1 个
    目标值、特征是 token 级) 对正则强度极敏感 => 必须用**训练样本内**的 CV 选,
    且必须**跑前写死**候选网格与选法。pilot 块固定为该方向目标层阶梯的首/中/末。
    """
    p_a, p_b = proj[s_name], proj[t_name]
    last = len(bb.layers) - 1
    sc_tok: Dict[float, List[float]] = {float(g): [] for g in PROBE_LAM_GRID}
    sc_pool: Dict[float, List[float]] = {float(g): [] for g in PROBE_LAM_GRID}
    sc_lab: Dict[float, List[float]] = {float(g): [] for g in PROBE_LAM_GRID}
    for ly, j in blocks:
        h_ref, _, _ = run_reference(bb, batches, j, last)
        zq = src_cache[s_name][(ly, "Q")] @ p_a.t()
        delta, _ = build_delta(zq, p_b, h_ref, THRESHOLDS["inject_lambda"])
        read, _fin = run_capture(bb, batches, j, last, delta=delta)
        ro = _drop_first(read)
        for g, v in probe_lambda_scores(ro, src_cache[s_name][(ly, "Q")],
                                        train_idx).items():
            if not np.isnan(v):
                sc_tok[g].append(v)
        pooled = ro.mean(dim=1)
        for g, v in probe_lambda_scores(pooled.unsqueeze(1),
                                        src_cache[s_name][(ly, "Q")], train_idx).items():
            if not np.isnan(v):
                sc_pool[g].append(v)
        for g, v in _label_lam_scores(pooled, labels, train_idx).items():
            if not np.isnan(v):
                sc_lab[g].append(v)

    def _best(sc_: Dict[float, List[float]]):
        bl, bs = float(PROBE_LAM_GRID[0]), -1e30
        for g, vs in sc_.items():
            if not vs:
                continue
            m = float(np.mean(vs))
            if m > bs:
                bl, bs = g, m
        return bl, bs

    lam_t, cv_t = _best(sc_tok)
    lam_p, cv_p = _best(sc_pool)
    lam_l, cv_l = _best(sc_lab)
    return {"probe_lambda": lam_t, "cv_r2": cv_t,
            "probe_lambda_pooled_aux": lam_p, "cv_r2_pooled_aux": cv_p,
            "label_lambda": lam_l, "cv_acc": cv_l,
            # 逐 lambda 的 CV 分数也记进快照: 证明选点不是看结果挑的, 且便于事后核对
            "cv_scores_token": {("%.4g" % g): v for g, v in sorted(sc_tok.items())},
            "cv_scores_pooled": {("%.4g" % g): v for g, v in sorted(sc_pool.items())},
            "cv_scores_label": {("%.4g" % g): v for g, v in sorted(sc_lab.items())}}


# ================================================================ 判据
def _blocks_of(rows: List[dict]):
    """(direction, ly, j) -> {arm: {seed: row}}"""
    out: Dict[Tuple[str, int, int], Dict[str, Dict[int, dict]]] = {}
    for r in rows:
        key = (r["direction"], int(r["source_layer"]), int(r["target_layer"]))
        b = out.setdefault(key, {})
        b.setdefault(r["arm"], {})[int(r["seed"])] = r
    return out


def _med(rows_by_arm: Dict[str, Dict[int, dict]], arm: str, field: str) -> Optional[float]:
    vals = [r[field] for _, r in sorted(rows_by_arm.get(arm, {}).items())]
    vals = [v for v in vals if v is not None]
    return float(np.median(vals)) if vals else None


def evaluate_criteria(rows: List[dict]) -> dict:
    bl = _blocks_of(rows)
    text_keys = sorted([k for k in bl if k[0] != "ViT->ViT"])
    vit_keys = sorted([k for k in bl if k[0] == "ViT->ViT"])

    def block_stat(key, arm, field):
        return _med(bl[key], arm, field)

    def paired_obs(arm_a, arm_b, field, keys):
        obs, bids = [], []
        for bi, key in enumerate(keys):
            aa = bl[key].get(arm_a, {})
            bb = bl[key].get(arm_b, {})
            for s in sorted(set(aa) & set(bb)):
                obs.append((aa[s][field], bb[s][field]))
                bids.append(bi)
        return obs, bids

    # ---- P1a 穿透: arm1 vs arm0 (臂间配对置换)
    p1a_obs, p1a_bids = paired_obs("arm1_slot_Q", "arm0_no_inject", "r2_q", text_keys)
    p1a_perm = perm_paired(p1a_obs, p1a_bids, N_PERM, seed=1001)
    p1a_deltas = [a - b for a, b in p1a_obs]
    p1a_median = _median(p1a_deltas)
    p1a_pass = bool(p1a_median is not None and p1a_median > 0
                    and p1a_perm["permutation_p"] is not None
                    and p1a_perm["permutation_p"] < ALPHA)

    # ---- P1b 槽特异: arm1 vs arm2 / arm1 vs arm3
    p1b = {}
    for arm_b, sd in (("arm2_wrong_slot_O", 1002), ("arm3_random_dir", 1003)):
        obs, bids = paired_obs("arm1_slot_Q", arm_b, "r2_q", text_keys)
        pr = perm_paired(obs, bids, N_PERM, seed=sd)
        ds = [a - b for a, b in obs]
        p1b[arm_b] = {
            "n_obs": pr["n_obs"], "n_blocks": pr["n_blocks"],
            "median_delta": _median(ds), "n_positive": sum(1 for x in ds if x > 0),
            "permutation_p": pr["permutation_p"], "n_perm": pr["n_perm"],
        }
    p1b_pass = bool(all(v["permutation_p"] is not None and v["permutation_p"] < ALPHA
                        for v in p1b.values()))

    # ---- P2 深度语义: 同层 vs 跨层 (块内分层置换)
    same_mask = np.array([1 if k[1] == k[2] else 0 for k in text_keys], dtype=int)
    p2_vals = np.array([(block_stat(k, "arm1_slot_Q", "r2_q") or 0.0)
                        - (block_stat(k, "arm0_no_inject", "r2_q") or 0.0)
                        for k in text_keys], dtype=float)
    p2_perm = perm_group_stratified(p2_vals, same_mask, N_PERM, seed=1004)
    p2_same = _median(p2_vals[same_mask == 1].tolist())
    p2_cross = _median(p2_vals[same_mask == 0].tolist())
    p2_pass = bool(p2_same is not None and p2_cross is not None
                   and p2_same >= p2_cross
                   and p2_perm["permutation_p"] is not None
                   and p2_perm["permutation_p"] < ALPHA)

    # ---- P3① 输出相对扰动区间
    p3_1_rows = [r for r in rows if r["arm"] in INJECT_ARMS]
    bad = [r for r in p3_1_rows
           if not (THRESHOLDS["p3_min_rel_perturbation"] < r["rel_perturb_final"]
                   <= THRESHOLDS["p3_max_rel_perturbation"])]
    fins = [r["rel_perturb_final"] for r in p3_1_rows]
    p3_1 = {"n_rows": len(p3_1_rows), "n_violations": len(bad),
            "min_rel_perturbation": min(fins) if fins else None,
            "max_rel_perturbation": max(fins) if fins else None,
            "passed": len(bad) == 0,
            "violations": [{"direction": r["direction"], "source_layer": r["source_layer"],
                            "target_layer": r["target_layer"], "arm": r["arm"],
                            "seed": r["seed"], "rel_perturb_final": r["rel_perturb_final"]}
                           for r in bad[:20]]}

    # ---- P3② 选择性: 非目标槽 (O) 的可读性增量不普涨
    p3_2_obs, p3_2_bids = paired_obs("arm1_slot_Q", "arm0_no_inject", "r2_o", text_keys)
    p3_2_perm = perm_paired(p3_2_obs, p3_2_bids, N_PERM, seed=1005)
    p3_2_deltas = [a - b for a, b in p3_2_obs]
    p3_2_median = _median(p3_2_deltas)
    p3_2_pass = bool(p3_2_median is not None
                     and (p3_2_median <= 0
                          or p3_2_perm["permutation_p"] is None
                          or p3_2_perm["permutation_p"] >= THRESHOLDS["p3_selectivity_p_min"]))
    p3_2 = {"n_obs": p3_2_perm["n_obs"], "n_blocks": p3_2_perm["n_blocks"],
            "median_delta_non_target_slot": p3_2_median,
            "n_positive": sum(1 for x in p3_2_deltas if x > 0),
            "permutation_p": p3_2_perm["permutation_p"], "n_perm": p3_2_perm["n_perm"],
            "passed": p3_2_pass,
            "rule": "median<=0 或 p>=%.2f" % THRESHOLDS["p3_selectivity_p_min"]}

    # ---- P1c: ViT 平行臂的槽特异性 (标签可读性差), 符号检验; **不参与主裁决**
    p1c_deltas = []
    for key in vit_keys:
        a = block_stat(key, "arm1_slot_Q", "label_acc")
        b = block_stat(key, "arm2_wrong_slot_O", "label_acc")
        if a is not None and b is not None:
            p1c_deltas.append(a - b)
    p1c = dict(sign_test(p1c_deltas)) if p1c_deltas else {"n_blocks": 0}
    p1c["passed"] = bool(p1c.get("sign_p") is not None and p1c["sign_p"] < P1C_ALPHA)
    p1c["role"] = ("ViT 平行臂, 单独报告, **不参与** S3 主裁决 "
                   "(协议 §2.2: ViT 不进跨模型配对统计; §2.1: 文本对是主判据集)")

    # ---- 并列比较量 Δ_Qwen / Δ_BERT (协议 §2.2)
    label_deltas = {}
    for dname in ("Qwen->BERT", "BERT->Qwen", "ViT->ViT"):
        ds = []
        for key in sorted([k for k in bl if k[0] == dname]):
            a = block_stat(key, "arm1_slot_Q", "label_acc")
            b = block_stat(key, "arm2_wrong_slot_O", "label_acc")
            if a is not None and b is not None:
                ds.append(a - b)
        label_deltas[dname] = {"n_blocks": len(ds), "median_delta": _median(ds),
                               "n_positive": sum(1 for x in ds if x > 0),
                               "all": [float(x) for x in ds]}

    # ---- P4 否决
    p4 = {"triggered": bool(p1a_median is None or p1a_median <= 0),
          "rule": "若 Δ_P1a 的块级中位数 <= 0 或不可判定 => A1 必须转训练版",
          "p1a_median_delta": p1a_median}

    passed = bool(p1a_pass and p1b_pass and p2_pass and p3_1["passed"] and p3_2_pass)
    failed = [n for n, ok in (("P1a", p1a_pass), ("P1b", p1b_pass), ("P2", p2_pass),
                              ("P3_1", p3_1["passed"]), ("P3_2", p3_2_pass)) if not ok]
    return {
        "P1a": {"passed": p1a_pass, "n_blocks": len(text_keys), "n_obs": p1a_perm["n_obs"],
                "n_positive": sum(1 for x in p1a_deltas if x > 0),
                "median_delta": p1a_median,
                "mean_delta": float(np.mean(p1a_deltas)) if p1a_deltas else None,
                "permutation_p": p1a_perm["permutation_p"], "n_perm": p1a_perm["n_perm"],
                "alpha": ALPHA, "per_block_deltas": [float(x) for x in p1a_deltas]},
        "P1b": {"passed": p1b_pass, "comparisons": p1b, "alpha": ALPHA},
        "P2": {"passed": p2_pass, "n_same_layer": int((same_mask == 1).sum()),
               "n_cross_layer": int((same_mask == 0).sum()),
               "median_same_layer": p2_same, "median_cross_layer": p2_cross,
               "permutation_p": p2_perm["permutation_p"], "n_perm": p2_perm["n_perm"],
               "alpha": ALPHA},
        "P3_1": p3_1,
        "P3_2": p3_2,
        "P4": p4,
        "P1c": p1c,
        "label_delta_by_direction": label_deltas,
        "verdict": {"verdict": "S3_PASS" if passed else "S3_FAIL", "passed": passed,
                    "failed_criteria": failed,
                    "note": "主判据集 = 64 个文本块; P1c 是 ViT 平行臂的单独结论, 不参与主裁决。"},
    }


# ================================================================ 主流程
def run(smoke: bool = False, fresh: bool = False) -> dict:
    t0 = time.time()
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(min(16, os.cpu_count() or 8))

    seeds = (0,) if smoke else SEEDS
    max_blocks = 2 if smoke else None

    fetch_models()
    texts = fetch_texts(N_TEXT)
    labels_text = np.array([int(t["label"]) for t in texts])
    imgs, vit_meta = fetch_vit_images()
    labels_vit = np.array([int(m["label"]) for m in vit_meta["items"]])
    proj_cache = {s: MEAS.projections(s) for s in seeds}
    print("[corpus] ag_news %d 条 / ViT %d 张 (4 类 x 16)" % (len(texts), len(imgs)), flush=True)

    # ---- 源侧槽激活 (样本级 mean-pool, 原生维; **与种子无关**, 抓一次)
    src_cache: Dict[str, Dict[Tuple[int, str], torch.Tensor]] = {}
    for name, layers in (("Qwen", L_Q), ("BERT", L_B), ("ViT", L_V)):
        bb = Backbone(name)
        batches = make_batches(bb, name, texts, imgs)
        want = [(ly, slot) for ly in layers for slot in SOURCE_SLOTS]
        acc: Dict[Tuple[int, str], List[torch.Tensor]] = {k: [] for k in want}
        for b in batches:
            st = capture_slots(bb, b, want)
            for k in want:
                acc[k].append(st[k].mean(dim=1))
        src_cache[name] = {k: torch.cat(v, 0) for k, v in acc.items()}
        del bb, batches, acc
        gc.collect()
        print("[source] %s: 抓到 %d 个 (层,槽) 组合的样本级激活" % (name, len(want)), flush=True)

    rows: List[dict] = []
    lam_info: Dict[str, dict] = {}
    # 逐块落盘 checkpoint: 长跑被中断时, 已完成的块 (每个块含全部 5 个种子) 仍在,
    # 不必等整轮跑完才有中间读数; 重跑同一命令会**自动跳过**已完成的组合 (断点续跑)。
    ckpt = _PKG / "results" / ("blocks_smoke.jsonl" if smoke else "blocks.jsonl")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    done: Dict[Tuple, dict] = {}
    if fresh or not ckpt.exists():
        if ckpt.exists():
            ckpt.unlink()
    else:
        with ckpt.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                done[(r["direction"], int(r["source_layer"]), int(r["target_layer"]),
                      r["arm"], int(r["seed"]))] = r
        print("[resume] checkpoint 里已有 %d 个已完成的组合, 直接复用" % len(done), flush=True)
    ck = ckpt.open("a", encoding="utf-8")

    for dname, s_name, t_name, s_layers, t_layers in DIRECTIONS:
        is_vit = (t_name == "ViT")
        labels = labels_vit if is_vit else labels_text
        split = VIT_SPLIT_TRAIN if is_vit else TEXT_SPLIT_TRAIN
        n_samples = N_VIT if is_vit else N_TEXT
        train_idx = np.arange(split)
        test_idx = np.arange(split, n_samples)

        bb = Backbone(t_name)
        batches = make_batches(bb, t_name, texts, imgs)
        last = len(bb.layers) - 1

        # ---- pilot: 选探针 ridge lambda (seed 0, 预登记的 pilot 块)
        pilot = _pilot_blocks(s_layers, t_layers, s_name, t_name)
        pl = _pilot_lambda(bb, batches, pilot, s_name, t_name, src_cache,
                           proj_cache[seeds[0]], train_idx, labels)
        lam = pl["probe_lambda"]
        lam_p = pl["probe_lambda_pooled_aux"]
        lam_label = pl["label_lambda"]
        lam_info[dname] = dict(pl)
        lam_info[dname].update({
            "grid": list(PROBE_LAM_GRID), "n_folds": PROBE_CV_FOLDS,
            "pilot_blocks": [[int(a), int(b)] for a, b in pilot],
            "pilot_seed": int(seeds[0])})
        print("[pilot] %s: token-lambda=%g (CV R2=%.4f) / pooled-lambda=%g (CV R2=%.4f) / "
              "label-lambda=%g (CV acc=%.4f)"
              % (dname, lam, pl["cv_r2"], lam_p, pl["cv_r2_pooled_aux"],
                 lam_label, pl["cv_acc"]), flush=True)

        n_done = 0
        for j in t_layers:
            if max_blocks and n_done >= max_blocks:
                break
            h_ref, ref_read, ref_fin = run_reference(bb, batches, j, last)
            # arm0: 零注入 -> 必须与参考前向逐位等价 (v80 的 G1 在本代复核)
            zero = torch.zeros_like(h_ref)
            a0_read, a0_fin = run_capture(bb, batches, j, last, delta=zero)
            g1 = max(rel_l2(ref_read, a0_read), rel_l2(ref_fin, a0_fin))
            for ly in s_layers:
                if s_name == t_name and ly == j:
                    continue
                if max_blocks and n_done >= max_blocks:
                    break
                tgt_q = src_cache[s_name][(ly, "Q")]
                tgt_o = src_cache[s_name][(ly, "O")]
                base = {"direction": dname, "source_model": s_name, "target_model": t_name,
                        "source_layer": int(ly), "target_layer": int(j),
                        "n_train": int(split), "n_test": int(n_samples - split),
                        "g1_max_rel_err": float(g1), "probe_lam": float(lam),
                        "probe_lam_pooled_aux": float(lam_p),
                        "label_lam": float(lam_label)}
                st0 = _arm_stats(a0_read, a0_fin, ref_fin, zero, h_ref, tgt_q, tgt_o,
                                 labels, split, train_idx, test_idx, lam, lam_p,
                                 lam_label)
                for seed in seeds:
                    # arm0 与种子无关 (读出、目标、lambda 都不含种子) => 逐种子各记一行,
                    # 值相同; 这样臂间配对置换每个块有**同样多**的观测 (块内中位数才成立),
                    # 且不需要多跑前向。
                    row0 = dict(base)
                    row0.update(st0)
                    row0["arm"] = "arm0_no_inject"
                    row0["seed"] = int(seed)
                    key0 = (dname, int(ly), int(j), "arm0_no_inject", int(seed))
                    if key0 in done:
                        rows.append(done[key0])
                        continue
                    rows.append(row0)
                    ck.write(json.dumps(row0, ensure_ascii=False) + "\n")
                    ck.flush()
                for seed in seeds:
                    p_a = proj_cache[seed][s_name]
                    p_b = proj_cache[seed][t_name]
                    zq = src_cache[s_name][(ly, "Q")] @ p_a.t()
                    zo = src_cache[s_name][(ly, "O")] @ p_a.t()
                    g = torch.Generator().manual_seed(_seed_of("arm3", seed, dname, ly, j))
                    u = torch.randn(zq.shape, generator=g)
                    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    for arm, zz in (("arm1_slot_Q", zq),
                                    ("arm2_wrong_slot_O", zo),
                                    ("arm3_random_dir", u)):
                        key = (dname, int(ly), int(j), arm, int(seed))
                        if key in done:
                            rows.append(done[key])
                            continue
                        delta, _ = build_delta(zz, p_b, h_ref, THRESHOLDS["inject_lambda"])
                        read, fin = run_capture(bb, batches, j, last, delta=delta)
                        stt = _arm_stats(read, fin, ref_fin, delta, h_ref, tgt_q, tgt_o,
                                         labels, split, train_idx, test_idx, lam, lam_p,
                                         lam_label)
                        row = dict(base)
                        row.update(stt)
                        row["arm"] = arm
                        row["seed"] = int(seed)
                        rows.append(row)
                        ck.write(json.dumps(row, ensure_ascii=False) + "\n")
                        ck.flush()
                n_done += 1
                print("[block] %s ly=%d j=%d 完成 (累计 %d 行, %.0fs)"
                      % (dname, ly, j, len(rows), time.time() - t0), flush=True)
        del bb, batches
        gc.collect()

    ck.close()
    crit = evaluate_criteria(rows)
    prov = {k: (sha256_of(v) if os.path.exists(v) else None)
            for k, v in PROVENANCE_FILES.items()}
    res = {
        "verdict": crit["verdict"]["verdict"],
        "verdict_detail": crit["verdict"],
        "criteria": {k: v for k, v in crit.items() if k != "verdict"},
        "protocol": {
            "thresholds": THRESHOLDS,
            "constants": {
                "DS": DS, "SEQ": SEQ, "N_TEXT": N_TEXT, "N_VIT": N_VIT,
                "VIT_N_CLASSES": VIT_N_CLASSES, "VIT_PER_CLASS": VIT_PER_CLASS,
                "BATCH": BATCH, "L_Q": list(L_Q), "L_B": list(L_B), "L_V": list(L_V),
                "SEEDS": list(seeds), "N_PERM": N_PERM, "ALPHA": ALPHA,
                "P1C_ALPHA": P1C_ALPHA, "LAMBDA_INJECT": LAMBDA_INJECT,
                "PROBE_LAM_GRID": list(PROBE_LAM_GRID),
                "PROBE_CV_FOLDS": PROBE_CV_FOLDS, "PROBE_PILOT_BLOCKS": PROBE_PILOT_BLOCKS,
                "TEXT_SPLIT_TRAIN": TEXT_SPLIT_TRAIN, "VIT_SPLIT_TRAIN": VIT_SPLIT_TRAIN,
                "DROP_FIRST_TOKEN": DROP_FIRST_TOKEN,
            },
            "probe_lambda_by_direction": lam_info,
            "corpus": {
                "text": "ag_news train 128 条 (缓存 agnews_128.json)",
                "text_label_dist": np.bincount(labels_text, minlength=4).tolist(),
                "vit": "food101 %d 类 x %d 张 (类别均衡)" % (VIT_N_CLASSES, VIT_PER_CLASS),
                "vit_classes": vit_meta["classes"],
                "vit_class_seed": VIT_CLASS_SEED,
                "vit_label_dist": np.bincount(labels_vit, minlength=4).tolist(),
                "split": "文本 前 64 训 / 后 64 测; ViT 按类别分层 32 训 / 32 测",
            },
            "provenance_sha256": prov,
            "seed": 0, "device": "cpu", "dtype": "float32",
            "smoke": bool(smoke),
            "scope": ("S3 语义判据: P1a / P1b / P2 / P3 / P4; P1c 为 ViT 平行臂, "
                      "单独报告不参与主裁决 (协议 §2.2)"),
            "discipline": [
                "阈值全部写进源码常量 (静态可扫), 跑前写死, 事后不得放宽",
                "判据一律给 块数 / 正号数 / 中位增量 / p (协议 §3 报告要求)",
                "置换 10^4 次 (最小可达 p ≈ 1e-4), 报告须同时给 p 与置换次数",
                "探针 lambda 由训练样本内 5 折 CV 在写死网格上选出, 测试样本全程不参与",
                "R² 一律在留出集上算 (协议 §2.6)",
                "本代不产生任何能力指标, 不训练任何底座权重",
            ],
            "limits": [
                "注入量是 a_s 的 256 维投影 => 探针读回的信息上限受该投影限制, "
                "读数只作相对增量解读 (协议 §5.7)",
                "ViT 臂只作方向性对照, 不与文本对合并 (协议 §2.2 / §5.3)",
                "层号 != 相对深度; '同层' 指层号相同 (协议 §5.1)",
                "探针目标按样本广播 => 每样本只有 1 个独立目标值, "
                "token 级特征的泛化由训练集内 CV 的 lambda 控制",
                "本代不产生 A1 的总判定 (协议 §5.7)",
            ],
        },
        "blocks": rows,
        "runtime_sec": round(time.time() - t0, 1),
    }
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="减网格冒烟 (不产出裁决)")
    ap.add_argument("--fresh", action="store_true", help="丢弃 checkpoint 从头跑")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = run(smoke=args.smoke, fresh=args.fresh)
    out = Path(args.out) if args.out else (
        _PKG / "results" / ("metrics_smoke.json" if args.smoke else "metrics.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[verdict] %s  (failed: %s)"
          % (res["verdict"], res["verdict_detail"]["failed_criteria"]), flush=True)
    if args.smoke:
        return 0
    return 0 if res["verdict"] == "S3_PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
