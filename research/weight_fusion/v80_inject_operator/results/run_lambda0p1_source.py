"""s2_operator.py - A1 S2: 注入算子下沉到真模型层边界 (四臂跑通 + 机械闸门).

背景: docs/A1_DESIGN.md §8 (S2 注入算子) + docs/S3_PROTOCOL.md §2.3 / §2.5 / §4.
v79 (S2-0) 的负结果已把注入算子口径定死: 保留共享空间载体, **不学任何学出来的映射**.

本代做什么:
  把注入算子实现成「真模型层边界」的 forward_pre_hook 注入, 四臂跑通:
    arm0_no_inject    零注入 (打断 -> 续跑), 必须与原前向**逐位等价**
    arm1_slot_Q      源模型第 l 层 Q 槽激活 -> 共享空间 -> 映回目标隐空间
    arm2_wrong_slot_O 同一源层的 O 槽 (错槽对照), 强度同
    arm3_random_dir  共享空间内的随机方向 (同载体 / 随机内容), 同范数

注入算子 (协议 §2.3 修订后的口径, 本代照抄, 不引入任何新映射):
    a_s   = 源模型第 l 层槽 s 的激活 (样本级 mean-pool, 原生维)
    z     = a_s @ P_A.T              (映到共享空间, 256 维)
    Δ_B   = z @ P_B                 (映回目标隐空间; **没有任何学出来的映射**)
    λ_i   = 0.1 * ||h_B^{j,i}||_F / (sqrt(T) * ||Δ_raw,i||_2)
    h_B^j <- h_B^j + λ_i * Δ_raw,i    (逐样本归一化, 广播到该样本的 T 个 token)

机械闸门 (判据全部跑前写死, 见 THRESHOLDS):
    G1 arm0 逐位等价          rel_err == 0 (协议 §4 开跑前清单第 2 条)
    C2 注入范数比精确        ||delta||_F / ||h_B^j||_F == 0.1 (逐样本)
    C3 读出层扰动可见        文本对 64 块里落进 (1e-3, 0.5) 的块数 >= 32 (每臂)
    C4 没把模型搞坏          全部块 final 输出相对扰动 <= 0.5
    C5 arm3 与真实内容不相关  |cos| >= 0.35 的样本数 == 0
    C6 全部有限              无 NaN / Inf

本代**不产生**任何 P1a/P1b/P2/P3 语义判据 —— 那是 S3 的事 (协议 §2.5 定的四臂,
本代只验「四臂跑得通」与机械前提)。
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

# xet 后端在本机会卡在 read-token 端点; 只对**本进程**生效, 不写任何持久配置。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch


# ---------------------------------------------------------------- 常量 (跑前写死)
DS = 256                      # 共享空间维度 (与 v62 一致)
SEQ = 128                     # 文本序列长
BATCH = 16                    # S2 机械闸门每批样本数 (S3 用全量 128)
N_TEXT = 128                  # 文本条数 (与 v77/v78/v79 同一缓存)
N_IMG = 16                    # ViT 臂本代取的图片数 (机械闸门, 类别平衡不影响)
PROJ_SEED = 0                 # 共享空间正交投影的种子 (与 v77/v79 同)
RAND_SEED = 20261117          # arm3 随机方向的种子基准 (逐块派生)

#: 层阶梯 (协议 §2.1)
L_Q = (0, 1, 2, 3, 6, 12, 18, 23)     # Qwen 24 层
L_B = (0, 1, 2, 3)                    # BERT 4 层
L_V = (0, 3, 6, 9, 11)                # ViT 12 层

SOURCE_SLOTS = ("Q", "O")
ARMS = ("arm1_slot_Q", "arm2_wrong_slot_O", "arm3_random_dir")

#: (方向名, 源模型, 目标模型, 源层阶梯, 目标层阶梯)
DIRECTIONS = (
    ("Qwen->BERT", "Qwen", "BERT", L_Q, L_B),
    ("BERT->Qwen", "BERT", "Qwen", L_B, L_Q),
    ("ViT->ViT", "ViT", "ViT", L_V, L_V),
)

THRESHOLDS = {
    # G1 arm0 逐位等价 (协议 §4 开跑前清单第 2 条: rel_err = 0)
    "G1_arm0_rel_err_max": 0.0,
    # C2 注入范数比 (协议 §2.3: 逐样本归一化到 0.1)
    "C2_norm_ratio_target": 0.1,
    "C2_norm_ratio_atol": 1e-6,
    # C3 读出层扰动可见 (下界沿用 v77 B3 的 1e-3, 上界 0.5)
    "C3_min_rel_perturbation": 1e-3,
    "C3_max_rel_perturbation": 0.5,
    "C3_min_blocks_in_range": 32,
    # C4 没把模型搞坏
    "C4_max_rel_perturbation": 0.5,
    # C5 arm3 与真实内容不相关 (共享空间 256 维下随机方向的 |cos| ~ N(0, 1/256))
    "C5_arm3_cos_threshold": 0.35,
    "C5_arm3_cos_violations_max": 0,
}


# ---------------------------------------------------------------- 按路径载入 (复用不复制)
def _load_by_path(path: Path, name: str):
    spec = ilu.spec_from_file_location(name, str(path))
    mod = ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_PKG = Path(__file__).resolve().parent
_V77 = _load_by_path(_PKG.parent / "v77_true_model_flow" / "true_model.py",
                     "true_model_v77_by_path")

MEAS = _V77.MEAS
SLOT_PATHS = _V77.SLOT_PATHS
Backbone = _V77.Backbone
Injector = _V77.Injector
text_batches = _V77.text_batches
capture_slots = _V77.capture_slots
fetch_models = _V77.fetch_models
fetch_texts = _V77.fetch_texts
fetch_images = _V77.fetch_images
sha256_of = _V77.sha256_of
CACHE_DIR = _V77.CACHE_DIR

PROVENANCE_FILES = {
    "v62_measure": _V77.PROVENANCE_FILES["v62_measure"],
    "v77_true_model": str(_PKG.parent / "v77_true_model_flow" / "true_model.py"),
    "v80_s2_operator": str(_PKG / "s2_operator.py"),
}


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- 层边界捕获
def capture_layers(bb, batch, out_layers: Sequence[int], in_layers: Sequence[int] = ()):
    """一次前向抓若干层的**输出**(forward-hook) 与**输入**(pre-hook)。

    口径 (协议 §2.6): hidden_states[j] = 第 j 层的**输入** = 第 j-1 层的输出。
    本代读出层 = 注入层 j 的**输出** = hidden_states[j+1] = 协议说的「第 j+1 层」。
    """
    store: Dict[Tuple[str, int], torch.Tensor] = {}
    handles = []
    for ly in sorted(set(int(x) for x in out_layers)):
        def post(module, args, kwargs, out, _k=ly):
            o = out[0] if isinstance(out, tuple) else out
            store[("out", _k)] = o.detach()
        handles.append(bb.layers[ly].register_forward_hook(post, with_kwargs=True))
    for ly in sorted(set(int(x) for x in in_layers)):
        def pre(module, args, kwargs, _k=ly):
            hs = kwargs.get("hidden_states", None)
            if hs is None and len(args) > 0:
                hs = args[0]
            store[("in", _k)] = hs.detach()
        handles.append(bb.layers[ly].register_forward_pre_hook(pre, with_kwargs=True))
    try:
        with torch.no_grad():
            bb.model(**batch)
    finally:
        for h in handles:
            h.remove()
    return store


def make_batch(bb, name: str, texts, imgs) -> dict:
    if name == "ViT":
        proc = bb.proc(images=imgs[:BATCH], return_tensors="pt")
        return {k: v for k, v in proc.items()}
    return text_batches(bb, texts)[0]


# ---------------------------------------------------------------- 注入量构造
def build_delta(z_shared: torch.Tensor, p_target: torch.Tensor,
                h_ref: torch.Tensor, ratio: float):
    """构造注入量并逐样本归一化。

    z_shared : [N, 256] 共享空间方向
    p_target : [256, D_B] 目标模型到共享空间的正交投影 (行正交)
    h_ref    : [N, T, D_B] 目标模型第 j 层的输入 (参考前向)

    返回 (delta [N, T, D_B], lam [N])。逐样本满足 ||delta_i||_F = ratio * ||h_i||_F。
    """
    n, t, d = h_ref.shape
    raw = z_shared @ p_target                                   # [N, D_B]
    raw_norm = raw.norm(dim=-1).clamp_min(1e-12)                # [N]
    h_norm = h_ref.reshape(n, -1).norm(dim=-1).clamp_min(1e-12)  # [N]
    lam = ratio * h_norm / (math.sqrt(t) * raw_norm)
    delta = (raw * lam[:, None]).unsqueeze(1).expand(n, t, d).contiguous()
    return delta, lam


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).pow(2).sum().sqrt() / (a.pow(2).sum().sqrt() + 1e-12))


def cos_rows(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    return (z1 * z2).sum(-1) / (z1.norm(dim=-1) * z2.norm(dim=-1) + 1e-12)


# ---------------------------------------------------------------- 主流程
def run() -> dict:
    t0 = time.time()
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(min(16, os.cpu_count() or 8))

    fetch_models()
    texts = fetch_texts(N_TEXT)
    imgs = fetch_images(N_IMG)
    proj = MEAS.projections(PROJ_SEED)

    # ---- 源侧槽激活: 样本级 mean-pool (原生维), 与 v77/v78/v79 同口径
    src_cache: Dict[str, Dict[Tuple[int, str], torch.Tensor]] = {}
    for name, layers in (("Qwen", L_Q), ("BERT", L_B), ("ViT", L_V)):
        bb = Backbone(name)
        batch = make_batch(bb, name, texts, imgs)
        want = [(ly, slot) for ly in layers for slot in SOURCE_SLOTS]
        st = capture_slots(bb, batch, want)
        src_cache[name] = {k: st[k].mean(dim=1).clone() for k in want}
        del bb, batch, st
        gc.collect()
        print("[source] %s: 抓到 %d 个 (层,槽) 组合的激活" % (name, len(want)), flush=True)

    arm0_rows: List[dict] = []
    block_rows: List[dict] = []
    cos_rows_all: List[dict] = []
    blk = 0

    for dname, s_name, t_name, s_layers, t_layers in DIRECTIONS:
        bb = Backbone(t_name)
        batch = make_batch(bb, t_name, texts, imgs)
        n_layers = len(bb.layers)
        last = n_layers - 1
        ref = capture_layers(bb, batch,
                             out_layers=list(range(n_layers)),
                             in_layers=list(t_layers))
        print("[target] %s: %d 层, 参考前向完成 (方向 %s)" % (t_name, n_layers, dname), flush=True)

        # ---- G1: 零注入的「打断 -> 续跑」必须与原前向逐位等价
        for j in t_layers:
            zero = torch.zeros_like(ref[("in", j)])
            with Injector(bb.layers[j], zero):
                out0 = capture_layers(bb, batch, out_layers=list(range(n_layers)))
            per_layer = [rel_l2(ref[("out", ly)], out0[("out", ly)]) for ly in range(n_layers)]
            arm0_rows.append({"direction": dname, "target_layer": int(j),
                              "max_rel_err": max(per_layer),
                              "per_layer_rel_err": [round(x, 15) for x in per_layer]})
            del out0
        print("[arm0] %s: %d 个注入点的零注入等价已核 (max rel_err=%.3g)"
              % (dname, len(t_layers), max(r["max_rel_err"] for r in arm0_rows[-len(t_layers):])),
              flush=True)

        # ---- 四臂
        p_a = proj[s_name]
        p_b = proj[t_name]
        z_cache = {}
        for ly in s_layers:
            for slot in SOURCE_SLOTS:
                z_cache[(ly, slot)] = src_cache[s_name][(ly, slot)] @ p_a.t()

        for j in t_layers:
            h_ref = ref[("in", j)]
            for ly in s_layers:
                if s_name == t_name and ly == j:
                    continue                      # ViT 自跨层臂: 排除同层
                blk += 1
                g = torch.Generator().manual_seed(RAND_SEED + blk)
                u = torch.randn(z_cache[(ly, "Q")].shape, generator=g)
                u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                arms = {
                    "arm1_slot_Q": z_cache[(ly, "Q")],
                    "arm2_wrong_slot_O": z_cache[(ly, "O")],
                    "arm3_random_dir": u,
                }
                for arm in ARMS:
                    z = arms[arm]
                    delta, lam = build_delta(z, p_b, h_ref,
                                             THRESHOLDS["C2_norm_ratio_target"])
                    with Injector(bb.layers[j], delta):
                        out = capture_layers(bb, batch, out_layers=sorted({j, last}))
                    o_ref = ref[("out", j)]
                    o_inj = out[("out", j)]
                    f_ref = ref[("out", last)]
                    f_inj = out[("out", last)]
                    n = h_ref.shape[0]
                    ratio = (delta.reshape(n, -1).norm(dim=-1)
                             / h_ref.reshape(n, -1).norm(dim=-1))
                    dh = (o_inj - o_ref).mean(dim=1)
                    dd = delta.mean(dim=1)
                    align = float(((dh * dd).sum(-1)
                                   / (dh.pow(2).sum(-1).sqrt() * dd.pow(2).sum(-1).sqrt()
                                      + 1e-12)).mean())
                    finite = bool(torch.isfinite(o_inj).all() and torch.isfinite(f_inj).all())
                    block_rows.append({
                        "direction": dname,
                        "source_model": s_name,
                        "target_model": t_name,
                        "source_layer": int(ly),
                        "target_layer": int(j),
                        "arm": arm,
                        "n_samples": int(n),
                        "norm_ratio_median": round(float(ratio.median()), 9),
                        "norm_ratio_max_abs_dev": round(float((ratio - THRESHOLDS["C2_norm_ratio_target"]).abs().max()), 12),
                        "rel_perturb_readout": round(rel_l2(o_ref, o_inj), 9),
                        "rel_perturb_final": round(rel_l2(f_ref, f_inj), 9),
                        "align_readout": round(align, 6),
                        "finite": finite,
                    })
                    del out, delta
                # C5: arm3 与 arm1/arm2 的余弦 (共享空间; P 行正交 => 目标空间等价)
                c31 = cos_rows(arms["arm3_random_dir"], arms["arm1_slot_Q"])
                c32 = cos_rows(arms["arm3_random_dir"], arms["arm2_wrong_slot_O"])
                thr = THRESHOLDS["C5_arm3_cos_threshold"]
                cos_rows_all.append({
                    "direction": dname,
                    "source_layer": int(ly),
                    "target_layer": int(j),
                    "n_samples": int(c31.numel()),
                    "max_abs_cos_arm3_vs_arm1": round(float(c31.abs().max()), 6),
                    "max_abs_cos_arm3_vs_arm2": round(float(c32.abs().max()), 6),
                    "n_violations": int((c31.abs() >= thr).sum() + (c32.abs() >= thr).sum()),
                })
            print("[arms] %s: 目标层 %d 跑完 (%d 源层 x 3 臂)" % (dname, j, len(s_layers)), flush=True)

        del bb, batch, ref
        gc.collect()

    # ---------------------------------------------------------------- 汇总
    def stat(vals: List[float]) -> dict:
        if not vals:
            return {"min": None, "median": None, "max": None, "n": 0}
        arr = np.asarray(vals, dtype=float)
        return {"min": round(float(arr.min()), 9),
                "median": round(float(np.median(arr)), 9),
                "max": round(float(arr.max()), 9),
                "n": int(arr.size)}

    summary_by_arm = {}
    for arm in ARMS:
        per_arm = [r for r in block_rows if r["arm"] == arm]
        text_rows = [r for r in per_arm if r["direction"] != "ViT->ViT"]
        vit_rows = [r for r in per_arm if r["direction"] == "ViT->ViT"]
        in_range_text = sum(
            1 for r in text_rows
            if THRESHOLDS["C3_min_rel_perturbation"] < r["rel_perturb_readout"]
            < THRESHOLDS["C3_max_rel_perturbation"])
        in_range_vit = sum(
            1 for r in vit_rows
            if THRESHOLDS["C3_min_rel_perturbation"] < r["rel_perturb_readout"]
            < THRESHOLDS["C3_max_rel_perturbation"])
        summary_by_arm[arm] = {
            "n_blocks_total": len(per_arm),
            "n_blocks_text": len(text_rows),
            "n_blocks_vit": len(vit_rows),
            "n_in_range_readout_text": in_range_text,
            "n_in_range_readout_vit": in_range_vit,
            "rel_perturb_readout": stat([r["rel_perturb_readout"] for r in per_arm]),
            "rel_perturb_final": stat([r["rel_perturb_final"] for r in per_arm]),
            "align_readout": stat([r["align_readout"] for r in per_arm]),
            "norm_ratio_max_abs_dev": max([r["norm_ratio_max_abs_dev"] for r in per_arm] or [0.0]),
            "all_finite": all(r["finite"] for r in per_arm),
        }

    n_text_blocks = len(set((r["direction"], r["source_layer"], r["target_layer"])
                            for r in block_rows if r["direction"] != "ViT->ViT"))
    n_vit_blocks = len(set((r["direction"], r["source_layer"], r["target_layer"])
                           for r in block_rows if r["direction"] == "ViT->ViT"))

    criteria = {
        "G1_arm0_bit_equivalent": all(r["max_rel_err"] <= THRESHOLDS["G1_arm0_rel_err_max"]
                                       for r in arm0_rows),
        "C2_norm_ratio_exact": all(v["norm_ratio_max_abs_dev"] <= THRESHOLDS["C2_norm_ratio_atol"]
                                    for v in summary_by_arm.values()),
        "C3_readout_perturbation_visible": all(
            v["n_in_range_readout_text"] >= THRESHOLDS["C3_min_blocks_in_range"]
            for v in summary_by_arm.values()),
        "C4_no_model_break": all(r["rel_perturb_final"] <= THRESHOLDS["C4_max_rel_perturbation"]
                                  for r in block_rows),
        "C5_arm3_decorrelated": all(r["n_violations"] <= THRESHOLDS["C5_arm3_cos_violations_max"]
                                     for r in cos_rows_all),
        "C6_all_finite": all(v["all_finite"] for v in summary_by_arm.values()),
    }
    verdict = "S2_OPERATOR_READY" if all(criteria.values()) else "S2_OPERATOR_FAIL"

    corpus_text_path = CACHE_DIR / ("agnews_%d.json" % N_TEXT)
    corpus_img_path = CACHE_DIR / ("food101_%d.pkl" % N_IMG)

    return {
        "protocol": {
            "step": "A1 S2 (注入算子下沉到真模型层边界 + 四臂跑通)",
            "scope": ("机械闸门: 注入算子跑得通 + 四臂可执行 + 零注入逐位等价; "
                      "**不产生**任何 P1a/P1b/P2/P3 语义判据 (属 S3)"),
            "operator": {
                "formula": ("z = a_s @ P_A.T ; Δ_B = z @ P_B ; "
                            "λ_i = 0.1 * ||h_B^{j,i}||_F / (sqrt(T) * ||Δ_raw,i||_2) ; "
                            "h_B^j <- h_B^j + λ_i * Δ_raw,i"),
                "no_learned_mapping": True,
                "learned_mapping_note": ("协议 §2.3 修订记录 (2026-11-17 拍板): 保留共享空间载体, "
                                         "不做任何学出来的映射 ⇒ 本代源码里没有 align/ridge/procrustes "
                                         "之类的拟合步骤"),
                "token_alignment": ("样本级广播: 两个模型 token 网格不可配对 (v79 已证), "
                                    "⇒ 源槽激活按样本 mean-pool 后广播到目标的 T 个 token"),
                "readout_layer": ("注入层 j 的**输出** = hidden_states[j+1] = 协议 §2.6 的「第 j+1 层」"),
                "arm3_carrier": ("arm3 的随机方向取在**共享空间 256 维**内再映回目标空间 "
                                 "⇒ 与 arm1/arm2 走同一载体几何, 只换内容"),
            },
            "grid": {
                "L_Q": list(L_Q), "L_B": list(L_B), "L_V": list(L_V),
                "n_text_blocks": n_text_blocks, "n_vit_blocks": n_vit_blocks,
                "n_arms": len(ARMS), "arms": list(ARMS),
            },
            "sample_size": {
                "per_block": BATCH,
                "note": ("S2 是机械闸门, 每块取 1 批 (16 条) 即可验机械性质; "
                         "S3 的语义读数用全量 128 条 (协议 §2.7)"),
            },
            "thresholds": THRESHOLDS,
            "discipline": [
                "阈值跑前写死 (源码常量 THRESHOLDS, 静态可扫), 事后不得放宽",
                "arm0 零注入必须与原前向逐位等价 (协议 §4 开跑前清单第 2 条), 不达标不开跑",
                "本代不产生语义判据, 不得被引用成 P1a/P1b/P2/P3 的证据",
                "注入算子不含任何学出来的映射 (协议 §2.3 修订记录)",
                "arm3 逐块独立取随机方向 (种子 = RAND_SEED + 块序号), 可复算",
            ],
            "corpus": {
                "text": "fancyzhx/ag_news train 128 条 (同一缓存 agnews_128.json)",
                "text_sha256": sha256_of_bytes(corpus_text_path.read_bytes()) if corpus_text_path.exists() else None,
                "image": "ethz/food101 train 前 %d 张 (机械闸门只用图像内容, 不做标签读数)" % N_IMG,
                "image_cache": str(corpus_img_path),
                "cache_dir": str(CACHE_DIR),
            },
            "models": {
                "Qwen": "Qwen/Qwen2.5-0.5B-Instruct",
                "BERT": "google/bert_uncased_L-4_H-256_A-4",
                "ViT": "google/vit-base-patch16-224-in21k",
            },
            "device": "cpu", "dtype": "float32",
            "proj_seed": PROJ_SEED, "rand_seed": RAND_SEED,
            "reused_from": ["v77 true_model.py (Backbone / Injector / capture_slots / text_batches / 语料)",
                            "v62 measure.py (SLOT_PATHS / projections)"],
            "provenance_sha256": {k: sha256_of(v) for k, v in PROVENANCE_FILES.items()},
            "declared_modeling_choices": [
                "fp32 on CPU (避开 WDDM 下被桌面占满的显存; 也让零注入的逐位等价可核)",
                "注入用 forward_pre_hook 打在 decoder layer 输入处, 不手工逐层调用",
                "槽激活: 投影类槽取模块输出, O 槽取模块输入 (v62 槽语义, 与 v77/v78/v79 同)",
                "跨模型载体: 样本级 mean-pool 后广播 (token 网格不可配对)",
                "注入强度: 逐样本 Frobenius 范数比 0.1 (delta 广播到该样本全部 token)",
                "arm3 的随机方向在共享空间内取 (同载体对照), 逐块独立",
            ],
        },
        "G1_arm0": {
            "n_checks": len(arm0_rows),
            "max_rel_err": max([r["max_rel_err"] for r in arm0_rows] or [None]),
            "rows": arm0_rows,
        },
        "summary_by_arm": summary_by_arm,
        "C5_cos": {
            "n_blocks": len(cos_rows_all),
            "max_abs_cos_arm3_vs_arm1": max([r["max_abs_cos_arm3_vs_arm1"] for r in cos_rows_all] or [0.0]),
            "max_abs_cos_arm3_vs_arm2": max([r["max_abs_cos_arm3_vs_arm2"] for r in cos_rows_all] or [0.0]),
            "total_violations": sum(r["n_violations"] for r in cos_rows_all),
            "rows": cos_rows_all,
        },
        "blocks": block_rows,
        "criteria": criteria,
        "verdict": verdict,
        "limits": [
            "本代只验机制: 不产生任何 P1a/P1b/P2/P3 语义判据, 不得被引用成语义层证据",
            "每块只取 1 批 (16 条): 机械性质可验, 统计功效不在本代",
            "注入量是 a_s 的 256 维投影: 读数只作相对增量解读 (协议 §5 第 7 条)",
            "arm3 取在共享空间内 (同载体对照): 它排除的是「同载体随便什么内容都能流」, "
            "不排除「目标空间内任意扰动都能流」",
            "ViT 臂只作方向性对照, 不进跨模型配对统计 (协议 §2.2)",
            "注入不改变任何底座权重 ⇒ 与权重级 G1/G2 正交",
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
    return 0 if res["verdict"] == "S2_OPERATOR_READY" else 1


if __name__ == "__main__":
    sys.exit(main())
