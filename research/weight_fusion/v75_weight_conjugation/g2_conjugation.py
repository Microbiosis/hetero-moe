# -*- coding: utf-8 -*-
"""g2_conjugation.py — 槽级共轭的权重级有效性检验 (G1 口径回归 + G2 主判据)。

主线问题 (docs/MAINLINE_ROUTE.md §1.4 / §4):
    G1 权重级共轭: 从 proj/align 导出作用在权重上的映射, 共轭后同槽跨架构权重确实可比
    G2 共轭有效性: 共轭后"同槽跨架构"的权重相似度显著高于"异槽跨架构"

跑之前先查清七件事, 因为它们直接决定结论能不能成立:

  (1) 口径: v62 §2 的谱统计 (JS + 置换) 必须在**原生权重**上逐位复现 -> G1 回归闸门。
      同时记录 v62 measure.py 的 sha256, 保证后续复算能证明口径同源。

  (2) **前提审计 — v62 的"内切"信号里有多少是形状/族效应?**
      v62 §2 的"内切"结论 (JS 0.0302 vs 0.0401, p=0.0002) 把三类配对混在一起:
        同槽 attention、异槽 attention、同槽 FFN、异槽 FFN、**跨族 (attention vs FFN)**。
      而"异功能"一侧的 1440 对里有 768 对是**跨族**的 —— 因为同名槽必同族,
      跨族配对按构造只能落在"异功能"一侧。
      ⇒ 若只看**合并后**的 headline, 族构成差异会伪装成"功能对应"。
      本文件按族分开重算, 并给出**族平衡**统计量 (各族分离度的等权平均 + 族分层置换)。

  (3) **等形状**: FFN 本体的输入空间是 FFN 空间, 没有跨模型对应物;
      只有复合算子 contract∘expand 落在模型隐空间, 才能进共轭。
      故共轭只在等形状单元上进行 (attention 四槽 + FFN 复合),
      前提审计也用同一套单元, 保证与实验口径一致。

  (4) **共轭后用什么量**: 相似变换 A·M·A⁻¹ **严格保持奇异值** ⇒
      "共轭后再比 JS 谱"这个口径**测不到共轭本身** (共轭前后 JS 谱逐位相同)。
      正交共轭更是谱上的空操作; 非正交共轭后奇异值被 A 的条件数主导
      (JS 从 0.016 爆到 0.47), 变成"搬运强度"的度量。
      本文件以**函数相似度**为主口径: 固定共享输入 X, 比较两个算子在同一批输入上
      的行为 (线性 CKA)。它对相似变换敏感, 是"权重级可合并"的直接度量。

  (5) **G2 必须拆成两问**:
      G2-a 保留检验: 同槽相似度**本来**就存在吗? 存在才谈得上"共轭后仍保留"。
      G2-b 生成检验: 槽级搬运是否**造出**了同槽偏好 (强于随机搬运/错槽搬运/槽无关搬运)?
      前者无功效时后者仍然有效, 但结论的措辞必须跟着改 —— 不许把"生成"说成"保留"。

  (6) **两种搬运测试, 循环性不同**:
      同层拟合 (fit layer = measure layer): A_s 是**按槽**拟合的, 必然把 m1 的槽 s
        送到 m2 的槽 s 附近 ⇒ "同槽更近"是构造出来的, **只能当机械通路自检**。
      跨层拟合 (测第 l 层, 在第 (l+1)%4 层拟合): 拟合层与测量层不重叠 ⇒ 非循环,
        检验的是"槽级对齐能否跨深度迁移", 这是 G2 唯一可判定的口径。
      两个都报, 但**只有跨层的进入判据**。

  (7) **统计单位**: 实验单位是 (模型对 × 层) 这 12 个块, 种子是 nuisance。
      各种子的观测先并进同一块, 再做块内分层置换 (只在块内打乱标签);
      块间对比用**配对置换检验** (块内逐观测交换两臂标签), 比符号检验更有功效。

臂 (docs/MAINLINE_ROUTE.md §4.3 三臂 + 两个额外对照):
    A0  不搬          N = M1
    B   按槽搬        N = M1 · A_sᵀ     A_s 由**该槽**在拟合层的输出拟合 (闭式岭回归)
    C   随机搬        N = M1 · (U A_s Vᵀ)ᵀ  与 B 的 A 奇异值完全相同、方向随机
    C'  错槽搬        N = M1 · A_{t}ᵀ   用**别的槽**拟合的对齐去搬当前槽 (强对照)
    B+  槽无关搬      N = M1 · A_poolᵀ  A_pool 由**所有槽汇总**拟合 (排除循环论证)

粒度 (同一套臂, 三种标签口径):
    slot       同单元名 => 同槽 (v62 语义; 正本 §4.4 的主判据)
    family     同功能族 => 同族 (attention 族 vs FFN 族)
    slot_attn  仅 attention 族内, 同单元名 => 同槽 (排除 FFN 参与的形状混淆)

已知口径局限 (必须随数字一起引用):
    共享空间算子 M = P·Wᵀ·Pᵀ 在 BERT (D=256=DS) 上与 W **相似** (谱完整保留),
    在 Qwen (896) / ViT (768) 上是**压缩投影** (谱被截断)。
    ⇒ 共轭后的**谱**跨模型不可直接比 (故 JS 只作次要口径);
    **函数相似度 (CKA)** 不受影响, 因为它比较的是"同一批输入下的行为"。
    FFN 复合算子只是各自 FFN 的线性部分 (Qwen 的 SwiGLU gate 路缺失)。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util as ilu
import itertools
import json
import os
import sys
import zlib
from math import comb
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------- 常量
SLICE_REPO = "redmingwei/backbone-slices"
DS = 256                      # 共享空间维度 (与 v62 一致)
DIMS = {"Qwen": 896, "BERT": 256, "ViT": 768}

#: 可共轭的**等形状**功能单元: attention 四槽 + FFN 复合算子 (contract∘expand)
ATT = ("Q", "K", "V", "O")
UNITS = ATT + ("FFN",)

#: v62 的六个原生槽 (含矩形 FFN), 仅用于口径回归与形状效应审计
V62_SLOTS = ("Q", "K", "V", "O", "expand", "contract")

LAYERS = ("0", "1", "2", "3")
SEEDS_DEFAULT = 5
N_PERM = 5000
N_REP = 2048                  # 共享输入样本数
RIDGE_LAM = 1.0

#: 判据阈值 (跑之前写死)
TOL_A0 = 0.05                 # G1 口径回归相对容差
P_G2 = 0.01                   # G2 显著性阈值
P_POWER = 0.05                # 功效闸门
C_REL_TOL = 0.20              # 随机对照分离度不得超过 B 的这个比例
INSAMPLE_RETR_MIN = 0.50     # 同层拟合的槽检索准确率下限 (机械通路自检)

SLICE_DIR = os.environ.get(
    "BACKBONE_SLICES_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "hetero-moe", "backbone-slices"),
)


def _seed_of(*parts) -> int:
    """确定性种子。不能用内置 hash(): 字符串哈希每进程加随机盐 => 不可复现。"""
    return zlib.crc32("|".join(str(p) for p in parts).encode("utf-8")) % (2 ** 31)


def _v62_measure():
    """按路径载入 v62 的 measure.py。

    按**文件路径**载入, 而不是包级 import: 包级 import 会连带触发 v62 包的
    __init__ (transformers / model.py 等重依赖), 按路径只取 measure.py 本身。
    复用其 cka/js/projections, 保证口径同源。
    这一跨线依赖在 MANIFEST.implementation.note 里显式声明;
    measure.py 的 sha256 记进结果, 证明复算时口径同源。
    """
    v62 = (Path(__file__).resolve().parents[2] / "aligner" / "v62_dimension_projection_space")
    spec = ilu.spec_from_file_location("v62_measure", str(v62 / "measure.py"))
    mod = ilu.module_from_spec(spec)
    sys.modules["v62_measure"] = mod
    spec.loader.exec_module(mod)
    return mod


MEAS = _v62_measure()
V62_MEASURE_PATH = (Path(__file__).resolve().parents[2] / "aligner"
                    / "v62_dimension_projection_space" / "measure.py")


def v62_measure_sha256() -> str:
    return hashlib.sha256(V62_MEASURE_PATH.read_bytes()).hexdigest()


# ---------------------------------------------------------------- 原料
def load_wall() -> Dict[Tuple[str, str, str], torch.Tensor]:
    """载入三个底座的权重切片 -> {(model, layer, slot): tensor}。"""
    wall: Dict[Tuple[str, str, str], torch.Tensor] = {}
    for key, name in (("qwen", "Qwen"), ("bert", "BERT"), ("vit", "ViT")):
        p = os.path.join(SLICE_DIR, key + ".pt")
        if not os.path.exists(p):
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(SLICE_REPO, key + ".pt", repo_type="model")
        pack = torch.load(p, map_location="cpu", weights_only=True)
        for k, v in pack.items():
            ly, slot = k.split("_", 1)
            wall[(name, ly, slot)] = v.float()
    return wall


def native_units(wall) -> Dict[Tuple[str, str, str], torch.Tensor]:
    """等形状的功能单元: {(model, unit, layer): [D_m, D_m]}。

    attention 四槽只取方阵 (Qwen 的 K/V 是 GQA [128,896], 非方阵 => 排除,
    与 v62 的 MEASURED_SUBSPACES 口径一致)。
    FFN 单元 = contract ∘ expand 的复合算子 (方阵, 位于模型隐空间) ——
    FFN 本体的输入空间是 FFN 空间, 没有跨模型对应物, 只有复合后才有。

    已知近似: Qwen 是 SwiGLU (gate+up 两路), 切片只含 up_proj,
    故三模型的 FFN 复合算子都只是各自 FFN 的**线性部分**, 不是完整非线性 FFN。
    """
    out: Dict[Tuple[str, str, str], torch.Tensor] = {}
    for (m, ly, slot), W in wall.items():
        if slot in ATT and W.shape[0] == DIMS[m] and W.shape[1] == DIMS[m]:
            out[(m, slot, ly)] = W.clone()
    for (m, ly) in {(k[0], k[1]) for k in wall}:
        up, dn = wall.get((m, ly, "expand")), wall.get((m, ly, "contract"))
        if up is None or dn is None or dn.shape[1] != up.shape[0]:
            continue
        C = dn @ up
        if C.shape[0] == DIMS[m] and C.shape[1] == DIMS[m]:
            out[(m, "FFN", ly)] = C
    return out


def shared_ops(nat, proj) -> Dict[Tuple[str, str, str], torch.Tensor]:
    """共享空间算子 M = P · Wᵀ · Pᵀ (与 v62 slot_repr 同口径: y = x·M)。"""
    return {k: proj[k[0]] @ v.t() @ proj[k[0]].t() for k, v in nat.items()}


# ---------------------------------------------------------------- 谱工具
def spec64(Mx: torch.Tensor) -> np.ndarray:
    """归一化奇异值谱, 64 点对数采样 (与 v62 measure_spectral 同口径)。"""
    sv = torch.linalg.svdvals(Mx)
    sv = sv / sv.sum()
    idx = np.linspace(0, len(sv) - 1, 64).astype(int)
    return sv.numpy()[idx]


def v62_spectral(wall) -> dict:
    """v62 §2 口径的原生谱统计 (逐位复现用)。"""
    spec = {k: spec64(v) for k, v in wall.items()}
    keys = list(spec)
    pairs = []
    for a, b in itertools.combinations(keys, 2):
        if a[0] == b[0]:
            continue
        pairs.append((a, b, float(MEAS.js(spec[a], spec[b])), a[2] == b[2]))
    same = np.array([d for _, _, d, s in pairs if s])
    diff = np.array([d for _, _, d, s in pairs if not s])
    rng = np.random.default_rng(0)
    allv = np.array([d for _, _, d, _ in pairs])
    lab = np.array([1 if s else 0 for _, _, _, s in pairs])
    obs = diff.mean() - same.mean()
    cnt = sum(1 for _ in range(N_PERM)
              if (lambda pm: allv[pm == 0].mean() - allv[pm == 1].mean() >= obs)(rng.permutation(lab)))
    denom = np.sqrt((same.var() + diff.var()) / 2)
    per_slot = {}
    for slot in V62_SLOTS:
        v = [x for a, _b, x, s in pairs if s and a[2] == slot]
        if v:
            per_slot[slot] = float(np.mean(v))
    return {
        "n_pairs": len(pairs),
        "same_function_js_mean": float(same.mean()),
        "same_function_js_median": float(np.median(same)),
        "diff_function_js_mean": float(diff.mean()),
        "diff_function_js_median": float(np.median(diff)),
        "cohen_d": float(obs / denom) if denom > 0 else 0.0,
        "permutation_p": cnt / N_PERM,
        "per_slot_js": per_slot,
    }


# ---------------------------------------------------------------- 统计
def perm_test(pairs: Sequence[Tuple[float, bool]], higher_is_similar: bool,
              n_perm: int = N_PERM) -> dict:
    """全局置换检验 (v62 口径的标签交换): 同槽标签是否可交换。

    pairs = [(相似度, is_same)]。分离度统一定义为 **同槽更相似时为正**:
      higher_is_similar=True  (CKA) => 分离度 = mean(same) - mean(diff)
      higher_is_similar=False (JS)   => 分离度 = mean(diff) - mean(same)
    """
    if not pairs:
        return {"n_pairs": 0}
    vals = np.array([float(d) for d, _ in pairs])
    lab = np.array([1 if s else 0 for _, s in pairs])
    same = vals[lab == 1]
    diff = vals[lab == 0]
    if same.size == 0 or diff.size == 0:
        return {"n_pairs": int(vals.size), "n_same": int(same.size),
                "n_diff": int(diff.size), "separation": None, "permutation_p": None}
    sep = float(same.mean() - diff.mean()) if higher_is_similar else float(diff.mean() - same.mean())
    rng = np.random.default_rng(0)
    cnt = 0
    for _ in range(n_perm):
        pm = rng.permutation(lab)
        s = vals[pm == 1].mean() - vals[pm == 0].mean()
        if not higher_is_similar:
            s = -s
        if s >= sep:
            cnt += 1
    denom = np.sqrt((same.var() + diff.var()) / 2)
    return {
        "n_pairs": int(vals.size), "n_same": int(same.size), "n_diff": int(diff.size),
        "same_sim_mean": float(same.mean()), "diff_sim_mean": float(diff.mean()),
        "separation": sep,
        "cohen_d": float(sep / denom) if denom > 0 else 0.0,
        "permutation_p": cnt / n_perm,
    }


def strat_perm(blocks: Sequence[Dict], metric: str, higher_is_similar: bool,
               n_perm: int = N_PERM, seed: int = 0) -> dict:
    """**块内分层**置换检验: 每个块 (模型对 × 层) 内部单独打乱同槽标签, 再汇总。

    块 = 实验单位 (模型对 × 层), 种子是 nuisance ⇒ 各种子的观测先并进同一个块,
    置换只在块内进行, 不会因为种子数而虚增样本量。
    """
    arrs = []
    for b in blocks:
        pairs = b.get(metric + "_pairs") or []
        if not pairs:
            continue
        vals = np.array([float(d) for d, _ in pairs])
        lab = np.array([1 if s else 0 for _, s in pairs])
        if lab.sum() == 0 or (1 - lab).sum() == 0:
            continue
        arrs.append((vals, lab))
    if not arrs:
        return {"n_blocks": 0, "separation": None, "permutation_p": None}

    def separation(assign: List[np.ndarray]) -> Optional[float]:
        seps = []
        for (vals, _), a in zip(arrs, assign):
            s = vals[a == 1]
            d = vals[a == 0]
            if s.size == 0 or d.size == 0:
                return None
            sep = float(s.mean() - d.mean())
            seps.append(sep if higher_is_similar else -sep)
        return float(np.mean(seps)) if seps else None

    obs = separation([lab for _, lab in arrs])
    if obs is None:
        return {"n_blocks": 0, "separation": None, "permutation_p": None}
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        assign = [rng.permutation(lab) for _, lab in arrs]
        s = separation(assign)
        if s is not None and s >= obs:
            cnt += 1
    return {"n_blocks": len(arrs), "separation": obs, "permutation_p": cnt / n_perm}


def sign_test(vals: Sequence[float]) -> dict:
    """单侧符号检验 (分离度>0 视为"同槽更近")。"""
    vals = [float(v) for v in vals]
    n = len(vals)
    if n == 0:
        return {"n_blocks": 0, "sign_p": None}
    pos = sum(1 for v in vals if v > 0)
    p = sum(comb(n, k) for k in range(pos, n + 1)) / (2 ** n)
    return {"n_blocks": n, "n_positive": pos, "sign_p": float(p),
            "mean_separation": float(np.mean(vals)),
            "median_separation": float(np.median(vals))}


def median_over_seeds(vals: Sequence[float]) -> dict:
    """多种子取中位数 (v40 教训: 均值会被单种子主导)。"""
    vals = [float(v) for v in vals]
    if not vals:
        return {}
    arr = np.array(vals)
    return {"median": float(np.median(arr)),
            "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            "all": [float(v) for v in arr]}


def merge_blocks(blocks: Sequence[Dict]) -> List[Dict]:
    """把各种子的块按 (模型对, 层) 合并 —— 块是实验单位, 种子是 nuisance。

    观测用 (seed, unit_u, unit_v) 作键, 这样两个臂之间能按观测对齐做配对置换。
    """
    merged: Dict[Tuple[str, str], Dict] = {}
    for b in blocks:
        key = (b["pair"], b["layer"])
        m = merged.setdefault(key, {"pair": b["pair"], "layer": b["layer"], "obs": {}})
        for o in b.get("obs") or []:
            m["obs"][o["key"]] = o
    out = []
    for key in sorted(merged):
        m = merged[key]
        obs = [m["obs"][k] for k in sorted(m["obs"])]
        m["obs_list"] = obs
        m["cka_pairs"] = [(o["cka"], o["same"]) for o in obs]
        m["js_pairs"] = [(o["js"], o["same"]) for o in obs]
        cs = [o["cka"] for o in obs if o["same"]]
        cd = [o["cka"] for o in obs if not o["same"]]
        js_ = [o["js"] for o in obs if o["same"]]
        jd = [o["js"] for o in obs if not o["same"]]
        if cs and cd:
            m["cka_separation"] = float(np.mean(cs) - np.mean(cd))
        if js_ and jd:
            m["js_separation"] = float(np.mean(jd) - np.mean(js_))
        out.append(m)
    return out


def paired_perm(blocks_a: Sequence[Dict], blocks_b: Sequence[Dict], metric: str = "cka",
                higher_is_similar: bool = True, n_perm: int = N_PERM, seed: int = 0) -> dict:
    """**配对置换检验**: 块内逐观测交换两臂的标签, 看观测到的臂间差是否罕见。

    比"块配对符号检验"更有功效: 它用上了每个观测的数值, 而不是只数块的正负号。
    观测按 (seed, unit_u, unit_v) 对齐 ⇒ 两臂比的是**同一批**算子对。
    """
    ka = {(b["pair"], b["layer"]): b for b in blocks_a}
    kb = {(b["pair"], b["layer"]): b for b in blocks_b}
    common = sorted(set(ka) & set(kb))
    pairs = []
    for k in common:
        oa = {o["key"]: o for o in ka[k]["obs_list"]}
        ob = {o["key"]: o for o in kb[k]["obs_list"]}
        for okey in sorted(set(oa) & set(ob)):
            pairs.append((oa[okey][metric], ob[okey][metric], oa[okey]["same"]))
    if not pairs:
        return {"n_obs": 0, "mean_diff": None, "permutation_p": None}
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    lab = np.array([1 if p[2] else 0 for p in pairs])

    def stat(aa: np.ndarray, bb: np.ndarray) -> Optional[float]:
        seps = []
        for arr in (aa, bb):
            s = arr[lab == 1]
            d = arr[lab == 0]
            if s.size == 0 or d.size == 0:
                return None
            sep = float(s.mean() - d.mean())
            seps.append(sep if higher_is_similar else -sep)
        return seps[0] - seps[1]

    obs = stat(a, b)
    if obs is None:
        return {"n_obs": len(pairs), "mean_diff": None, "permutation_p": None}
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        flip = rng.random(len(pairs)) < 0.5
        aa = np.where(flip, b, a)
        bb = np.where(flip, a, b)
        s = stat(aa, bb)
        if s is not None and s >= obs:
            cnt += 1
    return {"n_obs": len(pairs), "mean_diff": obs, "permutation_p": cnt / n_perm,
            "n_blocks": len(common)}


# ---------------------------------------------------------------- 功能族
def family_of(unit: str) -> str:
    return "attention" if unit in ATT else "ffn"


def gran_same(u: str, v: str, gran: str) -> bool:
    """某个粒度下, 单元 u 与 v 是否算"同功能"。"""
    if gran == "slot":
        return u == v
    if gran == "family":
        return family_of(u) == family_of(v)
    if gran == "slot_attn":
        return u == v
    raise ValueError(gran)


# ---------------------------------------------------------------- 前提审计
def shape_effect_audit(wall) -> dict:
    """v62 §2 的形状/族效应审计: 把配对按功能族拆开, 看"内切"信号到底来自哪里。

    关键: 跨族配对按构造只能落在"异功能"一侧 (同名槽必同族),
    所以"异功能"一侧天然混入了矩形 FFN 谱 vs 方阵 attention 谱的形状差。
    """
    spec = {k: spec64(v) for k, v in wall.items()}
    keys = list(spec)
    buckets: Dict[str, List[Tuple[float, bool]]] = {}
    for a, b in itertools.combinations(keys, 2):
        if a[0] == b[0]:
            continue
        fa, fb = family_of(a[2]), family_of(b[2])
        cls = ("A_attention_family" if fa == fb == "attention" else
               "B_ffn_family" if fa == fb == "ffn" else "C_cross_family")
        d = float(MEAS.js(spec[a], spec[b]))
        buckets.setdefault(cls, []).append((d, a[2] == b[2]))
    out: Dict[str, dict] = {}
    for c, pairs in sorted(buckets.items()):
        rec = perm_test(pairs, higher_is_similar=False)
        same = [d for d, s in pairs if s]
        diff = [d for d, s in pairs if not s]
        rec["same_js_mean"] = float(np.mean(same)) if same else None
        rec["diff_js_mean"] = float(np.mean(diff)) if diff else None
        rec["mean_js_all"] = float(np.mean([d for d, _ in pairs]))
        rec["note"] = ("同槽对按构造为 0 (同名槽必同族); 该类全部计入'异功能'一侧"
                       if not same else "")
        out[c] = rec
    within = buckets["A_attention_family"] + buckets["B_ffn_family"]
    out["D_within_family_only"] = perm_test(within, higher_is_similar=False)

    # 族平衡统计量: 各族分离度的等权平均, 并按族分层置换。
    # 若"内切"真的存在 (同槽跨架构更像), 它应在**每一族内**都成立;
    # 合并 headline 的显著性可能只是两族 JS 水平不同造成的构成效应。
    fam_pairs = {"attention": buckets["A_attention_family"], "ffn": buckets["B_ffn_family"]}
    sep_by_fam = {}
    for f, pairs in fam_pairs.items():
        r = perm_test(pairs, higher_is_similar=False)
        sep_by_fam[f] = r.get("separation")
    bal = sum(v for v in sep_by_fam.values() if v is not None) / max(1, len(sep_by_fam))
    # 分层置换: 每族内部独立打乱标签, 再算等权平均分离度
    rng = np.random.default_rng(0)
    arrs = []
    for f, pairs in fam_pairs.items():
        vals = np.array([d for d, _ in pairs])
        lab = np.array([1 if s else 0 for _, s in pairs])
        if lab.sum() == 0 or (1 - lab).sum() == 0:
            continue
        arrs.append((vals, lab))
    cnt = 0
    for _ in range(N_PERM):
        seps = []
        ok = True
        for vals, lab in arrs:
            pm = rng.permutation(lab)
            s, d = vals[pm == 1], vals[pm == 0]
            if s.size == 0 or d.size == 0:
                ok = False
                break
            seps.append(float(d.mean() - s.mean()))
        if ok and seps and float(np.mean(seps)) >= bal:
            cnt += 1
    out["F_family_balanced"] = {
        "separation_by_family": sep_by_fam,
        "balanced_separation": bal,
        "family_stratified_permutation_p": cnt / N_PERM,
        "note": ("各族分离度等权平均 + 族分层置换; 若这个数不显著, "
                 "则合并 headline 的显著性主要来自族构成差异, 而非功能槽对应"),
    }
    out["_reading"] = (
        "v62 §2 的 p=0.0002 混入了'跨族'(矩形 FFN 谱 vs 方阵 attention 谱)这一形状事实: "
        "跨族配对按构造只落在'异功能'一侧。族平衡统计量才是"
        "**可迁移到共轭层面**的那部分信号。"
    )
    return out


def native_premise(nat) -> dict:
    """原生 (不搬、不投影) 等形状单元上的同槽分离 —— G2-a 保留检验的基线与功效闸门。

    配对方式: 模型对 (m1, m2) 内, m1 的每个单元 u 与 m2 的**每个**单元 v 配对
    (含 u == v 的同槽对, 也含 u != v 的异槽对)。
    """
    spec = {k: spec64(v) for k, v in nat.items()}
    out: Dict[str, dict] = {}
    for gran in GRANS:
        pairs: List[Tuple[float, bool]] = []
        blocks: List[Dict] = []
        for (m1, m2) in itertools.combinations(sorted(DIMS), 2):
            for layer in LAYERS:
                us1 = [u for u in UNITS if (m1, u, layer) in nat]
                us2 = [u for u in UNITS if (m2, u, layer) in nat]
                bp: List[Tuple[float, bool]] = []
                for u in us1:
                    for v in us2:
                        if gran == "slot_attn" and (u not in ATT or v not in ATT):
                            continue
                        same = gran_same(u, v, gran)
                        d = float(MEAS.js(spec[(m1, u, layer)], spec[(m2, v, layer)]))
                        pairs.append((d, same))
                        bp.append((d, same))
                if bp:
                    blocks.append({"pair": m1 + "x" + m2, "layer": layer, "js_pairs": bp})
        rec = perm_test(pairs, higher_is_similar=False)
        sep_blocks = []
        for b in blocks:
            s = [d for d, sm in b["js_pairs"] if sm]
            d_ = [d for d, sm in b["js_pairs"] if not sm]
            if s and d_:
                sep_blocks.append(float(np.mean(d_) - np.mean(s)))
        rec["block_sign"] = sign_test(sep_blocks)
        rec["n_blocks"] = len(blocks)
        out[gran] = rec
    return out


# ---------------------------------------------------------------- 搬运 (transport)
def fit_transport(U: torch.Tensor, V: torch.Tensor, lam: float = RIDGE_LAM) -> torch.Tensor:
    """闭式岭回归: 求 A 使 U·Aᵀ ≈ V (U/V 为中心化后的输出矩阵)。

    A 的含义: 把"模型 m1 的算子输出"搬到"模型 m2 的算子输出"所在的坐标基。
    这就是 §1.4 的 G1 —— 从激活层对齐导出作用在**权重**上的映射。
    闭式解而非梯度下降: 确定性、可复现、无迭代超参。
    """
    d = U.shape[1]
    G = U.t() @ U + lam * torch.eye(d, dtype=U.dtype)
    return torch.linalg.solve(G, U.t() @ V).t().contiguous()


def random_matched(A: torch.Tensor, seed: int) -> torch.Tensor:
    """随机对照: 与 A **奇异值完全相同**、方向随机的矩阵 (U·A·Vᵀ)。

    为什么要同奇异值: 随机对照的强度必须与真实对齐可比, 否则"B 比 C 好"可能
    只是因为 C 太弱。同奇异值 => 两者搬运能力相同, 只差方向。
    注意 U·A·Vᵀ = U'·Σ·V'ᵀ (U',V' 正交), 即一个与 A **谱相同但方向无关**的矩阵,
    它不含 A 学到的任何槽对应信息。
    """
    g = torch.Generator().manual_seed(seed)
    d = A.shape[0]
    U = torch.linalg.qr(torch.randn(d, d, generator=g))[0]
    V = torch.linalg.qr(torch.randn(d, d, generator=g))[0]
    return U @ A @ V.t()


def outputs(Mx: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """算子在一批共享输入上的输出 (行向量约定 y = x·M), 中心化。"""
    Y = X @ Mx
    return Y - Y.mean(0, keepdim=True)


# ---------------------------------------------------------------- 臂
ARMS = ("A0_none", "B_slot", "C_random", "C_wrongslot", "Bplus_pooled")
GRANS = ("slot", "family", "slot_attn")


class Ctx:
    """一个种子下的全部上下文 (投影 / 共享输入 / 预计算的参考侧产物)。"""

    def __init__(self, seed: int, nat):
        torch.manual_seed(_seed_of("proj", seed))
        self.proj = {m: torch.nn.init.orthogonal_(torch.empty(DS, DIMS[m])) for m in DIMS}
        self.sh = shared_ops(nat, self.proj)
        g = torch.Generator().manual_seed(_seed_of("X", seed))
        self.X = torch.randn(N_REP, DS, generator=g)
        self.outR: Dict[Tuple[str, str, str], torch.Tensor] = {}
        self.specR: Dict[Tuple[str, str, str], np.ndarray] = {}
        for k, M in self.sh.items():
            self.outR[k] = outputs(M, self.X)
            self.specR[k] = spec64(M)


def build_transports(ctx: Ctx, fit_layer: str):
    """在给定拟合层上构造各臂所需的搬运矩阵。

    按槽搬运 A_s (B): 每个 (单元, 模型对) 单独拟合。
    槽无关搬运 A_pool (B+): 把该模型对**所有可用单元**的输出首尾拼接后拟合一次。
    """
    A_slot: Dict[Tuple[str, str, str], torch.Tensor] = {}
    A_pool: Dict[Tuple[str, str], torch.Tensor] = {}
    for (m1, m2) in itertools.combinations(sorted(DIMS), 2):
        units = [u for u in UNITS
                 if (m1, u, fit_layer) in ctx.sh and (m2, u, fit_layer) in ctx.sh]
        if not units:
            continue
        Us, Vs = [], []
        for u in units:
            U = ctx.outR[(m1, u, fit_layer)]
            V = ctx.outR[(m2, u, fit_layer)]
            Us.append(U)
            Vs.append(V)
            A_slot[(u, m1, m2)] = fit_transport(U, V)
        A_pool[(m1, m2)] = fit_transport(torch.cat(Us, 0), torch.cat(Vs, 0))
    return A_slot, A_pool


def transport(ctx: Ctx, arm: str, A_slot, A_pool, unit, m1, m2, layer):
    """按臂把模型 m1 的某单元算子搬到模型 m2 的基。"""
    M1 = ctx.sh[(m1, unit, layer)]
    if arm == "A0_none":
        return M1
    if arm == "B_slot":
        A = A_slot.get((unit, m1, m2))
        return None if A is None else M1 @ A.t()
    if arm == "C_random":
        A = A_slot.get((unit, m1, m2))
        if A is None:
            return None
        R = random_matched(A, _seed_of("rand", unit, m1, m2, layer))
        return M1 @ R.t()
    if arm == "C_wrongslot":
        # 用**别的槽**拟合出来的对齐去搬当前槽 (强对照: 强度同, 语义错)
        others = sorted(u for u in UNITS if u != unit and (u, m1, m2) in A_slot)
        if not others:
            return None
        pick = others[_seed_of("wrong", unit, m1, m2, layer) % len(others)]
        return M1 @ A_slot[(pick, m1, m2)].t()
    if arm == "Bplus_pooled":
        A = A_pool.get((m1, m2))
        return None if A is None else M1 @ A.t()
    raise ValueError(arm)


def run_arm(ctx: Ctx, arm: str, A_slot, A_pool, layers, seed: int) -> dict:
    """跑一个臂: 按三种粒度收集同/异单元配对的相似度 (块 = 模型对 × 层)。"""
    res = {g: {"cka_pairs": [], "js_pairs": [], "blocks": []} for g in GRANS}
    for (m1, m2) in itertools.combinations(sorted(DIMS), 2):
        for layer in layers:
            units = [u for u in UNITS
                     if (m1, u, layer) in ctx.sh and (m2, u, layer) in ctx.sh]
            if len(units) < 2:
                continue
            sims: Dict[str, List[Tuple[float, float, str]]] = {}
            for u in units:
                N = transport(ctx, arm, A_slot, A_pool, u, m1, m2, layer)
                if N is None:
                    continue
                outN = outputs(N, ctx.X)
                specN = spec64(N)
                for v in units:
                    R = (m2, v, layer)
                    sims.setdefault(u, []).append((
                        float(MEAS.cka(outN, ctx.outR[R])),
                        float(MEAS.js(specN, ctx.specR[R])),
                        v,
                    ))
            for gran in GRANS:
                obs = []
                for u, lst in sims.items():
                    for c, j, v in lst:
                        if gran == "slot_attn" and (u not in ATT or v not in ATT):
                            continue
                        obs.append({"key": (seed, u, v), "u": u, "v": v,
                                    "cka": c, "js": j, "same": gran_same(u, v, gran)})
                if not obs:
                    continue
                res[gran]["cka_pairs"].extend([(o["cka"], o["same"]) for o in obs])
                res[gran]["js_pairs"].extend([(o["js"], o["same"]) for o in obs])
                cs = [o["cka"] for o in obs if o["same"]]
                cd = [o["cka"] for o in obs if not o["same"]]
                js_ = [o["js"] for o in obs if o["same"]]
                jd = [o["js"] for o in obs if not o["same"]]
                if cs and cd and js_ and jd:
                    res[gran]["blocks"].append({
                        "pair": m1 + "x" + m2, "layer": layer, "obs": obs,
                        "cka_pairs": [(o["cka"], o["same"]) for o in obs],
                        "js_pairs": [(o["js"], o["same"]) for o in obs],
                        "cka_separation": float(np.mean(cs) - np.mean(cd)),
                        "js_separation": float(np.mean(jd) - np.mean(js_)),
                    })
    return res


def retrieval(ctx: Ctx, arm: str, A_slot, A_pool, layers) -> dict:
    """检索准确率: 把 m1 的单元 u 搬过去后, 在 m2 的全部单元里最像的是谁?

    slot 口径: 最像的必须是**同名单元** (零假设 = 随机命中)。
    family 口径: 最像的必须与 u **同族** (二分类, 零假设 = 1/2)。
    """
    hit_slot = hit_fam = tot = 0
    chance_slot = 0.0
    for (m1, m2) in itertools.combinations(sorted(DIMS), 2):
        for layer in layers:
            cands = [v for v in UNITS if (m2, v, layer) in ctx.sh]
            if not cands:
                continue
            for u in UNITS:
                if (m1, u, layer) not in ctx.sh or u not in cands:
                    continue
                N = transport(ctx, arm, A_slot, A_pool, u, m1, m2, layer)
                if N is None:
                    continue
                outN = outputs(N, ctx.X)
                sc = {v: float(MEAS.cka(outN, ctx.outR[(m2, v, layer)])) for v in cands}
                best = max(sc, key=sc.get)
                hit_slot += int(best == u)
                hit_fam += int(gran_same(u, best, "family"))
                tot += 1
                chance_slot += 1.0 / len(cands)
    if tot == 0:
        return {"n": 0}
    p0f = 0.5
    pvf = sum(comb(tot, i) * p0f ** i * (1 - p0f) ** (tot - i) for i in range(hit_fam, tot + 1))
    p0s = chance_slot / tot
    pvs = sum(comb(tot, i) * p0s ** i * (1 - p0s) ** (tot - i) for i in range(hit_slot, tot + 1))
    return {"n": tot, "slot_hits": hit_slot, "slot_accuracy": hit_slot / tot,
            "slot_chance": p0s, "slot_binom_p": float(min(1.0, pvs)),
            "family_hits": hit_fam, "family_accuracy": hit_fam / tot,
            "family_chance": p0f, "family_binom_p": float(min(1.0, pvf))}


# ---------------------------------------------------------------- 主
def v62_published() -> dict:
    """v62 §2 已发布数字 (从 v62 指标快照读, 不手抄)。"""
    p = (Path(__file__).resolve().parents[2] / "aligner" / "v62_dimension_projection_space"
         / "results" / "metrics.json")
    return json.loads(p.read_text(encoding="utf-8"))["spectral"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=SEEDS_DEFAULT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true", help="少种子, 仅供冒烟")
    args = ap.parse_args(argv)
    seeds = list(range(min(args.seeds, 2))) if args.quick else list(range(args.seeds))

    print("[1/5] 载入权重切片 ...", flush=True)
    wall = load_wall()
    nat = native_units(wall)
    print("      原始张量 %d, 等形状功能单元 %d" % (len(wall), len(nat)), flush=True)

    print("[2/5] G1 口径回归 (原生六槽全四层, 应逐位复现 v62 §2) ...", flush=True)
    pub = v62_published()
    a0 = v62_spectral(wall)
    rel = {
        "same_function_js_mean": abs(a0["same_function_js_mean"] - pub["same_function_js_mean"])
        / pub["same_function_js_mean"],
        "diff_function_js_mean": abs(a0["diff_function_js_mean"] - pub["diff_function_js_mean"])
        / pub["diff_function_js_mean"],
    }
    a0_ok = bool(max(rel.values()) <= TOL_A0 and a0["n_pairs"] == pub["n_pairs"])
    print("      同功能 %.6f (v62 %.6f, 相对差 %.3f%%)"
          % (a0["same_function_js_mean"], pub["same_function_js_mean"],
             100 * rel["same_function_js_mean"]), flush=True)
    print("      异功能 %.6f (v62 %.6f, 相对差 %.3f%%)"
          % (a0["diff_function_js_mean"], pub["diff_function_js_mean"],
             100 * rel["diff_function_js_mean"]), flush=True)
    print("      置换 p=%.4f (v62 %.4f)  配对数 %d (v62 %d)  => %s"
          % (a0["permutation_p"], pub["permutation_p"], a0["n_pairs"], pub["n_pairs"],
             "PASS" if a0_ok else "FAIL"), flush=True)
    print("      v62 measure.py sha256=%s" % v62_measure_sha256()[:16], flush=True)

    print("[3/5] 前提审计: v62 的'内切'信号里有多少是形状/族效应? ...", flush=True)
    shape = shape_effect_audit(wall)
    for c, rec in shape.items():
        if c.startswith("_"):
            continue
        if c == "F_family_balanced":
            print("      %-22s 族平衡分离度 %+.5f  族分层置换 p=%.4f"
                  % (c, rec["balanced_separation"], rec["family_stratified_permutation_p"]),
                  flush=True)
            continue
        print("      %-22s n_same=%-4d n_diff=%-4d 分离度 %s p=%s 均值JS %.5f"
              % (c, rec.get("n_same", 0), rec.get("n_diff", 0),
                 ("%+.5f" % rec["separation"]) if rec.get("separation") is not None else "n/a",
                 ("%.4f" % rec["permutation_p"]) if rec.get("permutation_p") is not None else "n/a",
                 rec.get("mean_js_all", float("nan"))), flush=True)
    premise = native_premise(nat)
    print("      --- 原生等形状单元上的同槽分离 (G2-a 功效闸门) ---", flush=True)
    for gran, rec in premise.items():
        print("      %-12s n_same=%-4d n_diff=%-4d 分离度 %s p=%s (块 %s/%s 正 p=%s)"
              % (gran, rec.get("n_same", 0), rec.get("n_diff", 0),
                 ("%+.5f" % rec["separation"]) if rec.get("separation") is not None else "n/a",
                 ("%.4f" % rec["permutation_p"]) if rec.get("permutation_p") is not None else "n/a",
                 rec.get("block_sign", {}).get("n_positive", 0),
                 rec.get("block_sign", {}).get("n_blocks", 0),
                 ("%.4f" % rec["block_sign"]["sign_p"])
                 if rec.get("block_sign", {}).get("sign_p") is not None else "n/a"), flush=True)

    print("[4/5] G2 主实验 (%d 种子, 跨层拟合, 三粒度) ..." % len(seeds), flush=True)
    acc: Dict[str, dict] = {arm: {g: {"cka_pairs": [], "js_pairs": [], "blocks": []}
                                    for g in GRANS} for arm in ARMS}
    acc_ins = {arm: {g: {"blocks": []} for g in GRANS} for arm in ARMS}
    acc_ret = {arm: [] for arm in ARMS}
    acc_ins_ret = {arm: [] for arm in ARMS}
    align_quality: List[dict] = []
    for sd in seeds:
        ctx = Ctx(sd, nat)
        # 每层的搬运矩阵: 测第 l 层 => 在第 (l+1)%4 层拟合 (拟合层与测量层不重叠)
        transports = {layer: build_transports(ctx, LAYERS[(LAYERS.index(layer) + 1) % len(LAYERS)])
                      for layer in LAYERS}
        # 同层拟合 (循环论证版): 只用于机械通路自检
        ins_slot, ins_pool = build_transports(ctx, "1")
        aq = {}
        for (u, m1, m2), A in ins_slot.items():
            U = ctx.outR[(m1, u, "1")]
            V = ctx.outR[(m2, u, "1")]
            aq["%s:%s->%s" % (u, m1, m2)] = float(MEAS.cka(U @ A.t(), V))
        align_quality.append(aq)
        for arm in ARMS:
            d = acc[arm]
            for layer in LAYERS:
                A_slot, A_pool = transports[layer]
                r = run_arm(ctx, arm, A_slot, A_pool, [layer], sd)
                for gran in GRANS:
                    d[gran]["cka_pairs"].extend(r[gran]["cka_pairs"])
                    d[gran]["js_pairs"].extend(r[gran]["js_pairs"])
                    d[gran]["blocks"].extend(r[gran]["blocks"])
            r_ins = run_arm(ctx, arm, ins_slot, ins_pool, ["1"], sd)
            for gran in GRANS:
                acc_ins[arm][gran]["blocks"].extend(r_ins[gran]["blocks"])
            ret = {"slot_hits": 0, "family_hits": 0, "n": 0}
            for layer in LAYERS:
                A_slot, A_pool = transports[layer]
                rr = retrieval(ctx, arm, A_slot, A_pool, [layer])
                ret["slot_hits"] += rr.get("slot_hits", 0)
                ret["family_hits"] += rr.get("family_hits", 0)
                ret["n"] += rr.get("n", 0)
            acc_ret[arm].append(ret)
            acc_ins_ret[arm].append(retrieval(ctx, arm, ins_slot, ins_pool, ["1"]))
        print("      seed %d done" % sd, flush=True)

    arms: Dict[str, dict] = {}
    merged_blocks: Dict[str, Dict[str, List[Dict]]] = {}
    for arm in ARMS:
        d = acc[arm]
        entry = {"granularities": {}}
        merged_blocks[arm] = {}
        for gran in GRANS:
            blocks = merge_blocks(d[gran]["blocks"])
            merged_blocks[arm][gran] = blocks
            cka_sep = [b["cka_separation"] for b in blocks]
            js_sep = [b["js_separation"] for b in blocks]
            ins_blocks = merge_blocks(acc_ins[arm][gran]["blocks"])
            ins_sep = [b["cka_separation"] for b in ins_blocks]
            entry["granularities"][gran] = {
                "cka_global_perm": perm_test(d[gran]["cka_pairs"], higher_is_similar=True),
                "cka_strat_perm": strat_perm(blocks, "cka", True),
                "cka_block": sign_test(cka_sep),
                "cka_block_median": median_over_seeds(cka_sep),
                "js_global_perm": perm_test(d[gran]["js_pairs"], higher_is_similar=False),
                "js_strat_perm": strat_perm(blocks, "js", False),
                "js_block": sign_test(js_sep),
                "in_sample_cka_block": sign_test(ins_sep),
                "in_sample_block_median": median_over_seeds(ins_sep),
            }
        ret = acc_ret[arm]
        tot = sum(r["n"] for r in ret)
        ins = acc_ins_ret[arm]
        ins_tot = sum(r.get("n", 0) for r in ins)
        ins_hits = sum(r.get("slot_hits", 0) for r in ins)
        entry["retrieval"] = {
            "n_total": tot,
            "slot_hits_total": sum(r["slot_hits"] for r in ret),
            "slot_accuracy": (sum(r["slot_hits"] for r in ret) / tot) if tot else 0.0,
            "family_hits_total": sum(r["family_hits"] for r in ret),
            "family_accuracy": (sum(r["family_hits"] for r in ret) / tot) if tot else 0.0,
            "in_sample_slot_accuracy": (ins_hits / ins_tot) if ins_tot else 0.0,
            "in_sample_n": ins_tot,
        }
        arms[arm] = entry

    # 配对对比: B 相对每个对照臂, 以及每个对照臂相对 A0 (null)
    paired: Dict[str, dict] = {}
    for gran in GRANS:
        b_blocks = merged_blocks["B_slot"][gran]
        entry = {"B_minus_" + other: paired_perm(b_blocks, merged_blocks[other][gran])
                 for other in ("A0_none", "C_random", "C_wrongslot", "Bplus_pooled")}
        for other in ("C_random", "C_wrongslot", "Bplus_pooled"):
            entry[other + "_minus_A0_none"] = paired_perm(
                merged_blocks[other][gran], merged_blocks["A0_none"][gran])
        entry["B_block_sign_minus_" + "A0_none"] = sign_test(
            [b["cka_separation"] for b in b_blocks])
        paired[gran] = entry

    print("[5/5] 判据 ...", flush=True)

    def sep(arm: str, gran: str) -> float:
        return arms[arm]["granularities"][gran]["cka_block_median"].get("median", 0.0)

    verdicts: Dict[str, dict] = {}
    crit_by_gran: Dict[str, dict] = {}
    for gran in GRANS:
        b_sep, a0_sep = sep("B_slot", gran), sep("A0_none", gran)
        c_sep, cp_sep = sep("C_random", gran), sep("C_wrongslot", gran)
        bp_sep = sep("Bplus_pooled", gran)
        base_p = premise[gran].get("permutation_p")
        native_power = bool(base_p is not None and base_p < P_POWER)
        ins_retr = arms["B_slot"]["retrieval"]["in_sample_slot_accuracy"]
        pm = paired[gran]
        crit = {
            "transport_path_works_in_sample": bool(ins_retr >= INSAMPLE_RETR_MIN),
            "native_baseline_detectable_at_this_granularity": native_power,
            "B_separation_positive": bool(b_sep > 0),
            "B_strat_perm_p_lt_0.01": bool(
                (arms["B_slot"]["granularities"][gran]["cka_strat_perm"].get("permutation_p")
                 or 1.0) < P_G2),
            "B_separation_not_smaller_than_A0": bool(b_sep >= a0_sep),
            "C_random_separation_small": bool(
                abs(c_sep) <= C_REL_TOL * abs(b_sep) if b_sep else abs(c_sep) < 1e-9),
            "B_beats_C_wrongslot_paired": bool(
                (pm["B_minus_C_wrongslot"].get("permutation_p") or 1.0) < P_G2),
            "B_beats_Bplus_pooled_paired": bool(
                (pm["B_minus_Bplus_pooled"].get("permutation_p") or 1.0) < P_G2),
            "B_beats_C_random_paired": bool(
                (pm["B_minus_C_random"].get("permutation_p") or 1.0) < P_G2),
        }
        crit_by_gran[gran] = crit

        # ---- G2-a 保留检验: 需要原生基线有功效, 否则不可判定 ----
        if not crit["transport_path_works_in_sample"]:
            va = "G2a_INCONCLUSIVE_TRANSPORT_PATH_BROKEN"
        elif not native_power:
            va = "G2a_INCONCLUSIVE_NO_BASELINE_POWER"
        elif not crit["B_separation_positive"] or not crit["B_strat_perm_p_lt_0.01"]:
            va = "G2a_FAIL_NO_SEPARATION"
        elif not crit["B_separation_not_smaller_than_A0"]:
            va = "G2a_FAIL_TRANSPORT_HURTS"
        else:
            va = "G2a_PASS"

        # ---- G2-b 生成检验: 不依赖原生基线, 只问搬运是否造出偏好 ----
        if not crit["transport_path_works_in_sample"]:
            vb = "G2b_INCONCLUSIVE_TRANSPORT_PATH_BROKEN"
        elif not crit["B_beats_C_random_paired"]:
            vb = "G2b_FAIL_RANDOM_AS_GOOD"
        elif not crit["B_beats_C_wrongslot_paired"]:
            vb = "G2b_FAIL_WRONG_SLOT_AS_GOOD"
        elif not crit["B_beats_Bplus_pooled_paired"]:
            vb = "G2b_FAIL_CIRCULAR"
        elif not crit["B_separation_positive"]:
            vb = "G2b_FAIL_NO_SEPARATION"
        else:
            vb = "G2b_PASS"
        verdicts[gran] = {"G2a_preservation": va, "G2b_creation": vb}

    crit = {
        "G1_reproduces_v62_within_5pct": a0_ok,
        "premise_native_slot_detectable": bool(
            (premise["slot"].get("permutation_p") or 1.0) < P_POWER),
    }
    primary = verdicts["slot"]["G2a_preservation"]
    out = {
        "protocol": {
            "ds": DS, "dims": DIMS, "units": list(UNITS), "attn_units": list(ATT),
            "v62_slots": list(V62_SLOTS), "layers": list(LAYERS),
            "fit_scheme": "cyclic leave-one-layer-out (测第 l 层 => 在第 (l+1)%4 层拟合)",
            "in_sample_scheme": "同层拟合 (循环论证版), 仅作机械通路自检",
            "seeds": seeds, "n_perm": N_PERM, "n_rep": N_REP, "ridge_lambda": RIDGE_LAM,
            "experimental_unit": "(模型对 × 层) 12 个块; 种子是 nuisance, 各种子观测并入同一块",
            "primary_metric": "cka (函数相似度, 越大越像)",
            "secondary_metric": "js (谱相似度, 越小越像)",
            "granularities": {
                "slot": "同单元名 => 同槽 (v62 语义, 正本 §4.4 主判据)",
                "family": "同功能族 => 同族 (attention 族 vs FFN 族)",
                "slot_attn": "仅 attention 族内, 同单元名 => 同槽",
            },
            "slice_dir": SLICE_DIR,
            "v62_measure_sha256": v62_measure_sha256(),
            "transport_forms": {
                "A0_none": "N = M1",
                "B_slot": "N = M1 · A_sᵀ (按槽拟合的输出对齐)",
                "C_random": "N = M1 · (U A_s Vᵀ)ᵀ (同奇异值、随机方向)",
                "C_wrongslot": "N = M1 · A_{t}ᵀ (用别的槽的对齐搬当前槽)",
                "Bplus_pooled": "N = M1 · A_poolᵀ (槽无关汇总对齐)",
            },
            "known_caveats": [
                "共享空间算子 M=P·Wᵀ·Pᵀ 在 BERT(D=256=DS) 上与 W 相似(谱完整保留), "
                "在 Qwen(896)/ViT(768) 上是压缩投影(谱被截断) ⇒ 共轭后的谱跨模型不可直接比, "
                "JS 只作次要口径; 函数相似度(CKA)不受此影响。",
                "FFN 复合算子只是各自 FFN 的线性部分 (Qwen 的 SwiGLU gate 路缺失)。",
                "相似变换严格保持奇异值 ⇒ 用 JS 谱衡量'共轭是否保住了相似性'在数学上无效; "
                "本代主口径是函数相似度。",
            ],
        },
        "g1_native_spectral": {
            "measured": a0, "v62_published": pub, "relative_error": rel, "passes": a0_ok,
        },
        "premise_shape_effect_audit": shape,
        "premise_native_equishaped": premise,
        "arms": arms,
        "paired_contrasts": paired,
        "criteria": crit,
        "criteria_by_granularity": crit_by_gran,
        "verdict_by_granularity": verdicts,
        "verdict": primary,
        "alignment_quality": align_quality,
    }

    for gran in GRANS:
        print("      --- 粒度 %s ---" % gran)
        for arm in ARMS:
            g = arms[arm]["granularities"][gran]
            print("      %-14s CKA 分离 %+.5f 分层p=%.4f sign p=%.4f (%d/%d 正) | 检索 slot %.3f "
                  "(同层自检 %.3f) / family %.3f"
                  % (arm, g["cka_block_median"].get("median", 0.0),
                     g["cka_strat_perm"].get("permutation_p") or 1.0,
                     g["cka_block"].get("sign_p") or 1.0, g["cka_block"].get("n_positive", 0),
                     g["cka_block"].get("n_blocks", 0),
                     arms[arm]["retrieval"]["slot_accuracy"],
                     arms[arm]["retrieval"]["in_sample_slot_accuracy"],
                     arms[arm]["retrieval"]["family_accuracy"]), flush=True)
        for k, v in paired[gran].items():
            if "permutation_p" in v:
                print("      配对 %-26s 差 %+.5f 置换 p=%.4f (n=%d)"
                      % (k, v.get("mean_diff") or 0.0, v.get("permutation_p") or 1.0,
                         v.get("n_obs", 0)), flush=True)
    for k, v in crit.items():
        print("      [%s] %s" % ("PASS" if v else "FAIL", k), flush=True)
    for gran, v in verdicts.items():
        print("      VERDICT(%s): 保留=%s 生成=%s" % (gran, v["G2a_preservation"],
                                                     v["G2b_creation"]), flush=True)
    print("      VERDICT(主判据 slot, G2a): %s" % primary, flush=True)

    dest = args.out or str(Path(__file__).resolve().parent / "results" / "metrics.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    payload = json.dumps(out, ensure_ascii=False, indent=2).replace(chr(10), chr(13) + chr(10))
    with open(dest, "wb") as f:
        f.write(payload.encode("utf-8"))
    print("      结果写入 %s" % dest, flush=True)
    return 0 if primary == "G2a_PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
