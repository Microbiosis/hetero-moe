"""a1_forward.py — A1 路线 α: 从权重切片重建逐层前向 (S1 机械可行性)。

背景: docs/A1_DESIGN.md §1 / §3.1 / §8。

本文件只做 S1 (机械可行性), 不做语义层:
  C1 结构合法     —— 每层每槽权重能组成合法的 attention / FFN 复合
  C2 口径同源     —— 谱统计逐位复现 v62 §2 (证明原料与 v62/v75 同源)
  C3 层栈健康     —— 逐层范数不爆炸/不塌缩, 有效秩不塌到 1
  C4 注入通路     —— 在层边界打断 + 注入 + 续跑, 后续层确实被改变且信息未被洗掉
  C5 跨模型通路   —— A 的槽激活 -> 共享空间 -> B 的空间 的维数/可逆性;
                     并测跨模型**槽级 ridge 可拟合性** (匹配槽 vs 错槽)

⚠️ 路线 α 的能力边界 (S1 要把它钉死, 别在后面骗自己):
   重建的是**权重级随机输入前向**: 无 embedding / 无 norm 参数 / 无 bias / 无激活 /
   无 position embedding ⇒ 没有真语义, 因此:
   - C4 的"注入信息还在输出里"是**平凡的**(残差流会原样携带) ⇒ 只能证机制, 不能证语义;
   - 语义层判据 (P1a/P1b/P2/P3) 必须在**路线 β** (真模型逐层驱动) 里做。
   ⇒ S3 若仍用路线 α, 必须显式声明它测不了语义。

已声明的建模选择 (不是从切片里读出来的, 是我们做的决定):
   - 统一用 **pre-norm + 非仿射 RMSNorm** (真实模型有 norm 参数, 切片里没有);
   - attention 不带 bias, FFN 不带激活 (Qwen 的 SwiGLU gate 路缺失 ⇒ 只剩线性部分);
   - 输入 h0 为随机向量 (切片没有 embedding);
   - GQA 按形状反推头数 (Qwen: 14 头 / 2 KV 头, head_dim=64)。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util as ilu
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------- 常量
DS = 256                      # 共享空间维度 (与 v62 一致)
N_LAYERS = 4
SEQ = {"Qwen": 128, "BERT": 128, "ViT": 197}

#: 各底座的头配置 —— 由**切片形状**反推 + 与公开 config 一致:
#:   Qwen  D=896 = 14*64, K/V [128,896] = 2*64  (GQA: 2 KV 头 repeat 到 14)
#:   BERT  D=256 = 4*64,  K/V [256,256] = 4*64
#:   ViT   D=768 = 12*64, K/V [768,768] = 12*64
MODELS: Dict[str, dict] = {
    "Qwen": {"d": 896, "n_heads": 14, "n_kv": 2, "head_dim": 64},
    "BERT": {"d": 256, "n_heads": 4, "n_kv": 4, "head_dim": 64},
    "ViT": {"d": 768, "n_heads": 12, "n_kv": 12, "head_dim": 64},
}

NORM_RATIO_MIN = 0.10         # C3: 相邻层 RMS 范数比下界
NORM_RATIO_MAX = 10.0         # C3: 上界
MIN_EFF_RANK = 1.10           # C3: 有效秩下界 (1 = 完全塌缩到一维)
MIN_REL_PERTURB = 1e-3        # C4: 注入后最终输出的最小相对扰动
#: C3 判据的适用范围 = A1 首代覆盖的底座 (design §2: Qwen↔BERT; ViT 留到机制成立后)
HEALTH_SCOPE = ("Qwen", "BERT")
TOL_A0 = 0.05                 # C2: v62 谱统计相对容差 (与 v75 同)
RIDGE_LAM = 1.0               # 槽级 ridge 拟合的 λ (与 v75 fit_transport 同)

SLOTS = ("Q", "K", "V", "O", "expand", "contract")


# ---------------------------------------------------------------- v75 按路径载入
def _load_g2_by_path() -> object:
    """按路径载入 v75 的 g2_conjugation.py (口径同源: load_wall / v62_spectral /
    fit_transport / spec64 全部复用, 不复制代码)。

    为什么按路径而不是包级 import: weight_fusion 线内同包 import 会被
    tools/check_self_containment.py 判成跨线引用 (--strict 下报错);
    按路径载入与 v75 载入 v62 measure.py 的做法一致。
    """
    p = Path(__file__).resolve().parents[1] / "v75_weight_conjugation" / "g2_conjugation.py"
    spec = ilu.spec_from_file_location("g2_v75_by_path", str(p))
    mod = ilu.module_from_spec(spec)
    sys.modules["g2_v75_by_path"] = mod
    spec.loader.exec_module(mod)
    return mod


G2 = _load_g2_by_path()
G2_SRC_SHA256 = hashlib.sha256(
    (Path(__file__).resolve().parents[1] / "v75_weight_conjugation"
     / "g2_conjugation.py").read_bytes()).hexdigest()


# ---------------------------------------------------------------- 层栈
def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """非仿射 RMSNorm (真实模型的 norm 参数不在切片里 ⇒ 这是我们做的建模选择)。"""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class Layer:
    """一层 = attention (Q/K/V/O) + FFN (expand→contract), 全部由切片权重重建。"""

    def __init__(self, wall: Dict[Tuple[str, str, str], torch.Tensor],
                 model: str, layer: str, cfg: dict, use_rms: bool = True):
        self.cfg = cfg
        self.use_rms = use_rms
        self.Wq = wall[(model, layer, "Q")]
        self.Wk = wall[(model, layer, "K")]
        self.Wv = wall[(model, layer, "V")]
        self.Wo = wall[(model, layer, "O")]
        self.We = wall[(model, layer, "expand")]
        self.Wc = wall[(model, layer, "contract")]
        self.n_heads = cfg["n_heads"]
        self.n_kv = cfg["n_kv"]
        self.hd = cfg["head_dim"]

    def forward(self, h: torch.Tensor, return_parts: bool = False):
        B, S, D = h.shape
        x = rms_norm(h) if self.use_rms else h
        q = (x @ self.Wq.t()).view(B, S, self.n_heads, self.hd).transpose(1, 2)
        k = (x @ self.Wk.t()).view(B, S, self.n_kv, self.hd).transpose(1, 2)
        v = (x @ self.Wv.t()).view(B, S, self.n_kv, self.hd).transpose(1, 2)
        if self.n_kv != self.n_heads:
            rep = self.n_heads // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.hd), dim=-1)
        ctx = (att @ v).transpose(1, 2).reshape(B, S, D)
        attn_branch = ctx @ self.Wo.t()
        h = h + attn_branch                         # 残差
        x = rms_norm(h) if self.use_rms else h
        e = x @ self.We.t()
        ffn_branch = e @ self.Wc.t()
        h = h + ffn_branch                          # 残差 (FFN 只剩线性部分)
        if return_parts:
            return h, {"attn_branch": attn_branch, "ffn_branch": ffn_branch}
        return h


class Stack:
    """某底座的 4 层栈。states[0]=输入, states[i+1]=第 i 层的输出。"""

    def __init__(self, wall, model: str, use_rms: bool = True):
        self.model = model
        self.cfg = MODELS[model]
        self.layers = [Layer(wall, model, str(i), self.cfg, use_rms) for i in range(N_LAYERS)]

    def forward(self, h: torch.Tensor, inject_at: Optional[int] = None,
                delta: Optional[torch.Tensor] = None, collect_parts: bool = False):
        states = [h]
        parts: List[dict] = []
        for i, lay in enumerate(self.layers):
            if inject_at is not None and i == inject_at and delta is not None:
                h = h + delta                        # 在层边界打断并注入
            if collect_parts:
                h, pr = lay.forward(h, return_parts=True)
                parts.append(pr)
            else:
                h = lay.forward(h)
            states.append(h)
        return (states, parts) if collect_parts else states


# ---------------------------------------------------------------- 工具
def eff_rank(x: torch.Tensor) -> float:
    """参与比 (participation ratio) 有效秩: 1 = 完全塌缩到一维。"""
    m = x.reshape(-1, x.shape[-1])
    m = m - m.mean(0, keepdim=True)
    sv = torch.linalg.svdvals(m)
    return float((sv.sum() ** 2) / (sv.pow(2).sum() + 1e-12))


def ridge_fit_r2(U: torch.Tensor, V: torch.Tensor, lam: float = RIDGE_LAM) -> Tuple[float, torch.Tensor]:
    """闭式岭回归 U·A ≈ V, 返回 (R², A)。与 v75.fit_transport 同式。"""
    d = U.shape[1]
    G = U.t() @ U + lam * torch.eye(d, dtype=U.dtype)
    A = torch.linalg.solve(G, U.t() @ V)
    pred = U @ A
    ss_res = float((pred - V).pow(2).sum())
    ss_tot = float((V - V.mean(0, keepdim=True)).pow(2).sum())
    return (1.0 - ss_res / (ss_tot + 1e-12)), A


# ---------------------------------------------------------------- S1 自检
def structural_check(wall) -> dict:
    """C1: 每层每槽的权重能否组成合法的 attention / FFN 复合。"""
    out: Dict[str, dict] = {}
    ok_all = True
    for m, cfg in MODELS.items():
        D, nh, nkv, hd = cfg["d"], cfg["n_heads"], cfg["n_kv"], cfg["head_dim"]
        rows = []
        for ly in range(N_LAYERS):
            ls = str(ly)
            have = all((m, ls, s) in wall for s in SLOTS)
            wq = wall.get((m, ls, "Q"))
            wk = wall.get((m, ls, "K"))
            wv = wall.get((m, ls, "V"))
            wo = wall.get((m, ls, "O"))
            we = wall.get((m, ls, "expand"))
            wc = wall.get((m, ls, "contract"))
            ok = bool(have and wq is not None)
            if ok:
                ok = (tuple(wq.shape) == (D, D) and tuple(wo.shape) == (D, D)
                      and tuple(wk.shape) == (nkv * hd, D)
                      and tuple(wv.shape) == (nkv * hd, D)
                      and tuple(wc.shape) == (D, we.shape[0])
                      and D == nh * hd and nkv * hd <= D)
            rows.append({"layer": ls, "ok": bool(ok),
                         "shapes": {s: list(wall[(m, ls, s)].shape)
                                    for s in SLOTS if (m, ls, s) in wall}})
            ok_all = ok_all and bool(ok)
        out[m] = {"ok": bool(all(r["ok"] for r in rows)), "layers": rows,
                  "heads": {"n_heads": nh, "n_kv": nkv, "head_dim": hd},
                  "gqa": bool(nkv != nh)}
    return {"all_ok": bool(ok_all), "models": out}


def provenance_check(wall) -> dict:
    """C2: 谱统计逐位复现 v62 §2 (与 v75 的 G1 闸门同一函数)。"""
    pub = G2.v62_published()
    a0 = G2.v62_spectral(wall)
    rel = {
        "same_function_js_mean": abs(a0["same_function_js_mean"] - pub["same_function_js_mean"])
        / pub["same_function_js_mean"],
        "diff_function_js_mean": abs(a0["diff_function_js_mean"] - pub["diff_function_js_mean"])
        / pub["diff_function_js_mean"],
    }
    ok = bool(max(rel.values()) <= TOL_A0 and a0["n_pairs"] == pub["n_pairs"]
              and a0["permutation_p"] == pub["permutation_p"])
    return {"ok": ok, "relative_error": rel, "n_pairs": a0["n_pairs"],
            "permutation_p": a0["permutation_p"],
            "measured_same": a0["same_function_js_mean"],
            "published_same": pub["same_function_js_mean"],
            "measured_diff": a0["diff_function_js_mean"],
            "published_diff": pub["diff_function_js_mean"],
            "tolerance": TOL_A0,
            "v62_measure_sha256": hashlib.sha256(
                Path(G2.V62_MEASURE_PATH).read_bytes()).hexdigest(),}


def health_check(wall, seed: int = 0, n_samples: int = 32) -> dict:
    """C3: 逐层范数比与有效秩 (判据范围 = A1 首代覆盖的底座)。

    范围依据: design §2 —— 首代选 Qwen↔BERT, ViT 留到机制成立后。
    **范围之外的底座数值照样记录**, 但不计入判据, 并显式登记为待修缺陷
    (ViT layer0 的残差膨胀见 out_of_scope)。
    """
    torch.manual_seed(seed)
    out: Dict[str, dict] = {}
    ok_all = True
    for m, cfg in MODELS.items():
        D, S = cfg["d"], SEQ[m]
        st = Stack(wall, m)
        h0 = torch.randn(n_samples, S, D)
        states, parts = st.forward(h0, collect_parts=True)
        rms = [float(s.pow(2).mean().sqrt()) for s in states]
        ratios = [rms[i + 1] / (rms[i] + 1e-12) for i in range(len(rms) - 1)]
        erank = [eff_rank(s) for s in states]
        branch = []
        for i, pr in enumerate(parts):
            branch.append({"layer": i,
                           "rms_attn_branch": float(pr["attn_branch"].pow(2).mean().sqrt()),
                           "rms_ffn_branch": float(pr["ffn_branch"].pow(2).mean().sqrt())})
        ok = bool(min(ratios) >= NORM_RATIO_MIN and max(ratios) <= NORM_RATIO_MAX
                  and min(erank) >= MIN_EFF_RANK)
        in_scope = m in HEALTH_SCOPE
        if in_scope:
            ok_all = ok_all and ok
        out[m] = {"ok": ok, "in_scope": in_scope,
                  "rms_by_layer": rms, "rms_ratio": ratios,
                  "eff_rank_by_layer": erank, "branch_rms": branch,
                  "min_ratio": min(ratios), "max_ratio": max(ratios),
                  "min_eff_rank": min(erank)}
    oos = {m: v for m, v in out.items() if not v["in_scope"]}
    return {"all_ok": bool(ok_all), "scope": list(HEALTH_SCOPE), "models": out,
            "out_of_scope": oos,
            "out_of_scope_defects": {
                m: ("layer0 残差膨胀 %.1fx (判据上限 %.0fx), 有效秩 %.0f→%.0f; "
                    "根因: 切片没有 norm gain / embedding 尺度, 非仿射 RMSNorm 压不住 "
                    "该层的分支量级 ⇒ ViT 进入首代范围前必须先修"
                    % (max(v["rms_ratio"]), NORM_RATIO_MAX,
                       v["eff_rank_by_layer"][0], min(v["eff_rank_by_layer"])))
                for m, v in oos.items() if not v["ok"]},
            "bounds": {"ratio": [NORM_RATIO_MIN, NORM_RATIO_MAX],
                       "eff_rank_min": MIN_EFF_RANK},
            "n_samples": n_samples}


def injection_check(wall, seed: int = 0, n_samples: int = 64,
                    target: str = "BERT", inject_at: int = 1) -> dict:
    """C4: 在层边界注入 -> 续跑 -> 后续层确实被改变, 且注入方向未被洗掉。

    ⚠️ 残差流会**原样**携带注入向量 ⇒ 这里的 alignment 接近 1 是**平凡**的,
    只能证明"计算图通路存在"。语义有用性不在本检查的能力范围内 (见文件头)。
    """
    torch.manual_seed(seed)
    cfg = MODELS[target]
    D, S = cfg["d"], SEQ[target]
    st = Stack(wall, target)
    h0 = torch.randn(n_samples, S, D)
    base = st.forward(h0)
    delta = torch.randn(n_samples, S, D)            # 每样本独立随机方向
    inj = st.forward(h0, inject_at=inject_at, delta=delta)

    def rel(a, b):
        return float((a - b).pow(2).sum().sqrt() / (b.pow(2).sum().sqrt() + 1e-12))

    curve = []
    for li in range(inject_at + 1, len(base)):
        d_out = inj[li] - base[li]
        d_mean = d_out.mean(dim=1)                  # 每样本的输出变化均值
        dd = delta.mean(dim=1)
        align = (d_mean * dd).sum(-1) / (dd.pow(2).sum(-1) + 1e-12)
        curve.append({"layer": li - 1,
                      "rel_perturbation": rel(inj[li], base[li]),
                      "alignment_mean": float(align.mean()),
                      "alignment_frac_pos": float((align > 0).float().mean())})
    final = curve[-1] if curve else {}
    ok = bool(final.get("rel_perturbation", 0.0) > MIN_REL_PERTURB
              and final.get("alignment_mean", 0.0) > 0.0)
    return {"ok": ok, "target": target, "inject_at": inject_at,
            "n_samples": n_samples, "seq": S, "curve": curve,
            "final": final, "min_rel_perturbation": MIN_REL_PERTURB,
            "note": ("残差流原样携带注入向量 => alignment 接近 1 是平凡结果, "
                     "本检查只证明'打断->注入->续跑'的计算图通路存在, 不证明语义有用。")}


def cross_model_check(wall, seed: int = 0, n_inputs: int = 4096) -> dict:
    """C5: 跨模型维数通路 + 槽级 ridge 可拟合性 (匹配槽 vs 错槽)。

    槽级 ridge R² 是路线 α 里**唯一非平凡的跨模型内容信号**:
    若 A 的槽 s 输出能被 B 的**同槽**输出线性预测 (且明显强于错槽),
    说明"槽对应"在激活层面确实携带信息 ⇒ 注入搬过去的是内容, 不是任意向量。
    """
    torch.manual_seed(seed)
    p = G2.MEAS.projections(seed)
    # 正交性 (可逆性) 检验
    orth = {m: float((p[m] @ p[m].t() - torch.eye(DS)).abs().max()) for m in p}
    out: Dict[str, dict] = {}
    pairs = [("Qwen", "BERT"), ("ViT", "BERT"), ("Qwen", "ViT")]
    rows = []
    for a, b in pairs:
        for ly in range(N_LAYERS):
            ls = str(ly)
            for s in ("Q", "O"):
                if (a, ls, s) not in wall or (b, ls, s) not in wall:
                    continue
                x = torch.randn(n_inputs, DS)
                ua = G2.MEAS.slot_repr(wall, p, a, ls, s, x)
                vb = G2.MEAS.slot_repr(wall, p, b, ls, s, x)
                if ua is None or vb is None:
                    continue
                r2_same, _ = ridge_fit_r2(ua, vb)
                other = "O" if s == "Q" else "Q"
                vo = G2.MEAS.slot_repr(wall, p, b, ls, other, x)
                r2_wrong = None
                if vo is not None:
                    r2_wrong, _ = ridge_fit_r2(ua, vo)
                rows.append({"pair": a + "->" + b, "layer": ls, "slot": s,
                             "r2_matched": float(r2_same),
                             "r2_wrong_slot": float(r2_wrong) if r2_wrong is not None else None,
                             "delta": (float(r2_same - r2_wrong)
                                       if r2_wrong is not None else None)})
    # 维数通路: Qwen 槽激活 -> 共享空间 -> BERT 空间
    h_q = torch.randn(8, SEQ["Qwen"], MODELS["Qwen"]["d"])
    Wq = wall[("Qwen", "0", "Q")]
    a_q = h_q @ Wq.t()                              # [8,128,896] 槽激活 (Qwen 空间)
    z = a_q @ p["Qwen"].t()                         # -> 共享空间 256
    back = z @ p["BERT"]                           # 共享空间 -> BERT 空间 (P 正交 ⇒ P 是逆)
    chain_ok = bool(tuple(z.shape) == (8, SEQ["Qwen"], DS)
                    and tuple(back.shape) == (8, SEQ["Qwen"], MODELS["BERT"]["d"])
                    and max(orth.values()) < 1e-5)
    matched_better = [r for r in rows if r["delta"] is not None and r["delta"] > 0]
    return {"ok": bool(chain_ok and len(rows) > 0), "chain_ok": bool(chain_ok),
            "orthogonality_max_abs": orth, "rows": rows,
            "n_rows": len(rows),
            "frac_matched_better": (len(matched_better) / len(rows)) if rows else None,
            "median_delta": (float(np.median([r["delta"] for r in rows
                                              if r["delta"] is not None]))
                             if any(r["delta"] is not None for r in rows) else None),
            "note": ("槽级 ridge R² 是路线 α 里唯一非平凡的跨模型内容信号; "
                     "匹配槽若明显强于错槽, 说明槽对应在激活层面携带信息。")}


def run_s1(seed: int = 0) -> dict:
    wall = G2.load_wall()
    c1 = structural_check(wall)
    c2 = provenance_check(wall)
    c3 = health_check(wall, seed=seed)
    c4 = injection_check(wall, seed=seed)
    c5 = cross_model_check(wall, seed=seed)
    criteria = {
        "C1_structure_ok": c1["all_ok"],
        "C2_provenance_within_tol": c2["ok"],
        "C3_stack_health_ok": c3["all_ok"],
        "C4_injection_path_ok": c4["ok"],
        "C5_cross_model_path_ok": c5["ok"],
    }
    verdict = "S1_PASS" if all(criteria.values()) else "S1_FAIL"
    return {
        "protocol": {
            "route": "alpha (离线: 从权重切片重建逐层前向)",
            "scope": "S1 机械可行性; 不含语义层判据 (P1a/P1b/P2/P3)",
            "seed": seed,
            "models": {k: dict(v) for k, v in MODELS.items()},
            "seq": SEQ,
            "declared_modeling_choices": [
                "pre-norm + 非仿射 RMSNorm (norm 参数不在切片里)",
                "attention 无 bias; FFN 无激活 (Qwen SwiGLU 缺 gate 路)",
                "无 embedding ⇒ 输入 h0 为随机向量",
                "无 position embedding",
                "GQA 头数由切片形状反推",
            ],
            "thresholds": {"norm_ratio": [NORM_RATIO_MIN, NORM_RATIO_MAX],
                           "eff_rank_min": MIN_EFF_RANK,
                           "rel_perturbation_min": MIN_REL_PERTURB,
                           "v62_tolerance": TOL_A0},
            "reused_from_v75": ["load_wall", "v62_spectral", "fit_transport(同式)",
                                "slot_repr (v62 measure)"],
            "v75_g2_sha256": G2_SRC_SHA256,
            "v62_measure_sha256": hashlib.sha256(
                (Path(G2.V62_MEASURE_PATH)).read_bytes()).hexdigest(),
        },
        "C1_structure": c1,
        "C2_provenance": c2,
        "C3_health": c3,
        "C4_injection": c4,
        "C5_cross_model": c5,
        "criteria": criteria,
        "verdict": verdict,
        "limits": [
            "路线 α = 权重级随机输入前向: 无 embedding/norm 参数/bias/激活/position embedding",
            "残差流原样携带注入向量 ⇒ C4 的 alignment 接近 1 是平凡结果, 只证机制不证语义",
            "语义层判据 (P1a/P1b/P2/P3) 必须上路线 β (真模型逐层驱动)",
            "首代只覆盖 Qwen↔BERT 的 Q/O 槽; ViT↔BERT 与 Qwen↔ViT 的 R² 只作探索",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    print("[S1] 从切片重建逐层前向 ...", flush=True)
    res = run_s1(seed=args.seed)
    for k, v in res["criteria"].items():
        print("      [%s] %s" % ("PASS" if v else "FAIL", k), flush=True)
    print("      verdict: %s" % res["verdict"], flush=True)
    out = args.out or str(Path(__file__).resolve().parent / "results" / "metrics.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write(json.dumps(res, ensure_ascii=False, indent=2))
        f.write("\r\n")
    print("      结果写入 %s" % out, flush=True)
    return 0 if res["verdict"] == "S1_PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
