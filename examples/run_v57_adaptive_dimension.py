"""V57.0 — 自适应维度: 在真实表征上读出 r(t) 曲线.

> 归属: aligner (投影空间线的收口), v56 的后继
> 复现: bash run.sh v57
> 不训练任何模型: 只做子空间跟踪, 因此不需要 GPU, 也不受 n 小的限制以外的影响

要证明的一件事
--------------
**固定 D 在数据流的几乎每个位置都是错的。**

三阶段数据流 (每阶段 10 批):
    P1  只用文本侧 (h_base)              真实秩低
    P2  文本 + 表格 (h_base, c)          真实秩升高
    P3  只用文本侧                       真实秩回落

自适应空间应当: P1 低 → P2 扩张 → P3 坍缩。
若它只是一路涨, 那它就是"只会扩张", 等于换了个名字的固定维度。
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from research.aligner.v57_adaptive_dimension.operators import (
    AdaptiveSubspace, SpectralProject,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "research/aligner/v57_adaptive_dimension/results")
ART = os.environ.get("V57_ART_DIR", OUT)
CACHE = os.environ.get("V57_CACHE", "/tmp/v56_repr.npz")   # 复用 v56 的缓存
BATCH = int(os.environ.get("V57_BATCH", "32"))
PHASE_STEPS = int(os.environ.get("V57_PHASE_STEPS", "10"))
FIXED_D = [96, 252, 384]


def eff_rank_pr(X: torch.Tensor) -> float:
    X = X.float()
    X = X - X.mean(0, keepdim=True)
    s = torch.linalg.svdvals(X)
    s = s[s > 1e-12]
    return float(s.sum() ** 2 / (s ** 2).sum()) if s.numel() else 0.0


def load():
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return (torch.from_numpy(z["h_base"]).float(),
                torch.from_numpy(z["c"]).float())
    raise SystemExit(f"缺 {CACHE}（v56 的缓存）; 设 V57_CACHE 指向 h_base/c 的 npz")


def main():
    t0 = time.time()
    print("=" * 80)
    print("V57.0 — 自适应维度: r(t) 在真实表征上的轨迹")
    print("=" * 80)
    H, C = load()
    n, d_h = H.shape
    d_c = C.shape[1]
    print(f"h_base {tuple(H.shape)}  c_passive {tuple(C.shape)}")

    # 两侧各自归一化尺度, 否则拼接后被大的一侧压死
    H = H / H.std()
    C = C / C.std()

    # 拼接后的环境空间 = 文本侧 + 表格侧
    d_all = d_h + d_c
    B = torch.linalg.qr(torch.randn(d_all, min(d_all, 512)))[0]  # 随机把两侧混到同一坐标系
    def embed(h, c=None):
        if c is None:
            x = torch.cat([h, torch.zeros(h.shape[0], d_c)], 1)
        else:
            x = torch.cat([h, c], 1)
        return x @ B

    rng = np.random.RandomState(0)
    sp = AdaptiveSubspace(d_all, death_window=2)
    phases = [("P1 文本侧", "h"), ("P2 文本+表格", "hc"), ("P3 文本侧", "h")]
    rows = []
    for label, mode in phases:
        for step in range(PHASE_STEPS):
            idx = rng.choice(n, BATCH, replace=False)
            h = H[idx]
            c = C[idx] if mode == "hc" else None
            Z = embed(h, c)
            coords = sp(Z, update=True)
            rows.append({"phase": label, "step": step, "r": sp.r,
                         "captured": round(sp.log.captured[-1], 4),
                         "born": sp.log.born[-1], "died": sp.log.died[-1]})
            if step % 4 == 0 or step == PHASE_STEPS - 1:
                print(f"  {label:<12} step{step:>2}  r={sp.r:>4}  captured={sp.log.captured[-1]:.3f}"
                      f"  +{sp.log.born[-1]}/-{sp.log.died[-1]}")
        print(f"  —— {label} 末 r={sp.r}")

    traj = sp.log.r
    print("\n" + "=" * 80)
    print("判决")
    print("=" * 80)
    seg = len(traj) // 3
    r1, r2, r3 = max(traj[:seg]), max(traj[seg:2 * seg]), traj[-1]
    expanded = r2 > r1
    collapsed = r3 < r2
    print(f"  P1 末 r={r1}  →  P2 峰值 r={r2}  →  P3 末 r={r3}")
    print(f"  V1 扩张 (P2 > P1): {'过' if expanded else '未过'}")
    print(f"  V2 坍缩 (P3 < P2): {'已观测到 (died 合计 %d)' % sum(sp.log.died) if collapsed else '未过'}")
    print(f"  V3 维度不是常数: r 的取值集合大小 = {len(set(traj))} (若为 1 则等于固定维度)")
    print(f"  V4 固定 D 的代价: 轨迹最大 r={max(traj)}, 若固定 D={FIXED_D[0]} 则"
          f" 最多只能容纳 {max(traj) / FIXED_D[0]:.1f} 分之一所需维度")

    os.makedirs(ART, exist_ok=True)
    payload = {
        "schema_version": "1.0", "version": "57.0",
        "package": "v57_adaptive_dimension", "line": "aligner",
        "config": {"batch": BATCH, "phase_steps": PHASE_STEPS,
                   "ambient": d_all, "d_h": d_h, "d_c": d_c,
                   "death_window": 2, "cache": CACHE},
        "phases": [[lbl, mode] for lbl, mode in phases],
        "trajectory": traj,
        "rows": rows,
        "verdicts": {"V1_expand": bool(expanded), "V2_collapse": bool(collapsed),
                     "V3_not_constant": len(set(traj)) > 1,
                     "r_max": int(max(traj)), "r_final": int(traj[-1]),
                     "born_total": int(sum(sp.log.born)),
                     "died_total": int(sum(sp.log.died))},
        "fixed_d_baselines": FIXED_D,
    }
    mpath = os.path.join(ART, "metrics.json")
    json.dump(payload, open(mpath, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n落盘 {mpath}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
