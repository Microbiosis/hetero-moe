"""V55.0 — 中枢带宽的实测引理 (只读, 不需要训练, 不需要底座权重)。

测的是 **v33/v49 融合层在初始化处** 的雅可比结构, 不是训练后的解。
之所以在 init 处测有意义: P_m (正交随机, 冻结) 与专家 FFN (随机, 冻结) 在
v33 里从不训练, 只有 W_router / (gamma, beta, alpha) 训练 → 该层对输入的
一阶响应在架构层面就已固定, 训练只改它的调制系数。

三个量
------
  rank(J_alpha)      跨专家通道的秩 (路由对融合表示的全部作用都过这条通道)
  ||J_route||_F/||J||_F   该通道占输出敏感度的份额
  effrank / stable / top-k mass of J_out   局部项的方向分布 (是否与数据子空间对齐)

用法
----
    python research/aligner/v55_hub_bandwidth/measure.py            # 打印 + 落盘
    V55_SEED=1 V55_D=256 python research/aligner/v55_hub_bandwidth/measure.py

坑 (实测踩到)
-------------
  · 秩阈值必须相对且不能取 1e-8: 秩-1 矩阵的求和项数值噪声在 1e-9 量级,
    相对 s[0] 可达 1e-6。本脚本用 1e-5 相对阈值, 并同时报告相邻谱隙。
  · torch.autograd.functional.jacobian(..., vectorize=False) 对 custom
    autograd.Function (SparseRouterSTE) 会静默给出**错误的**秩 (top_k=2 报 3)。
    本脚本用显式基向量循环, 不用 vmap, 也不用 functional.jacobian。
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FUSION = REPO / "research" / "scale_heterogeneous" / "v33_scale_hetero" / "fusion.py"

SEED = int(os.environ.get("V55_SEED", "0"))
D = int(os.environ.get("V55_D", "256"))
E = 4
FF = 512
MODAL_DIMS = [2560, 1024, 1024]
REL_TOL = 1e-5


def _load_fusion():
    spec = importlib.util.spec_from_file_location("v55_fusion", FUSION)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def jac(fn, out_dim, x0):
    """显式基向量循环 (不用 functional.jacobian / vmap, 见模块 docstring)。"""
    rows = []
    for i in range(out_dim):
        x = x0.clone().requires_grad_(True)
        y = fn(x).reshape(-1)
        rows.append(torch.autograd.grad(y[i], x, retain_graph=True)[0].reshape(-1))
    return torch.stack(rows).detach()


def spectrum_report(M, rel_tol=REL_TOL):
    s = torch.linalg.svdvals(M)
    keep = s > s[0] * rel_tol
    r = int(keep.sum())
    sk = s[:r]
    # 相邻谱隙: 被保留的最后一个 vs 被丢弃的第一个 (全保留时记 0)
    gap = round(float((s[r - 1] - s[r]) / s[0]), 8) if r < s.numel() else 0.0
    return {
        "shape": list(M.shape),
        "rank": r,
        "sigma_head": [round(float(v), 6) for v in s[:6].tolist()],
        "sigma_floor_rel": round(float(s[-1] / s[0]), 12),
        "spectral_gap_after_rank": gap,
        "stable_rank": round(float(sk.sum() / sk.max()), 3),
        "eff_rank": round(float((sk.sum() ** 2) / (sk ** 2).sum()), 3),
        "top1_mass": round(float(sk[0] / sk.sum()), 4),
        "top19_mass": round(float(sk[:19].sum() / sk.sum()), 4),
        "fro_norm": round(float(M.norm()), 6),
    }


def run_one(mm, top_k, x0, d=D, e=E, ff=FF, modal_dims=None, heavy=True):
    """单档实测. heavy=False 时跳过 J_local / J_out (测试用, 快 ~50×)."""
    modal_dims = modal_dims or MODAL_DIMS
    layer = mm.CrossModalFusionLayer(d, ff, e, top_k, modal_dims)
    layer.eval()

    with torch.no_grad():
        logits = torch.nn.functional.linear(x0, layer.W_router)
        alpha = torch.softmax(logits, -1)
        _, idx = torch.topk(alpha, top_k, dim=-1)
        mask = torch.zeros(e)
        mask.scatter_(-1, idx[0], 1.0)
        alpha_hat = alpha * mask

    def alpha_fn(x):
        # 前向真映射 (mask 在 x0 的 top-k 区域里是常数, 故局部雅可比即此)
        return torch.softmax(torch.nn.functional.linear(x, layer.W_router), -1) * mask

    def expert_fn(x, k):
        h = x * layer.gammas[k] + layer.betas[k]
        f = layer._swiglu(h, layer.w_gates[k], layer.w_ups[k], layer.w_downs[k])
        return f

    with torch.no_grad():
        F = [expert_fn(x0, k)[0] for k in range(e)]

    J_alpha = jac(alpha_fn, e, x0)                       # [e, d]
    J_route = sum(F[k][:, None] * J_alpha[k][None, :] for k in range(e))   # [d, d]

    row = {
        "top_k": top_k,
        "n_nonzero_weights": int(mask.sum()),
        "gate_states": int(math.comb(e, top_k)),          # 全局门状态数 = C(e, top_k)
        "gate_bits": round(math.log2(math.comb(e, top_k)), 4),
        "J_alpha": spectrum_report(J_alpha),
        "J_alpha_theory_rank": min(top_k, e - 1),
        "J_route": spectrum_report(J_route),
    }
    if not heavy:
        return row

    J_local = sum(
        alpha_hat[0, k] * layer.alphas[k].item()
        * jac(lambda x, k=k: expert_fn(x, k), d, x0)
        for k in range(e)
    )
    J = J_route + J_local
    row.update({
        "J_local": spectrum_report(J_local),
        "J_out": spectrum_report(J),
        "route_share_fro": round(float(J_route.norm() / J.norm()), 4),
        "local_share_fro": round(float(J_local.norm() / J.norm()), 4),
    })
    return row


def check() -> int:
    """`--check`: 把已落盘的快照与定理上界对拍, 不重算 (CI 用, <1s)."""
    path = HERE / "results" / "metrics.json"
    if not path.exists():
        print(f"[FAIL] 缺少 {path}; 先跑一次 measure.py")
        return 1
    m = json.loads(path.read_text(encoding="utf-8"))
    e = m["provenance"]["num_experts"]
    bad = [r["top_k"] for r in m["rows"] if r["J_alpha"]["rank"] != min(r["top_k"], e - 1)
           or r["J_route"]["rank"] > min(r["top_k"], e - 1)]
    if bad:
        print(f"[FAIL] 定理 3 在 top_k={bad} 上与快照不符")
        return 1
    print(f"[PASS] 定理 3 在全部 {len(m['rows'])} 档上与快照一致 (M={e}); "
          f"v33 默认 = {m['v33_default']['J_alpha']['rank']} 维 / "
          f"{m['v33_default']['gate_bits']} bit")
    return 0


def main():
    if "--check" in sys.argv[1:]:
        return check()

    torch.manual_seed(SEED)
    mm = _load_fusion()
    x0 = torch.randn(1, D) * 0.5

    rows = [run_one(mm, k, x0) for k in (1, 2, 3, 4)]

    try:
        sha = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        sha = "unknown"

    out = {
        "schema_version": "1.0",
        "package": "v55_hub_bandwidth",
        "line": "aligner",
        "kind": "measured_lemma",  # 只读实测, 不是训练实验
        "provenance": {
            "code_sha": sha,
            "torch": torch.__version__,
            "device": "cpu",
            "seed": SEED,
            "d_shared": D,
            "num_experts": E,
            "d_ff": FF,
            "modal_dims": MODAL_DIMS,
            "rel_tol": REL_TOL,
            "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "note": "v33/v49 融合层在**初始化**处的雅可比结构; P_m 与专家 FFN 在 v33 中冻结, 故一阶响应是架构量",
        "rows": rows,
        "v33_default": next(r for r in rows if r["top_k"] == 2),
    }

    dst = HERE / "results" / "metrics.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    hdr = f"{'top_k':>5} {'rank(J_a)':>10} {'theory':>7} {'route%':>7} {'rank(J_route)':>13} {'eff(J_out)':>10} {'stable':>7} {'top19':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['top_k']:>5} {r['J_alpha']['rank']:>10} {r['J_alpha_theory_rank']:>7} "
            f"{100 * r['route_share_fro']:>6.1f}% {r['J_route']['rank']:>13} "
            f"{r['J_out']['eff_rank']:>10.1f} {r['J_out']['stable_rank']:>7.1f} "
            f"{r['J_out']['top19_mass']:>7.3f}"
        )
    print(f"\n-> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
