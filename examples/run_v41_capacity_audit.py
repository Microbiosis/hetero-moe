"""v41.0 — 对齐器阶梯的容量-泄漏双重审计 (runner).

审计两个一直挂账的未闭项:

**E2(容量混杂)**  docs 记录的对齐器阶梯 linear 1.0436 → attn 0.3448 → mixture 0.2575
→ mom 0.2645, 但参数量是 325,632 → 720,640 → 2,291,968 → 1,704,448, **性能序与容量序
完全一致**, 无法判定"设计"还是"规模"。

**关键构造**: ``MixtureAligner(n_shared_experts=1)`` 的参数量**恰好等于**
``CrossArchAttnAligner`` 的 720,640。这是构造出来的精确等容量对, 因此
``attn`` vs ``mixture-n1`` 在容量严格相同的前提下比较两种机制。

**泄漏**  v13 的 ``eval_teacher`` 在训练同源数据上评估 (in-sample)。本 runner 对每个臂
同时报 in-sample 与 held-out (同分布、不同抽样, ``prepare_data`` 换一个足够大的
base seed), 分离"机制增益"与"记住训练样本"。

用法::

    python3 examples/run_v41_capacity_audit.py              # 6 臂 × 5 seed
    V41_SEEDS=1 python3 examples/run_v41_capacity_audit.py  # 冒烟
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from research.aligner.v10_embedding import AlignedFusionLayer, CrossArchAttnAligner
from research.aligner.v13_mixture import MixtureAligner
from research.aligner.v15_nested import MoMAligner
from research.aligner.v41_capacity_audit import count_trainable, heldout_seed, ladder_table

# 协议常量与数据构造全部复用 v13, 保证与被审计对象同口径
from examples.run_v13_distill import eval_teacher, prepare_data, train_teacher
from examples.run_v8_full import D_SHARED, N_CLS, S, SEEDS, STEPS, LR, load_encoders, make_pools

HELDOUT_OFFSET = 1000

# (臂名, 对齐器构造) —— 参数在 make_teacher 里按 encoders 的实际维度填
ARMS = ["attn", "mixture-n1", "mixture-n2", "mixture-n4", "mixture-n8", "mom"]


def make_teacher(kind, attn_pools, encoders):
    """与 v13 的 make_teacher 同构, 只换 aligner。"""
    (_, _, D_bert, _), (_, _, D_llama, _), (_, _, D_vit, _) = encoders
    modal_dims = [D_bert, D_llama, D_vit]
    if kind == "attn":
        aligner = CrossArchAttnAligner(modal_dims=modal_dims, d_shared=D_SHARED, num_heads=4)
    elif kind.startswith("mixture"):
        n = int(kind.split("-n")[1])
        aligner = MixtureAligner(modal_dims=modal_dims, d_shared=D_SHARED,
                                 n_shared_experts=n, num_heads=4)
    elif kind == "mom":
        aligner = MoMAligner(modal_dims=modal_dims, d_shared=D_SHARED,
                             n_inner=2, n_outer=2, num_heads=4)
    else:
        raise ValueError(kind)
    return AlignedFusionLayer(d_shared=D_SHARED, d_ff=4 * D_SHARED,
                              attn_pools=attn_pools, modal_dims=modal_dims,
                              aligner=aligner, c1_alpha=0.1)


def run_arm(kind, seed, encoders, attn_pools):
    """训练一个臂, 返回 (params, in_sample_mse, heldout_mse)。"""
    torch.manual_seed(seed)
    modal_seqs, modal_indices, targets = prepare_data(seed, encoders)
    teacher = make_teacher(kind, attn_pools, encoders)
    train_teacher(teacher, modal_seqs, modal_indices, targets)
    mse_in = eval_teacher(teacher, modal_seqs, modal_indices, targets)

    # held-out: 同分布、不同抽样
    ho_seqs, ho_idx, ho_tgt = prepare_data(heldout_seed(seed, HELDOUT_OFFSET), encoders)
    mse_out = eval_teacher(teacher, ho_seqs, ho_idx, ho_tgt)

    # 容量必须只数**对齐器**: AlignedFusionLayer 自带的 gammas/betas/alphas/W_router
    # 各臂完全相同, 数进去会把等容量对淹没在共享项里。
    return count_trainable(teacher.aligner), count_trainable(teacher), mse_in, mse_out


def main():
    seeds = int(os.environ.get("V41_SEEDS", SEEDS))
    print("=" * 92)
    print("V41.0 — 对齐器阶梯: 容量-泄漏双重审计")
    print("=" * 92)
    print(f"{len(ARMS)} 臂 × {seeds} seed")
    print("E2 关键构造: mixture-n1 与 attn 参数量应当完全相等")
    print("评估口径: in-sample(v13 原口径) + held-out(同分布不同抽样)")
    print()

    encoders = load_encoders()
    attn_pools = make_pools(encoders)

    raw = {k: [] for k in ARMS}
    params = {}
    params_total = {}
    print(f"{'seed':>4} | " + " | ".join(f"{k:>22}" for k in ARMS))
    print("-" * 92)
    for seed in range(seeds):
        cells = []
        for kind in ARMS:
            try:
                p_aligner, p_total, mi, mo = run_arm(kind, seed, encoders, attn_pools)
                params[kind] = p_aligner
                params_total[kind] = p_total
                raw[kind].append((mi, mo))
                cells.append(f"{p_aligner:>9,} {mi:>5.3f}/{mo:>5.3f}")
            except Exception as e:
                raw[kind].append((float("nan"), float("nan")))
                cells.append(f"{'FAILED':>22}")
                print(f"  [FAIL {kind}] {type(e).__name__}: {e}")
        print(f"{seed:>4} | " + " | ".join(cells))

    # ── 汇总 ──────────────────────────────────────────────────────────────
    print("-" * 92)
    rows = []
    for kind in ARMS:
        vals = [(a, b) for a, b in raw[kind] if not np.isnan(a)]
        if not vals:
            continue
        rows.append({"arm": kind, "params": params[kind],
                     "mse_in_sample": float(np.mean([a for a, _ in vals])),
                     "mse_heldout": float(np.mean([b for _, b in vals]))})
    shared = (params_total["attn"] - params["attn"]) if "attn" in params_total else 0
    print(f"\n(各臂共享的 AlignedFusionLayer 自带可训练参数 = {shared:,}"
          f"; 表中只列**对齐器**参数量)")
    print("\n均值 (in-sample / held-out):")
    print(ladder_table(rows, base_arm="attn"))

    attn_row = next((r for r in rows if r["arm"] == "attn"), None)
    n1_row = next((r for r in rows if r["arm"] == "mixture-n1"), None)
    print("\n=== E2 裁决: 等容量对 (attn vs mixture-n1) ===")
    if attn_row and n1_row:
        assert attn_row["params"] == n1_row["params"], "等容量对构造失效"
        print(f"  参数量: {attn_row['params']:,} == {n1_row['params']:,}  ✓ 容量严格相同")
        for k, lbl in (("mse_in_sample", "in-sample"), ("mse_heldout", "held-out ")):
            a, m = attn_row[k], n1_row[k]
            d = (a - m) / a * 100
            print(f"  {lbl}: attn {a:.4f} vs mixture-n1 {m:.4f}  → mixture 优 {d:+.1f}%")
    print("\n=== 泄漏裁决: in-sample 与 held-out 的落差 ===")
    for r in sorted(rows, key=lambda x: x["params"]):
        gap = r["mse_heldout"] - r["mse_in_sample"]
        rel = gap / max(r["mse_in_sample"], 1e-9) * 100
        print(f"  {r['arm']:<14} 落差 {gap:+.4f} ({rel:+.1f}%)")
    print("=" * 92)

    payload = {
        "experiment": "v41_capacity_audit",
        "seeds": seeds,
        "constants": {"D_SHARED": D_SHARED, "S": S, "STEPS": STEPS, "LR": LR,
                      "HELDOUT_OFFSET": HELDOUT_OFFSET},
        "params_aligner": params,
        "params_total": params_total,
        "shared_fusion_params": shared,
        "rows": rows,
        "raw": {k: v for k, v in raw.items()},
    }
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "research", "aligner", "v41_capacity_audit", "results", "raw.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"写入: {out}")


if __name__ == "__main__":
    main()
