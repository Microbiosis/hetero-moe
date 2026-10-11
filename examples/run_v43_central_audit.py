"""v43.0 — 中枢机制的结构-容量-泄漏审计 (runner).

审计 ``v9_gate_central``: 论文 ``neurips_2025.tex`` 整篇建立在它的核心主张上 ——
"加性广播(v8)摧毁路由, 温度调制(v9)修复它"。

三条审计:

1. **容量**  不需要实验: v9 gate 的中枢参数 (512) 是 v8 broadcast (1024) 的**一半**,
   混杂方向与担心的相反。见包内 ``workspace_audit`` 与单测。
2. **容量与机制是否可分离**  不需要实验: 把 gate 展宽到 broadcast 的容量必须改用
   逐专家温度, 随机检验显示保序破坏率从 0% 升到 ~4.7%。
3. **泄漏**  **需要实验**: ``examples/run_v9_gate_layer.py:121-124`` 在训练同源数据上
   评估。本 runner 对每个臂同时报 in-sample 与 held-out。

协议复用: 层构造、训练器、数据构造全部 import ``examples/run_v9_gate_layer.py``,
只新增 held-out 评估。

用法::

    python3 examples/run_v43_central_audit.py                # 3 臂 × 5 seeds
    V43_SEEDS=1 python3 examples/run_v43_central_audit.py    # 冒烟
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from research.routing_evolution.v43_central_audit import (
    broadcast_params, capacity_ratio, gate_params, is_identity_at_init,
    order_break_rate,
)

from examples.run_v9_gate_layer import make_layer, make_trainer
from examples.run_v8_full import (
    D_SHARED, N_CLS, B, S, STEPS, LR, SEEDS,
    load_encoders, make_pools, encode_text, encode_code, encode_image,
)

MODES = ["v7-baseline", "v9-gate-c1", "v9-gate-full"]
HELDOUT_OFFSET = 1000


def build_data(seed, encoders):
    """与 v9 的 run_seed 同构的数据构造; 换一个足够大的 seed 即得到同分布不同抽样。"""
    (bert_tok, bert, _, _), (llama_tok, llama, _, _), (vit_proc, vit, _, _) = encoders
    texts = ["cat dog bird", "animal pet wild", "feline canine fowl"]
    h_text = encode_text(texts, bert_tok, bert)
    codes = ["def hello():", "return True", "pass None"]
    h_code = encode_code(codes, llama_tok, llama)
    img = np.random.randint(0, 256, (224, 224, 3))
    h_img = encode_image(img, vit_proc, vit)

    modal_seqs, modal_indices = [], []
    for cls, h_block in enumerate([h_text, h_code, h_img]):
        n = h_block.shape[0]
        for i in range(6):
            src = h_block[i % n]
            g = torch.Generator().manual_seed(seed * 10 + cls * 6 + i)
            modal_seqs.append(src + 0.1 * torch.randn(src.shape, generator=g))
            modal_indices.append(cls)

    targets = []
    for cls in range(N_CLS):
        ds = cls * (D_SHARED // N_CLS); de = (cls + 1) * (D_SHARED // N_CLS)
        t = torch.zeros(S, D_SHARED); t[:, ds:de] = 1.0
        for _ in range(6):
            targets.append(t)
    return modal_seqs, modal_indices, targets


def project(modal_seqs, modal_indices, layer):
    return [F.linear(h.unsqueeze(0), layer.P_m[cls])
            for h, cls in zip(modal_seqs, modal_indices)]


def evaluate(layer, x_list, targets):
    with torch.no_grad():
        return sum(F.mse_loss(layer(x), t.unsqueeze(0)).item()
                   for x, t in zip(x_list, targets)) / B


def run_seed(mode, seed, encoders, attn_pools):
    torch.manual_seed(seed)
    modal_seqs, modal_indices, targets = build_data(seed, encoders)
    layer = make_layer(mode, attn_pools, encoders)
    x_train = project(modal_seqs, modal_indices, layer)

    trainer = make_trainer(layer)
    for _ in range(STEPS):
        trainer.step(x_train, [t.unsqueeze(0) for t in targets])

    mse_in = evaluate(layer, x_train, targets)

    ho_seqs, ho_idx, ho_tgt = build_data(seed + HELDOUT_OFFSET, encoders)
    x_ho = project(ho_seqs, ho_idx, layer)
    mse_out = evaluate(layer, x_ho, ho_tgt)

    # 结构性质 (只在 gate 臂上测)
    struct = {}
    for name in ("cw_attn", "cw_ffn"):
        if hasattr(layer, name):
            ws = getattr(layer, name)
            ok, dev = is_identity_at_init(ws, torch.zeros(1, 1, 4))
            struct[f"{name}_init_identity"] = {"identity": ok, "max_dev": dev}
    return mse_in, mse_out, struct


def main():
    seeds = int(os.environ.get("V43_SEEDS", SEEDS))
    print("=" * 90)
    print("V43.0 — 中枢机制审计: 容量 / 结构 / 泄漏")
    print("=" * 90)
    d, m = D_SHARED, len(MODES)
    print(f"容量 (d_shared={d}, M=3):")
    print(f"  v8 BroadcastWorkspace = {broadcast_params(d,3)['total']:,}  = c[{d}] + U[3,{d}]")
    print(f"  v9 GateStyleWorkspace = {gate_params(d,3)['total']:,}  = c[{d}] + W_t[{d}]"
          f"   → 比值 {capacity_ratio(d,3):.2f}×  (混杂方向与担心的相反)")
    br = order_break_rate(n_trials=20000)
    print(f"保序性随机检验 ({br['n_trials']} 次):")
    print(f"  标量温度 (gate 实际形态)      : 破坏率 {br['scalar_break_rate']:.3f}")
    print(f"  逐专家温度 (展宽到 broadcast 容量): 破坏率 {br['wide_break_rate']:.3f}")
    print(f"  ⇒ 买到容量就必须放弃保序 —— 容量与机制在中枢这里**不可分离**")
    print()
    print(f"泄漏: v9 runner 在训练同源数据上评估, 本 runner 另报 held-out")
    print(f"{len(MODES)} 臂 × {seeds} seeds")
    print("-" * 90)

    encoders = load_encoders()
    attn_pools = make_pools(encoders)

    raw = {k: [] for k in MODES}
    struct = None
    for seed in range(seeds):
        for mode in MODES:
            try:
                mi, mo, st_struct = run_seed(mode, seed, encoders, attn_pools)
                raw[mode].append((mi, mo))
                if st_struct and struct is None:
                    struct = st_struct
                print(f"  [seed {seed}] {mode:>14}  in={mi:.4f}  held-out={mo:.4f}  "
                      f"落差={mo-mi:+.4f}")
            except Exception as e:
                raw[mode].append((float("nan"), float("nan")))
                print(f"  [seed {seed}] {mode:>14} FAILED: {type(e).__name__}: {e}")

    print("-" * 90)
    print(f"{'arm':>16} | {'in-sample 中位数':>15} | {'held-out 中位数':>15} | {'落差':>9} | {'相对落差':>9}")
    print("-" * 90)
    rows = []
    for k in MODES:
        vals = [(a, b) for a, b in raw[k] if not np.isnan(a)]
        if not vals:
            continue
        mi = statistics.median([a for a, _ in vals])
        mo = statistics.median([b for _, b in vals])
        rows.append({"arm": k, "in_median": mi, "out_median": mo, "gap": mo - mi})
        print(f"{k:>16} | {mi:>15.4f} | {mo:>15.4f} | {mo-mi:>+9.4f} | "
              f"{(mo-mi)/max(mi,1e-9)*100:>+8.1f}%")

    base = next((r for r in rows if r["arm"] == "v7-baseline"), None)
    if base:
        print("\n=== v9 主张在 held-out 上是否存活 ===")
        for r in rows:
            if r["arm"] == "v7-baseline":
                continue
            for lbl, key in (("in-sample ", "in_median"), ("held-out ", "out_median")):
                d_ = (base[key] - r[key]) / max(base[key], 1e-9) * 100
                print(f"  {r['arm']:<14} {lbl}: {r[key]:.4f} vs v7 {base[key]:.4f} → {d_:+.1f}%")
    if struct:
        print(f"\n=== 结构性质 (初始状态) ===")
        for k, v in struct.items():
            print(f"  {k}: identity={v['identity']}  max_dev={v['max_dev']:.3e}")
    print("=" * 90)

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "research", "routing_evolution", "v43_central_audit", "results", "raw.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"experiment": "v43_central_audit", "seeds": seeds, "modes": MODES,
                   "capacity": {"broadcast": broadcast_params(D_SHARED, 3),
                                "gate": gate_params(D_SHARED, 3),
                                "ratio": capacity_ratio(D_SHARED, 3)},
                   "order_break_rate": br,
                   "rows": rows, "struct": struct,
                   "raw": {k: v for k, v in raw.items()}}, f, indent=2, ensure_ascii=False)
    print(f"写入: {out}")


if __name__ == "__main__":
    main()
