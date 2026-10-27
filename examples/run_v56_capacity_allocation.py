"""V56.0 — 维度作为容量分配: hidden 宽度 × 投影方式 × 头类型 的析因.

> 归属路线: scale_heterogeneous (容器范式), v39 的直接后继
> 复现: python examples/run_v56_capacity_allocation.py
> 缓存: h_base / c_passive 落 npz, 改超参重跑不需要再加载容器

复用 v39 的协议 (不重新发明): Adult XOR 数据 / MSA-4B 浅层 h_base /
TabLDM 被动入口 c_passive。**只把头宽度与投影方式变成变量。**

网格: 2 表征(proj/native) × 2 头(linear/mlp) × 10 宽度 × 2 臂(A/C) × 5 seeds
"""
import gc
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from research.scale_heterogeneous.v56_capacity_allocation.audit import (
    WIDTHS, REPRS, HEADS, PREDICTIONS, D_ENTRY, D_BIG, N_CLS,
    build_input, make_head, n_params, train_head,
    hidden_act_rank, block_ablate, rank_report, eff_rank_pr,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "research/scale_heterogeneous/v56_capacity_allocation/results")
CACHE = os.environ.get("V56_CACHE", "/tmp/v56_repr.npz")
ART = os.environ.get("V56_ART_DIR", OUT)

N_TOTAL = int(os.environ.get("V56_N", "512"))
STEPS = int(os.environ.get("V56_STEPS", "500"))
SEEDS = int(os.environ.get("V56_SEEDS", "5"))
N_BIG_LAYERS = int(os.environ.get("V56_BIG_LAYERS", "1"))
DATASET = os.environ.get("V56_DATASET", "adult").lower()
S_LEN = 32
SEL_WIDTHS = [int(x) for x in os.environ.get("V56_WIDTHS", ",".join(map(str, WIDTHS))).split(",")]


# ------------------------------------------------------------------ 表征 (算一次, 缓存)

def container_text_repr(big_model, tok, texts):
    ids = tok(texts, padding="max_length", truncation=True, max_length=S_LEN,
              return_tensors="pt").input_ids
    with torch.no_grad():
        return big_model(ids).last_hidden_state.float().mean(dim=1)


def tabldm_passive(X_table):
    from tabldm import TabLDMClassifier
    from research.scale_heterogeneous.v39_g2_interaction.paths import TABLDM_CKPT
    clf = TabLDMClassifier(model_path=TABLDM_CKPT, device="cpu")
    d = X_table.shape[1]
    rng = np.random.RandomState(42)
    clf.fit(rng.randn(12, d).astype(np.float32), rng.randint(0, 10, 12))
    m = clf.model_
    reps = []
    h = m.icl_predictor.register_forward_hook(
        lambda mod, inp, out: reps.append(inp[0].detach() if isinstance(inp, tuple) else inp.detach()))
    clf.predict_proba(X_table.numpy())
    h.remove()
    r = reps[0]
    n = X_table.shape[0]
    rep = r[:, -n:, :].mean(dim=0).float()
    del clf, m
    gc.collect()
    return rep


def get_reprs():
    """返回 (h_base, c_passive, y)。优先读缓存 —— 改头宽度不该重跑 4B 容器。"""
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        print(f"  缓存命中: {CACHE}  h_base={z['h_base'].shape} c={z['c'].shape}")
        return (torch.from_numpy(z["h_base"]), torch.from_numpy(z["c"]),
                torch.from_numpy(z["y"]).long())

    from research.scale_heterogeneous.v39_g2_interaction import DATASETS, MSA_DIR, load_big_container
    from transformers import AutoTokenizer

    load_fn, ds_name = DATASETS[DATASET]
    texts, X, y = load_fn(N_TOTAL, 42, decouple=True)
    print(f"  {ds_name}: {len(texts)} 样本, 正样本率 {y.float().mean():.2%}")

    t0 = time.time()
    big = load_big_container(n_layers=N_BIG_LAYERS)
    tok = AutoTokenizer.from_pretrained(MSA_DIR)
    h_base = container_text_repr(big, tok, texts)
    del big
    gc.collect()
    print(f"  容器表征 {tuple(h_base.shape)} ({time.time()-t0:.0f}s)")

    t0 = time.time()
    c = tabldm_passive(X)
    print(f"  TabLDM 被动入口 {tuple(c.shape)} ({time.time()-t0:.0f}s)")

    os.makedirs(os.path.dirname(CACHE) or ".", exist_ok=True)
    np.savez(CACHE, h_base=h_base.numpy(), c=c.numpy(), y=y.numpy())
    print(f"  已缓存 → {CACHE}")
    return h_base, c, y


# ------------------------------------------------------------------ 主流程

def main():
    t0 = time.time()
    print("=" * 82)
    print("V56.0 — 维度作为容量分配 (hidden 宽度 × 投影方式 × 头类型)")
    print("=" * 82)
    print("问题: v39 把 hidden=128 与 ProjHead(512→2560) 当成固定超参;")
    print("      本代把两者变成变量, 问「宽度是否真的是约束」")
    print(f"网格: {len(REPRS)} 表征 × {len(HEADS)} 头 × {len(SEL_WIDTHS)} 宽度 × 2 臂 × {SEEDS} seeds")
    print()

    h_base, c, y = get_reprs()
    rng = np.random.RandomState(1234)
    perm = rng.permutation(len(y))
    tr_idx = torch.tensor(perm[:-128])
    te_idx = torch.tensor(perm[-128:])
    print(f"  {len(tr_idx)}/{len(te_idx)} 划分 (与 v39 同构造)\n")

    # ---- 维度诊断: 「维度」这个概念的可执行形式 ----
    print("=" * 82)
    print("维度诊断 (Q3 的前提: 输入本身占几个方向)")
    print("=" * 82)
    ranks = [rank_report("h_base (容器输出)", h_base),
             rank_report("c_passive (TabLDM)", c)]
    cat_native = torch.cat([h_base, c], 1)
    ranks.append(rank_report("concat native", cat_native))
    for r in ranks:
        print(f"  {r['name']:<22} ambient={r['ambient_dim']:>5}  "
              f"eff_rank(PR)={r['eff_rank_pr']:>7.2f}  "
              f"eff_rank(熵)={r['eff_rank_entropy']:>7.2f}  "
              f"占用 {r['occupied_frac']*100:.2f}%")

    # 每侧对标签的可分性 (单侧天花板)
    print("\n  单侧线性探针 (XOR 下应接近 0.5):")
    for nm, Z in (("h_base only", h_base), ("c only", c)):
        torch.manual_seed(0)
        _, a = train_head(Z, y, tr_idx, te_idx, head_mode="linear", steps=300)
        print(f"    {nm:<14} acc={a:.3f}")

    # ---- 主网格 ----
    results = {}
    zero_c = torch.zeros_like(c)
    for repr_mode in REPRS:
        torch.manual_seed(0)
        P = torch.nn.Linear(D_ENTRY, D_BIG)
        torch.nn.init.orthogonal_(P.weight)
        torch.nn.init.zeros_(P.bias)
        for p in P.parameters():
            p.requires_grad_(False)

        for head_mode in HEADS:
            for w in (SEL_WIDTHS if head_mode == "mlp" else [0]):
                key = f"{repr_mode}|{head_mode}|{w}"
                accs, drops_e, drops_h, act_ranks, npar = {}, {}, {}, {}, {}
                for arm, cc in (("A", zero_c), ("C", c)):
                    for s in range(SEEDS):
                        x, d_hb, d_en = build_input(h_base, cc, repr_mode, P)
                        head, a = train_head(x, y, tr_idx, te_idx, hidden=w,
                                             head_mode=head_mode, steps=STEPS, seed=s)
                        accs.setdefault(arm, []).append(a)
                        ra = hidden_act_rank(head, x)
                        if ra is not None:
                            act_ranks.setdefault(arm, []).append(ra)
                        if s == 0:
                            bl = block_ablate(head, x, y, te_idx, d_hb, d_en)
                            drops_e.setdefault(arm, []).append(bl["drop_entry"])
                            drops_h.setdefault(arm, []).append(bl["drop_hbase"])
                            npar[arm] = n_params(head)
                results[key] = {
                    "repr": repr_mode, "head": head_mode, "hidden": w,
                    "acc_A": statistics.mean(accs["A"]),
                    "acc_C": statistics.mean(accs["C"]),
                    "acc_C_sd": statistics.stdev(accs["C"]) if SEEDS > 1 else 0.0,
                    "gain": statistics.mean(accs["C"]) - statistics.mean(accs["A"]),
                    "act_rank_C": (statistics.mean(act_ranks["C"]) if "C" in act_ranks else None),
                    "drop_entry_C": (statistics.mean(drops_e["C"]) if "C" in drops_e else None),
                    "drop_hbase_C": (statistics.mean(drops_h["C"]) if "C" in drops_h else None),
                    "params": npar.get("C"),
                }
                r = results[key]
                print(f"  {repr_mode:<7} {head_mode:<7} w={w:<5} "
                      f"A={r['acc_A']:.3f} C={r['acc_C']:.3f} (sd {r['acc_C_sd']:.3f}) "
                      f"gain={r['gain']:+.3f} "
                      f"act_rank={r['act_rank_C'] if r['act_rank_C'] is None else round(r['act_rank_C'],1)} "
                      f"drop(entry)={r['drop_entry_C']:+.3f} "
                      f"params={r['params']}")

    # ---- 判决 ----
    print("\n" + "=" * 82)
    print("预登记判决 P1–P4")
    print("=" * 82)
    lin = [v["acc_C"] for k, v in results.items() if v["head"] == "linear"]
    mlp = {v["hidden"]: v["acc_C"] for k, v in results.items()
           if v["head"] == "mlp" and v["repr"] == "native"}
    p1 = max(lin) <= 0.80
    best = max(mlp.values()) if mlp else 0
    p2 = (mlp.get(16, 0) >= 0.95 * best) if mlp else False
    proj_v = {v["hidden"]: v["acc_C"] for k, v in results.items()
              if v["head"] == "mlp" and v["repr"] == "proj"}
    diffs = [abs(proj_v[w] - mlp[w]) for w in mlp if w in proj_v]
    p3 = max(diffs) <= 0.03 if diffs else False
    big_w = [w for w in mlp if w >= 1024]
    p4 = all(results[f"native|mlp|{w}"]["act_rank_C"] is not None and
             results[f"native|mlp|{w}"]["act_rank_C"] < 0.05 * w for w in big_w) if big_w else False

    print(f"  P1 linear 头解不开 XOR (max={max(lin):.3f} ≤ 0.80): {'过' if p1 else '未过'}")
    print(f"  P2 hidden=16 已达最好宽度的 95% ({mlp.get(16, 0):.3f} vs {best:.3f}): "
          f"{'过' if p2 else '未过'}")
    print(f"  P3 proj vs native 差 ≤0.03 (max diff={max(diffs):.3f}): {'过' if p3 else '未过'}")
    print(f"  P4 大宽度下激活有效秩 < 宽度 5%: {'过' if p4 else '未过'}")

    os.makedirs(ART, exist_ok=True)
    payload = {"schema_version": "1.0", "version": "56.0",
               "package": "v56_capacity_allocation", "line": "scale_heterogeneous",
               "config": {"n": N_TOTAL, "steps": STEPS, "seeds": SEEDS,
                          "big_layers": N_BIG_LAYERS, "dataset": DATASET,
                          "widths": SEL_WIDTHS, "reprs": list(REPRS), "heads": list(HEADS)},
               "rank_diagnostics": ranks,
               "predictions": PREDICTIONS,
               "verdicts": {"P1": p1, "P2": p2, "P3": p3, "P4": p4},
               "results": results}
    mpath = os.path.join(ART, "metrics.json")
    json.dump(payload, open(mpath, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n落盘: {mpath}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
