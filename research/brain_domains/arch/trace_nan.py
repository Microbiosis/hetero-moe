"""追查训练期 NaN: eager vs compile(default) vs compile(reduce-overhead)。

上一轮的证据链:
  - 单步验证 (3 fwd + 1 bwd): compile 三种 mode 全部数值正确 (Δ~6e-8)
  - 900 步训练: real/shuffled (α 可训练) 在 3/5 种子 NaN; none (α 冻结) 全部正常
  => NaN 只在「长时间训练 + α 可训练」时出现。

本脚本用同一 seed/数据/步数, 逐 100 步记录 loss 与各参数范数,
定位 NaN 首次出现的时刻与哪个量先炸。
"""

from __future__ import annotations

import json
import math
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from domain_control_compiled import (Model, make_env, gen_data, K, L,  # noqa: E402
                                     N_TRAIN, N_EVAL, BATCH, LR, STEPS, D_INNER)

SEED = 1          # 上一轮 NaN 的种子
VARIANT = "real"  # 上一轮 NaN 的变体


def diag(model, u, ids, x, dev):
    """返回 loss 与关键量范数。"""
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    with torch.no_grad():
        y = m(u, ids)
        loss = F.mse_loss(y, x).item()
        finite = torch.isfinite(y).all().item()
        stats = {}
        for i, layer in enumerate(m.layers[:1]):
            stats["alpha_max"] = layer.alpha.abs().max().item()
            stats["U_max"] = layer.U.abs().max().item()
            stats["gamma_max"] = layer.gamma.abs().max().item()
            stats["A_log_max"] = layer.A_log.abs().max().item()
            stats["dtw_max"] = layer.dt_proj.weight.abs().max().item()
            # 关键: 有效 A 的范围与单块累积指数
            a_sh = -torch.exp(layer.A_log)
            off = torch.einsum("blr,rdn->bldn", layer.alpha[ids] * layer.domain_gate, layer.U)
            phi = torch.exp(torch.tanh(off) * layer.C_BOUND)
            a_eff = a_sh.unsqueeze(0).unsqueeze(0) * phi
            stats["a_eff_min"] = a_eff.min().item()
            stats["a_eff_max"] = a_eff.max().item()
            if torch.isfinite(a_eff).all():
                # 块内最负累积 (block=32)
                stats["blk_cummin"] = (a_eff.min() * 0.1 * 32).item()
                stats["exp_of_cummin"] = math.exp(max(-700, stats["blk_cummin"]))
    return loss, bool(finite), stats


def train(mode, dev, steps=STEPS, log_every=100):
    torch.manual_seed(SEED)
    model = Model().to(dev)
    if mode != "eager":
        model = torch.compile(model, mode=mode)

    env_a, env_b = make_env(SEED)
    ids_tr, u_tr, x_tr = gen_data(env_a, env_b, N_TRAIN, 10_000 + SEED)
    ids_tr, u_tr, x_tr = ids_tr.to(dev), u_tr.to(dev), x_tr.to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    gb = torch.Generator().manual_seed(40_000 + SEED)

    print(f"\n########## mode = {mode} ##########")
    log = []
    nan_at = None
    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        idx = torch.randint(0, N_TRAIN, (BATCH,), generator=gb).to(dev)
        pred = model(u_tr[idx], ids_tr[idx])
        loss = F.mse_loss(pred, x_tr[idx])

        if not torch.isfinite(loss):
            nan_at = step
            print(f"  [{mode}] step {step}: **loss 变 NaN** — 停止")
            break

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):
            nan_at = step
            print(f"  [{mode}] step {step}: **grad_norm = {gn.item()}** — 停止")
            break
        opt.step()

        if step % log_every == 0 or step == 1:
            l, fin, st = diag(model, u_tr[:64], ids_tr[:64], x_tr[:64], dev)
            log.append({"step": step, "loss": round(l, 6), "finite": fin,
                        "gnorm": round(gn.item(), 4),
                        "alpha_max": round(st.get("alpha_max", 0), 4),
                        "U_max": round(st.get("U_max", 0), 4),
                        "a_eff_min": round(st.get("a_eff_min", 0), 3),
                        "blk_cummin": round(st.get("blk_cummin", 0), 1),
                        "exp_cummin": f"{st.get('exp_of_cummin', 0):.2e}"})
            print(f"  step {step:4d} loss {l:.5f} finite={fin} | "
                  f"|alpha|max {st.get('alpha_max',0):.3f} |U|max {st.get('U_max',0):.3f} | "
                  f"a_eff_min {st.get('a_eff_min',0):.2f} | 块累积 {st.get('blk_cummin',0):.1f} | "
                  f"exp {st.get('exp_of_cummin',0):.1e}")

    dt = time.time() - t0
    print(f"  [{mode}] {dt:.1f}s  ({dt/max(1,step)*1000:.1f} ms/步)  nan_at={nan_at}")
    return {"mode": mode, "nan_at": nan_at, "elapsed_s": round(dt, 1),
            "ms_per_step": round(dt / max(1, step) * 1000, 1), "log": log}


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print(f"训练期 NaN 追查 | {dev} {torch.cuda.get_device_name(0) if dev.type=='cuda' else ''}")
    print(f"torch {torch.__version__} | seed={SEED} variant={VARIANT} steps={STEPS}")
    print("=" * 72)

    out = []
    for mode in ["eager", "default", "reduce-overhead"]:
        try:
            out.append(train(mode, dev))
        except Exception as e:
            print(f"  [{mode}] 崩溃: {type(e).__name__}: {str(e)[:200]}")
            out.append({"mode": mode, "error": f"{type(e).__name__}: {str(e)[:200]}"})

    print("\n===== 汇总 =====")
    for r in out:
        if "error" in r:
            print(f"  {r['mode']:18s} 崩溃: {r['error'][:80]}")
        else:
            print(f"  {r['mode']:18s} nan_at={str(r['nan_at']):>6s}  "
                  f"{r['ms_per_step']:6.1f} ms/步")

    print("\n--- JSON ---")
    print(json.dumps(out, ensure_ascii=False))
    print("完成。")


if __name__ == "__main__":
    main()
