"""严格验证 torch.compile 的数值正确性 — 上一轮 NaN 事故的追查。

背景:
  ablate2 测出 compile(reduce-overhead) 26.3 ms (3.9×), 但**未检查输出是否有限**。
  随后 domain_control_compiled 训练时 3/5 种子在域激活变体上出现 NaN
  (none 变体全正常, 因其 alpha 冻结且 gate=0 绕开了域路径)。
  => 那个 3.9× 不可信, 必须先证明 compile 输出正确。

本脚本对每种 mode 做三件事:
  1. 前向输出: NaN/Inf 检查 + 与 eager 的 max|diff|
  2. 反向梯度: NaN 检查 + 与 eager 的 grad 差异
  3. 只有通过 1、2 才测速
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
from fast_scan import chunked_scan  # noqa: E402
from domain_control_compiled import CompiledSSM, Model, K  # noqa: E402

L, B, STEPS = 80, 32, 20


def finite(t):
    return bool(torch.isfinite(t).all().item())


def run_eager(dev, seed=0):
    """eager 参考: 前向 + 反向, 返回 (y, grads)。"""
    torch.manual_seed(seed)
    m = Model().to(dev)
    # 模拟训练后的域分化状态 (alpha 非零) — 正是出 NaN 的条件
    with torch.no_grad():
        for layer in m.layers:
            layer.alpha.normal_(0, 0.5)
    u = torch.randn(B, L, device=dev)
    ids = torch.randint(0, K, (B, L), device=dev)
    tgt = torch.randn(B, L, device=dev)
    y = m(u, ids)
    loss = F.mse_loss(y, tgt)
    loss.backward()
    return y.detach(), {n: p.grad.detach().clone() for n, p in m.named_parameters()
                        if p.grad is not None}


def run_compiled(dev, mode, seed=0):
    torch.manual_seed(seed)
    m = Model().to(dev)
    with torch.no_grad():
        for layer in m.layers:
            layer.alpha.normal_(0, 0.5)
    cm = torch.compile(m, mode=mode)
    u = torch.randn(B, L, device=dev)
    ids = torch.randint(0, K, (B, L), device=dev)
    tgt = torch.randn(B, L, device=dev)
    # 多跑几次让 CUDA graph 录制完成
    for _ in range(3):
        y = cm(u, ids)
    loss = F.mse_loss(y, tgt)
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in m.named_parameters()
             if p.grad is not None}
    return y.detach(), grads


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print(f"torch.compile 数值验证 | {dev} {torch.cuda.get_device_name(0) if dev.type=='cuda' else ''}")
    print(f"torch {torch.__version__}")
    print("=" * 72)

    y_ref, g_ref = run_eager(dev)
    print(f"\n[eager] 输出 finite = {finite(y_ref)}  |max| = {y_ref.abs().max().item():.3e}")

    out = {"eager_finite": finite(y_ref)}
    for mode in ["default", "reduce-overhead", "max-autotune"]:
        print(f"\n--- mode = {mode} ---")
        try:
            y, g = run_compiled(dev, mode)
            fy = finite(y)
            dy = (y_ref - y).abs().max().item() if fy else float("nan")

            # 梯度检查
            bad_grads = []
            worst_g = 0.0
            for name, gr in g_ref.items():
                if name not in g:
                    bad_grads.append(f"{name}:missing")
                    continue
                gg = g[name]
                if not finite(gg):
                    bad_grads.append(f"{name}:nan")
                    continue
                d = (gr - gg).abs().max().item()
                worst_g = max(worst_g, d)

            ok = fy and dy < 1e-4 and not bad_grads and worst_g < 1e-4
            out[mode] = {"forward_finite": fy, "forward_maxdiff": dy,
                         "grad_maxdiff": worst_g, "bad_grads": bad_grads[:3],
                         "ok": ok}
            print(f"  前向 finite={fy}  max|Δ|={dy:.3e}")
            print(f"  梯度 max|Δ|={worst_g:.3e}  异常梯度 {len(bad_grads)} 个"
                  + (f" (例: {bad_grads[:2]})" if bad_grads else ""))
            print(f"  判定: {'✅ 数值正确' if ok else '❌ 数值不可用'}")

            if ok:
                torch.manual_seed(0)
                m = Model().to(dev)
                with torch.no_grad():
                    for layer in m.layers:
                        layer.alpha.normal_(0, 0.5)
                cm = torch.compile(m, mode=mode)
                u = torch.randn(B, L, device=dev)
                ids = torch.randint(0, K, (B, L), device=dev)
                tgt = torch.randn(B, L, device=dev)
                opt = torch.optim.AdamW(cm.parameters(), lr=1e-3)
                for _ in range(5):
                    loss = F.mse_loss(cm(u, ids), tgt)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(STEPS):
                    loss = F.mse_loss(cm(u, ids), tgt)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                ms = (time.time() - t0) / STEPS * 1000
                out[mode]["ms"] = round(ms, 1)
                print(f"  测速: {ms:.1f} ms/步")
        except Exception as e:
            print(f"  ❌ 失败: {type(e).__name__}: {str(e)[:200]}")
            out[mode] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

    # eager 基准测速
    torch.manual_seed(0)
    m = Model().to(dev)
    with torch.no_grad():
        for layer in m.layers:
            layer.alpha.normal_(0, 0.5)
    u = torch.randn(B, L, device=dev)
    ids = torch.randint(0, K, (B, L), device=dev)
    tgt = torch.randn(B, L, device=dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(3):
        loss = F.mse_loss(m(u, ids), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(STEPS):
        loss = F.mse_loss(m(u, ids), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    eager_ms = (time.time() - t0) / STEPS * 1000
    out["eager_ms"] = round(eager_ms, 1)
    print(f"\n[eager 测速] {eager_ms:.1f} ms/步")
    for mode in ["default", "reduce-overhead", "max-autotune"]:
        if isinstance(out.get(mode), dict) and "ms" in out[mode]:
            print(f"  {mode}: {eager_ms / out[mode]['ms']:.2f}× vs eager")

    print("\n--- JSON ---")
    print(json.dumps(out, ensure_ascii=False))
    print("完成。")


if __name__ == "__main__":
    main()
