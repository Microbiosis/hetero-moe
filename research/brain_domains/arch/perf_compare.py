"""fast_scan 优化验证 + 性能对比 (HF Jobs / T4).

三步:
  [1] 对拍: chunked_scan vs scan_reference (含非整除长度 / 多块边界)
  [2] 端到端等价: StableDomainSSM 的 scan=fast vs scan=ref 全模型前向对比
  [3] 性能: 同模型下 scan=ref vs scan=fast 的训练步速率 + profiler 内核数
"""

from __future__ import annotations

import json
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, ".")
from fast_scan import chunked_scan, scan_reference, verify_equivalence  # noqa: E402
from cloud_bench import StableDomainSSM, DomainStack  # noqa: E402

K, D_INNER, N_LAYER, RANK, D_STATE = 8, 256, 4, 8, 16


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(1, D_INNER)
        self.stack = DomainStack(N_LAYER, cls=StableDomainSSM, d_inner=D_INNER,
                                 d_state=D_STATE, n_domains=K, rank=RANK)
        self.head = nn.Linear(D_INNER, 1)

    def forward(self, u, ids, scan="fast"):
        h = self.inp(u.unsqueeze(-1))
        for layer in self.stack.layers:
            h = h + layer(h, ids, scan=scan)
        return self.head(h).squeeze(-1)


def model_equiv(dev):
    """端到端: 同一权重下 scan=fast 与 scan=ref 的前向输出对比。"""
    print("\n=== [2] 端到端等价 (全模型前向) ===")
    torch.manual_seed(3)
    model = Model().to(dev).eval()
    ok_all = True
    with torch.no_grad():
        for B, L in [(4, 80), (2, 512)]:
            u = torch.randn(B, L, device=dev) * 0.1
            ids = torch.randint(0, K, (B, L), device=dev)
            y_ref = model(u, ids, scan="ref")
            y_fast = model(u, ids, scan="fast")
            d = (y_ref - y_fast).abs().max().item()
            ok_all &= d < 1e-5
            print(f"  B={B} L={L:4d}  max|Δ| = {d:.3e}  {'PASS' if d < 1e-5 else 'FAIL'}")
    return ok_all


def bench(dev, scan, B, L, steps, warmup=3):
    torch.manual_seed(0)
    model = Model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    u = torch.randn(B, L, device=dev)
    ids = torch.randint(0, K, (B, L), device=dev)
    x = torch.randn(B, L, device=dev)

    def one():
        pred = model(u, ids, scan=scan)
        loss = torch.nn.functional.mse_loss(pred, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(warmup):
        one()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        one()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / steps * 1000.0


def profile_kernels(dev, scan, B=32, L=80, steps=3):
    from torch.profiler import profile, ProfilerActivity
    torch.manual_seed(0)
    model = Model().to(dev)
    u = torch.randn(B, L, device=dev)
    ids = torch.randint(0, K, (B, L), device=dev)
    x = torch.randn(B, L, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def one():
        pred = model(u, ids, scan=scan)
        loss = torch.nn.functional.mse_loss(pred, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    one()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t0 = time.time()
        for _ in range(steps):
            one()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        wall = time.time() - t0
    total_dev, count = 0.0, 0
    for evt in prof.key_averages():
        total_dev += getattr(evt, "self_device_time_total", 0) or 0
        count += evt.count or 0
    return wall / steps * 1000.0, count // steps, total_dev / 1e6 / steps


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"
    print("=" * 70)
    print(f"fast_scan 验证+性能对比 | {dev} {name} | torch {torch.__version__}")
    print("=" * 70)

    if not verify_equivalence(dev):
        print("\n对拍失败 — 终止。")
        sys.exit(1)
    if not model_equiv(dev):
        print("\n端到端对拍失败 — 终止。")
        sys.exit(1)

    print("\n=== [3] 性能对比 (训练步 ms; 全模型 fwd+bwd+opt) ===")
    out = {}
    for B, L, steps in [(32, 80, 20), (32, 320, 10), (64, 512, 5)]:
        ms_ref = bench(dev, "ref", B, L, steps)
        ms_fast = bench(dev, "fast", B, L, steps)
        speedup = ms_ref / ms_fast
        out[f"B{B}_L{L}"] = {"ref_ms": round(ms_ref, 1), "fast_ms": round(ms_fast, 1),
                             "speedup": round(speedup, 2)}
        print(f"  B={B:<4} L={L:<4} | ref {ms_ref:9.1f} ms | fast {ms_fast:8.1f} ms | 提速 {speedup:5.2f}×")
        if ms_ref > 3000 and steps > 5:
            break

    print("\n=== [4] profiler (B=32, L=80) ===")
    for scan in ["ref", "fast"]:
        wall_ms, kcount, cuda_ms = profile_kernels(dev, scan)
        out[f"prof_{scan}"] = {"wall_ms": round(wall_ms, 1), "kernels": kcount,
                               "cuda_busy_ms": round(cuda_ms, 1)}
        print(f"  scan={scan:4s} 墙钟 {wall_ms:8.1f} ms/步 | 内核数 {kcount:<7} | CUDA 忙 {cuda_ms:8.1f} ms")

    print("\n--- JSON ---")
    print(json.dumps(out))
    print("完成。")


if __name__ == "__main__":
    main()
