#!/usr/bin/env python3
r"""exp102_diag2.py — 诊断：**adapter 的权重到底动了没有？**

## 为什么必须查这个

C0 的「损失没降」有两种完全不同的成因，处置方式相反：
- **(a) 梯度不通**（bug）⇒ 例如 token 拼接处被 detach、或者 zero-init 的末层把梯度挡死
  ⇒ ⇒ **再训 10 小时也不会动** ⇒ 应当立刻改实现；
- **(b) 梯度通、只是学得慢** ⇒ ⇒ 那就只需要加训练量。

⇒ ⇒ 判据：读回 `adapter_last.pt` ⇒ 打印**各层权重的范数** + 与初始化（末层零）的偏离
⇒ ⇒ **末层权重若仍为 0 ⇒ 梯度没到 ⇒ (a)**；**若已非零 ⇒ 梯度到了 ⇒ (b)**。

## 输出

`/domains/runs/illusionbrain-align/exp102/adapter_diag2.json` + stdout
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

EXP_DIR = Path(__file__).resolve().parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/domains/runs/illusionbrain-align/exp102/adapter_last.pt")
    ap.add_argument("--out", default="/domains/runs/illusionbrain-align/exp102/adapter_diag2.json")
    args = ap.parse_args()

    import torch
    import exp102_train as T

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    step = ck.get("step")
    log(f"[ckpt] {args.ckpt}｜step={step}｜keys={len(sd)}")

    rep = {"ckpt": args.ckpt, "step": step, "layers": {}}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        rep["layers"][k] = {"shape": list(v.shape),
                            "norm": float(v.float().norm()),
                            "abs_max": float(v.float().abs().max()),
                            "nonzero_frac": float((v.float() != 0).float().mean())}
        log(f"  {k:28} shape={tuple(v.shape)} ‖w‖={rep['layers'][k]['norm']:.6f} "
            f"max|w|={rep['layers'][k]['abs_max']:.6f} 非零占比={rep['layers'][k]['nonzero_frac']:.4f}")

    # 末层（零初始化）是否已偏离 0 ⇒ 梯度是否真的到达
    last = [k for k in sd if k.endswith("net.3.weight") or k.endswith("net.3.bias")]
    moved = False
    for k in last:
        v = sd[k].float()
        nz = float(v.abs().max())
        rep["last_layer"] = {"key": k, "abs_max": nz, "nonzero_frac": float((v != 0).float().mean())}
        log(f"[末层/零初始化] {k}: max|w|={nz:.8f} 非零占比={rep['last_layer']['nonzero_frac']:.6f}")
        if nz > 0:
            moved = True
    rep["gradient_reached_adapter"] = bool(moved)
    rep["verdict"] = ("梯度到了 adapter ⇒ 只是**学得慢** ⇒ 加训练量即可"
                      if moved else
                      "**梯度没到 adapter** ⇒ 实现有 bug ⇒ 再训也没用 ⇒ 要改实现")
    log(f"[判决] {rep['verdict']}")

    Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str),
                              encoding="utf-8")
    log(f"写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
