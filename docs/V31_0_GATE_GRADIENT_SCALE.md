# V31.0 — 梯度重缩放门控中枢

> **本文件是薄指针 (thin pointer), 非权威副本.**
> 权威规范: [`research/central_stability/v31_gate_gradient_scale/SPEC.md`](../research/central_stability/v31_gate_gradient_scale/SPEC.md)
> 机器可读声明: [`research/central_stability/v31_gate_gradient_scale/MANIFEST.json`](../research/central_stability/v31_gate_gradient_scale/MANIFEST.json)
> 指标快照: [`research/central_stability/v31_gate_gradient_scale/results/metrics.json`](../research/central_stability/v31_gate_gradient_scale/results/metrics.json)

## 机制

在 tanh 前对温度标量做梯度重缩放, 修复 v9 的 tanh 饱和:

```
t  = W_t · c
t' = t / (1 + |t|)                    压缩进 (-1, 1)
τ  = 1 + α · tanh(t')
z' = z / τ
```

## 与 v9 / v30 的对照(均由包内单测实测)

| | 完美退化 | 饱和区梯度 | 渐近衰减 |
|---|---|---|---|
| v9 纯 tanh | ✓ | **0.000e+00** | 指数到精确 0 |
| v30 softplus | **✗**(初始 temperature=1.0693) | 存活(正尾) | 正尾 O(1) / 负尾 float32 下溢为 0 |
| **v31 梯度重缩放** | **✓**(实测偏差 0.000000) | **存活**(6.7e-01) | **~1/t 多项式**(g·t 在 t≥25 后 ≈ 常数) |

> ⚠️ 本包 workspace.py 的 docstring 称「梯度 ∂scaled/∂t_scalar 始终有下界(不会为 0)」——
> **实测为 ~1/t 多项式衰减, 仍趋于 0**, 只是比 tanh 的指数慢。该过强主张已由
> `tests/test_v31_gate_gradient_scale.py` 第 7 项钉成事实。

## 单元测试

```bash
python3 tests/test_v31_gate_gradient_scale.py   # 9 项
```
