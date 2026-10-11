# V44.0 — 四种中枢模式的容量-泄漏审计

> **本文件是薄指针 (thin pointer), 非权威副本.**
> 权威规范: [`research/central_stability/v44_hub_modes_audit/SPEC.md`](../research/central_stability/v44_hub_modes_audit/SPEC.md)
> 机器可读声明: [`research/central_stability/v44_hub_modes_audit/MANIFEST.json`](../research/central_stability/v44_hub_modes_audit/MANIFEST.json)
> 指标快照: [`research/central_stability/v44_hub_modes_audit/results/metrics.json`](../research/central_stability/v44_hub_modes_audit/results/metrics.json)

## 一句话结论

> 主表称 adaptive-ema 在 v22 上最低(0.2972), 但 **in-sample 最优是 router-norm**(0.2835)、**held-out 最优也是它**(0.3342, 5/5 全胜); adaptive-ema held-out 仅 **+0.3%** 且泛化落差 25.0% 最差 ⇒ **主表结论不成立**。

## 复现

```bash
python3 examples/run_v44_hub_modes_audit.py
```
