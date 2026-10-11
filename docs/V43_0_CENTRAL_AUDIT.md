# V43.0 — 中枢机制审计:容量 / 结构 / 泄漏

> **本文件是薄指针 (thin pointer), 非权威副本.**
> 权威规范: [`research/routing_evolution/v43_central_audit/SPEC.md`](../research/routing_evolution/v43_central_audit/SPEC.md)
> 机器可读声明: [`research/routing_evolution/v43_central_audit/MANIFEST.json`](../research/routing_evolution/v43_central_audit/MANIFEST.json)
> 指标快照: [`research/routing_evolution/v43_central_audit/results/metrics.json`](../research/routing_evolution/v43_central_audit/results/metrics.json)

## 一句话结论

> gate 的中枢参数(512)是 broadcast(1,024)的一半 ⇒ 容量混杂方向与担心相反; 完美退化与保序**成立**(构造性质, 泄漏无法伪造); 但 +44.5% 的 in-sample 增益在 held-out 塌到 **+4.0%(3/5)**。

## 复现

```bash
python3 examples/run_v43_central_audit.py
```
