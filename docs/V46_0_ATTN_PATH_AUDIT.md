# V46.0 — 「增益集中在 attention 通路」的样本外审计

> **本文件是薄指针 (thin pointer), 非权威副本.**
> 权威规范: [`research/routing_evolution/v46_attn_path_audit/SPEC.md`](../research/routing_evolution/v46_attn_path_audit/SPEC.md)
> 机器可读声明: [`research/routing_evolution/v46_attn_path_audit/MANIFEST.json`](../research/routing_evolution/v46_attn_path_audit/MANIFEST.json)
> 指标快照: [`research/routing_evolution/v46_attn_path_audit/results/metrics.json`](../research/routing_evolution/v46_attn_path_audit/results/metrics.json)

## 一句话结论

> attn 相对 ffn 的通路差: in-sample **+82.7%(5/5)**, held-out **+70.6%(5/5)** ⇒ **主张成立**, 不依赖 in-sample 口径。

## 复现

```bash
python3 examples/run_v46_attn_path_audit.py
```
