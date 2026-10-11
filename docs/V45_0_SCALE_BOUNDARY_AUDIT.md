# V45.0 — 语料规模边界:方向判定与样本外存活性

> **本文件是薄指针 (thin pointer), 非权威副本.**
> 权威规范: [`research/corpus_scale/v45_scale_boundary_audit/SPEC.md`](../research/corpus_scale/v45_scale_boundary_audit/SPEC.md)
> 机器可读声明: [`research/corpus_scale/v45_scale_boundary_audit/MANIFEST.json`](../research/corpus_scale/v45_scale_boundary_audit/MANIFEST.json)
> 指标快照: [`research/corpus_scale/v45_scale_boundary_audit/results/metrics.json`](../research/corpus_scale/v45_scale_boundary_audit/results/metrics.json)

## 一句话结论

> **摘要与主表方向相反**(摘要写 gate→adaptive-ema, 表是 adaptive-ema→gate), 已更正摘要。边界 in-sample 存在、held-out **不稳健**; 大规模档中位数与配对两种口径互相矛盾 ⇒ 报告为**未在样本外确立**。

## 复现

```bash
python3 examples/run_v45_scale_boundary_audit.py
```
