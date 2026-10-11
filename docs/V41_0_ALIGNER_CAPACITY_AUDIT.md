# V41.0 — 对齐器阶梯的容量-泄漏审计

> **本文件是薄指针 (thin pointer), 非权威副本.**
> 权威规范: [`research/aligner/v41_capacity_audit/SPEC.md`](../research/aligner/v41_capacity_audit/SPEC.md)
> 机器可读声明: [`research/aligner/v41_capacity_audit/MANIFEST.json`](../research/aligner/v41_capacity_audit/MANIFEST.json)
> 指标快照: [`research/aligner/v41_capacity_audit/results/metrics.json`](../research/aligner/v41_capacity_audit/results/metrics.json)

## 一句话结论

> MixtureAligner(n=1) 与 CrossArchAttnAligner 参数量**逐位相等**(720,640), 仍优 **+36.5%**(5/5 配对全胜) ⇒ **E2 容量混杂排除**, 对齐器阶梯成立。

## 复现

```bash
python3 examples/run_v41_capacity_audit.py
```
