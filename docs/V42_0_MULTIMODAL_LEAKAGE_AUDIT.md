# V42.0 — 多模态蒸馏泄漏审计

> **本文件是薄指针 (thin pointer), 非权威副本.**
> 权威规范: [`research/multimodal/v42_leakage_audit/SPEC.md`](../research/multimodal/v42_leakage_audit/SPEC.md)
> 机器可读声明: [`research/multimodal/v42_leakage_audit/MANIFEST.json`](../research/multimodal/v42_leakage_audit/MANIFEST.json)
> 指标快照: [`research/multimodal/v42_leakage_audit/results/metrics.json`](../research/multimodal/v42_leakage_audit/results/metrics.json)

## 一句话结论

> v21 学生损失为 `0.7·MSE(y,teacher)+0.3·MSE(y,target)` ⇒ **β_h=0.3**, 与 v16 同构。关掉泄漏后学生 0.3270 vs 教师 0.3255 = **−0.5%**, **5/5 种子不优于教师** ⇒ **+50.5% 的裕度归零**。

## 复现

```bash
python3 examples/run_v42_leakage_audit.py
```
