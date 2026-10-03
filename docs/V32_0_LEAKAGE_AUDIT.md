# V32.0 — 目标泄漏审计 (target-leakage audit)

> **本文件是薄指针 (thin pointer), 非权威副本.**
> 权威规范: [`research/distillation/v32_leakage_audit/SPEC.md`](../research/distillation/v32_leakage_audit/SPEC.md)
> 机器可读声明: [`research/distillation/v32_leakage_audit/MANIFEST.json`](../research/distillation/v32_leakage_audit/MANIFEST.json)
> 指标快照: [`research/distillation/v32_leakage_audit/results/metrics.json`](../research/distillation/v32_leakage_audit/results/metrics.json)
> 权威测试: [`research/distillation/v32_leakage_audit/tests/test_package.py`](../research/distillation/v32_leakage_audit/tests/test_package.py)
> 原始证据: `research/distillation/v32_leakage_audit/results/v32_*.json`（四份，随包分发）

原因: 让每一代成为**自包含独立子包**——规范、指标快照、原始证据与测试同处一目录,
整目录拷走即可独立阅读与复核。规范见
[`docs/GENERATION_MANIFEST_SCHEMA.md`](GENERATION_MANIFEST_SCHEMA.md)。

---

## 快速索引(自 MANIFEST 摘录)

- **版本**: v32
- **研究线**: distillation（审计代）
- **一句话**: 把损失里的 0.3 直接目标项移走后，学生**不再**超越教师（配对差 −0.0009 ± 0.0038，10 seeds，不显著）；表观收益全部来自目标泄漏。
- **谱系**: 见 MANIFEST.json 的 `generation` 字段
- **泄漏审计**: 本代即执行方，状态 `clean`（自身结论均在无泄漏条件下测得）

## 复核方式

```bash
python research/distillation/v32_leakage_audit/tests/test_package.py
python -c "from research.distillation.v32_leakage_audit import leakage_verdict; print(leakage_verdict())"
```
