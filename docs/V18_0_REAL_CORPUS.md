# V18.0 — 真实 Wikipedia 语料蒸馏

> **本文件是薄指针 (thin pointer), 非权威副本.**
> 权威规范: [`research/distillation/v18_real_corpus/SPEC.md`](../research/distillation/v18_real_corpus/SPEC.md)
> 机器可读声明: [`research/distillation/v18_real_corpus/MANIFEST.json`](../research/distillation/v18_real_corpus/MANIFEST.json)
> 指标快照: [`research/distillation/v18_real_corpus/results/metrics.json`](../research/distillation/v18_real_corpus/results/metrics.json)
> 权威测试: [`research/distillation/v18_real_corpus/tests/test_package.py`](../research/distillation/v18_real_corpus/tests/test_package.py)

原因: 让每一代成为**自包含独立子包**——规范、指标快照、测试与代码同处一目录,
整目录拷走即可独立阅读与复核. 规范见
[`docs/GENERATION_MANIFEST_SCHEMA.md`](GENERATION_MANIFEST_SCHEMA.md).

---

## 快速索引(自 MANIFEST 摘录)

- **版本**: v18
- **研究线**: distillation
- **谱系**: 见 MANIFEST.json 的 `generation` 字段
- **泄漏审计**: 见 MANIFEST.json 的 `leakage_audit` 字段
- **复现**: `python3 tests/test_v18_*.py` (转发层) 或直接跑包内 `tests/test_package.py`
