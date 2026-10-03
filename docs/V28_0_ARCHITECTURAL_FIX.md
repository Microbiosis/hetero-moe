# V28.0 — 架构层面修正 router-norm / adaptive-ema

> **本文件是薄指针 (thin pointer), 非权威副本.**
> 权威规范: [`research/corpus_scale/v28_architectural_fix/SPEC.md`](../research/corpus_scale/v28_architectural_fix/SPEC.md)
> 机器可读声明: [`research/corpus_scale/v28_architectural_fix/MANIFEST.json`](../research/corpus_scale/v28_architectural_fix/MANIFEST.json)
> 指标快照: [`research/corpus_scale/v28_architectural_fix/results/metrics.json`](../research/corpus_scale/v28_architectural_fix/results/metrics.json)
> 权威测试: [`research/corpus_scale/v28_architectural_fix/tests/test_package.py`](../research/corpus_scale/v28_architectural_fix/tests/test_package.py)

原因: 让每一代成为**自包含独立子包**——规范、指标快照、测试与代码同处一目录.
规范见 [`docs/GENERATION_MANIFEST_SCHEMA.md`](GENERATION_MANIFEST_SCHEMA.md).
