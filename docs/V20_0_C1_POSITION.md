# V20.0 — H5 假设: 中枢 broadcast 位置 (attn vs ffn)

> **本文件是薄指针 (thin pointer), 非权威副本.**
> 权威规范: [`research/central_diagnostics/v20_c1_position/SPEC.md`](../research/central_diagnostics/v20_c1_position/SPEC.md)
> 机器可读声明: [`research/central_diagnostics/v20_c1_position/MANIFEST.json`](../research/central_diagnostics/v20_c1_position/MANIFEST.json)
> 指标快照: [`research/central_diagnostics/v20_c1_position/results/metrics.json`](../research/central_diagnostics/v20_c1_position/results/metrics.json)
> 权威测试: [`research/central_diagnostics/v20_c1_position/tests/test_package.py`](../research/central_diagnostics/v20_c1_position/tests/test_package.py)

原因: 让每一代成为**自包含独立子包**——规范、指标快照、测试与代码同处一目录.
规范见 [`docs/GENERATION_MANIFEST_SCHEMA.md`](GENERATION_MANIFEST_SCHEMA.md).
