# V74.0 — 审计脚本 /v62pkg 自包含

> 本文件是指针。权威规范在
> [`research/aligner/v74_audit_pkg/SPEC.md`](../research/aligner/v74_audit_pkg/SPEC.md)

**一句话**：让审计脚本支持直接从 /v62pkg 读取算子文件，notebook 不必再 git clone 整个仓库。

| 入口 | 位置 |
|---|---|
| 权威规范 | `research/aligner/v74_audit_pkg/SPEC.md` |
| 机器可读声明 | `research/aligner/v74_audit_pkg/MANIFEST.json` |
| 指标快照 | `research/aligner/v74_audit_pkg/results/metrics.json` |
| 权威测试 | `research/aligner/v74_audit_pkg/tests/test_package.py` |

## ⚠️ 本代无端到端数字

Kaggle 批量验证**尚未执行**。本代只验证机制语义，**不得**把「代码就位」说成「已验证有效」。
