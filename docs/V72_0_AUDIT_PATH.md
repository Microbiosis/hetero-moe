# V72.0 — 审计脚本 projspace 定位修复

> 本文件是指针。权威规范在
> [`research/aligner/v72_audit_path/SPEC.md`](../research/aligner/v72_audit_path/SPEC.md)

**一句话**：修审计脚本的 projspace 定位——用绝对路径 + 包目录 fallback，不再依赖 Kaggle 的工作目录。

| 入口 | 位置 |
|---|---|
| 权威规范 | `research/aligner/v72_audit_path/SPEC.md` |
| 机器可读声明 | `research/aligner/v72_audit_path/MANIFEST.json` |
| 指标快照 | `research/aligner/v72_audit_path/results/metrics.json` |
| 权威测试 | `research/aligner/v72_audit_path/tests/test_package.py` |

## ⚠️ 本代无端到端数字

Kaggle 批量验证**尚未执行**。本代只验证机制语义，**不得**把「代码就位」说成「已验证有效」。
