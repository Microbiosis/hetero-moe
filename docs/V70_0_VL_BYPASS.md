# V70.0 — VL 跨模态融合旁路

> 本文件是指针。权威规范在
> [`research/aligner/v70_vl_bypass/SPEC.md`](../research/aligner/v70_vl_bypass/SPEC.md)

**一句话**：接入跨模态 VL 融合旁路（锚点共注意力 + Hadamard-Bilinear 门控），并修掉视觉 patch 数硬编码与 ViT hidden 全零两个真实 bug。

| 入口 | 位置 |
|---|---|
| 权威规范 | `research/aligner/v70_vl_bypass/SPEC.md` |
| 机器可读声明 | `research/aligner/v70_vl_bypass/MANIFEST.json` |
| 指标快照 | `research/aligner/v70_vl_bypass/results/metrics.json` |
| 权威测试 | `research/aligner/v70_vl_bypass/tests/test_package.py` |

## ⚠️ 本代无端到端数字

Kaggle 批量验证**尚未执行**。本代只验证机制语义，**不得**把「代码就位」说成「已验证有效」。
