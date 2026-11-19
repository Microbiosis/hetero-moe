# V65.0 — SNLI-VE 真多模态

> 本文件是指针。权威规范在
> [`research/aligner/v65_snli_ve/SPEC.md`](../research/aligner/v65_snli_ve/SPEC.md)

**一句话**：把评测任务从 ag_news 4 分类换成 SNLI-VE 3 类蕴含判断——后者要求图文配对推理，把多模态从「输入拼接」变成「必须用视觉证据」。

| 入口 | 位置 |
|---|---|
| 权威规范 | `research/aligner/v65_snli_ve/SPEC.md` |
| 机器可读声明 | `research/aligner/v65_snli_ve/MANIFEST.json` |
| 指标快照 | `research/aligner/v65_snli_ve/results/metrics.json` |
| 权威测试 | `research/aligner/v65_snli_ve/tests/test_package.py` |

## ⚠️ 本代无端到端数字

Kaggle 批量验证**尚未执行**。本代只验证机制语义，**不得**把「代码就位」说成「已验证有效」。
