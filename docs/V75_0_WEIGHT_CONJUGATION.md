# V75.0 — 槽级共轭的权重级有效性检验

> 本文件是指针。权威规范在
> [`research/weight_fusion/v75_weight_conjugation/SPEC.md`](../research/weight_fusion/v75_weight_conjugation/SPEC.md)

**一句话**：把主线生死线 G2 跑成可证伪实验 —— 槽级搬运后，同槽跨架构的权重并没有比异槽更相似。

| 入口 | 位置 |
|---|---|
| 权威规范 | `research/weight_fusion/v75_weight_conjugation/SPEC.md` |
| 机器可读声明 | `research/weight_fusion/v75_weight_conjugation/MANIFEST.json` |
| 指标快照 | `research/weight_fusion/v75_weight_conjugation/results/metrics.json` |
| 权威测试 | `research/weight_fusion/v75_weight_conjugation/tests/test_package.py` |

## ⚠️ 本代是负结果

- G1 口径回归 PASS（原生六槽逐位复现 v62 §2）。
- **G2-a 保留检验 INCONCLUSIVE**：原生基线在 JS 与 CKA 两种口径下都无功效 ⇒ 不是否证，是不可判定。
- **G2-b 生成检验 FAIL**：按槽搬运未在预登记 α=0.01 上击败同谱随机搬运与槽无关搬运。

引用本代必须带上「负结果」与「α=0.01 是跑前写死的」这两条，不得按 0.05 事后放行。

复现需要权重切片缓存，步骤见 SPEC.md §8。
