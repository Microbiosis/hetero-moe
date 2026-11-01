# V59.0 — 面积律（token 轴）：负结果

> 本文件是指针。权威规范在
> [`research/aligner/v59_token_area/SPEC.md`](../research/aligner/v59_token_area/SPEC.md)

把 v58 的面积律算子搬到 token 轴，并加三条对照（N4 纯噪声正控 / N0 样本打乱 / N1 切口敏感）。

## 结果

**χ 与切口无关 ⇒ 面积律在 token 轴上平凡成立、不携带信息**（N1 0/8 全败）。同时证伪 v58 的阈值方法：iid 零分布系统性低估谱尾部（最高 5.7×）。

| 入口 | 位置 |
|---|---|
| 权威规范 | `research/aligner/v59_token_area/SPEC.md` |
| 机器可读声明 | `research/aligner/v59_token_area/MANIFEST.json` |
| 指标快照 | `research/aligner/v59_token_area/results/metrics.json` |
| 权威测试 | `research/aligner/v59_token_area/tests/test_package.py` |
