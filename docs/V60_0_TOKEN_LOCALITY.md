# V60.0 — token 局部性：预登记判据全败

> 本文件是指针。权威规范在
> [`research/aligner/v60_token_locality/SPEC.md`](../research/aligner/v60_token_locality/SPEC.md)

把观测量换成**位置对余弦相似度** corr(d, M̄(d))，并加 P2 置换可分判据挡掉 M 矩阵自身的量级结构。

## 结果

**P1/P2 全败 ⇒ 线性相关意义下无局部性**，且方向相反（远端更相似）。P2 抓到 P1 的三处假阳性。补充分析（v60.1，事后）发现 Δ 有稳定效应但三臂方向分裂。

| 入口 | 位置 |
|---|---|
| 权威规范 | `research/aligner/v60_token_locality/SPEC.md` |
| 机器可读声明 | `research/aligner/v60_token_locality/MANIFEST.json` |
| 指标快照 | `research/aligner/v60_token_locality/results/metrics.json` |
| 权威测试 | `research/aligner/v60_token_locality/tests/test_package.py` |
