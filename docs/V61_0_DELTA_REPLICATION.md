# V61.0 — Δ 跨模型复制（主结论被文献削弱）

> 本文件是指针。权威规范在
> [`research/aligner/v61_delta_replication/SPEC.md`](../research/aligner/v61_delta_replication/SPEC.md)

把 v60.1 的事后探索改为**预登记**检验：固定 Δ ≡ M̄(1) − mean(M̄(2..S−1))，在 3 模型 × 2 长度 × 8 层上重复，并加 R4 非交替判据。

## 结果

R1/R2 全过 48/48；**R4 按字面未通过**（违规全在 centered 臂）。⚠️ 主结论已被文献大幅削弱：Song & Zhong (2023) 已给出同样的分解，raw 臂的「相邻更相似」是重发现，且据 Zuo et al. (2025) 可能是因果注意力掩码的架构副产物。**先读 LITERATURE.md 与 FINDINGS.md 的警示**。

| 入口 | 位置 |
|---|---|
| 权威规范 | `research/aligner/v61_delta_replication/SPEC.md` |
| 机器可读声明 | `research/aligner/v61_delta_replication/MANIFEST.json` |
| 指标快照 | `research/aligner/v61_delta_replication/results/metrics.json` |
| 权威测试 | `research/aligner/v61_delta_replication/tests/test_package.py` |
