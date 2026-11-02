# V58.0 — 膜 / 面积律算子

> 本文件是指针。权威规范在
> [`research/aligner/v58_entanglement_area/SPEC.md`](../research/aligner/v58_entanglement_area/SPEC.md)

v57 只有一个全局标量 r，无法表达「这里维度多、那里维度少」。本代把 r 拆成**逐切口的 χ 剖面**，用 Ryu–Takayanagi 面积律定规则：每个切口的维数由跨越它的净耦合通道数决定。

## 结果

合成侧算子 13/13 全过；**真实表征 B2/B3 未过 ⇒ 膜假说不成立**（`acceptance.outcome = fail`）。

| 入口 | 位置 |
|---|---|
| 权威规范 | `research/aligner/v58_entanglement_area/SPEC.md` |
| 机器可读声明 | `research/aligner/v58_entanglement_area/MANIFEST.json` |
| 指标快照 | `research/aligner/v58_entanglement_area/results/metrics.json` |
| 权威测试 | `research/aligner/v58_entanglement_area/tests/test_package.py` |
