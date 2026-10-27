# V55.0 — 中枢带宽: 融合算子的秩判据 (扩展文档)

> **本文件是薄指针 (thin pointer), 非权威副本.**
> 权威正文: [`research/aligner/v55_hub_bandwidth/SPEC.md`](../research/aligner/v55_hub_bandwidth/SPEC.md)
> 机器可读声明: [`research/aligner/v55_hub_bandwidth/MANIFEST.json`](../research/aligner/v55_hub_bandwidth/MANIFEST.json)
> 实测快照: [`research/aligner/v55_hub_bandwidth/results/metrics.json`](../research/aligner/v55_hub_bandwidth/results/metrics.json)
> 复现: `python research/aligner/v55_hub_bandwidth/measure.py` (CPU ~40s, 不加载底座权重)

原因: 让每一代成为**自包含独立子包**——规范、指标快照、测试与代码同处一目录.
规范见 [`docs/GENERATION_MANIFEST_SCHEMA.md`](GENERATION_MANIFEST_SCHEMA.md).

---

**一句话**: v48–v53 六代审计反复观察到「margin 与源专家身份同生共死」, 但一直只有观察没有机制。
本代给出机制, 而且是可算的、结构性的:

> **v33 融合层的跨专家通道秩恒等于 `min(top_k, M-1)`** —— 对**任意**参数取值成立 (定理 3)。
> v33 默认配置 ($M=4$, `top_k`=2) 下是 **2 维**, 对应 $C(4,2)=6$ 个门状态 = **2.58 bit**。
> 而这条 2 维通道承担了融合层 **68.5%** 的雅可比能量。

与之并置的是 v49 的空间诊断: 数据并集在 8192-token 口径下 95% 方差只有 **4 维**, 99% 是 **19 维**,
而共享空间是 **256 维**。于是**两个方向的错配同时存在**:

| 通道 | 宽度 | 数据需要 | 诊断 |
|---|---|---|---|
| 跨专家门 $J_\alpha$ | 2 | ≥4–19 | **过窄** |
| 共享空间 $x$ | 256 | 4–19 | **过宽 (13–64×)** |

过窄的那条决定了「只能分辨是哪个底座」(2 bit 恰好够编一个源标签); 过宽的那条提供了「任意标签都能
拟合」的自由度 (所以 +20%~+70% 的 margin 是两个同样坏掉的模型之间的比较)。v50/v51/v52/v53 的
全部读数落在这两句话上。

**本文不主张**那篇 ETH 论文能给融合算子带来任何加速: 它算配分函数, 算子算点值, 遍历项
$(2D+1)^{2\tau n}$ 塌成 $O(1)$ 之后算法本体什么都不剩 (§1.1)。可搬的只有它把「精确可分解」
化归成秩条件的那一步 (§1.2 → 定理 1/2)。

**待验证** (§5, 预登记五条): 核心是 **H1** —— `top_k` 消融下 margin 应跟踪 `min(k, M-1)`,
即 $k=1,2,3,4$ 呈 **升 → 升 → 平**。一个「容量越多越好」的叙事预测单调升, 一个「过宽有害」的叙事
预测单调降, 两者都会在 $k{=}3\to4$ 被打脸。这是本判据唯一的强预测。
