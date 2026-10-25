# V49.0 — 投影空间三重检验 (非线性 × 条件依赖 × 样本量)

> **薄指针层**: 权威正文在 `research/aligner/v49_projection_space/SPEC.md` (随仓库分发)。
>
> - 规范: [`research/aligner/v49_projection_space/SPEC.md`](../research/aligner/v49_projection_space/SPEC.md)
> - 机器可读声明: `research/aligner/v49_projection_space/MANIFEST.json`
> - 指标快照: `research/aligner/v49_projection_space/results/metrics.json`
> - 复现: `bash run.sh v49` (本地) / HF Space `redmingwei/hetero-moe` (远端, 每格一跑可续跑)
>
> **一句话**: v48 说"让投影可训练"得到的是**拟合能力**; v49 在**参数量对齐**、
> **算力对齐**下用 2×2 析因 + 样本量自变量证明: 非线性映射能拿到线性抓不到的那份
> 结构 (held-out 低 5-10 倍, 过拟合间隙小一半), 条件依赖确实在起作用 (打乱条件
> 退化 25%-47%) 但需要数据 —— 同时暴露出更尖锐的问题: **v33 架构的"融合"收益
> 并不来自融合** (可训练臂的"融合 vs 单专家"增益全部塌到 ~0)。
