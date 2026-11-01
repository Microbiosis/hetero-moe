# V62.0 — 维度投影空间

> 本文件是指针。权威规范在
> [`research/aligner/v62_dimension_projection_space/SPEC.md`](../research/aligner/v62_dimension_projection_space/SPEC.md)

**一句话**：v48–v57 六代都在问「D_shared 该多大」，答案一直是「不是它」。
本代换问法：**空间 = Σ_s S_s**，每个功能槽一个子空间，三条正交的轴（槽 / 模型 / 对齐）。

| 入口 | 位置 |
|---|---|
| 权威规范 | `research/aligner/v62_dimension_projection_space/SPEC.md` |
| 机器可读声明 | `research/aligner/v62_dimension_projection_space/MANIFEST.json` |
| 指标快照 | `research/aligner/v62_dimension_projection_space/results/metrics.json` |
| 权威测试 | `research/aligner/v62_dimension_projection_space/tests/test_package.py` |
| 复算脚本 | `research/aligner/v62_dimension_projection_space/measure.py` |
| 打包模型 | https://huggingface.co/redmingwei/HeteroMoEProjSpace-v2 |

## 核心结论

| 结论 | 实测 |
|---|---|
| 同功能跨架构权重谱更相似 | JS 0.0302 vs 0.0401，置换 p=0.0002，d=0.25 |
| 两两对齐都到 0.987+ | Q/O 与 K/V 无差别 |
| 三模型共用一个变换失败 | Q 0.6767 / O 0.7168 |
| 失败变量是**模型数**，不是功能类型 | 去掉 Qwen → 0.988 |
| 环路一致性可修复 | 0.677 → 0.9925，路径差 0.0038 |
| 等维对齐无损，压维丢信息 | 24 维保真 0.165；256 维 0.9998 |
| 端到端不塌缩 | held-out 0.9016，均衡 0.8774 |

## 一处必须记住的偏离

v62 的融合层**解冻了专家**（v33 是 frozen + `x.detach()`，全流程只训 2.3K 参数）。
端到端结果在解冻前提下取得，**不能**与 v33 的 +26.7% 直接比较。
