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
| 端到端不塌缩 | held-out **0.9076**，专家均衡 **0.612**（v62.3；此前 0.9016/0.8774 是 v62.0 的，统计的是死代码 router） |

## 版本说明（端到端数字）

端到端数字随三次修复变化，**只有 v62.3 的有效**：

| 版本 | held-out | 专家均衡 | 说明 |
|---|---|---|---|
| v62.0 | 0.9016 | 0.8774 | **作废** —— 那组均衡统计的是从不参与计算的 router |
| v62.1 | 0.9141 | 0.5736 | router 接上了，但槽上做 top-k ⇒ 槽门控塌缩（熵 0.0209） |
| v62.2 | 0.9101 | 0.492 | 槽门控改全槽软门控 + 熵正则，但 **router 仍是随机的**（lazy bug） |
| **v62.3** | **0.9076** | **0.612** | router 改 eager 创建，真正被优化。当前有效 |

v62.0/v62.1/v62.2 的槽门控结论一律无效 —— 那个 router 从未被训练。

## 一处必须记住的偏离

v62 的融合层**解冻了专家**（v33 是 frozen + `x.detach()`，全流程只训 2.3K 参数）。
端到端结果在解冻前提下取得，**不能**与 v33 的 +26.7% 直接比较。
