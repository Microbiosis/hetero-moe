# V40.0 — 分派式多教师

> **本文件是薄指针 (thin pointer), 非权威副本.**
> 权威规范: [`research/distillation/v40_dispatch/SPEC.md`](../research/distillation/v40_dispatch/SPEC.md)
> 机器可读声明: [`research/distillation/v40_dispatch/MANIFEST.json`](../research/distillation/v40_dispatch/MANIFEST.json)
> 指标快照: [`research/distillation/v40_dispatch/results/metrics.json`](../research/distillation/v40_dispatch/results/metrics.json)

## 一句话

> v17 的归因只对了伤害、没说出增益:无泄漏分派消除了加权和的损害
> (中位数 0.3344 → 0.3195), 但没有超过单教师;而教师侧明明存在 13% 的互补
> headroom,**没有任何形式能把它传给学生**。

## 为什么有 v40 这个版本

本代检验 `v17_multi_teacher/MANIFEST.json` 的 `root_cause_identified_later`
(失败源于「同时多目标回归」的实现形式),该归因**从未被测试**。

它原本被实现为 v17 的子版本(v17b / version 17.1),但内容上是引入新机制
(per-sample 分派 + 教师侧诊断)并产出新负结果,不是 v17 的一个变体,
因此提升为独立顶层版本包。

## 四条结论

| # | 结论 | 证据 |
|---|---|---|
| 1 | **v17 负结果复现**:加权三教师 −6.7%(v17)→ −4.7%(v40 中位数) | `metrics.json:verdict.v17_negative_reproduced` |
| 2 | **v17 归因部分成立但不足**:实现形式是**伤害来源**,不是增益来源 | `metrics.json:claim` |
| 3 | **形式不重要**:加权 / 均值 / 轮转分派中位数 0.3335–0.3398 彼此无差别 | `metrics.json:verdict.form_invariance` |
| 4 | **第三条 capacity ceiling**:教师侧 13% headroom 传不到学生 | `metrics.json:verdict.capacity_ceiling` |

## 稳健性:为什么用中位数

多教师臂的种子间 sd 是单教师的 5–12 倍。`dispatch-oracle` 均值 +6.0%,
中位数仅 +0.6%、去掉最好种子 +1.1% —— **是单种子假象**,不得引用为结论。

## 与外部工作的关系

MOPD(arXiv:2606.30406)用 per-prompt 分派取代加权求和,但其教师是**领域专家**
(math / IF / SWE, 输入域有划分);v17 的教师是**对齐器阶梯三档**(同数据同目标,
能力完全重叠、无域划分),且 6 个样本**共享同一目标**。

> **分派的前提条件是硬的, 不是补丁。** 本代对此给出定量否证。
> 详见 [[Obsidian/论文笔记/MOPD多教师同策略蒸馏笔记]]。

## 复现

```bash
python3 examples/run_v40_dispatch.py          # 8 arm × 5 seed, CPU 约 9 分钟
V40_SEEDS=1 python3 examples/run_v40_dispatch.py   # 冒烟
```
