<!-- 权威副本. 本文件是 research/distillation/v40_dispatch/SPEC.md -->
<!-- 机器可读声明见同目录 MANIFEST.json; 指标快照见 results/metrics.json -->

# V40.0 — 分派式多教师

> **日期**: 2026-10-10
> **版本定位**: 独立顶层版本包 (原为 v17 子版本 v17b/17.1, 当日提升为 v40/40.0)
> **状态**: ✅ Verified (12/12 单元测试 PASS, 8 arm × 5 seed = 40 run)
> **复现命令**: `python3 examples/run_v40_dispatch.py`
> **一句话结论**: **v17 的归因只对了伤害、没说出增益** ——
> 无泄漏分派消除了加权和的损害(中位数 0.3344 → 0.3195), 但**没有超过单教师**;
> 而教师侧明明存在 13% 的互补 headroom, **没有任何形式能把它传给学生**。

---

## 1. 动机:一个从未被检验的归因

v17 (`research/distillation/v17_multi_teacher/`) 得出负结果: 多教师蒸馏
三个变体全部退步 (multi-2 −16.7% / multi-3 −6.7% / weighted −3.4%)。

后来该包的 `MANIFEST.json` 加了一条 `root_cause_identified_later`, 把失败归因于
**实现形式**而非多教师本身:

```python
# research/distillation/v17_multi_teacher/distill.py:51
total = total + self.weights[name] * l          # 加权和

# examples/run_v17_multi_teacher.py:159
student_outs = {kind: y for kind in teacher_kinds}  # 同一张量 y 别名进全部教师槽位
```

> 「代码中不存在任何按样本的分派路径; 对照 MOPD (arXiv:2606.30406) 的
> per-prompt 单教师分派, 本代从未测过分派式多教师。」

**这条归因从未被测试。** 本代就是那个测试。

外部对照 MOPD 用 per-prompt 分派取代加权求和, 在 Qwen3-30B-A3B 上做到归一化
0.9373 (Mix-RL 0.8818 / Cascade RL 0.7752)。但**前置条件不同**, 见 §2。

## 2. 为什么分派在 v17 上先天受限(读代码得到的结构性事实)

v17 的 `teacher_kinds = ["attn", "mixture", "mom"]` 走**同一个 `make_teacher`
+ 同一份 `train_teacher` 数据与目标** (`examples/run_v17_multi_teacher.py:107-118`),
只是架构不同:

| | MOPD 的教师 | v17 的教师 |
|---|---|---|
| 是什么 | **领域专家** (math / IF / SWE) | **对齐器阶梯三档** |
| 输入域 | **有划分**, 域标签唯一确定分派 | **完全重叠**, 人人都有意见 |
| 分派是否有唯一解 | 有 | **无** |

更关键: `prepare_data` 的 `for _ in range(6): targets.append(t)` 追加的是
**同一张量对象**, 所以 6 个 BERT 模态样本**共享同一个目标**
(本代实测 `targets_all_identical: true`)。

> **结论: v17 的 harness 里不存在 MOPD 那种"分派"的语义。**
> 本代因此不假装能复现 MOPD, 而是测量"分派作为操作"在本设置下的实际表达力。

## 3. 设计:8 个 arm

| arm | 分派信号 | 泄漏 | 角色 |
|---|---|---|---|
| `weighted-3-v17style` | — | 否 | **复现锚点**(v17 逐样本循环原样) |
| `weighted-3` | — | 否 | 加权和(批量版) |
| `consensus-target` | 朝**教师均值**蒸馏 | 否 | 不选谁, 直接朝融合目标 |
| `single-mixture` | — | 否 | 基线(= v17 teacher-single) |
| `single-mom` | — | 否 | v17 teacher-best-single |
| `dispatch-cyclic` | 样本序号轮转 | 否 | **零信息**分派基线 |
| `dispatch-consensus` | 留一共识最近者 | 否 | **无泄漏分派** |
| `dispatch-oracle` | 目标 MSE 最小者 | **⚠ 是** | 仅作上界 |

### 协议复用与两处刻意偏离

教师构造 / `prepare_data` / `train_teacher` / 评估口径**全部从 v17 直接 import**,
保证 arm 之间以及与 v17 之间可比。两处偏离:

1. **评估输入固定**: v17 的评估 `torch.randint` 消耗全局 RNG, 不同 arm 抽到
   不同评估样本。本代用独立固定种子的 generator, 所有 arm 面对**同一批**样本
   (配对比较)。
2. **教师按 seed 训一次并在 8 个 arm 间共享** (v17 每个 mode 重训一遍)。
   锚点臂 `weighted-3-v17style` 保留 v17 的逐样本循环, 用来验证批量化不是混杂
   —— 实测两者中位数相差 <0.2%。

## 4. 结果(5 seeds, **中位数**口径)

基线中位数 0.3194 (single-mixture)。

| arm | 中位数 | sd | vs 基线 | 去最好种子 | 泄漏 |
|---|---:|---:|---:|---:|---|
| **dispatch-oracle** | **0.3174** | 0.0358 | **+0.6%** | +1.1% | ⚠ 是 |
| dispatch-consensus | 0.3195 | 0.0031 | −0.1% | −0.4% | 否 |
| **single-mixture** | **0.3194** | 0.0072 | +0.0% | −0.8% | 否 |
| single-mom | 0.3201 | 0.0032 | −0.2% | −0.7% | 否 |
| consensus-target | 0.3335 | 0.0370 | −4.4% | −7.4% | 否 |
| weighted-3-v17style | 0.3338 | 0.0367 | −4.5% | −7.4% | 否 |
| weighted-3 | 0.3344 | 0.0379 | −4.7% | −7.6% | 否 |
| dispatch-cyclic | 0.3398 | 0.0372 | −6.4% | −8.2% | 否 |

### 为什么用中位数而不是均值

多教师臂的种子间 sd 是单教师的 **5–12 倍**(0.036 vs 0.003–0.007)。
均值口径被单个好种子主导:

- `dispatch-oracle` 均值 **+6.0%**, 但中位数 **+0.6%**、去掉最好种子后 **+1.1%**
- 驱动它的 seed 0 单跑出 0.2369, 其余 4 个 seed 是 0.3174 / 0.3100 / 0.3204 / 0.3180

> **+6.0% 是单种子假象。** 这正是 v32 建立的纪律在本代的应用:
> 任何增益都要过稳健性检验才能进结论。

### 教师侧诊断(无需训练, seed 0)

| 量 | 值 |
|---|---:|
| 教师两两平均平方距离 | 0.1360 |
| attn 对目标 MSE | 0.3180 |
| mixture 对目标 MSE | 0.3212 |
| mom 对目标 MSE | 0.3262 |
| **共识均值对目标 MSE** | **0.2764** |
| 6 个样本目标是否全同 | **True** |

**共识均值比最优单教师低 13.0%。多教师携带的互补信息在教师侧确实存在。**

### 分派退化

| arm | 实际用到的教师数 |
|---|---|
| dispatch-consensus | 1–2(5 个 seed 中大多只选中 1 个) |
| dispatch-oracle | 2–3 |
| dispatch-cyclic | 3 |

## 5. 结论

### 5.1 v17 的负结果复现成立(版本定位: v40 是独立顶层版本, 不是 v17 的变体)

| | v17 报告 | v40 复现 |
|---|---:|---:|
| 加权三教师 vs 单教师 | −6.7% | **−4.7%**(中位数) |

同符号同量级。锚点臂 `weighted-3-v17style`(0.3338)与批量版 `weighted-3`(0.3344)
相差 0.2%, 说明批量化不是混杂。

### 5.2 v17 的归因:部分成立但不足

`MANIFEST.json` 说失败源于「同时多目标回归的实现形式」。

- ✅ **成立的部分**: 实现形式确实是**伤害来源**。加权中位数 0.3344 → 无泄漏分派
  0.3195, 损害消失, 方差也从 0.037 降到 0.003。
- ❌ **不足的部分**: 修复后只**回到单教师水平, 没有超过**。

所以准确表述是: **"实现形式"解释了 v17 为什么会受伤, 但没有解释多教师本该有的增益
为什么不存在。**

### 5.3 v17 正文的结论未被推翻

v17 正文写「教师的质量远比数量重要, 一个选得好的教师胜过教师集成」。

**六个无泄漏 arm 没有一个在稳健口径上超过 single-mixture。** 该结论在
无泄漏条件下成立。

### 5.4 一条新负结果:形式不重要

加权 / 均值 / 轮转分派三种多教师形式的**中位数是 0.3335–0.3398, 彼此无差别**,
且都劣于单教师。

> **形式不重要, 单教师 vs 多教师才重要。** 这与 v12 / v17 / v27 / v28 是同一条
> 规律的第五次出现, 但这次把它收得更紧: 连"换个更好的形式"这条退路也堵上了。

### 5.5 第三条 capacity ceiling

教师侧有 13% 的互补 headroom, 但 `consensus-target`(朝均值蒸馏)的中位数是
0.3335, **反而劣于单教师**。

> 瓶颈不在"选哪个教师", 也不在"用什么形式融合", 而在
> **学生无法吸收融合后的目标**。

这是同族证据的第三条: v15 对齐器天花板 → v39 加性融合头天花板 →
v23b toy 数据天花板 → **v40 学生容量天花板**。

### 5.6 对 MOPD 类比的定量裁决

MOPD 的增益来自 per-prompt 分派到**域专家**; v17 教师无域划分 → consensus 分派
退化为 1–2 个教师。

> **分派的前提条件是硬的, 不是补丁。** MOPD 的手段在 v17 上不可移植,
> 这不是 v17 实现得不好, 是问题结构不同。

## 6. 局限(诚实边界)

- 全部在 CPU 沙箱: `D_SHARED=256`, 序列长 16, 合成 class-direction 目标。
- 教师是**对齐器变体**而非领域专家, 因此本代**不能**外推到 MOPD 的能力整合场景。
  只能说"分派在无域划分时不可移植"。
- 5 seeds 对 sd≈0.036 的臂偏少; 结论建立在**中位数 + 去最好种子**两个稳健口径上,
  而非 p 值。
- 教师按 seed 共享, 因此 arm 之间是**配对**的; 跨 seed 的绝对值不可比。
- 学生是 TinyBERT + LoRA(0.6% 参数)。§5.5 的 capacity ceiling 解释**未做容量
  扫描验证**, 是基于"教师侧 headroom 存在而学生吃不到"的推断。

## 7. 下一步

`results/metrics.json:next_experiment`: 若 capacity ceiling 成立, **放宽学生容量**
应让 `consensus-target` 臂吃到那 13% 的教师侧 headroom; 若仍吃不到, 则说明瓶颈
在**目标空间的对齐**而非容量 —— 那将指向 v39 式的融合头表达力问题, 而不是容量问题。

## 8. 工程产出

- `v40_dispatch/dispatch.py` — 分派族算子(纯 torch, 无跨线 import)
- `v40_dispatch/tests/test_package.py` — 12 项单元测试(权威副本)
- `examples/run_v40_dispatch.py` — 8 arm × 5 seed runner
- `results/metrics.json` / `results/raw.json`
