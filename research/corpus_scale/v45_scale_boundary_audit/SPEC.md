<!-- 权威副本. 本文件是 research/corpus_scale/v45_scale_boundary_audit/SPEC.md -->
<!-- 机器可读声明见同目录 MANIFEST.json; 指标快照见 results/metrics.json -->

# V45.0 — 语料规模边界:方向判定与样本外存活性

> **日期**: 2026-10-10
> **状态**: ✅ Verified (7/7 单元测试 PASS, 2 规模 × 2 模式 × 5 seeds = 20 run)
> **复现命令**: `python3 examples/run_v45_scale_boundary_audit.py` (CPU 约 15 分钟)
> **审计对象**: `paper.tex` 的**理论主张**
> **一句话**: **摘要与论文自己的主表方向相反;而且边界在样本外读数下不稳健。**

---

## 1. 论文内部先自相矛盾

`paper.tex` 摘要(第 74 行):

> *"establish a corpus-scale boundary showing that the optimal hub mode switches
> **from gate-style to adaptive-EMA** as data scale grows"*

论文主表(同一篇,第 200–207 行):

| 规模 | gate | adaptive-ema | 最优 |
|---|---:|---:|---|
| tiny (8/4/4) | 0.7097 | **0.7035** | adaptive-ema |
| small (18/6/6) | 0.7075 | **0.6984** | adaptive-ema |
| medium (32/12/12) | **0.6971** | 0.7136 | gate |
| large (64/24/24) | **0.7029** | 0.7136 | gate |

**方向是 `adaptive-ema → gate`,与摘要的 `gate → adaptive-ema` 相反。**

论文的贡献点(第 93 行)、讨论(225)、结论(233)**三处都与主表一致**,
只有摘要写反。已用单测
`test_paper_table_direction_is_ada_to_gate` 与
`test_abstract_claim_is_contradicted_by_its_own_table` 把这个矛盾锁死。

## 2. 关键前提:held-out 数字一直存在,只是没人汇总

`examples/run_v26_real_corpus.py` 与 `examples/run_v29_large_corpus_v26.py`
**本来就在同一个 `run_seed` 里同时算 `train_fuse` 和 `ho_fuse`**,并都返回了。
只是这些 held-out 数字从未被汇总进任何结论。本代做的第一件事就是把那份
**已经算好却没人看的数字**取出来用。

## 3. 两点检验(5 seeds,中位数)

| 规模 | 模式 | in-sample | held-out | 落差 |
|---|---|---:|---:|---:|
| small | gate | 0.3183 | 0.4572 | +43.7% |
| small | **adaptive-ema** | **0.3017** | **0.4472** | +48.2% |
| large | **gate** | **0.2729** | 0.4618 | +69.2% |
| large | adaptive-ema | 0.2985 | **0.4184** | +40.2% |

- **in-sample 口径**:adaptive-ema → gate,**边界存在**(与主表一致)
- **held-out 口径**:adaptive-ema → adaptive-ema,**边界消失**

## 4. ⚠️ 但大规模档的 held-out 读数自相矛盾

| 口径 | large 档 held-out 结论 | 依据 |
|---|---|---|
| **中位数**(不配对) | adaptive-ema 优 → **无边界** | 0.4184 < 0.4618 |
| **配对**(逐 seed) | gate 优 → **有边界** | adaptive-ema 仅 1/5 胜,配对中位 −3.6% |

**成因**:adaptive-ema 在 large 档有一个重灾种子 ——
**seed 1 的 held-out = 0.767**,而同 seed 的 gate = 0.543。
这一个异常值把 adaptive-ema 的**中位数**拉低,于是两种口径给出相反的胜者。

> **按本项目自 v40 起的纪律,两种口径必须同时报告。**
> 本代不宣称边界存在或不存在,只宣称:**该边界在这套读数下不稳健。**

## 5. 裁决

| 主张 | 裁决 |
|---|---|
| 摘要的 `gate → adaptive-ema` | **不成立** —— 与自己的主表相反;held-out 两种口径下也不支持 |
| 主表的 `adaptive-ema → gate` | **in-sample 下成立,held-out 中位数口径下不成立** |

## 6. 局限(必须一起读)

- **只测了 4 档里的 2 档**(small / large),**medium 未测** —— 而翻转点恰恰在 medium。
- **小语料档 held-out 的优势仅 +0.5%(3/5)**,本就在噪声内。
- 四个臂的 held-out 落差都在 **+40% ~ +69%**,说明本 harness 在真实语料上泛化本就困难,
  **绝对值不可跨规模比较**,只有同 seed 配对可用。
- CPU 沙箱,5 seeds。

**因此本代不足以推翻整个边界理论**,只能说明:
**该理论目前缺少稳健的样本外支撑,且论文摘要的陈述与自己的数据矛盾,必须先更正。**

## 7. 又一次单种子陷阱

**1 seed 冒烟给出「in-sample 无边界 / held-out 有边界」,5 seed 后反转为
「in-sample 有边界 / held-out 无边界」——两个口径的结论都翻转了。**

若按冒烟汇报会得出完全相反的判断。这与 v40 的 `dispatch-oracle`、
v43 的 `gate-c1` 同族,是本项目第三次踩到。

## 8. 工程产出

- `v45_scale_boundary_audit/boundary.py` — 边界方向判定 / 方向一致性检验
- `v45_scale_boundary_audit/tests/test_package.py` — 7 项单元测试(权威副本)
- `examples/run_v45_scale_boundary_audit.py`
- `results/metrics.json` / `results/raw.json`
