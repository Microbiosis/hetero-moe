<!-- 权威副本. 本文件是 research/central_stability/v44_hub_modes_audit/SPEC.md -->
<!-- 机器可读声明见同目录 MANIFEST.json; 指标快照见 results/metrics.json -->

# V44.0 — 四种中枢模式的容量-泄漏审计

> **日期**: 2026-10-10
> **状态**: ✅ Verified (8/8 单元测试 PASS, 4 模式 × 5 seeds = 20 run)
> **复现命令**: `python3 examples/run_v44_hub_modes_audit.py` (CPU 约 8 分钟)
> **审计对象**: `v22_central_theory` 的四模式比较 —— **`paper.tex` 主表的依据**
> **一句话**: **主表结论不成立。** in-sample 最优是 router-norm 而非 adaptive-ema;
> adaptive-ema 的样本外优势塌到 +0.3%(3/5),而 router-norm 是唯一 5/5 全胜的模式。

---

## 0. 前置:主表此前根本无法复现

`examples/run_v22_theory_compare.py` 第 14 行是一个**多余的 `"""`**, 提前关闭了模块 docstring,
使整文件 SyntaxError —— **v22 的四模式比较从未被成功运行过**, `paper.tex` 主表不可复现。

修复: 删除该行。此后本代才第一次真正跑出 v22 的四模式数字。

## 1. 容量:gate 最省,其余三种都是它的两倍

| 模式 | 中枢参数 | 相对 gate | 构成 |
|---|---:|---:|---|
| **gate** | **512** | 1.00× | `c[256] + W_t[256]` |
| broadcast | 1,024 | 2.00× | `c[256] + U[3,256]` |
| router-norm | 1,024 | 2.00× | `c[256] + U[3,256]` |
| adaptive-ema | 1,025 | 2.00× | `c[256] + U[3,256] + decay[1]` |

## 2. 结果(5 seeds,同 seed 配对中位数;基线 = broadcast)

| 模式 | in-sample 中位 | 泛化落差 | **held-out 中位** | vs broadcast in | **vs broadcast out** | 胜出 |
|---|---:|---:|---:|---:|---:|---:|
| broadcast | 0.3236 | 9.3% | 0.3538 | +0.0% | +0.0% | — |
| gate | 0.2983 | 16.1% | 0.3462 | +7.9% | **+5.8%** | 3/5 |
| **router-norm** | **0.2835** | 17.9% | **0.3342** | **+9.6%** | +4.3% | **5/5** |
| adaptive-ema | 0.2859 | **25.0%** | 0.3573 | +4.0% | **+0.3%** | 3/5 |

逐 seed 的 held-out 配对增益:

```
gate          +6  −7  +20  +7  −6      → 3/5
router-norm   +4  +0  +11  +10  +2      → 5/5
adaptive-ema  −6  +0  −38  +4  +2      → 3/5
```

## 3. 裁决

`paper.tex` 主表写: *"Adaptive-EMA achieves the lowest MSE on v22 (0.2972)"*。

- ❌ **in-sample 最优就不是 adaptive-ema** —— router-norm 0.2835 < adaptive-ema 0.2859
- ❌ **held-out 更不成立** —— router-norm 0.3342 < adaptive-ema 0.3573
- ❌ **adaptive-ema 的优势不稳定** —— held-out 仅 +0.3%,3/5 胜出,含单种子 **−38%**
- ❌ **它是泛化最差的** —— 落差 25.0%,四模式最高:in-sample 领先、样本外垫底

> **router-norm 是唯一在 held-out 上 5/5 全胜 broadcast 的模式。**

## 4. 容量交互:在中枢这条线上容量同样不是主导

gate 只有 512 参数(其余的两倍),样本外拿到 **+5.8%**;router-norm 1,024 参数拿到 **+4.3%**;
adaptive-ema 1,025 参数拿到 **+0.3%**。

> **更多参数没有换来更好的样本外表现。** 这与 v43 在 gate vs broadcast 上的结论同向:
> 中枢这条线上,结构性质(保序、完美退化)比参数量更决定成败。

## 5. 局限

- held-out 除换噪声外**还换了随机图像**,漂移强于"只换噪声";所有臂看到同一份,
  故配对比较有效,**绝对值不可与 in-sample 比**。
- 5 seeds 分辨不出 +0.3% 这种量级 —— adaptive-ema 的塌陷是"未被支持"而非"被证伪";
  但**它不是最优**这一点在两种口径下都成立。
- 全部在 CPU 沙箱:`D_shared=256`,序列长 16,合成 class-direction 目标。

## 6. 未闭环(优先级最高)

- **`v29` 的语料规模边界仍 pending** —— 它才是 `paper.tex` 的**理论主张**
  ("optimal hub mode switches from gate-style to adaptive-EMA as data scale grows"),
  且 v29 报"大语料下 gate 退步 −16.3%",那个数字同样只有 in-sample 依据。
- `v30` / `v31` 的 tanh 饱和修复未审。

## 7. 工程产出

- `v44_hub_modes_audit/audit.py` — 容量表 / 配对增益 / 泛化落差
- `v44_hub_modes_audit/tests/test_package.py` — 8 项单元测试(权威副本)
- `examples/run_v44_hub_modes_audit.py` — 4 模式 × 5 seed + held-out
- `results/metrics.json` / `results/raw.json`
