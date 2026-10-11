<!-- 权威副本. 本文件是 research/routing_evolution/v43_central_audit/SPEC.md -->
<!-- 机器可读声明见同目录 MANIFEST.json; 指标快照见 results/metrics.json -->

# V43.0 — 中枢机制审计:容量 / 结构 / 泄漏

> **日期**: 2026-10-10
> **状态**: ✅ Verified (8/8 单元测试 PASS, 3 臂 × 5 seeds = 15 run)
> **复现命令**: `python3 examples/run_v43_central_audit.py` (CPU 约 15 分钟)
> **审计对象**: `v9_gate_central` —— `neurips_2025.tex` **整篇**建立在它上面
> **一句话**: **容量方向相反(排除),结构论证成立(保住),但 +44.5% 是 in-sample 数字,在 held-out 上塌到噪声内。**

---

## 1. 三条审计,两条不需要实验

### 1.1 容量混杂 —— 方向与担心的相反

| 中枢 | 参数构成 | 总数 |
|---|---|---:|
| v8 `BroadcastWorkspace` | `c[256] + U[3,256]` | **1,024** |
| v9 `GateStyleWorkspace` | `c[256] + W_t[256]` | **512** |

**gate 的中枢参数是 broadcast 的一半**,而且 gate 的参数量**与专家数无关**(标量温度),
broadcast 的 `U[M,d]` 随 M 线性增长。

> v9 相对 v8 的 +44.5% **不可能来自参数更多**。容量混杂在此处的方向与担心相反。

### 1.2 容量与机制**不可分离** —— 与 v41 对齐器不同

v41 之所以能做等容量对照,是因为可以把 `n_shared_experts` 调到任何值而机制不变。
中枢这里做不到:gate 的容量来自 `W_t`,要把它展宽到 broadcast 的 1,024 参数,
必须令 `W_t ∈ R^{M×d}`,即**每个专家一个温度**。

随机检验 (20,000 次, M=3):

| 形态 | 保序破坏率 |
|---|---:|
| 标量温度(gate 实际形态) | **0.000** |
| 逐专家温度(展宽到 broadcast 容量) | **0.049** |

> **买到容量就必须放弃保序。** 容量与机制在中枢这里**不可分离**——
> 这不是方法缺陷,是 gate 之所以有效的机制本身要求的。
> 与 v41"可以构造等容量对"的结论恰好相反,两条并读才是完整图景。

### 1.3 两条结构性论证:成立,且泄漏无法伪造

| 性质 | 检验 | 结果 |
|---|---|---|
| **完美退化** | 初始 `W_t=0, c=0` ⇒ `temperature=1` ⇒ 输出恒等于输入 | ✓ 最大偏差 **0.000e+00** |
| **保序** | 标量温度不改变 M 个 logit 的次序 | ✓ 20,000 次检验破坏率 **0.000** |

> 这两条是 v9 论文的**核心论证**,它们是构造性质而非测量结果。
> 本代在 held-out 上发现的样本外问题**不影响它们**。

## 2. 泄漏:v9 的 +44.5% 不成立于样本外

`examples/run_v9_gate_layer.py:121-124` 在**训练同源数据**上评估
(第 119 行训练用 `x_shared_list` + `targets`,第 121-124 行评估用同一批)。

配对比较 (同 seed 内与 v7-baseline 比):

| 臂 | in-sample 中位 | 胜出 | **held-out 中位** | 胜出 |
|---|---:|---:|---:|---:|
| v9-gate-c1 | +52.2% | 4/5 | **+4.0%** | **3/5** |
| v9-gate-full | +93.4% | 5/5 | **+23.3%** | 4/5 |

泛化落差(相对值):v7-baseline **+127%**,gate-c1 **+278%**,gate-full **+2179%**。

### 三点必须同时说明(不要读成"gate 无效")

1. 论文的 **+44.5% 对应 gate-c1 臂**,该臂样本外确实塌到噪声内(+4.0%,3/5)
2. **完整 v9(gate + C-2 + C-3)仍保留 +23.3%,4/5 seed**
3. 本 harness 的 held-out 本身很严苛——**连 v7 基线都从 1.096 劣化到 2.488(2.3×)**,
   所以绝对样本外数值对所有臂都差,**只有同 seed 配对比较是有效读数**

## 3. 本代自身的一次纪律考验

**1 seed 冒烟给出 gate-c1 held-out +42.3%(存活),5 seed 后变成 +4.0%(不存活)。**

| seed 数 | gate-c1 held-out 配对中位 | 结论 |
|---|---:|---|
| 1 | +42.3% | 存活 |
| 5 | **+4.0%** | **不存活** |

> 若按 v40 的教训只看冒烟,会得出**完全相反**的结论。这与 v40 的 `dispatch-oracle`
> 均值假象是同一族错误。**已记入 `results/metrics.json:verdict_headline.method_note`。**

## 4. 顺带修的两个既有缺陷

两个都让 **v9 的结果无法复现**:

1. `examples/run_v9_gate_layer.py` 把 `make_v9_layer` 从 `_primitives.central_mechanism` 导入,
   而该符号在 `research.routing_evolution.v9_gate_central` → **ImportError, runner 根本无法启动**
2. `research/routing_evolution/v9_gate_central/layer.py` 引用了不存在的
   `v9_gate_central.workspace` 子模块(该符号实际在 `_primitives`)

与 v33 runner 的 ImportError 同类,都是"文档说能跑、实际跑不起来"。

## 5. 局限(诚实边界)

- held-out 构造除换噪声外**还换了随机图像**(`numpy` RNG 前进),是比"只换噪声"更强的漂移。
  所有臂看到同一份,故配对比较有效,但**绝对值不可与 in-sample 比**。
- gate-c1 的 held-out 是 +4.0% / 3/5 —— 这是"**不被支持**",不是"**被证伪**"。
  5 个 seed 不足以把 +4% 和 0 区分开。
- 全部在 CPU 沙箱:`D_shared=256`,序列长 16,合成 class-direction 目标。
- **未审 v19/v20/v22/v25–v31 的中枢变体**(softplus / grad-scale / router-norm / adaptive-ema)
  仍是 pending。v29 曾报告"大语料下 gate 反而退步 −16.3%",该结论同样只有 in-sample 依据。

## 6. 未闭环(按优先级)

1. **`neurips_2025.tex` 的 +44.5% 需要改口径** —— 它是 in-sample 数字,且 gate-c1 臂样本外不存活
2. v30/v31 的 tanh 饱和修复、以及 v22 的四种中枢模式比较,都仍只有 in-sample 依据
3. 剩余 pending 且被论文引用的: v7 / v18 / v22 / v25 / v29(v13 已由 v41 结清、v21 已由 v42 推翻)

## 7. 工程产出

- `v43_central_audit/workspace_audit.py` — 容量解析式 / 完美退化 / 保序随机检验
- `v43_central_audit/tests/test_package.py` — 8 项单元测试(权威副本)
- `examples/run_v43_central_audit.py` — 3 臂 × 5 seed + held-out
- `results/metrics.json` / `results/raw.json`
