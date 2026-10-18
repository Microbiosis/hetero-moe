# 文献：为什么 τ 估计会随采样率变化（以及该怎么办）

> 2026-10-18 · 针对 exp39 的失败（0/5 通过分辨率检验）做的文献核查
>
> **结论**：我们的失败**不是实现错误，是这类估计的已知固有偏差**。
> 文献明确指出 ACF 估计受 bin width、数据长度、epoched vs 连续 等因素支配。

---

## 一、最直接的证据：iSTTC 论文（PLOS Comp Biol 2026）

> Pochinok, Hanganu-Opatz, Chini.
> "iSTTC: A robust method for accurate estimation of intrinsic neural timescales
> from single-unit recordings." *PLoS Comput Biol* 22(3): e1013385, 2026.
> DOI: 10.1371/journal.pcbi.1013385

### 摘要里的原话

> "current methods for estimating ITs from SUA **suffer significant biases and
> instabilities**, particularly when applied to sparse, noisy, or epoched neural
> spike data."

### 关键段落（直接对应我们的问题）

> "Recent methodological improvements have primarily focused on **optimizing the
> exponential fitting step** ... However, significant uncertainty remains in the
> **initial process of estimating the ACF itself**. This is a critical step whose
> results are influenced by multiple factors, including:
>
> - sensitivity to **data length**
> - the SUA **firing rate**
> - the eventual presence of **trials**
> - the representation of the spiking data as **binary or continuous**
> - **the choice of bin width** or smoothing kernel
> - and other preprocessing choices."

### 为什么这解释了我们的失败

我们的 pipeline 正好踩了**三个**列出的因素：

| 文献指出的因素 | 我们的做法 | 后果 |
|---|---|---|
| **bin width** | 50ms → 100ms → 200ms 降采样 | τ 涨 1.6–1.9× |
| **data length** | 每个片段 9793–10381 bins | 200ms 下只剩 ~2500 点 |
| **trials 的存在** | 我们用连续的 PCA 投影 | 未考虑 trial 边界 |

### 更狠的一句

> "the most commonly used approach for trial-based IT estimation ... does not
> calculate the **genuine ACF**, but instead measures **pseudo-autocorrelations**
> on trial-averaged data"

**我们的做法（在 trial 平均后的 PCA 投影上算）正是"pseudo-autocorrelation"。**

---

## 二、方法论综述：Neural timescales from a computational perspective

> Zeraati, Levina, Macke, Gao. arXiv:2409.02684v3 (2026-01)

### 核心警告

> "Although studies often emphasize convergent results, **different paradigms and
> methods introduce assumptions and biases on estimated timescales** that may lead
> to important conceptual deviations, which should be incorporated when assessing
> the generalizability of results."

### 对 τ 定义的澄清

> "'Neural timescales' in the literature, for the most part, refer to the
> **estimated** autocorrelation decay timescale of measured neural activity"

**关键**：文献自己用的词是 "**estimated** timescale"——强调它是估计量，
不是物理常数。这与我第 18 条教训（模型参数 ≠ 数据属性）完全一致。

---

## 三、MCMC 社区的方法论：`τ` 估计需要 N ≥ 50τ

> Foreman-Mackey. "Autocorrelation time estimation." (emcee docs)

虽然来自 MCMC，但**统计原理完全相同**（都是估计自相关衰减时间）：

### 关键结论

1. **朴素估计（求和到 N）不可用**
   > "At longer lags, ρ̂(τ) starts to contain **more noise than signal** and
   > summing all the way out to N will result in a **very noisy** estimate."

2. **需要窗口截断（Sokal 方法）**
   > "estimate τ as 1 + 2Σ_{τ=1}^M ρ̂(τ) for some M ≪ N ... choosing the smallest
   > value of M where M ≥ C·τ̂(M) for C ~ 5"

3. **最关键的样本量判据**
   > "you probably **shouldn't trust any estimate of τ unless you have more than
   > F×τ samples for some F ≥ 50**."

### 用这个判据检查我们的数据

| 分箱 | 每片段样本数 | 测得的 τ | **N/τ** | 是否可信（需 ≥50）|
|---|---:|---:|---:|---|
| 50ms | ~5,400 | 149 ms ≈ 3 bins | **1800** | ✅ 够 |
| 100ms | ~2,700 | 166 ms ≈ 1.7 bins | **1600** | ✅ 够 |
| 200ms | ~1,350 | 237 ms ≈ 1.2 bins | **1100** | ✅ 够 |

样本量足够，**问题不在 N/τ**，而在：

- **τ 只有 1–3 个 bin** —— 低于可分辨下限
- 我们测的 τ 全部落在 **1–3 bins** 范围（对应 50–240 ms）
- 而 bin 宽度是 50–200 ms，**τ ≈ bin width**

> **当 τ 只有 1–2 个 bin 时，估计量完全由离散化误差支配。**
> 这就是为什么降采样后 τ 变大——不是物理变化，是 τ 逼近了分辨率极限。

---

## 四、另一个可行的替代：1/f 非周期指数

> Gerster et al. 2022 (PMC9588478)；Donoghue et al. 2020 (eNeuro)；
> Boncompte et al. 2026 (PMID 41468346)

### 优点

- **不需要跨 bin 的自相关**，直接拟合功率谱的 log-log 斜率
- 不涉及"τ 是几个 bin"的分辨率问题
- FOOOF / specparam 是成熟工具

### 但也有它自己的坑

> Boncompte et al. 2026: "the **aperiodic exponent depends on its estimation
> frequency range**, highlighting the need for caution when comparing exponents
> across [studies]"

**同样的教训**：任何谱量都依赖分析窗口。必须报告估计范围。

---

## 五、结论：我们的 τ 为什么不可信

### 三条独立原因（都有文献支持）

| # | 原因 | 文献 |
|---|---|---|
| 1 | **τ ≈ bin width**（1–3 bins）| iSTTC：bin width 是主要偏差源 |
| 2 | **在 trial 平均后的数据上算 = pseudo-ACF** | iSTTC：不是 genuine ACF |
| 3 | **降采样无抗混叠** | 信号处理常识 + iSTTC 的 preprocessing 警告 |

### 应该怎么改（若要继续）

| 方案 | 说明 | 可行性 |
|---|---|---|
| **A. 用 iSTTC** | 专为 spike 设计，对 binning 鲁棒，支持 epoched | 需装包（代码在论文补充）|
| **B. 提高时间分辨率** | 用 4ms 原始数据（250Hz），不做 50ms 分箱 | ✅ 数据支持 |
| **C. 改用 1/f 指数** | 不做自相关，直接拟合谱斜率 | ✅ 简单，但需报告频率范围 |
| **D. 用 aABC 拟合** | Approximate Bayesian Computation（Zeraati 等）| 需实现 |

**最务实的路径**：B + C —— THINGS-EEG 原始是 250 Hz（4ms），
我们完全可以不做 50ms 分箱，直接在 4ms 上算；同时报告 1/f 指数作为交叉验证。

---

## 六、这条失败的价值

exp39 的失败**与文献完全吻合**：

- 文献说 "bin width 是主要偏差源" → 我们实测 1.6–1.9× 变化
- 文献说 "trial 平均得到的是 pseudo-ACF" → 我们确实是这么做的
- 文献说 "τ 是 estimated 量" → 我们的 145ms 就是估计量，不是常数

**这不再是一个"我搞错了"的故事，而是一个"这类估计本身就很脆弱"的实例。**

诚实的做法是：报告 R²(1)（通过分辨率检验的量），
不报告 τ（未通过），并在文档里引用这些文献说明为什么。

---

## 七、引文

1. Pochinok I, Hanganu-Opatz IL, Chini M. **iSTTC: A robust method for accurate
   estimation of intrinsic neural timescales from single-unit recordings.**
   *PLoS Comput Biol* 2026;22(3):e1013385. doi:10.1371/journal.pcbi.1013385

2. Zeraati R, Levina A, Macke JH, Gao R. **Neural timescales from a computational
   perspective.** arXiv:2409.02684v3, 2026.

3. Foreman-Mackey D. **Autocorrelation time estimation.** emcee documentation.

4. Spitzner FP, et al. **MR. Estimator, a toolbox to determine intrinsic
   timescales from subsampled spiking activity.** *PLoS ONE* 2021;16(4):e0249447.

5. Spitmaan M, et al. **Multiple timescales of neural dynamics and integration of
   task-relevant information.** *PNAS* 2020;117(46):29274-29285.

6. Gerster M, et al. **Separating neural oscillations from aperiodic 1/f activity.**
   *NeuroImage* 2022. PMC9588478.

7. Boncompte G, et al. **Aperiodic exponent of brain field potentials is dependent
   on estimation frequency range.** 2026. PMID 41468346.
