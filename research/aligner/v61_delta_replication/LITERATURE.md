# 文献对照：v60–v61 的发现有多少是重发现

> 检索时间：2026-10-28。**在此之前 v58–v61 全程未做过文献检索。**

## 一、最相关：这个分解已经有人提出并系统研究过

**Song & Zhong, "Uncovering hidden geometry in Transformers via disentangling
position and context"**, arXiv:2310.04861（2023-10，ICLR/OpenReview 在审）。
[论文页](https://openreview.net/forum?id=1M0qIxVKf6)

他们的分解：

```
h_{c,t} = μ + pos_t + ctx_c + resid_{c,t}
```

| 项 | 含义 |
| --- | --- |
| `μ` | 全局均值向量 |
| `pos_t` | **跨 context 在位置 t 上的均值** |
| `ctx_c` | **跨位置在 context c 上的均值** |
| `resid` | 残差 |

### 与我三臂的逐项对应

| 我的臂 | 变换 | 在他们框架中的含义 |
| --- | --- | --- |
| `raw` | 原样 | `μ + pos_t + ctx_c + resid` |
| `detrend` | `X − mean_over_samples` | 去掉 `pos_t` → `μ + ctx_c + resid` |
| `centered` | `X − mean_over_positions` | 去掉 `ctx_c` → `μ + pos_t + resid` |

**三条臂正好是他们分解的三项。** 这不是巧合——这个分解是按"对哪个轴取均值"
穷举出来的，任何认真做过轴消融的人都会落到同一组。

### 他们的结论，与我的重合部分

| 他们的发现 | 我的对应读数 | 判定 |
| --- | --- | --- |
| `(pos_t)` 是**低维、连续、常呈螺旋形** | `raw` Δ > 0（48/48）；`centered` Δ > 0 | **重发现** |
| `(ctx_c)` 按话题呈**簇结构** | 未测 | — |
| `pos_t` 与 `ctx_c` **近乎正交** | 未测（但可解释我 `centered`/`raw` 的 Δ 相近） | 未验证 |

**⇒ 我 v61 说的"位置通用成分平滑，使相邻位置更相似"不是新发现。**
它是 Song & Zhong 2023 的结论之一，而且他们做得多得多：跨层追踪螺旋形状、
量化低维性、给出可视化。我只是在 3 个模型上重测了它的一个标量推论。

## 二、`raw` 的 Δ>0 还可能不是"学到的"位置结构

**Zuo, Guerzhoy & Guerzhoy, "Position Information Emerges in Causal Transformers
Without Positional Encodings via Similarity of Nearby Embeddings"**,
COLING 2025 / arXiv:2501.00073。

> We observe that **nearby embeddings are more similar to each other than faraway
> embeddings**, allowing the transformer to potentially reconstruct the positions
> of tokens. We show that this pattern can occur in **both the trained and the
> randomly initialized** Transformer models with **causal attention and no
> positional encodings** over a common range of hyperparameters.

三点直接冲击我的解释：

1. **"相邻 embedding 更相似"是他们的核心观察**，措辞几乎与我一致。
2. 他们在**随机初始化**、**完全没有位置编码**的模型上也能复现
   ⇒ 这不来自 RoPE 或其他位置编码，而是**因果注意力掩码本身**的产物。
3. 因此我 v61 里"位置通用成分"这个说法暗示了"学到了位置表示"，
   **这个暗示是错的**——它更可能是架构副产品。

**⇒ 我之前关于 RoPE 的猜测应当撤回。** RoPE 并非必要条件。

## 三、为什么 `raw` 的余弦全都那么高：anisotropy

**Godey, de la Clergerie & Sagot, "Anisotropy Is Inherent to Self-Attention
in Transformers"**, EACL 2024 / arXiv:2401.12143。

> anisotropy … makes them unexpectedly close to each other in terms of angular
> distance (cosine-similarity) … anisotropy is actually **inherent to
> Transformers-based models**.

- 他们跨文本/视觉/语音验证，且与训练目标无关 ⇒ 不是"长尾词频"造成的。
- 机制：为了让注意力**尖锐**（低熵），`QKᵀ` 的点积必须大，
  于是 Q 与 K 的分布在训练早期就朝同一方向漂移。
- 直接对应我 `raw` 臂的读数：big 层内的 `M̄` 在 0.43–0.52 之间（全部偏高），
  而**跨样本、跨位置**的正常差异只有 0.02–0.05 量级。

**⇒ `raw` 的 Δ=0.086 是在这个窄锥内部的微小起伏，不是"整体相似"。**

## 四、`centered` 是一个标准变换——因此它的副作用值得写下来

**Mu & Viswanath, "All-but-the-Top: Simple and Effective Postprocessing for
Word Representations"**, ICLR 2018 / arXiv:1702.01417。

> 先减去均值 μ，再去掉前若干个主成分。（该文的 centering 步骤即为
> 我 `centered` 臂所做的操作。）

**Sasaki et al. (2023)**、**Hämmerl et al. (ACL 2023 Findings)** 等继续使用。
⇒ **均值中心化是去各向异性的标准第一步**，被大量论文当作无害预处理。

我的发现：**在 token 轴上做逐样本均值中心化，会引入一个与模型无关的
硬代数约束，产生 0.188（S=16）的假奇偶结构。** 见 FINDINGS 第七节。

### 为什么这会成为伪影（我现在的解释）

`centered` 做的是 **逐样本** 减均值：`x'_t = x_t − (1/S)Σ_u x_u`。
于是对每个样本都**精确**有 `Σ_t x'_t = 0`，即

```
Gram G = X'X'ᵀ 满足  G·1 = 0   （全 1 向量落进零空间）
      ⟺ 每行非对角元素之和 = −‖x'_i‖²
```

这是**恒等式，不是统计规律**，所以：

- 与模型无关 ⇒ 三个架构完全不同的模型给出同一值（0.18775 / 0.18810 / 0.18812）
- 强度随 `1/S` ⇒ 约束在小 S 上更紧（0.188 @ S=16 → 0.090 @ S=32）

**注意**：Song & Zhong 的 `pos_t` / `ctx_c` 是**跨样本**取均值，
不施加"单样本和为 0"这种硬约束。所以他们的分解没有这个伪影。
**我的 `centered` 臂 ≠ 他们的"去掉 ctx_c"** —— 我用了更粗暴的逐样本均值，
而正确的做法是跨样本回归掉 `ctx_c`。

## 五、余弦本身的可解释性也有专文批评

**Steck, Ekanadham & Kallus, "Is Cosine-Similarity of Embeddings Really About
Similarity?"**, arXiv:2403.05440。

> We derive analytically how cosine-similarity can yield **arbitrary and
> therefore meaningless `similarities'**… we caution against blindly using
> cosine-similarity.

与我的经验一致：v60 主判据（`corr(d, M̄(d))`）与 v60.1（Δ）给出**相反**的结论，
就是因为余弦在接近 −1（中心化后）时高度非线性，线性相关系数不可用。

## 六、对 v58–v61 的重新定位

| 代 | 我原本的表述 | 文献定位 |
| --- | --- | --- |
| v58 | "任意切块轴上面积律平凡" | 与 [Song&Zhong] 的 `pos_t`/`ctx_c` 正交性相关，但我没做轴消融就下了结论 |
| v59 | "χ 度量的是'两侧是否同源'" | 无直接对应，但我没控制 anisotropy，读数已被各向异性污染 |
| v60 | "线性相关意义下无局部性" | 与 [Steck] 对余弦的批评一致 |
| v61 | "位置结构两层且方向相反" | **`raw` 部分 = [Song&Zhong] 结论 1 = [Zuo] 的核心观察**；`detrend` 部分性质待查 |

## 七、还可能是新的部分

据本轮检索，**没有**找到以下内容的直接对应：

1. **`detrend` 臂的负 Δ**（去掉 `pos_t` 后，残差在相邻位置**反相关**）——
   即"相邻 token 的内容部分互相区分"。Song & Zhong 只描述了 `resid` 是否存在，
   没有报告它的距离结构。
2. **余弦中心化在 token 轴上产生的奇偶伪影**（第四节）。
   这更像一条方法论警告：把 All-but-the-Top 的 centering 从"词表轴"
   搬到"序列轴"时会出现假结构。

但要注意第 1 条：**"没搜到"不等于"新的"**。这个现象可能已有名字
（如 token-level redundancy / local decorrelation），只是我没用对关键词。

## 八、结论：应当如何改写 v61 的主张

**原主张**（过于自信）：
> 位置结构有两层且方向相反：共享位置成分使相邻更相似，残差在相邻位置互相区分。

**修改后**：
> 在 3 个模型 × 2 个长度上，把隐状态按 Song & Zhong (2023) 的
> `μ + pos_t + ctx_c + resid` 分解后：
> (a) 保留 `pos_t` 的臂给出相邻更相似（Δ≈0.09–0.11）——
>     这与该文对 `pos_t` "低维、连续、平滑"的描述一致，**不是新发现**，
>     且据 Zuo et al. (2025) 它可能只是因果注意力的架构副产物；
> (b) 去掉 `pos_t` 后，残差在相邻位置反相关（Δ≈−0.31）——
>     这部分我未在文献中找到对应，但也未做充分检索，**只能记为待确认**。
> (c) 逐样本均值中心化（All-but-the-Top 的 centering 搬到序列轴）
>     会引入与模型无关的奇偶伪影，强度 ~3/S —— 这是方法论警告，不是模型性质。

## 参考文献

1. Song, J., & Zhong, Y. (2023). Uncovering hidden geometry in Transformers via
   disentangling position and context. arXiv:2310.04861.
2. Zuo, C., Guerzhoy, P., & Guerzhoy, M. (2025). Position Information Emerges in
   Causal Transformers Without Positional Encodings via Similarity of Nearby
   Embeddings. COLING 2025, 9418–9430. arXiv:2501.00073.
3. Godey, N., de la Clergerie, É., & Sagot, B. (2024). Anisotropy Is Inherent to
   Self-Attention in Transformers. EACL 2024. arXiv:2401.12143.
4. Mu, J., & Viswanath, P. (2018). All-but-the-Top: Simple and Effective
   Postprocessing for Word Representations. ICLR 2018. arXiv:1702.01417.
5. Steck, H., Ekanadham, C., & Kallus, N. (2024). Is Cosine-Similarity of
   Embeddings Really About Similarity? arXiv:2403.05440.
6. Haviv, A., Ram, O., Press, O., Izsak, P., & Levy, O. (2022). Transformer
   Language Models without Positional Encodings Still Learn Positional
   Information. TACL. arXiv:2203.16634.
7. Su, J., et al. (2021). RoFormer: Enhanced Transformer with Rotary Position
   Embedding. arXiv:2104.09864.
