# V51.0 — 架构异构线 (v6/v7) 的「融合收益」审计

> **日期**: 2026-10-25
> **状态**: 🚧 结果已出, 独立复算待做 (leakage_audit: pending)
> **复现**: `bash run.sh v51` / HF Space `redmingwei/hetero-moe`
> **一句话**: v50 审计了规模异构线 (v33); 本代把同样的两把尺子用到**架构异构线** (v6/v7) 上,
> 并发现这条线有一个更根本的问题: **它的目标块就是模态下标 —— 任务本身就是"识别来自哪个底座"。**

---

## 0. 被审计的对象

论文 §6.1/§6.2 与摘要:

> (a) "FFN-only routing beats the best single expert by $+20.3\%$ ... the first demonstration that
>      genuinely different pretrained architectures can share one sparse router"
> (b) "dual-path routing raises the margin over averaging to $+84.8\%$"
> (c) "the cross-architecture gain concentrates in the attention path ($+82.7\%$ in-sample, $+70.6\%$ held-out)"

前提事实: v6/v7 的 `P_m` 同样是 **"正交初始化, 冻结"** (见
`research/routing_evolution/v7_attention/fusion.py`: "P_m 矩形投影 (D_m → D_shared), 冻结")。

## 1. ★ 一个更根本的问题: 这条线的任务就是"识别来源底座"

v7 的 runner 里写得很直接:

```python
# 4. 目标: 每模态一个 class direction
targets = []
for cls in range(N_CLS):        # cls = 模态下标 (text / code / image)
    ds = cls * (D_SHARED // N_CLS)
    ...
```

即 **目标块 = 模态下标**。所以"融合优于单专家"在这条线上等价于
"**能分清模态的模型赢过分不清的模型**" —— 与"底座之间的语义融合"无关。
任何能识别模态的机制都会赢。

## 2. 两把尺子 + 一个探针

| 仪器 | 做法 | 判据 |
|---|---|---|
| **尺 1 目标置换** | 模态 m 的目标块改成 π(m) (固定无不动点错位)。任务难度不变 (仍只需认出模态再查表), 只有标签对应关系变了 | margin 若不变 → margin 是**识别机制**的产物, 与底座间任何语义关系无关 |
| **尺 2 有意义的空间** | 把冻结随机线性 P_m 换成**非线性可训练**抬升 (v50 验证过的配方) | margin 若塌 → 与 v50 在规模异构线上的结论一致 |
| **探针** | 从融合输出线性探测 (a) 源模态身份 (b) 目标块 (均为 held-out; 随机基线 1/3) | 融合输出编码的是来源还是语义 |

四臂 = 尺1 × 尺2: A 冻结+对齐 (v7 原配置) / B 冻结+置换 / C 可训练+对齐 / D 可训练+置换。

## 3. 协议

**完全复用 v7 的 runner** (模型、attn Q/K/V 抽取、层构造、目标构造) → 数字与论文同源。
TinyBERT (文本, D=312) + TinyLlama-110M (代码, D=768) + ViT-tiny (图像, D=192);
D_shared=256; 5 seeds; 100 步。v7 原本**没有 held-out**, 本代补了 (不同扰动种子) 并同时报两口径。

## 4. 结果 (5 seeds, held-out)

| 臂 | fuse | attn_only | ffn_only | avg | best single | vs 单专家 | vs 平均 | 探针 源模态 | 探针 目标块 |
|---|---|---|---|---|---|---|---|---|---|
| A 冻结随机 + 对齐 (v7 原配置) | 2.159 | 2.209 | 7.272 | 7.243 | 7.288 | **+70.4%** | +70.2% | 1.000 | 1.000 |
| B 冻结随机 + **目标置换** | 1.996 | 2.045 | 7.312 | 7.262 | 7.316 | **+72.7%** | +72.5% | 1.000 | 1.000 |
| C 非线性可训练 + 对齐 | 0.0315 | 0.0314 | 0.0324 | 0.0324 | 0.0322 | **+2.3%** | +2.8% | 1.000 | 1.000 |
| D 非线性可训练 + 置换 | 0.0291 | 0.0292 | 0.0305 | 0.0306 | 0.0305 | **+4.4%** | +4.7% | 1.000 | 1.000 |

## 5. 读数

- **尺 1 (A vs B): margin 完全不变** (+70.4% → +72.7%, 差异在噪声内)。
  把"模态 m → 块 m"改成"模态 m → 块 π(m)"之后, 融合相对单专家的领先幅度纹丝不动
  → **那个 margin 与底座之间的任何语义对应关系无关**, 它只取决于"能否分清模态"。
  这是"margin 是识别机制产物"的**决定性证据**。
- **尺 2 (A vs C): margin 塌掉** (+70.4% → +2.3%), 同时绝对误差从 2.159 降到 0.0315
  (**68 倍**) → 与 v50 在规模异构线上的发现**完全一致**。
- **探针: 四臂的源模态与目标块可解码性都是 1.000** → 因为任务本身就是模态识别,
  模型把它解得完美。**这解释了 margin 为何存在: 它是一个模态检测器的性能。**
- **★ attention 通路优势也随之消失**: A 臂 attn_only 2.209 vs ffn_only 7.272 (通路差 +69%,
  与论文的 +82.7% 同向); 而 C/D 臂 (有意义空间) 下 attn 0.0314 vs ffn 0.0324 → **通路差只剩 3%**。
  → 论文的第二个头条主张 ("增益集中在 attention 通路") 与 +26.7% 一样, **是随机空间下的产物**。

## 6. 因此论文必须改写的主张

| 原表述 | 改写后 |
|---|---|
| "FFN-only routing beats the best single expert by +20.3% ... the first demonstration that genuinely different pretrained architectures can share one sparse router" | "该配置的**目标块就是模态下标**; margin 衡量的是**模态识别**, 且在目标置换 (标签对应关系改变) 下**完全不变** → 它不构成'不同架构共享一个路由器'的证据" |
| "dual-path routing raises the margin over averaging to +84.8%" | 同上, 属识别机制的领先幅度, 在有意义空间下塌到 +2%~+5% |
| "the cross-architecture gain concentrates in the attention path (+82.7% / +70.6% held-out)" | "在冻结随机空间下确实观察到 attention 通路领先 (+69% held-out), 但在有意义空间下该领先**降到 3%** → 该主张是空间性质的产物, 而非架构异构性的产物" |

## 7. 诚实声明

1. 探针自检有效 (可分数据 train acc 0.98 / 噪声 0.44), 但**是线性探针**。
2. v6/v7 的原协议**没有 held-out**; 本代补的 held-out 只换扰动种子 (内容不变),
   因为 v7 的"内容"只有 3 段文本 + 3 段代码 + 1 张**随机图像** → 内容维度的泛化无法测。
3. ViT 专家的输入是**随机噪声图像** → 该专家的表征几乎不含语义, 这本身是协议弱点,
   但对"识别模态"的任务来说反而更容易 (随机图像极好认) → margin 被进一步放大。
4. 单一合成协议; 真实语料未验证; `leakage_audit: pending` → 独立复算前不得作为终稿主张。
