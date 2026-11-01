# V62.0 — 维度投影空间：空间不是大小的数，是按功能槽张成的和

> **文档 ID**: HMF-DIM-PROJ-SPACE-1.0
> **日期**: 2026-10-31
> **状态**: ✅ Verified（测量 + 算子 + 端到端训练 + 完整打包）
> **归属**: aligner 线，接 v48 → v49 → v50 → v51 → v52 → v53 → v57 → v58 → v59 → v60 → v61
> **复现**: `python research/aligner/v62_dimension_projection_space/measure.py` / `pytest tests/test_package.py`
> **一句话**: v48–v57 六代都在问「D_shared 该多大」，答案一直是「不是它」。
> 本代换问法：**空间 = Σ_s S_s**，每个功能槽一个子空间，三条正交的轴（槽 / 模型 / 对齐）。

---

## 0. 前六代问错了什么

| 代 | 做法 | 隐含前提 |
|---|---|---|
| v48–v54 | 扫 `D_shared ∈ {96,252,256,384}` | 维度是超参 |
| v55 | 证跨专家通道秩 ≡ `min(top_k, M−1)` | 秩由架构定死 |
| v56 | 扫 hidden `∈ {4..2048}` | 维度仍是超参 |
| **v57** | 维度变成**被观测的状态** | 空间就是状态 |

v57 的结论是「一个固定的数不可能同时适配秩为 3 和秩为 30 的数据」。
本代接受这个结论，但**不再追问那个数**——转而问：**空间是由什么张成的？**

答案：由功能槽张成。

```
空间 = Σ_s S_s ,   dim(空间) = Σ_s dim(S_s)
```

三条正交的轴：

| 轴 | 内容 |
|---|---|
| **槽轴** | 功能槽 s ∈ {Q,K,V,O}，各自一个子空间，维数独立实测 |
| **模型轴** | 每个模型在每个子空间里有自己的投影 `proj[s\|m]` |
| **对齐轴** | 成对对齐 `align[s\|m1>m2]`，**可复合**（`A_ac = A_bc ∘ A_ab`） |

---

## 1. 功能槽确实存在，且形式各异

按 `nn.Module` 结构对齐（不按参数名）：

| 槽 | Qwen | BERT | ViT | 形式差异 |
|---|---|---|---|---|
| Q | `self_attn.q_proj` | `attention.self.query` | `attention.q_proj` | — |
| K | `self_attn.k_proj` | `attention.self.key` | `attention.k_proj` | **Qwen 是 GQA，K 为 [128,896] 非方阵** |
| V | `self_attn.v_proj` | `attention.self.value` | `attention.v_proj` | 同上 |
| O | `self_attn.o_proj` | `attention.output.dense` | `attention.o_proj` | — |
| expand | `up_proj + gate_proj` | `intermediate.dense` | `mlp.fc1` | **SwiGLU 两路 vs 单路 GELU** |
| contract | `mlp.down_proj` | `output.dense` | `mlp.fc2` | — |

结构差异不止名字：pre-norm（Qwen/ViT）vs post-norm（BERT）、SwiGLU vs GELU、GQA vs MHA。

---

## 2. 同功能的权重谱跨架构确实更相似

`n = 1728 对`（3 模型 × 4 层 × 6 槽的全部跨模型组合）

| 量 | 值 |
|---|---|
| 同功能 JS 均值 / 中位 | **0.0302** / 0.0166 |
| 不同功能 JS 均值 / 中位 | 0.0401 / 0.0255 |
| Cohen d | 0.252 |
| 置换检验 p（5000 次） | **0.0002** |

**结论成立，但效应量中等（d=0.25）**，不能过度解读成「三个模型的对应层几乎一样」。

各槽通用性差异很大：

| 槽 | 跨模型 JS | 读作 |
|---|---|---|
| `contract` | **0.0087** | 最通用 |
| `O` | 0.0140 | 很通用 |
| `expand` | 0.0245 | 较通用 |
| `V` | 0.0406 | 中等 |
| `Q` | 0.0431 | 偏特异 |
| `K` | **0.0504** | 最特异 |

---

## 3. 两两都能对齐，三个不行 —— 而这不是 Qwen 的错

共享等维变换（256×256），对齐后 CKA：

| 槽 | Qwen×BERT | Qwen×ViT | BERT×ViT | **三模型** | 去 Qwen |
|---|---|---|---|---|---|
| Q | 0.9893 | 0.9883 | 0.9879 | **0.6767** | 0.9879 |
| O | 0.9898 | 0.9890 | 0.9876 | **0.7168** | 0.9876 |
| K | — | — | 0.9873 | — | — |
| V | — | — | 0.9882 | — | — |

**关键读数**：两两全部 0.987+；一旦要求**一个变换同时对齐三个**，掉到 0.677 / 0.717。

**这个失败与 Qwen 的架构无关**——去掉 Qwen，Q/O 立刻回到 0.988。真正的原因是
**一个 256×256 的变换对 3 个模型过定**：它要同时满足三对约束。

> ⚠️ 修正一处早期误判：K/V 之所以「看起来更容易对齐」，
> 不是因为它们更通用，而是因为 **Qwen 的 K/V 是 [128,896] 非方阵被跳过了**，只参与两模型。

---

## 4. 环路一致性约束修复了它

既然失败源于「多条对齐路径不自洽」，就显式要求它们自洽：

```python
consistency_loss = mean over (a,b,c) of MSE( W_{b→c} @ W_{a→b} , W_{a→c} )
```

| 量 | 训练前 | 训练后 |
|---|---|---|
| consistency loss | 0.007817 | **0.000000** |
| 三模型对齐 CKA | 0.677 | **0.9925** |
| ViT→BERT 直达 | — | 0.9925 |
| ViT→Qwen→BERT 复合 | — | 0.9925 |
| 两路径差 | — | 0.0038 |

---

## 5. 等维对齐：不压也不升

这是本代最反直觉的一条。秩扫描（K/V 槽，共享变换）：

| rank | CKA | 保真度 |
|---|---|---|
| 8 | 0.9981 | 0.060 |
| 24 | 0.9982 | **0.165** |
| 128 | 0.9982 | 0.712 |
| 192 | 0.9982 | 0.934 |
| **256** | 0.9982 | **0.9998** |
| 384 | 0.9982 | 0.99999 |
| 768 | 0.9983 | 1.0000 |

**CKA 从 8 维到 768 维全是 0.998** —— 对齐质量与维度无关。
**保真度才是维度的函数** —— 压到 24 维只剩 16.5% 信息，256 维无损，再往上无收益。

⇒ 结论：**等维对齐（256→256 正交变换）**。
早期版本用 24/28 维省参数，代价是丢 83% 信息，而 CKA 指标看不到这个损失。

---

## 6. 对照实验：对齐不是变换自己制造的

| 组 | 秩 前→后 | 对齐后 CKA |
|---|---|---|
| 真对齐 K | 168 → 30 | **0.9872** |
| 真对齐 V | 182 → 24 | **0.9918** |
| 假对齐 K×V（跨功能混合） | 200 → 198 | 0.414 |
| 假对齐 Q×O | 218 → 219 | 0.403 |
| 随机表示 | 676 → 666 | **0.057** |

随机表示对齐后 CKA 仅 0.057、秩几乎不降，说明**变换本身不能凭空压缩或对齐**。
真对齐的秩下降（168→30）与高 CKA 都是真实结构。

---

## 7. 多子空间并集

对齐后各槽表示的有效秩（95% 方差，4 层）：

| 槽 | 各层 前→后 | 降幅 |
|---|---|---|
| K | 154→34, 168→30, 160→33, 175→27 | **78–85%** |
| V | 179→24, 182→24, 187→23, 184→24 | **87–88%** |
| Q | 166→151, 195→166, 195→166, 198→166 | 9–16% |
| O | 194→165, 205→170, 208→172, 209→173 | 15–17% |

全空间（10 个表示）从 212–225 维降到 **131–139 维**（占 256 的 53%）。

---

## 8. 算子

### 8.1 DimProjectionSpace

```
输入  {model: {slot: [B,S,256]}}
  ↓ 每模型每槽投影到该槽子空间       proj[s|m]: 256 → dim_s
  ↓ 同位置 token 跨模型对齐（成对复合）align[s|m→ref]
  ↓ 合并成一个结构化 token
输出  struct [B,S,4,256]   canonical [B,S,1024]   alpha [B,S,4]
```

**token 数 S 不变** —— 合并发生在**模型轴**，不是 token 轴。

三个能力：

```python
op.separate(struct, 'K')                     # 分离：取出一个槽
op.recombine(struct, {'K': 0, 'V': 1})       # 重组：token0 的 K + token1 的 V
op.consistency_loss()                        # 环路一致性
op.alignment_loss(pairs)                     # 对齐监督
```

序列长度不同（Qwen/BERT 128，ViT 197）时用**交叉注意力**对齐，不截断也不池化到同一长度——序列长度本身是信息。

### 8.2 FusionOverProjSpace

| | v33 CrossModalFusionLayer | v62 FusionOverProjSpace |
|---|---|---|
| 输入 | 256 | **1024**（= Σ 子空间） |
| 专家 FFN | 256→1024→256 | **1024→4096→1024** |
| 专家参数 | 3.1M | **50.3M** |
| 专家状态 | frozen + `x.detach()` | **解冻，参与训练** |

> ⚠️ **解冻专家偏离 v33 原设定**。v48 SPEC 记载：v33 全流程真正被训练的只有
> `W_router` + 每专家 (γ,β,α) ≈ 2.3K 参数。本代的端到端结果在解冻前提下取得，
> **不能**与 v33 的 +26.7% 直接比较。

---

## 9. 端到端训练

数据：ag_news 4 分类，1 万条真实文本的 Qwen/BERT 隐藏状态 + 5 千张 food101 的 ViT 隐藏状态。
**按类别分层 80/20**（train 7998 / val 2002，四类比例对齐）。

| 指标 | 值 |
|---|---|
| held-out 准确率 | **0.9016**（随机 0.25） |
| 专家负载 | [0.545, 0.478, 0.484, 0.493] |
| 均衡系数 | **0.8774** |
| 路由熵 | 0.5016 |

**与先前三次失败的对照**：

| 实验 | 准确率 | 均衡系数 |
|---|---|---|
| 专家冻结 | ≈0.25（随机） | — |
| 专家解冻（256 维融合层） | ≈0.25 | 0.001（塌缩） |
| 共享融合 + 路由 | — | 0.0001（塌缩） |
| **v62（1024 维 + 投影空间）** | **0.9016** | **0.8774** |

前三次塌缩的共同根因：**目标函数只与模态有关、与内容无关**，路由拿不到有意义的梯度。
v62 换成真实分类任务（交叉熵 + 真实标签），路由与专家才得到有效监督。

> ⚠️ 局限：专家权重是**随机初始化**（项目从未从真实 FFN 抽取过 v33 的专家权重），
> 解冻后从 5000 万随机参数开始训练。0.9016 是 ag_news 分类准确率，不是通用能力。

---

## 10. 完整打包

**https://huggingface.co/redmingwei/HeteroMoEProjSpace-v2**

```
645,190,802 参数  |  2.58 GB
  qwen       494.03M
  bert        11.17M
  vit         86.39M
  proj_space   2.76M
  fusion      50.34M
```

| 验证 | 结果 |
|---|---|
| `missing` / `unexpected` | **0 / 0** |
| 加载后自洽（同实例两次前向） | **0.000e+00** |
| 跨实例 `max\|Δy\|` | 9.5e-07 |

**不是 bit 级一致**，原因明确：打包时 ViT 走 BERT 风格命名（`encoder.layer.N.attention.attention.query`），
重建时走 HF ViT 命名（`layers.N.attention.q_proj`），数学等价但浮点路径不同。
`model.remap_vit_state()` 负责两者互转。

---

## 11. 本代证伪的假设

| 假设 | 判定 | 证据 |
|---|---|---|
| 「Q/O 跨架构对不齐，所以该路由」 | **证伪** | 两两对齐 0.987+，与 K/V 无差别 |
| 「Qwen 的 GQA 结构导致对齐失败」 | **证伪** | 去掉 Qwen 后 CKA 仍 0.988；单对 Qwen×BERT 也 0.989 |
| 「K/V 更通用所以更容易融合」 | **证伪** | K/V 只是恰好只有 2 个模型参与 |
| 「压维可以省参数且不掉效果」 | **证伪** | 24 维保真度 0.165；CKA 指标看不到这个损失 |
| 「升维能增强表达」 | **证伪** | 256 已无损，384/512/768 无额外收益 |
| 「专家冻结时路由能学到内容结构」 | **证伪** | 探针 ≈ 随机（0.25）；专家随机且冻结时路由无内容梯度可用 |

---

## 12. 与本线前代的关系

| 代 | 主张 | v62 的处理 |
|---|---|---|
| v48 | D_shared=256 过配，数据只占 5–47 维 | **接受**，但改为按槽分配而非全局一个数 |
| v48 | 底座间有共享几何，线性方法抓不到 | **部分解决**：按槽分开抓，V/K 抓到（秩降 87%），Q/O 需成对复合 |
| v55 | 跨专家通道秩 ≡ min(top_k, M−1) | 未涉及（本代不碰路由带宽） |
| v57 | 维度应是状态不是超参 | **接受**，但实现为「每槽一个状态」而非单一标量 r |
| v58–v61 | 面积律 / token 局部性 | 未涉及（空间轴任意性问题上代已判定平凡） |

---

## 13. 复现

```bash
# 测量（需 huggingface_hub；权重切片来自 redmingwei/backbone-slices，约 295MB）
python research/aligner/v62_dimension_projection_space/measure.py

# 权威测试
python research/aligner/v62_dimension_projection_space/tests/test_package.py

# 净室验证：干净 clone + 只从 Hub 下载，检验「别人拿到能不能用」
python research/aligner/v62_dimension_projection_space/cleanroom.py --json /tmp/cr.json
```

加载打包模型（`trust_remote_code=True` 需要 Hub 上的 `modeling_hetero_moe_projspace.py`）：

```python
from transformers import AutoModel
m = AutoModel.from_pretrained('redmingwei/HeteroMoEProjSpace-v2', trust_remote_code=True)
print(m.describe())
```

---

## 13b. 净室验证（第三方能不能用起来）

`cleanroom.py` 在**干净目录**里只靠 `git clone` + Hub 下载跑 8 步，与 `tests/test_package.py`
分开：后者验**实现对不对**（作者本机），前者验**交付能不能被用起来**。

| 步骤 | 内容 | 结果 |
|---|---|---|
| 1a | 从 GitHub `--depth 1` clone | ✓ commit 8330a7a |
| 1b | 子包 9 个必备文件齐全 | ✓ |
| 2a | 版本：torch 2.10.0 / transformers 5.16.1 | — |
| 3a | clone 出来的包直接 import + 前向 | ✓ 路由槽 Q,O |
| 4a | 包内权威测试在 clone 里跑通 | ✓ 12/12 |
| 5a | `AutoModel.from_pretrained(trust_remote_code=True)` | ✓ 645.19M |
| 6a | 用 Hub 的 example_inputs 复现 expected_y | ✓ Δ=7.9e-05 |
| 6b | 缺模态可用（只给 Qwen） | ✓ |

### 净室暴露并修掉的两个真问题

**问题一：`trust_remote_code=True` 当时根本无法工作。**
第一次净室跑第 5 步就失败：

```
ValueError: The checkpoint you are trying to load has model type
`hetero_moe_projspace` but Transformers does not recognize this architecture.
```

原因：Hub 上只有 `config.json` 和权重，**没有建模代码**，`auto_map` 也没写。
`save_pretrained` 只保存配置与权重，不保存 Python 文件。
修法：生成一个**自包含单文件** `modeling_hetero_moe_projspace.py`（把
projspace.py / fusion.py / model.py 内联，不 import 仓库任何模块），
并在 config.json 加 `auto_map` 指向它。

**问题二：只给文本、不给图像时模型直接崩。**

```
AssertionError: router 输入宽度 1024 != 声明 1536（实际可用模型与声明不符）
```

router 宽度在 `__init__` 里按「声明的全部模型」算死（Q/O 槽各 3 个模型 = 1536），
但前向时只传入 Qwen+BERT 就只有 1024 维。
修法：按**声明**的模型数固定宽度，缺失的模态用零向量占位
（`nn.Linear` 无 bias，零输入 → 零贡献 = 「该模态缺席」，语义正确）。
已补为第 12 项回归测试 `test_partial_modality`（5 种输入组合）。

### 精度：权重完美，但数值复现受版本影响

| 项 | 结果 |
|---|---|
| 权重加载 | missing 0 / unexpected 0 |
| 同实例两次前向 | Δ = 0.000e+00（确定性） |
| 跨实例复现 expected_y | Δ = 7.911e-05 |

`expected_y` 在 **transformers 5.21.0** 下生成，净室环境是 **5.16.1**。
同一实例内完全确定（Δ=0），说明差异来自不同版本骨干 forward 的实现细节，
不是加载错误。`cleanroom.py` 的容差设为 1e-3（相对量级约 2e-6），
既能抓住真错，也不会因版本漂移误报。

> 复现方若要 bit 级一致，需 pin `transformers==5.21.0`。

---

## 14. 未做的事（明确列出）

- **没做**：bit 级跨版本复现（需 pin transformers 版本；实测 5.16.1 vs 5.21.0 差 7.9e-05）

- **没做**：把 v48–v61 的其它观测量（面积律、token 局部性）纳入本算子
- **没做**：在 v33 的真实 fuse_mse 协议上重跑（本代用的是 ag_news 分类）
- **没做**：专家权重从真实 FFN 抽取（项目从未做过）
- **没做**：rank 的逐槽最优解（本代统一取等维 256）
