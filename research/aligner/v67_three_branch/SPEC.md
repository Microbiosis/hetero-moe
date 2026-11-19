# V67.0 — 去掉 SSM 分支（3 异构 proposer）

> **文档 ID**: HMF-THREE-BRANCH-1.0
> **日期**: 2026-11-12（补登记；工作完成于 2026-11-04，commit `5414c9c`；2026-11-06 补 `v67b` 两个修复）
> **状态**: 代码就位，**Kaggle 批量验证待执行**
> **归属**: aligner 线，接 v66
> **一句话**: 把 MoA proposer 从四分支降为三分支（Attention / Conv / SwiGLU）——避开 SSM 状态维度与显存、并行扫描之间的三角冲突。

---

## 0. 本代性质

**本代没有端到端数字。**本代只改分支组成，不改评估协议；端到端数字待 Kaggle 批量跑。

但本代**有一条有出处的实测结论**（来自 commit `c496d6d` 的记录）：见 §2。

---

## 1. 机制

`HeterogeneousProposerOp` 的分支轮转从 `kind = i % 4` 改为 `kind = i % 3`：

| kind | 分支 | 归纳偏置 |
|---|---|---|
| 0 | `CausalSelfAttentionSubOp` | 长程内容依赖 |
| 1 | `CausalDepthwiseConvSubOp` | 局部时序模式 |
| 2 | `SwiGLUMLPSubOp` | 逐 token 非线性变换 |

删掉的是 `SelectiveSSMSubOp` 分支。**类本身仍留在文件里**（删的是接线，不是研究）——这样后续换回只需改一个数字，且 v66 折腾过的 SSM 扫描实现没有白写。

---

## 2. 为什么删 SSM：两条独立理由

### 2.1 机制冲突

SSM 的状态维度 `d_state` 与显存占用、并行扫描的中间张量构成三角冲突（详见 `v66_vsr_triton/SPEC.md` §4.1）：chunkwise SSD 要 6 维中间张量、Kogge-Stone 在 D256×N16 下要 1.5GB、`diff_log_a` 在 25GB 量级——都必 OOM。

### 2.2 实测更差（不是"删了省显存"，是"删了更好"）

commit `c496d6d` 的实测记录：

| 配置 | held-out |
|---|---|
| 3 分支（Attn/Conv/SwiGLU） | **0.64** |
| 4 分支 + zero-OOM SSM | 0.53 |

⇒ SSM 分支不仅难跑，还把精度拖累了 0.11。这条是本代最硬的依据。

---

## 3. 本代附带的两个修复（v67b，2026-11-06）

1. **ViT 197 patch → 文本 128 token 池化**（commit `312fedf`）：VL 融合要求两路序列等长，否则 `v_vis + t_txt` 逐位相加会崩。修法是按 `linspace` 索引把视觉序列池化到文本长度。
2. **raw 兜底列表补 `vl_fusion.py` / `moa_fusion.py`**（commit `24a6f48`）：Kaggle clone 不完整时走 raw 下载兜底，漏了这两个文件会让 VL / MoA 分支 import 失败。

---

## 4. 权威测试：6 项（本地 CPU）

| # | 判据 | 实测 |
|---|---|---|
| 1 | 轮转确实是 `i % 3` | `kind = i % 3` |
| 2 | proposer 里不再实例化 SSM 分支 | 0 处 |
| 3 | 三种归纳偏置按序轮转 | Attn/Conv/SwiGLU ×2 |
| 4 | 前向仍可跑、形状不变 | `(2,4,12,64)` |
| 5 | `SelectiveSSMSubOp` 类仍在文件里（未接线而非已删除） | 类在 line 563 |
| 6 | zero-OOM 变体的 SSM 通路仍在（v68 的实验底座） | `branch_kinds = [i % 4 ...]` + SSM 实例化 |

---

## 5. 已知缺陷（本代登记，不在本代修）

**`profile_moa_operator_complexity` 仍按四分支估算参数。**

`moa_operators.py:921` 里是 `kind = i % 4` 且注释仍写 "Attention + SSM + DepthwiseConv + SwiGLU"，
而实际 proposer 从 v67 起是三分支。⇒ **该函数在 v67 之后给出的参数账与实现不符，不可直接引用。**

修它需要动 v62 工作目录里的公共代码（可能影响其他代对它的引用），故本代只登记、不擅改。
证据：`research/aligner/v62_dimension_projection_space/moa_operators.py:921`。

---

## 6. 复现

```bash
python research/aligner/v67_three_branch/tests/test_package.py     # 6/6，本地 CPU
```

---

*本代无端到端数字；任何引用不得把「代码就位」说成「已验证有效」。*
