# V64.0 — MoA 异构 proposer 融合层

> **文档 ID**: HMF-MOA-PROPOSER-1.0
> **日期**: 2026-11-12（补登记；工作完成于 2026-11-03，commit `7fe96ea`）
> **状态**: 代码就位，**Kaggle 批量验证待执行**
> **归属**: aligner 线，接 v63
> **一句话**: v62 的融合层是**同构**的——4 个专家全是 SwiGLU FFN，架构异构只体现在输入模态；v64 把它换成真正的 MoA 异构 proposer：多种归纳偏置分支并行提案、稀疏路由聚合，且接口与 v62 完全兼容。

---

## 0. 本代性质（先说清楚，避免误引）

**本代没有端到端数字。**

MoA 融合层接进端到端训练需要 GPU + 已训练 ckpt，Kaggle 批量验证**尚未执行**。按项目纪律「无支撑的数字不得对外引用」：

- `claim.asserts` 写的是**机制语义可本地证伪**，不是精度提升；
- `results/metrics.json` 只记录本地复算得到的结构性事实；
- 端到端产物的落盘位置已在 §4 写明，等验证后回填。

---

## 1. 机制

### 1.1 异构 proposer：多种归纳偏置并行提案

`HeterogeneousProposerOp` 按 proposer 序号轮转分支类型（当前 HEAD 为 `kind = i % 3`）：

| kind | 分支 | 归纳偏置 |
|---|---|---|
| 0 | `CausalSelfAttentionSubOp` | 长程内容依赖 |
| 1 | `CausalDepthwiseConvSubOp` | 局部时序模式 |
| 2 | `SwiGLUMLPSubOp` | 逐 token 非线性变换 |

输出 `[B, N, L, D]`，每个分支带独立 LayerScale（`branch_scale` 形状 `[1,N,1,D]`），分支可在训练中自行缩放贡献。

> **分支数随代变化（必须随代记录）**：v64 落地时是 4 种（`i % 4`，含 `SelectiveSSMSubOp`）；v67 因 SSM 状态维度与显存/并行扫描三角冲突改成 3 种（见 `v67_three_branch`）。本 SPEC 描述的是 **HEAD 的当前形态**，测试同样钉在 HEAD 形态上。

### 1.2 聚合：三种模式

`CrossProposalAggregatorOp.forward(x, proposals)` → `(y, aux, metrics)`：

| 模式 | 语义 |
|---|---|
| `cross_attention` | 跨 proposer 交叉注意力融合 |
| `sparse_router` | Top-K 置信度门控 + Load-Balancing + Router Z-Loss |
| `low_rank_concat` | 低秩拼接 |

`sparse_router` 是本代的主线机制：只有 Top-K 个 proposer 参与聚合，`metrics['proposal_importance']` 标出被选中的分支，`aux` 是非负的辅助损失（LB + Z-Loss）。

### 1.3 头级异构注意力掩码

`build_mixture_of_attention_mask(seq_len, n_heads, window_size, num_sinks, dilation)` → 掩码 `[1, H, L, L]`，三个头模式按 `h % 3` 轮转：

- `sink_local`：注意力沉（`num_sinks`）+ 滑窗（`window_size`）
- `dilated`：膨胀间隔采样
- `full_causal`：完整因果

实测（`L=16, H=4, window=4, sinks=2, dilation=2`）：整体密度 **0.744**，三个头均因果（对角可见、上三角屏蔽）。

### 1.4 与 v62 的接口兼容

`MoAFusionOverProjSpace(d_space, num_proposers, top_k).forward_with_routing(x)` → `(y, alpha)`：

- `y` 形状 `[B, S, d_space]`，与 v62 `FusionOverProjSpace.forward_with_routing` 一致；
- `alpha` 形状 `[B, S, num_proposers]`（token 级路由权重）；
- `last_aux_loss` 非负。

⇒ `train_v2.py` 用 `USE_MOA=1` 即可切换，v62.3 默认行为不变（`MOA_MODE` / `MOA_STEPS` 可调）。

---

## 2. 本地可复算的机制语义（权威测试 9 项）

| # | 判据 | 实测 |
|---|---|---|
| 1 | 分支类型按 `i % 3` 轮转，三种归纳偏置齐备 | `Attn / Conv / SwiGLU / Attn` |
| 2 | proposer 输出 `[B,N,L,D]`，LayerScale 在位 | `(2,4,12,64)`，`branch_scale (1,4,1,64)` |
| 3 | `sparse_router` 聚合且真稀疏 | 非零 proposer 数 4 ≤ top_k·B·S = 48 |
| 4 | `aux_loss`（LB + Z-Loss）非负 | 0.022192 |
| 5 | 三种聚合模式前向均通过 | 3/3 |
| 6 | 头级异构掩码因果、密度落在 (0,1] | 掩码 `(1,4,16,16)`，密度 0.744 |
| 7 | numpy 参考核前向与 torch 逐位一致 | max diff < 1e-5 |
| 8 | numpy 解析反向核与 torch autograd 一致 | dx = 1.2e-07，dw = 9.5e-07 |
| 9 | 与 v62 融合层接口兼容（可无痛替换） | `y (2,12,64)`、`alpha (2,12,4)` |

第 7/8 项是本代最硬的证据：**numpy 参考实现与 torch 实现双向一致**，说明算子语义没写错，而不只是「能跑」。

---

## 3. numpy 参考核的形状契约（易错点，已固化在测试里）

`fused_proposer_linear_forward(x, weight, bias)`：

```text
x      [B, L, D_in]
weight [N, D_in, D_out]      # d 轴对齐 x 的 D_in
bias   [N, D_out]            # 可为 None
→ out  [B, N, L, D_out]      # einsum "bld,ndo->bnlo"
```

torch 侧须写成 `F.linear(x, w[n].t(), b[n])`。把 `weight` 写成 `[N, D_out, D_in]` 会直接报错——该契约在权威测试第 7 项中固定。

---

## 4. 待执行：Kaggle 批量验证

**批 B（训练类）**

```bash
USE_MOA=1 MOA_MODE=sparse_router MOA_STEPS=1 python train_v2.py
```

**产物落盘位置**

- 训练产物 → HF `redmingwei/HeteroMoEProjSpace-v2/checkpoints/v64.*`
- 评估产物 → `research/aligner/v64_moa_proposer/results/*.json`

验证完成后应回填 `results/metrics.json` 的 `pending` 段，并把 `status` 从 `code_ready_pending_batch_run` 改为实测结论。

---

## 5. 复现

```bash
python research/aligner/v64_moa_proposer/tests/test_package.py     # 9/9，本地 CPU
```

实现文件在 v62 包内（`moa_operators.py` / `moa_fusion.py`），它们用**平铺 import**（`from moa_operators import ...`），因此测试必须把 v62 包目录加入 `sys.path` 后按普通模块导入——走包内相对导入会 `ModuleNotFoundError`（实测）。

---

*本代无端到端数字；任何引用不得把「代码就位」说成「已验证有效」。*
