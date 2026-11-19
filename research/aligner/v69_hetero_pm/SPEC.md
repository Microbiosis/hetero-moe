# V69.0 — 异构 P_m（维度调解）

> **文档 ID**: HMF-HETERO-PM-1.0
> **日期**: 2026-11-12（补登记；工作完成于 2026-11-04，commit `88edadf` / `ac2c951`）
> **状态**: 机制**已本地实测**；端到端精度**待 Kaggle 批量验证**
> **归属**: aligner 线，接 v68
> **一句话**: 异构 P_m——大模型降维、小模型先局部提炼再稠密升维并加门控，替代"统一压到 256 维冻结随机正交"。

---

## 0. 本代性质

本代的**机制语义已本地实测**（可在 CPU 上执行真实类源码并验证），
但**没有端到端精度数字**——异构 P_m 要接进训练才见分晓，需 Kaggle 批量跑。

---

## 1. 问题

BERT(768) / ViT(1152) 对接 Qwen(2048) 时，原来那条"统一 256 维冻结随机正交"的路两头都不讨好：

| 做法 | 问题 |
|---|---|
| 统一压到 d_shared=256 | 信息残废（v62 实测：24 维保真度只有 0.165，256 维才 0.9998） |
| 补零升到 2048 | 稀疏，一半维度恒零 |

---

## 2. 机制

`pm_reconcile=True` 时，`d_shared` 从 256 提到 `pm_target`（默认 1024），每个模态按自身维度走不同路径：

| 模态 | 维度 | 走的路 | 类 |
|---|---|---|---|
| Qwen | 2048 | 降维 + LayerNorm | `DimDownProj` |
| BERT | 768 | 局部提炼 → 稠密升维 → 门控 | `DimReconcileBlock` |
| ViT | 1152 | 同上 | `DimReconcileBlock` |

`DimReconcileBlock` 三段：

```text
refined = refine_norm(x + local_refine(x))   # 局部提炼（多头注意力或 MLP）+ 残差
up      = up_proj(refined)                    # 稠密升维（不是补零）
g       = sigmoid(gate(refined))              # 门控 0~1，保护原生信息
out     = out_norm(up * g)
```

局部提炼的实现按可整除性退化：`d_src % n_heads == 0` 用 `nn.MultiheadAttention`，否则退化为 `Linear→GELU→Linear`。

**默认路径不变**：`pm_reconcile=False` 时仍是原来的"统一正交冻结 P_m"，可随时回退。

**v69b 追加**：调解/降维输出后再接一个 `target × target` 的**冻结正交去相关**，避免各模态调解后的表示互相缠绕。

---

## 3. 权威测试：7 项（本地 CPU）

测试直接**执行 `train_v2.py` 里的真实类源码**（用 AST 取出类片段后 exec），
而不是把类抄一份到测试里——抄的那份必然与实现漂移。

| # | 判据 | 实测 |
|---|---|---|
| 1 | `DimDownProj` 把大模型降到 target 维 | (2,5,2048) → (2,5,1024) |
| 2 | `DimReconcileBlock` 稠密升维，**没有补零** | 1024 个输出维无一为死维 |
| 3 | 不能整除时退化为 MLP 提炼，两条路都能跑 | 768→Attn，770→MLP |
| 4 | 默认路径仍是正交冻结 P_m | 3 组 [256,D]，行正交误差 < 1e-4，requires_grad=False |
| 5 | `pm_reconcile` 路径：d_shared=1024、无 P_m、换成 modal_proj | DimDownProj+DimReconcileBlock×2 |
| 6 | `PM_RECONCILE` / `PM_TARGET` 可切 | 默认 off/1024，可切 on/2048 |
| 7 | 调解路径在真实形状上能前向 | y (2,8,1024)，alpha (2,8,4) |

第 2 项是本代的核心断言：**"稠密升维"和"补零"在输出上可区分**——补零会留下一批恒零维。

---

## 4. 待执行

```bash
PM_RECONCILE=1 PM_TARGET=1024 python train_v2.py
```

**产物落盘位置**：训练产物 → HF `redmingwei/HeteroMoEProjSpace-v2/checkpoints/v69.*`；评估产物 → `research/aligner/v69_hetero_pm/results/*.json`。

---

## 5. 复现

```bash
python research/aligner/v69_hetero_pm/tests/test_package.py     # 7/7，本地 CPU
```

---

*本代机制已实测，端到端精度待验证。*
