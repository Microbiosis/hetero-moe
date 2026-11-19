# V70.0 — VL 跨模态融合旁路

> **文档 ID**: HMF-VL-BYPASS-1.0
> **日期**: 2026-11-12（补登记；工作完成于 2026-11-04/05，commit `a64388d` / `70708a9` / `e4da8ae` / `6c02fd8` / `9ee9b0b` / `633b951`）
> **状态**: 机制**已本地实测**；端到端精度**待 Kaggle 批量验证**
> **归属**: aligner 线，接 v69
> **一句话**: 接入跨模态 VL 融合旁路——四槽把三模态预混合进 canonical，视觉与文本的细粒度交互在进融合层前就被抹平了；旁路直接吃分离的两路流做显式交互。

---

## 0. 本代性质

本代交付算子（`vl_fusion.py`，从用户仓库 `neuro-physical-hierarchical-moa-backbone` 抽出最小依赖）
与若干真实 bug 修复。算子机制**已本地实测**，端到端精度**待验证**。

---

## 1. 机制

`CrossModalVisionLanguageFusionSubOp`：

```text
v_vis = silu(depthwise_conv1d(v_raw))        # 视觉支路：双向局部空间上下文
t_txt = silu(t_raw)                          # 文本支路
joint = 0.5 * (v_vis + t_txt)
z     = attn_pool(anchors, joint) @ v_vis    # 低秩跨模态锚点共注意力（R 个锚点）
c_align = attn(t_txt, z) @ z                 # 文本对锚点求对齐
out = g ⊙ v_vis + (1-g) ⊙ t_txt + β ⊙ ((v_vis ⊙ t_txt + c_align) / √(1+λ‖·‖²))
g = sigmoid(α_vis ⊙ v_vis + α_txt ⊙ t_txt)   # Hadamard-Bilinear 门控
```

两个关键性质（都已在测试里锁住）：

1. **锚点共注意力零距离衰减**：早期图像 patch 与晚期文本 token 之间没有距离惩罚 ⇒ 末位文本能影响首位视觉输出（实测首位输出变化 9.8e-04）。
2. **门控是凸组合**：`β=0` 时输出逐维落在视觉流与文本流之间。

输出以**残差**方式并联回 canonical（`canon = canon + vl_out_proj(vl_out)`），不破坏四槽结构。

---

## 2. 本代修掉的真实 bug

| # | bug | 修法 | commit |
|---|---|---|---|
| 1 | `vl_out` 是 256 维，却要加到 1024 维 canonical 上 | 加 `vl_out_proj = Linear(d_shared, d_space)` 先投再残差 | `e4da8ae` |
| 2 | patch 维硬编码 128，实际 SigLIP2 是 197 | 改用动态 `Lp` | `9ee9b0b` |
| 3 | ViT hidden 全零（下载失败静默填占位） | 成功率断言 + 灰图占位 + UA 头 | `aa8a8d7`（v66）/ `6c02fd8` |
| 4 | Kaggle 上从包目录 import 失败 | 算子独立成 `vl_fusion.py`，train_v2 从包目录 import | `70708a9` |
| 5 | VSR 图像下载走 COCO 联网，Kaggle 上不通 | 改从 HF 预打包 zip（3 parts）拉取 | `633b951` |

另外 `6c02fd8` 交付了**视觉 hidden 有效性检查**：样本间 cosine + patch 差异 + 灰图对照——
用来回答"视觉通道到底有没有携带信息"。

---

## 3. 权威测试：7 项（本地 CPU）

| # | 判据 | 实测 |
|---|---|---|
| 1 | 算子前向形状正确 | (2,12,64) |
| 2 | `forward_from_projected` 两路序列长度必须相等 | 不等长报错 |
| 3 | β=0 时输出是凸组合 | 逐维落在两路之间 |
| 4 | 锚点共注意力零距离衰减 | 末位文本改动使首位输出变化 9.8e-04 |
| 5 | 门控参数梯度非零 | α_vis / α_txt / β_bilin + 两路输入梯度均非零 |
| 6 | `USE_VL` 可切 | 默认 off，可切 on |
| 7 | VL 输出投到 canonical 维度再残差 | `vl_out_proj` + 残差并联在位 |

---

## 4. 待执行

```bash
USE_VL=1 python train_v2.py
```

**产物落盘位置**：训练产物 → HF `redmingwei/HeteroMoEProjSpace-v2/checkpoints/v70.*`；评估产物 → `research/aligner/v70_vl_bypass/results/*.json`。

---

## 5. 复现

```bash
python research/aligner/v70_vl_bypass/tests/test_package.py     # 7/7，本地 CPU
```

---

*本代机制已实测，端到端精度待验证。*
