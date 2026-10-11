<!-- 权威副本. 本文件是 research/aligner/v10_embedding/SPEC.md; docs/V10_0_ALIGNER.md 是指向本文件的薄指针. -->
<!-- 机器可读声明见同目录 MANIFEST.json; 指标快照见 results/metrics.json -->

# V10.0 — 语义对齐器 (Semantic Aligner) 路线

> **日期**: 2026-08-26
> **状态**: ✅ Verified (5/5 PASS)
> **复现命令**: `python3 examples/run_v10_aligner.py`
> **核心结论**: **attn < mean < linear**, 语义对齐器显著优于线性 P_m

---

## 1. 动机

用户问题: *"能不能使用一个巧妙的工具将不同的底座联系为一个新的底座?"*

v6/v7/v8/v9 的跨架构对齐用的是**线性投影** `P_m ∈ R^{D_shared × D_m}`——异构底座独立投影, 互不知晓。

v10.0 提出**语义对齐器 (SemanticAligner)**: 跨底座共享 K/V 注意力, 让异构底座的"同一概念"在共享空间自然对齐。

## 2. v10 与 v8/v9 的路线关系

```
v8  → 探索三层中枢叠加 (C-1/C-2/C-3)         # 中枢整体能带来多少增益?
v9  → 修正单独中枢 token (gate-style)        # 中枢 token 单独能不能工作?
v10 → 更好的跨架构对齐 (语义对齐器)            # 线性 P_m 能不能被替代?
```

**三条路线平行, 都接 v7**, 不互相依赖。

## 3. 设计: 三个语义对齐器

### LinearAligner (v6/v7/v9 默认)
```python
x_shared_m = h_m · P_m       # 每个底座独立投影
```
- 简单, 但跨架构的非线性差异被压平

### CrossArchMeanAligner (工具 A)
```python
z_m = h_m · W_down_m              # D_m → d_shared (降维)
x_shared = mean(z_A, z_B, z_C)     # 跨底座均值
```
- 简单, 但没有"概念对齐"

### CrossArchAttnAligner (工具 B)
```python
z_m = h_m · W_down_m                          # 各底座降维
z_mean = mean(z_A, z_B, z_C)                   # 跨底座平均
shared_k = z_mean · W_k
shared_v = z_mean · W_v                        # 共享 K/V
q_m = z_m · W_q                                # 各自 Q
α_m = softmax(q_m · shared_k^T / √D)          # 查询共享概念
concept_m = α_m · shared_v
aligned_m = z_m + concept_m                     # 残差
x_shared = mean(aligned_A, aligned_B, aligned_C)
```

**关键创新**: 不是简单的 `h · P_m` 投影, 而是**跨底座的注意力**——每个底座的 token 查询"共享概念空间"。共享的 K/V 让不同底座的同一概念在 attention 后自然对齐。

## 4. 实验设计

3 变体 × 5 seeds, 跨架构 (TinyBERT + TinyLlama + ViT):

| mode | aligner | 说明 |
|---|---|---|
| v9-linear | 矩形线性 P_m | 基线 |
| v10-mean | 跨底座均值 | 简化对照 |
| v10-attn | 跨底座共享 K/V | 完整路线 |

## 5. 实验结果

```
        mode |  fuse MSE |   vs linear
   v9-linear |    1.0436 |       +0.0%
    v10-mean |    0.4753 |      +54.5%
    v10-attn |    0.3448 |      +67.0%
```

| mode | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | 均值 |
|---|---:|---:|---:|---:|---:|---:|
| v9-linear | 1.0120 | 1.7701 | 0.8530 | 0.6639 | 0.9192 | **1.0436** |
| v10-mean  | 0.3997 | 0.4204 | 0.3824 | 0.5990 | 0.5749 | **0.4753** |
| v10-attn  | 0.3715 | 0.3201 | 0.2979 | 0.3333 | 0.4012 | **0.3448** |

**`attn < mean < linear` 严格成立** — 语义对齐路线显著有效。

## 6. 结论

### 6.1 语义对齐显著优于线性投影

- v10-attn vs v9-linear: **+67.0%** 改善 (fuse MSE 1.04 → 0.34)
- v10-mean vs v9-linear: **+54.5%** 改善 (fuse MSE 1.04 → 0.48)
- 共享 K/V 注意力比简单均值更优: attn 比 mean 再改善 **27.5%**

### 6.2 这印证了什么

**"用巧妙的工具联系不同底座"是有效的**——v6/v7 的线性 P_m 把异构信息压平了。v10 的共享 K/V 让不同底座的 token 在统一语义空间"互相看见", 信息保留度大幅提升。

### 6.3 与 v9 gate-style 中枢的关系

v10 复用了 v9 的 gate-style 中枢 (`central_mode="gate"`, `freeze_u=True`)。说明**v9 的中枢修正和 v10 的语义对齐是叠加增益**——两个改进可以组合使用。

## 7. 工程产出

- `v10_embedding/__init__.py`
- `v10_embedding/aligner.py` — `CrossArchMeanAligner` + `CrossArchAttnAligner`
- `v10_embedding/fusion.py` — `AlignedFusionLayer` (替换 P_m 为 aligner)
- `examples/run_v10_aligner.py` — 3 变体 × 5 seeds 对照
- `tests/test_v10_embedding.py` — 8 项单元测试
- `pyproject.toml` — packages.include 加 `v10_embedding*`

## 8. 给 v11 的方向

v10 验证了"语义对齐器显著优于线性 P_m", 但 v10 仍然是**适配器**模式 (跨底座协同, 推理时所有底座都需存活)。

下一步 (v11) 可以走**嵌合 + 蒸馏**路线:
- 用 v10 的语义对齐器作为"教师" (跨架构协同)
- 蒸馏到单底座 (例如 TinyBERT + LoRA)
- 让 TinyBERT 学会"什么时候调用 Llama 风格思考" + "什么时候调用 ViT 风格思考"

详见 `docs/V10_1_DISTILLATION_ROADMAP.md` (待写)。

## 9. 一句话总结

> **v10.0 语义对齐器 (跨底座共享 K/V 注意力) 显著优于 v9.0 线性 P_m: attn 改善 +67.0%, mean 改善 +54.5%. 验证了"用巧妙的工具联系不同底座"是有效的——不同底座的 token 在共享概念空间自然对齐, 信息保留度大幅提升.**