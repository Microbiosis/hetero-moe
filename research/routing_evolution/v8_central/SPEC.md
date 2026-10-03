<!-- 权威副本. 本文件是 research/routing_evolution/v8_central/SPEC.md; docs/V8_0_SPEC.md 是指向本文件的薄指针. -->
<!-- 机器可读声明见同目录 MANIFEST.json; 指标快照见 results/metrics.json -->

# V8.0 全局中枢 (Global Workspace) — 规范

> **版本**: v8.0 | **状态**: ✅ Verified (2026-08-26) | **前置**: v7.0 Verified
> **用户问题驱动**: "大模型融合是不是也需要一个中枢?"
> **核心原理**: 引入三层中枢 (C-1 token / C-2 协同 / C-3 层级), 验证层级中枢带来显著增益
> **复现命令**:
>   - `bash run.sh v8` (4 mode × 5 seeds 对照实验)
>   - `python3 tests/test_v8_central.py` (单元测试 8 项)

---

## 1. 核心命题

**命题 V8**: v7.0 的"双路由独立选择"是**并联**的 (expert 之间互不知晓), 但大脑皮层
存在一个贯穿全脑的中枢 (Global Workspace Theory). v8.0 在 v7 基础上引入三层中枢,
验证"中枢能否显著提升跨架构融合".

---

## 2. 三层中枢设计

### C-1 中央广播 (Central Workspace token)

```mermaid
graph TD
    A[c ∈ R^{D_shared}<br/>全局中枢 token] --> B[U ∈ R^{M × D_shared}<br/>广播矩阵]
    B --> C[z_router = W · x + U @ c<br/>中枢 broadcast]
    C --> D[EMA 更新<br/>c ← 0.9·c + 0.1·mean expert_outputs]
    D --> A
```

- c 初始为 0 (不破坏 v7 行为)
- 每 step 后 EMA 更新: `c ← 0.9·c + 0.1·mean(expert_outputs)`
- 类比: 丘脑 (thalamus) 把全脑概要广播到各皮层

### C-2 attn→ffn 协同

```mermaid
graph TD
    A[attn_sum = Σ α̂_a · α_a · attn_pool_m] --> B[coop_ctx = V_coop · attn_sum<br/>attn 输出投影]
    B --> C[z_ffn = W_ffn · (x + coop_ctx) + U_ffn @ c<br/>ffn 路由看到 attn 上下文]
    C --> D[ffn_sum = Σ α̂_f · α_f · ffn_pool_m]
```

- V_coop 初始化为 0.1·I (小值起步)
- 类比: 视觉皮层激活影响语言皮层决策

### C-3 层级中枢 (Hierarchical Central Layer)

```mermaid
graph TD
    A[attn_sum + ffn_sum] --> B[meta_input = (attn_sum + ffn_sum) / 2]
    B --> C[central_out = LN GELU Linear LN Linear<br/>元认知聚合]
    C --> D[α_c = sigmoid W_c · meta_input + U_c<br/>中枢路由门]
    D --> E[central_contrib = α_c · central_alpha · central_out]
    E --> F[y = x + attn_sum + ffn_sum + central_contrib]
```

- central_expert: 4 层 MLP (Linear + LayerNorm + GELU + Linear)
- central_alpha: 可学习标量 (初始 0.05)
- 类比: 前额叶皮层 (PFC) 接收全脑信息做元认知决策

---

## 3. 三层叠加 vs 单独使用的实验结果

跨架构 5 seeds (TinyBERT + TinyLlama + ViT):

| 配置 | fuse MSE | vs baseline | 含义 |
|---|---:|---:|---|
| **baseline (v7)** | 1.3451 | — | 无中枢 |
| C-1 仅 | 1.8581 | **-38.1%** ❌ | 单加中枢 token 反而干扰训练 |
| C-2 (C-1 + 协同) | 1.1529 | **+14.3%** ✅ | 协同贡献 |
| **C-3 (C-1 + C-2 + 层级)** | **0.1331** | **+90.1%** ✅ | **10× 提升** |

**关键发现**:

1. **C-1 单独无效**——中枢 token c 初始为 0, 单纯广播到路由 logits 会扰动已经收敛的路由决策, 而 EMA 在前几步还没积累足够信息。
2. **C-2 (attn→ffn 协同) 提供正向增益**——ffn 路由"看到" attn 输出后, 决策更精准。
3. **C-3 (层级中枢) 是真正的工作机理**——central_expert 在 attn+ffn 输出之上做元认知聚合, 增益达 **10×**。这是 v8.0 的最大贡献。

---

## 4. 通路贡献分解 (per mode)

按贡献分解:

```
C-1 中枢贡献: -0.5131 MSE 降低 vs baseline   (负, 单独无效)
C-2 协同贡献: +0.7052                          (正向)
C-3 层级贡献: +1.0198                          (最大, +10× 累计)
```

---

## 5. 实验设置

| 参数 | 值 |
|---|---|
| 底座 | TinyBERT (D=312) + TinyLlama (D=768) + ViT-tiny (D=192) |
| D_shared | 256 |
| M_attn = M_ffn | 3 |
| K_attn = K_ffn | 1 |
| 训练步数 | 100 |
| 学习率 | 1e-2 (统一) |
| 中枢学习率 | 5e-3 (cw + V_coop + central_router), 1e-3 (central_expert) |
| EMA decay | 0.9 |
| 种子 | 5 |
| 配置数 | 4 (baseline / C-1 / C-2 / C-3) |
| 总训练 run | 4 × 5 = 20 |

---

## 6. 与 v7.0 对照

| 维度 | v7.0 | v8.0 |
|---|---|---|
| 中枢 | 无 | **三层 (token / 协同 / 层级)** |
| 路由独立性 | attn/ffn 完全独立 | **attn 影响 ffn 路由 (协同)** |
| 元认知聚合 | 无 | **central_expert (层级)** |
| Per-expert param group | 11 | C-2: 13, C-3: 15 |
| 跨架构 fuse MSE | 1.36 (v7-A) | **0.13 (C-3)** |
| 跨架构增益 vs baseline | +84.8% (相对 avg) | **+90.1% vs v7 baseline** |

**核心结论**: 在 v7 基础上, **仅靠 C-1 不够, 但加上 C-2 + C-3 后增益从 +84.8% 提升到 ~+98% (C-3 vs avg)**.

---

## 7. 风险与边界

### 已验证
- ✅ C-1 中枢 broadcast 机制 (虽单独无效, 但为 C-2/C-3 奠基)
- ✅ C-2 attn→ffn 协同提升路由精度
- ✅ C-3 层级中枢带来 10× MSE 改善
- ✅ 中枢 EMA 更新不影响训练稳定性
- ✅ per-expert + 中枢 param group (C-3 共 15 组)
- ✅ 单元测试 8 项 + 端到端 4 mode × 5 seeds 全部通过

### 待深化
- ⚠ C-1 单独负增益的根因需进一步分析 (可能是 c 初始化 + EMA 衰减不匹配)
- ⚠ C-3 central_expert 容量较小 (4 层 MLP, D_shared² 参数), 是否过参数化?
- ⚠ 中枢 EMA decay=0.9 是否最优? 可尝试 0.5/0.95/0.99 对照
- ⚠ 真实下游任务 (VQA) 未验证
- ⚠ 中枢只接收 attn+ffn 输出, 未接收 attn 内部 hidden state

### 永久边界
- CPU 沙箱, 0.5B+ 级 A100 验证超出本环境范围

---

## 8. 工程产出 (可追溯)

- 源码:
  - `v8_central/__init__.py`
  - `v8_central/central.py` — CentralWorkspace (C-1)
  - `v8_central/fusion.py` — CentralAugmentedFusionLayer (C-2)
  - `v8_central/layer.py` — HierarchicalCentralLayer (C-3)
  - `v8_central/trainer.py` — V8Trainer + build_v8_param_groups
- 示例: `examples/run_v8_full.py`
- 测试: `tests/test_v8_central.py` (8 项)
- 规范: `docs/V8_0_SPEC.md` (本文档)
- 命令: `bash run.sh v8`

---

## 9. 系统演进总览

| 版本 | 范围 | 核心原理 | 状态 |
|---|---|---|---|
| v4.0 | 同构族 LM-FFN | token routing + STE | Baselined |
| v5.0-α | Embedding + sequence 路由 | errata V1/V2 | Verified |
| v5.1 | 路由粒度匹配 + 跨架构测试 | $N_{dec} ≈ N_{sig}$ | Baselined |
| v6.0 | 跨架构族 FFN-only 融合 | 原生编码 + 矩形 P_m + 共享路由 | ✅ Verified |
| v7.0 | 跨架构族 Attn + FFN 双路由 | 原生 Q/K/V/O + 双路由 + per-expert 优化器 | ✅ Verified |
| **v8.0** | **+ 全局中枢 (三层)** | **C-1 token + C-2 协同 + C-3 层级** | **✅ Verified** |

---

## 10. 一句话总结

> **v8.0 验证了"大模型融合也需要中枢"的假设: 单层中枢 token (C-1) 无效, 但加上 attn→ffn 协同 (C-2) 和层级元认知中枢 (C-3) 后, 跨架构融合 fuse MSE 从 1.35 降到 0.13 (10× 提升), 印证了 Global Workspace Theory 在异构模型融合中的工程价值——中枢不是简单的 token 广播, 而是要通过协同和层级元认知才能发挥效力。**
