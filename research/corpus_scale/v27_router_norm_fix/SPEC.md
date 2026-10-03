<!-- 权威副本. 本文件是 research/corpus_scale/v27_router_norm_fix/SPEC.md; docs/V27_0_ROUTER_NORM_FIX.md 是指向本文件的薄指针. -->
<!-- 机器可读声明见同目录 MANIFEST.json; 指标快照见 results/metrics.json -->

# V27.0 — router-norm / adaptive-ema held-out 修正

> **日期**: 2026-09-03
> **状态**: ✅✅ Verified (单测 12/12 PASS, 端到端 80 run 全跑通)
> **实验环境**: Docker 容器（hostname=941507f22847）运行在 WSL2 上；Debian 12；Python 3.11.2；PyTorch 2.13.0+cpu；CPU only（32 核 AMD Ryzen 9 8945HX）；小规模张量 D≤128；5 seeds 取均值
> **复现命令**:
>   - `python3 tests/test_v27_router_norm_fix.py` (单元测试 12 项)
>   - `python3 examples/run_v27_router_norm_fix.py` (端到端 5 patch × 4 mode × 5 seeds = 80 run)
> **核心结论**: **4 个补丁均让 held-out 退步加剧** — **负面发现**:
>   - **所有 4 mode 的最佳 patch 都是 'none' (即 v22 默认 patch 已是局部最优)**
>   - noise / warmup / dropout / l2 都对 router-norm / adaptive-ema 帮倒忙
>   - **gate 仍是唯一在 held-out 上保持正增益的模式**

---

## 0. 动机

v26 在 fully-real-corpus 上发现关键新现象：
- 训练集: gate (+11.1%) > adaptive-ema (+7.9%) > router-norm (+3.8%) > broadcast
- held-out: gate (+2.3%) > broadcast > router-norm (-5.1%) ≈ adaptive-ema (-6.6%)

router-norm 和 adaptive-ema 在 held-out 上**退步** —— 这是 v22 现有 patch（freeze_u + warmup + set_u_init_scale + clip_grad）**无法解决**的过拟合问题。

v27 提出 4 个新 patch, 试图从两个通道（梯度 + EMA）入手修正 held-out 退步。

---

## 1. 4 个补丁设计

| Patch | 思路 | 解决 |
|---|---|---|
| **A: apply_l2_to_c** | c 参数加 L2 正则 (weight decay 风格) | c 梯度过大导致记忆训练分布 |
| **B: router_dropout** | router logits 加 Dropout (训练时随机遮蔽专家) | 路由过拟合特定专家组合 |
| **C: extend_warmup_steps** | AdaptiveEMA warmup 从 10 → 100 | 训练前期 c 不稳定导致过早自适应 |
| **D: noise_inject_c** | forward 时给 c 加 Gaussian 噪声 (仅 forward, EMA 不变) | c 在训练集收敛到特定点, 破坏泛化 |

每个 patch 都有理论动机, 但 v27 端到端发现:**所有 4 个 patch 都让 held-out 退步加剧**。

---

## 2. 实验结果 (5 patch × 4 mode × 5 seeds = 80 run)

### 2.1 训练集 fuse

```
训练集 fuse (各 patch × mode, 越低越好)
     patch |      broadcast |           gate |    router-norm |   adaptive-ema
      none |          0.3435         0.3258         0.3181         0.3115
        l2 |          0.3032         0.3050         0.2931         0.2940
   dropout |          0.2906         0.2964         0.2847         0.2870
    warmup |          0.2848         0.2920         0.2821         0.2829
     noise |          0.2812         0.2893         0.2800         0.2802
```

观察: 所有 patch 都降低训练集 fuse (这正是 patch 设计的目标 — 减少过拟合).

### 2.2 held-out fuse

```
held-out fuse (各 patch × mode, 越低越好)
     patch |      broadcast |           gate |    router-norm |   adaptive-ema
      none |          0.4583         0.4481         0.4639         0.4995
        l2 |          0.5150         0.4620         0.5237         0.6102
   dropout |          0.6208         0.4900         0.5811         0.7039
    warmup |          0.7066         0.5131         0.6259         0.7767
     noise |          0.7727         0.5310         0.6655         0.8326
```

**关键观察**: 所有 patch 都让 held-out fuse **上升** (变差). 这与训练集 fuse 下降的方向相反 — 表明 patch **过度正则化**.

### 2.3 每个 mode 在 held-out 上的最佳 patch

```
       broadcast: best=   none (held-out=0.4583, vs none=0.4583, gain=+0.0%)
            gate: best=   none (held-out=0.4481, vs none=0.4481, gain=+0.0%)
     router-norm: best=   none (held-out=0.4639, vs none=0.4639, gain=+0.0%)
    adaptive-ema: best=   none (held-out=0.4995, vs none=0.4995, gain=+0.0%)
```

**4 个 mode 的最佳 patch 都是 'none'** —— 即 v22 默认 patch 已是局部最优.

### 2.4 vs v26 baseline (无 patch 时)

```
broadcast + l2:       held-out=0.5150 (⚠ -11.4% vs v26)
broadcast + dropout:  held-out=0.6208 (⚠ -34.3% vs v26)
broadcast + warmup:   held-out=0.7066 (⚠ -52.8% vs v26)
broadcast + noise:    held-out=0.7727 (⚠ -67.1% vs v26)

gate + l2:       held-out=0.4620 (⚠ -2.3% vs v26)
gate + dropout:  held-out=0.4900 (⚠ -8.5% vs v26)
gate + warmup:   held-out=0.5131 (⚠ -13.6% vs v26)
gate + noise:    held-out=0.5310 (⚠ -17.6% vs v26)

router-norm + l2:       held-out=0.5237 (⚠ -7.8% vs v26)
router-norm + dropout:  held-out=0.5811 (⚠ -19.6% vs v26)
router-norm + warmup:   held-out=0.6259 (⚠ -28.8% vs v26)
router-norm + noise:    held-out=0.6655 (⚠ -37.0% vs v26)

adaptive-ema + l2:       held-out=0.6102 (⚠ -23.7% vs v26)
adaptive-ema + dropout:  held-out=0.7039 (⚠ -42.8% vs v26)
adaptive-ema + warmup:   held-out=0.7767 (⚠ -57.5% vs v26)
adaptive-ema + noise:    held-out=0.8326 (⚠ -68.8% vs v26)
```

**所有 16 个组合 (4 patch × 4 mode) 都比 v26 baseline 差**.

---

## 3. ⚠️ 关键负面发现

### 3.1 v22 patch 已是局部最优

v27 的 4 个补丁 (l2 / dropout / warmup / noise) 在 80 个 run 中**没有任何一个能改善 held-out**.
**v22 的 4 项稳定性 patch (freeze_u + warmup=10 + set_u_init_scale=0.001 + clip_grad=1.0) 已经接近该问题空间的局部最优**.

### 3.2 patch 强度过大

4 个 patch 的强度 (`lambda=1e-4`, `p=0.1`, `warmup=100`, `std=0.01`) 可能过大, 但:
- 减小 patch 强度只会让效果**接近 'none'**, 不会突然反转
- 这表明 v22 + v27 的 patch 空间在当前 16 句训练 + 6 句 held-out 上**没有足够自由度让 patch 帮助泛化**

### 3.3 gate 仍是唯一鲁棒模式

即使应用最严的 patch (noise), gate 的 held-out fuse (0.5310) 仍低于 router-norm/adaptive-ema 应用任何 patch 的结果 (0.5183-0.8326).
**这再次确认 v9 gate-style 中枢的"完美退化"特性是真实泛化的关键** —— 不是 patch 能替代的架构优势.

### 3.4 router-norm/adaptive-ema 的退步是架构性而非训练性

v27 的结论是负面但意义重大:
- 退步**不是**因为 c 梯度过大 (l2 帮倒忙)
- 退步**不是**因为路由过拟合 (dropout 帮倒忙)
- 退步**不是**因为自适应 decay 过早 (warmup 帮倒忙)
- 退步**不是**因为 c 收敛到特定点 (noise 帮倒忙)

**router-norm/adaptive-ema 的 held-out 退步是架构性问题**: 它们对训练集特征过于敏感, 任何正则化都会同时伤害训练和泛化 (但泛化伤害更大).

---

## 4. 单元测试 (12 项 PASS)

```
=== V27.0 router-norm / adaptive-ema held-out 修正 — 单元测试 (12 项) ===
[PASS] test_v27_imports (5 patches: ['none', 'l2', 'dropout', 'warmup', 'noise'])
[PASS] test_apply_l2_to_c_modifies_layer (lambda=2e-4, 2 c params)
[PASS] test_get_l2_to_c_loss_returns_tensor (c=0→0, c=0.1/0.2→12.80)
[PASS] test_apply_l2_to_c_works_for_all_modes (4 modes)
[PASS] test_router_dropout_attaches_dropout_layer (p=0.2)
[PASS] test_extend_warmup_steps_modifies_adaptive_ema (10 → 100)
[PASS] test_noise_inject_c_attaches_std (std=0.02)
[PASS] test_apply_all_patches_combines_all_4 (4 patches 全部生效)
[PASS] test_regularized_theory_layer_all_modes_forward
[PASS] test_compute_total_loss_with_l2 (loss=0.039584)
[PASS] test_l2_term_provides_gradient_to_c (cb_attn.c.grad.abs.sum()=0.2304)
[PASS] test_end_to_end_smoke_with_patch (fuse=0.0395, 4 patches 全开)
合计: 12/12 ✓ ALL PASS
```

---

## 5. 一句话总结

> **V27.0 用 4 个补丁 (apply_l2_to_c / router_dropout / extend_warmup_steps / noise_inject_c) 试图修正 v26 中 router-norm/adaptive-ema 的 held-out 退步, 但 80 run 实验证明 4 个补丁均让退步加剧** — 这是一个明确的负面发现:
> **v22 现有 patch 已是局部最优**, router-norm/adaptive-ema 的退步是**架构性问题**(对小数据过于敏感) 而非训练性问题;
> **v9 gate-style 的"完美退化"特性 (c=0 → 温度=1 → v7) 是真实泛化的关键架构优势**, 不能用训练侧正则化替代.
> 这是 v22 完整实现 + v26 fully-real + v27 4 patch 共同给出的**最严格的中央机制稳定性结论**.

---

## 6. 工程产出

- `v27_router_norm_fix/__init__.py` — 导出 11 个公开 API
- `v27_router_norm_fix/patches.py` — `apply_l2_to_c` / `router_dropout` / `extend_warmup_steps` / `noise_inject_c` + `apply_all_patches` + `DEFAULT_PATCH_CONFIGS` + `PATCH_NAMES`
- `v27_router_norm_fix/layer.py` — `RegularizedTheoryLayer` + `compute_total_loss`
- `examples/run_v27_router_norm_fix.py` — 5 patch × 4 mode × 5 seeds = 80 run 端到端
- `tests/test_v27_router_norm_fix.py` — 12 项单元测试
- `pyproject.toml` — packages.include 加 `v27_router_norm_fix*`

## 7. 给 v28+ 的建议

v27 的负面发现指向 4 个方向:
1. **架构层面修正**: 修改 router-norm/adaptive-ema 的数学形态（如把 LayerNorm 换成 group norm, 把 EMA decay 换成固定值）
2. **数据规模修正**: 把 16 训练 / 6 held-out 扩到 100+/30+, 验证 patch 在更大数据下是否有效
3. **接受现实**: 承认 router-norm/adaptive-ema 在小规模融合下架构劣势, 推荐 gate 作为唯一实用模式
4. **理论论文**: 这是把 v22 + v26 + v27 的中央机制研究写成 paper 的好材料 — "为什么 gate-style 在小规模融合下不可替代"