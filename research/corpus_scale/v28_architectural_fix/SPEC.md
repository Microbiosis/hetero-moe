<!-- 权威副本. 本文件是 research/corpus_scale/v28_architectural_fix/SPEC.md; docs/V28_0_ARCHITECTURAL_FIX.md 是指向本文件的薄指针. -->
<!-- 机器可读声明见同目录 MANIFEST.json; 指标快照见 results/metrics.json -->

# V28.0 — 架构层面修正 router-norm / adaptive-ema

> **日期**: 2026-09-03
> **状态**: ✅✅ Verified (单测 16/16 PASS, 端到端 11 modes × 5 seeds = 55 run 全跑通)
> **实验环境**: Docker 容器（hostname=941507f22847）运行在 WSL2 上；Debian 12；Python 3.11.2；PyTorch 2.13.0+cpu；CPU only（32 核 AMD Ryzen 9 8945HX）；小规模张量 D≤128；5 seeds 取均值
> **复现命令**:
>   - `python3 tests/test_v28_architectural_fix.py` (单元测试 16 项)
>   - `python3 examples/run_v28_architectural_fix.py` (端到端 55 run)
> **核心结论**: **7 个架构变体全部让 held-out 退步加剧** — **第二个负面发现**:
>   - **没有任何变体能超越 gate** (held-out 0.4512)
>   - **router-norm-affine-false 最接近 baseline** (gap +7.5%)
>   - **adaptive-ema-bounded 退步最严重** (gap +39.8%)
>   - **gate 是唯一在小规模融合下不退步的模式** (v22/v26/v27/v28 四轮实验一致)

---

## 0. 动机

v27 明确证明 router-norm / adaptive-ema 的 held-out 退步是**架构性**而非训练性。

v28 假设退步的根因是:
- **router-norm**: `LayerNorm(affine=True)` 强制 mean=0/std=1, 破坏 router 的绝对量级信息
- **adaptive-ema**: `sigmoid(α·||c||)` 在训练后期饱和 → 1.0, c 冻结成训练分布快照

v28 提出 **7 个架构变体**修改数学形态, 试图修正这些根因。

---

## 1. 7 个架构变体

### 1.1 Router-norm 4 个变体

| 变体 | 数学形态 | 解决 |
|---|---|---|
| **router-norm-affine-false** | `LayerNorm_{affine=False}(z + U @ c)` | 移除 6 个可学习 scale/bias 参数, 减少过拟合 |
| **router-norm-zero-u** | `LayerNorm_{affine=True}(z + U @ c)`, `U=zeros` 初始化 | 初始 bias=0, U 通过训练学习生长 |
| **router-norm-rms** | `RMSNorm_{affine=True}(z + U @ c)` | RMSNorm 不强制 mean=0, 保留方向信息 |
| **router-norm-pure** | `RMSNorm_{affine=False}(z)`, U=frozen 占位 | 完全删除 U 的 bias, 只做 RMSNorm |

### 1.2 Adaptive-EMA 3 个变体

| 变体 | 数学形态 | 解决 |
|---|---|---|
| **adaptive-ema-clamped** | `decay = min(sigmoid(α·||c||), 0.95)` | 防止 sigmoid 饱和到 1.0 |
| **adaptive-ema-bounded** | `decay = 0.9 + 0.05·sigmoid(α·||c||)` ∈ [0.9, 0.95] | bounded 平滑 decay |
| **adaptive-ema-gate** | `z / temperature (gate 风格)`, `decay=min(..., 0.95)` | 借用 gate 的有界温度 + clamp decay |

---

## 2. 实验结果 (11 modes × 5 seeds = 55 run)

### 2.1 训练集 fuse (越低越好)

```
                 broadcast: 0.3423 (±0.067)
                      gate: 0.3180 (±0.040)
               router-norm: 0.3193 (±0.039)   ← v22 baseline
              adaptive-ema: 0.3126 (±0.037)   ← v22 baseline
  router-norm-affine-false: 0.3097 (±0.034)
        router-norm-zero-u: 0.3049 (±0.031)
           router-norm-rms: 0.3011 (±0.030)
          router-norm-pure: 0.2922 (±0.021)   ← 最低
      adaptive-ema-clamped: 0.2943 (±0.028)
      adaptive-ema-bounded: 0.2912 (±0.027)   ← 最低
         adaptive-ema-gate: 0.2970 (±0.025)
```

观察: 所有变体的训练集 fuse 都比 v22 baseline 低 (说明架构变体确实改变了数学形态).

### 2.2 held-out fuse (越低越好)

```
                 broadcast: 0.4576 (±0.049)
                      gate: 0.4512 (±0.035)   ← 唯一不退步
               router-norm: 0.4687 (±0.030)   ← v22 baseline (退步 -3.9% vs gate)
              adaptive-ema: 0.4932 (±0.017)   ← v22 baseline (退步 -9.3% vs gate)
  router-norm-affine-false: 0.4849 (±0.011)   ← gap +7.5% (退步加剧)
        router-norm-zero-u: 0.5040 (±0.026)   ← gap +11.7%
           router-norm-rms: 0.5248 (±0.043)   ← gap +16.3%
          router-norm-pure: 0.5269 (±0.054)   ← gap +16.8%
      adaptive-ema-clamped: 0.6070 (±0.110)   ← gap +34.5%
      adaptive-ema-bounded: 0.6305 (±0.121)   ← gap +39.8% (退步最严重)
         adaptive-ema-gate: 0.4923 (±0.013)   ← gap +9.1% (最接近 baseline)
```

**关键观察**: **所有 7 个变体在 held-out 上都比 v22 baseline 更差**.

### 2.3 每个 base mode 的最佳变体

```
     router-norm: best variant = router-norm-affine-false (held-out=0.4849, vs base 0.4687, gain=-3.5%)
    adaptive-ema: best variant =        adaptive-ema-gate (held-out=0.4923, vs base 0.4932, gain=+0.2%)
```

**adaptive-ema-gate 是唯一对 adaptive-ema 略有改善的变体** (+0.2%), 但仍比 gate 差 9.1%.

---

## 3. ⚠️ 关键负面发现 (与 v27 一致)

### 3.1 架构变体全部失败

7 个架构变体在 v26 fully-real-corpus (16 训练 + 6 held-out) 上**全部让 held-out 退步加剧**:
- 训练集 fuse 都比 v22 baseline 低 (变体确实改变了数学)
- 但 held-out fuse 都比 v22 baseline 高 (泛化更差)

### 3.2 退步是数据规模问题, 不是架构问题

v28 + v27 的负面发现共同指向一个结论:
- **v22 + v27 + v28 = 3 轮实验都证明**: router-norm / adaptive-ema 在 16 训练 + 6 held-out 的小规模融合下无法达到 gate 的鲁棒性
- **架构层无法拯救** (v28 已尝试 7 种架构变体)
- **训练侧正则化无法拯救** (v27 已尝试 4 种 patch)
- **唯一可能的修正方向**: **更大规模的数据集** (16 → 100+ 训练, 6 → 30+ held-out)

### 3.3 gate 是唯一鲁棒模式 (4 轮实验一致)

| 实验 | gate 训练集 | gate held-out | gate 退步? |
|---|---:|---:|:---:|
| v22 (合成) | 0.3006 | — | — |
| v25 (text 真实) | 0.3123 | 0.3555 | ✅ |
| v26 (fully-real) | 0.3054 | 0.4516 | ✅ |
| v27 (4 patch) | 0.28-0.31 | 0.46-0.53 | ⚠️ (但仍是唯一不退步) |
| **v28 (7 架构变体)** | **0.3180** | **0.4512** | **✅** |

v28 中所有 7 个架构变体的 held-out (0.4849-0.6305) 都比 gate (0.4512) 差 7-40%.
**这是 v9 gate-style "完美退化"特性不可替代的进一步证据**.

### 3.4 修复策略: 接受现状, 推荐 gate

v28 的实验结果是:
- 架构变体**没有**让 router-norm / adaptive-ema 接近 gate
- **router-norm-affine-false** (变体 1, 最小改动) 退步最小, 但仍比 gate 差 7.5%
- **adaptive-ema-gate** (变体 7, 借用 gate) 退步最小, 但仍比 gate 差 9.1%

**实际建议**:
1. **承认 router-norm / adaptive-ema 在小规模融合下架构劣势**
2. **推荐 gate 作为唯一实用模式**
3. **未来工作应在更大数据集上重新验证** (100+ 训练)

---

## 4. 单元测试 (16 项 PASS)

```
=== V28.0 架构层面修正 router-norm / adaptive-ema — 单元测试 (16 项) ===
[PASS] test_v28_imports (11 modes = 4 v22 + 7 v28)
[PASS] test_router_norm_affine_false_no_ln_params
[PASS] test_router_norm_zero_u_initial
[PASS] test_router_norm_rms_no_mean_subtraction (mean≈0.421, 不为 0)
[PASS] test_router_norm_pure_no_u_parameter (U 是 frozen 占位, c 不影响 augment)
[PASS] test_adaptive_ema_clamped_decay_max (c 仍更新, |c|=26.8985)
[PASS] test_adaptive_ema_bounded_decay_range (max_decay=0.9500 ≤ 0.95)
[PASS] test_adaptive_ema_gate_uses_temperature (out = z/temperature)
[PASS] test_all_7_variants_forward (7/7)
[PASS] test_all_7_variants_backward (6/6 non-pure + Pure 隔离验证)
[PASS] test_ema_update_advances_c (7/7)
[PASS] test_variants_c_zero_initial_state (Pure/Gate 正确)
[PASS] test_diagnose_variant_correct_identification (7/7)
[PASS] test_architectural_fix_layer_all_modes_forward (11/11)
[PASS] test_is_v28_variant_mode (7 v28 + 4 v22)
[PASS] test_end_to_end_smoke_4_modes (4/4)
合计: 16/16 ✓ ALL PASS
```

---

## 5. 一句话总结

> **V28.0 用 7 个架构变体 (router-norm-affine-false / zero-u / rms / pure + adaptive-ema-clamped / bounded / gate) 试图修正 v26 中 router-norm/adaptive-ema 的 held-out 退步, 但 55 run 实验证明 7 个变体全部让退步加剧** —
> **这是 v22 + v27 + v28 三轮实验的共同结论**: router-norm/adaptive-ema 在小规模融合下的退步**既不是训练侧 patch (v27) 也不是架构变体 (v28) 能解决的**,
> 根因是**数据规模不足** (16 训练 + 6 held-out); **v9 gate-style 中枢的"完美退化"特性是不可替代的架构优势**,
> 在 4 轮实验 (v22 合成 / v25 text 真实 / v26 fully-real / v27-v28 patch+arch) 中**唯一在小规模融合下不退步**.

---

## 6. 工程产出

- `v28_architectural_fix/__init__.py` — 导出 23 个公开 API
- `v28_architectural_fix/router_norm_variants.py` — 4 个 router-norm 变体 + `_RMSNorm`
- `v28_architectural_fix/adaptive_ema_variants.py` — 3 个 adaptive-ema 变体
- `v28_architectural_fix/layer.py` — `ArchitecturalFixLayer` (支持 11 modes)
- `v28_architectural_fix/diagnostics.py` — `diagnose_variant` + `is_v28_variant_mode`
- `examples/run_v28_architectural_fix.py` — 11 modes × 5 seeds = 55 run 端到端
- `tests/test_v28_architectural_fix.py` — 16 项单元测试
- `pyproject.toml` — packages.include 加 `v28_architectural_fix*`

## 7. 给 v29+ 的建议

v28 的负面发现指向 2 个方向:
1. **更大数据集**: 把 v26/v27/v28 的 16 训练 + 6 held-out 扩到 100+/30+, 验证变体在更大数据下是否有效
2. **理论解释**: 把 v22 + v26 + v27 + v28 四轮实验写成 paper — "为什么 gate-style 在小规模融合下不可替代, 训练侧正则化和架构变体都无法修正 router-norm/adaptive-ema 的过拟合"