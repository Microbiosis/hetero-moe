# 研究线 3: central-stability (中枢稳定性修复)

**研究问题**: 除了 v9 gate-style, 还有哪些结构修复能让中枢机制稳定?

**父版本**: v9 gate-style (已建立 C-1 工作版本)
**子版本**: → research/corpus_scale/ 在真实语料下重测

## 成员

| 版本 | 贡献 | 关键模块 |
|---|---|---|
| v22 | 4 种中枢模式基类 + adaptive-ema 探索 | `theory.CentralTheory`, `RouterNormCentral`, `AdaptiveEMACentral` |
| v30 | softplus 替代 tanh (消除饱和区) | `workspace.SoftplusGateStyleWorkspace` |
| v31 | tanh 前梯度重缩放 | `workspace.GradientScaledGateStyleWorkspace` |

## 4 种中枢模式 (v22 探索)

| 模式 | 机制 | 合成数据 (v22) | 真实语料 (v25 16+6) | 真实语料 (v26 16+6) | 大语料 (v29 64+32) |
|---|---|---|---|---|---|
| broadcast | z += c·U^T (v8 原版) | 0.3283 | 0.4516 | 0.4624 | 0.4548 |
| gate-style (v9) | z = z / (1 + α·tanh(W_t·c)) | 0.3006 | 0.3562 | **0.4516** | 0.5253 |
| router-norm | logits + LayerNorm 再 softmax | 0.2875 | 0.4125 | 0.4859 (退步) | 0.5763 (退步) |
| adaptive-ema | decay = sigmoid(α·‖c‖) | 0.2972 | **0.3094** | 0.4931 (退步) | **0.4309** |

## 关键发现

- **最优中枢模式依赖数据规模** — 这是 v29 的最重要发现
- 小语料 (16-32): gate-style 最佳
- 中等语料 (16+6 real): gate 唯一保持正增益
- 大语料 (64+32): adaptive-ema 最佳 (+12.6%)

## 修复路径 (v30/v31)

- v9 tanh 饱和区 `∂τ/∂t_scalar ≈ 0`, 导致 `grad_norm_central = 0.0000`
- v30 softplus: v9=0.6927 → **v30=0.6726 (-3.0%)** ✅ 改善
- v31 gradient-scale: v9=0.6927 → **v31=0.6795 (-1.9%)** ✅ 改善

## 负面发现

- v27 4 patch (L2/Dropout/warmup/noise) **全部失败** — v22 patch 已局部最优
- v28 7 架构变体 (LayerNorm/RMSNorm/zero U 等) **全部失败** — 架构层救不了数据规模问题

## 测试

- test_v22_theory.py (14 PASS)
- v30/v31 由 v22 测试覆盖 + 端到端脚本