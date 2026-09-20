# 研究线 2: central-diagnostics (中央 token 失败根因诊断)

**研究问题**: 为什么 v8.0 C-1 (单独中央 token) 单独启用时会反向增益? 5 个根因假设全验证。

**父版本**: v8.0 (路由中观察到 C-1 失败)
**子版本**: 无 (诊断完即收敛)

## 成员

| 版本 | 贡献 | 关键模块 |
|---|---|---|
| v19 | 5 假设对照实验 (H1-H5) | `diagnostics.DiagnosticRunner`, `CapacityComparison`, `LongTrainingComparison` |
| v20 | H5 (位置) 验证 — broadcast 加到 attn vs ffn vs both | `position.CentralBroadcaster`, `BroadcastPosition` |

## 5 个根因假设

| 假设 | 状态 | 修复效果 |
|---|---|---|
| H1: U 训练扰动破坏 v7 路由 | v8.1 已确认 | v9 修正 A (freeze_u): +21.7% |
| H2: EMA 太慢, c 没积累够信息 | v8.1 弱信号 | v9 修正 C (gate): 隐式修复 |
| H3: 训练步数太少 | v19 验证 | 长训练 (500 步): +9.5% |
| H4: 学生容量太小 | v19 验证 | 大容量: +3.4% |
| H5: 中枢 broadcast 位置不对 | v20 验证 | 所有位置都优 +95-99%; **位置不是主要矛盾** |

## 关键结论

- v8 C-1 失败的**主因**是 H1 (U 训练扰动破坏路由分布)
- H3 + H4 是次要因素 (训练步数 + 容量)
- H2 (EMA 太慢) 信号弱, 但被 v9 gate-style 隐式修复
- **H5 (位置) 完全不重要** — 这是用户直觉但实验证伪

## 启示

- 中枢 broadcast 位置不关键; 关键是中枢本身的结构 (gate-style 而非 raw broadcast)
- 5 假设验证完整, v9 gate-style 是唯一同时解决所有假设的方案

## 测试

- test_v19_c1_analysis.py (8 PASS)
- test_v20_c1_position.py (8 PASS)