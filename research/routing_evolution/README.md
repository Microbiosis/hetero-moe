# 研究线 1: routing-evolution (路由机制演化)

**研究问题**: 异构底座之间的路由机制如何演化?

**父版本**: hetero_fusion v4.0/v6 baseline (矩形 P_m 投影)
**子版本**: 后续所有研究线的根

## 成员

| 版本 | 贡献 | 关键模块 |
|---|---|---|
| v7 | 跨架构族 Attn + FFN 双路由 | `attn_pool.AttnPool`, `fusion.CrossArchAttnFFNFusionLayer`, `trainer.V7Trainer` |
| v8 | 全局中枢三路线 (C-1 broadcast / C-2 协同 / C-3 层级) | `central.CentralWorkspace` (历史, 现已迁移), `fusion.CentralAugmentedFusionLayer`, `layer.HierarchicalCentralLayer` |
| v9 | gate-style 中枢修正 C (v8 C-1 失败的根因修复) | `layer.GateStyleWorkspace`, `workspace.make_workspace` |

## 演进脉络

```
v6 baseline (v4.0 规范)
   │
   ├─ v7: 双路由 (attn + ffn 独立选择, per-expert param group)
   │
   └─ v8: 加入中枢
       │
       └─ v9: C-1 broadcast 失败 → gate-style 修正 (温度 τ = 1 + 0.1·tanh(W_t·c))
              │
              ├─→ research/central_diagnostics/   (v19, v20 根因分析)
              └─→ research/central_stability/     (v22, v30, v31 稳定性修复)
```

## 关键结论

- v7: 双路由本身 +84% (相对 v6)
- v8.0 C-1 broadcast: **-71.9% (反向)** — 唯一挑战中枢 token
- v8.0 C-3 层级中枢: **+88.1%** — 中枢"必须是有容量、有结构的元认知模块"
- v9.0 gate-style: **+44.5%** (C-1 单独); **+88.1%** (C-3) — 架构层修复

## 负面发现 / 边界

- v8 C-1 broadcast 单独启用会反向增益 — 引出 v9 根因分析
- v8 C-1 + C-2 + C-3 全开: 增益 < 仅 C-3 — 全开不如单 C-3

## 测试

- test_v7_attention.py (8 PASS)
- test_v8_central.py (8 PASS)
- test_v9_gate.py (12 PASS)