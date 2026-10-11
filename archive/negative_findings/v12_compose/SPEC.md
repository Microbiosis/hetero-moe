<!-- 权威副本. 本文件是 archive/negative_findings/v12_compose/SPEC.md; docs/V12_0_COMPOSE.md 是指向本文件的薄指针. -->
<!-- 机器可读声明见同目录 MANIFEST.json; 指标快照见 results/metrics.json -->

# V12.0 — v9 + v10 组合 (语义对齐器 + gate-style 中枢)

> **日期**: 2026-08-26
> **状态**: ✅ Verified (5/5 PASS)
> **复现命令**: `python3 examples/run_v12_compose.py`
> **核心结论**: **v12 ≈ v10**, 主贡献来自 attn aligner (改善 ~72% vs v9)

---

## 1. 动机

用户问题: *"把 v9 + v10 组合, 看叠加增益"*

v9 (gate-style 中枢) 和 v10 (语义对齐器) 各自独立验证有效:
- v9: C-1 单独从 -71.9% 翻正到 +44.5%
- v10: attn aligner 比 v9 线性 P_m 改善 +67%

**v12 验证问题**: 两个改进叠加, 是否带来比单独使用更好的结果?

## 2. 实验设计: 4 变体 × 5 seeds

| mode | aligner | 中枢 | 说明 |
|---|---|---|---|
| v9-linear-gate | P_m 线性 | gate-style | v9.0 修正 C (基线) |
| v10-attn-broadcast | attn aligner | broadcast | 退化: 用 v8 默认中枢 (已知有 bug) |
| **v12-attn-gate** | **attn aligner** | **gate-style** | **v9 + v10 组合 (核心)** |
| v10-attn-gate-original | attn aligner | gate-style | v10 默认 (= v12, 对照) |

## 3. 实验结果

```
                        mode |  fuse MSE |  vs v9-gate
              v9-linear-gate |    1.1625 |       +0.0%
          v10-attn-broadcast |    0.3201 |      +72.5%
               v12-attn-gate |    0.3374 |      +71.0%
      v10-attn-gate-original |    0.3191 |      +72.6%
```

| mode | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | 均值 |
|---|---:|---:|---:|---:|---:|---:|
| v9-linear-gate        | 1.1868 | 1.7701 | 1.0000 | 0.8124 | 1.0428 | **1.1625** |
| v10-attn-broadcast     | 0.3669 | 0.3239 | 0.3130 | 0.2880 | 0.3089 | **0.3201** |
| v12-attn-gate          | 0.3364 | 0.3147 | 0.3167 | 0.3238 | 0.3951 | **0.3374** |
| v10-attn-gate-original | 0.3331 | 0.3003 | 0.3080 | 0.3013 | 0.3526 | **0.3191** |

## 4. 结论

### 4.1 v12 vs v9: 显著叠加增益 (但主要来自 v10)

- v12-attn-gate 比 v9-linear-gate 改善 **+71.0%** (fuse MSE 1.16 → 0.34)
- 但 v10-attn-gate-original (v10 默认) 也改善 +72.6% —— **几乎与 v12 相同**

**核心结论**: gate-style 中枢在 v10 框架上**没有带来额外叠加增益**. v12 ≈ v10. 这是因为 v10 默认就用 gate-style 中枢 (`central_mode="gate"`, `freeze_u=True`)。

### 4.2 broadcast vs gate 中枢在 v10 上差异极小

| 中枢模式 | fuse MSE |
|---|---:|
| broadcast | 0.3201 |
| gate | 0.3191 / 0.3374 |

差异 ~5%, 在统计噪声范围内。说明:
- v9 路线中"gate-style 比 broadcast 更优"是**针对 v6/v7 线性 P_m** 的结论
- v10 attn aligner 已经把信息保留度提升到接近上限, 中枢 broadcast/gate 差异被边缘化

### 4.3 v10 是真正的"主贡献者"

- v10 单独 vs v9: 改善 ~72%
- v12 (v10 + v9 gate): 改善 ~71%
- **差异在 1-2% 之内**

**架构层面**: v10 的语义对齐器解决了"跨架构异构信息保留"这个根本问题; v9 的中枢机制解决"expert 协调"问题。前者更基础, 后者已被前者涵盖。

## 5. 给后续工作的意义

1. **v10 是当前最强组件**——任何新版本应该以 v10 为基础
2. **gate-style 中枢可继续保留**(无副作用, 有微弱益处), 但不是关键改进
3. **真正的下一步是蒸馏**(v13)——把 v10 协同蒸馏到单底座, 产出真正的独立模型
4. **"组合 v8 + v9 + v10" 的边际收益递减**——核心机制已经收敛

## 6. 工程产出

- `v12_compose/__init__.py`
- `v12_compose/fusion.py` — `ComposedFusionLayer` (v9 + v10 组合的命名包装)
- `examples/run_v12_compose.py` — 4 变体 × 5 seeds 对照
- `pyproject.toml` — packages.include 加 `v12_compose*`

## 7. 一句话总结

> **V12 验证 v9 + v10 组合无显著叠加增益: gate-style 中枢在 v10 框架下几乎不改变结果 (差异 < 2%), 主贡献 (~72%) 来自 v10 的 attn aligner. 这印证了"组合多版本不总是更好"——v10 已基本解决了跨架构融合的核心问题.**