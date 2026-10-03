# V14.0 — MixtureAligner vs CrossArchAttnAligner 端到端对照

> **日期**: 2026-08-26
> **状态**: ✅ Verified (5/5 PASS)
> **复现命令**: `python3 examples/run_v14_aligner_compare.py`
> **核心结论**: **`mixture < attn < linear` 严格成立**, 更复杂的对齐器更好

---

## 1. 动机

v13 提供了 MixtureAligner (跨底座 + 跨专家注意力), 但**没在端到端跑过**.
v14 真正端到端对照 3 种 aligner, 验证"更复杂的对齐器是否真的更好".

## 2. 设计: 3 种 aligner 端到端对照

| aligner | 来源 | 设计 |
|---|---|---|
| linear | v9 P_m 矩形投影 | 每底座独立线性投影 |
| attn | v10 CrossArchAttnAligner | 跨底座共享 K/V, 各底座独立 Q |
| **mixture** | **v13 MixtureAligner** | **跨底座 + 跨专家 (M × N 个专家), 跨专家注意力** |

## 3. 实验结果

```
      mode |  fuse MSE |   vs linear
    linear |    1.1409 |       +0.0%
      attn |    0.3376 |      +70.4%
   mixture |    0.2575 |      +77.4%
```

每个 seed 的明细:
```
linear    seed 0..4:  0.9323, 1.5723, 0.7832, 1.5840, 0.8325  (mean 1.1409)
attn      seed 0..4:  0.3205, 0.3251, 0.3278, 0.3151, 0.3995  (mean 0.3376)
mixture   seed 0..4:  0.2501, 0.2529, 0.3064, 0.2234, 0.2546  (mean 0.2575)
```

## 4. 结论

### 4.1 `mixture < attn < linear` 严格成立

- **mixture 比 attn 改善 23.7%** (0.34 → 0.26)
- **mixture 比 linear 改善 77.4%** (1.14 → 0.26)
- **attn 比 linear 改善 70.4%** (1.14 → 0.34)

**核心结论**: 更复杂的对齐器 (跨底座 + 跨专家) 持续优于简化版本. 表达能力阶梯:
1. linear (无注意力)
2. attn (跨底座共享 K/V, 表达力中)
3. mixture (跨底座 + 跨专家, 表达力强)

### 4.2 MixtureAligner 的额外成本

- **参数量**: mixture 比 attn 多 M × N 个专家参数 (本次: 3 × 4 = 12 个专家)
- **计算量**: 跨专家池注意力 O(M²·N²), attn 是 O(M²)
- **本次实验**: CPU 上 mixture 比 attn 慢约 3-4 倍, 但 fuse MSE 改善 24%

**性价比**: 如果推理时延不敏感, mixture 是当前最强选择; 如果时延敏感, attn 是更好的折中.

### 4.3 v13 的 MixtureAligner 不再"纸上谈兵"

v13 引入了 MixtureAligner 但只在 unit test 跑过. v14 验证了**它在端到端真实任务上有效**, 比 v10 的 CrossArchAttnAligner 还好 24%.

## 5. 与 v6-v13 的演进关系

```
v6   linear P_m       → 1.14  (基线)
v10  attn aligner     → 0.34  (-70%)
v13  mixture aligner  → 0.26  (-77%)  ← v14 验证
v13+ student distill → 0.32  (单底座 + LoRA)
```

**当前最强**: v13 MixtureAligner (fuse 0.26)
**当前最实用**: v13 学生蒸馏 (单底座可独立部署, fuse 0.32)

## 6. 工程产出

- `examples/run_v14_aligner_compare.py` — 3 变体 × 5 seeds 对照
- 复用现有子包 (v9/v10/v13), 无新模块

## 7. 给 v15 的方向

1. **更复杂的对齐器**: Mixture of Mixture (跨 Mixture aligner + 跨底座)
2. **Mixture vs Mean + 蒸馏**: 把 mixture aligner 当教师, 看学生蒸馏能否再优
3. **A100 部署**: CPU 沙箱限制了 mixture 的规模验证

## 8. 一句话总结

> **V14 验证 v13 的 MixtureAligner 端到端有效: 比 v10 CrossArchAttnAligner 再改善 23.7% (fuse 0.34 → 0.26). `mixture < attn < linear` 严格成立, 更复杂的对齐器持续优于简化版本. 这是"表达能力阶梯"在跨架构融合上的清晰验证.**