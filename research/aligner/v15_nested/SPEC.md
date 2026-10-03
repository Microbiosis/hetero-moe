<!-- 权威副本. 本文件是 research/aligner/v15_nested/SPEC.md; docs/V15_0_MOM_NESTED.md 是指向本文件的薄指针. -->
<!-- 机器可读声明见同目录 MANIFEST.json; 指标快照见 results/metrics.json -->

# V15.0 — Mixture of Mixture (MoM) 嵌套对齐器

> **日期**: 2026-08-26
> **状态**: ✅ Verified (5/5 PASS)
> **复现命令**: `python3 examples/run_v15_mom_compare.py`
> **核心结论**: **`mixture < mom < attn` 部分成立** — 嵌套 (mom) 比 attn 改善但**不比 mixture 更优**, 表达能力到达瓶颈

---

## 1. 动机

v13 MixtureAligner 端到端优于 v10 CrossArchAttnAligner (fuse 0.34 → 0.26, 23%).
v14 验证"更复杂的对齐器更好"在 v10 → v13 阶梯上成立.

v15 验证这个阶梯**是否单调** — 把对齐器推到两级嵌套 (Mixture of Mixture), 看是否还能进一步提升.

## 2. 设计: MoMAligner

### 两级嵌套专家池

```
Level 1 (底座级):
    每底座贡献 n_inner 个原始专家 (Linear-ReLU-Linear)
    M × n_inner 个专家跨底座池 attn -> 共享表示 1

Level 2 (池级):
    共享表示 1 通过 n_outer 个高层专家
    n_outer 个高层专家跨池 attn -> 最终统一表示
```

### 与 v13 MixtureAligner 的区别

| 维度 | v13 MixtureAligner | v15 MoMAligner |
|---|---|---|
| 层数 | 1 层 | 2 层嵌套 |
| 专家数 | M × n (单层) | M × n_inner (内层) + n_outer (外层) |
| 总参数量 | O(M × n × D²) | O((M × n_inner + n_outer) × D²) |
| 抽象层级 | 跨底座 + 跨专家 | 跨底座 + 跨专家 + 跨"高层专家" |

## 3. 实验结果 (3 变体 × 5 seeds)

```
      mode |  fuse MSE |    vs attn
      attn |    0.3216 |      +0.0%
   mixture |    0.2525 |     +21.5%
       mom |    0.2645 |     +17.8%
```

每个 seed 的明细:
```
attn      seed 0..4:  0.3168, 0.2855, 0.2963, 0.3284, 0.3811
mixture   seed 0..4:  0.2566, 0.2497, 0.3072, 0.2240, 0.2249
mom       seed 0..4:  0.2291, 0.2324, 0.2753, 0.3097, 0.2758
```

## 4. 结论

### 4.1 mom 比 attn 改善, 但**比 mixture 略差**

- mom vs attn: **+17.8%** 改善
- mom vs mixture: -4.7% (退步)
- mixture vs attn: **+21.5%** 改善

**核心结论**: 嵌套对齐器**不比单层 mixture 更好**. 这可能是因为:
1. 两级 mean 把信号平滑, 嵌套反而损失信息
2. mixture 已经足够强, 嵌套边际收益递减
3. **表达能力阶梯不再单调** — mixture 已是当前最优架构

### 4.2 表达能力的天花板

```
v6   linear P_m        fuse 1.14  (基线)
v10  attn aligner      fuse 0.34  (-70%)
v13  mixture           fuse 0.26  (-77%)  ← 当前最强
v15  mom (嵌套)        fuse 0.26  (-77%)  ← 无进一步提升
```

**结论**: 对齐器的"表达能力阶梯"在 mixture 这一档到达瓶颈. 嵌套不带来增益.

### 4.3 这是 v15 的重要发现

v15 不是失败, 而是验证了**对齐器有天花板**. 这一发现避免后续工作:
- 不要再投入"更复杂对齐器"
- 应该转向**蒸馏** (v13 已验证) 或**更大规模** (A100) 等其他方向

## 5. 工程产出

- `v15_nested/__init__.py`
- `v15_nested/aligner.py` — `MoMAligner` (两级嵌套)
- `examples/run_v15_mom_compare.py` — 3 变体 × 5 seeds 对照
- `tests/test_v15_nested.py` — 8 项单元测试
- `pyproject.toml` — packages.include 加 `v15_nested*`

## 6. 一句话总结

> **V15 验证对齐器表达能力有天花板: MoM (嵌套) 比 attn 改善 17.8% 但**比 mixture 略差 4.7%**. 单层 mixture (v13) 已是当前最强架构, 嵌套不带来增益. 这是项目里**第二个负面发现** (第一个是 v12 组合无叠加), 共同指向一个结论: 当前路线已接近能力上限, 下一突破应在 A100 规模或蒸馏上.**