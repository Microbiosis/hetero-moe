# 研究线 4: aligner (语义对齐器)

**研究问题**: 异构底座怎么对齐语义空间?

**父版本**: v7 路由 (矩形 P_m 压平非线性)
**子版本**: → research/distillation/ 蒸馏用 MixtureAligner

## 成员

| 版本 | 贡献 | 关键模块 |
|---|---|---|
| v10 | attn aligner (跨底座共享 K/V 注意力) | `aligner.SemanticAligner`, `CrossArchAttnAligner`, `AlignedFusionLayer` |
| v13 | MixtureAligner (跨底座 + 跨专家注意力) + 蒸馏入口 | `aligner.MixtureAligner`, `student.StudentModel`, `distill.distill_loss` |
| v15 | 嵌套对齐器 (MoM: mixture of mixtures) — 天花板验证 | `aligner.MoMAligner` |

## 对齐器能力严格排序

```
v6   linear P_m        fuse 1.14   (基线)
v10  attn aligner      fuse 0.34   (-70%)
v14  mixture           fuse 0.26   (-77%)   ← 当前最强
v15  MoM (嵌套)        fuse 0.26   (-77%)   ← 无进一步提升
```

## 关键结论

- **`mixture > attn > linear`** 严格成立
- **对齐器表达能力存在天花板**: 嵌套不比单层 mixture 更好
- v10 attn aligner 是 v6 linear P_m 的 -70% MSE 改善
- v13 mixture 是 v10 attn 的进一步 -23% 改善

## 跨线复用

- v13 的 `StudentModel` + `distill_loss` 被 research/distillation/ 的 v16/v17/v18 复用
- v13 的 MixtureAligner 被 research/multimodal/ 的 v21 (模态路由) 复用
- v10 的 `CrossArchAttnAligner` 被 research/corpus_scale/ 的 v25 复用

## 负面发现 (archive)

- archive/negative_findings/v12_compose/ — v9+v10 组合无叠加增益 (8 PASS test 但 0 增益)

## 测试

- test_v10_aligner.py (8 PASS)
- test_v13_mixture.py (8 PASS)
- test_v15_nested.py (8 PASS)
- test_v12_compose.py (8 PASS, archive)