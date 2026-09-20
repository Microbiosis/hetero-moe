# 研究线 5: distillation (蒸馏到独立模型)

**研究问题**: 能不能从多底座协同中蒸馏出真正独立的单底座?

**父版本**: research/aligner/v13_mixture (教师)
**子版本**: → research/multimodal/ (v21 BERT+ViT 多模态学生)

## 成员

| 版本 | 贡献 | 关键模块 |
|---|---|---|
| v16 | 真实预训练学生 (TinyBERT + LoRA, 85,632 参数 0.60%) | `student.BERTStudent`, `student.LoRAAdapter` |
| v17 | 多教师蒸馏 (2/3/3-weighted) | `child.MultiTeacherDistiller` |
| v18 | 真实 Wikipedia 语料蒸馏 | `corpus.MiniCorpus`, `cycle_batches` |

## 关键数据

| 版本 | 输入 | fuse | vs 教师 |
|---|---|---:|---:|
| v13 (教师) | 多底座 mixture | 0.330 | — |
| v16 (学生) | 真实 TinyBERT + LoRA | **0.135** | **超 +47.5%** |
| v18 (学生) | 真实 Wikipedia 蒸馏 | 0.157 | 修正 v16 过拟合假象 |
| v17 (多教师) | 2/3 教师加权 | 退步 | -3.4% ~ -16.7% |

## 关键结论

- **单教师蒸馏可行且超教师** — v16 真实预训练 +47.5%
- **多教师蒸馏反而退步** — 蒸馏组合有天花板
- **真实语料下蒸馏仍然可行** — v18 修正 v16 过拟合假象

## 当前最强单底座独立模型

**v16 + BERT (真实预训练)** — 1 个 TinyBERT + 85,632 LoRA 参数 (0.60%), 蒸馏后 fuse 0.135, 推理无需任何其他底座。

## 跨线复用

- v18 的 `MiniCorpus` 被 research/corpus_scale/ 的 v25/v26 复用 (跨版本集成)
- v13 的 `StudentModel` 是 v16 的父类

## 测试

- test_v16_bert.py (8 PASS)
- test_v17_multi.py (8 PASS)
- test_v18_real_corpus.py (8 PASS)