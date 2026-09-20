# 研究线 6: multimodal (多模态扩展)

**研究问题**: 多模态融合能否超越单模态?

**父版本**: research/distillation/v16_bert_student (教师为 v13 mixture)
**子版本**: 无 (v24 之后是真实 CLIP 教师)

## 成员

| 版本 | 贡献 | 关键模块 |
|---|---|---|
| v21 | BERT-text + ViT-image + 模态路由器 + LoRA (蒸馏自 v13 mixture 教师) | `student.MultiModalStudent`, `LoRAHiddenAdapter` |
| v23 | Mini-VQA 4 类 toy 任务 | `dataset.MiniVQADataset`, `generate_synthetic_vqa_sample` |
| v24 | CLIP 预训练教师 + 真实多模态数据 | `teacher.CLIPTeacher`, `corpus.MiniImageTextCorpus` |

## 关键数据

| 版本 | 任务 | 结果 | 结论 |
|---|---|---|---|
| v21 | 多模态蒸馏 | fuse 0.16, **比 v13 mixture 教师超 50.5%** | 多模态可蒸馏 |
| v23 | toy VQA 4 类 | text-only 83.3% / image-only 100.0% / multimodal 91.7% | toy 太简单 |
| v23b (archive) | 8 类陷阱 toy | multimodal 50% = 随机 | **toy 数据集根本不可行** |

## 当前最强多模态独立模型

**v21 + BERT + ViT (多模态)** — BERT-text + ViT-image + 模态路由器 + LoRA, 蒸馏后 fuse 0.16。

## 负面发现

- **v23b**: 8 类陷阱设计下所有模型 (含 multimodal) 仍 50% = 随机水平
  - 已移到 `archive/negative_findings/v23b_hard_vqa/` (从 v23 抽出)
  - 启示: toy 数据集根本不可行, 必须用真实数据 (v24)

## 测试

- test_v21_multimodal.py (8 PASS)
- test_v23_vqa.py (8 PASS)
- test_v23_vqa_hard.py (8 PASS, archive)
- test_v24_clip.py (8 PASS)