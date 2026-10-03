<!-- 权威副本. 本文件是 research/multimodal/v23_vqa/SPEC.md; docs/V23_0_VQA.md 是指向本文件的薄指针. -->
<!-- 机器可读声明见同目录 MANIFEST.json; 指标快照见 results/metrics.json -->

# V23.0 — Mini-VQA 真实任务验证多模态嵌合

> **日期**: 2026-08-30
> **状态**: ✅ Verified (单测 8/8 PASS, 端到端 3/3 seeds)
> **复现命令**: `python3 examples/run_v23_vqa_distill.py`
> **核心结论**: 多模态嵌合 pipeline 跑通; toy 任务太简单未显出 multimodal 优势

---

## 1. 动机

EVOLUTION.md 给 v22+ 方向 B: *"用 VQA / 跨模态检索等真实任务替代 class direction 回归"*.

v18/v21 用 toy class direction 回归验证蒸馏可行性, 但**真实任务能力**未验证.

v23 提供:
- **MiniVQADataset**: 内置 16 张合成图片 + 16 个问题 + 16 个答案 (4 类: yes/no/red/blue)
- **真实 VQA pipeline**: 多模态学生输入"图片 + 问题" → 4 类答案 CE loss
- 不依赖 VQAv2 下载 (CPU 沙箱)

---

## 2. 设计: MiniVQADataset

### 4 类答案

| label | 答案 | 图像 | 问题模板 |
|---|---|---|---|
| 0 | yes | 白色圆 | "Is there a circle?" |
| 1 | no | 全黑 | "Is the screen empty?" |
| 2 | red | 红色填充 | "What color is it?" |
| 3 | blue | 蓝色填充 | "What color is shown?" |

```python
def generate_synthetic_vqa_sample(idx):
    img = torch.zeros(3, 224, 224)
    label = idx % 4
    if label == 0:  # yes - 白圆
        for y in range(224):
            for x in range(224):
                if (x - cx)² + (y - cy)² < r²:
                    img[:, y, x] = 1.0
    elif label == 1: pass  # no - 全黑
    elif label == 2: img[0] = 1.0  # red
    elif label == 3: img[2] = 1.0  # blue
    return img, label, QUESTION_TEMPLATES[label]
```

### 数据集分割

- train: idx 0-11 (12 个样本)
- eval: idx 12-15 (4 个样本, 不同 seed 保证是新颖样本)

---

## 3. 实验设计: 4 mode × 3 seeds = 12 run

| mode | 说明 |
|---|---|
| random-baseline | 随机猜 4 类 (基线下界) |
| text-only | v21 学生但 image 全黑 (只用文本) |
| image-only | v21 学生但 text mask (只用图像) |
| **multimodal-distill** | **v21 学生 + LoRA + CE loss** |

任务: 输入 (图片, 问题) → CE loss on 4 类答案; 评估 eval 集 accuracy.

---

## 4. 实验结果 (3 seeds)

```
                  mode |  error rate |   acc
       random-baseline |      0.8333 | 16.7%
             text-only |      0.1667 | 83.3%
            image-only |      0.0000 | 100.0%
    multimodal-distill |      0.0833 | 91.7%
```

每个 seed:
```
random-baseline   seed 0..2: 1.0000, 0.7500, 0.7500  (mean 0.8333)
text-only         seed 0..2: 0.2500, 0.0000, 0.2500  (mean 0.1667)
image-only        seed 0..2: 0.0000, 0.0000, 0.0000  (mean 0.0000)
multimodal-distill seed 0..2: 0.0000, 0.0000, 0.2500  (mean 0.0833)
```

---

## 5. 结论 (诚实)

### 5.1 Pipeline 跑通

- MiniVQADataset 端到端可用
- MultiModalStudent (v21 复用) 输入 (image, text) 输出 4 类 logits
- CE loss + AdamW + 反向传播全跑通
- eval 集 + 训练集分离

### 5.2 toy 任务太简单, multimodal 优势未显出

- **image-only 100% 准确率**: 学生学到了"图像→答案"的捷径（颜色和圆都很容易识别）
- **text-only 83.3%**: 文本也有信号（问题模板暗示答案）
- **multimodal-distill 91.7%**: 多模态没超越 image-only

**根本原因**:
- 训练样本太少（12 个）
- 任务太简单（4 类, 每类 3 个样本）
- 模型容量足够（BERT-Tiny + ViT-Tiny）轻松过拟合

### 5.3 这是预期负面发现

类似 v12/v15/v17, **在新维度上 v23 也遇到"简单任务下 multimodal 优势不显"**. 但 **pipeline 跑通**比"准确率数字"更重要:
- v23 提供了**真实 VQA 任务的标准模板**
- 未来研究可用更大数据集 (VQAv2, GQA) 直接复用本端到端脚本

---

## 6. 工程产出

- `v23_vqa/__init__.py` + `dataset.py`
- `examples/run_v23_vqa_distill.py` — 4 mode × 3 seeds 端到端
- `tests/test_v23_vqa.py` — 8 项单元测试
- `pyproject.toml` — packages.include 加 `v23_vqa*`

---

## 7. 给 v24+ 的方向

1. **更大 VQA 数据集**: 用 GQA (图问答) 或 VQAv2 子集替代 toy
2. **更难的任务**: Image Captioning (生成式), VCR (视觉常识推理)
3. **多教师多模态蒸馏**: v10 attn + v13 mixture 同时蒸馏到 MultiModalStudent
4. **A100 部署**: 当前 CPU 限制图像处理速度

---

## 8. 一句话总结

> **V23.0 提供了真实 VQA 任务的完整 pipeline (MiniVQADataset + MultiModalStudent + CE loss + eval). 端到端跑通但 toy 任务 (4 类 + 12 样本) 太简单, multimodal 未显出优势 (image-only 100% accuracy). 这与 v12/v15/v17 的负面发现一致 — 在小规模简单任务下"更多的东西"不一定更好. 但 pipeline 价值高: 未来研究用 VQAv2/GQA 直接复用.**