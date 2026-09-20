"""v23.0 — MiniVQADataset (内置合成 VQA 数据集).

设计:
    16 个样本, 4 个类别:
        0: yes  (有/是)
        1: no   (无/不是)
        2: red  (颜色)
        3: blue (颜色)

每个样本包含:
    image: [3, 224, 224] tensor (合成图像)
    question: str (固定 4 种模板之一)
    answer_label: int (0-3)
    tokenized_question: [max_length] input_ids (用 BERT tokenizer)

合成图像规则:
    yes/no 类别: 随机位置画一个白色圆, 是 -> yes, 不是 -> no
    red 类别: 整张图片填充红色
    blue 类别: 整张图片填充蓝色
"""
from __future__ import annotations

from typing import List, Tuple

import torch


# 4 个问题模板
QUESTION_TEMPLATES = {
    0: "Is there a circle?",      # yes/no
    1: "Is the screen empty?",     # yes/no
    2: "What color is it?",         # color (red/blue)
    3: "What color is shown?",      # color (red/blue)
}

ANSWER_LABELS = ["yes", "no", "red", "blue"]


def generate_synthetic_vqa_sample(idx: int, image_size: int = 224) -> Tuple[torch.Tensor, int, str]:
    """生成第 idx 个合成 VQA 样本.

    Returns:
        image: [3, image_size, image_size] tensor
        answer_label: int (0-3)
        question: str (4 种模板之一)
    """
    img = torch.zeros(3, image_size, image_size)
    label = idx % 4
    if label == 0:  # yes (有圆)
        # 在随机位置画一个白色圆
        cy = torch.randint(20, image_size - 20, (1,)).item()
        cx = torch.randint(20, image_size - 20, (1,)).item()
        r = 15
        for y in range(image_size):
            for x in range(image_size):
                if (x - cx) ** 2 + (y - cy) ** 2 < r ** 2:
                    img[:, y, x] = 1.0
    elif label == 1:  # no (空屏)
        pass  # 全黑
    elif label == 2:  # red
        img[0] = 1.0  # R 通道
    elif label == 3:  # blue
        img[2] = 1.0  # B 通道
    return img, label, QUESTION_TEMPLATES[label]


class MiniVQADataset:
    """内置 16 个样本的 mini-VQA 数据集.

    split 规则:
        train: idx 0-11 (12 个样本)
        eval:  idx 12-15 (4 个样本, 不同 idx 保证是新颖样本)
    """

    def __init__(self, num_train: int = 12, image_size: int = 224, seed: int = 0):
        self.num_train = num_train
        self.image_size = image_size
        self.train_samples = self._generate(num_train, seed)
        self.eval_samples = self._generate(4, seed + 1000)

    def _generate(self, n: int, seed: int) -> List[Tuple]:
        torch.manual_seed(seed)
        out = []
        for i in range(n):
            img, label, question = generate_synthetic_vqa_sample(i, self.image_size)
            out.append({
                "image": img,
                "label": label,
                "question": question,
            })
        return out

    def __len__(self):
        return len(self.train_samples)

    def __getitem__(self, idx):
        s = self.train_samples[idx]
        return {
            "image": s["image"],
            "label": s["label"],
            "question": s["question"],
        }

    def all_train(self):
        return list(self.train_samples)

    def all_eval(self):
        return list(self.eval_samples)