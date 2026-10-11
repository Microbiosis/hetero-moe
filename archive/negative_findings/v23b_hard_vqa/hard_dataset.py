"""v23b — HardVQADataset (8 类, 24 训练样本, 陷阱设计).

8 个类别 (label 0-7):
    0: yes_circle   白圆 + "Is there a circle?"
    1: no_circle    全黑 + "Is there a circle?"
    2: yes_square   白方块 + "Is there a square?"
    3: no_square    全黑 + "Is there a square?"
    4: red_circle   红圆 + "What color is it?"        (image 关键)
    5: blue_circle  蓝圆 + "What color is it?"        (image 关键)
    6: red_square   红方块 + "How many sides does it have?" (trap: 答 4 不需颜色)
    7: blue_square  蓝方块 + "How many sides does it have?" (trap)

关键设计:
    - label 4-5 用 "What color?" 问题, 只有 image 能区分
    - label 6-7 用 "How many sides?" 问题, image 颜色不影响答案 (trap)
    - 学生必须学会"看问题决定看什么模态"才能优于单模态

train: 24 个样本 (label 0-7 各 3 个)
eval:  8 个样本 (label 0-7 各 1 个, train/eval 完全分离)
"""
from __future__ import annotations

from typing import List, Tuple

import torch


# 8 类答案
HARD_ANSWER_LABELS = [
    "yes_circle", "no_circle", "yes_square", "no_square",
    "red_circle", "blue_circle", "red_square", "blue_square",
]

# 8 类对应的问题模板
HARD_QUESTION_TEMPLATES = {
    0: "Is there a circle?",
    1: "Is there a circle?",
    2: "Is there a square?",
    3: "Is there a square?",
    4: "What color is it?",
    5: "What color is it?",
    6: "How many sides does it have?",
    7: "How many sides does it have?",
}


def generate_hard_vqa_sample(idx: int, image_size: int = 224) -> Tuple[torch.Tensor, int, str]:
    """生成第 idx 个 hard VQA 样本.

    Returns:
        image: [3, image_size, image_size] tensor
        label: int (0-7)
        question: str
    """
    import math
    img = torch.zeros(3, image_size, image_size)
    label = idx % 8

    if label in (0, 4, 5):  # circle 类
        # 画圆
        cy = image_size // 2
        cx = image_size // 2
        r = image_size // 6
        for y in range(image_size):
            for x in range(image_size):
                if (x - cx) ** 2 + (y - cy) ** 2 < r ** 2:
                    if label == 4:    # red_circle
                        img[0, y, x] = 1.0  # R 通道
                    elif label == 5:  # blue_circle
                        img[2, y, x] = 1.0  # B 通道
                    else:                # yes_circle (label=0)
                        img[:, y, x] = 1.0  # 白
    elif label in (2, 6, 7):  # square 类
        # 画方块
        s = image_size // 4
        cx = image_size // 2
        cy = image_size // 2
        for y in range(cy - s, cy + s):
            for x in range(cx - s, cx + s):
                if 0 <= y < image_size and 0 <= x < image_size:
                    if label == 6:    # red_square
                        img[0, y, x] = 1.0  # R 通道
                    elif label == 7:  # blue_square
                        img[2, y, x] = 1.0  # B 通道
                    else:                # yes_square (label=2)
                        img[:, y, x] = 1.0  # 白
    # label 1, 3 (no_circle / no_square): 全黑

    return img, label, HARD_QUESTION_TEMPLATES[label]


class HardVQADataset:
    """8 类 24 样本 hard VQA 数据集.

    train: 24 个样本 (每类 3 个, idx 0-7 各 3 次)
    eval:  8 个样本 (每类 1 个, idx 8-15)
    """

    def __init__(self, image_size: int = 224, train_per_class: int = 3, seed: int = 0):
        self.image_size = image_size
        self.train_per_class = train_per_class
        # train: idx 0-23 (8 类 × 3)
        # 不同 idx 通过 idx % 8 取类别, idx // 8 取样本编号 (影响位置/随机种子)
        self.train_samples = self._generate(8 * train_per_class, seed, is_train=True)
        # eval: idx 24-31, 不同 seed 保证是不同位置
        self.eval_samples = self._generate(8, seed + 10000, is_train=False)

    def _generate(self, n: int, seed: int, is_train: bool) -> List[dict]:
        torch.manual_seed(seed)
        out = []
        for i in range(n):
            img, label, question = generate_hard_vqa_sample(i, self.image_size)
            # 用 idx 序号影响位置 (而非完全固定)
            # 不同 idx 圆/方块位置会不同
            if is_train:
                # train: idx 0-23, label = idx % 8
                img_label = i % 8
            else:
                # eval: idx 0-7, label = idx % 8
                img_label = i % 8
            out.append({
                "image": img,
                "label": img_label,
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
