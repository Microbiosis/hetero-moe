#!/usr/bin/env python3
"""exp73b — 直接验证 THINGS-EEG 的「测试概念 vs 训练概念」是否重叠。

## 为什么需要这个

exp73 做的是 **200 选 1 图像解码**。一个必须先回答的前提问题：

> 200 张测试图的**概念**，是否出现在 1654 个训练概念里？

- 若**重叠**：解码器可能靠"同概念、同语义"取巧，数字是虚高的。
- 若**不相交（concept-disjoint）**：数字就是标准协议下的值，可与文献比较。

exp73 内置了一个**免标签**的 ViT 特征近邻探针来回答这个问题，
但那是**间接**证据。这个脚本给出**直接**证据：从桶内图像清单取概念名。

## 证据来源（桶内，无需下载 .pt 标签）

    hf buckets ls -R hf://buckets/illusion-brain/training-data/eeg/things-eeg/Image_set/training_images/
    hf buckets ls -R hf://buckets/illusion-brain/training-data/eeg/things-eeg/Image_set/test_images/

结构（实测）：

    training_images/<NNNNN_concept>/<concept>_NN{b,s}.jpg   → 1654 概念 × 10 图
    test_images/<NNNNN_concept>/<concept>_NN{b,s}.jpg         →  200 概念 ×  1 图

行序即特征行序：ViT-H-14_features_train.pt 的第 i 行 ↔ 训练清单第 i 行。

## 用法

    hf buckets ls -R <train_dir> > train_images.txt
    hf buckets ls -R <test_dir>  > test_images.txt
    python exp73b_concept_overlap.py train_images.txt test_images.txt
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PAT = re.compile(r"Image_set/(?:training_images|test_images)/(\d+)_(.+?)/")


def parse(path):
    ids, names = [], []
    unmatched = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = PAT.search(line)
            if m:
                ids.append(int(m.group(1)))
                names.append(m.group(2))
            elif line.strip() and not line.startswith(("Hint:", "---", "\x1b")):
                unmatched.append(line.strip()[:110])
    return ids, names, unmatched


def main(train_txt, test_txt):
    tr_ids, tr_names, tr_un = parse(train_txt)
    te_ids, te_names, te_un = parse(test_txt)

    print(f"训练行 {len(tr_names)}  测试行 {len(te_names)}")
    if tr_un:
        print(f"  ! 未匹配的训练行 {len(tr_un)}，样例: {tr_un[:2]}")
    if te_un:
        print(f"  ! 未匹配的测试行 {len(te_un)}，样例: {te_un[:2]}")

    groups = []
    consistent = True
    for i in range(0, len(tr_names), 10):
        g = tr_names[i:i + 10]
        if len(set(g)) != 1:
            consistent = False
            g = [f"<不一致:{sorted(set(g))}>"]
        groups.append(g[0])
    print(f"训练概念组 {len(groups)}（应为 1654），每组 10 张且组内同名: {consistent}")
    print(f"训练概念字母序: {groups == sorted(groups)}")
    print(f"测试概念 {len(te_names)}，互不相同: {len(set(te_names)) == len(te_names)}")
    print(f"测试概念字母序: {te_names == sorted(te_names)}")

    overlap = sorted(set(te_names) & set(groups))
    print("\n" + "=" * 72)
    print(f"概念重叠：{len(overlap)} / {len(te_names)}")
    if overlap:
        print("  重叠样例:", overlap[:10])
        print("  ⇒ 测试概念**部分**出现在训练概念里 ⇒ 解码数字可能含同概念泄漏")
    else:
        print("  ⇒ 测试概念与训练概念**完全不相交**（concept-disjoint）")
        print("  ⇒ exp73 的 200 选 1 数字是**新概念**泛化，不是同概念取巧")
    print("=" * 72)

    print(f"训练概念 id 范围: {min(tr_ids):05d}–{max(tr_ids):05d}（{len(set(tr_ids))} 个）")
    print(f"测试概念 id 范围: {min(te_ids):05d}–{max(te_ids):05d}（{len(set(te_ids))} 个）")
    print(f"id 区间有交集: {bool(set(tr_ids) & set(te_ids))}"
          "  （id 是各自分割内的编号，不是全局概念 id）")
    return 0 if not overlap else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
