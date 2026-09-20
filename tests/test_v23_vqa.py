"""V23.0 — MiniVQADataset (合成 VQA) 单元测试 (8 项)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from research.multimodal.v23_vqa import MiniVQADataset, generate_synthetic_vqa_sample, QUESTION_TEMPLATES, ANSWER_LABELS


def test_generate_synthetic_vqa_sample_shapes():
    """1. 合成 VQA 样本形状正确"""
    img, label, question = generate_synthetic_vqa_sample(idx=0, image_size=224)
    assert img.shape == (3, 224, 224), f"image shape {img.shape}"
    assert isinstance(label, int)
    assert isinstance(question, str)
    assert label in [0, 1, 2, 3]
    print("[PASS] test_generate_synthetic_vqa_sample_shapes")
    return True


def test_generate_synthetic_vqa_label_cycling():
    """2. 4 个类别循环 (idx % 4)"""
    labels = [generate_synthetic_vqa_sample(i)[1] for i in range(8)]
    expected = [0, 1, 2, 3, 0, 1, 2, 3]
    assert labels == expected, f"label 循环错误: {labels}"
    print("[PASS] test_generate_synthetic_vqa_label_cycling")
    return True


def test_question_template_per_label():
    """3. 4 个 label 对应 4 个问题模板"""
    for label in range(4):
        img, l, q = generate_synthetic_vqa_sample(idx=label)
        assert l == label
        assert q == QUESTION_TEMPLATES[label], f"label {label} 用了错误模板 {q}"
    print("[PASS] test_question_template_per_label")
    return True


def test_mini_vqa_dataset_train_eval_split():
    """4. train 和 eval 不重叠"""
    dataset = MiniVQADataset(num_train=12)
    assert len(dataset.train_samples) == 12
    assert len(dataset.eval_samples) == 4
    # 训练 idx 0-11, eval idx 12-15 (不同 idx → 不同合成图)
    train_imgs = [s["image"].sum() for s in dataset.all_train()]
    eval_imgs = [s["image"].sum() for s in dataset.all_eval()]
    # 不同 idx 应生成不同图 (圆位置/颜色)
    assert train_imgs != eval_imgs, "train 和 eval 用了相同图?"
    print("[PASS] test_mini_vqa_dataset_train_eval_split")
    return True


def test_mini_vqa_dataset_getitem():
    """5. __getitem__ 返回正确字段"""
    dataset = MiniVQADataset()
    s = dataset[0]
    assert "image" in s
    assert "label" in s
    assert "question" in s
    assert s["image"].shape == (3, 224, 224)
    assert isinstance(s["label"], int)
    print("[PASS] test_mini_vqa_dataset_getitem")
    return True


def test_mini_vqa_yes_no_samples_have_distinguishable_images():
    """6. yes/no 类别图像可区分 (yes 有白圆, no 全黑)"""
    dataset = MiniVQADataset(num_train=8)  # 含 yes/no 两次
    # idx 0 (yes) 有白圆, idx 1 (no) 全黑
    yes_img = dataset.train_samples[0]["image"]
    no_img = dataset.train_samples[1]["image"]
    assert yes_img.sum() > 0, "yes 类别应有白圆"
    assert no_img.sum() == 0, "no 类别应全黑"
    # pixel count 显著不同
    assert (yes_img > 0).sum() > (no_img > 0).sum() + 100, "yes 应有显著更多白色像素"
    print(f"[PASS] test_mini_vqa_yes_no_samples_have_distinguishable_images "
          f"(yes={int((yes_img>0).sum())} px, no={int((no_img>0).sum())} px)")
    return True


def test_mini_vqa_color_samples_have_distinguishable_images():
    """7. red/blue 类别颜色可区分 (R 通道 vs B 通道)"""
    dataset = MiniVQADataset(num_train=8)
    red_img = dataset.train_samples[2]["image"]   # idx=2 -> label=2 -> red
    blue_img = dataset.train_samples[3]["image"]  # idx=3 -> label=3 -> blue
    # red: R 通道=1, B 通道=0
    assert red_img[0].mean() == 1.0, "red 应 R 通道=1"
    assert red_img[2].mean() == 0.0, "red 应 B 通道=0"
    # blue: R 通道=0, B 通道=1
    assert blue_img[0].mean() == 0.0, "blue 应 R 通道=0"
    assert blue_img[2].mean() == 1.0, "blue 应 B 通道=1"
    print("[PASS] test_mini_vqa_color_samples_have_distinguishable_images")
    return True


def test_mini_vqa_eval_set_size_and_distinct_from_train():
    """8. eval 集大小正确, 且与 train 完全分离"""
    dataset = MiniVQADataset(num_train=12)
    assert len(dataset.eval_samples) == 4
    # eval 用了不同 seed, 应生成不同图
    train_seeds = set()
    eval_seeds = set()
    for i, s in enumerate(dataset.train_samples):
        train_seeds.add(int(s["image"].sum().item()))
    for s in dataset.eval_samples:
        eval_seeds.add(int(s["image"].sum().item()))
    # 至少有 4 个不同的 sum (yes/no/red/blue 像素数不同)
    assert len(train_seeds) >= 3, f"train 集图像应有 ≥3 种像素数, 实测 {len(train_seeds)}"
    assert len(eval_seeds) >= 3, f"eval 集图像应有 ≥3 种像素数, 实测 {len(eval_seeds)}"
    print(f"[PASS] test_mini_vqa_eval_set_size_and_distinct_from_train "
          f"(train={len(train_seeds)} unique sums, eval={len(eval_seeds)} unique sums)")
    return True


TESTS = [
    test_generate_synthetic_vqa_sample_shapes,
    test_generate_synthetic_vqa_label_cycling,
    test_question_template_per_label,
    test_mini_vqa_dataset_train_eval_split,
    test_mini_vqa_dataset_getitem,
    test_mini_vqa_yes_no_samples_have_distinguishable_images,
    test_mini_vqa_color_samples_have_distinguishable_images,
    test_mini_vqa_eval_set_size_and_distinct_from_train,
]


if __name__ == "__main__":
    print("=== V23.0 MiniVQADataset — 单元测试 (8 项) ===")
    passed = 0
    for fn in TESTS:
        try:
            if fn():
                passed += 1
        except Exception as e:
            print(f"[FAIL] {fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"---\n合计: {passed}/{len(TESTS)} {'✓ ALL PASS' if passed == len(TESTS) else '✗ HAS FAIL'}")
    import sys
    sys.exit(0 if passed == len(TESTS) else 1)