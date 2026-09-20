"""V24.0 — CLIPTeacher + MiniImageTextCorpus 单元测试 (8 项)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from research.multimodal.v24_clip import CLIPTeacher, MiniImageTextCorpus, generate_color_image, COLOR_RGB


def test_generate_color_image_shape():
    """1. generate_color_image 输出 [3, H, W] 正确"""
    img = generate_color_image("red", image_size=224)
    assert img.shape == (3, 224, 224)
    assert img[0].mean() == 1.0  # R 通道 = 1
    assert img[1].mean() == 0.0  # G 通道 = 0
    assert img[2].mean() == 0.0  # B 通道 = 0
    print("[PASS] test_generate_color_image_shape")
    return True


def test_color_rgb_mapping():
    """2. 11 个颜色都有 RGB 定义"""
    expected = ["red", "blue", "green", "yellow", "black", "white",
                "purple", "orange", "brown", "pink", "gray"]
    for color in expected:
        assert color in COLOR_RGB, f"missing {color}"
        assert len(COLOR_RGB[color]) == 3
    print(f"[PASS] test_color_rgb_mapping ({len(COLOR_RGB)} colors)")
    return True


def test_mini_image_text_corpus_train_size():
    """3. train 集大小正确"""
    corpus = MiniImageTextCorpus(train_size=12)
    assert len(corpus.train_samples) == 12
    assert len(corpus.eval_samples) == 4
    print(f"[PASS] test_mini_image_text_corpus_train_size (12 train + 4 eval)")
    return True


def test_corpus_samples_have_required_fields():
    """4. 每个样本含 image, text, color 字段"""
    corpus = MiniImageTextCorpus()
    for s in corpus.all_train() + corpus.all_eval():
        assert "image" in s
        assert "text" in s
        assert "color" in s
        assert s["image"].shape == (3, 224, 224)
        assert isinstance(s["text"], str)
    print(f"[PASS] test_corpus_samples_have_required_fields")
    return True


def test_text_color_consistency():
    """5. text 与 color 一致 (e.g. 'a red square' -> 'red')"""
    corpus = MiniImageTextCorpus()
    for s in corpus.all_train() + corpus.all_eval():
        # text 包含 color 字符串
        assert s["color"] in s["text"], f"text '{s['text']}' 不含 color '{s['color']}'"
        # 颜色 RGB 一致
        r, g, b = COLOR_RGB[s["color"]]
        assert abs(s["image"][0].mean() - r) < 0.01
        assert abs(s["image"][1].mean() - g) < 0.01
        assert abs(s["image"][2].mean() - b) < 0.01
    print(f"[PASS] test_text_color_consistency")
    return True


def test_clip_teacher_loads_and_freezes():
    """6. CLIPTeacher 加载并冻结所有参数"""
    teacher = CLIPTeacher()
    # 所有参数应冻结
    n_trainable = sum(p.numel() for p in teacher.model.parameters()
                       if p.requires_grad)
    assert n_trainable == 0, f"CLIP 有 {n_trainable} 个可训练参数, 应全部冻结"
    assert teacher.text_dim == 512, f"text_dim 应=512, 实测 {teacher.text_dim}"
    print(f"[PASS] test_clip_teacher_loads_and_freezes (text_dim={teacher.text_dim})")
    return True


def test_clip_teacher_encode_image_and_text():
    """7. CLIP encode_image / encode_text 输出 [B, 512]"""
    teacher = CLIPTeacher()
    corpus = MiniImageTextCorpus()
    s = corpus.all_train()[0]
    # 编码 image
    pixel_values = s["image"].unsqueeze(0)  # [1, 3, 224, 224]
    img_feat = teacher.encode_image(pixel_values)
    assert img_feat.shape == (1, 512), f"img_feat shape {img_feat.shape}"
    # 编码 text
    from transformers import CLIPTokenizer
    text_inputs = teacher.processor(text=[s["text"]], return_tensors="pt",
                                     padding=True, truncation=True)
    txt_feat = teacher.encode_text(text_inputs["input_ids"], text_inputs["attention_mask"])
    assert txt_feat.shape == (1, 512)
    print(f"[PASS] test_clip_teacher_encode_image_and_text")
    return True


def test_clip_teacher_fused_target_shape():
    """8. CLIPTeacher.fused_target 输出 [B, 512] 融合特征"""
    teacher = CLIPTeacher()
    corpus = MiniImageTextCorpus()
    s = corpus.all_train()[0]
    pixel_values = s["image"].unsqueeze(0)
    text_inputs = teacher.processor(text=[s["text"]], return_tensors="pt",
                                     padding=True, truncation=True)
    # alpha=0.5 默认 (text + image 平均)
    target = teacher.fused_target(
        pixel_values, text_inputs["input_ids"], text_inputs["attention_mask"],
        alpha=0.5
    )
    assert target.shape == (1, 512)
    # alpha=0 (纯 image)
    target_img = teacher.fused_target(
        pixel_values, text_inputs["input_ids"], text_inputs["attention_mask"],
        alpha=0.0
    )
    assert target_img.shape == (1, 512)
    # 验证 alpha 影响
    assert not torch.allclose(target, target_img), "alpha 应影响 target"
    print(f"[PASS] test_clip_teacher_fused_target_shape")
    return True


TESTS = [
    test_generate_color_image_shape,
    test_color_rgb_mapping,
    test_mini_image_text_corpus_train_size,
    test_corpus_samples_have_required_fields,
    test_text_color_consistency,
    test_clip_teacher_loads_and_freezes,
    test_clip_teacher_encode_image_and_text,
    test_clip_teacher_fused_target_shape,
]


if __name__ == "__main__":
    print("=== V24.0 CLIPTeacher + MiniImageTextCorpus — 单元测试 (8 项) ===")
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