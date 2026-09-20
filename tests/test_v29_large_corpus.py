"""V29.0 — 大型真实语料 单元测试 (12 项).

测试覆盖:
    1.  LargeTextCorpus 64 train + 24 held-out
    2.  LargeTextCorpus.train_strings 返回 list[str]
    3.  LargeTextCorpus held-out ∩ train = ∅
    4.  LargeCodeCorpus 32 train + 12 held-out
    5.  LargeCodeCorpus 字符长度 ≤ 60
    6.  LargeCodeCorpus held-out ∩ train = ∅
    7.  LargeImageCorpus 50 train + 16 held-out
    8.  LargeImageCorpus held-out 主题名 ∩ train = ∅
    9.  LargeImageCorpus 6 种形状都能生成
    10. generate_wiki_image 单主题 → [224, 224, 3] uint8
    11. prepare_v29_real_corpus_data 接口存在
    12. 端到端冒烟: 1 seed × 1 mode × 1 corpus 跑通
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 冒烟用例经 v29 layer 惰性导入 examples/run_v8_full (其依赖 experiment_env),
# 故需把 examples/ 加入 import 路径。
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))

import math

import numpy as np
import torch

from research.corpus_scale.v29_large_corpus import (
    LargeTextCorpus, LargeCodeCorpus, LargeImageCorpus,
    generate_wiki_image, WIKI_COLORS_LARGE,
    prepare_v29_real_corpus_data, prepare_v29_held_out_data,
)
from research.corpus_scale.v29_large_corpus.text_corpus import (
    _LARGE_TEXT_CORPUS, _LARGE_TEXT_HELD_OUT,
)
from research.corpus_scale.v29_large_corpus.code_corpus import (
    _LARGE_PY_CORPUS, _LARGE_PY_HELD_OUT,
)
from research.corpus_scale.v29_large_corpus.image_corpus import (
    _LARGE_WIKI_THEMES, _LARGE_HELD_OUT_THEMES,
)


# ---------------------------------------------------------------------------
# text 测试
# ---------------------------------------------------------------------------


def test_large_text_corpus_size():
    """1. LargeTextCorpus 64 train + 24 held-out."""
    assert len(_LARGE_TEXT_CORPUS) == 64, f"train 应 64, 实测 {len(_LARGE_TEXT_CORPUS)}"
    assert len(_LARGE_TEXT_HELD_OUT) == 24, f"held_out 应 24, 实测 {len(_LARGE_TEXT_HELD_OUT)}"
    print(f"[PASS] test_large_text_corpus_size (64 train + 24 held-out)")
    return True


def test_large_text_corpus_train_strings_return_list():
    """2. LargeTextCorpus.train_strings 返回 list[str]."""
    corpus = LargeTextCorpus.__new__(LargeTextCorpus)
    corpus.train_sentences = _LARGE_TEXT_CORPUS
    corpus.held_out = _LARGE_TEXT_HELD_OUT
    corpus.tokenizer = None
    corpus.max_length = 16

    train_strs = corpus.train_strings()
    assert isinstance(train_strs, list)
    assert len(train_strs) == 64
    assert all(isinstance(s, str) for s in train_strs)
    eval_strs = corpus.eval_strings()
    assert isinstance(eval_strs, list)
    assert len(eval_strs) == 24
    print(f"[PASS] test_large_text_corpus_train_strings_return_list (64 train + 24 eval)")
    return True


def test_large_text_corpus_held_out_disjoint_from_train():
    """3. LargeTextCorpus held-out ∩ train = ∅."""
    train_set = set(_LARGE_TEXT_CORPUS)
    held_out_set = set(_LARGE_TEXT_HELD_OUT)
    overlap = train_set & held_out_set
    assert len(overlap) == 0, f"held-out 与 train 有重叠: {overlap}"
    print(f"[PASS] test_large_text_corpus_held_out_disjoint_from_train (intersection=∅)")
    return True


# ---------------------------------------------------------------------------
# code 测试
# ---------------------------------------------------------------------------


def test_large_code_corpus_size():
    """4. LargeCodeCorpus 32 train + 12 held-out."""
    assert len(_LARGE_PY_CORPUS) == 32, f"train 应 32, 实测 {len(_LARGE_PY_CORPUS)}"
    assert len(_LARGE_PY_HELD_OUT) == 12, f"held_out 应 12, 实测 {len(_LARGE_PY_HELD_OUT)}"
    print(f"[PASS] test_large_code_corpus_size (32 train + 12 held-out)")
    return True


def test_large_code_corpus_snippet_length_reasonable():
    """5. LargeCodeCorpus 字符长度 ≤ 60 (适配 ≤ 16 BPE token)."""
    for s in _LARGE_PY_CORPUS + _LARGE_PY_HELD_OUT:
        assert len(s) <= 60, f"snippet 过长 ({len(s)}): {s!r}"
    max_len = max(len(s) for s in _LARGE_PY_CORPUS + _LARGE_PY_HELD_OUT)
    print(f"[PASS] test_large_code_corpus_snippet_length_reasonable "
          f"(44 片段均 ≤ 60 字符, max={max_len})")
    return True


def test_large_code_corpus_held_out_disjoint_from_train():
    """6. LargeCodeCorpus held-out ∩ train = ∅."""
    train_set = set(_LARGE_PY_CORPUS)
    held_out_set = set(_LARGE_PY_HELD_OUT)
    overlap = train_set & held_out_set
    assert len(overlap) == 0, f"held-out 与 train 有重叠: {overlap}"
    print(f"[PASS] test_large_code_corpus_held_out_disjoint_from_train (intersection=∅)")
    return True


# ---------------------------------------------------------------------------
# image 测试
# ---------------------------------------------------------------------------


def test_large_image_corpus_size():
    """7. LargeImageCorpus 50 train + 16 held-out."""
    assert len(_LARGE_WIKI_THEMES) == 50, f"train 应 50, 实测 {len(_LARGE_WIKI_THEMES)}"
    assert len(_LARGE_HELD_OUT_THEMES) == 16, f"held_out 应 16, 实测 {len(_LARGE_HELD_OUT_THEMES)}"
    print(f"[PASS] test_large_image_corpus_size (50 train + 16 held-out)")
    return True


def test_large_image_corpus_held_out_disjoint_from_train():
    """8. LargeImageCorpus held-out 主题名 ∩ train 主题名 = ∅."""
    train_set = set(t[0] for t in _LARGE_WIKI_THEMES)
    held_out_set = set(t[0] for t in _LARGE_HELD_OUT_THEMES)
    overlap = train_set & held_out_set
    assert len(overlap) == 0, f"held-out 与 train 主题名重叠: {overlap}"
    print(f"[PASS] test_large_image_corpus_held_out_disjoint_from_train (主题名交集=∅)")
    return True


def test_large_image_corpus_6_shapes():
    """9. LargeImageCorpus 6 种形状都能生成."""
    shapes = set(t[2] for t in _LARGE_WIKI_THEMES + _LARGE_HELD_OUT_THEMES)
    expected = {"circle", "rect", "triangle", "wave", "hexagon", "star"}
    assert shapes == expected, f"形状集 应 = {expected}, 实测 {shapes}"
    # 每种形状都能生成
    for shape in expected:
        # 找对应主题
        theme = next((t for t in _LARGE_WIKI_THEMES + _LARGE_HELD_OUT_THEMES if t[2] == shape), None)
        assert theme is not None
        img = generate_wiki_image(theme, image_size=224)
        assert img.shape == (224, 224, 3)
        assert img.dtype == np.uint8
    print(f"[PASS] test_large_image_corpus_6_shapes (circle/rect/triangle/wave/hexagon/star)")
    return True


def test_generate_wiki_image_single_theme():
    """10. generate_wiki_image 单主题 → [224, 224, 3] uint8."""
    corpus = LargeImageCorpus(image_size=224)
    for theme in _LARGE_WIKI_THEMES[:3] + _LARGE_HELD_OUT_THEMES[:3]:
        img = generate_wiki_image(theme, image_size=224)
        assert img.shape == (224, 224, 3)
        assert img.dtype == np.uint8
        # 像素不全为 0 (有内容)
        assert img.sum() > 0
    print(f"[PASS] test_generate_wiki_image_single_theme (6 主题生成正确)")
    return True


# ---------------------------------------------------------------------------
# 集成层测试
# ---------------------------------------------------------------------------


def test_prepare_v29_real_corpus_data_function_exists():
    """11. prepare_v29_real_corpus_data 接口存在."""
    import inspect
    sig = inspect.signature(prepare_v29_real_corpus_data)
    params = list(sig.parameters.keys())
    expected = ['seed', 'encoders', 'text_corpus', 'code_corpus', 'image_corpus',
                'n_per_class', 'S', 'D_SHARED', 'N_CLS']
    for p in expected:
        assert p in params, f"缺少参数: {p}"

    sig2 = inspect.signature(prepare_v29_held_out_data)
    params2 = list(sig2.parameters.keys())
    expected2 = ['encoders', 'text_corpus', 'code_corpus', 'image_corpus',
                 'S', 'D_SHARED', 'N_CLS', 'n_per_class']
    for p in expected2:
        assert p in params2, f"prepare_v29_held_out_data 缺少参数: {p}"
    print(f"[PASS] test_prepare_v29_real_corpus_data_function_exists")
    return True


def test_end_to_end_smoke_with_real_corpus():
    """12. 端到端冒烟: 1 seed × 1 mode × 1 corpus 跑通.

    用 stub encoders + LargeTextCorpus/LargeCodeCorpus/LargeImageCorpus.
    """
    # Stub encoders
    class _StubTok:
        pad_token_id = 0
        def __call__(self, sentences=None, **kw):
            if sentences is None:
                sentences = ["stub"]
            if not isinstance(sentences, list):
                sentences = [sentences]
            return {"input_ids": torch.randint(0, 100, (len(sentences), 16))}

    class _StubVitProc:
        def __call__(self, images=None, **kw):
            return {"pixel_values": torch.randn(1, 3, 224, 224)}

    class _StubModel:
        def __init__(self, D_m):
            self.D_m = D_m
        def __call__(self, input_ids_or_pixels):
            if input_ids_or_pixels.dim() == 2:
                return type("O", (), {"last_hidden_state": torch.randn(input_ids_or_pixels.shape[0], 16, self.D_m) * 0.1})()
            else:
                return type("O", (), {"last_hidden_state": torch.randn(1, 197, self.D_m) * 0.1})()

    encoders = (
        (_StubTok(), _StubModel(312), 312, None),  # bert
        (_StubTok(), _StubModel(768), 768, None),  # llama
        (_StubVitProc(), _StubModel(192), 192, None),  # vit
    )
    text_corpus = LargeTextCorpus.__new__(LargeTextCorpus)
    text_corpus.train_sentences = _LARGE_TEXT_CORPUS
    text_corpus.held_out = _LARGE_TEXT_HELD_OUT
    text_corpus.tokenizer = None
    text_corpus.max_length = 16

    code_corpus = LargeCodeCorpus.__new__(LargeCodeCorpus)
    code_corpus.train_snippets = _LARGE_PY_CORPUS
    code_corpus.held_out = _LARGE_PY_HELD_OUT
    code_corpus.tokenizer = None
    code_corpus.max_length = 16

    image_corpus = LargeImageCorpus(image_size=224)

    # prepare_real_corpus_data
    modal_seqs, modal_indices, targets = prepare_v29_real_corpus_data(
        seed=0, encoders=encoders,
        text_corpus=text_corpus, code_corpus=code_corpus, image_corpus=image_corpus,
        n_per_class=6, S=16, D_SHARED=256, N_CLS=3,
    )
    assert len(modal_seqs) == 18, f"应 18 个 modal_seqs, 实测 {len(modal_seqs)}"
    assert len(modal_indices) == 18
    assert all(0 <= m <= 2 for m in modal_indices)
    # 6 per class
    for cls in range(3):
        assert sum(1 for m in modal_indices if m == cls) == 6
    # held-out
    ho_seqs, ho_indices, ho_targets = prepare_v29_held_out_data(
        encoders=encoders,
        text_corpus=text_corpus, code_corpus=code_corpus, image_corpus=image_corpus,
        S=16, D_SHARED=256, N_CLS=3, n_per_class=6,
    )
    assert len(ho_seqs) == 18
    print(f"[PASS] test_end_to_end_smoke_with_real_corpus")
    return True


TESTS = [
    test_large_text_corpus_size,
    test_large_text_corpus_train_strings_return_list,
    test_large_text_corpus_held_out_disjoint_from_train,
    test_large_code_corpus_size,
    test_large_code_corpus_snippet_length_reasonable,
    test_large_code_corpus_held_out_disjoint_from_train,
    test_large_image_corpus_size,
    test_large_image_corpus_held_out_disjoint_from_train,
    test_large_image_corpus_6_shapes,
    test_generate_wiki_image_single_theme,
    test_prepare_v29_real_corpus_data_function_exists,
    test_end_to_end_smoke_with_real_corpus,
]


if __name__ == "__main__":
    print("=== V29.0 大型真实语料 — 单元测试 (12 项) ===")
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
