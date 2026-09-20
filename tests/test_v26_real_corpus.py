"""V26.0 — fully-real-corpus 单元测试 (11 项).

v26 新增:
    MiniCodeCorpus (16 真实 Python 片段 + 6 held-out)
    MiniImageCorpus (16 Wikipedia 主题结构图 + 6 held-out)
    prepare_v26_real_corpus_data / prepare_v26_held_out_data (集成层)

测试覆盖:
    1. MiniCodeCorpus import + 长度
    2. MiniCodeCorpus.train_strings 返回 list[str]
    3. MiniCodeCorpus held-out ∩ train = ∅
    4. MiniCodeCorpus 片段 ≤60 字符 (合理 token 长度)
    5. MiniImageCorpus import + 长度
    6. MiniImageCorpus.train_batch 返回 [B, 224, 224, 3] uint8
    7. MiniImageCorpus.eval_batch 返回 [6, 224, 224, 3] uint8
    8. MiniImageCorpus held-out ∩ train = ∅
    9. generate_wiki_image 单主题生成正确
    10. train_batch 随机性 (≥5 unique sigs in 20 batches)
    11. 集成层 prepare_v26_real_corpus_data 接口存在 (stub 验证)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from research.corpus_scale.v26_real_corpus import (
    MiniCodeCorpus,
    MiniImageCorpus,
    generate_wiki_image,
    WIKI_COLORS,
    prepare_v26_real_corpus_data,
    prepare_v26_held_out_data,
)


# ---------------------------------------------------------------------------
# 测试 1-4: MiniCodeCorpus
# ---------------------------------------------------------------------------


def test_mini_code_corpus_constants():
    """1. MiniCodeCorpus 训练 16 + held-out 6 (与 v18 镜像)."""
    from research.corpus_scale.v26_real_corpus.code_corpus import _MINI_PY_CORPUS, _PY_HELD_OUT
    assert len(_MINI_PY_CORPUS) == 16, f"train 应 16 句, 实测 {len(_MINI_PY_CORPUS)}"
    assert len(_PY_HELD_OUT) == 6, f"held_out 应 6 句, 实测 {len(_PY_HELD_OUT)}"
    print(f"[PASS] test_mini_code_corpus_constants (16 train + 6 held-out)")
    return True


def test_mini_code_corpus_train_strings_return_list():
    """2. MiniCodeCorpus.train_strings 返回 list[str]."""
    corpus = MiniCodeCorpus.__new__(MiniCodeCorpus)
    from research.corpus_scale.v26_real_corpus.code_corpus import _MINI_PY_CORPUS, _PY_HELD_OUT
    corpus.train_snippets = _MINI_PY_CORPUS
    corpus.held_out = _PY_HELD_OUT
    corpus.tokenizer = None
    corpus.max_length = 16

    train_strs = corpus.train_strings()
    assert isinstance(train_strs, list)
    assert len(train_strs) == 16
    assert all(isinstance(s, str) for s in train_strs)
    eval_strs = corpus.eval_strings()
    assert isinstance(eval_strs, list)
    assert len(eval_strs) == 6
    print(f"[PASS] test_mini_code_corpus_train_strings_return_list (16 train + 6 eval)")
    return True


def test_mini_code_corpus_held_out_disjoint_from_train():
    """3. MiniCodeCorpus held-out ∩ train = ∅."""
    from research.corpus_scale.v26_real_corpus.code_corpus import _MINI_PY_CORPUS, _PY_HELD_OUT
    train_set = set(_MINI_PY_CORPUS)
    held_out_set = set(_PY_HELD_OUT)
    assert train_set & held_out_set == set(), "held-out 与 train 有重叠"
    print(f"[PASS] test_mini_code_corpus_held_out_disjoint_from_train (intersection=∅)")
    return True


def test_mini_code_corpus_snippet_length_reasonable():
    """4. 所有 Python 片段字符长度 ≤ 60 (保证 ≤16 token)."""
    from research.corpus_scale.v26_real_corpus.code_corpus import _MINI_PY_CORPUS, _PY_HELD_OUT
    for s in _MINI_PY_CORPUS + _PY_HELD_OUT:
        # 60 字符 ≈ 16-20 BPE token (适配 S=16)
        assert len(s) <= 60, f"snippet 过长 ({len(s)}): {s!r}"
    print(f"[PASS] test_mini_code_corpus_snippet_length_reasonable "
          f"(22 片段均 ≤ 60 字符, max={max(len(s) for s in _MINI_PY_CORPUS + _PY_HELD_OUT)})")
    return True


# ---------------------------------------------------------------------------
# 测试 5-10: MiniImageCorpus
# ---------------------------------------------------------------------------


def test_mini_image_corpus_constants():
    """5. MiniImageCorpus 训练 16 + held-out 6 (与 v18 镜像)."""
    from research.corpus_scale.v26_real_corpus.image_corpus import _WIKI_THEMES, _HELD_OUT_THEMES
    assert len(_WIKI_THEMES) == 16, f"train 应 16 主题, 实测 {len(_WIKI_THEMES)}"
    assert len(_HELD_OUT_THEMES) == 6, f"held_out 应 6 主题, 实测 {len(_HELD_OUT_THEMES)}"
    print(f"[PASS] test_mini_image_corpus_constants (16 train + 6 held-out)")
    return True


def test_mini_image_corpus_train_batch_shape():
    """6. MiniImageCorpus.train_batch 返回 [B, 224, 224, 3] uint8."""
    corpus = MiniImageCorpus(image_size=224)
    batch = corpus.train_batch(batch_size=4)
    assert batch.shape == (4, 224, 224, 3), f"shape 应 (4, 224, 224, 3), 实测 {batch.shape}"
    assert batch.dtype == np.uint8, f"dtype 应 uint8, 实测 {batch.dtype}"
    print(f"[PASS] test_mini_image_corpus_train_batch_shape (4, 224, 224, 3) uint8")
    return True


def test_mini_image_corpus_eval_batch_shape():
    """7. MiniImageCorpus.eval_batch 返回 [6, 224, 224, 3] uint8."""
    corpus = MiniImageCorpus(image_size=224)
    batch = corpus.eval_batch(batch_size=6)
    assert batch.shape == (6, 224, 224, 3)
    assert batch.dtype == np.uint8
    print(f"[PASS] test_mini_image_corpus_eval_batch_shape (6, 224, 224, 3) uint8")
    return True


def test_mini_image_corpus_held_out_disjoint_from_train():
    """8. MiniImageCorpus held-out 主题 ∩ train 主题 = ∅."""
    corpus = MiniImageCorpus(image_size=224)
    train_set = set(t[0] for t in corpus.train_themes)  # 用主题名 (str) 作 key
    held_out_set = set(t[0] for t in corpus.held_out)
    assert train_set & held_out_set == set(), f"held-out 与 train 主题重叠: {train_set & held_out_set}"
    print(f"[PASS] test_mini_image_corpus_held_out_disjoint_from_train (主题名交集=∅)")
    return True


def test_generate_wiki_image_single_theme():
    """9. generate_wiki_image 单主题 → [224, 224, 3] uint8 图像."""
    corpus = MiniImageCorpus(image_size=224)
    for theme in corpus.train_themes[:3]:
        img = generate_wiki_image(theme, image_size=224)
        assert img.shape == (224, 224, 3)
        assert img.dtype == np.uint8
        # 像素不全为 0 (有内容)
        assert img.sum() > 0
        # 像素不全相同 (有变化, 不是纯色块)
        assert img.std() > 0
    print(f"[PASS] test_generate_wiki_image_single_theme (3 主题均生成有效图像)")
    return True


def test_mini_image_corpus_random_sampling():
    """10. train_batch 随机性 (≥5 unique sigs in 20 batches)."""
    corpus = MiniImageCorpus(image_size=224)
    seen = set()
    for _ in range(20):
        batch = corpus.train_batch(batch_size=4)
        sig = float(batch.flatten().sum())  # 用总和作 hash 代理
        seen.add(sig)
    assert len(seen) >= 5, f"多样性不足, 仅 {len(seen)} 种 sig"
    print(f"[PASS] test_mini_image_corpus_random_sampling ({len(seen)} unique sigs in 20 batches)")
    return True


# ---------------------------------------------------------------------------
# 测试 11: 集成层接口存在
# ---------------------------------------------------------------------------


def test_prepare_v26_real_corpus_data_function_exists():
    """11. prepare_v26_real_corpus_data 接口存在 (stub 验证)."""
    import inspect
    sig = inspect.signature(prepare_v26_real_corpus_data)
    params = list(sig.parameters.keys())
    expected = ['seed', 'encoders', 'text_corpus', 'code_corpus', 'image_corpus',
                'n_per_class', 'S', 'D_SHARED', 'N_CLS']
    for p in expected:
        assert p in params, f"缺少参数: {p}"
    # 返回类型注解: Tuple[List[Tensor], List[int], List[Tensor]]
    sig2 = inspect.signature(prepare_v26_held_out_data)
    params2 = list(sig2.parameters.keys())
    expected2 = ['encoders', 'text_corpus', 'code_corpus', 'image_corpus',
                 'S', 'D_SHARED', 'N_CLS', 'n_per_class']
    for p in expected2:
        assert p in params2, f"prepare_v26_held_out_data 缺少参数: {p}"
    print(f"[PASS] test_prepare_v26_real_corpus_data_function_exists")
    return True


TESTS = [
    test_mini_code_corpus_constants,
    test_mini_code_corpus_train_strings_return_list,
    test_mini_code_corpus_held_out_disjoint_from_train,
    test_mini_code_corpus_snippet_length_reasonable,
    test_mini_image_corpus_constants,
    test_mini_image_corpus_train_batch_shape,
    test_mini_image_corpus_eval_batch_shape,
    test_mini_image_corpus_held_out_disjoint_from_train,
    test_generate_wiki_image_single_theme,
    test_mini_image_corpus_random_sampling,
    test_prepare_v26_real_corpus_data_function_exists,
]


if __name__ == "__main__":
    print("=== V26.0 fully-real-corpus — 单元测试 (11 项) ===")
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
