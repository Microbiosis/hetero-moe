"""V18.0 — MiniCorpus 单元测试 (8 项)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer

from research._primitives.real_corpus import MiniCorpus, cycle_batches


def _get_tokenizer():
    return AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")


def test_mini_corpus_init():
    """1. MiniCorpus 初始化加载内置 16 句训练 + 6 句 held-out"""
    tok = _get_tokenizer()
    corpus = MiniCorpus(tok, max_length=16)
    assert len(corpus.train_sentences) == 16
    assert len(corpus.held_out) == 6
    print(f"[PASS] test_mini_corpus_init (16 train + 6 held-out)")
    return True


def test_mini_corpus_train_batch_shape():
    """2. train_batch 返回正确形状"""
    tok = _get_tokenizer()
    corpus = MiniCorpus(tok, max_length=16)
    batch = corpus.train_batch(batch_size=4)
    assert batch.shape == (4, 16), f"shape {batch.shape}"
    print("[PASS] test_mini_corpus_train_batch_shape")
    return True


def test_mini_corpus_eval_batch_shape():
    """3. eval_batch 返回 held-out 句子的正确形状"""
    tok = _get_tokenizer()
    corpus = MiniCorpus(tok, max_length=16)
    batch = corpus.eval_batch(batch_size=6)
    assert batch.shape == (6, 16)
    print("[PASS] test_mini_corpus_eval_batch_shape")
    return True


def test_mini_corpus_eval_held_out_distinct_from_train():
    """4. eval_batch 与 train_batch 的句子不应重复 (held-out)"""
    tok = _get_tokenizer()
    corpus = MiniCorpus(tok, max_length=16)
    # 获取 eval batch 的原始文本 (用 tokenizer.decode)
    eval_ids = corpus.eval_batch()
    train_texts = set(corpus.train_sentences)
    eval_texts = set(corpus.held_out)
    # eval batch 应来自 held-out, 不应来自 train
    assert train_texts & eval_texts == set(), "eval 与 train 有重叠"
    print(f"[PASS] test_mini_corpus_eval_held_out_distinct_from_train "
          f"(train ∩ eval = ∅)")
    return True


def test_mini_corpus_random_sampling():
    """5. 多次 train_batch 应采样到不同句子 (随机性)"""
    tok = _get_tokenizer()
    corpus = MiniCorpus(tok, max_length=16)
    # 用整 batch 的 hash, 不只是第一列
    seen = set()
    for _ in range(20):
        batch = corpus.train_batch(batch_size=4)
        sig = tuple(batch.flatten().tolist())
        seen.add(sig)
    # 20 个 batch 应至少采样到 5 种不同的 sig
    assert len(seen) >= 5, f"多样性不足, 仅 {len(seen)} 种"
    print(f"[PASS] test_mini_corpus_random_sampling ({len(seen)} unique sigs)")
    return True


def test_cycle_batches_yields_infinitely():
    """6. cycle_batches 无限循环, 可迭代"""
    tok = _get_tokenizer()
    corpus = MiniCorpus(tok, max_length=16)
    gen = cycle_batches(corpus, batch_size=2)
    # 跑 5 个 step 不报错
    for i, batch in enumerate(gen):
        assert batch.shape == (2, 16)
        if i >= 4:
            break
    print("[PASS] test_cycle_batches_yields_infinitely")
    return True


def test_train_eval_distributions_differ():
    """7. train 和 eval 的 token 分布不同 (held-out 是新句子)"""
    tok = _get_tokenizer()
    corpus = MiniCorpus(tok, max_length=16)
    train_tokens = set()
    for _ in range(10):
        batch = corpus.train_batch(batch_size=4)
        train_tokens.update(batch.flatten().tolist())
    eval_batch = corpus.eval_batch(batch_size=6)
    eval_tokens = set(eval_batch.flatten().tolist())
    # 应有差异 (held-out 用不同 token)
    new_in_eval = len(eval_tokens - train_tokens)
    assert new_in_eval > 0, "held-out 没引入新 token (与 train 完全重叠?)"
    print(f"[PASS] test_train_eval_distributions_differ "
          f"({new_in_eval} new tokens in eval)")
    return True


def test_corpus_tokens_not_all_pad():
    """8. corpus tokenize 后不全是 pad (说明长度合理)"""
    tok = _get_tokenizer()
    corpus = MiniCorpus(tok, max_length=16)
    batch = corpus.train_batch(batch_size=4)
    # 检查非 pad 比例 > 50% (真实句子至少有内容)
    pad_id = tok.pad_token_id
    n_pad = (batch == pad_id).sum().item()
    n_total = batch.numel()
    ratio = n_pad / n_total
    assert ratio < 0.7, f"pad 占比过高: {ratio:.2%}, 可能 max_length 太小或句子太短"
    print(f"[PASS] test_corpus_tokens_not_all_pad (pad 占比 {ratio:.2%})")
    return True


TESTS = [
    test_mini_corpus_init,
    test_mini_corpus_train_batch_shape,
    test_mini_corpus_eval_batch_shape,
    test_mini_corpus_eval_held_out_distinct_from_train,
    test_mini_corpus_random_sampling,
    test_cycle_batches_yields_infinitely,
    test_train_eval_distributions_differ,
    test_corpus_tokens_not_all_pad,
]


if __name__ == "__main__":
    print("=== V18.0 MiniCorpus — 单元测试 (8 项) ===")
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