"""V25.0 — RealCorpus + CentralTheory 跨版本集成单元测试 (10 项)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 端到端用例会经 v25 layer 惰性导入 examples/run_v8_full (其依赖 experiment_env),
# 故需把 examples/ 加入 import 路径。
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))

import math
import statistics

import numpy as np
import torch
import torch.nn.functional as F

from research._primitives.real_corpus import MiniCorpus
from research._primitives.central_theory import (
    CentralTheory, freeze_u, set_u_init_scale, diagnose_theory_layer,
)
from research.corpus_scale.v25_corpus_central import (
    RealCorpusTheoryLayer,
    build_v25_param_groups,
    prepare_real_corpus_data,
    prepare_held_out_data,
)


def _make_stub_corpus():
    """构造一个不依赖真实 tokenizer 的 stub MiniCorpus (用 _MINI_CORPUS 直接赋)."""
    from research._primitives.real_corpus.corpus import _MINI_CORPUS, _HELD_OUT
    bert_tok = None  # 仅用于构造 MiniCorpus 对象, 不实际调用 tokenizer
    corpus = MiniCorpus.__new__(MiniCorpus)
    corpus.tokenizer = bert_tok
    corpus.max_length = 16
    corpus.train_sentences = _MINI_CORPUS
    corpus.held_out = _HELD_OUT
    return corpus


def _make_random_attn_pools(d_shared=256):
    """构造 stub AttnPool 列表 (3 个)."""
    from research._primitives.attention import AttnPool
    pools = []
    for D_m in [312, 768, 192]:
        w_q = torch.randn(D_m, D_m) * 0.02
        w_k = torch.randn(D_m, D_m) * 0.02
        w_v = torch.randn(D_m, D_m) * 0.02
        w_o_native = torch.eye(D_m)
        ap = AttnPool(D_m=D_m, D_shared=d_shared, num_heads=4,
                       w_q=w_q, w_k=w_k, w_v=w_v, w_o_native=w_o_native)
        pools.append(ap)
    return pools


class _StubTextTok:
    """文本 tokenizer stub: 只接受 sentences 列表, 返回 input_ids."""
    pad_token_id = 0
    def __call__(self, sentences=None, **kw):
        if sentences is None and "sentences" not in kw:
            sentences = ["stub"]
        if not isinstance(sentences, list):
            sentences = [sentences]
        return {"input_ids": torch.randint(0, 100, (len(sentences), 16))}


class _StubVitProc:
    """图像 processor stub: 只接受 images, 返回 pixel_values."""
    def __call__(self, images=None, **kw):
        return {"pixel_values": torch.randn(1, 3, 224, 224)}


# ---------------------------------------------------------------------------
# 单测 (10 项)
# ---------------------------------------------------------------------------


def test_v25_imports():
    """1. v25 公开 API 全部 import OK."""
    from research.corpus_scale.v25_corpus_central import (
        RealCorpusTheoryLayer,
        build_v25_param_groups,
        prepare_real_corpus_data,
        prepare_held_out_data,
    )
    assert RealCorpusTheoryLayer is not None
    assert build_v25_param_groups is not None
    assert prepare_real_corpus_data is not None
    assert prepare_held_out_data is not None
    print("[PASS] test_v25_imports")
    return True


def test_mini_corpus_train_sentences_shape_compatible():
    """2. MiniCorpus.train_sentences 数量与 v25 期望的 [6, S, D_m] 兼容."""
    corpus = _make_stub_corpus()
    assert len(corpus.train_sentences) >= 6, f"train_sentences 至少 6 句, 实测 {len(corpus.train_sentences)}"
    assert len(corpus.held_out) >= 6, f"held_out 至少 6 句, 实测 {len(corpus.held_out)}"
    # 每句长度合理 (8-20 词)
    for s in corpus.train_sentences[:3]:
        assert 5 <= len(s.split()) <= 20
    print(f"[PASS] test_mini_corpus_train_sentences_shape_compatible (train={len(corpus.train_sentences)}, held_out={len(corpus.held_out)})")
    return True


def test_real_corpus_theory_layer_all_4_modes_forward():
    """3. RealCorpusTheoryLayer 4 mode 都能 forward (输出 shape 一致)."""
    pools = _make_random_attn_pools()
    for mode in ["broadcast", "gate", "router-norm", "adaptive-ema"]:
        layer = RealCorpusTheoryLayer(
            d_shared=256, num_experts=3,
            modal_dims=[312, 768, 192], attn_pools=pools, mode=mode,
        )
        # Stub 输入: [6, 16, D_m] per modality
        h_text = torch.randn(6, 16, 312)
        h_code = torch.randn(6, 16, 768)
        h_img = torch.randn(6, 16, 192)
        y = layer([h_text, h_code, h_img])
        assert y.shape == (6, 16, 256), f"{mode}: shape 应 = (6, 16, 256), 实测 {y.shape}"
    print("[PASS] test_real_corpus_theory_layer_all_4_modes_forward")
    return True


def test_real_corpus_theory_layer_all_4_modes_backward():
    """4. 4 mode 都能 backward (梯度流到关键参数).

    注意: c 在初始为零时梯度路径依赖其模式, 这里主要验证 W_router / V_coop / attn W_o / U (非 router-norm freeze 后)。
    """
    pools = _make_random_attn_pools()
    for mode in ["broadcast", "gate", "router-norm", "adaptive-ema"]:
        layer = RealCorpusTheoryLayer(
            d_shared=256, num_experts=3,
            modal_dims=[312, 768, 192], attn_pools=pools, mode=mode,
        )
        h_text = torch.randn(6, 16, 312)
        h_code = torch.randn(6, 16, 768)
        h_img = torch.randn(6, 16, 192)
        y = layer([h_text, h_code, h_img])
        target = torch.zeros_like(y)
        loss = F.mse_loss(y, target)
        loss.backward()
        # V_coop 必有梯度 (C-2 协同矩阵)
        assert layer.V_coop.grad is not None and layer.V_coop.grad.abs().sum() > 0, \
            f"{mode}: V_coop.grad 异常"
        # attn W_o 必有梯度
        assert layer.attn_pools[0].W_o.grad is not None and layer.attn_pools[0].W_o.grad.abs().sum() > 0, \
            f"{mode}: attn W_o.grad 异常"
        # router (W_router_attn / W_router_ffn) 必有梯度
        assert layer.W_router_attn.grad is not None and layer.W_router_attn.grad.abs().sum() > 0, \
            f"{mode}: W_router_attn.grad 异常"
        assert layer.W_router_ffn.grad is not None and layer.W_router_ffn.grad.abs().sum() > 0, \
            f"{mode}: W_router_ffn.grad 异常"
    print("[PASS] test_real_corpus_theory_layer_all_4_modes_backward")
    return True


def test_freeze_u_still_works_in_real_corpus_layer():
    """5. freeze_u 在 RealCorpusTheoryLayer 内仍生效 (router-norm 模式)."""
    pools = _make_random_attn_pools()
    layer = RealCorpusTheoryLayer(
        d_shared=256, num_experts=3,
        modal_dims=[312, 768, 192], attn_pools=pools, mode="router-norm",
    )
    # 默认应已 freeze_u (DEFAULT_FREEZE_U_FOR_ROUTER_NORM=True)
    assert layer.cb_attn.U.requires_grad is False, "router-norm 默认 freeze_u"
    assert layer.cb_ffn.U.requires_grad is False
    print("[PASS] test_freeze_u_still_works_in_real_corpus_layer")
    return True


def test_build_v25_param_groups_has_per_expert_groups():
    """6. build_v25_param_groups 产出与 v22 一致的 per-expert param group 结构."""
    pools = _make_random_attn_pools()
    layer = RealCorpusTheoryLayer(
        d_shared=256, num_experts=3,
        modal_dims=[312, 768, 192], attn_pools=pools, mode="gate",
    )
    groups = build_v25_param_groups(layer)
    expert_tags = [g["expert"] for g in groups]
    # 期望包含: router_attn, attn_0/1/2, router_ffn, ffn_0/1/2_adapter, ffn_0/1/2_alpha, central_workspace, central_coop
    assert "router_attn" in expert_tags
    assert "router_ffn" in expert_tags
    assert all(f"attn_{i}" in expert_tags for i in range(3))
    assert all(f"ffn_{i}_adapter" in expert_tags for i in range(3))
    assert all(f"ffn_{i}_alpha" in expert_tags for i in range(3))
    assert "central_workspace" in expert_tags
    assert "central_coop" in expert_tags
    # 学习率隔离
    lrs = {g["expert"]: g["lr"] for g in groups}
    assert lrs["router_attn"] == 1e-4
    assert lrs["attn_0"] == 1e-2
    assert lrs["ffn_0_alpha"] == 1e-3
    assert lrs["central_workspace"] == 5e-3
    assert lrs["central_coop"] == 5e-3
    print(f"[PASS] test_build_v25_param_groups_has_per_expert_groups ({len(groups)} groups)")
    return True


def test_ema_control_train_vs_eval():
    """7. EMA 控制: 训练时更新, eval 时不更新 (对齐 v8._ema_enabled 语义)."""
    pools = _make_random_attn_pools()
    layer = RealCorpusTheoryLayer(
        d_shared=256, num_experts=3,
        modal_dims=[312, 768, 192], attn_pools=pools, mode="router-norm",
    )
    h_text = torch.randn(6, 16, 312)
    h_code = torch.randn(6, 16, 768)
    h_img = torch.randn(6, 16, 192)
    # 训练: c 应变化
    layer.train()
    c_before = layer.cb_attn.c.data.clone()
    _ = layer([h_text, h_code, h_img])
    c_after_train = layer.cb_attn.c.data.clone()
    assert not torch.allclose(c_after_train, c_before), "训练时 c 应更新"
    # eval: c 不变
    layer.eval()
    c_before_eval = layer.cb_attn.c.data.clone()
    with torch.no_grad():
        _ = layer([h_text, h_code, h_img])
    c_after_eval = layer.cb_attn.c.data.clone()
    assert torch.allclose(c_after_eval, c_before_eval), "eval 时 c 不应更新"
    print("[PASS] test_ema_control_train_vs_eval")
    return True


def test_prepare_real_corpus_data_returns_correct_structure():
    """8. prepare_real_corpus_data 返回结构正确 (18 个样本, 6 per class)."""
    # Stub encoders: 用专用 _StubTextTok / _StubVitProc
    class _StubModel:
        def __call__(self, input_ids):
            class _O:
                last_hidden_state = torch.randn(input_ids.shape[0], 16, 312)
            return _O()
    encoders = (
        (_StubTextTok(), _StubModel(), None, None),  # bert (text)
        (_StubTextTok(), _StubModel(), None, None),  # llama (text/code 共享 token 接口)
        (_StubVitProc(), _StubModel(), None, None),  # vit (image)
    )
    corpus = _make_stub_corpus()
    modal_seqs, modal_indices, targets = prepare_real_corpus_data(
        seed=0, encoders=encoders, corpus=corpus,
        n_per_class=6, S=16, D_SHARED=256, N_CLS=3,
    )
    assert len(modal_seqs) == 18, f"应 18 个 modal_seqs, 实测 {len(modal_seqs)}"
    assert len(modal_indices) == 18
    assert all(0 <= m <= 2 for m in modal_indices)
    # 6 per class
    for cls in range(3):
        assert sum(1 for m in modal_indices if m == cls) == 6
    # targets shape
    assert all(t.shape == (16, 256) for t in targets)
    print(f"[PASS] test_prepare_real_corpus_data_returns_correct_structure")
    return True


def test_prepare_held_out_data_returns_correct_structure():
    """9. prepare_held_out_data 返回结构正确 (held-out 6 句重复到 18 个样本)."""
    class _StubModel:
        def __call__(self, input_ids):
            class _O:
                last_hidden_state = torch.randn(input_ids.shape[0], 16, 192)
            return _O()
    encoders = (
        (_StubTextTok(), _StubModel(), None, None),
        (_StubTextTok(), _StubModel(), None, None),
        (_StubVitProc(), _StubModel(), None, None),
    )
    corpus = _make_stub_corpus()
    modal_seqs, modal_indices, targets = prepare_held_out_data(
        encoders=encoders, corpus=corpus,
        S=16, D_SHARED=256, N_CLS=3, n_per_class=6,
    )
    assert len(modal_seqs) == 18
    assert len(modal_indices) == 18
    # held-out 句子应与 train_sentences 不重叠 (MiniCorpus 设计保证)
    for sent in [corpus.held_out[0]]:
        assert sent not in corpus.train_sentences, "held_out 应与 train_sentences 不重叠"
    print(f"[PASS] test_prepare_held_out_data_returns_correct_structure")
    return True


def test_end_to_end_smoke_run_produces_finite_fuse():
    """10. 端到端冒烟: stub encoders + 1 seed × 4 mode 全跑通, fuse MSE 有限."""
    class _StubBertModel:
        def __call__(self, input_ids):
            class _O:
                last_hidden_state = torch.randn(input_ids.shape[0], 16, 312) * 0.1
            return _O()
    class _StubLlamaModel:
        def __call__(self, input_ids):
            class _O:
                last_hidden_state = torch.randn(input_ids.shape[0], 16, 768) * 0.1
            return _O()
    class _StubVitModel:
        def __call__(self, pixel_values):
            class _O:
                last_hidden_state = torch.randn(1, 197, 192) * 0.1
            return _O()

    encoders = (
        (_StubTextTok(), _StubBertModel(), 312, None),
        (_StubTextTok(), _StubLlamaModel(), 768, None),
        (_StubVitProc(), _StubVitModel(), 192, None),
    )
    pools = _make_random_attn_pools()
    corpus = _make_stub_corpus()

    # 只跑 1 seed × 4 mode × 5 STEPS (冒烟, 不跑完整 100 step)
    results = {}
    for mode in ["broadcast", "gate", "router-norm", "adaptive-ema"]:
        torch.manual_seed(0)
        modal_seqs, modal_indices, targets = prepare_real_corpus_data(
            seed=0, encoders=encoders, corpus=corpus,
            n_per_class=6, S=16, D_SHARED=256, N_CLS=3,
        )
        h_by_modal = [[] for _ in range(3)]
        t_by_modal = [[] for _ in range(3)]
        for i, m in enumerate(modal_indices):
            h_by_modal[m].append(modal_seqs[i])
            t_by_modal[m].append(targets[i])
        for cls in range(3):
            h_by_modal[cls] = torch.stack(h_by_modal[cls], dim=0)
            t_by_modal[cls] = torch.stack(t_by_modal[cls], dim=0)

        layer = RealCorpusTheoryLayer(
            d_shared=256, num_experts=3,
            modal_dims=[312, 768, 192], attn_pools=pools, mode=mode,
        )
        opt = torch.optim.AdamW(build_v25_param_groups(layer), lr=1e-2)
        for _ in range(5):
            opt.zero_grad(set_to_none=True)
            y = layer(h_by_modal)
            loss = sum(F.mse_loss(y, t_by_modal[c]) for c in range(3)) / 3
            loss.backward()
            torch.nn.utils.clip_grad_norm_(layer.parameters(), 1.0)
            opt.step()
        # 评估
        layer.eval()
        with torch.no_grad():
            y = layer(h_by_modal)
            fuse = 0.0
            for c in range(3):
                for s in range(6):
                    fuse += F.mse_loss(y[s:s+1], t_by_modal[c][s:s+1]).item()
            fuse /= 18
        results[mode] = fuse
        assert math.isfinite(fuse), f"{mode}: fuse 应有限, 实测 {fuse}"
    print(f"[PASS] test_end_to_end_smoke_run_produces_finite_fuse ({results})")
    return True


TESTS = [
    test_v25_imports,
    test_mini_corpus_train_sentences_shape_compatible,
    test_real_corpus_theory_layer_all_4_modes_forward,
    test_real_corpus_theory_layer_all_4_modes_backward,
    test_freeze_u_still_works_in_real_corpus_layer,
    test_build_v25_param_groups_has_per_expert_groups,
    test_ema_control_train_vs_eval,
    test_prepare_real_corpus_data_returns_correct_structure,
    test_prepare_held_out_data_returns_correct_structure,
    test_end_to_end_smoke_run_produces_finite_fuse,
]


if __name__ == "__main__":
    print("=== V25.0 RealCorpus + CentralTheory — 单元测试 (10 项) ===")
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
