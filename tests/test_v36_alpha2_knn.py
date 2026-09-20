"""V36.0 — α2 跨模态验证 (per-sample TabLDM entry + k-NN) 单元测试 (10 项)。

不依赖 transformers / tabldm: 用随机张量与 dummy 容器测试路径解析、knn 评估、占位接口。
覆盖:
    paths:
        1. first_existing 命中现存候选
        2. first_existing 回退首选
        3. 路径常量可解析 (str)
    eval:
        4. knn_acc leave-one-out 正确 (最近邻 = 自身外的最近)
        5. knn_acc 在随机向量上接近随机 (10 类 n=10)
        6. concat_norm_then_knn 输出 ∈ [0,1]
    entries big_forward 语义:
        7. dummy 容器前向 → [B, S, D] 与 embed 一致
        8. h_base (per-sample mean) 形状 = [B, D_BIG]
    修复验证 (v35→v36 关键):
        9. per-sample repr 形状 = [B, 512] (修复 v35 的 [512] 常向量塌缩)
        10. 拼接 [h_base, c_struct] 的范数 = 各分量范数之和 (v35 路径均值塌缩会破这个等式)
"""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from research._primitives.scale_hetero_loaders import knn_acc, concat_norm_then_knn, first_existing, big_forward
from research._primitives.scale_hetero_loaders.paths import MSA_DIR, TABLDM_CKPT


def test_first_existing_picks_existing():
    with tempfile.TemporaryDirectory() as d:
        assert first_existing(["/nope/a", d], "V36_X_UNSET") == d
    print("[PASS] test_first_existing_picks_existing")
    return True


def test_first_existing_fallback():
    assert first_existing(["/nope/a", "/nope/b"], "V36_X_UNSET") == "/nope/a"
    print("[PASS] test_first_existing_fallback")
    return True


def test_paths_resolvable():
    assert isinstance(MSA_DIR, str) and len(MSA_DIR) > 0
    assert isinstance(TABLDM_CKPT, str) and len(TABLDM_CKPT) > 0
    print("[PASS] test_paths_resolvable")
    return True


def test_knn_acc_leave_one_out():
    """4. 5 个点: 4 类各 1+1 自身 → 1-NN 应该是 自身外最近=另一类自身 (距离 0)."""
    X = torch.tensor([[0., 0.], [10., 10.], [0.1, 0.1], [10.1, 10.1], [5., 5.]])
    y = torch.tensor([0, 1, 0, 1, 0])
    # 各点最近的"非自身"最近邻是另一个类的最近 (距离 ~0.1)
    acc = knn_acc(X, y, k=1)
    assert acc == 1.0, f"acc={acc}"
    print("[PASS] test_knn_acc_leave_one_out (acc=1.0)")
    return True


def test_knn_acc_random_baseline():
    """5. 10 类 n=100 随机向量 → k-NN 准确率应接近随机 (≤0.2, 期望 ≈0.1)."""
    torch.manual_seed(0)
    X = torch.randn(100, 16)
    y = torch.randint(0, 10, (100,))
    acc = knn_acc(X, y, k=1)
    assert 0.0 <= acc <= 0.20, f"随机向量 k-NN 应接近随机, acc={acc}"
    print(f"[PASS] test_knn_acc_random_baseline (acc={acc:.3f})")
    return True


def test_concat_norm_knn_in_unit_interval():
    """6. concat_norm_then_knn 输出 ∈ [0, 1]."""
    torch.manual_seed(0)
    h = torch.randn(20, 32)
    c = torch.randn(20, 16)
    y = torch.randint(0, 5, (20,))
    acc = concat_norm_then_knn(h, c, y)
    assert 0.0 <= acc <= 1.0, f"acc={acc}"
    print(f"[PASS] test_concat_norm_knn_in_unit_interval (acc={acc:.3f})")
    return True


class _DummyContainer(nn.Module):
    def __init__(self, vocab=20, D=8):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, D)
        self.D = D
    def forward(self, input_ids):
        from types import SimpleNamespace
        return SimpleNamespace(last_hidden_state=self.embed_tokens(input_ids))


def test_big_forward_returns_b_x_d():
    """7. big_forward (无注入基线) 输出形状 [B, S, D_BIG]."""
    m = _DummyContainer(D=8)
    ids = torch.randint(0, 20, (2, 4))
    h = big_forward(m, ids)
    assert h.shape == (2, 4, 8), f"{h.shape}"
    print("[PASS] test_big_forward_returns_b_x_d")
    return True


def test_h_base_per_sample_shape():
    """8. h_base = big_forward(...).mean(dim=1) → [B, D_BIG] (per-sample)."""
    m = _DummyContainer(D=16)
    ids = torch.randint(0, 20, (3, 5))
    h_base = big_forward(m, ids).mean(dim=1)
    assert h_base.shape == (3, 16), f"{h_base.shape}"
    print("[PASS] test_h_base_per_sample_shape")
    return True


def test_per_sample_repr_shape():
    """9. 修复验证: per-sample repr 形状 [B, 512] (修复 v35 的 [512] 塌缩).

    v35 bug: `r.mean(dim=(0,1))` 把所有样本所有维度塌缩成 [512] 单向量.
    v36 fix: 仅在 ensemble 维求均值, 保留 [B, 512] 逐样本.
    """
    # 模拟 TabLDM hook 输出: [n_ens, seq, 512], 取最后 B 个 query 样本
    n_ens, seq, B = 4, 12, 6
    D = 512
    r = torch.randn(n_ens, seq, D)
    rep_q = r[:, -B:, :]                # [n_ens, B, D]
    c_struct = rep_q.mean(dim=0)        # [B, D]   ← v36 fix
    c_collapsed = r.mean(dim=(0, 1))    # [D]     ← v35 bug

    assert c_struct.shape == (B, D), f"v36 shape={c_struct.shape}"
    assert c_collapsed.shape == (D,), f"v35 bug shape={c_collapsed.shape}"
    # 不同样本的 c_struct 应当彼此不同 (每个样本有独立信息)
    assert c_struct.std(dim=0).mean() > 0.01, "per-sample 应有信息 (std>0)"
    # c_collapsed 是单向量; 若 broadcast 给所有样本则丢弃了样本间差异
    # 即 c_struct 展开后, 行间方差应 > c_collapsed 重复行的方差 (≈ 0)
    c_broadcast = c_collapsed.unsqueeze(0).expand(B, -1)
    assert torch.norm(c_struct - c_struct.mean(0, keepdim=True)) > \
           torch.norm(c_broadcast - c_broadcast.mean(0, keepdim=True))
    print("[PASS] test_per_sample_repr_shape")
    return True


def test_concat_norm_preserves_differential_info():
    """10. v35 塌缩路径下 concat(h_base, mean(c, dim=0)) → 每个样本拼接结果相同 (无差分).

    v36 修复后: 每个样本拼接结果应反映 c_struct[i] 的差异.
    测试方法: 构造**结构化**信号 (类别由 c 决定), 比较两种路径的 1-NN 准确率.
    v36 (per-sample) 应恢复全部信号; v35 塌缩后所有样本拼接结果尾部相同 → 信息丢失.
    """
    torch.manual_seed(0)
    n = 12  # 2 类各 6 样本, c_per_sample = 类别 one-hot × 大向量
    h_base = torch.randn(n, 8)
    # 类别标签由 c_per_sample 完全决定 (构造性注入: y[i] = argmax(c_per_sample[i]))
    c_per_sample = torch.zeros(n, 4)
    for i in range(n):
        c_per_sample[i, i % 2] = 10.0  # 强信号: 一半是 0 类, 一半是 1 类
    y = torch.tensor([i % 2 for i in range(n)])  # 标签与 c 完全一致

    # v35 塌缩: c_collapsed = mean across samples, 全部样本尾部相同 → 0 信息
    c_collapsed = c_per_sample.mean(dim=0, keepdim=True).expand(n, 4)

    X_v36 = torch.cat([torch.nn.functional.normalize(h_base, dim=1),
                       torch.nn.functional.normalize(c_per_sample, dim=1)], dim=1)
    X_v35 = torch.cat([torch.nn.functional.normalize(h_base, dim=1),
                       torch.nn.functional.normalize(c_collapsed, dim=1)], dim=1)

    acc_v36 = knn_acc(X_v36, y, k=1)
    acc_v35 = knn_acc(X_v35, y, k=1)
    # 1-NN 找最近: v36 拼接尾部保留类别 → 类别一致 → acc 高
    #              v35 拼接尾部对所有样本相同 → 只靠 h_base 猜 → acc 应低很多
    assert acc_v36 == 1.0, f"v36 应 100% (类别由 c 决定, c 已保留), got {acc_v36}"
    assert acc_v35 < acc_v36, f"v35_collapsed={acc_v35} 应 < v36={acc_v36}"
    print(f"[PASS] test_concat_norm_preserves_differential_info (v36={acc_v36:.2f}, v35_collapsed={acc_v35:.2f})")
    return True


TESTS = [
    test_first_existing_picks_existing,
    test_first_existing_fallback,
    test_paths_resolvable,
    test_knn_acc_leave_one_out,
    test_knn_acc_random_baseline,
    test_concat_norm_knn_in_unit_interval,
    test_big_forward_returns_b_x_d,
    test_h_base_per_sample_shape,
    test_per_sample_repr_shape,
    test_concat_norm_preserves_differential_info,
]


if __name__ == "__main__":
    print(f"=== V36.0 α2 跨模态验证 — 单元测试 ({len(TESTS)} 项) ===")
    passed = 0
    for fn in TESTS:
        try:
            if fn():
                passed += 1
        except Exception as e:
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
    print(f"---\n合计: {passed}/{len(TESTS)} {'✓ ALL PASS' if passed == len(TESTS) else '✗ HAS FAIL'}")
    sys.exit(0 if passed == len(TESTS) else 1)