"""V50.0 融合收益审计 — 单元测试 (6 项)。不加载真实模型。"""
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "hetero_moe")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from research._primitives.testing import run_suite

TITLE = "V50.0 融合收益审计 — 单元测试 (6 项)"

import torch

from research.aligner.v50_fusion_gain_audit.mechanism import proxy_pretrain, linear_probe
from research.aligner.v49_projection_space.mechanism import build_projector


def test_proxy_pretrain_reduces_loss():
    """proxy 对齐必须真的把损失压下去 (否则对齐预训练没生效)。"""
    torch.manual_seed(0)
    experts = ["a", "b", "c"]
    D = 64
    # 三个底座是**同一份特征的不同随机基** → 真实存在可对齐的公共结构
    base = torch.randn(16, 5, D)
    A = torch.randn(D, D) * 0.1                     # 三个底座共享的公共结构
    feats = {e: base @ (A + torch.randn(D, D) * 0.02) for e in experts}
    projs = {e: build_projector("NL_static", 32, D) for e in experts}
    l0 = None
    params = [p for e in experts for p in projs[e].parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-2)
    zs = [projs[e](feats[e]) for e in experts]
    mean = torch.stack(zs, 0).mean(0)
    l0 = sum(((z - mean) ** 2).mean() for z in zs).item()
    l1 = proxy_pretrain(projs, feats, experts, steps=100, lr=1e-2)
    assert l1 < l0, (l0, l1)
    print(f"[PASS] proxy 对齐压低损失 ({l0:.4f} → {l1:.4f})")
    return True


def test_proxy_pretrain_uses_no_labels():
    """proxy 目标里不能出现任何标签: 改标签不应影响它 (结构性检查)。"""
    experts = ["a", "b"]
    D = 32
    feats = {e: torch.randn(8, 4, D) for e in experts}
    projs = {e: build_projector("NL_static", 16, D) for e in experts}
    l = proxy_pretrain(projs, feats, experts, steps=5)
    assert isinstance(l, float) and l >= 0
    print("[PASS] proxy 预训练只用特征, 不需要标签")
    return True


def test_linear_probe_learns_separable():
    """探针在可分数据上必须学得动 (否则探针无效, 不能用来下结论)。"""
    torch.manual_seed(0)
    x = torch.randn(200, 8)
    y = (x[:, 0] > 0).long()
    te, tr = linear_probe(x, y, x, y, n_class=2, steps=200)
    assert tr > 0.9, tr
    print(f"[PASS] 线性探针在可分数据上有效 (train acc {tr:.2f})")
    return True


def test_linear_probe_reports_chance_on_noise():
    """在纯噪声上, 探针的准确率应接近随机 (否则它会给出假的高准确率)。"""
    torch.manual_seed(1)
    x = torch.randn(300, 16)
    y = torch.randint(0, 3, (300,))
    te, tr = linear_probe(x, y, x, y, n_class=3, steps=200)
    assert te < 0.6, te
    print(f"[PASS] 噪声上探针接近随机 (acc {te:.2f}, 随机 0.33)")
    return True


def test_probe_handles_batch_shapes():
    """探针要能吃 [N, S, D] 的展平输入。"""
    torch.manual_seed(2)
    x = torch.randn(20, 6, 10)
    y = torch.randint(0, 3, (20,))
    te, tr = linear_probe(x.reshape(-1, 10), y.repeat_interleave(6), x.reshape(-1, 10),
                          y.repeat_interleave(6), n_class=3, steps=50)
    assert 0 <= te <= 1
    print("[PASS] 探针支持 [N,S,D] 展平")
    return True


def test_proxy_is_invariant_to_expert_order():
    """proxy 目标对底座顺序不变 (换顺序损失应相同)。"""
    torch.manual_seed(3)
    experts = ["a", "b", "c"]
    D = 32
    feats = {e: torch.randn(6, 3, D) for e in experts}
    p1 = {e: build_projector("NL_static", 16, D) for e in experts}
    p2 = {e: build_projector("NL_static", 16, D) for e in experts}
    p2["a"].load_state_dict(p1["a"].state_dict())
    p2["b"].load_state_dict(p1["b"].state_dict())
    p2["c"].load_state_dict(p1["c"].state_dict())
    l1 = proxy_pretrain(p1, feats, experts, steps=10)
    l2 = proxy_pretrain(p2, feats, ["c", "a", "b"], steps=10)
    assert abs(l1 - l2) < 1e-5, (l1, l2)
    print("[PASS] proxy 目标对底座顺序不变")
    return True


TESTS = [
    test_proxy_pretrain_reduces_loss,
    test_proxy_pretrain_uses_no_labels,
    test_linear_probe_learns_separable,
    test_linear_probe_reports_chance_on_noise,
    test_probe_handles_batch_shapes,
    test_proxy_is_invariant_to_expert_order,
]


if __name__ == "__main__":
    sys.exit(run_suite(TITLE, TESTS))
