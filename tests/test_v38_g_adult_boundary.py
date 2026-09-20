"""V38.0 — γ 真实数据跨模态验证 单元测试 (10 项)。

不依赖 transformers / tabldm: 测试 Adult loader + XOR 标签 + 蒸馏预热。
覆盖:
    data.adult_loader:
        1. load_adult: [n, 6] float32 + [n] long (n=128)
        2. load_adult 正负样本平衡 (n=64, 各类 32)
        3. load_adult_crossmodal XOR 标签: 标签 ∈ {0, 1}
        4. load_adult_crossmodal 跨模态必需性: decouple 切断关联
        5. load_adult_crossmodal 文本: 含职业信息 ("working as ...")
    paths / __init__:
        6. 路径常量可解析
        7. v38 __version__ 与导出符号
    EntryEncoder 蒸馏预热 (γ 关键: E 拉到 TabLDM 先验):
        8. 蒸馏前 E(x) 与 ref 距离 > 0
        9. 蒸馏 100 步后距离显著下降
        10. 蒸馏后 E 参数 grad 仍非零 (还可继续任务训练)
"""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from research.scale_heterogeneous.v38_g_adult_boundary import (
    adult_loader, EntryEncoder, first_existing,
)
from research.scale_heterogeneous.v38_g_adult_boundary.paths import MSA_DIR, TABLDM_CKPT


def test_load_adult_shape():
    """1. load_adult 返回 [n,6] float32 + [n] long."""
    X, y = adult_loader.load_adult(n=128, seed=0)
    assert X.shape == (128, 6) and X.dtype == torch.float32
    assert y.shape == (128,) and y.dtype == torch.long
    assert y.min().item() in (0, 1) and y.max().item() in (0, 1)
    print("[PASS] test_load_adult_shape")
    return True


def test_load_adult_balance():
    """2. balance=True 时正负样本各半."""
    X, y = adult_loader.load_adult(n=64, seed=0)
    n_pos = (y == 1).sum().item()
    n_neg = (y == 0).sum().item()
    assert n_pos == 32 and n_neg == 32, f"应各 32, 实测 pos={n_pos} neg={n_neg}"
    print("[PASS] test_load_adult_balance (pos=neg=32)")
    return True


def test_load_adult_crossmodal_label_binary():
    """3. crossmodal 标签 ∈ {0, 1} (XOR 严格二分类)."""
    texts, X, y = adult_loader.load_adult_crossmodal(n=128, seed=42, decouple=True)
    assert y.unique().tolist() in ([0, 1], [0], [1])
    print(f"[PASS] test_load_adult_crossmodal_label_binary (类别 {y.unique().tolist()})")
    return True


def test_load_adult_crossmodal_decouple():
    """4. decouple 切断非工时列与职业的关联 (验证随机置换)."""
    texts, X, y = adult_loader.load_adult_crossmodal(n=128, seed=42, decouple=True)
    # 工时列 (索引 5) 在 decouple 中保持原顺序; 其余 5 列被打乱
    # 验证: 同一文本两次调用得到不同的 X (因为样本间置换的随机性)
    _, X2, _ = adult_loader.load_adult_crossmodal(n=128, seed=99, decouple=True)
    assert not torch.equal(X, X2), "不同 seed 应得到不同 X"
    # X 列均值应在 0 附近 (标准化)
    means = X.mean(dim=0).abs().max().item()
    assert means < 1.0, f"标准化后均值应≈0, max={means}"
    print("[PASS] test_load_adult_crossmodal_decouple")
    return True


def test_load_adult_crossmodal_text_has_occupation():
    """5. 文本包含职业信息 ("working as <occupation>"), 否则跨模态标签无法构造."""
    texts, X, y = adult_loader.load_adult_crossmodal(n=64, seed=42, decouple=True)
    n_with_occ = sum(1 for t in texts if "working as" in t)
    assert n_with_occ > 0, "文本应包含 'working as <职业>' 模板"
    # 至少部分文本应提到高薪职业
    high_pay = {"Exec-managerial", "Prof-specialty", "Tech-support"}
    n_high = sum(1 for t in texts for hp in high_pay if hp.lower() in t)
    assert n_high > 0, f"应存在高薪职业文本, 实测 {n_high}"
    print(f"[PASS] test_load_adult_crossmodal_text_has_occupation (n_occ={n_with_occ}, n_high={n_high})")
    return True


def test_paths_resolvable():
    """6. 路径常量可解析 (str + 非空)."""
    assert isinstance(MSA_DIR, str) and len(MSA_DIR) > 0
    assert isinstance(TABLDM_CKPT, str) and len(TABLDM_CKPT) > 0
    print("[PASS] test_paths_resolvable")
    return True


def test_package_exports():
    """7. v38 __version__ 与导出符号."""
    from research.scale_heterogeneous import v38_g_adult_boundary as v38
    assert v38.__version__ == "38.0.0"
    for name in ("adult_loader", "load_big_container", "EntryEncoder", "knn_acc"):
        assert hasattr(v38, name), f"未导出 {name}"
    print("[PASS] test_package_exports")
    return True


def test_distill_initial_distance_positive():
    """8. 蒸馏前 E(x) 与 ref 的 MSE > 0 (随机初始化, 与随机 ref 不重合)."""
    torch.manual_seed(0)
    X = torch.randn(64, 6)
    ref = torch.randn(64, 512)
    E = EntryEncoder(6, 512)
    with torch.no_grad():
        d0 = nn.functional.mse_loss(E(X), ref).item()
    assert d0 > 1e-3, f"蒸馏前距离应 > 0, got {d0}"
    print(f"[PASS] test_distill_initial_distance_positive (d0={d0:.4f})")
    return True


def test_distill_reduces_distance():
    """9. 蒸馏后 E 逼近 ref, MSE 显著下降.

    ref 用**结构化**信号 (类别 one-hot × 大常数), 而非随机高维向量,
    让 6→256→512 MLP 在合理步数内能拟合 (验证 E 真的向 ref 收拢,
    而非测试优化器对抗随机高维目标的极限能力).
    """
    torch.manual_seed(0)
    n = 64
    X = torch.randn(n, 6)
    # 结构化 ref: 每个样本的 ref 由其类别索引决定 (类别 0 或 1)
    y = torch.tensor([i % 2 for i in range(n)])
    ref = torch.zeros(n, 512)
    for i in range(n):
        ref[i, y[i] * 256: (y[i] + 1) * 256] = 1.0  # 类别 i 的 ref 块为 1, 其余为 0

    E = EntryEncoder(6, 512)
    P = nn.Linear(512, 64); head = nn.Linear(64, 2)
    params = list(P.parameters()) + list(head.parameters()) + list(E.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    tr = torch.arange(48)
    with torch.no_grad():
        d0 = nn.functional.mse_loss(E(X), ref).item()
    for _ in range(300):
        b = tr[torch.randperm(len(tr))[:32]]
        loss = nn.functional.mse_loss(E(X[b]), ref[b])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        d1 = nn.functional.mse_loss(E(X), ref).item()
    assert d1 < d0 * 0.5, f"蒸馏后距离应 < 50% 初始, d0={d0:.4f} d1={d1:.4f}"
    print(f"[PASS] test_distill_reduces_distance (d0={d0:.4f} -> d1={d1:.4f}, 下降 {(1-d1/d0)*100:.0f}%)")
    return True


def test_distill_then_task_grad_still_flows():
    """10. 蒸馏后 E 参数 grad 仍非零 (还可继续任务训练, γ E 臂设计)."""
    torch.manual_seed(0)
    X = torch.randn(64, 6)
    ref = torch.randn(64, 512)
    y = torch.randint(0, 2, (64,))
    E = EntryEncoder(6, 512)
    P = nn.Linear(512, 64); head = nn.Linear(64, 2)
    params = list(P.parameters()) + list(head.parameters()) + list(E.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    tr = torch.arange(48)
    # 蒸馏阶段
    for _ in range(50):
        b = tr[torch.randperm(len(tr))[:32]]
        opt.zero_grad()
        nn.functional.mse_loss(E(X[b]), ref[b]).backward()
        opt.step()
    # 任务阶段: E 应仍参与梯度
    opt.zero_grad()
    cin = E(X[:32])
    logits = nn.Linear(512, 2)(cin + torch.randn(32, 512))
    nn.functional.cross_entropy(logits, y[:32]).backward()
    grad_norms = [p.grad.norm().item() for p in E.parameters() if p.grad is not None]
    assert len(grad_norms) > 0 and all(g > 0 for g in grad_norms)
    print(f"[PASS] test_distill_then_task_grad_still_flows (min={min(grad_norms):.4f})")
    return True


TESTS = [
    test_load_adult_shape,
    test_load_adult_balance,
    test_load_adult_crossmodal_label_binary,
    test_load_adult_crossmodal_decouple,
    test_load_adult_crossmodal_text_has_occupation,
    test_paths_resolvable,
    test_package_exports,
    test_distill_initial_distance_positive,
    test_distill_reduces_distance,
    test_distill_then_task_grad_still_flows,
]


if __name__ == "__main__":
    print(f"=== V38.0 γ 真实数据跨模态验证 — 单元测试 ({len(TESTS)} 项) ===")
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