"""V63.0 — 九条路线执行矩阵 单元测试。

只依赖 torch + numpy；不触碰 v62 包的模型权重（那需要 GPU + 已训练 ckpt）。
本测试验证的是**机制语义**而非精度 —— 本代没有端到端数字（见 results/metrics.json）。

每个 test_* 必须在末尾 return True（run_suite 以返回真值判定通过）。
"""
import importlib.util
import os
import py_compile
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_pkg = os.path.dirname(_here)
_root = _pkg
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "hetero_moe")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch

TITLE = "V63.0 九条路线执行矩阵 — 单元测试"

# ---------------------------------------------------------------- 载入子模块
# v63 的子目录是扁平脚本（非 Python 包），用 importlib 按路径载入
def _load(relpath, name):
    path = os.path.join(_pkg, relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_hard = _load(os.path.join("routing", "hard_router.py"), "v63_hard_router")
_dyn = _load(os.path.join("routing", "dynamic_router.py"), "v63_dynamic_router")
_audit = _load(os.path.join("audit", "audit_v63.py"), "v63_audit")

_B, _S, _K = 2, 16, 4


# ---------------------------------------------------------------- 1
def test_all_nine_scripts_compile():
    """1. 九条路线全部脚本语法可编译（路线 #2/#3 复用 v62 模块，不在本包内）"""
    n = 0
    for sub in ("routing", "robustness", "interpret", "deploy", "audit"):
        d = os.path.join(_pkg, sub)
        for f in os.listdir(d):
            if f.endswith(".py"):
                py_compile.compile(os.path.join(d, f), doraise=True)
                n += 1
    assert n >= 8, "预期至少 8 个脚本，实际 %d" % n
    print("[PASS] test_all_nine_scripts_compile (%d 个脚本编译通过)" % n)
    return True


# ---------------------------------------------------------------- 2
def test_hard_router_sparse_and_normalized():
    """2. HardLoadRouter: top-k 硬稀疏，每 token 恰好 2 槽非零，权重和为 1"""
    r = _hard.HardLoadRouter(n_slots=_K)
    logits = torch.randn(_B, _S, _K)
    gate, aux = r(logits)
    assert gate.shape == (_B, _S, _K), gate.shape
    nz = (gate > 0).sum(-1)
    assert (nz == 2).all(), "top-k=2 应每 token 恰好 2 槽非零，实测 %s" % nz.unique().tolist()
    assert torch.allclose(gate.sum(-1), torch.ones(_B, _S), atol=1e-5)
    print("[PASS] test_hard_router_sparse_and_normalized (硬稀疏 k=2, 和=1)")
    return True


# ---------------------------------------------------------------- 3
def test_hard_router_capacity_penalty_positive_when_imbalanced():
    """3. HardLoadRouter: 负载不均时容量惩罚 > 0，均匀时为 0

    容量约束按批次平均概率算：mean_p > capacity_factor/K 的部分被罚。
    """
    r = _hard.HardLoadRouter(n_slots=_K, capacity_factor=1.25, aux_weight=1.0)
    # 人为偏置：槽 0 压倒性占优 ⇒ 超容量 ⇒ 罚 > 0
    skewed = torch.zeros(_B, _S, _K)
    skewed[..., 0] = 10.0
    _, aux_skew = r(skewed)
    pen_skew = float(aux_skew.detach())
    assert pen_skew > 0, "偏斜输入应产生正容量惩罚，实测 %r" % pen_skew
    # 完全均衡 ⇒ 各槽 mean_p = 1/K < capacity_factor/K ⇒ 无罚
    _, aux_flat = r(torch.zeros(_B, _S, _K))
    pen_flat = float(aux_flat.detach())
    assert abs(pen_flat) < 1e-6, "均衡输入不应有惩罚，实测 %r" % pen_flat
    print("[PASS] test_hard_router_capacity_penalty_positive_when_imbalanced "
          "(偏斜罚 %.4f / 均衡罚 %.2e)" % (pen_skew, pen_flat))
    return True


# ---------------------------------------------------------------- 4
def test_dynamic_router_slot_count_bounds():
    """4. DynamicSlotRouter: 每 token 激活槽数落在 [min_slots, max_slots]"""
    r = _dyn.DynamicSlotRouter(n_slots=_K, tau=0.5, min_slots=1, max_slots=4)
    logits = torch.randn(_B, _S, _K)
    gate, n_active = r(logits, training=True)
    assert gate.shape == (_B, _S, _K), gate.shape
    assert n_active.shape == (_B, _S), n_active.shape
    assert (n_active >= 1).all(), "激活槽数应 >= min_slots=1"
    assert (n_active <= 4).all(), "激活槽数应 <= max_slots=4"
    assert torch.allclose(gate.sum(-1), torch.ones(_B, _S), atol=1e-5)
    print("[PASS] test_dynamic_router_slot_count_bounds (激活槽数 %d–%d)"
          % (int(n_active.min()), int(n_active.max())))
    return True


# ---------------------------------------------------------------- 5
def test_dynamic_router_differentiable():
    """5. DynamicSlotRouter: 训练态门控对 logits 可微（Gumbel-sigmoid 的前提）"""
    r = _dyn.DynamicSlotRouter(n_slots=_K)
    logits = torch.randn(_B, _S, _K, requires_grad=True)
    gate, _ = r(logits, training=True)
    gate.sum().backward()
    assert logits.grad is not None, "梯度未回传到 logits"
    assert torch.isfinite(logits.grad).all(), "梯度含 NaN/Inf"
    assert float(logits.grad.abs().sum()) > 0, "梯度全零 —— 门控不可微"
    print("[PASS] test_dynamic_router_differentiable (梯度回传非零)")
    return True


# ---------------------------------------------------------------- 6
def test_dynamic_router_eval_mode_is_deterministic():
    """6. DynamicSlotRouter: 推理态无随机性（阈值化），两次前向一致"""
    r = _dyn.DynamicSlotRouter(n_slots=_K)
    logits = torch.randn(_B, _S, _K)
    g1, n1 = r(logits, training=False)
    g2, n2 = r(logits, training=False)
    assert torch.equal(g1, g2), "推理态两次前向不一致"
    assert torch.equal(n1, n2)
    print("[PASS] test_dynamic_router_eval_mode_is_deterministic (推理态确定性)")
    return True


# ---------------------------------------------------------------- 7
def test_audit_criteria_thresholds_present():
    """7. 审计脚本固化 8 项判据（含 v62 的关键阈值），可离线复用"""
    import inspect
    src = inspect.getsource(_audit)
    for key in ("A3_balance", "A4_slot_entropy"):
        assert key in src, "审计缺少判据 %s" % key
    # v62 实测: 均衡 0.612 / 槽熵 0.647 ⇒ 阈值 0.3 / 0.4 有据
    assert "0.3" in src, "缺少均衡阈值 0.3"
    assert "0.4" in src, "缺少槽熵阈值 0.4"
    print("[PASS] test_audit_criteria_thresholds_present (判据阈值在位)")
    return True


# ---------------------------------------------------------------- 8
def test_audit_effective_rank_and_cka_math():
    """8. 审计的两个数学工具在已知输入上给出正确值

    cka(X,X)=1（自相似），cka(X,-X)=1（中心化后符号无关）；
    effective_rank 对 rank-r 矩阵应约等于 r。
    """
    import numpy as np
    torch.manual_seed(0)
    X = np.random.randn(64, 16)
    assert abs(_audit.cka(X, X) - 1.0) < 1e-6, _audit.cka(X, X)
    assert abs(_audit.cka(X, -X) - 1.0) < 1e-6, _audit.cka(X, -X)
    Z = np.random.randn(64, 16) @ np.random.randn(16, 4)
    er = _audit.effective_rank(Z)
    assert 3 <= er <= 5, "effective_rank 应约等于 4，实测 %d" % er
    print("[PASS] test_audit_effective_rank_and_cka_math (cka=1, eff_rank=%d)" % er)
    return True


TESTS = [
    test_all_nine_scripts_compile,
    test_hard_router_sparse_and_normalized,
    test_hard_router_capacity_penalty_positive_when_imbalanced,
    test_dynamic_router_slot_count_bounds,
    test_dynamic_router_differentiable,
    test_dynamic_router_eval_mode_is_deterministic,
    test_audit_criteria_thresholds_present,
    test_audit_effective_rank_and_cka_math,
]

if __name__ == "__main__":
    from research._primitives.testing import run_suite
    sys.exit(run_suite(TITLE, TESTS))
