"""V54.0 评价器换轨 — 单元测试 (9 项)。不加载模型。"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "hetero_moe")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from research._primitives.testing import run_suite

TITLE = "V54.0 评价器换轨 — 单元测试 (9 项)"

import numpy as np
import torch

from research.aligner.v54_evaluator_swap.audit import (
    CLASSES, N_CLASSES, EVALUATORS, EvaluatorHead, eval_loss, baselines,
    decode_accuracy, calibration, make_split, build_targets,
)


def test_three_evaluators_registered():
    """三档必须齐全 (MSE 对照不被悄悄删掉)。"""
    assert EVALUATORS == ("mse", "gauss_nll", "ce"), EVALUATORS
    print(f"[PASS] 三档齐全: {EVALUATORS}")
    return True


def test_param_parity():
    """参数等量纪律: 相对 MSE 档的额外参数必须可数且极小。"""
    d = 96
    extra = {ev: EvaluatorHead(ev, d).extra_params() for ev in EVALUATORS}
    assert extra["mse"] == 0, extra
    assert extra["ce"] == 1, extra                  # 仅可学习温度 τ
    assert extra["gauss_nll"] == d + 1, extra       # 逐样本标量 logvar
    print(f"[PASS] 额外参数: mse=0 ce=1 gauss_nll={d + 1} (相对 lift 的几千参数可忽略)")
    return True


def test_integer_divisibility_enforced():
    """块状目标要求 D_shared 能被 n_classes 整除, 否则必须报错而不是静默错。"""
    for bad in (256, 100):
        if bad % N_CLASSES != 0:
            try:
                EvaluatorHead("ce", bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"d_shared={bad} 应被拒绝")
    EvaluatorHead("ce", 96)
    print(f"[PASS] 整除约束生效 (n_classes={N_CLASSES}: 96 可以, 256 被拒)")
    return True


def test_all_zero_mse_equals_one_over_n():
    """MSE(预测全零) = 1/n_classes 是精确等式 —— 这是整个换轨的出发点。"""
    labels = list(range(N_CLASSES)) * 2
    t = build_targets(labels, 96, N_CLASSES, torch.device("cpu"), n_tokens=4)
    got = float(((0 - t) ** 2).mean())
    assert abs(got - 1.0 / N_CLASSES) < 1e-6, (got, 1.0 / N_CLASSES)
    print(f"[PASS] 全零 MSE={got:.6f} == 1/{N_CLASSES}={1 / N_CLASSES:.6f} (精确, 非估计)")
    return True


def test_ce_uniform_matches_predict_mean_baseline():
    """两条独立推导必须撞到同一个数: CE 均匀预测的 MSE == predict_mean 基线。"""
    d, s, n = 96, 3, 6
    head = EvaluatorHead("ce", d)
    with torch.no_grad():
        head.tau_raw.fill_(1e6)                     # 温度极大 → softmax 均匀
    out = torch.zeros(n, s, d)
    tgt = build_targets([i % N_CLASSES for i in range(n)], d, N_CLASSES,
                        torch.device("cpu"), n_tokens=s)
    lab = torch.tensor([i % N_CLASSES for i in range(n)])
    _, info = eval_loss(head, out, tgt, lab)
    want = baselines("ce")["uniform_pred_mse"]
    assert abs(info["mse"] - want) < 1e-6, (info["mse"], want)
    assert abs(info["loss"] - baselines("ce")["uniform"]) < 1e-5
    print(f"[PASS] CE 均匀预测 mse={info['mse']:.6f} == predict_mean 基线={want:.6f}; "
          f"CE loss={info['loss']:.4f} == ln n")
    return True


def test_nll_baseline_exact():
    """NLL 基线必须带常数项: 最优 logvar 下 = 0.5(1+ln(mse))。漏掉 1 就是错的。"""
    d, s, n = 96, 2, 8
    head = EvaluatorHead("gauss_nll", d)
    out = torch.zeros(n, s, d)
    tgt = build_targets([i % N_CLASSES for i in range(n)], d, N_CLASSES,
                        torch.device("cpu"), n_tokens=s)
    lab = torch.tensor([i % N_CLASSES for i in range(n)])
    err = float(((0 - tgt) ** 2).mean())
    with torch.no_grad():
        head.logvar_head.weight.zero_()
        head.logvar_head.bias.fill_(float(np.log(err)))
    loss, _ = eval_loss(head, out, tgt, lab)
    want = baselines("gauss_nll")["all_zero"]
    assert abs(float(loss) - want) < 1e-4, (float(loss), want)
    print(f"[PASS] NLL 基线精确: 实测={float(loss):.6f} 解析={want:.6f}")
    return True


def test_constant_logvar_spearman_is_zero():
    """常数 logvar 的秩相关必须返回 0 (否则 E4 会假通过)。"""
    lv = torch.full((16,), -2.0)
    err = torch.rand(16)
    assert abs(calibration(lv, err)) < 1e-9
    rho = calibration(torch.arange(16).float(), torch.arange(16).float())
    assert rho > 0.99, rho
    print(f"[PASS] 常数 logvar → spearman=0; 完美单调 → spearman={rho:.3f}")
    return True


def test_ce_output_lives_on_simplex():
    """CE 档的输出必须落在单纯形上, 否则它的 MSE 与 MSE 档不可比。"""
    d, s, n = 96, 3, 4
    head = EvaluatorHead("ce", d)
    with torch.no_grad():
        head.tau_raw.fill_(0.0)
    out = torch.randn(n, s, d) * 3
    q = head.forward(out)
    blk = d // N_CLASSES
    blocks = q["pred"].reshape(n, s, N_CLASSES, blk)
    # 1) block 内必须是常数 (概率摊平), 不是每个维度各自取值
    spread = blocks.max(-1).values - blocks.min(-1).values
    assert float(spread.max()) < 1e-6, f"block 内不等: {float(spread.max())}"
    # 2) 每个 block 的取值 / blk 求和 == 1, 即概率归一
    mass = blocks.mean(-1).sum(-1)
    assert torch.allclose(mass, torch.ones_like(mass), atol=1e-5), mass
    assert q["pred"].min() >= 0
    print(f"[PASS] CE 输出: block 内常数 (spread<1e-6) 且概率归一 "
          f"(max 偏差 {float((mass - 1).abs().max()):.2e})")
    return True


def test_evaluator_swap_is_paired():
    """配对前提: 同一 seed 重建的模型权重必须逐位相同。"""
    from research.aligner.v54_evaluator_swap.audit import ResidualLift
    torch.manual_seed(7)
    a = ResidualLift(96, 2560)
    torch.manual_seed(7)
    b = ResidualLift(96, 2560)
    for (n1, p1), (n2, p2) in zip(a.named_parameters(), b.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2), n1
    print("[PASS] 同 seed 重建逐位相同 → 三档配对成立")
    return True


TESTS = [
    test_three_evaluators_registered,
    test_param_parity,
    test_integer_divisibility_enforced,
    test_all_zero_mse_equals_one_over_n,
    test_ce_uniform_matches_predict_mean_baseline,
    test_nll_baseline_exact,
    test_constant_logvar_spearman_is_zero,
    test_ce_output_lives_on_simplex,
    test_evaluator_swap_is_paired,
]


if __name__ == "__main__":
    sys.exit(run_suite(TITLE, TESTS))
