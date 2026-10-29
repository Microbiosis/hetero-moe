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

TITLE = "V54.0 评价器换轨 — 单元测试 (16 项)"

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


# ----------------------------------------------------------------------
# 完整性闸门 (integrity.py) —— 这一组是**回归测试**, 全部来自真实踩到的洞
# ----------------------------------------------------------------------

def _fake_results(d=96, evs=("mse", "gauss_nll", "ce"), arms=("frozen_random", "learned_lift"),
                  seeds=(2000, 2001), fuse_l=0.10, fuse_f=0.16):
    """造一份结构合法的 results, 只填判决用得到的字段。"""
    out = {}
    for ev in evs:
        for arm in arms:
            rows = []
            for sd in seeds:
                rows.append({
                    "seed": sd,
                    "held_out": {"fuse": fuse_l if arm == "learned_lift" else fuse_f,
                                 "avg": 0.2, "best_single": 0.2},
                    "acc": {"fuse": 1.0 / N_CLASSES if arm == "frozen_random" else 0.5,
                            "best_single": 0.45},
                    "held_out_calibration": 0.5,
                    "eff_rank": 30.0 if arm == "learned_lift" else 20.0,
                })
            out[f"D{d}:{ev}:{arm}"] = {"rows": rows}
    return out


def _grid(d=96, evs=("mse", "gauss_nll", "ce"), arms=("frozen_random", "learned_lift"),
          seeds=(2000, 2001)):
    from research.aligner.v54_evaluator_swap.integrity import expected_grid
    return expected_grid([d], list(evs), list(arms), len(seeds), seeds[0])


def _crit():
    from research.aligner.v54_evaluator_swap.audit import baselines
    N = N_CLASSES

    def e1(rl, rf, ctx):
        return rl["held_out"]["fuse"] < rf["held_out"]["fuse"]

    def e2(rl, rf, ctx):
        return abs(rf["acc"]["fuse"] - 1.0 / N) <= 0.05

    return {"E1": {"fn": e1}, "E2": {"fn": e2, "only_ev": ("ce",)}}


def test_gate_empty_grid_is_not_pass():
    """★ 回归: 空网格曾判 all_pass=True (None 被过滤掉)。必须为 False。"""
    from research.aligner.v54_evaluator_swap.integrity import judge_verdicts
    v, a, ok = judge_verdicts({}, _grid(), _crit())
    assert ok is False, "空网格判通过了 —— 这就是要防的那个 bug"
    assert a["_completeness"]["ok"] is False
    assert v["D96"]["E1"] is None
    print(f"[PASS] 空网格 → all_pass=False, 缺 {a['_completeness']['n_cells_want']} 格")
    return True


def test_gate_missing_one_cell_blocks():
    """★ 少一格也不给结论 (不能"差不多就发")。"""
    from research.aligner.v54_evaluator_swap.integrity import judge_verdicts
    r = _fake_results()
    r.pop("D96:mse:learned_lift")
    v, a, ok = judge_verdicts(r, _grid(), _crit())
    assert ok is False and a["_completeness"]["ok"] is False
    assert a["_completeness"]["missing_cells"] == ["D96:mse:learned_lift"]
    print(f"[PASS] 缺 1 格 → all_pass=False; 缺格被点名")
    return True


def test_gate_missing_one_seed_blocks():
    """★ 一格少一个 seed 也算不完备 —— 否则 80% 规则会被"只剩 2 个 seed"钻空子。"""
    from research.aligner.v54_evaluator_swap.integrity import judge_verdicts
    r = _fake_results()
    r["D96:mse:learned_lift"]["rows"].pop()
    v, a, ok = judge_verdicts(r, _grid(), _crit())
    assert ok is False
    assert "D96:mse:learned_lift" in a["_completeness"]["bad_seed_sets"]
    print(f"[PASS] 缺 1 个 seed → all_pass=False, 记入 bad_seed_sets")
    return True


def test_gate_complete_and_true_passes():
    """完备 + 全真 → 通过 (闸门不能把所有事都判失败)。"""
    from research.aligner.v54_evaluator_swap.integrity import judge_verdicts
    v, a, ok = judge_verdicts(_fake_results(), _grid(), _crit())
    assert ok is True, (v, a["_gate"])
    assert a["_completeness"]["n_obs_have"] == a["_completeness"]["n_obs_want"] == 12
    print(f"[PASS] 完备且全真 → all_pass=True ({a['_completeness']['n_obs_have']} 观测)")
    return True


def test_gate_complete_but_false_fails():
    """完备但有一条为假 → 不通过, 且**不是** None (要能区分"没过"与"没算")。"""
    from research.aligner.v54_evaluator_swap.integrity import judge_verdicts
    r = _fake_results(fuse_l=0.30, fuse_f=0.16)      # learned 更差
    v, a, ok = judge_verdicts(r, _grid(), _crit())
    assert ok is False
    assert v["D96"]["E1"] is False, v["D96"]
    assert a["_gate"]["no_undecided"] is True        # 有结论, 只是结论为假
    print("[PASS] 完备但 E1 为假 → all_pass=False, 且 no_undecided=True "
          "(区分『没过』与『没算』)")
    return True


def test_gate_undecidable_blocks():
    """判据返回 None → 该判据"无结论", 整体不得判通过。"""
    from research.aligner.v54_evaluator_swap.integrity import judge_verdicts
    r = _fake_results()
    for k in r:
        for row in r[k]["rows"]:
            row.pop("held_out_calibration", None)

    def e4(rl, rf, ctx):
        rho = rl.get("held_out_calibration")
        return None if rho is None else rho >= 0.2

    crit = dict(_crit())
    crit["E4"] = {"fn": e4, "only_ev": ("gauss_nll",)}
    v, a, ok = judge_verdicts(r, _grid(), crit)
    assert v["D96"]["E4"] is None
    assert ok is False and a["_gate"]["no_undecided"] is False
    print("[PASS] 判据无结论 → E4=None, all_pass=False (不会被当成通过)")
    return True


def test_gate_only_ev_scales_expectation():
    """only_ev 的预期观测数必须按适用评价器缩放, 否则 E2 永远"观测不足"。"""
    from research.aligner.v54_evaluator_swap.integrity import judge_verdicts
    v, a, ok = judge_verdicts(_fake_results(), _grid(), _crit())
    assert a["D96"]["E2"]["n_expected"] == 2, a["D96"]["E2"]     # 1 D × 1 seed × 仅 ce
    assert a["D96"]["E1"]["n_expected"] == 6, a["D96"]["E1"]     # 3 档 × 2 seed
    print(f"[PASS] only_ev 缩放正确: E2 预期 {a['D96']['E2']['n_expected']} (ce×2 seed), "
          f"E1 预期 {a['D96']['E1']['n_expected']} (3 档×2 seed)")
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
    test_gate_empty_grid_is_not_pass,
    test_gate_missing_one_cell_blocks,
    test_gate_missing_one_seed_blocks,
    test_gate_complete_and_true_passes,
    test_gate_complete_but_false_fails,
    test_gate_undecidable_blocks,
    test_gate_only_ev_scales_expectation,
]


if __name__ == "__main__":
    sys.exit(run_suite(TITLE, TESTS))
