"""V32 目标泄漏审计 — 单元测试 (8 项).

这些断言全部**从包内 results/ 的原始记录重算**, 不对着写死的字面量空转.
若哪天有人篡改证据、改错符号约定, 或把泄漏项偷偷加回去, 本套件必须转红.

数值口径的两个易错点, 已在测试中显式钉住:
    1. paired_gap := teacher - pure_student (负值 = 学生更差)
    2. 标准差用**样本**标准差 (ddof=1); ddof=0 会得到 ±0.0036 而非论文的 ±0.0038
"""
import os
import statistics
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "hetero_moe")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from research._primitives.testing import run_suite  # noqa: E402

TITLE = "V32 目标泄漏审计 (variance isolation / beta scan / 迁移) — 单元测试 (8 项)"

from research.distillation.v32_leakage_audit import (  # noqa: E402
    RESULTS_DIR,
    beta_scan_curve,
    beta_scan_is_monotonic,
    leakage_verdict,
    load_records,
    paired_gap,
    published_headline,
    super_deterministic_table,
    transfer_accuracy,
    variance_isolation_table,
)

CONDITIONS = ["base", "no-int4", "soft-route", "deterministic"]


def test_evidence_ships_with_package():
    """1. 四份原始记录随包分发 —— 审计不得依赖仓库根的 results/."""
    for name in ("v32_variance_isolation", "v32_beta_scan",
                 "v32_super_deterministic", "v32_representation_transfer"):
        p = RESULTS_DIR / f"{name}.json"
        assert p.exists(), f"包内缺少原始记录: {p}"
        assert p.stat().st_size > 0, f"原始记录为空: {p}"
    assert RESULTS_DIR.parent.name == "v32_leakage_audit", \
        f"results/ 不在包内: {RESULTS_DIR}"
    print("[PASS] test_evidence_ships_with_package")
    return True


def test_paired_gap_sign_convention():
    """2. 配对差约定为 teacher - student, 且标准差取自**逐 seed 配对差**.

    符号约定错了会把"学生不超越教师"读成"学生反超教师" —— 与论文正面冲突,
    所以显式钉死, 而不是让它隐含在别处。
    """
    g = paired_gap([1.0, 2.0, 3.0], [0.5, 2.5, 2.0])
    # 逐 seed 差 = [0.5, -0.5, 1.0] -> mean 1/3, 样本 std 0.7638
    assert abs(g["mean"] - 0.3333333) < 1e-4, f"均值算错: {g['mean']}"
    assert abs(g["std"] - 0.7637626) < 1e-4, f"样本标准差算错: {g['std']}"
    # 配对差的 std 必须与"把两组数摊平一起算 std"不同 —— 否则说明没做配对
    pooled = statistics.stdev([1.0, 2.0, 3.0, 0.5, 2.5, 2.0])
    assert abs(g["std"] - pooled) > 1e-3, \
        f"配对差 std {g['std']:.4f} 与摊平 std {pooled:.4f} 相同, 配对可能没生效"
    # 负值 = 学生更差
    assert paired_gap([1.0], [2.0])["mean"] < 0, "符号约定反了: 学生更差时 gap 应为负"
    print("[PASS] test_paired_gap_sign_convention")
    return True


def test_headline_number_reproduces_paper():
    """3. 从原始记录重算出的头条数字与论文/v16 MANIFEST 一致 (-0.0009 +- 0.0038).

    这是本包最有价值的一条: 论文那句"纯蒸馏不超越教师"不是抄来的, 是算出来的.
    """
    h = published_headline()
    assert h["condition"] == "deterministic", f"头条条件应为 deterministic, 实为 {h['condition']}"
    assert h["seeds"] == 10, f"应为 10 seeds, 实为 {h['seeds']}"
    assert abs(h["paired_gap"] - (-0.0009)) < 5e-4, \
        f"配对差与论文不符: {h['paired_gap']:.4f} vs -0.0009"
    assert abs(h["paired_gap_std"] - 0.0038) < 5e-4, \
        f"配对差标准差与论文不符: {h['paired_gap_std']:.4f} vs 0.0038 (注意 ddof=1)"
    print(f"[PASS] test_headline_number_reproduces_paper "
          f"(gap {h['paired_gap']:+.4f} ± {h['paired_gap_std']:.4f})")
    return True


def test_gap_is_noise_in_every_condition():
    """4. 四个教师随机性条件下配对差都在噪声内 —— 核心发现.

    若哪天把 INT4 噪声或硬 Top-K 路由的影响真的测出来了, 本条必须转红。
    """
    table = variance_isolation_table()
    for cond in CONDITIONS:
        assert cond in table, f"缺条件 {cond}"
        gap = table[cond]["paired_gap"]
        assert abs(gap["mean"]) < 0.01, \
            f"{cond} 条件下配对差 {gap['mean']:+.4f} 已超出噪声 (±0.01)"
        assert gap["n"] == 10, f"{cond} 应为 10 seeds, 实为 {gap['n']}"
    print("[PASS] test_gap_is_noise_in_every_condition "
          + " ".join(f"{c}={table[c]['paired_gap']['mean']:+.4f}" for c in CONDITIONS))
    return True


def test_mixed_objective_produces_the_headline_margin():
    """5. 表观收益确实来自混合目标 (泄漏项) —— 移走后优势消失.

    这一对断言必须同时成立: 混合远低于教师, 纯蒸馏贴着教师。
    只断其中一半, "泄漏"这个解释就不成立。
    """
    h = published_headline()
    assert h["mixed_student_mse"] < 0.6 * h["teacher_mse"], \
        f"混合学生 {h['mixed_student_mse']:.4f} 未显著低于教师 {h['teacher_mse']:.4f}"
    assert abs(h["pure_student_mse"] - h["teacher_mse"]) < 0.01, \
        f"纯学生 {h['pure_student_mse']:.4f} 应贴着教师 {h['teacher_mse']:.4f}"
    print(f"[PASS] test_mixed_objective_produces_the_headline_margin "
          f"(mixed {h['mixed_student_mse']:.4f} vs pure {h['pure_student_mse']:.4f} "
          f"vs teacher {h['teacher_mse']:.4f})")
    return True


def test_beta_scan_is_continuous_not_threshold():
    """6. beta_h 0→1 连续过渡, 无跳变 —— 收益与泄漏权重成比例, 非机制被激活."""
    curve = beta_scan_curve()
    assert len(curve) == 11, f"应有 11 个 beta 取值, 实为 {len(curve)}"
    assert beta_scan_is_monotonic(), "学生 MSE 未随 beta_h 单调不增"
    beta0, beta1 = curve[0], curve[-1]
    assert beta0["beta_h"] == 0.0 and beta1["beta_h"] == 1.0, "beta 端点不对"
    assert abs(beta0["student_mse"] - beta0["teacher_mse"]) < 0.01, \
        f"beta=0 时学生应贴着教师: {beta0['student_mse']:.4f} vs {beta0['teacher_mse']:.4f}"
    assert beta1["student_mse"] < beta0["student_mse"], "beta=1 应远低于 beta=0"
    # 无阈值跳变: 最大单步降幅应远小于总降幅 (阈值效应会在某一步吃掉大头)
    drops = [a["student_mse"] - b["student_mse"] for a, b in zip(curve, curve[1:])]
    assert max(drops) < 0.5 * (beta0["student_mse"] - beta1["student_mse"]), \
        f"存在单步跳变 {max(drops):.4f}, 形态更像阈值而非线性过渡"
    print(f"[PASS] test_beta_scan_is_continuous_not_threshold "
          f"({beta0['student_mse']:.4f} → {beta1['student_mse']:.4f}, 单步最大降 {max(drops):.4f})")
    return True


def test_super_deterministic_falsifies_jensen():
    """7. 20 次前向平均的超确定教师与确定教师完全一致 —— Jensen 优势不可测量."""
    s = super_deterministic_table()
    assert "super-deterministic" in s and "deterministic" in s, "缺条件"
    assert abs(s["super-deterministic"]["teacher_mse_mean"]
               - s["deterministic"]["teacher_mse_mean"]) < 1e-6, \
        "超确定教师应与确定教师数值相同"
    for cond in ("deterministic", "super-deterministic"):
        assert abs(s[cond]["gap_pure_mean"]) < 0.01, \
            f"{cond} 下纯学生配对差 {s[cond]['gap_pure_mean']:+.4f} 超出噪声"
    print("[PASS] test_super_deterministic_falsifies_jensen "
          f"(超确定教师 {s['super-deterministic']['teacher_mse_mean']:.4f} "
          f"= 确定教师 {s['deterministic']['teacher_mse_mean']:.4f})")
    return True


def test_verdict_and_transfer():
    """8. 迁移实验与整代判定: 纯学生不劣于混合学生, 且总判定为"纯蒸馏不超越教师"."""
    t = transfer_accuracy()
    assert t["pure"]["eval_accuracy"] >= t["mixed"]["eval_accuracy"], \
        f"纯学生迁移准确率 {t['pure']['eval_accuracy']:.4f} 应不低于混合 {t['mixed']['eval_accuracy']:.4f}"
    v = leakage_verdict()
    assert v["pure_beats_teacher"] is False, \
        f"总判定应为纯蒸馏不超越教师, 实得 pure_beats_teacher={v['pure_beats_teacher']}"
    assert v["gap_within_noise_in_every_condition"] is True, "存在配对差超出噪声的条件"
    assert v["beta_scan_monotonic"] is True, "beta 扫描非单调"
    assert v["super_deterministic_teacher_identical"] is True, "超确定教师与确定教师不一致"
    assert v["transfer_prefers_leak_free"] is True, "迁移实验未支持无泄漏条件"
    assert isinstance(v["conclusion"], str) and v["conclusion"], "结论文本为空"
    print(f"[PASS] test_verdict_and_transfer (transfer pure {t['pure']['eval_accuracy']:.3f} "
          f">= mixed {t['mixed']['eval_accuracy']:.3f}; pure_beats_teacher={v['pure_beats_teacher']})")
    return True


TESTS = [
    test_evidence_ships_with_package,
    test_paired_gap_sign_convention,
    test_headline_number_reproduces_paper,
    test_gap_is_noise_in_every_condition,
    test_mixed_objective_produces_the_headline_margin,
    test_beta_scan_is_continuous_not_threshold,
    test_super_deterministic_falsifies_jensen,
    test_verdict_and_transfer,
]


if __name__ == "__main__":
    sys.exit(run_suite(TITLE, TESTS))
