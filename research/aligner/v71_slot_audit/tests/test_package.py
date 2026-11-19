"""V71.0 — 四槽 / 三模态独立性审计 权威测试。

本代引入独立审计脚本 `slot_independence_audit.py`（借鉴 neuro-physical 仓库的
五维独立性审计方法论），回答两个问题：

    Dim II  四槽 canonical 的**表示独立性** —— Q/K/V/O 是否名实相符，
            还是四槽其实只是「源身份探测器」（联合子空间秩 / 两两 |cos| / VIF）
    Dim III 三模态输入**梯度子空间独立性** —— 各模态是否各司其职，
            还是某个模态的梯度方向与另一个高度共线

⚠️ 脚本在 v73 被重写（加严了权重完整性断言与 P_m 投影）。
   本代测试锁的是**它自己的贡献**：两段审计都在、判据口径不变、能真跑出报告。

被锁住的契约:
    1. 审计脚本能端到端跑完（合成 ckpt + 合成 cache），产出两份结构化报告
    2. Dim II 报告联合子空间秩 / 两两 |cos| / VIF
    3. Dim III 报告三对子空间主夹角
    4. Dim II 判据 = 满秩 且 两两 |cos| < 0.6
    5. Dim III 判据 = 最大夹角 < 0.9
    6. 默认参数：n=400 / k=8 / grad_n=16
    7. 审计对象是四槽 canonical（槽名取自 proj_space，不是写死的 QKVO）

每个 test_* 必须在末尾 return True。
"""
import os
import sys
import tempfile

_here = os.path.dirname(os.path.abspath(__file__))
_pkg = os.path.dirname(_here)
_v62 = os.path.join(os.path.dirname(_pkg), "v62_dimension_projection_space")
_root = _pkg
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "hetero_moe")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)
if _v62 not in sys.path:
    sys.path.insert(0, _v62)

import _audit_kit as K

TITLE = "V71.0 四槽 / 三模态独立性审计"

SCRIPT = "slot_independence_audit.py"

_CACHE = {}


def _run():
    """跑一次审计（结果在进程内缓存，多个 test 共用一次运行）。"""
    if "res" not in _CACHE:
        tmp = tempfile.mkdtemp(prefix="v71_audit_")
        ckpt, cache = K.make_audit_inputs(tmp, n=24, seq=16)
        rc, out, err = K.run_slot_audit(ckpt, cache, sufs="_v65", n=16, grad_n=4)
        _CACHE["res"] = (rc, out, err)
    return _CACHE["res"]


# ---------------------------------------------------------------- 1
def test_audit_script_runs_end_to_end():
    """1. 审计脚本能端到端跑完并产出两份结构化报告"""
    rc, out, err = _run()
    assert rc == 0, "审计脚本退出码 %d\n%s" % (rc, err[-800:])
    assert "[DIM2_RESULT]" in out, "缺 Dim II 报告"
    assert "[DIM3_RESULT]" in out, "缺 Dim III 报告"
    assert "[done]" in out, "脚本未跑到结尾"
    print("[PASS] test_audit_script_runs_end_to_end (rc=0, 两份报告齐备)")
    return True


# ---------------------------------------------------------------- 2
def test_dim2_reports_rank_cos_and_vif():
    """2. Dim II 报告联合子空间秩 / 两两 |cos| / VIF"""
    rc, out, err = _run()
    d2 = K.parse_dim_result(out, "DIM2_RESULT")
    for key in ("rank", "mean_cos", "max_cos", "vif"):
        assert key in d2, "DIM2_RESULT 缺 %s: %s" % (key, d2)
    assert d2["rank"].startswith("4/"), "四槽应报 4 槽秩，实测 %s" % d2["rank"]
    print("[PASS] test_dim2_reports_rank_cos_and_vif (rank=%s, max_cos=%s)"
          % (d2["rank"], d2["max_cos"]))
    return True


# ---------------------------------------------------------------- 3
def test_dim3_reports_three_pairwise_angles():
    """3. Dim III 报告三对子空间主夹角"""
    rc, out, err = _run()
    d3 = K.parse_dim_result(out, "DIM3_RESULT")
    assert "angles" in d3, "DIM3_RESULT 缺 angles: %s" % d3
    angles = [a.strip() for a in d3["angles"].strip("[]").split(",") if a.strip()]
    assert len(angles) == 3, "三模态应给出 3 对夹角，实测 %s" % d3["angles"]
    assert "max" in d3, d3
    print("[PASS] test_dim3_reports_three_pairwise_angles (%s, max=%s)"
          % (d3["angles"], d3.get("max")))
    return True


# ---------------------------------------------------------------- 4
def test_dim2_criterion_is_rank_and_cosine():
    """4. Dim II 判据 = 满秩 且 两两 |cos| < 0.6（口径不能被悄悄改掉）"""
    src = K.src(SCRIPT)
    assert "rank == len(slots) and cos[off].max() < 0.6" in src, \
        "Dim II 判据不再是「满秩 + 两两 |cos| < 0.6」"
    assert "[DIM2_RESULT]" in src, "Dim II 缺结构化结果行"
    print("[PASS] test_dim2_criterion_is_rank_and_cosine (rank==4 且 max|cos|<0.6)")
    return True


# ---------------------------------------------------------------- 5
def test_dim3_criterion_is_max_angle():
    """5. Dim III 判据 = 最大夹角 < 0.9"""
    src = K.src(SCRIPT)
    assert "np.max(sc)) < 0.9" in src or "float(np.max(sc)) < 0.9" in src, \
        "Dim III 判据不再是「最大夹角 < 0.9」"
    assert "[DIM3_RESULT]" in src, "Dim III 缺结构化结果行"
    print("[PASS] test_dim3_criterion_is_max_angle (最大夹角 < 0.9)")
    return True


# ---------------------------------------------------------------- 6
def test_default_sample_parameters():
    """6. 默认参数：n=400 / k=8 / grad_n=16（Dim III 只用少量样本，CPU 反传太慢）"""
    import argparse
    src = K.src(SCRIPT)
    assert "'--n'" in src and "default=400" in src, "缺 --n 默认 400"
    assert "'--k'" in src and "default=8" in src, "缺 --k 默认 8"
    assert "'--grad_n'" in src and "default=16" in src, "缺 --grad_n 默认 16"
    print("[PASS] test_default_sample_parameters (n=400, k=8, grad_n=16)")
    return True


# ---------------------------------------------------------------- 7
def test_audit_object_is_four_slot_canonical():
    """7. 审计对象是四槽 canonical，槽名取自 proj_space（不是写死 QKVO）"""
    src = K.src(SCRIPT)
    assert "slots = ps.slot_names" in src, "槽名应取自 proj_space.slot_names"
    assert "cn[..., i*D:(i+1)*D]" in src, "未按槽切 canonical"
    assert len(K.src(SCRIPT)) > 0
    print("[PASS] test_audit_object_is_four_slot_canonical (槽名取自 ps.slot_names)")
    return True


TESTS = [
    test_audit_script_runs_end_to_end,
    test_dim2_reports_rank_cos_and_vif,
    test_dim3_reports_three_pairwise_angles,
    test_dim2_criterion_is_rank_and_cosine,
    test_dim3_criterion_is_max_angle,
    test_default_sample_parameters,
    test_audit_object_is_four_slot_canonical,
]

if __name__ == "__main__":
    from research._primitives.testing import run_suite
    sys.exit(run_suite(TITLE, TESTS))
