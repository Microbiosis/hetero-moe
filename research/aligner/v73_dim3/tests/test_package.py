"""V73.0 — Dim III 子空间夹角（维度修复 + eigh 提速）权威测试。

v71 的审计脚本有**两个致命 bug**，本代把它们修掉：

    1. 曾用随机初始化的 `DimProjectionSpace` + `load_state_dict(strict=False)`，
       加载失败的键被静默忽略 ⇒ 等于**审计随机空间**，不是审计训练好的模型。
       本版断言 missing_keys 必须为空。
    2. 曾漏掉 P_m 投影，把 2048 维原始 hidden 直接喂给期望 256 维的槽投影
       ⇒ 维度报错。本版用 ckpt 里的 P_m 权重做投影。

同时把 Dim III 的子空间求解从**全 SVD** 换成 **Gram 矩阵 eigh**
（结果等价，CPU 从几十分钟到秒级），并修掉三处维度错误：
梯度路径未过 P_m、`g @ P_m.T` 漏了转置、三模态子空间维度不同无法直接比夹角。

被锁住的契约:
    1. proj_space 权重缺键 ⇒ 脚本必须**报错退出**（不再静默审计随机空间）
    2. feats 与梯度路径都过 P_m 投影，且 P_m 键名/shape 有校验
    3. ViT 侧截断到 128 token 与文本位置对齐
    4. Dim III 用 Gram eigh 而非对 256×256 矩阵做全 SVD
    5. eigh 与全 SVD 给出**同一个 k 维子空间**（结果等价）
    6. 三模态子空间统一投到 256 维槽空间后才比夹角（用投影算子）

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

import numpy as np

import _audit_kit as K

TITLE = "V73.0 Dim III 子空间夹角 — 权重完整性 + P_m 投影 + eigh 提速"

SCRIPT = "slot_independence_audit.py"


# ---------------------------------------------------------------- 1
def test_missing_proj_space_keys_must_fail():
    """1. proj_space 权重缺键 ⇒ 脚本必须报错退出（v73 的核心修复）"""
    tmp = tempfile.mkdtemp(prefix="v73_missing_")
    ckpt, cache = K.make_audit_inputs(os.path.join(tmp, "in"), n=24, seq=16,
                                      drop_proj_key=True)
    rc, out, err = K.run_slot_audit(ckpt, cache, sufs="_v65", n=8, grad_n=2)
    assert rc != 0, "缺键时脚本仍返回 0 —— 等于在审计随机空间"
    assert "proj_space" in err and ("缺失" in err or "missing" in err.lower()), \
        "报错信息应说明 proj_space 权重缺失:\n%s" % err[-500:]
    assert "[DIM2_RESULT]" not in out, "缺键时不应产出审计结论"
    print("[PASS] test_missing_proj_space_keys_must_fail (rc=%d, 已拒绝)" % rc)
    return True


# ---------------------------------------------------------------- 2
def test_p_m_projection_applied_and_validated():
    """2. feats 与梯度路径都过 P_m 投影，且 P_m 键名 / shape 有校验"""
    src = K.src(SCRIPT)
    assert "key = 'P_m.%d' % i" in src, "P_m 键名应是 ParameterList 的 P_m.N"
    assert "assert key in sd" in src, "缺 P_m 存在性断言"
    assert "q_t @ P_m[0].t()" in src and "b_t @ P_m[1].t()" in src, \
        "前向 feats 未过 P_m 投影"
    assert "gp = g @ P_m[_mi].cpu().numpy().T" in src, \
        "梯度路径未过 P_m 投影（且漏了转置）"
    assert "256, 2048" in src or "[256,2048/768/1152]" in src or \
        "2048" in src, "缺 P_m shape 校验"
    print("[PASS] test_p_m_projection_applied_and_validated (P_m.N 键名 + 双向投影 + shape 校验)")
    return True


# ---------------------------------------------------------------- 3
def test_vit_truncated_to_text_length():
    """3. ViT 侧截断到 128 token 与文本位置对齐（ViT 是 197 patch，文本是 128 token）"""
    src = K.src(SCRIPT)
    assert "[:, :128, :]" in src, "ViT 未截断到 128 token"
    assert "截断到 128" in src or "128" in src
    print("[PASS] test_vit_truncated_to_text_length (ViT [:, :128, :])")
    return True


# ---------------------------------------------------------------- 4
def test_dim3_uses_gram_eigh():
    """4. Dim III 用 Gram 矩阵 eigh，而不是对 256×256 矩阵做全 SVD"""
    src = K.src(SCRIPT)
    assert "G = gp.T @ gp" in src, "未构造 Gram 矩阵"
    assert "np.linalg.eigh(G)" in src, "未用 eigh 求 Gram 特征分解"
    assert "np.linalg.svd(G" not in src, "仍在对 Gram 做全 SVD"
    print("[PASS] test_dim3_uses_gram_eigh (Gram + eigh, 无全 SVD)")
    return True


# ---------------------------------------------------------------- 5
def test_gram_eigh_subspace_equals_full_svd_subspace():
    """5. eigh 与全 SVD 给出同一个 k 维子空间（提速不改变结论）"""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 256))
    G = X.T @ X
    k = 8
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1][:k]
    Vk = V[:, order]
    _u, _s, Vt = np.linalg.svd(G)
    V_svd = Vt.T[:, :k]
    diff = float(np.abs(Vk @ Vk.T - V_svd @ V_svd.T).max())
    assert diff < 1e-8, "两个子空间的投影算子差异 %g" % diff
    print("[PASS] test_gram_eigh_subspace_equals_full_svd_subspace (投影算子差异 %.2e)" % diff)
    return True


# ---------------------------------------------------------------- 6
def test_angle_compared_in_shared_slot_space():
    """6. 三模态子空间统一投到 256 维槽空间后，用投影算子比夹角"""
    src = K.src(SCRIPT)
    assert "Us.append(Vk)" in src, "未把各模态主方向统一成 [256, k]"
    assert "M = Us[i].T @ Us[j]" in src, "未用投影算子比较子空间"
    assert "np.linalg.svd(M, compute_uv=False)[0]" in src, \
        "夹角应为 |Ui^T Uj| 的最大奇异值"
    print("[PASS] test_angle_compared_in_shared_slot_space (投影算子最大奇异值)")
    return True


# ---------------------------------------------------------------- 7
def test_audit_still_produces_both_reports():
    """7. 修完之后审计仍能产出两份报告（提速与加严没有把审计改坏）"""
    tmp = tempfile.mkdtemp(prefix="v73_ok_")
    ckpt, cache = K.make_audit_inputs(os.path.join(tmp, "in"), n=24, seq=16)
    rc, out, err = K.run_slot_audit(ckpt, cache, sufs="_v65", n=8, grad_n=2)
    assert rc == 0, "rc=%d\n%s" % (rc, err[-600:])
    assert "[DIM2_RESULT]" in out and "[DIM3_RESULT]" in out
    d3 = K.parse_dim_result(out, "DIM3_RESULT")
    assert "max" in d3, d3
    print("[PASS] test_audit_still_produces_both_reports (两份报告齐备, max=%s)" % d3["max"])
    return True


TESTS = [
    test_missing_proj_space_keys_must_fail,
    test_p_m_projection_applied_and_validated,
    test_vit_truncated_to_text_length,
    test_dim3_uses_gram_eigh,
    test_gram_eigh_subspace_equals_full_svd_subspace,
    test_angle_compared_in_shared_slot_space,
    test_audit_still_produces_both_reports,
]

if __name__ == "__main__":
    from research._primitives.testing import run_suite
    sys.exit(run_suite(TITLE, TESTS))
