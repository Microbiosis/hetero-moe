"""V69.0 — 异构 P_m（维度调解）权威测试。

问题：BERT(768) / ViT(1152) 对接 Qwen(2048) 时，两条老路都不通——
    · 统一压到 d_shared=256 ⇒ 信息残废（保真度 0.165 @24 维的那种事）
    · 补零升到 2048 ⇒ 稀疏，一半维度恒零

解法（异构 P_m）：
    · 大维度模态（Qwen）→ DimDownProj 降维
    · 小维度模态（BERT/ViT）→ DimReconcileBlock：局部提炼（多头注意力或 MLP）
      + 稠密升维 + sigmoid 门控保护原生信息
    · d_shared 从 256 提升到 pm_target（默认 1024）

被锁住的契约:
    1. DimDownProj 把大模型降到 target 维
    2. DimReconcileBlock 稠密升维，**没有补零**（每个输出维都有方差）
    3. d_src 不能被 n_heads 整除时退化为 MLP 提炼（且仍能跑）
    4. 默认路径仍是「统一正交冻结 P_m」（行为不变，可回退）
    5. pm_reconcile 路径把 d_shared 提到 pm_target，并改用 modal_proj
    6. PM_RECONCILE / PM_TARGET 环境变量可切

本代测试直接**执行 train_v2.py 里的真实类源码**（用 AST 取出类片段后 exec），
而不是抄一份到测试里 —— 抄的那份会与实现漂移。
每个 test_* 必须在末尾 return True。
"""
import os
import sys

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

import torch
import torch.nn as nn

import _audit_kit as K

TITLE = "V69.0 异构 P_m — 维度调解"

TRAIN = "train_v2.py"
MODAL_DIMS = (2048, 768, 1152)


def _ns():
    """把 train_v2.py 里的真实类源码取出来执行（不 import train_v2 —— 它会开训）。"""
    import projspace
    import fusion
    g = {
        "torch": torch,
        "nn": nn,
        "os": os,
        "MEASURED": projspace.MEASURED_SUBSPACES,
        "DimProjectionSpace": projspace.DimProjectionSpace,
        "FusionOverProjSpace": fusion.FusionOverProjSpace,
    }
    for cls in ("DimDownProj", "DimReconcileBlock", "HeteroMoEProjSpace"):
        exec(compile(K.class_source(TRAIN, cls), cls, "exec"), g)
    return g


# ---------------------------------------------------------------- 1
def test_dim_down_proj_reduces_to_target():
    """1. DimDownProj 把大维度模态降到 target 维（Qwen 2048 -> 1024）"""
    g = _ns()
    blk = g["DimDownProj"](2048, 1024)
    x = torch.randn(2, 5, 2048)
    y = blk(x)
    assert y.shape == (2, 5, 1024), y.shape
    assert torch.isfinite(y).all()
    print("[PASS] test_dim_down_proj_reduces_to_target (%s -> %s)" % (tuple(x.shape), tuple(y.shape)))
    return True


# ---------------------------------------------------------------- 2
def test_dim_reconcile_block_dense_upprojection():
    """2. DimReconcileBlock 稠密升维——**不是补零**（每个输出维都有方差）"""
    g = _ns()
    blk = g["DimReconcileBlock"](768, 1024)
    x = torch.randn(4, 7, 768)
    y = blk(x)
    assert y.shape == (4, 7, 1024), y.shape
    per_dim_std = y.std(dim=(0, 1))
    dead = int((per_dim_std < 1e-6).sum())
    assert dead == 0, "有 %d 个输出维恒为常数（补零会这样），实测 std 最小 %.3g" % (
        dead, float(per_dim_std.min()))
    print("[PASS] test_dim_reconcile_block_dense_upprojection (1024 维全部非死维)")
    return True


# ---------------------------------------------------------------- 3
def test_dim_reconcile_block_attn_or_mlp_fallback():
    """3. d_src 不能被 n_heads 整除时退化为 MLP 提炼（两条路都要能跑）"""
    g = _ns()
    attn = g["DimReconcileBlock"](768, 1024, n_heads=4)
    assert attn.use_attn is True, "768 %% 4 == 0，应走多头注意力"
    mlp = g["DimReconcileBlock"](770, 1024, n_heads=4)
    assert mlp.use_attn is False, "770 %% 4 != 0，应退化为 MLP"
    for blk, d in ((attn, 768), (mlp, 770)):
        y = blk(torch.randn(2, 4, d))
        assert y.shape == (2, 4, 1024), (d, y.shape)
        assert torch.isfinite(y).all(), d
    print("[PASS] test_dim_reconcile_block_attn_or_mlp_fallback (768->Attn, 770->MLP)")
    return True


# ---------------------------------------------------------------- 4
def test_default_path_keeps_frozen_orthogonal_pm():
    """4. 默认路径仍是「统一正交冻结 P_m」——行为不变，可随时回退"""
    g = _ns()
    model = g["HeteroMoEProjSpace"](modal_dims=MODAL_DIMS, d_shared=256)
    assert hasattr(model, "P_m"), "默认路径应保留 P_m"
    assert len(model.P_m) == 3, len(model.P_m)
    for i, D in enumerate(MODAL_DIMS):
        p = model.P_m[i]
        assert tuple(p.shape) == (256, D), (i, p.shape)
        assert p.requires_grad is False, "P_m[%d] 应冻结" % i
        gram = p @ p.t()
        err = float((gram - torch.eye(256)).abs().max())
        assert err < 1e-4, "P_m[%d] 行不正交，误差 %g" % (i, err)
    assert model.d_shared == 256, model.d_shared
    print("[PASS] test_default_path_keeps_frozen_orthogonal_pm (3 组 [256,D] 正交冻结)")
    return True


# ---------------------------------------------------------------- 5
def test_pm_reconcile_path_replaces_p_m():
    """5. pm_reconcile 路径：d_shared 提到 pm_target，改用 modal_proj，不再有 P_m"""
    g = _ns()
    model = g["HeteroMoEProjSpace"](modal_dims=MODAL_DIMS, pm_reconcile=True, pm_target=1024)
    assert model.d_shared == 1024, model.d_shared
    assert not hasattr(model, "P_m"), "pm_reconcile 路径不应再创建 P_m"
    assert len(model.modal_proj) == 3, len(model.modal_proj)
    kinds = [type(m).__name__ for m in model.modal_proj]
    assert kinds[0] == "DimDownProj", kinds
    assert kinds[1] == "DimReconcileBlock" and kinds[2] == "DimReconcileBlock", kinds
    print("[PASS] test_pm_reconcile_path_replaces_p_m (d_shared=1024, %s)" % ",".join(kinds))
    return True


# ---------------------------------------------------------------- 6
def test_pm_reconcile_env_switch():
    """6. PM_RECONCILE / PM_TARGET 环境变量可切"""
    ns = K.exec_assigns(TRAIN, ["_USE_PMR", "_PM_TARGET"])
    assert ns["_USE_PMR"] is False, ns["_USE_PMR"]
    assert ns["_PM_TARGET"] == 1024, ns["_PM_TARGET"]
    ns = K.exec_assigns(TRAIN, ["_USE_PMR", "_PM_TARGET"],
                        env={"PM_RECONCILE": "1", "PM_TARGET": "2048"})
    assert ns["_USE_PMR"] is True, ns["_USE_PMR"]
    assert ns["_PM_TARGET"] == 2048, ns["_PM_TARGET"]
    print("[PASS] test_pm_reconcile_env_switch (默认 off/1024, 可切 on/2048)")
    return True


# ---------------------------------------------------------------- 7
def test_reconcile_forward_runs_on_real_shapes():
    """7. 调解路径在真实形状上能前向（2048/768/1152 三模态）"""
    g = _ns()
    model = g["HeteroMoEProjSpace"](modal_dims=MODAL_DIMS, pm_reconcile=True, pm_target=1024)
    model.eval()
    hidden = {
        "Qwen": torch.randn(2, 8, 2048),
        "BERT": torch.randn(2, 8, 768),
        "ViT": torch.randn(2, 8, 1152),
    }
    with torch.no_grad():
        y, alpha, aux = model(hidden)
    assert y.shape == (2, 8, model.d_space), y.shape
    assert alpha.shape == (2, 8, 4), alpha.shape
    assert torch.isfinite(y).all(), "输出含 NaN/Inf"
    assert model.d_space == 1024, model.d_space
    print("[PASS] test_reconcile_forward_runs_on_real_shapes (y %s, alpha %s, d_space=%d)"
          % (tuple(y.shape), tuple(alpha.shape), model.d_space))
    return True


TESTS = [
    test_dim_down_proj_reduces_to_target,
    test_dim_reconcile_block_dense_upprojection,
    test_dim_reconcile_block_attn_or_mlp_fallback,
    test_default_path_keeps_frozen_orthogonal_pm,
    test_pm_reconcile_path_replaces_p_m,
    test_pm_reconcile_env_switch,
    test_reconcile_forward_runs_on_real_shapes,
]

if __name__ == "__main__":
    from research._primitives.testing import run_suite
    sys.exit(run_suite(TITLE, TESTS))
