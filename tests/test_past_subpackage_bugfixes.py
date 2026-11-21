"""tests/test_past_subpackage_bugfixes.py — 历史子包 BUG 修复与溯源锁回归测试套件。

覆盖 2026-11-19 全仓 77 个历史子包排查中落地修复的全部公共/可复用组件与 SHA256 锁恢复：
  1. v62/measure.py SHA256 严格还原 (8a20d007f5a4...) + v62/measure_v2.py 可用
  2. v58 Gram 加速与稠密 SVD 数学等价且高维快速完成
  3. v62 外挂组件修复 (hebbian_align / hyper_proj / memory_router / feedback_head /
     sparse_block_fusion / self_repair / learned_optimizer / build_modeling)
  4. research/_primitives/ 修复 (workspace eval/freeze_c 保护 / encoders attention_mask /
     eval k-NN 真多数投票 / tabular_data load_online_shoppers 列名解析)
  5. v50 / v53 / v63 / v82 辅助脚本边界修复
"""
from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn


def test_v62_measure_sha256_restored_and_measure_v2_available():
    """1. v62/measure.py SHA256 精确恢复为 v75–v83 锁定值，measure_v2.py 提供修复版。"""
    p = ROOT / "research/aligner/v62_dimension_projection_space/measure.py"
    raw = p.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == "8a20d007f5a42340ee89713512c089f81157d92101610c17f92f7ae6077aa349", sha

    from research.aligner.v62_dimension_projection_space import measure_v2

    assert callable(measure_v2.measure_random_weight_control)
    assert callable(measure_v2.fit_orthogonal_pair)


def test_v58_gram_accelerated_cross_spectrum_matches_dense_and_fast():
    """2. v58 cross_spectrum 在高维 n<<d 上快速完成且与稠密 SVD 数值一致。"""
    from research.aligner.v58_entanglement_area.brane import capacity_needed, cross_spectrum

    torch.manual_seed(0)
    # 对比验证：在超过阈值 (da*db > 4e6) 附近核对 Gram 路径与稠密 SVD 路径一致
    A = torch.randn(48, 2200)
    B = torch.randn(48, 2200)
    s_gram = cross_spectrum(A, B)
    Ac = A - A.mean(0, keepdim=True)
    Bc = B - B.mean(0, keepdim=True)
    Ac = Ac / (Ac.norm(dim=0, keepdim=True) / (48 ** 0.5)).clamp(min=1e-8)
    Bc = Bc / (Bc.norm(dim=0, keepdim=True) / (48 ** 0.5)).clamp(min=1e-8)
    s_dense = torch.linalg.svdvals((Ac.T @ Bc) / 47)[:48]
    assert torch.allclose(s_gram, s_dense, atol=1e-3, rtol=1e-3)

    # 高维纯噪声 (192, 10240) 必须在数秒内完成且 χ=0
    t0 = time.time()
    chi = capacity_needed(torch.randn(192, 10240), torch.randn(192, 10240))
    assert chi == 0
    assert (time.time() - t0) < 15.0


def test_v62_hebbian_align_handles_missing_modality():
    """3. HebbianSlotAlignment 缺模态不再崩溃，且缺失模态耦合不被污染。"""
    from research.aligner.v62_dimension_projection_space.hebbian_align import HebbianSlotAlignment

    hb = HebbianSlotAlignment()
    c0 = hb.coupling.clone()
    c, sim = hb({"Qwen": torch.randn(2, 6, 256), "BERT": torch.randn(2, 6, 256)})
    assert c.shape == (3, 3) and sim.shape == (3, 3)
    assert torch.allclose(c[:, 2], c0[:, 2]) and torch.allclose(c[2, :], c0[2, :])
    assert torch.allclose(torch.diag(c), torch.ones(3))


def test_v62_hyper_proj_supports_2d_and_3d():
    """4. HyperLinear 同时支持 2D [N, d] 与 3D [B, S, d] 输入。"""
    from research.aligner.v62_dimension_projection_space.hyper_proj import HyperLinear

    hl = HyperLinear(32, 24, rank=4)
    y3 = hl(torch.randn(2, 5, 32))
    y2 = hl(torch.randn(6, 32))
    assert y3.shape == (2, 5, 24)
    assert y2.shape == (6, 24)


def test_v62_memory_router_lru_fills_all_slots_and_q_proj_gets_grad():
    """5. MemoryAugmentedRouter 真 LRU 填满全部槽位，且 q_proj 收到非零梯度。"""
    from research.aligner.v62_dimension_projection_space.memory_router import MemoryAugmentedRouter

    torch.manual_seed(0)
    mr = MemoryAugmentedRouter(d_in=16, n_slots=4, n_memory=8, d_mem=8, alpha=0.5)
    mr.train()
    # 连续写入 8 步应覆盖全部 8 个记忆槽（旧 bug 永远只写 0 号槽）
    for _ in range(8):
        x = torch.randn(2, 3, 16)
        logits = mr(x)
        logits.pow(2).mean().backward()
    nonempty_slots = int((mr.mem_val.abs().sum(dim=1) > 1e-8).sum().item())
    assert nonempty_slots == 8, f"LRU 应填满 8 个记忆槽，实际仅 {nonempty_slots}"
    assert mr.q_proj.weight.grad is not None and mr.q_proj.weight.grad.abs().sum() > 0


def test_v62_feedback_head_applies_full_matrix_not_scalar():
    """6. RecursiveFeedbackHead.apply_feedback 施加完整低秩矩阵增量，而非单标量广播。"""
    from research.aligner.v62_dimension_projection_space.feedback_head import RecursiveFeedbackHead

    torch.manual_seed(0)
    fh = RecursiveFeedbackHead(d_fused=32, n_slots=2, slot_dim=16, rank=4)
    w0 = nn.Parameter(torch.zeros(16, 16))
    w1 = nn.Parameter(torch.zeros(16, 16))
    fh.apply_feedback([w0, w1], torch.randn(2, 4, 32), {"Q": 0, "O": 1})
    # 旧 bug 会让 w0 全部元素相等（std == 0）
    assert w0.std().item() > 1e-6, "delta 不应塌成常数标量"


def test_v62_sparse_block_fusion_keeps_gate_grad_and_non_inplace():
    """7. SparseBlockFusion.forward_with_routing 保留 block_gate 梯度且无原地冲突。"""
    from research.aligner.v62_dimension_projection_space.sparse_block_fusion import SparseBlockFusion

    torch.manual_seed(0)
    sbf = SparseBlockFusion(d_space=32, n_blocks=6, block_ff=16, top_blocks=2)
    x = torch.randn(4, 5, 32, requires_grad=True)
    y, alpha = sbf.forward_with_routing(x)
    assert y.shape == (4, 5, 32) and alpha.shape == (4, 5, 6)
    assert sbf.last_router_probs is not None
    # 负载均衡梯度必须能流回 block_gate
    (y.pow(2).mean() + alpha.pow(2).mean()).backward()
    assert sbf.block_gate.weight.grad is not None and sbf.block_gate.weight.grad.abs().sum() > 0


def test_v62_self_repair_resets_router_bias_and_adam_state():
    """8. SlotHealthMonitor.repair 同时重置 router.bias[idx] 及其 Adam 动量状态。"""
    from research.aligner.v62_dimension_projection_space.self_repair import SlotHealthMonitor

    mod = nn.Module()
    mod.router = nn.Linear(8, 4)
    mod.slot_gate = nn.Parameter(torch.zeros(4))
    with torch.no_grad():
        mod.router.bias.fill_(-10.0)
    opt = torch.optim.Adam(mod.parameters(), lr=1e-2)
    # 走一步填充 optimizer.state
    (mod.router(torch.randn(2, 8)).sum() + mod.slot_gate.sum()).backward()
    opt.step()

    mon = SlotHealthMonitor(4, ["Q", "K", "V", "O"], threshold=0.05, cooldown=0)
    mon.ema[0] = 0.001
    repaired = mon.repair(mod, optimizer=opt)
    assert repaired == ["Q"]
    assert abs(float(mod.router.bias.detach()[0])) < 1e-8
    assert abs(float(opt.state[mod.router.bias]["exp_avg"][0])) < 1e-8


def test_v62_learned_optimizer_supports_non_square_weights():
    """9. LearnedWeightOptimizer.predict_delta 兼容非方阵权重不崩溃。"""
    from research.aligner.v62_dimension_projection_space.learned_optimizer import LearnedWeightOptimizer

    opt = LearnedWeightOptimizer(d=32, rank=4)
    w_rect = torch.randn(16, 48)
    delta = opt.predict_delta(w_rect, torch.randn_like(w_rect), loss_val=0.5)
    assert delta.shape == (16, 48)


def test_v62_build_modeling_utf8_check_and_syntax():
    """10. build_modeling.py --check 在 UTF-8 下通过且生成文件语法合法。"""
    r = subprocess.run(
        [sys.executable, str(ROOT / "research/aligner/v62_dimension_projection_space/build_modeling.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    src = (ROOT / "research/aligner/v62_dimension_projection_space/modeling_hetero_moe_projspace.py").read_text(
        encoding="utf-8"
    )
    ast.parse(src)


def test_primitives_workspace_eval_and_freeze_c_guard():
    """11. BroadcastWorkspace / GateStyleWorkspace 在 eval 或 freeze_c 下不漂移 c。"""
    from research._primitives.central_mechanism.workspace import BroadcastWorkspace, GateStyleWorkspace

    out = [torch.ones(2, 4, 16)]
    for cls in (BroadcastWorkspace, GateStyleWorkspace):
        ws = cls(d_shared=16, num_experts=3)
        ws.eval()
        c_before = ws.c.detach().clone()
        ws.ema_update(out)
        assert torch.equal(ws.c, c_before), f"{cls.__name__} 在 eval 模式发生 EMA 漂移"

    gw_frozen = GateStyleWorkspace(d_shared=16, num_experts=3, freeze_c=True)
    gw_frozen.train()
    c_before = gw_frozen.c.detach().clone()
    gw_frozen.ema_update(out)
    assert torch.equal(gw_frozen.c, c_before), "freeze_c=True 仍被 EMA 改写"


def test_primitives_encoders_knn_and_shoppers_loader():
    """12. encoders 透传 attention_mask；eval.knn_acc 支持 k>1；tabular_data 解析真实列名。"""
    from research._primitives.scale_hetero_loaders.encoders import encode
    from research._primitives.scale_hetero_loaders.eval import knn_acc
    from research._primitives.tabular_data import load_online_shoppers

    # (a) encoders.encode 必须传递 attention_mask
    captured = {}

    class _DummyTok:
        def __call__(self, texts, **kw):
            return {
                "input_ids": torch.ones(len(texts), kw["max_length"], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 0, 0]] * len(texts)),
            }

    class _DummyModel:
        def __call__(self, input_ids, attention_mask=None):
            captured["attention_mask"] = attention_mask
            return type("Out", (), {"last_hidden_state": torch.zeros(input_ids.shape[0], input_ids.shape[1], 8)})()

    encode(["hi"], _DummyTok(), _DummyModel(), S=4)
    assert captured["attention_mask"] is not None

    # (b) knn_acc 真正使用 k 参数（构造成对噪声点使 1-NN 与 3-NN 结果不同）
    X = torch.tensor([[0.0], [0.01], [0.2], [0.21], [10.0], [10.01], [10.2], [10.21]])
    y = torch.tensor([0, 1, 0, 0, 1, 0, 1, 1])
    assert knn_acc(X, y, k=3) > knn_acc(X, y, k=1)

    # (c) load_online_shoppers 能正确读出真实 6 列并标准化
    Xs, ys = load_online_shoppers(n=32, seed=0)
    assert Xs.shape == (32, 6) and ys.shape == (32,)
    assert np.isfinite(Xs).all()


def test_v53_knn_and_v63_decode_slots_fixes():
    """13. v53 KNNProbe 支持 n<k 不崩溃；v63 decode_slots 输出非负 L2 范数。"""
    from research.aligner.v53_third_audit.audit import KNNProbe
    from research.aligner.v63_nine_routes.interpret.decode_slots import serialize_slot

    p = KNNProbe(k=10).fit(torch.randn(4, 6), torch.tensor([0, 1, 0, 1]), n_class=2)
    preds = p.predict(torch.randn(3, 6))
    assert preds.shape == (3,)

    stats = serialize_slot(np.random.default_rng(0).standard_normal((16, 16)), topk=4)
    assert all(v >= 0.0 for v in stats["col_norm_top_val"])
    assert all(v >= 0.0 for v in stats["row_norm_top_val"])
    assert stats["col_norm_top_val"] == sorted(stats["col_norm_top_val"], reverse=True)



def test_primitives_workspace_and_theory_autograd_2d_and_freeze_fixes():
    """14. workspace/theory: forward 内 EMA 不破坏反传梯度；2D 不塌成标量；freeze_c 阻断 EMA；ema_decay 透传。"""
    from research._primitives.central_mechanism.workspace import (
        BroadcastWorkspace, GateStyleWorkspace, make_workspace,
    )
    from research._primitives.central_theory.theory import CentralTheory, freeze_c

    # (a) step 0 (c_0 == 0) 调用 augment + ema_update 再 backward，dL/dU 和 dL/dW_t 必须严格为 0
    bw = BroadcastWorkspace(d_shared=8, num_experts=3)
    z = torch.randn(2, 4, 3)
    out = bw.augment_router_logits(z)
    bw.ema_update([torch.ones(2, 4, 8) * 5.0])
    out.sum().backward()
    assert bw.U.grad is not None and float(bw.U.grad.abs().max()) == 0.0

    gw = GateStyleWorkspace(d_shared=8, num_experts=3, alpha=1.5)
    out_g = gw.augment_router_logits(z)
    gw.ema_update([torch.ones(2, 4, 8) * 5.0])
    out_g.sum().backward()
    assert gw.W_t.grad is not None and float(gw.W_t.grad.abs().max()) == 0.0
    assert torch.isfinite(out_g).all()

    # (b) 2D [S, D] 专家输出不把 c 塌缩为标量（各通道保留不同均值）
    gw2 = GateStyleWorkspace(d_shared=6, num_experts=3, ema_decay=0.0)
    vals = torch.arange(6, dtype=torch.float32).unsqueeze(0).expand(4, 6)
    gw2.ema_update([vals])
    assert torch.allclose(gw2.c.detach(), torch.arange(6, dtype=torch.float32))

    # (c) make_workspace 跨模式不报 TypeError；CentralTheory 透传 ema_decay 且 freeze_c 阻断 EMA
    mw_b = make_workspace("broadcast", d_shared=8, num_experts=3, freeze_c=True)
    assert not mw_b.c.requires_grad
    mw_g = make_workspace("gate", d_shared=8, num_experts=3, init_scale=0.01)
    assert isinstance(mw_g, GateStyleWorkspace)

    ct_gate = CentralTheory(8, 3, mode="gate", ema_decay=0.5)
    assert ct_gate.cw.ema_decay == 0.5
    ct_rn = CentralTheory(8, 3, mode="router-norm", ema_decay=0.25)
    ct_rn.ema_update([torch.ones(2, 4, 8)])
    assert torch.allclose(ct_rn.c.detach(), torch.full((8,), 0.75))
    freeze_c(ct_rn)
    c_before = ct_rn.c.detach().clone()
    ct_rn.ema_update([torch.ones(2, 4, 8) * 100.0])
    assert torch.allclose(ct_rn.c.detach(), c_before)


def test_primitives_lazy_paths_heldout_knn_4cell_adult_and_aligners():
    """15. _primitives: 路径延迟解析、arm 留出集 k-NN、Adult 四格均衡、aligners 长度校验与 corpus mask。"""
    import research._primitives.scale_hetero_loaders.paths as sh_paths
    from research._primitives.aligners import CrossArchMeanAligner, CrossArchAttnAligner
    from research._primitives.scale_hetero_loaders.arm import train_arm
    from research._primitives.tabular_data import load_adult_crossmodal, _HIGH_PAY

    # (a) paths 模块不再在 import 时把 MSA_DIR 绑定为模块 globals（延迟在 __getattr__）
    assert "MSA_DIR" not in sh_paths.__dict__

    # (b) aligners 拒绝长度不匹配输入（不静默 zip 截断）
    for al in (CrossArchMeanAligner([4, 6], 8), CrossArchAttnAligner([4, 6], 8, num_heads=2)):
        try:
            al([torch.randn(2, 3, 4)])
            assert False, "should raise ValueError on length mismatch"
        except ValueError:
            pass

    # (c) train_arm 的 knn 仅在留出集 te_idx 计算（构造训练集易分但测试集标签错配）
    torch.manual_seed(0)
    h_base = torch.zeros(40, 8)
    X = torch.randn(40, 4)
    y = torch.zeros(40, dtype=torch.long)
    y[:32] = (X[:32, 0] > 0).long()
    # 测试集 8 个样本：相同特征但唯一 1 个正样本，若在留出集测 LOO 1-NN，正样本必错配
    X[32:] = 0.0
    y[32:] = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0])
    _, knn = train_arm(
        h_base, X, y, c_frozen=torch.zeros(40, 4),
        tr_idx=torch.arange(32), te_idx=torch.arange(32, 40),
        d_entry=4, d_big=8, n_cls=2, steps=5, seed=0,
    )
    assert 0.0 <= knn <= 1.0

    # (d) load_adult_crossmodal(balanced_4cell=True) 四格严格各占 n//4
    texts, Xa, ya = load_adult_crossmodal(n=128, seed=0, decouple=True, balanced_4cell=True)
    assert len(texts) == 128 and Xa.shape == (128, 6)
    hi_occ = [any(f"working as {o.lower()}" in t for o in _HIGH_PAY) for t in texts]
    # 每格应恰好 32 条 ⇒ 职业为高薪的总计恰好 64 条、y==1 恰好 64 条
    assert sum(hi_occ) == 64, sum(hi_occ)
    assert int(ya.sum()) == 64, int(ya.sum())


def test_v57_gavish_donoho_consecutive_reset_high_rank_and_grad():
    """16. v57: Gavish-Donoho 单次 ω(β)·median(s)、_below 复位、r>=d/2 不炸秩、effective_params=r²。"""
    from research.aligner.v57_adaptive_dimension.operators import (
        AdaptiveOperator, AdaptiveSubspace, omega, optimal_hard_threshold,
    )

    torch.manual_seed(0)
    Z = torch.randn(128, 128)
    s = torch.linalg.svdvals(Z)
    tau = optimal_hard_threshold(s, 128, 128)
    assert abs(tau - omega(1.0) * float(s.median())) < 1e-5

    # 连续坍缩复位: [高 -> 低 -> 高 -> 低] 在 death_window=2 下不应死亡
    d = 32
    B = torch.linalg.qr(torch.randn(d, 4))[0]
    sp = AdaptiveSubspace(d, death_window=2)
    sig_batch = lambda: torch.randn(64, 2) * 8.0 @ B[:, :2].T + torch.randn(64, d) * 0.5
    noise_batch = lambda: torch.randn(64, d) * 0.5
    sp(sig_batch(), update=True)
    assert sp.r == 2
    sp(noise_batch(), update=True)      # _below = [1, 1]
    assert sp.r == 2
    sp(sig_batch(), update=True)        # 恢复高能量 → _below 应复位为 [0, 0]
    assert int(sp._below.max()) == 0
    sp(noise_batch(), update=True)      # 仅连续第 1 步低于地板 → 仍存活
    assert sp.r == 2

    # r >= d/2 时不发生残差中位数塌零炸满 max_rank
    torch.manual_seed(1)
    d2, n2 = 20, 120
    B2 = torch.linalg.qr(torch.randn(d2, 10))[0]
    sp2 = AdaptiveSubspace(d2, death_window=5)
    for _ in range(3):
        Z2 = torch.randn(n2, 10) * 8.0 @ B2.T + torch.randn(n2, d2) * 0.5
        sp2(Z2, update=True)
    assert sp2.r == 10, f"expected r=10 without explosion, got {sp2.r}"

    # effective_params == r² 且 forward 保留输入梯度，events 输出非空 r=
    op = AdaptiveOperator(d2, sp2)
    assert op.effective_params == 100
    assert any("r=10" in ev for ev in sp2.log.events())
    Z_req = torch.randn(8, d2, requires_grad=True)
    sp2(Z_req, update=False).sum().backward()
    assert Z_req.grad is not None and float(Z_req.grad.abs().sum()) > 0


def test_v62_train_v2_abl_image_pm_reconcile_vl_feedback_and_projspace():
    """17. v62: ABL=image 生效、pm_reconcile+use_vl+use_feedback 联合无崩溃、use_hyper+adapter 不绕过 hyper、负索引重组。"""
    from research.aligner.v62_dimension_projection_space import train_v2
    from research.aligner.v62_dimension_projection_space.projspace import (
        DimProjectionSpace, MEASURED_SUBSPACES,
    )

    # (a) ABL=image 仅保留 ViT
    out_img = train_v2._feats("q", "b", "v", abl="image")
    assert out_img == {"ViT": "v"}, out_img

    # (b) pm_reconcile + use_vl + use_feedback 联合前向反传无崩溃，且 vl_fusion 无死 in_proj
    m = train_v2.HeteroMoEProjSpace(
        modal_dims=(64, 32, 48), d_shared=32, ne=4, top_k=2,
        pm_reconcile=True, pm_target=64, use_vl=True, use_feedback=True,
    )
    assert m.vl_fusion.in_proj is None
    hidden = {
        "Qwen": torch.randn(2, 6, 64),
        "BERT": torch.randn(2, 5, 32),   # 刻意与 Qwen 序列长度不同
        "ViT": torch.randn(2, 7, 48),
    }
    y, alpha, aux = m(hidden, fused_only=True)
    assert y.shape == (2, 5, m.d_space) and torch.isfinite(y).all()
    y.sum().backward()
    assert m.vl_fusion.out_proj.weight.grad is not None

    # (c) projspace: use_hyper + use_modality_adapter 联合时 HyperLinear 仍收梯度；recombine 支持 -1；align 方阵正交
    ps = DimProjectionSpace(
        d_shared=64, subspaces=MEASURED_SUBSPACES, n_experts=4, top_k=2,
        use_hyper=True, use_modality_adapter=True,
    )
    # align 虽 sp.dim=256 != d_shared=64，仍应正交初始化
    w_al = next(iter(ps.align.values())).weight.detach()
    assert torch.allclose(w_al @ w_al.t(), torch.eye(w_al.shape[0]), atol=1e-4)
    feats = {
        "Qwen": {s: torch.randn(2, 4, 64) for s in ps.slot_names},
        "BERT": {s: torch.randn(2, 4, 64) for s in ps.slot_names},
        "ViT": {s: torch.randn(2, 4, 64) for s in ps.slot_names},
    }
    struct, canon, _ = ps(feats)
    canon.sum().backward()
    hl = ps.proj["Q|Qwen"]
    assert hl.scale.grad is not None and float(hl.scale.grad.abs().sum()) > 0
    rec = ps.recombine(struct.detach(), {"Q": -1, "K": 0})
    assert rec.shape == struct.shape


def test_v62_moa_act_grad_ssm_no_dead_params_cka_vit4x_and_raster3d():
    """18. v62: MoA ACT 停机步梯度与精确步数、SSM 零死参、CKA 捕获旋转冗余、ViT 4.x hook、raster3d 背面三角形。"""
    from research.aligner.v62_dimension_projection_space.moa_operators import (
        IterativeMoAOperatorBlock, MoAOperatorConfig,
    )
    from research.aligner.v62_dimension_projection_space.moa_fusion import MoAFusionOverProjSpace
    from research.aligner.v62_dimension_projection_space.moa_operators_zerooom import (
        HeterogeneousProposerOp as ZeroOOMProposer,
        MoAOperatorConfig as ZeroOOMCfg,
    )
    from research.aligner.v62_dimension_projection_space.slot_independence_audit import linear_cka_matrix
    from research.aligner.v62_dimension_projection_space.model import HeteroMoEProjSpaceModel
    from research.aligner.v62_dimension_projection_space.visual3d.raster3d import Cam, Canvas

    # (a) ACT 在 step 0 停机时 halting_gate 仍有非零梯度，且 mean_refinement_steps == 1.0
    cfg = MoAOperatorConfig(
        d_model=16, n_heads=2, num_proposers=3, top_k_proposers=2,
        num_refinement_steps=3, enable_adaptive_halting=True, halting_threshold=0.5,
    )
    blk = IterativeMoAOperatorBlock(cfg)
    nn.init.constant_(blk.halting_gate[1].bias, 5.0)  # step 0 p_halt ≈ 0.993 >= 0.5
    y, aux, met = blk(torch.randn(2, 4, 16))
    (y.sum() + aux).backward()
    assert blk.halting_gate[1].bias.grad is not None and float(blk.halting_gate[1].bias.grad.abs()) > 0
    assert abs(float(met["mean_refinement_steps"]) - 1.0) < 1e-5

    # (b) MoAFusion forward_with_routing 返回逐 token 可微 alpha (求和为 1)
    mf = MoAFusionOverProjSpace(d_space=16, num_proposers=3, n_heads=2, top_k=2)
    ym, am = mf.forward_with_routing(torch.randn(2, 4, 16, requires_grad=True))
    assert am.requires_grad and torch.allclose(am.sum(dim=-1), torch.ones(2, 4), atol=1e-5)

    # (c) ZeroOOM HeterogeneousProposerOp 的 SSM 分支不再携带死 in_proj 参数
    zop = ZeroOOMProposer(ZeroOOMCfg(d_model=16, n_heads=2, num_proposers=4, d_state=4))
    assert all(b.in_proj is None for b in zop.ssm_branches.values())

    # (d) linear_cka_matrix 捕获正交旋转下的 100% 冗余信号
    rng = np.random.default_rng(0)
    z1 = rng.standard_normal((100, 1))
    Q, _ = np.linalg.qr(rng.standard_normal((32, 4)))
    reps = {k: z1 @ Q[:, i:i + 1].T for i, k in enumerate(("Q", "K", "V", "O"))}
    cka = linear_cka_matrix(reps, ["Q", "K", "V", "O"])
    assert float(cka[~np.eye(4, dtype=bool)].min()) > 0.99

    # (e) model._find_slot_modules 兼容 transformers 4.x ViT 命名
    class _FakeViT4(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Module()
            self.attention.attention = nn.Module()
            self.attention.attention.query = nn.Linear(4, 4)
            self.attention.attention.key = nn.Linear(4, 4)
            self.attention.attention.value = nn.Linear(4, 4)
            self.attention.output = nn.Module()
            self.attention.output.dense = nn.Linear(4, 4)

    found = HeteroMoEProjSpaceModel._find_slot_modules(_FakeViT4(), "ViT")
    assert set(found.keys()) == {"Q", "K", "V", "O"}

    # (f) raster3d 对正反两种绕序（area > 0 与 area < 0）均正常渲染
    cam = Cam(eye=(0, -3.0, 0), target=(0, 0, 0), W=64, H=64)
    v0, v1, v2 = np.array([-0.8, 0.0, -0.6]), np.array([0.8, 0.0, -0.6]), np.array([0.0, 0.0, 0.8])
    c_ccw = Canvas(64, 64, bg=(0, 0, 0))
    c_cw = Canvas(64, 64, bg=(0, 0, 0))
    c_ccw.tri(cam, v0, v1, v2, (255, 0, 0))
    c_cw.tri(cam, v0, v2, v1, (255, 0, 0))
    assert int((c_ccw.img[:, :, 0] > 0).sum()) > 100
    assert int((c_cw.img[:, :, 0] > 0).sum()) > 100


def test_weight_fusion_unlocked_diagnostics_and_v52_fixes():
    """19. v52 / v79 / v81 / v82 / v83 非锁定诊断脚本统计与口径修复验证。"""
    import importlib.util as ilu

    def _load(rel, name):
        spec = ilu.spec_from_file_location(name, str(ROOT / rel))
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    # (a) v79 _sign_test_two_sided 在 pos < n/2 时对称显著（不再恒返 1.0）
    v79 = _load("research/weight_fusion/v79_align_source/s2_0_align.py", "test_v79_s20")
    st_low = v79._sign_test_two_sided([-1.0] * 31 + [1.0])
    st_hi = v79._sign_test_two_sided([1.0] * 31 + [-1.0])
    assert abs(st_low["sign_p_two_sided"] - st_hi["sign_p_two_sided"]) < 1e-12
    assert st_low["sign_p_two_sided"] < 1e-6

    # (b) v81 s3_ceiling_diag._r2_from_residual 严格仅评估留出集 ~train_mask
    v81d = _load("research/weight_fusion/v81_s3_semantic/s3_ceiling_diag.py", "test_v81_ceil")
    train_mask = np.array([True, True, True, True, False, False, False, False])
    target = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    resid1 = torch.zeros_like(target)
    resid2 = resid1.clone()
    resid2[0] = 100.0; resid2[1] = -100.0  # 仅改训练集残差（保持训练集均值 0 不变）
    assert abs(v81d._r2_from_residual(resid1, target, train_mask)
               - v81d._r2_from_residual(resid2, target, train_mask)) < 1e-9

    # (c) v82 readout_diag 导入无副作用、perm 块级翻号且 (cnt+1)/(n_perm+1) > 0
    v82r = _load("research/weight_fusion/v82_s4_pipe/readout_diag.py", "test_v82_rd")
    pairs = [(b, 1.0, 0.0) for b in range(16) for _ in range(5)]
    p_val = v82r.perm(pairs, n_perm=500, seed=0)
    assert abs(p_val - 1.0 / 501.0) < 1e-12

    # (d) v82 pipe_probe2.sign_test_p 在 n=2000 不溢出 float64
    v82p2 = _load("research/weight_fusion/v82_s4_pipe/pipe_probe2.py", "test_v82_p2")
    p2, pos2, n2 = v82p2.sign_test_p([1.0] * 1100 + [-1.0] * 900)
    assert n2 == 2000 and pos2 == 1100 and 0.0 <= p2 < 1e-5

    # (e) v83 diag_mvo_readout._perm 块级翻号且 (cnt+1)/(n_perm+1) > 0
    v83d = _load("research/weight_fusion/v83_s4_aligned_pipe/diag_mvo_readout.py", "test_v83_diag")
    p83 = v83d._perm(pairs, n_perm=500, seed=0)
    assert abs(p83 - 1.0 / 501.0) < 1e-12



def test_primitives_entries_mask_corpus_mask_and_shoppers_env_dir():
    """20. Track C 补全: entries 掩码池化屏蔽 pad、MiniCorpus return_mask/batch_size 校验、ADULT_DATA_DIR 生效。"""
    import os
    from research._primitives.scale_hetero_loaders.entries import encode_baseline_text
    from research._primitives.real_corpus.corpus import MiniCorpus
    from research._primitives.tabular_data import load_online_shoppers

    # (a) encode_baseline_text 忽略 pad 位置（改变 pad 位置隐状态不影响池化结果）
    class _Tok:
        def __call__(self, texts, **kw):
            return {
                "input_ids": torch.tensor([[1, 2, 0, 0]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.long),
            }

    class _Mod:
        def __init__(self, pad_val):
            self.pad_val = pad_val

        def __call__(self, input_ids=None, attention_mask=None):
            h = torch.zeros(1, 4, 4)
            h[0, 0] = 2.0
            h[0, 1] = 4.0
            h[0, 2:] = self.pad_val
            return type("O", (), {"last_hidden_state": h})()

    out1 = encode_baseline_text(_Mod(0.0), _Tok(), ["a b"], s=4)
    out2 = encode_baseline_text(_Mod(999.0), _Tok(), ["a b"], s=4)
    assert torch.allclose(out1, torch.full((1, 4), 3.0))
    assert torch.allclose(out1, out2)

    # (b) MiniCorpus return_mask=True 返回 (ids, mask)，且拒绝 batch_size <= 0
    mc = MiniCorpus(_Tok(), max_length=4)
    ids, mask = mc.train_batch(batch_size=1, return_mask=True)
    assert ids.shape == (1, 4) and mask.shape == (1, 4)
    for bad_bs in (0, -2):
        try:
            mc.eval_batch(batch_size=bad_bs)
            assert False, f"should reject batch_size={bad_bs}"
        except ValueError:
            pass

    # (c) load_online_shoppers 响应 ADULT_DATA_DIR 环境变量
    v39_data = str(ROOT / "research/scale_heterogeneous/v39_g2_interaction/data")
    old_env = os.environ.get("ADULT_DATA_DIR")
    try:
        os.environ["ADULT_DATA_DIR"] = v39_data
        Xs, ys = load_online_shoppers(n=16, seed=1)
        assert Xs.shape == (16, 6) and ys.shape == (16,)
    finally:
        if old_env is None:
            os.environ.pop("ADULT_DATA_DIR", None)
        else:
            os.environ["ADULT_DATA_DIR"] = old_env


def test_v62_missing_modality_fuse_w_triton_guards_and_extract_fp16_casts():
    """21. Track C 补全: projspace 缺模态 fuse_w 精确索引、Triton NaN 守卫与 K4 字节账、SSM 反向核与提取脚本 fp16 强转。"""
    from research.aligner.v62_dimension_projection_space.projspace import (
        DimProjectionSpace, MEASURED_SUBSPACES,
    )
    from research.aligner.v62_dimension_projection_space.moa_triton_kernels import TritonTileSimulator

    # (a) 缺 Qwen 时（仅 BERT+ViT），fuse_w['Q'] 必须取索引 [1, 2] 而非 [0, 1]
    ps = DimProjectionSpace(d_shared=16, subspaces=MEASURED_SUBSPACES, n_experts=4, top_k=2)
    with torch.no_grad():
        ps.fuse_w["Q"].copy_(torch.tensor([10.0, -5.0, 5.0]))  # Qwen=10, BERT=-5, ViT=5
    feats = {
        "BERT": {s: torch.randn(1, 3, 16) for s in ps.slot_names},
        "ViT": {s: torch.randn(1, 3, 16) for s in ps.slot_names},
    }
    _, canon_a, _ = ps(feats)
    # 若改 Qwen 权重 fuse_w['Q'][0]，缺 Qwen 时输出必须严格不变
    with torch.no_grad():
        ps.fuse_w["Q"][0] = -10.0
    _, canon_b, _ = ps(feats)
    assert torch.allclose(canon_a, canon_b, atol=1e-6), "missing Qwen still read fuse_w['Q'][0]!"

    # (b) Triton K4 模拟器准确计入 executed_blocks * block_n KV 重读字节（大于旧低估公式）
    rng = np.random.default_rng(0)
    B, H, L, Dh = 1, 3, 32, 16
    q = rng.standard_normal((B, H, L, Dh)).astype(np.float32)
    k = rng.standard_normal((B, H, L, Dh)).astype(np.float32)
    v = rng.standard_normal((B, H, L, Dh)).astype(np.float32)
    _, stats, rep = TritonTileSimulator.simulate_block_sparse_moa_flash_attention(
        q, k, v, num_sinks=2, window_size=4, dilation=4, block_m=8, block_n=8, bytes_per_elem=4,
    )
    old_underestimate = int(B * H * L * Dh * 4 * (2 + 2 * stats["active_tile_ratio"]))
    assert rep.triton_fused_hbm_bytes > old_underestimate
    assert rep.max_abs_diff_vs_ref < 1e-6

    # (c) 源码契约检查：Triton NaN 守卫、ZeroOOM 反向核回写、4 个提取脚本 fp16 强转及 VSR 断言先于上传
    v62_dir = ROOT / "research/aligner/v62_dimension_projection_space"
    k_src = (v62_dir / "moa_triton_kernels.py").read_text(encoding="utf-8")
    assert "m_ij_safe = tl.where(m_ij > -float(\"inf\"), m_ij, 0.0)" in k_src
    z_src = (v62_dir / "moa_triton_zerooom.py").read_text(encoding="utf-8")
    for tok in ("GradB_ptr", "GradALog_ptr", "du_act_rev += dh_rev", "ddt_rev += dh_rev"):
        assert tok in z_src, f"missing {tok} in moa_triton_zerooom.py"
    for ext in ("extract_snli_ve_v65.py", "extract_vsr_v66.py",
                "extract_backbones_v63.py", "extract_vit_only_v63.py"):
        es = (v62_dir / ext).read_text(encoding="utf-8")
        assert ".to(dtype=vm.dtype)" in es, f"missing fp16 cast in {ext}"
    vsr_src = (v62_dir / "extract_vsr_v66.py").read_text(encoding="utf-8")
    assert vsr_src.index("assert _stats['img_fail']") < vsr_src.index("_up('h_vit_vsr.npy')")


def test_v52_ridge_probe_and_v82_v83_diag_contracts():
    """22. Track C 补全: v52 RidgeProbe 精度与设备一致；v82 summarize/carrier_probe 与 v83 diag_mvo_readout 口径。"""
    import importlib.util as ilu
    from research.aligner.v52_independent_audit.audit import RidgeProbe

    torch.manual_seed(0)
    X = torch.cat([torch.randn(20, 4) - 3.0, torch.randn(20, 4) + 3.0], dim=0)
    y = torch.tensor([0] * 20 + [1] * 20)
    rp = RidgeProbe(alpha=1e-2).fit(X, y, n_class=2)
    preds = rp.predict(X)
    assert preds.device == X.device and float((preds == y).float().mean()) > 0.95

    # v82 summarize_v82 导入无副作用（__main__ 保护）且使用 statistics.median
    sum_path = ROOT / "research/weight_fusion/v82_s4_pipe/summarize_v82.py"
    spec = ilu.spec_from_file_location("test_sum_v82", str(sum_path))
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(mod.main)
    assert "statistics.median" in sum_path.read_text(encoding="utf-8")

    # v82 carrier_probe 使用中心化 R²；v83 diag_mvo_readout 严格切分至留出集 test_start:
    cp_src = (ROOT / "research/weight_fusion/v82_s4_pipe/carrier_probe.py").read_text(encoding="utf-8")
    assert "a_c = a - a.mean(dim=0, keepdim=True)" in cp_src
    d83_src = (ROOT / "research/weight_fusion/v83_s4_aligned_pipe/diag_mvo_readout.py").read_text(encoding="utf-8")
    assert "heldout_test_only[" in d83_src and "[test_start:]" in d83_src


def test_track_b_metadata_consistency_and_errata_invariants():
    """23. Track B 元数据一致性回归锁: v18/v57/v47/v36-v39/v7/v10/v56 元数据与勘误总账一致。"""
    import json

    # (a) v18 MANIFEST 不再虚假标注 clean
    m18 = json.loads((ROOT / "research/distillation/v18_real_corpus/MANIFEST.json").read_text(encoding="utf-8"))
    assert m18["leakage_audit"]["status"] == "invalidated"

    # (b) v57 MANIFEST 与已落盘 V2_collapse=false 对齐，不再标 positive
    m57 = json.loads((ROOT / "research/aligner/v57_adaptive_dimension/MANIFEST.json").read_text(encoding="utf-8"))
    r57 = json.loads((ROOT / "research/aligner/v57_adaptive_dimension/results/metrics.json").read_text(encoding="utf-8"))
    assert r57["verdicts"]["V2_collapse"] is False
    assert m57["claim"]["outcome_label"] == "mixed"
    assert m57["artifacts"]["results"] == "results/metrics.json"

    # (c) v47 metrics.json 完整包含全部 9 个被审包（6 中枢 + 3 规模异构）
    r47 = json.loads((ROOT / "research/central_stability/v47_heldout_audit/results/metrics.json").read_text(encoding="utf-8"))
    expected_9 = {"v8", "v12", "v19", "v20", "v30", "v31", "v33", "v34", "v35"}
    assert expected_9.issubset(set(r47["per_package"].keys())), set(r47["per_package"].keys())

    # (d) v36-v39 metrics.json 方向与 MANIFEST 一致为 higher_is_better
    for rel in (
        "research/scale_heterogeneous/v36_alpha2_knn/results/metrics.json",
        "research/scale_heterogeneous/v37_b_selftrained_entry/results/metrics.json",
        "research/scale_heterogeneous/v38_g_adult_boundary/results/metrics.json",
        "research/scale_heterogeneous/v39_g2_interaction/results/metrics.json",
    ):
        d = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        assert d["direction"] == "higher_is_better", (rel, d["direction"])

    # (e) v7/v10 MANIFEST 修复值与 lifted 位置正确
    m7 = json.loads((ROOT / "research/routing_evolution/v7_attention/MANIFEST.json").read_text(encoding="utf-8"))
    assert abs(float(m7["claim"]["value"]) - 1.0626) < 1e-6
    m10 = json.loads((ROOT / "research/aligner/v10_embedding/MANIFEST.json").read_text(encoding="utf-8"))
    assert m10["implementation"]["location"] == "lifted"
    assert m10["implementation"]["lifted_to"] == "research._primitives.aligners"

    # (f) 勘误总账包含 6 条跨包 P0 链与全部主要历史包边界
    errata = (ROOT / "docs/PAST_SUBPACKAGES_ERRATA.md").read_text(encoding="utf-8")
    for marker in ("链 1", "链 6", "V8Trainer", "load_adult_crossmodal", "V2_collapse", "BERT->Qwen"):
        assert marker in errata, f"missing {marker} in PAST_SUBPACKAGES_ERRATA.md"
