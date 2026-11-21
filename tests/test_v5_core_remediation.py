"""tests/test_v5_core_remediation.py — 全历史固有缺陷修复综合回归测试套件。

覆盖核心主包 hetero_moe v5.0、v62/v65 修复、防泄漏下游门禁以及工程治理工具补盲。
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from hetero_moe import (
    DownstreamEvalGate,
    ExpertSpec,
    HeteroFusionLayer,
    HeteroTrainer,
    IdentifiabilityGuard,
    InferenceRunner,
    SplitGuard,
)


def test_v5_true_heterogeneous_experts_different_widths_and_archs():
    """修复 1 (伪异构): 不同架构/不同宽度/外部自定义模块在同一 HeteroFusionLayer 共存并可微训练。"""
    torch.manual_seed(0)

    class CustomFrozenBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(18, 27), nn.ReLU(), nn.Linear(27, 22))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    specs = [
        ExpertSpec("swiglu_wide", arch="swiglu", d_in=20, d_ff=48, d_out=28, freeze_base=True, lora_rank=4),
        ExpertSpec("gelu_narrow", arch="gelu_mlp", d_in=24, d_ff=36, d_out=16, freeze_base=True, lora_rank=2),
        ExpertSpec("conv_gated", arch="conv_mlp", d_in=32, d_ff=40, d_out=32, freeze_base=True, lora_rank=0),
        ExpertSpec("custom_mod", arch="custom", d_in=18, d_out=22, freeze_base=True, lora_rank=4, module=CustomFrozenBlock()),
    ]
    layer = HeteroFusionLayer(d_model=32, d_ff=64, num_experts=4, top_k=2, gate_rank=4, expert_specs=specs)
    assert layer.is_heterogeneous is True
    assert len(layer.experts) == 4
    # 验证底座参数确实冻结，而 LoRA 与异构投影可训练
    for exp in layer.experts:
        assert all(not p.requires_grad for p in exp.base.parameters())
    x = torch.randn(3, 10, 32, requires_grad=True)
    y, z, ah = layer(x)
    assert y.shape == (3, 10, 32) and z.shape == (3, 10, 4) and ah.shape == (3, 10, 4)
    y.pow(2).mean().backward()
    assert x.grad is not None and x.grad.norm() > 0
    assert layer.W_router.grad is not None and layer.W_router.grad.norm() > 0


def test_v5_true_sparse_dispatch_matches_dense_and_skips_inactive_experts():
    """修复 2 (伪稀疏): 真稀疏分派与稠密数值一致，且零命中专家彻底跳过不执行。"""
    torch.manual_seed(7)
    D, D_FF, M, K = 32, 64, 8, 2
    l_sp = HeteroFusionLayer(D, D_FF, M, K, sparse_dispatch=True)
    l_dn = HeteroFusionLayer(D, D_FF, M, K, sparse_dispatch=False)
    l_dn.load_state_dict(l_sp.state_dict())

    x = torch.randn(4, 16, D)
    y_sp, z_sp, ah_sp = l_sp(x)
    y_dn, z_dn, ah_dn = l_dn(x)
    assert torch.allclose(y_sp, y_dn, atol=1e-5), f"稀疏/稠密偏差: {(y_sp - y_dn).abs().max()}"
    assert sum(l_sp.last_expert_token_counts) == 4 * 16 * K
    assert sum(l_dn.last_expert_token_counts) == 4 * 16 * M

    # 强制全部路由仅命中专家 0，验证专家 1 的模块一次都未被调用
    calls = [0, 0]

    class CountingMod(nn.Module):
        def __init__(self, idx: int):
            super().__init__()
            self.idx = idx
            self.lin = nn.Linear(D, D)

        def forward(self, inp: torch.Tensor) -> torch.Tensor:
            calls[self.idx] += 1
            return self.lin(inp)

    l_cnt = HeteroFusionLayer(
        D, D_FF, 2, 1, sparse_dispatch=True,
        expert_specs=[
            ExpertSpec("e0", "custom", d_in=D, d_out=D, module=CountingMod(0)),
            ExpertSpec("e1", "custom", d_in=D, d_out=D, module=CountingMod(1)),
        ],
    )
    with torch.no_grad():
        l_cnt.W_router.zero_()
        l_cnt.W_router[0].fill_(10.0)
    l_cnt(torch.ones(2, 6, D))
    assert calls == [1, 0], f"未命中专家应跳过，实际调用次数 {calls}"


def test_v5_grad_checkpoint_matches_exact_gradients_and_cross_layer_flow():
    """修复 3 (detach 显存陈述错误): grad_checkpoint 梯度严格等价，detach_input=False 保留跨层 FFN 梯度。"""
    torch.manual_seed(11)
    D, D_FF, M, K = 24, 48, 4, 2
    l_base = HeteroFusionLayer(D, D_FF, M, K, detach_input=False, grad_checkpoint=False, lora_rank=4)
    l_ckpt = HeteroFusionLayer(D, D_FF, M, K, detach_input=False, grad_checkpoint=True, lora_rank=4)
    l_det = HeteroFusionLayer(D, D_FF, M, K, detach_input=True, grad_checkpoint=False, lora_rank=4)
    l_ckpt.load_state_dict(l_base.state_dict())
    l_det.load_state_dict(l_base.state_dict())

    x0 = torch.randn(2, 8, D)
    x1 = x0.clone().requires_grad_(True)
    x2 = x0.clone().requires_grad_(True)
    x3 = x0.clone().requires_grad_(True)

    l_base(x1)[0].pow(2).mean().backward()
    l_ckpt(x2)[0].pow(2).mean().backward()
    l_det(x3)[0].pow(2).mean().backward()

    assert torch.allclose(x1.grad, x2.grad, atol=1e-5)
    for p1, p2 in zip(l_base.trainable_parameters, l_ckpt.trainable_parameters):
        if p1.grad is not None or p2.grad is not None:
            assert torch.allclose(p1.grad, p2.grad, atol=1e-5)
    # detach_input=False 时 x 额外收到穿过专家 FFN 的梯度（默认 0.02^3*0.05 量级约 ~1e-6，严格非零）
    assert (x1.grad - x3.grad).abs().max().item() > 1e-8


def test_v5_vector_gate_breaks_v55_theorem3_bandwidth_ceiling():
    """修复 4 (v55 Theorem 3 路由低秩瓶颈): gate_rank>0 将 J_route 秩从 <= min(K,M-1) 提升至 >= gate_rank。"""
    torch.manual_seed(3)
    D, D_FF, M, K = 32, 64, 4, 2
    x0 = torch.randn(1, 1, D)
    l_s = HeteroFusionLayer(D, D_FF, M, K, quant_bits=8, gate_rank=0, detach_input=True, sparse_dispatch=False)
    l_v = HeteroFusionLayer(D, D_FF, M, K, quant_bits=8, gate_rank=8, detach_input=True, sparse_dispatch=False)
    with torch.no_grad():
        for lyr in (l_s, l_v):
            for p in lyr.alphas:
                p.fill_(1.0)
            for wg, wu, wd in zip(lyr.w_gates, lyr.w_ups, lyr.w_downs):
                wg.normal_(0, 0.5)
                wu.normal_(0, 0.5)
                wd.normal_(0, 0.5)
        for u in l_v.gate_u:
            u.normal_(0, 0.3)

    def _rank(lyr):
        def _fn(v):
            return (lyr(v.view(1, 1, D))[0] - v.view(1, 1, D)).reshape(D)

        sv = torch.linalg.svdvals(torch.autograd.functional.jacobian(_fn, x0.reshape(D)))
        return int((sv > 1e-4 * sv.max()).sum().item())

    r_s = _rank(l_s)
    r_v = _rank(l_v)
    assert r_s <= min(K, M - 1), f"标量门控应服从 Theorem 3 (<=2), 实测 {r_s}"
    assert r_v >= 8 > min(K, M - 1), f"向量门控应突破 Theorem 3 (>2), 实测 {r_v}"


def test_v5_phase1_zero_z_loss_contamination():
    """修复 5 (Phase 1 z_loss 梯度污染): Phase 1 改变 lam_z 对前层适配器梯度零影响。"""
    def _l0_grad(lam_z: float) -> torch.Tensor:
        torch.manual_seed(5)
        layers = nn.ModuleList([HeteroFusionLayer(16, 32, 4, 2) for _ in range(2)])
        head = nn.Linear(16, 12, bias=False)
        tr = HeteroTrainer(layers, lm_head=head, lam_z=lam_z)
        tr.begin_phase(1)
        x = torch.randn(2, 6, 16)
        tgt = torch.randint(0, 12, (2, 6))
        tr.optimizer.zero_grad(set_to_none=True)
        y, louts = tr.forward(x)
        tr.compute_loss(y, tgt, louts)["total"].backward()
        return torch.cat([p.grad.detach().reshape(-1) for p in (*layers[0].gammas, *layers[0].betas, *layers[0].alphas)])

    g0 = _l0_grad(0.0)
    g100 = _l0_grad(100.0)
    assert torch.equal(g0, g100), f"Phase 1 z_loss 仍污染前层梯度: {(g0 - g100).abs().max()}"


def test_v5_renorm_topk_matches_inference_capacity_route_scale():
    """修复 6 (训练-推理幅值失配 + K=1 STE 梯度直通): 默认 renorm_topk=True 使路由权重和恒为 1，且 top_k=1 时 W_router 梯度非零。"""
    torch.manual_seed(9)
    layer = HeteroFusionLayer(16, 32, 4, 2)
    assert layer.renorm_topk is True
    _, _, ah = layer(torch.randn(3, 8, 16), phase1=False)
    assert torch.allclose(ah.sum(dim=-1), torch.ones(3, 8), atol=1e-5)

    # K=1 且 renorm_topk=True 时，W_router 必须收到非零任务梯度（修复商法则在 K=1 下恒零的缺陷）
    layer_k1 = HeteroFusionLayer(16, 32, 4, 1, renorm_topk=True)
    y_k1, _, ah_k1 = layer_k1(torch.randn(3, 8, 16), phase1=False)
    assert torch.allclose(ah_k1.sum(dim=-1), torch.ones(3, 8), atol=1e-5)
    y_k1.pow(2).mean().backward()
    assert layer_k1.W_router.grad is not None and layer_k1.W_router.grad.norm().item() > 0.0


def test_v5_inference_single_router_matmul_and_multilayer_skip_rate_bounded():
    """修复 7 (多层 skip_rate > 1 与重复路由计算): 4 层全溢出时 skip_rate <= 1.0 且无重复路由投影。"""
    torch.manual_seed(13)
    layers = nn.ModuleList([HeteroFusionLayer(16, 32, 4, 1) for _ in range(4)])
    with torch.no_grad():
        for lyr in layers:
            lyr.W_router.zero_()
            lyr.W_router[0].fill_(10.0)
    runner = InferenceRunner(layers, capacity_factor=1.0)
    res = runner.benchmark(torch.ones(2, 16, 16), capacity_factors=(1.0,), warmup=1, repeat=2)
    sr = res[1.0]["skip_rate"]
    assert 0.0 < sr <= 1.0, f"多层 skip_rate 越界: {sr}"
    assert len(res[1.0]["per_layer_skip_rates"]) == 4

    # 验证传入 routed_logits 时层内不再调用 W_router 线性投影
    orig_linear = torch.nn.functional.linear
    router_calls = [0]

    def _wrapped_linear(inp, weight, bias=None):
        if weight is layers[0].W_router:
            router_calls[0] += 1
        return orig_linear(inp, weight, bias)

    torch.nn.functional.linear = _wrapped_linear
    try:
        runner.forward(torch.ones(1, 4, 16))
    finally:
        torch.nn.functional.linear = orig_linear
    assert router_calls[0] == 1, f"每层路由矩阵乘应仅发生 1 次，实际 {router_calls[0]} 次"


def test_v5_downstream_eval_gate_catches_leakage_and_underdetermination():
    """修复 8 (评测门禁): 拦截三集重叠泄漏、高维欠定拟合与弱于单底座的假融合。"""
    # 1. 重叠划分必须报错
    try:
        SplitGuard.verify([0, 1, 2], [2, 3], [4, 5])
        raise AssertionError("未拦截 train∩val 重叠")
    except ValueError:
        pass

    # 2. v83 O(256) on N=64 欠定必须被拦截，r_safe 必须通过
    bad_id = IdentifiabilityGuard.check(64, IdentifiabilityGuard.orthogonal_group_dof(256))
    assert bad_id["ok"] is False
    r_safe = IdentifiabilityGuard.max_safe_procrustes_rank(64, 256)
    good_id = IdentifiabilityGuard.check(64, IdentifiabilityGuard.orthogonal_group_dof(r_safe))
    assert good_id["ok"] is True

    # 3. 弱于单底座的模型必须被 DownstreamEvalGate 判 FAIL
    y = torch.tensor([0, 1, 0, 1] * 25)
    strong_single = torch.zeros(100, 2)
    strong_single[torch.arange(100), y] = 3.0
    weak_fusion = torch.zeros(100, 2)
    rep = DownstreamEvalGate(n_boot=200, seed=0).evaluate_classification(
        y_true_test=y, fusion_logits_test=weak_fusion, single_logits_test={"s0": strong_single}
    )
    assert rep.passed is False and len(rep.reasons) >= 1


def test_v62_fusion_true_sparse_dispatch():
    """修复 9 (v62 融合层伪稀疏): FusionOverProjSpace 仅计算活跃 token。"""
    from research.aligner.v62_dimension_projection_space.fusion import FusionOverProjSpace

    torch.manual_seed(0)
    fus = FusionOverProjSpace(d_space=64, d_ff=128, num_experts=4, top_k=1)
    y, ah = fus.forward_with_routing(torch.randn(3, 10, 64))
    assert y.shape == (3, 10, 64)
    assert sum(fus.last_expert_token_counts) == 3 * 10 * 1


def test_v62_measure_random_weight_control_and_orthogonal_fix():
    """修复 10 (v62 measure_v2.py 随机对照与正交对齐，同时保留原 measure.py SHA256 锁): 同输入随机矩阵阴性对照揭示秩塌缩假象，正交对齐不虚高。"""
    import hashlib
    from research.aligner.v62_dimension_projection_space.measure_v2 import measure_random_weight_control

    raw_m = (ROOT / "research/aligner/v62_dimension_projection_space/measure.py").read_bytes()
    orig_sha = hashlib.sha256(raw_m.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")).hexdigest()
    assert orig_sha == "8a20d007f5a42340ee89713512c089f81157d92101610c17f92f7ae6077aa349"

    rc = measure_random_weight_control(ds=32, n_samples=256, iters=550, seed=0)
    assert rc["unconstrained_2_random_W"]["cka"] > 0.90
    assert rc["unconstrained_2_random_W"]["eff_rank_A"] < 10.0
    # 线性 CKA 对正交变换不变，同输入两随机矩阵原生 CKA 约为 0.50；正交对齐保满秩且不虚高
    assert rc["unconstrained_2_random_W"]["cka"] - rc["orthogonal_2_random_W"]["cka"] > 0.30
    assert abs(rc["orthogonal_2_random_W"]["eff_rank_Q"] - 32.0) < 1e-2


def test_v62_projspace_real_slot_flag_and_strict_mode():
    """修复 11 (v62 伪造 Q/K/V/O 槽): build_slot_feats 显式标注槽来源，strict_real_slots 可强制阻断启发式冒充。"""
    from research.aligner.v62_dimension_projection_space.projspace import DimProjectionSpace

    op = DimProjectionSpace(d_shared=32)
    projs = {"Qwen": torch.randn(2, 6, 32), "BERT": torch.randn(2, 6, 32)}
    op.build_slot_feats(projs)
    assert op.last_slot_source == "heuristic_fallback"
    try:
        op.build_slot_feats(projs, strict_real_slots=True)
        raise AssertionError("strict_real_slots=True 未拦截缺失真实槽激活")
    except ValueError:
        pass
    real = {m: {k: torch.randn(2, 6, 32) for k in op.slot_names} for m in projs}
    op.build_slot_feats(projs, slot_acts=real, strict_real_slots=True)
    assert op.last_slot_source == "real_slots"


def test_v62_train_v2_safe_import_and_three_way_split():
    """修复 12 (v62 train_v2 import 副作用 + 验证集选模泄漏): 安全 import + 严格分层三集切分。"""
    from research.aligner.v62_dimension_projection_space import train_v2

    labels = np.array([0, 1, 2, 3] * 50)
    tr, va, te = train_v2.stratified_three_way_split(labels, n_cls=4, seed=0)
    SplitGuard.verify(tr, va, te)
    assert len(tr) + len(va) + len(te) == len(labels)


def test_v65_snli_ve_no_caption_leakage_by_default():
    """修复 13 (v65 SNLI-VE caption 纯文本泄漏 + import 副作用): 默认仅喂假设 sentence2。"""
    from research.aligner.v62_dimension_projection_space import extract_snli_ve_v65 as ext

    entry = {"sentence1": "A dog runs on grass.", "sentence2": "An animal is outdoors."}
    assert ext.INCLUDE_CAPTION is False
    assert ext.build_snli_ve_text(entry) == "An animal is outdoors."
    assert ext.build_snli_ve_text(entry, include_caption=True) == "A dog runs on grass. An animal is outdoors."


def test_governance_checkers_detect_dynamic_loads_and_unmanaged_dirs():
    """修复 14 (工程治理盲区): check_self_containment 拦截动态跨线加载，validate_generations 曝光非治理子域。"""
    from tools import check_self_containment as csc
    from tools import validate_generations as vg

    pkgs = csc.discover_research_packages()
    known = csc._build_pkg_line_index(pkgs)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "sneaky.py"
        p.write_text(
            "import importlib.util\n"
            "spec = importlib.util.spec_from_file_location('x', '../aligner/v62_dimension_projection_space/measure.py')\n",
            encoding="utf-8",
        )
        dyn = csc.parse_dynamic_loads(p, "v99_new_pkg", known)
        assert "v62_dimension_projection_space" in dyn

    scope = vg.audit_unmanaged_scope(ROOT, vg.discover_generations())
    assert any("brain_domains" in s for s in scope["unmanaged_dirs"])
    assert any("v62_dimension_projection_space" in s for s in scope["bloated_packages"])

    # 在临时目录内构造内嵌 .venv，避免依赖未入版本库的本地 research/vqc/.venv 导致干净克隆失败
    with tempfile.TemporaryDirectory() as td_venv:
        venv_cfg = Path(td_venv) / "research" / "mock_domain" / ".venv" / "pyvenv.cfg"
        venv_cfg.parent.mkdir(parents=True, exist_ok=True)
        venv_cfg.write_text("home = /usr/bin\n", encoding="utf-8")
        mock_scope = vg.audit_unmanaged_scope(Path(td_venv), [])
        assert any(".venv" in s for s in mock_scope["embedded_venvs"])


def test_v5_1_internal_lora_weight_quant_train_capacity_tp_and_momentum_preservation():
    """修复 15 (v5.1 五项核心工程闭环):
    (a) 内部子层 LoRA (internal_lora=True) 深入 SwiGLU/GELU/custom 非线性子层且 base 参数严格保持冻结；
    (b) 专家权重级 INT4 伪量化 (quant_target='weight'|'both') 生效且可微；
    (c) 训练期容量感知路由 (train_capacity_factor) 强制执行容量上界并回传 STE 梯度；
    (d) 张量并行 TensorParallelSwiGLU (列并行+行并行+All-Reduce) 与未切分 swiglu_forward 数值等价；
    (e) HeteroTrainer 从 Phase 1 切换 Phase 2 时无损保留已有 Adam 一阶/二阶动量状态 (exp_avg / exp_avg_sq)。
    """
    from hetero_moe import TensorParallelSwiGLU, swiglu_forward

    torch.manual_seed(2026)
    D, D_FF = 16, 32

    # (a) 内部子层 LoRA
    class CustomMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(D, 24)
            self.l2 = nn.Linear(24, D)

        def forward(self, inp: torch.Tensor) -> torch.Tensor:
            return self.l2(torch.relu(self.l1(inp)))

    specs = [
        ExpertSpec("sw", "swiglu", d_in=D, d_ff=D_FF, d_out=D, freeze_base=True, lora_rank=4, internal_lora=True),
        ExpertSpec("ge", "gelu_mlp", d_in=D, d_ff=D_FF, d_out=D, freeze_base=True, lora_rank=4, internal_lora=True),
        ExpertSpec("cu", "custom", d_in=D, d_out=D, freeze_base=True, lora_rank=4, module=CustomMLP(), internal_lora=True),
    ]
    lyr = HeteroFusionLayer(D, D_FF, 3, top_k=2, expert_specs=specs, quant_target="both", train_capacity_factor=1.25)
    for exp in lyr.experts:
        assert exp.sublayer_loras is not None and len(exp.sublayer_loras) >= 2
        assert all(not p.requires_grad for p in exp.base.parameters())
        # 使子层 LoRA B 非零以验证第一层子层 LoRA A 同样获得非零梯度
        for adp in exp.sublayer_loras.values():
            nn.init.normal_(adp.B, std=0.05)

    x = torch.randn(4, 8, D, requires_grad=True)
    y, _, ah = lyr(x)
    # (c) 验证 train_capacity_factor=1.25 生效: 每个专家接纳 token 数 <= int(1.25 * 32 / 3) = 13
    for cnt in lyr.last_expert_token_counts:
        assert cnt <= int(1.25 * 32 / 3)
    y.pow(2).mean().backward()
    for exp in lyr.experts:
        for adp in exp.sublayer_loras.values():
            assert adp.A.grad is not None and adp.A.grad.norm() > 0
            assert adp.B.grad is not None and adp.B.grad.norm() > 0

    # (b) 验证权重级 INT4 伪量化与无量化输出存在预期的权重离散化差异
    lyr_noq = HeteroFusionLayer(D, D_FF, 3, top_k=2, quant_bits=4, quant_target="none")
    lyr_wq = HeteroFusionLayer(D, D_FF, 3, top_k=2, quant_bits=4, quant_target="weight")
    lyr_wq.load_state_dict(lyr_noq.state_dict())
    y_noq, _, _ = lyr_noq(x.detach())
    y_wq, _, _ = lyr_wq(x.detach())
    assert not torch.allclose(y_noq, y_wq, atol=1e-6)

    # (d) 张量并行 Column-Parallel + Row-Parallel + All-Reduce 严格等价于单卡 swiglu_forward
    wg = torch.randn(D_FF, D)
    wu = torch.randn(D_FF, D)
    wd = torch.randn(D, D_FF)
    tp_mod = TensorParallelSwiGLU(wg, wu, wd, num_shards=4)
    out_ref = swiglu_forward(x.detach(), wg, wu, wd)
    out_tp = tp_mod(x.detach())
    assert torch.allclose(out_ref, out_tp, atol=1e-5)

    # (e) HeteroTrainer Phase 1 -> Phase 2 保留 Adam exp_avg / exp_avg_sq 状态
    t_layers = nn.ModuleList([HeteroFusionLayer(D, D_FF, 3, top_k=1, lora_rank=2)])
    head = nn.Linear(D, 8, bias=False)
    trainer = HeteroTrainer(t_layers, lm_head=head)
    trainer.begin_phase(1)
    trainer.step(torch.randn(2, 6, D), torch.randint(0, 8, (2, 6)))
    g0 = t_layers[0].gammas[0]
    assert g0 in trainer.optimizer.state and "exp_avg" in trainer.optimizer.state[g0]
    saved_exp_avg = trainer.optimizer.state[g0]["exp_avg"].clone()
    trainer.begin_phase(2)
    assert g0 in trainer.optimizer.state
    assert torch.allclose(trainer.optimizer.state[g0]["exp_avg"], saved_exp_avg)
    # 且 W_router 已被增量加入 param_groups 并可在 Phase 2 正常更新
    trainer.step(torch.randn(2, 6, D), torch.randint(0, 8, (2, 6)))
    assert t_layers[0].W_router in trainer.optimizer.state

