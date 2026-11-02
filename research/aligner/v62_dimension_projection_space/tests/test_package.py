"""V62.2 — 维度投影空间 单元测试 (16 项)。

只依赖 torch 即可跑；涉及 transformers 的那一项在缺依赖时自动 SKIP。
每个 test_* 必须在末尾 return True（run_suite 以返回真值判定通过）。
"""
import json
import math
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "hetero_moe")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch
import torch.nn.functional as F

from research._primitives.testing import run_suite
from research.aligner.v62_dimension_projection_space.projspace import (
    DimProjectionSpace, MEASURED_SUBSPACES,
)
from research.aligner.v62_dimension_projection_space.fusion import FusionOverProjSpace

TITLE = "V62.2 维度投影空间 — 单元测试 (16 项)"

_DS = 256
_B, _S = 2, 12


def _feats(seed=0):
    torch.manual_seed(seed)
    return {m: {k: torch.randn(_B, _S, _DS) for k in MEASURED_SUBSPACES}
            for m in ['Qwen', 'BERT', 'ViT']}


def _train_align(op, steps=800):
    """构造同源输入的三个视角，训练成对对齐 + 环路一致性。"""
    torch.manual_seed(0)
    n = 512
    base = torch.randn(n, _DS)
    # detach：表示只当固定输入，反传不回穿到 op.proj
    src = {m: op.proj[f'Q|{m}'](base + torch.randn(n, _DS) * 0.1).detach()
           for m in ['Qwen', 'BERT', 'ViT']}
    pairs = {('Q', ('Qwen', 'BERT')): (src['Qwen'], src['BERT']),
             ('Q', ('BERT', 'ViT')): (src['BERT'], src['ViT']),
             ('Q', ('Qwen', 'ViT')): (src['Qwen'], src['ViT'])}
    opt = torch.optim.Adam(list(op.align.parameters()), lr=1e-3)
    for _ in range(steps):
        opt.zero_grad()
        (op.alignment_loss(pairs) + 10.0 * op.consistency_loss()).backward()
        opt.step()
    return src


# ---------------------------------------------------------------- 1
def test_specs_match_metrics():
    """1. 算子里的规格与 results/metrics.json 的实测值逐位一致"""
    mp = os.path.join(os.path.dirname(_here), 'results', 'metrics.json')
    al = json.load(open(mp))['alignment']
    assert abs(MEASURED_SUBSPACES['Q'].cka_pair - al['pairwise']['Q']['BERT×ViT']) < 1e-12
    assert abs(MEASURED_SUBSPACES['Q'].cka_triple - al['triple']['Q']) < 1e-12
    assert abs(MEASURED_SUBSPACES['O'].cka_triple - al['triple']['O']) < 1e-12
    assert MEASURED_SUBSPACES['K'].cka_triple is None   # K/V 只有两模型参与
    assert MEASURED_SUBSPACES['V'].cka_triple is None
    print('[PASS] test_specs_match_metrics (规格与实测逐位一致)')
    return True


# ---------------------------------------------------------------- 2
def test_forward_shapes():
    """2. forward 输出 struct / canonical / alpha 三种表示"""
    op = DimProjectionSpace(_DS)
    struct, canon, alpha = op(_feats())
    d = max(sp.dim for sp in op.subspaces.values())
    assert struct.shape == (_B, _S, op.n_slots, d), struct.shape
    assert canon.shape == (_B, _S, op.total_dim), canon.shape
    assert alpha.shape == (_B, _S, op.n_experts), alpha.shape
    print('[PASS] test_forward_shapes (struct/canonical/alpha 形状正确)')
    return True


# ---------------------------------------------------------------- 3
def test_token_count_preserved():
    """3. token 数不变 —— 合并发生在模型轴，不是 token 轴"""
    op = DimProjectionSpace(_DS)
    struct, canon, _ = op(_feats())
    assert struct.shape[1] == _S and canon.shape[1] == _S
    print(f'[PASS] test_token_count_preserved (S={_S} 保持不变)')
    return True


# ---------------------------------------------------------------- 4
def test_separate():
    """4. 分离：取出单个槽的分量"""
    op = DimProjectionSpace(_DS)
    struct, _, _ = op(_feats())
    for slot in op.slot_names:
        part = op.separate(struct, slot)
        assert part.shape == (_B, _S, op.subspaces[slot].dim), part.shape
    print('[PASS] test_separate (每个槽都能单独取出)')
    return True


# ---------------------------------------------------------------- 5
def test_recombine():
    """5. 重组：token0 的 K + token1 的 V → 新 token"""
    op = DimProjectionSpace(_DS)
    struct, _, _ = op(_feats())
    rec = op.recombine(struct, {'K': 0, 'V': 1})
    ki, vi = op.slot_names.index('K'), op.slot_names.index('V')
    assert torch.allclose(rec[:, 5, ki, :], struct[:, 0, ki, :])
    assert torch.allclose(rec[:, 5, vi, :], struct[:, 1, vi, :])
    print('[PASS] test_recombine (跨 token 拼接正确)')
    return True


# ---------------------------------------------------------------- 6
def test_router_dense_distribution():
    """6. 槽门控是**全槽软分布**，不是 top-k 稀疏。

    v62.1 曾对槽做 top-k=2，实测每 token 最大权重均值 0.9933、熵 0.0209
    （top-2 理论上界才 0.5），等价于只剩一个槽。四个槽的投影在门控前
    已经算完，top-k 在这里不省算力、只丢信息，故改为全槽软门控。
    """
    op = DimProjectionSpace(_DS)
    _, _, alpha = op(_feats())
    # 1) 全槽都应有非零权重（软门控）
    nz = (alpha > 0).sum(-1)
    assert (nz == op.n_slots).all(), f'应全槽非零，实测 {nz.unique().tolist()}'
    # 2) 每 token 权重和为 1
    assert torch.allclose(alpha.sum(-1), torch.ones_like(alpha.sum(-1)), atol=1e-5)
    print(f'[PASS] test_router_dense_distribution (全 {op.n_slots} 槽软分布, 和为 1)')
    return True


def test_slot_entropy_loss():
    """6b. slot_entropy_loss 能正确表达「熵越低惩罚越大」。"""
    op = DimProjectionSpace(_DS)
    uniform = torch.full((2, 4, 4), 0.25)
    peaked = torch.zeros(2, 4, 4)
    peaked[..., 0] = 1.0
    l_uniform = float(op.slot_entropy_loss(uniform))
    l_peaked = float(op.slot_entropy_loss(peaked))
    assert l_uniform < l_peaked, (l_uniform, l_peaked)
    # 4 类均匀分布的熵 = ln4 = 1.3863（未归一化）→ loss = -1.3863
    assert abs(l_uniform + math.log(4)) < 1e-4, l_uniform
    print(f'[PASS] test_slot_entropy_loss (均匀 {l_uniform:.4f} < 尖峰 {l_peaked:.4f})')
    return True


# ---------------------------------------------------------------- 7
def test_route_slots_derived_from_measurement():
    """7. 路由槽由实测 cka_triple < 0.9 推出，不是硬编码"""
    op = DimProjectionSpace(_DS)
    expected = [k for k, sp in op.subspaces.items()
                if sp.cka_triple is not None and sp.cka_triple < 0.9]
    assert op.route_slots == expected == ['Q', 'O'], op.route_slots
    print('[PASS] test_route_slots_derived_from_measurement (路由槽 = Q,O)')
    return True


# ---------------------------------------------------------------- 8
def test_consistency_loss_decreases():
    """8. 环路一致性损失可优化到接近 0"""
    torch.manual_seed(0)
    op = DimProjectionSpace(_DS)
    l0 = float(op.consistency_loss().detach())
    _train_align(op)
    l1 = float(op.consistency_loss().detach())
    assert l1 < l0 * 0.01, f'{l0:.6f} → {l1:.6f}'
    print(f'[PASS] test_consistency_loss_decreases ({l0:.6f} → {l1:.6f})')
    return True


# ---------------------------------------------------------------- 9
def test_three_path_agreement():
    """9. 一致性训练后，直达路径与复合路径一致"""
    torch.manual_seed(0)
    op = DimProjectionSpace(_DS)
    src = _train_align(op)
    with torch.no_grad():
        direct = op.align['Q|ViT>BERT'](src['ViT'])
        compound = op.align['Q|Qwen>BERT'](op.align['Q|ViT>Qwen'](src['ViT']))
        diff = float((direct - compound).abs().mean())
        cos = float(F.cosine_similarity(direct, src['BERT'], dim=-1).mean())
    assert diff < 0.05, f'两路径差 {diff}'
    assert cos > 0.9, f'对齐后余弦 {cos}'
    print(f'[PASS] test_three_path_agreement (路径差 {diff:.4f}, 余弦 {cos:.4f})')
    return True


# ---------------------------------------------------------------- 10
def test_fusion_over_projspace():
    """10. 融合层吃 1024 维、专家解冻、top-k 稀疏"""
    d_space = sum(sp.dim for sp in MEASURED_SUBSPACES.values())
    assert d_space == 1024, d_space
    fus = FusionOverProjSpace(d_space)
    x = torch.randn(_B, _S, d_space)
    y, alpha = fus.forward_with_routing(x)
    assert y.shape == x.shape and alpha.shape == (_B, _S, fus.num_experts)
    assert ((alpha > 0).sum(-1) <= fus.top_k).all()
    assert all(p.requires_grad for p in fus.w_gates), '专家应解冻（v62 偏离 v33 的关键点）'
    fus.zero_grad()
    y.pow(2).mean().backward()
    g = sum(p.grad.abs().sum().item() for p in fus.w_gates if p.grad is not None)
    assert g > 0, '专家权重没收到梯度'
    print(f'[PASS] test_fusion_over_projspace (d_space=1024, 专家梯度 {g:.2f})')
    return True


# ---------------------------------------------------------------- 11
def test_vit_naming_reversible():
    """11. ViT 旧/新命名双向可逆（需 transformers，缺则 SKIP）"""
    try:
        from research.aligner.v62_dimension_projection_space.model import remap_vit_state
    except Exception as e:                                    # noqa: BLE001
        print(f'[SKIP] test_vit_naming_reversible (无 transformers: {type(e).__name__})')
        return True
    old = {'vit.encoder.layer.0.attention.attention.query.weight': torch.zeros(1),
           'vit.encoder.layer.0.attention.attention.key.weight': torch.zeros(1),
           'vit.encoder.layer.0.attention.attention.value.weight': torch.zeros(1),
           'vit.encoder.layer.0.attention.output.dense.weight': torch.zeros(1),
           'vit.encoder.layer.0.attention.output.LayerNorm.weight': torch.zeros(1),
           'vit.encoder.layer.0.intermediate.dense.weight': torch.zeros(1),
           'vit.encoder.layer.0.output.dense.weight': torch.zeros(1),
           'vit.encoder.layer.0.output.LayerNorm.weight': torch.zeros(1),
           'qwen.x': torch.zeros(1)}
    new = remap_vit_state(old, to_hf=True)
    assert 'vit.layers.0.attention.q_proj.weight' in new
    assert 'vit.layers.0.attention.o_proj.weight' in new
    assert 'vit.layers.0.layernorm_before.weight' in new
    assert 'vit.layers.0.mlp.fc1.weight' in new
    assert 'vit.layers.0.mlp.fc2.weight' in new
    assert 'vit.layers.0.layernorm_after.weight' in new
    assert 'qwen.x' in new, '非 vit.* 前缀不应被改动'
    back = remap_vit_state(new, to_hf=False)
    assert set(back) == set(old), (sorted(back), sorted(old))
    print('[PASS] test_vit_naming_reversible (ViT 命名双向可逆, 9 个 key 全部还原)')
    return True


# ---------------------------------------------------------------- 12
def test_partial_modality():
    """12. 缺模态可用 —— 净室测试发现的真 bug（只给文本时 router 宽度算错）"""
    op = DimProjectionSpace(_DS)
    torch.manual_seed(0)
    allm = {m: {k: torch.randn(_B, _S, _DS) for k in MEASURED_SUBSPACES}
            for m in ['Qwen', 'BERT', 'ViT']}
    cases = {
        '三模态': ['Qwen', 'BERT', 'ViT'],
        '仅文本': ['Qwen', 'BERT'],
        '仅 Qwen': ['Qwen'],
        '仅 BERT': ['BERT'],
        '仅 ViT': ['ViT'],
    }
    shapes = set()
    for name, ms in cases.items():
        struct, canon, alpha = op({m: allm[m] for m in ms})
        # 关键：输出宽度必须与三模态时一致，否则下游权重对不上
        shapes.add((canon.shape[-1], alpha.shape[-1]))
        assert struct.shape[2] == op.n_slots
    assert shapes == {(op.total_dim, op.n_experts)}, shapes
    print(f'[PASS] test_partial_modality ({len(cases)} 种输入组合, 输出宽度一致)')
    return True


# ---------------------------------------------------------------- 13
def test_router_actually_affects_output():
    """13. router 必须真的影响输出 —— 净室抓到的死代码 bug 回归测试。

    v62.0 的 bug：router 算出 alpha 后只 return，never 参与 canonical 计算。
    梯度不流向它，训练出的权重是随机的，而报告里的「专家负载/路由熵」
    统计的正是这个装饰品。本测试通过「改权重 → 看输出变不变」直接抓它。
    """
    torch.manual_seed(0)
    op = DimProjectionSpace(_DS)
    assert op.router is not None, 'router 应存在'

    torch.manual_seed(0)
    feats = {m: {k: torch.randn(_B, _S, _DS) for k in MEASURED_SUBSPACES}
             for m in ['Qwen', 'BERT', 'ViT']}
    with torch.no_grad():
        _, canon_a, aux_a = op(feats, return_aux=True)

    # 把 router 权重放大 50 倍 —— 若它真的参与计算，输出必然大改
    with torch.no_grad():
        op.router.weight.mul_(50.0)
        _, canon_b, aux_b = op(feats, return_aux=True)

    d_canon = float((canon_b - canon_a).abs().max())
    assert d_canon > 1e-4, (
        f'router 权重改了 50 倍，canonical 却只变 {d_canon:.3e} —— router 是死代码')
    print(f'[PASS] test_router_actually_affects_output (Δcanonical={d_canon:.4f})')
    return True


# ---------------------------------------------------------------- 14
def test_router_receives_gradient():
    """14. 梯度必须能到达 router。死代码时梯度为 0 或不存在。"""
    torch.manual_seed(0)
    op = DimProjectionSpace(_DS)
    torch.manual_seed(0)
    feats = {m: {k: torch.randn(_B, _S, _DS) for k in MEASURED_SUBSPACES}
             for m in ['Qwen', 'BERT', 'ViT']}
    op.zero_grad()
    _, canon, _ = op(feats)
    canon.pow(2).mean().backward()
    g = op.router.weight.grad
    assert g is not None, 'router 没有 gradient'
    gs = float(g.abs().sum())
    assert gs > 0, f'router 梯度为 0 —— 说明输出不依赖它'
    print(f'[PASS] test_router_receives_gradient (|grad|={gs:.4f})')
    return True


# ---------------------------------------------------------------- 15
def test_slot_gating_sums_to_one():
    """15. 有效门控（全局 × 逐 token）归一化后每 token 和为 1。"""
    torch.manual_seed(0)
    op = DimProjectionSpace(_DS)
    torch.manual_seed(0)
    feats = {m: {k: torch.randn(_B, _S, _DS) for k in MEASURED_SUBSPACES}
             for m in ['Qwen', 'BERT', 'ViT']}
    with torch.no_grad():
        _, _, aux = op(feats, return_aux=True)
    eff = aux['slot_gate'].view(1, 1, -1) * aux['alpha']
    eff = eff / (eff.sum(-1, keepdim=True) + 1e-9)
    s = eff.sum(-1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5), s
    # 槽门控是全槽软分布（v62.2），不再要求 top-k 稀疏
    assert (aux['alpha'] > 0).sum(-1).min() == op.n_slots
    print('[PASS] test_slot_gating_sums_to_one (门控和为 1, 全槽软分布)')
    return True


TESTS = [
    test_specs_match_metrics,
    test_forward_shapes,
    test_token_count_preserved,
    test_separate,
    test_recombine,
    test_router_dense_distribution,
    test_slot_entropy_loss,
    test_route_slots_derived_from_measurement,
    test_consistency_loss_decreases,
    test_three_path_agreement,
    test_fusion_over_projspace,
    test_vit_naming_reversible,
    test_partial_modality,
    test_router_actually_affects_output,
    test_router_receives_gradient,
    test_slot_gating_sums_to_one,
]


if __name__ == '__main__':
    sys.exit(run_suite(TITLE, TESTS))
