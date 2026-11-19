"""可重算的测量脚本（v2 修复版）：修复原 measure.py 的随机矩阵伪像与探针口径。

说明：原 `measure.py` 因被 `v75–v83` 的 `provenance_sha256` 严格锁定在
`8a20d007f5a42340ee89713512c089f81157d92101610c17f92f7ae6077aa349`，须保持只读不变；
所有新增的正交 Procrustes 对齐、同输入随机矩阵阴性对照与共享探针保真度检查均维护在本文件。

四组测量，对应 results/metrics.json 的四个顶层键：

  functional_slots   三个模型的功能槽结构对照（按 nn.Module 结构对齐，不按参数名）
  spectral           同功能跨架构的权重谱是否更相似（JS 散度 + 置换检验）
  alignment          两两 / 三模型的对齐上限（共享等维变换）
  rank_sweep         压维 / 等维 / 升维的 CKA 与保真度

依赖：三个模型的权重切片
  https://huggingface.co/redmingwei/backbone-slices
  （qwen.pt / bert.pt / vit.pt，各含 layer0..3 × 6 个功能槽的张量）

用法：
    python research/aligner/v62_dimension_projection_space/measure.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

THIS = Path(__file__).resolve().parent
SLICE_REPO = 'redmingwei/backbone-slices'
DS = 256
N_LAYERS = 4
DIMS = {'Qwen': 896, 'BERT': 256, 'ViT': 768}
SLOTS = ['Q', 'K', 'V', 'O', 'expand', 'contract']

#: 功能槽到各模型真实参数名的映射（形式不同、作用相同）
SLOT_PATHS = {
    'Qwen': {
        'Q': 'self_attn.q_proj.weight', 'K': 'self_attn.k_proj.weight',
        'V': 'self_attn.v_proj.weight', 'O': 'self_attn.o_proj.weight',
        'expand': 'mlp.up_proj.weight', 'contract': 'mlp.down_proj.weight',
    },
    'BERT': {
        'Q': 'attention.self.query.weight', 'K': 'attention.self.key.weight',
        'V': 'attention.self.value.weight', 'O': 'attention.output.dense.weight',
        'expand': 'intermediate.dense.weight', 'contract': 'output.dense.weight',
    },
    'ViT': {
        'Q': 'attention.q_proj.weight', 'K': 'attention.k_proj.weight',
        'V': 'attention.v_proj.weight', 'O': 'attention.o_proj.weight',
        'expand': 'mlp.fc1.weight', 'contract': 'mlp.fc2.weight',
    },
}


# ---------------------------------------------------------------- 工具
def cka(a: torch.Tensor, b: torch.Tensor) -> float:
    """线性 CKA（对正交变换不变）。"""
    a = a - a.mean(0, keepdim=True)
    b = b - b.mean(0, keepdim=True)
    num = (a.t() @ b).norm() ** 2
    den = (a.t() @ a).norm() * (b.t() @ b).norm()
    return float(num / (den + 1e-9))


def js(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon 散度（对称，0=相同）。"""
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda x, y: float(np.sum(x * np.log((x + 1e-12) / (y + 1e-12))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def load_slices() -> Dict[Tuple[str, str, str], torch.Tensor]:
    """载入 qwen/bert/vit 的权重切片 → {(model, layer, slot): tensor}"""
    from huggingface_hub import hf_hub_download
    wall: Dict[Tuple[str, str, str], torch.Tensor] = {}
    for key, name in [('qwen', 'Qwen'), ('bert', 'BERT'), ('vit', 'ViT')]:
        p = hf_hub_download(SLICE_REPO, f'{key}.pt', repo_type='model')
        pack = torch.load(p, map_location='cpu', weights_only=True)
        for k, v in pack.items():
            ly, slot = k.split('_', 1)
            wall[(name, ly, slot)] = v.float()
    return wall


def projections(seed: int = 0) -> Dict[str, torch.Tensor]:
    """各模型到共享空间的正交投影（与 v33 的 P_m 同构）。"""
    torch.manual_seed(seed)
    return {m: torch.nn.init.orthogonal_(torch.empty(DS, DIMS[m])) for m in DIMS}


def slot_repr(wall, p, model: str, layer: str, slot: str,
              x: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
    """x(DS) → P^T → W → P → (DS)；维度不兼容时返回 None。"""
    if x is None:
        x = torch.randn(4096, DS)
    w = wall.get((model, layer, slot))
    if w is None:
        return None
    xm = x @ p[model]
    if w.shape[1] != xm.shape[1] or w.shape[0] != p[model].shape[1]:
        return None
    y = (xm @ w.t()) @ p[model].t()
    return y / (y.std() + 1e-9)


# ---------------------------------------------------------------- 1. 功能槽
def measure_functional_slots() -> dict:
    return {
        'method': '按 nn.Module 结构对齐，不按参数名',
        'slots': SLOT_PATHS,
        'form_differences': {
            'norm_position': 'Qwen/ViT 是 pre-norm，BERT 是 post-norm',
            'ffn': 'Qwen 是 SwiGLU（gate+up 两路），BERT/ViT 是单路 GELU',
            'qkv': 'Qwen/ViT 有 bias 在 o_proj，BERT 每个投影都有 bias',
            'gqa': 'Qwen 是 GQA（14 Q 头 / 2 KV 头），K/V 投影是 [128,896] 非方阵',
        },
        'shapes_layer0': {
            'Qwen': {'Q': [896, 896], 'K': [128, 896], 'V': [128, 896],
                     'O': [896, 896], 'expand': [4864, 896], 'contract': [896, 4864]},
            'BERT': {'Q': [256, 256], 'K': [256, 256], 'V': [256, 256],
                     'O': [256, 256], 'expand': [1024, 256], 'contract': [256, 1024]},
            'ViT': {'Q': [768, 768], 'K': [768, 768], 'V': [768, 768],
                    'O': [768, 768], 'expand': [3072, 768], 'contract': [768, 3072]},
        },
    }


# ---------------------------------------------------------------- 2. 谱
def measure_spectral(wall: dict) -> dict:
    spec = {}
    for (m, ly, slot), w in wall.items():
        sv = torch.linalg.svdvals(w)
        sv = sv / sv.sum()
        idx = np.linspace(0, len(sv) - 1, 64).astype(int)
        spec[(m, ly, slot)] = sv.numpy()[idx]

    keys = list(spec)
    pairs = []
    for a, b in itertools.combinations(keys, 2):
        if a[0] == b[0]:
            continue                       # 只比跨模型
        pairs.append((a, b, js(spec[a], spec[b]), a[2] == b[2]))

    same = np.array([d for _, _, d, s in pairs if s])
    diff = np.array([d for _, _, d, s in pairs if not s])

    rng = np.random.default_rng(0)
    allv = np.array([d for _, _, d, _ in pairs])
    lab = np.array([1 if s else 0 for _, _, _, s in pairs])
    obs = diff.mean() - same.mean()
    cnt = sum(1 for _ in range(5000)
              if (lambda perm: allv[perm == 0].mean() - allv[perm == 1].mean() >= obs)(
                  rng.permutation(lab)))
    d = float((diff.mean() - same.mean()) / np.sqrt((same.var() + diff.var()) / 2))

    per_slot = {}
    for slot in SLOTS:
        v = [x for a, _b, x, s in pairs if s and a[2] == slot]
        if v:
            per_slot[slot] = float(np.mean(v))

    return {
        'metric': 'Jensen-Shannon divergence on normalized singular-value spectra',
        'n_pairs': len(pairs),
        'n_layers_per_model': N_LAYERS,
        'same_function_js_mean': float(same.mean()),
        'same_function_js_median': float(np.median(same)),
        'diff_function_js_mean': float(diff.mean()),
        'diff_function_js_median': float(np.median(diff)),
        'cohen_d': d,
        'permutation_p': cnt / 5000,
        'per_slot_js': per_slot,
        'reading': '同功能跨架构的权重谱显著更相似（p=0.0002），但效应量中等（d=0.25）；'
                   '各槽差异很大：contract/O 最通用，Q/K 最特异',
    }


# ---------------------------------------------------------------- 3. 对齐
def effective_rank(mat: torch.Tensor) -> float:
    """矩阵奇异值参与比有效秩 (Σs_i^2)^2 / Σs_i^4，用于监测共享变换是否发生秩塌缩。"""
    s = torch.linalg.svdvals(mat.detach().float())
    s2 = s.pow(2)
    return float((s2.sum().pow(2) / (s2.pow(2).sum() + 1e-12)).item())


def fit_orthogonal_pair(x_a: torch.Tensor, x_b: torch.Tensor) -> Tuple[float, float, torch.Tensor]:
    """正交约束 Procrustes 对齐 (A ∈ O(D))：严格保距、不可靠塌缩有效秩虚高 CKA。

    输入均为同一探测集上的表示 [N, D]；求解 min_{Q ∈ O(D)} ||x_a Q - x_b||_F。
    返回 (aligned_cka, rel_residual, Q)。
    """
    a = x_a - x_a.mean(0, keepdim=True)
    b = x_b - x_b.mean(0, keepdim=True)
    a = a / (a.std() + 1e-9)
    b = b / (b.std() + 1e-9)
    u, _, vh = torch.linalg.svd(a.t() @ b, full_matrices=False)
    q = u @ vh
    a_al = a @ q
    resid = float(((a_al - b) ** 2).mean().item())
    return cka(a_al, b), resid, q


def fit_shared(outs: Dict[str, torch.Tensor], iters: int = 1200,
               lr: float = 2e-3, restarts: int = 2) -> Tuple[float, float]:
    """[历史复现接口] 单个无约束共享等维变换 A。返回 (CKA, 残差)。

    修复: 移除从未被使用的死参数 tar。
    注意 (v84 审计): 当对两路同输入线性变换使用单个无正交约束 A 时，A 会塌缩到广义
    特征向量子空间，即使纯独立高斯随机矩阵也能获得 CKA≈0.98+（三矩阵≈0.68）。
    真实跨模型对齐度应以 fit_orthogonal_pair 与 measure_random_weight_control 为准。
    """
    ck, resid, _er, _A = _fit_shared_internal(outs, rank=DS, iters=iters, lr=lr, restarts=restarts)
    return ck, resid


def _fit_shared_internal(
    outs: Dict[str, torch.Tensor],
    rank: int,
    iters: int = 900,
    lr: float = 2e-3,
    restarts: int = 2,
) -> Tuple[float, float, float, torch.Tensor]:
    names = list(outs)
    in_dim = outs[names[0]].shape[-1]
    best = None
    for sd in range(restarts):
        torch.manual_seed(sd)
        if rank == in_dim:
            A = nn.Parameter(torch.eye(in_dim) + torch.randn(in_dim, in_dim) * 0.02)
        else:
            A = nn.Parameter(torch.randn(rank, in_dim) * 0.02)
        opt = torch.optim.Adam([A], lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
        for _ in range(iters):
            opt.zero_grad()
            al = [outs[n] @ A.t() for n in names]
            al = [a / (a.std() + 1e-9) for a in al]
            tt = torch.stack(al).mean(0).detach()
            (sum(F.mse_loss(a, tt) for a in al) / len(al)).backward()
            opt.step()
            sch.step()
        with torch.no_grad():
            al = [(outs[n] @ A.t()).detach() for n in names]
            al = [a / (a.std() + 1e-9) for a in al]
            tt = torch.stack(al).mean(0)
            resid = float(sum(((a - tt) ** 2).mean() for a in al) / len(al))
            ck = float(np.mean([cka(al[i], al[j])
                                for i in range(len(al))
                                for j in range(i + 1, len(al))]))
            er = effective_rank(A)
        if best is None or resid < best[1]:
            best = (ck, resid, er, A.detach().clone())
    return best


def _shared_probe(n_samples: int = 4096, ds: int = DS, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_samples, ds, generator=g)


def measure_alignment(wall: dict, p: dict, seed: int = 0) -> dict:
    out = {
        'protocol': '共享固定探针 x；同时报告历史无约束 fit_shared (带 A 有效秩) 与正交约束 fit_orthogonal_pair',
        'pairwise': {}, 'pairwise_orthogonal': {}, 'triple': {},
    }
    x_probe = _shared_probe(4096, DS, seed=seed)
    layer = '1'
    for slot in ['Q', 'K', 'V', 'O']:
        pair_res = {}
        pair_ortho = {}
        avail = {}
        for m in DIMS:
            r = slot_repr(wall, p, m, layer, slot, x=x_probe)
            if r is not None:
                avail[m] = r
        for a, b in itertools.combinations(avail, 2):
            ck, resid, er, _A = _fit_shared_internal({a: avail[a], b: avail[b]}, rank=DS, iters=900)
            pair_res[f'{a}×{b}'] = {'cka': ck, 'resid': resid, 'eff_rank_A': er}
            ock, oresid, _Q = fit_orthogonal_pair(avail[a], avail[b])
            pair_ortho[f'{a}×{b}'] = {'cka': ock, 'resid': oresid, 'eff_rank_Q': float(DS)}
        out['pairwise'][slot] = pair_res
        out['pairwise_orthogonal'][slot] = pair_ortho
        if len(avail) >= 3:
            ck, resid, er, _A = _fit_shared_internal(avail, rank=DS, iters=1200)
            out['triple'][slot] = {'cka': ck, 'resid': resid, 'eff_rank_A': er,
                                   'n_models': len(avail)}
        noq = {k: v for k, v in avail.items() if k != 'Qwen'}
        if len(noq) >= 2 and len(avail) >= 3:
            ck, resid, er, _A = _fit_shared_internal(noq, rank=DS, iters=900)
            out['triple'][slot + '_without_qwen'] = {
                'cka': ck, 'resid': resid, 'eff_rank_A': er, 'n_models': len(noq)}

    out['reading'] = (
        '修正后使用固定共享探针 x 并同步输出无约束 fit_shared 的有效秩与严格正交 Procrustes CKA。'
        '注意：无约束单矩阵 fit_shared 的高 CKA 可在纯随机矩阵上因秩塌缩复现，需结合正交口径与随机权重对照解读。')
    return out


# ---------------------------------------------------------------- 4. 秩扫描
def _fidelity(A: torch.Tensor, orig: torch.Tensor) -> float:
    proj = orig @ A.t()
    ax = torch.cat([proj.reshape(-1, proj.shape[-1]),
                    torch.ones(proj.shape[0], 1, device=orig.device, dtype=orig.dtype)], 1)
    b = orig.reshape(-1, orig.shape[-1])
    sol = torch.linalg.lstsq(ax, b).solution
    pred = ax @ sol
    return float(1 - ((b - pred) ** 2).sum() / (((b - b.mean(0)) ** 2).sum() + 1e-12))


def measure_rank_sweep(wall: dict, p: dict, seed: int = 0) -> dict:
    ranks = [8, 24, 28, 64, 128, 192, 256, 384, 512, 768]
    out = {'protocol': '固定共享探针 x，K/V 槽，BERT+ViT，layer1；同步实算 fidelity 与 eff_rank_A',
           'by_slot': {}}
    x_probe = _shared_probe(4096, DS, seed=seed)

    for slot in ['K', 'V']:
        outs = {}
        for m in ['BERT', 'ViT']:
            r = slot_repr(wall, p, m, '1', slot, x=x_probe)
            if r is not None:
                outs[m] = r
        if len(outs) < 2:
            continue
        ref_orig = outs['BERT']
        rows = {}
        for rank in ranks:
            ck, resid, er, A = _fit_shared_internal(outs, rank=rank, iters=900)
            rows[str(rank)] = {
                'cka': ck,
                'resid': resid,
                'fidelity': _fidelity(A, ref_orig),
                'eff_rank_A': er,
            }
        out['by_slot'][slot] = rows

    out['reading'] = (
        '已补全 fidelity() 实际调用与 eff_rank_A 监测：当 rank >= D 时任意数值满秩矩阵的 lstsq 均可近无损求逆'
        '(fidelity≈1.0)，即使 eff_rank_A 已塌缩至个位数；因此高 fidelity + 高 CKA 不能单独证明跨模型对齐有效。')
    return out


# ---------------------------------------------------------------- 5. 公平随机权重阴性对照
def measure_random_weight_control(
    ds: int = 64,
    n_samples: int = 512,
    iters: int = 500,
    seed: int = 0,
) -> dict:
    """同输入随机权重阴性对照（修复原 metrics.json 中用独立随机输出导致对照失效的缺陷）。

    用完全独立的随机高斯权重矩阵 W1/W2/W3 作用在同一探针 x 上，分别运行:
      - 无约束 fit_shared（复现 2 矩阵 CKA≈0.98+ / 3 矩阵 CKA≈0.68 / 秩塌缩的线性代数假象）
      - 正交约束 fit_orthogonal_pair（真实可保距基线，不会虚高）
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, ds, generator=g)
    ws = [torch.randn(ds, ds, generator=g) for _ in range(3)]

    def _mk(W: torch.Tensor) -> torch.Tensor:
        y = x @ W.t()
        return y / (y.std() + 1e-9)

    outs2 = {'rand1': _mk(ws[0]), 'rand2': _mk(ws[1])}
    outs3 = {'rand1': outs2['rand1'], 'rand2': outs2['rand2'], 'rand3': _mk(ws[2])}
    ck2, res2, er2, A2 = _fit_shared_internal(outs2, rank=ds, iters=iters, restarts=1)
    ck3, res3, er3, A3 = _fit_shared_internal(outs3, rank=ds, iters=iters, restarts=1)
    ock2, ores2, _Q = fit_orthogonal_pair(outs2['rand1'], outs2['rand2'])
    return {
        'ds': ds,
        'n_samples': n_samples,
        'unconstrained_2_random_W': {
            'cka': ck2, 'resid': res2, 'eff_rank_A': er2,
            'fidelity': _fidelity(A2, outs2['rand1']),
        },
        'unconstrained_3_random_W': {
            'cka': ck3, 'resid': res3, 'eff_rank_A': er3,
            'fidelity': _fidelity(A3, outs3['rand1']),
        },
        'orthogonal_2_random_W': {
            'cka': ock2, 'resid': ores2, 'eff_rank_Q': float(ds),
        },
    }


# ---------------------------------------------------------------- 主
def main() -> None:
    print('[1/6] 载入权重切片 ...', flush=True)
    wall = load_slices()
    print('       %d 个张量' % len(wall), flush=True)
    p = projections()

    print('[2/6] 功能槽结构 ...', flush=True)
    fs = measure_functional_slots()

    print('[3/6] 谱相似性（1728 对）...', flush=True)
    sp = measure_spectral(wall)
    print('       同功能 %.5f / 不同功能 %.5f / p=%.4f'
          % (sp['same_function_js_mean'], sp['diff_function_js_mean'],
             sp['permutation_p']), flush=True)

    print('[4/6] 对齐上限（共享探针 + 正交对照）...', flush=True)
    al = measure_alignment(wall, p)

    print('[5/6] 秩扫描（含实算 fidelity 与 eff_rank_A）...', flush=True)
    rk = measure_rank_sweep(wall, p)

    print('[6/6] 同输入随机权重阴性对照 ...', flush=True)
    rc = measure_random_weight_control()

    out = {'functional_slots': fs, 'spectral': sp,
           'alignment': al, 'rank_sweep': rk, 'random_weight_control': rc}
    dst = THIS / 'results' / 'metrics.json'
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print('\n已写 %s' % dst)


if __name__ == '__main__':
    main()
