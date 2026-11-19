"""v62+ 训练脚本（v84 固有缺陷修复版）。

修复四项固有缺陷:
    1. import 副作用保护: 顶层仅定义配置常量与模型类，下载缓存、训练、评估与上传全部封入
       main() 并由 if __name__ == '__main__' 保护，不再 import 即触发训练或 pip install。
    2. 消除验证集选模泄漏: 原脚本 80/20 切分后在 va_idx 上挑选 best_state 再把同一 va_idx
       报为 held-out acc；现改为严格分层三集划分 train/val/test (70%/15%/15%)，val 选模、
       未触碰 test 集报告最终准确率。
    3. 强制单底座线性探针基线对照: 在完全相同的 train/val/test 划分上逐一评估各单底座
       (Qwen/BERT/ViT) 线性探针准确率，报告 fusion_minus_best_single 净增量。
    4. 禁止静默错配伪多模态: 当图文样本数不一致时默认拒绝继续（除非显式设置
       ALLOW_MISMATCHED_PAIRING=1 复现历史 AG_News+Food101 错配实验）。
"""
import importlib.util as _ilu
import json
import os
import subprocess
import sys
import time
import urllib.request as _ur
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path as _Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============ 顶层配置常量（供测试契约读取，零副作用）============
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
HFT = os.environ.get('HF_TOKEN', '')
D_SHARED = 256
LR = float(os.environ.get('LR', '2e-4'))
STEPS = int(os.environ.get('STEPS', '1500'))
BS = int(os.environ.get('BS', '32'))
BAL_W = float(os.environ.get('BAL_W', '0.01'))
SLOT_ENT_W = float(os.environ.get('SLOT_ENT_W', '0.05'))
VER = os.environ.get('VER', '62.5')

_SUF = os.environ.get('CACHE_SUFFIX', '')
_MODAL_DIMS = tuple(int(x) for x in os.environ.get('MODAL_DIMS', '896,256,768').split(','))
R = os.environ.get('CACHE', 'redmingwei/hetero-moe-cache' + _SUF.replace('_', '-'))
_FILES_CACHE = ['h_qwen%s.npy' % _SUF, 'h_bert%s.npy' % _SUF, 'h_vit%s.npy' % _SUF,
                'labels%s.npy' % _SUF, 'food_labels%s.npy' % _SUF]
_LOCAL = os.environ.get('LOCAL_CACHE', '')
N_CLS = int(os.environ.get('N_CLS', '4'))   # ag_news=4, SNLI-VE=3
ABL = os.environ.get('ABL', 'all')          # all=融合, text=只文本(无ViT), image=只视觉
ALLOW_MISMATCHED_PAIRING = os.environ.get('ALLOW_MISMATCHED_PAIRING', '0') == '1'

_USE_HYPER = os.environ.get('USE_HYPER', '0') == '1'
_USE_FAST = os.environ.get('USE_FAST', '0') == '1'
_USE_RECUR = os.environ.get('USE_RECUR', '0') == '1'
_USE_MEM = os.environ.get('USE_MEM', '0') == '1'
_USE_HEBB = os.environ.get('USE_HEBB', '0') == '1'
_USE_ADAPT = os.environ.get('USE_ADAPT', '0') == '1'
_USE_FB = os.environ.get('USE_FB', '0') == '1'
_USE_LOPT = os.environ.get('USE_LOPT', '0') == '1'
_USE_SPARSE = os.environ.get('USE_SPARSE', '0') == '1'
_USE_SUPCON = os.environ.get('USE_SUPCON', '0') == '1'
_USE_MOA = os.environ.get('USE_MOA', '0') == '1'
_MOA_STEPS = int(os.environ.get('MOA_STEPS', '1'))
_MOA_MODE = os.environ.get('MOA_MODE', 'sparse_router')
_USE_PMR = os.environ.get('PM_RECONCILE', '0') == '1'
_PM_TARGET = int(os.environ.get('PM_TARGET', '1024'))
_USE_VL = os.environ.get('USE_VL', '0') == '1'
CTR_W = float(os.environ.get('CTR_W', '0.5'))

_URL = 'https://github.com/Microbiosis/hetero-moe.git'
_RAW = 'https://raw.githubusercontent.com/Microbiosis/hetero-moe/main'
_REL = 'research/aligner/v62_dimension_projection_space'
_FILES = {'projspace.py': 'v62_projspace', 'fusion.py': 'v62_fusion',
          'hyper_proj.py': 'v62_hyper_proj', 'self_repair.py': 'v62_self_repair',
          'recurrent_gate.py': 'v62_recurrent_gate',
          'hebbian_align.py': 'v62_hebbian_align',
          'memory_router.py': 'v62_memory_router',
          'modality_adapter.py': 'v62_modality_adapter',
          'feedback_head.py': 'v62_feedback_head',
          'learned_optimizer.py': 'v62_learned_optimizer',
          'vl_fusion.py': 'v62_vl_fusion',
          'moa_fusion.py': 'v62_moa_fusion',
          'sparse_block_fusion.py': 'v62_sparse_block_fusion'}


def _feats(hq, hb, hv, abl: str = ABL):
    if abl == 'image':
        return {'ViT': hv}
    d = {'Qwen': hq, 'BERT': hb}
    if abl != 'text':
        d['ViT'] = hv
    return d


def _find_repo():
    cands = []
    try:
        cands.append(_Path(__file__).resolve())
    except NameError:
        pass
    cands += [_Path('/kaggle/working/hetero-moe'),
              _Path.cwd() / 'hetero-moe',
              _Path('/workspace/hetero-moe')]
    for c in cands:
        for q in [c, *c.parents]:
            if (q / 'research').is_dir() and (q / 'hetero_moe').is_dir():
                return q
    return None


def _load_local_ops():
    """从本地包目录加载唯一算子实现，不触发网络下载。"""
    repo = _find_repo()
    pkg_dir = (repo / _REL) if repo is not None else _Path(__file__).resolve().parent
    if str(pkg_dir) not in sys.path:
        sys.path.insert(0, str(pkg_dir))
    if repo is not None and str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    def _load(name, fname):
        path = pkg_dir / fname
        spec = _ilu.spec_from_file_location(name, path)
        mod = _ilu.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    ps = _load('v62_projspace', 'projspace.py')
    fu = _load('v62_fusion', 'fusion.py')
    return ps.DimProjectionSpace, fu.FusionOverProjSpace, ps.MEASURED_SUBSPACES


try:
    DimProjectionSpace, FusionOverProjSpace, MEASURED = _load_local_ops()
except Exception:  # noqa: BLE001
    DimProjectionSpace = None
    FusionOverProjSpace = None
    MEASURED = None


def stratified_three_way_split(
    labels: np.ndarray,
    n_cls: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按类别分层切分严格互不相交的 train / val / test 三集，彻底杜绝选模泄漏。"""
    rng = np.random.default_rng(seed)
    tr_parts, va_parts, te_parts = [], [], []
    for c in range(n_cls):
        ids = np.where(labels == c)[0].copy()
        if len(ids) == 0:
            continue
        rng.shuffle(ids)
        n_c = len(ids)
        k_tr = max(1, int(n_c * train_frac))
        k_va = max(1, int(n_c * val_frac))
        if k_tr + k_va >= n_c:
            k_tr = max(1, n_c - 2)
            k_va = max(1, n_c - k_tr - 1)
        tr_parts.append(ids[:k_tr])
        va_parts.append(ids[k_tr:k_tr + k_va])
        te_parts.append(ids[k_tr + k_va:])
    tr_idx = np.sort(np.concatenate(tr_parts))
    va_idx = np.sort(np.concatenate(va_parts))
    te_idx = np.sort(np.concatenate(te_parts))
    assert len(set(tr_idx) & set(va_idx)) == 0
    assert len(set(va_idx) & set(te_idx)) == 0
    assert len(set(tr_idx) & set(te_idx)) == 0
    return tr_idx, va_idx, te_idx


def fit_single_backbone_probe(
    feats_np: np.ndarray,
    labels_np: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    te_idx: np.ndarray,
    n_cls: int,
    dev: str = 'cpu',
    steps: int = 200,
    lr: float = 1e-2,
) -> Dict[str, float]:
    """训练单个冻结底座的线性探针基线（train 训，val 选模，test 报告）。"""
    def _pool(arr, idx):
        x = np.asarray(arr[idx], dtype=np.float32)
        if x.ndim == 3:
            x = x.mean(axis=1)
        return torch.from_numpy(x).to(dev)

    x_tr, x_va, x_te = _pool(feats_np, tr_idx), _pool(feats_np, va_idx), _pool(feats_np, te_idx)
    y_tr = torch.as_tensor(labels_np[tr_idx], dtype=torch.long, device=dev)
    y_va = torch.as_tensor(labels_np[va_idx], dtype=torch.long, device=dev)
    y_te = torch.as_tensor(labels_np[te_idx], dtype=torch.long, device=dev)

    probe = nn.Linear(x_tr.shape[-1], n_cls).to(dev)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    best_va, best_sd = -1.0, None
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(x_tr), y_tr)
        loss.backward()
        opt.step()
        with torch.no_grad():
            va_acc = float((probe(x_va).argmax(-1) == y_va).float().mean().item())
            if va_acc > best_va:
                best_va = va_acc
                best_sd = {k: v.detach().clone() for k, v in probe.state_dict().items()}
    if best_sd is not None:
        probe.load_state_dict(best_sd)
    with torch.no_grad():
        te_acc = float((probe(x_te).argmax(-1) == y_te).float().mean().item())
    return {'val_acc': best_va, 'test_acc': te_acc}


# ============ v69: 异构 P_m（维度调解）============
class DimDownProj(nn.Module):
    """大维度模态 → target_dim 的降维投影。"""
    def __init__(self, d_src, d_target):
        super().__init__()
        self.proj = nn.Linear(d_src, d_target)
        self.norm = nn.LayerNorm(d_target)
        w = torch.empty(d_target, d_target)
        nn.init.orthogonal_(w)
        self.register_buffer('ortho', w.detach())

    def forward(self, x):
        return self.norm(self.proj(x)) @ self.ortho.t()


class DimReconcileBlock(nn.Module):
    """低维模态 → target_dim 的调解层：局部提炼 + 稠密升维 + 门控保护 + 冻结正交去相关。"""
    def __init__(self, d_src, d_target, n_heads=4):
        super().__init__()
        h = max(1, n_heads)
        self.use_attn = (d_src % h == 0) and d_src >= h
        if self.use_attn:
            self.local_refine = nn.MultiheadAttention(d_src, h, batch_first=True)
        else:
            self.local_refine = nn.Sequential(nn.Linear(d_src, d_src), nn.GELU(),
                                              nn.Linear(d_src, d_src))
        self.refine_norm = nn.LayerNorm(d_src)
        self.up_proj = nn.Linear(d_src, d_target)
        self.gate = nn.Linear(d_src, d_target)
        self.out_norm = nn.LayerNorm(d_target)
        self.register_buffer('ortho', self._make_ortho(d_target))

    @staticmethod
    def _make_ortho(d):
        w = torch.empty(d, d)
        nn.init.orthogonal_(w)
        return w.detach()

    def forward(self, x):
        if self.use_attn:
            refined, _ = self.local_refine(x, x, x)
        else:
            refined = self.local_refine(x)
        refined = self.refine_norm(x + refined)
        up = self.up_proj(refined)
        g = torch.sigmoid(self.gate(refined))
        out = self.out_norm(up * g)
        return out @ self.ortho.t()


class HeteroMoEProjSpace(nn.Module):
    def __init__(s, modal_dims=(896, 256, 768), d_shared=256, ne=4, top_k=2,
                 use_hyper=False, use_fast=False, hyper_rank=8, fast_decay=0.9,
                 use_recurrent_gate=False, use_memory_router=False,
                 use_hebbian=False, use_modality_adapter=False, use_feedback=False,
                 use_sparse=False, use_moa=False, moa_steps=1, moa_mode='sparse_router',
                 pm_reconcile=False, pm_target=1024, use_vl=False):
        super().__init__()
        s.pm_reconcile = pm_reconcile
        if pm_reconcile:
            s.d_shared = pm_target
            max_dim = max(modal_dims)
            s.modal_proj = nn.ModuleList([
                DimDownProj(D, pm_target) if D == max_dim
                else DimReconcileBlock(D, pm_target)
                for D in modal_dims])
        else:
            s.d_shared = d_shared
            s.P_m = nn.ParameterList([nn.Parameter(torch.empty(d_shared, D)) for D in modal_dims])
            for m in range(len(modal_dims)):
                nn.init.orthogonal_(s.P_m[m])
                s.P_m[m].requires_grad_(False)
        s.proj_space = DimProjectionSpace(
            s.d_shared, MEASURED, ne, top_k,
            use_hyper=use_hyper, hyper_rank=hyper_rank,
            use_recurrent_gate=use_recurrent_gate,
            use_memory_router=use_memory_router,
            use_hebbian=use_hebbian,
            use_modality_adapter=use_modality_adapter)
        s.d_space = s.proj_space.total_dim
        if use_moa:
            from moa_fusion import MoAFusionOverProjSpace
            s.fusion = MoAFusionOverProjSpace(
                s.d_space, num_proposers=ne, top_k=top_k,
                aggregator_mode=moa_mode, num_refinement_steps=moa_steps)
        elif use_sparse:
            from sparse_block_fusion import SparseBlockFusion
            s.fusion = SparseBlockFusion(
                s.d_space, n_blocks=16, block_ff=256, top_blocks=3, num_experts=ne)
        else:
            s.fusion = FusionOverProjSpace(
                s.d_space, None, ne, top_k, use_fast=use_fast, fast_decay=fast_decay)
        s.use_feedback = use_feedback
        if use_feedback:
            from feedback_head import RecursiveFeedbackHead
            # 每槽实际宽度取自 subspaces (pm_reconcile=True 时每槽宽 256 而非 s.d_shared=1024)
            s._fb_slot_dims = [s.proj_space.subspaces[k].dim for k in s.proj_space.slot_names]
            s.feedback_head = RecursiveFeedbackHead(
                s.d_space, s.proj_space.n_slots, max(s._fb_slot_dims), rank=4)
        s.use_vl = use_vl
        if use_vl:
            _vl_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else None
            for _cand in [_vl_dir, '/kaggle/working/hetero-moe/research/aligner/v62_dimension_projection_space']:
                if _cand and _cand not in sys.path and os.path.isdir(_cand):
                    sys.path.insert(0, _cand)
            from vl_fusion import CrossModalVisionLanguageFusionSubOp as _VLFusion
            # forward_from_projected 不经过 in_proj，设 use_in_proj=False 避免 64.5% 死参数
            s.vl_fusion = _VLFusion(d_model=d_shared, num_anchors=16, use_in_proj=False)
            # pm_reconcile=True 时 pr[m] 宽为 s.d_shared(1024)，先投至 d_shared(256) 再进入 vl_fusion
            s.vl_text_proj = nn.Linear(s.d_shared, d_shared, bias=False)
            s.vl_vis_proj = nn.Linear(s.d_shared, d_shared, bias=False)
            s.vl_out_proj = nn.Linear(d_shared, s.d_space, bias=False)

    def forward(s, hidden, fused_only=True, slot_acts=None):
        names = ['Qwen', 'BERT', 'ViT']
        pr = {}
        for i, m in enumerate(names):
            if m in hidden:
                if s.pm_reconcile:
                    pr[m] = s.modal_proj[i](hidden[m])
                else:
                    pr[m] = hidden[m] @ s.P_m[i].t()
        vl_out = None
        if getattr(s, 'use_vl', False):
            txt = [pr[m] for m in ['Qwen', 'BERT'] if m in pr]
            if txt:
                if len(txt) > 1 and any(t.shape[1] != txt[0].shape[1] for t in txt[1:]):
                    s_txt = min(t.shape[1] for t in txt)
                    txt = [
                        t if t.shape[1] == s_txt else nn.functional.adaptive_avg_pool1d(
                            t.transpose(1, 2), s_txt).transpose(1, 2)
                        for t in txt
                    ]
                t_stream = s.vl_text_proj(sum(txt) / len(txt))
            else:
                t_stream = None
            v_stream = s.vl_vis_proj(pr['ViT']) if 'ViT' in pr else None
            if t_stream is not None and v_stream is not None:
                if v_stream.shape[1] != t_stream.shape[1]:
                    v_stream = nn.functional.adaptive_avg_pool1d(
                        v_stream.transpose(1, 2), t_stream.shape[1]
                    ).transpose(1, 2)
                vl_out = s.vl_fusion.forward_from_projected(v_stream, t_stream)
        feats = s.proj_space.build_slot_feats(pr, slot_acts=slot_acts)
        struct, canon, aux = s.proj_space(feats, return_aux=True)
        aux['slot_source'] = getattr(s.proj_space, 'last_slot_source', 'unknown')
        if vl_out is not None:
            if vl_out.shape[1] != canon.shape[1]:
                vl_out = nn.functional.adaptive_avg_pool1d(
                    vl_out.transpose(1, 2), canon.shape[1]
                ).transpose(1, 2)
            canon = canon + s.vl_out_proj(vl_out)
        if fused_only:
            y, alpha = s.fusion.forward_with_routing(canon)
            aux['moa_aux_loss'] = getattr(s.fusion, 'last_aux_loss', 0.0)
            if s.use_feedback:
                conf, deltas = s.feedback_head(y)
                fb_terms = []
                offset = 0
                for i, k in enumerate(s.proj_space.slot_names):
                    dim_k = s.proj_space.subspaces[k].dim
                    part = canon[..., offset:offset + dim_k]
                    offset += dim_k
                    d_i = deltas[i][:dim_k, :dim_k]
                    fb_terms.append(part @ d_i.t())
                y = y + s.feedback_head.lambda_ * torch.cat(fb_terms, dim=-1)
                aux['feedback_conf'] = conf
            return y, alpha, aux
        return canon, aux


def main():
    t0 = time.time()
    print('device:', DEV, flush=True)
    if DEV == 'cuda':
        print(torch.cuda.get_device_name(0), flush=True)
    from huggingface_hub import hf_hub_download, HfApi

    print('\n[1/5] 载入缓存 ...', flush=True)
    if _LOCAL and os.path.isdir(_LOCAL):
        p = {f: os.path.join(_LOCAL, f) for f in _FILES_CACHE
             if os.path.exists(os.path.join(_LOCAL, f))}
    else:
        p = {f: hf_hub_download(R, f, repo_type='dataset') for f in _FILES_CACHE}

    h_q = np.load(p['h_qwen%s.npy' % _SUF], mmap_mode='r')
    h_b = np.load(p['h_bert%s.npy' % _SUF], mmap_mode='r')
    h_v = np.load(p['h_vit%s.npy' % _SUF], mmap_mode='r')
    y_t = np.load(p['labels%s.npy' % _SUF])

    if ABL != 'text' and h_v.shape[0] != h_q.shape[0] and not ALLOW_MISMATCHED_PAIRING:
        raise ValueError(
            f'视觉样本数 ({h_v.shape[0]}) 与文本样本数 ({h_q.shape[0]}) 不匹配！'
            '禁止静默取模伪多模态配对（如需复现历史错配审计请显式设置 ALLOW_MISMATCHED_PAIRING=1）。'
        )

    tr_idx, va_idx, te_idx = stratified_three_way_split(y_t, N_CLS, 0.70, 0.15, seed=0)
    print('   严格三集划分: train %d / val %d / test %d' % (len(tr_idx), len(va_idx), len(te_idx)), flush=True)

    # ---- 强制单底座线性探针基线 ----
    print('\n[2/5] 训练单底座线性探针基线 ...', flush=True)
    singles = {
        'Qwen': fit_single_backbone_probe(h_q, y_t, tr_idx, va_idx, te_idx, N_CLS, dev=DEV),
        'BERT': fit_single_backbone_probe(h_b, y_t, tr_idx, va_idx, te_idx, N_CLS, dev=DEV),
    }
    if ABL != 'text' and h_v.shape[0] == h_q.shape[0]:
        singles['ViT'] = fit_single_backbone_probe(h_v, y_t, tr_idx, va_idx, te_idx, N_CLS, dev=DEV)
    best_single_name = max(singles, key=lambda k: singles[k]['test_acc'])
    best_single_acc = singles[best_single_name]['test_acc']
    print('   单底座基线: %s | best_single=%s (%.4f)' % (singles, best_single_name, best_single_acc), flush=True)

    torch.manual_seed(0)
    inferred_dims = (int(h_q.shape[-1]), int(h_b.shape[-1]), int(h_v.shape[-1]))
    modal_dims = inferred_dims if inferred_dims != _MODAL_DIMS and 'MODAL_DIMS' not in os.environ else _MODAL_DIMS
    model = HeteroMoEProjSpace(
        modal_dims=modal_dims,
        use_hyper=_USE_HYPER, use_fast=_USE_FAST,
        use_recurrent_gate=_USE_RECUR, use_memory_router=_USE_MEM,
        use_hebbian=_USE_HEBB, use_modality_adapter=_USE_ADAPT,
        use_feedback=_USE_FB, use_sparse=_USE_SPARSE,
        use_moa=_USE_MOA, moa_steps=_MOA_STEPS, moa_mode=_MOA_MODE,
        pm_reconcile=_USE_PMR, pm_target=_PM_TARGET, use_vl=_USE_VL,
    ).to(DEV)
    head = nn.Linear(model.d_space, N_CLS).to(DEV)
    trainable = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(trainable, lr=LR)
    ce = nn.CrossEntropyLoss()

    def _eval_split(idx_arr):
        model.eval()
        head.eval()
        corr, tot = 0, 0
        with torch.no_grad():
            for s in range(0, len(idx_arr), 200):
                b = idx_arr[s:s + 200]
                hq = torch.from_numpy(np.asarray(h_q[b], dtype=np.float32)).to(DEV)
                hb = torch.from_numpy(np.asarray(h_b[b], dtype=np.float32)).to(DEV)
                vi = b % h_v.shape[0]
                hv = torch.from_numpy(np.asarray(h_v[vi], dtype=np.float32)).to(DEV)
                y, _, _ = model(_feats(hq, hb, hv), fused_only=True)
                pred = head(y.mean(1)).argmax(-1)
                corr += (pred == torch.as_tensor(y_t[b], device=DEV)).sum().item()
                tot += len(b)
        model.train()
        head.train()
        return corr / max(1, tot)

    rng = np.random.default_rng(0)
    best_val_acc, best_state, best_head = -1.0, None, None
    print('\n[3/5] 训练融合模型（val 选模，test 完全隔离）...', flush=True)
    for step in range(STEPS + 1):
        bi = np.sort(rng.choice(tr_idx, size=min(BS, len(tr_idx)), replace=False))
        hq = torch.from_numpy(np.asarray(h_q[bi], dtype=np.float32)).to(DEV)
        hb = torch.from_numpy(np.asarray(h_b[bi], dtype=np.float32)).to(DEV)
        vi = bi % h_v.shape[0]
        hv = torch.from_numpy(np.asarray(h_v[vi], dtype=np.float32)).to(DEV)
        lab = torch.as_tensor(y_t[bi], dtype=torch.long, device=DEV)

        y, alpha, aux = model(_feats(hq, hb, hv), fused_only=True)
        loss_ce = ce(head(y.mean(1)), lab)
        loss_consist = model.proj_space.consistency_loss()
        loss = loss_ce + 0.1 * loss_consist
        moa_aux = aux.get('moa_aux_loss', 0.0)
        if torch.is_tensor(moa_aux):
            loss = loss + moa_aux
        elif isinstance(moa_aux, (int, float)) and moa_aux != 0.0:
            loss = loss + moa_aux
        if step < STEPS:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
        if step % 250 == 0 or step == STEPS:
            va_acc = _eval_split(va_idx)
            if va_acc > best_val_acc:
                best_val_acc = va_acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_head = {k: v.detach().clone() for k, v in head.state_dict().items()}

    print('\n[4/5] 在未触碰的 test 集上评估最终权重 ...', flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
        head.load_state_dict(best_head)
    test_acc = _eval_split(te_idx)
    margin = test_acc - best_single_acc
    print('   val_acc=%.4f | test_acc=%.4f | best_single=%s(%.4f) | margin=%+.4f'
          % (best_val_acc, test_acc, best_single_name, best_single_acc, margin), flush=True)

    OUT = '/tmp/v2'
    os.makedirs(OUT, exist_ok=True)
    report = {
        'version': VER,
        'split': '分层严格三集 70/15/15 (train/val/test 零重叠)',
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'singles': singles,
        'best_single': {'name': best_single_name, 'test_acc': best_single_acc},
        'fusion_minus_best_single': margin,
    }
    with open(f'{OUT}/report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    if HFT:
        api = HfApi(token=HFT)
        CKPT_DIR = os.environ.get('CKPT_DIR', 'checkpoints/v%s' % VER)
        api.upload_file(path_or_fileobj=f'{OUT}/report.json',
                        path_in_repo=f'{CKPT_DIR}/report.json',
                        repo_id='redmingwei/HeteroMoEProjSpace-v2', repo_type='model',
                        commit_message=f'v{VER}: report.json')
    print('总耗时 %.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
