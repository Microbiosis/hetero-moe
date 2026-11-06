"""v71 四槽独立性审计（借鉴 neuro-physical 仓库 Dim II/III 方法论）。

针对 v48-v54 审计链结论「融合输出只是源身份探测器」——四槽 Q/K/V/O 可能名不副实。
对已训练 v65.5 模型的四槽 canonical 做：
  Dim II  前向表示独立性：四槽切片联合子空间秩 + 两两 |cos| + VIF
  Dim III 输入梯度子空间独立：三模态(Qwen/BERT/ViT)输入梯度的子空间秩 + |cos|

用法（Kaggle）: python slot_independence_audit.py --ckpt <ckpt.pt> --cache <dir>
"""
import os, sys, argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='含 proj_space state_dict 的 ckpt')
    ap.add_argument('--cache', required=True, help='cache-v65 目录')
    ap.add_argument('--suf', default='_v65')
    ap.add_argument('--n', type=int, default=400)
    args = ap.parse_args()

    import torch, os as _o, urllib.request as _ur
    # 自包含：下载 v62 算子文件到本地包目录（不依赖 git clone / notebook 编排）
    _BASE = 'https://raw.githubusercontent.com/Microbiosis/hetero-moe/aa7184d/research/aligner/v62_dimension_projection_space/'
    _FILES = ['projspace.py','hyper_proj.py','self_repair.py','recurrent_gate.py',
              'memory_router.py','hebbian_align.py','modality_adapter.py',
              'feedback_head.py','fusion.py','moa_operators.py','moa_fusion.py']
    _pkgdir = '/tmp/v62pkg'
    _o.makedirs(_pkgdir, exist_ok=True)
    for _f in _FILES:
        _p = _o.path.join(_pkgdir, _f)
        if not _o.path.exists(_p) or _o.path.getsize(_p) < 100:
            _ur.urlretrieve(_BASE + _f, _p)
    if _pkgdir not in sys.path:
        sys.path.insert(0, _pkgdir)
    import importlib.util as ilu
    spec = ilu.spec_from_file_location('v62_projspace',
                                       _o.path.join(_pkgdir, 'projspace.py'))
    ps_mod = ilu.module_from_spec(spec); sys.modules['v62_projspace'] = ps_mod
    spec.loader.exec_module(ps_mod)
    DimProjectionSpace = ps_mod.DimProjectionSpace; MEASURED = ps_mod.MEASURED_SUBSPACES
    print('projspace 目录:', _pkgdir, flush=True)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ps = DimProjectionSpace(256, MEASURED, 4, 2).to(dev)
    sd = torch.load(args.ckpt, map_location=dev)
    msd = sd.get('model', sd)
    ps.load_state_dict({k[len('proj_space.'):]: v for k, v in msd.items()
                        if k.startswith('proj_space.')}, strict=False)
    ps.eval()
    slots = ps.slot_names                                  # ['Q','K','V','O']
    # P_m: 模态 hidden -> d_shared(256) 正交冻结投影（v62 架构关键一步）
    P_M = []
    for _i in range(3):
        _k = None
        for _cand in ['P_m.%d' % _i, 'P_m.%d.weight' % _i]:
            if _cand in msd:
                _k = _cand; break
        if _k is None:
            raise KeyError('ckpt 缺少 P_m.%d；现有 P_m 键: %s'
                           % (_i, [x for x in msd if 'P_m' in x][:6]))
        P_M.append(msd[_k].to(dev))
    print('P_m shapes:', [tuple(x.shape) for x in P_M], flush=True)
    _EXP = [(256, 2048), (256, 768), (256, 1152)]
    for _i, _p in enumerate(P_M):
        if tuple(_p.shape) != _EXP[_i]:
            raise ValueError('P_m.%d 形状 %s，期望 %s' % (_i, tuple(_p.shape), _EXP[_i]))

    hq = np.load(f'{args.cache}/h_qwen{args.suf}.npy', mmap_mode='r')
    hb = np.load(f'{args.cache}/h_bert{args.suf}.npy', mmap_mode='r')
    hv = np.load(f'{args.cache}/h_vit{args.suf}.npy', mmap_mode='r')
    rng = np.random.default_rng(0)
    idx = rng.choice(hq.shape[0], min(args.n, hq.shape[0]), replace=False)

    def canon_of(hq_t, hb_t, hv_t):
        # 先过 P_m 投影（模态维 → 256），再进四槽投影空间
        pr = {'Qwen': hq_t @ P_M[0].t(),
              'BERT': hb_t @ P_M[1].t(),
              'ViT':  hv_t @ P_M[2].t()}
        feats = {m: {k: pr[m] for k in slots} for m in pr}
        _, canon, _ = ps(feats, return_aux=True)
        return canon

    # ---------- Dim II: 四槽前向表示独立性 ----------
    print('提取四槽 canonical ...', flush=True)
    reps = {k: [] for k in slots}
    with torch.no_grad():
        for s in range(0, len(idx), 64):
            b = idx[s:s+64]
            hq_t = torch.from_numpy(np.asarray(hq[b],dtype=np.float32)).to(dev)
            hb_t = torch.from_numpy(np.asarray(hb[b],dtype=np.float32)).to(dev)
            hv_t = torch.from_numpy(np.asarray(hv[b % hv.shape[0]],dtype=np.float32)).to(dev)
            canon = canon_of(hq_t, hb_t, hv_t)              # [B,S,1024]
            for i, k in enumerate(slots):
                reps[k].append(canon[..., i*256:(i+1)*256].mean(1).cpu().numpy())
    reps = {k: np.concatenate(v) for k, v in reps.items()}

    print('\n=== Dim II 四槽前向表示独立性 ===')
    F = []
    for k in slots:
        f = reps[k].ravel(); f = f - f.mean()
        F.append(f / (np.linalg.norm(f) + 1e-15))
    Fm = np.stack(F, axis=1)
    cos = np.abs(Fm.T @ Fm); off = ~np.eye(len(slots), dtype=bool)
    rank = int(np.linalg.matrix_rank(Fm, tol=1e-5))
    vifs = np.diag(np.linalg.inv(Fm.T @ Fm + 1e-8*np.eye(len(slots))))
    print('联合子空间秩: %d/%d (满秩才名实相符)' % (rank, len(slots)))
    print('两两|cos| 均值 %.4f 最大 %.4f' % (cos[off].mean(), cos[off].max()))
    print('VIF: %s (>10=严重共线)' % np.round(vifs, 2).tolist())
    print('判定:', '✅ 四槽表示独立' if (rank==len(slots) and cos[off].max()<0.6)
          else '❌ 四槽共线/秩不足 → 印证"融合只是源身份探测器"')

    # ---------- Dim III: 三模态输入梯度子空间独立 ----------
    print('\n=== Dim III 三模态输入梯度子空间独立 ===')
    b = idx[:64]
    hq_t = torch.from_numpy(np.asarray(hq[b],dtype=np.float32)).to(dev)
    hb_t = torch.from_numpy(np.asarray(hb[b],dtype=np.float32)).to(dev)
    hv_t = torch.from_numpy(np.asarray(hv[b % hv.shape[0]],dtype=np.float32)).to(dev)
    Gs = []
    _MIDX = {'Qwen': 0, 'BERT': 1, 'ViT': 2}
    for m, base in [('Qwen', hq_t), ('BERT', hb_t), ('ViT', hv_t)]:
        src = base.clone().requires_grad_(True)
        pr = {'Qwen': hq_t @ P_M[0].t(), 'BERT': hb_t @ P_M[1].t(),
              'ViT': hv_t @ P_M[2].t()}
        pr[m] = src @ P_M[_MIDX[m]].t()
        feats = {mm: {k: pr[mm] for k in slots} for mm in pr}
        _, canon, _ = ps(feats, return_aux=True)
        canon.sum().backward()
        g = src.grad.ravel().cpu().numpy(); g = g - g.mean()
        Gs.append(g / (np.linalg.norm(g) + 1e-15))
    # 三模态 hidden 维度不同（2048/768/1152），梯度向量长度不同 → 不能直接堆叠。
    # 改为：各自中心化后投影到公共子空间（随机高斯投影到 256 维）再比较方向；
    # 同时用各自长度做 Gram，报告两两 |cos|。
    _rng = np.random.default_rng(7)
    def _proj(g):
        d = g.shape[0]
        R = _rng.standard_normal((d, 256)) / np.sqrt(d)
        q = g @ R
        return q / (np.linalg.norm(q) + 1e-15)
    Gm = np.stack([_proj(g) for g in Gs], axis=1)
    gcos = np.abs(Gm.T @ Gm); goff = ~np.eye(3, dtype=bool)
    grank = int(np.linalg.matrix_rank(Gm, tol=1e-5))
    print('梯度子空间秩(公共投影后): %d/3' % grank)
    print('梯度两两|cos| 均值 %.4f 最大 %.4f' % (gcos[goff].mean(), gcos[goff].max()))
    print('判定:', '✅ 三模态梯度独立' if (grank==3 and gcos[goff].max()<0.8)
          else '⚠️ 三模态梯度共线 → 某模态冗余')

    print('\n[done]')

if __name__ == '__main__':
    main()
