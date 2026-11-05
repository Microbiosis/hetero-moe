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

    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import importlib.util as ilu
    spec = ilu.spec_from_file_location('v62_projspace', 'projspace.py')
    ps_mod = ilu.module_from_spec(spec); sys.modules['v62_projspace'] = ps_mod
    spec.loader.exec_module(ps_mod)
    DimProjectionSpace = ps_mod.DimProjectionSpace; MEASURED = ps_mod.MEASURED_SUBSPACES

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ps = DimProjectionSpace(256, MEASURED, 4, 2).to(dev)
    sd = torch.load(args.ckpt, map_location=dev)
    ps.load_state_dict(sd.get('proj_space', sd), strict=False)
    ps.eval()
    slots = ps.slot_names                                  # ['Q','K','V','O']

    hq = np.load(f'{args.cache}/h_qwen{args.suf}.npy', mmap_mode='r')
    hb = np.load(f'{args.cache}/h_bert{args.suf}.npy', mmap_mode='r')
    hv = np.load(f'{args.cache}/h_vit{args.suf}.npy', mmap_mode='r')
    rng = np.random.default_rng(0)
    idx = rng.choice(hq.shape[0], min(args.n, hq.shape[0]), replace=False)

    def canon_of(hq_t, hb_t, hv_t):
        feats = {'Qwen': {k: hq_t for k in slots},
                 'BERT': {k: hb_t for k in slots},
                 'ViT':  {k: hv_t for k in slots}}
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
    for m, base in [('Qwen', hq_t), ('BERT', hb_t), ('ViT', hv_t)]:
        src = base.clone().requires_grad_(True)
        pr = {'Qwen': hq_t.detach(), 'BERT': hb_t.detach(), 'ViT': hv_t.detach()}
        pr[m] = src
        feats = {mm: {k: pr[mm] for k in slots} for mm in pr}
        _, canon, _ = ps(feats, return_aux=True)
        canon.sum().backward()
        g = src.grad.ravel().cpu().numpy(); g = g - g.mean()
        Gs.append(g / (np.linalg.norm(g) + 1e-15))
    Gm = np.stack(Gs, axis=1)
    gcos = np.abs(Gm.T @ Gm); goff = ~np.eye(3, dtype=bool)
    grank = int(np.linalg.matrix_rank(Gm, tol=1e-5))
    print('梯度子空间秩: %d/3' % grank)
    print('梯度两两|cos| 均值 %.4f 最大 %.4f' % (gcos[goff].mean(), gcos[goff].max()))
    print('判定:', '✅ 三模态梯度独立' if (grank==3 and gcos[goff].max()<0.8)
          else '⚠️ 三模态梯度共线 → 某模态冗余')

    print('\n[done]')

if __name__ == '__main__':
    main()
