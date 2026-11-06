"""v73 四槽/三模态独立性审计（独立可运行版，无需训练）。

借鉴 neuro-physical 仓库五维独立性审计的 Dim II（表示独立）+ Dim III（梯度子空间独立）。

修掉 v72 版的两个致命 bug：
  1. 曾用随机初始化的 DimProjectionSpace + load_state_dict(strict=False)，
     加载失败的键被静默忽略 → 等于审计「随机空间」而非训练好的模型。
     本版断言 missing_keys 必须为空。
  2. 曾漏掉 P_m 投影，把 2048 维原始 hidden 直接喂给期望 256 维的槽投影 → 维度报错。
     本版用 ckpt 里的 P_m 权重做投影。

审计对象：训练好的 hetero-moe 模型（v65.5 权重）
  Dim II  四槽 canonical 的表示独立性：联合子空间秩 + 两两|cos| + VIF
  Dim III 三模态输入梯度的子空间夹角（各自 SVD 取 k 维主子空间再比）

用法（Kaggle）:
  python slot_independence_audit.py --ckpt <model.pt> --cache <cache_dir> --suf _v65
"""
import os, sys, argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--suf', default='_v65')
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--k', type=int, default=8, help='主子空间维数')
    ap.add_argument('--grad_n', type=int, default=16, help='Dim III 梯度样本数')
    args = ap.parse_args()

    import torch
    import importlib.util as ilu

    # 定位 v62 包目录（含 projspace.py）
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [here,
             '/kaggle/working/hetero-moe/research/aligner/v62_dimension_projection_space']
    pkg = next((c for c in cands
                if os.path.isfile(os.path.join(c, 'projspace.py'))), cands[0])
    sys.path.insert(0, pkg)
    spec = ilu.spec_from_file_location('v62_projspace',
                                       os.path.join(pkg, 'projspace.py'))
    ps_mod = ilu.module_from_spec(spec); sys.modules['v62_projspace'] = ps_mod
    spec.loader.exec_module(ps_mod)
    DimProjectionSpace = ps_mod.DimProjectionSpace
    MEASURED = ps_mod.MEASURED_SUBSPACES
    print('projspace 目录:', pkg, flush=True)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(args.ckpt, map_location='cpu')
    sd = ck.get('model', ck)

    # --- 关键：proj_space 必须完整加载，missing 不允许静默 ---
    ps_sd = {k[len('proj_space.'):]: v for k, v in sd.items() if k.startswith('proj_space.')}
    assert ps_sd, 'ckpt 里没有 proj_space.* 参数'
    ps = DimProjectionSpace(256, MEASURED, 4, 2)
    missing, unexpected = ps.load_state_dict(ps_sd, strict=False)
    assert not missing, 'proj_space 权重缺失(等于审计了随机空间): %s' % missing[:6]
    print('proj_space 加载完成 (%d 键), unexpected=%d' % (len(ps_sd), len(unexpected)), flush=True)
    ps = ps.to(dev).eval()

    # --- P_m 权重（ckpt 里）---
    P_m = []
    for i in range(3):
        key = 'P_m.%d' % i
        assert key in sd, 'ckpt 缺 %s' % key
        P_m.append(sd[key].to(dev).float())
    print('P_m shapes:', [tuple(p.shape) for p in P_m], flush=True)

    slots = ps.slot_names
    D = ps.d_shared
    cache = args.cache
    hq = np.load(f'{cache}/h_qwen{args.suf}.npy', mmap_mode='r')
    hb = np.load(f'{cache}/h_bert{args.suf}.npy', mmap_mode='r')
    hv = np.load(f'{cache}/h_vit{args.suf}.npy', mmap_mode='r')
    print('cache:', hq.shape, hb.shape, hv.shape, flush=True)

    def canon_of(q_t, b_t, v_t):
        pr = {'Qwen': q_t @ P_m[0].t(),
              'BERT': b_t @ P_m[1].t(),
              'ViT':  v_t @ P_m[2].t()}
        feats = {m: {k: pr[m] for k in slots} for m in pr}
        with torch.no_grad():
            _, cn, _ = ps(feats, return_aux=True)
        return cn

    idx = np.random.default_rng(0).choice(hq.shape[0], min(args.n, hq.shape[0]), replace=False)

    # ---------- Dim II ----------
    print('\n=== Dim II 四槽前向表示独立性 ===', flush=True)
    reps = {k: [] for k in slots}
    with torch.no_grad():
        for s in range(0, len(idx), 64):
            b = idx[s:s+64]
            q_t = torch.from_numpy(np.asarray(hq[b], dtype=np.float32)).to(dev)
            b_t = torch.from_numpy(np.asarray(hb[b], dtype=np.float32)).to(dev)
            v_t = torch.from_numpy(np.asarray(hv[b % hv.shape[0]], dtype=np.float32)).to(dev)
            cn = canon_of(q_t, b_t, v_t)
            for i, k in enumerate(slots):
                reps[k].append(cn[..., i*D:(i+1)*D].mean(1).cpu().numpy())
    reps = {k: np.concatenate(v) for k, v in reps.items()}

    Fv = []
    for k in slots:
        f = reps[k].ravel(); f = f - f.mean()
        Fv.append(f / (np.linalg.norm(f) + 1e-15))
    Fm = np.stack(Fv, axis=1)
    cos = np.abs(Fm.T @ Fm)
    off = ~np.eye(len(slots), dtype=bool)
    rank = int(np.linalg.matrix_rank(Fm, tol=1e-5))
    vifs = np.diag(np.linalg.inv(Fm.T @ Fm + 1e-8*np.eye(len(slots))))
    print('联合子空间秩: %d/%d (满秩才名实相符)' % (rank, len(slots)))
    print('两两|cos| 均值 %.4f 最大 %.4f' % (cos[off].mean(), cos[off].max()))
    print('VIF: %s (>10=严重共线)' % np.round(vifs, 2).tolist())
    print('判定:', '✅ 四槽表示独立' if (rank == len(slots) and cos[off].max() < 0.6)
          else '❌ 四槽共线/秩不足 → 印证"融合只是源身份探测器"')
    print('[DIM2_RESULT] rank=%d/%d mean_cos=%.4f max_cos=%.4f vif=%s' % (
        rank, len(slots), cos[off].mean(), cos[off].max(),
        np.round(vifs, 2).tolist()), flush=True)

    # ---------- Dim III ----------
    print('\n=== Dim III 三模态输入梯度子空间独立 ===', flush=True)
    b = idx[:args.grad_n]      # Dim III 只需少量样本算梯度（CPU 反传慢）
    q_t = torch.from_numpy(np.asarray(hq[b], dtype=np.float32)).to(dev)
    b_t = torch.from_numpy(np.asarray(hb[b], dtype=np.float32)).to(dev)
    # ViT 197 patch vs 文本 128 token → 截断到 128 使行数与 token 位置对齐
    v_t = torch.from_numpy(np.asarray(hv[b % hv.shape[0]][:, :128, :], dtype=np.float32)).to(dev)
    Us = []
    for name, base in [('Qwen', q_t), ('BERT', b_t), ('ViT', v_t)]:
        src = base.clone().requires_grad_(True)
        # 关键：必须先过 P_m 投影到 d_shared(256)，否则 2048 维直接喂 256 维槽投影会维度报错
        _mi = {'Qwen': 0, 'BERT': 1, 'ViT': 2}[name]
        pr = {'Qwen': q_t.detach() @ P_m[0].t(),
              'BERT': b_t.detach() @ P_m[1].t(),
              'ViT':  v_t.detach() @ P_m[2].t()}
        pr[name] = src @ P_m[_mi].t()
        feats = {m: {k: pr[m] for k in slots} for m in pr}
        _, cn, _ = ps(feats, return_aux=True)
        cn.sum().backward()
        g = src.grad.reshape(-1, src.shape[-1]).cpu().numpy()      # [N, D_m]
        # 用 Gram 矩阵 (D×D) 特征分解取右奇异向量：避免对 [N,D] 做完整 SVD（CPU 上极慢）
        # 主子空间取「输入方向」= V（g 的列空间），即梯度在输入维度上的主方向
        G = g.T @ g                                                # [D_m, D_m]
        w, V = np.linalg.eigh(G)                                   # 升序
        order = np.argsort(w)[::-1][:args.k]                        # 最大 k 个特征值
        Vk = V[:, order]                                           # [D_m, k] 列正交
        S = np.sqrt(np.maximum(w[order], 0))
        Us.append(Vk.T)                                            # [k, D_m] 行正交（便于比夹角）
        print('  %s 梯度 %s → 主方向 %s (能量 %.3f)' % (
            name, tuple(g.shape), Vk.shape,
            float((S**2).sum() / max((w**2).sum(), 1e-12))), flush=True)
    # 各模态主子空间维度不同(D_m×k) → 用投影算子比较夹角:
    # |Ui^T Pj Ui| 的最大奇异值 = 两个子空间夹角
    sc = []
    for i in range(3):
        for j in range(i+1, 3):
            Pi = Us[i].T @ Us[i]          # [D_i, D_i] 投影到 Qi 的列空间
            # 把 Qj 的基投到 Qi 的空间: Pi @ (列正交基 of Qj) 的奇异值
            M = Pi @ Us[j].T              # [D_i, k]
            sc.append(float(np.linalg.svd(M, compute_uv=False)[0]))
    print('三模态梯度子空间主夹角(cos): %s 均值 %.4f 最大 %.4f' % (
        np.round(sc, 4).tolist(), float(np.mean(sc)), float(np.max(sc))))
    print('判定:', '✅ 三模态梯度子空间独立（各司其职）' if float(np.max(sc)) < 0.9
          else '⚠️ 三模态梯度高度共线 → 某模态冗余')
    print('[DIM3_RESULT] angles=%s mean=%.4f max=%.4f' % (
        np.round(sc, 4).tolist(), float(np.mean(sc)), float(np.max(sc))), flush=True)
    print('\n[done]', flush=True)


if __name__ == '__main__':
    main()
