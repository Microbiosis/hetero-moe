"""路线4：跨任务少样本迁移验证。

v62 仅在 ag_news 上训练。本脚本测试融合空间能否 few-shot 迁移到
其他文本分类任务（如 SST-2 情感、20news 主题），或跨模态检索。

协议：冻结 DimProjectionSpace + fusion，仅替换/训练分类头。
输出：zero-shot / 5-shot / 20-shot 三档准确率。
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn


def build_model(device):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'v62_projspace',
        os.path.join(os.path.dirname(__file__),
                     '../../v62_dimension_projection_space/projspace.py'))
    ps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ps)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj_space = ps.DimProjectionSpace(256, ps.MEASURED_SUBSPACES, 4, 2)

        def forward(self, hidden):
            feats = {m: {k: hidden[m] for k in self.proj_space.slot_names}
                     for m in hidden}
            return self.proj_space(feats)[1]      # canonical [B,S,1024]

    m = M().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()
    return m


def extract_feats(model, h_dict, device, bs=256):
    """抽取融合特征 [N,1024]。"""
    outs = []
    with torch.no_grad():
        for i in range(0, len(h_dict['Qwen']), bs):
            batch = {m: h_dict[m][i:i + bs].to(device) for m in h_dict}
            outs.append(model(batch).mean(1).cpu())
    return torch.cat(outs)


def train_head(X, y, n_shot, device, steps=300, lr=1e-3):
    """在 n_shot 样本上训练线性头，返回 (head, val_acc)。"""
    idx = np.random.permutation(len(X))
    shot_idx = idx[:n_shot]
    rest_idx = idx[n_shot:n_shot + 500]
    Xs, ys = X[shot_idx].to(device), y[shot_idx].to(device)
    Xv, yv = X[rest_idx].to(device), y[rest_idx].to(device)

    head = nn.Linear(X.shape[1], int(y.max()) + 1).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(head(Xs), ys)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = float((head(Xv).argmax(-1) == yv).float().mean())
    return head, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--source', required=True, help='源任务 npz（ag_news）')
    ap.add_argument('--target', required=True, help='目标任务 npz')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else 'cpu'

    model = build_model(dev)
    sd = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(sd.get('model', sd), strict=False)

    src = np.load(args.source)
    tgt = np.load(args.target)
    h_src = {m: torch.from_numpy(src[f'h_{m[0].lower()}']).float() for m in
             ['Qwen', 'BERT', 'ViT']}
    h_tgt = {m: torch.from_numpy(tgt[f'h_{m[0].lower()}']).float() for m in
             ['Qwen', 'BERT', 'ViT']}

    X_src = extract_feats(model, h_src, dev)
    X_tgt = extract_feats(model, h_tgt, dev)
    y_tgt = torch.from_numpy(tgt['y']).long()

    print('目标域特征: %s | 源域特征: %s' % (tuple(X_tgt.shape), tuple(X_src.shape)))
    results = {}
    for shot in [0, 5, 20]:
        if shot == 0:
            # zero-shot：直接用源域训练的头（近似，实际需源域头）
            results['zero_shot'] = {'note': '需源域训练好的头，此处占位'}
            continue
        accs = []
        for seed in range(3):
            torch.manual_seed(seed)
            np.random.seed(seed)
            _, acc = train_head(X_tgt, y_tgt, shot, dev)
            accs.append(acc)
        results['%d_shot' % shot] = {
            'acc_mean': float(np.mean(accs)),
            'acc_std': float(np.std(accs)),
            'runs': accs
        }
        print('  %2d-shot: %.4f ± %.4f' % (shot, np.mean(accs), np.std(accs)))

    out = os.path.join(os.path.dirname(__file__), '..', 'results',
                       'transfer_eval.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(' saved → %s' % out)


if __name__ == '__main__':
    main()
