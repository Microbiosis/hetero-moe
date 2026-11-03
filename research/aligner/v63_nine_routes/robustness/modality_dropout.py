"""路线3：单模态缺失鲁棒性测试。

v62 仅在三模态同时输入上验证。本脚本测试当某模态 dropout 时，
模型是否 graceful degrade，四槽如何重新分配。

用法（Kaggle/本地，需 torch + 已训练权重）:
    python modality_dropout.py --ckpt <v62.4.pt> --data <capture.npz>
输出：4 模态组合 × 4 类别的准确率 + slot_alpha 重分配热力图。
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch


MODALITIES = ['Qwen', 'BERT', 'ViT']
COMBOS = [
    ('all', MODALITIES),
    ('qwen_only', ['Qwen']),
    ('bert_only', ['BERT']),
    ('vit_only', ['ViT']),
    ('no_qwen', ['BERT', 'ViT']),
    ('no_bert', ['Qwen', 'ViT']),
    ('no_vit', ['Qwen', 'BERT']),
]


def load_model(ckpt_path: str, device: str = 'cuda'):
    """加载 v62 模型（与 train_v2.py 相同的 HeteroMoEProjSpace 结构）。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'v62_projspace',
        os.path.join(os.path.dirname(__file__),
                     '../../v62_dimension_projection_space/projspace.py'))
    ps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ps)

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj_space = ps.DimProjectionSpace(256, ps.MEASURED_SUBSPACES, 4, 2)

        def forward(self, hidden):
            feats = {m: {k: hidden[m] for k in self.proj_space.slot_names}
                     for m in hidden}
            return self.proj_space(feats, return_aux=True)

    model = M().to(device)
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd.get('model', sd), strict=False)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', required=True, help='npz with h_q/h_b/h_v/y')
    ap.add_argument('--head', default=None, help='分类头 ckpt')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() else 'cpu'
    model = load_model(args.ckpt, dev)
    d = np.load(args.data)
    h = {m: torch.from_numpy(d[f'h_{m[0].lower()}']).float().to(dev)
         for m in MODALITIES}
    y = torch.from_numpy(d['y']).long().to(dev)

    head = None
    if args.head and os.path.exists(args.head):
        head = torch.nn.Linear(1024, int(y.max()) + 1).to(dev)
        head.load_state_dict(torch.load(args.head, map_location=dev))
        head.eval()

    results = {}
    print('%-12s %-8s %-10s %s' % ('combo', 'acc', 'slot_ent', 'slot_alpha_mean'))
    for name, mods in COMBOS:
        feats = {m: h[m] for m in mods}
        with torch.no_grad():
            struct, canon, aux = model(feats)
            alpha = aux['alpha']                      # [B,S,4]
            alpha_mean = alpha.mean(dim=(0, 1)).cpu().numpy().round(4)
            ent = float(-(alpha * torch.log(alpha + 1e-9)).sum(-1).mean()
                        / np.log(4))
            if head is not None:
                pred = head(canon.mean(1)).argmax(-1)
                acc = float((pred == y).float().mean())
            else:
                acc = float('nan')
        results[name] = {'acc': acc, 'slot_ent': ent,
                         'alpha_mean': alpha_mean.tolist()}
        print('%-12s %-8.4f %-10.4f %s'
              % (name, acc, ent, alpha_mean.tolist()))

    out = os.path.join(os.path.dirname(__file__), '..', 'results',
                       'modality_dropout.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print('\n saved → %s' % out)


if __name__ == '__main__':
    main()
