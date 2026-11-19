# -*- coding: utf-8 -*-
"""读出端补测 (post_hoc_only) —— 正文侧 (token 级 r2_q) 的内容特异 / 槽特异读数.

问题: v82 的六项判据只测了注意力侧的连通/寻址 (P5a/P5b) 与正文侧的整体增量 (P5c/P5d);
"正文侧的内容特异性与槽特异性各是多少" 没有单独测过 => 本脚本补测.

口径: 统计量与 v81 (v81_s3_semantic) 的 P1a/P1b **同款** (正文 token 级 r2_q 的臂间差)
=> arm1-arm2 可与 v81 的 P1b 直接比较; 基线 arm0 两代口径不同 (S3 管道不可见 /
v82 管道开内容 0) 但**臂间差不受 arm0 影响**.

置换: 块内符号翻转 + 块中位的均值, 10^4 次 (统计量与上游 v81 `perm_paired` 一致);
这里保留**本地实现**是为了把 RNG 种子 (11/12/13/14) 写死在本脚本里, 使已归档的
读数 (results/diag_readout.json) 可逐位复现.

用法: python research/weight_fusion/v82_s4_pipe/readout_diag.py
"""
import collections
import json
from pathlib import Path
import numpy as np

PKG = Path(__file__).resolve().parent


def perm(pairs, n_perm=10000, seed=0):
    """块级配对置换检验 (整块同翻号 + Phipson–Smyth (cnt+1)/(n_perm+1) 平滑)。"""
    if not pairs:
        return None
    byb = collections.defaultdict(list)
    for b, a, c in pairs:
        byb[b].append(a - c)
    lens = set(len(v) for v in byb.values())
    assert len(lens) == 1, lens
    D = np.array(list(byb.values()), dtype=float)  # [n_blocks, n_seeds]
    obs = float(np.median(D, axis=1).mean())
    rng = np.random.default_rng(seed)
    # 同一块内的 5 个 seed 共享确定性语料与层，必须按块整体翻号而非独立翻 seed
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, D.shape[0], 1))
    st = np.median(D[None, :, :] * signs, axis=2).mean(axis=1)
    cnt = int((st >= obs - 1e-15).sum())
    return float((cnt + 1) / (n_perm + 1))


def stat(pairs):
    d = np.array([x - y for _b, x, y in pairs])
    byb = collections.defaultdict(list)
    for b, x, y in pairs:
        byb[b].append(x - y)
    bm = np.array([np.median(v) for v in byb.values()])
    return {'n_obs': len(pairs), 'n_blocks': len(byb), 'median_obs_delta': float(np.median(d)),
            'n_positive_obs': int((d > 0).sum()), 'median_block_delta': float(np.median(bm)),
            'mean_block_delta': float(bm.mean())}


def main():
    with open(PKG / 'results' / 'blocks.jsonl', encoding='utf-8') as f:
        rows = [json.loads(l) for l in f if l.strip()]
    bl = collections.defaultdict(dict)
    for r in rows:
        bl[(r['direction'], r['source_layer'], r['target_layer'], r['seed'])][r['arm']] = r

    out = {'role': 'post_hoc_only —— **读出端解读诊断**, 不参与 S4-P 任何裁决, 不推翻任何阈值',
           'question': 'v82 的六项判据只测了注意力侧的连通/寻址 (P5a/P5b); 正文侧 (token 级 r2_q) 的内容特异与槽特异各是多少?',
           'note': '正文侧 r2_q 的臂间差是 S3 (v81) 同款口径 (P1a/P1b 的同一统计量), 故 arm1-arm2 可与 v81 的 P1b 直接比较; 基线 arm0 两代口径不同 (S3 管道不可见 / v82 管道开内容0) 但臂间差不受 arm0 影响。'}
    defs = [('arm1_minus_arm0_presence', 'arm1_slot_Q', 'arm0_pipe_zero', 11),
            ('arm1_minus_arm3_content_specificity', 'arm1_slot_Q', 'arm3_random_dir', 12),
            ('arm1_minus_arm2_slot_specificity', 'arm1_slot_Q', 'arm2_wrong_slot_O', 13),
            ('arm3_minus_arm0_random_noise', 'arm3_random_dir', 'arm0_pipe_zero', 14)]
    for name, a, c, sd in defs:
        # 块键取 k[:3] = (direction, source_layer, target_layer)（64 块 × 5 seed，而非把 seed 算进块键）
        pairs = [(k[:3], v[a]['r2_q'], v[c]['r2_q']) for k, v in bl.items() if a in v and c in v]
        s = stat(pairs); s['permutation_p'] = perm(pairs, seed=sd); s['n_perm'] = 10000
        out[name] = s
        bydir = {}
        for d in ('Qwen->BERT', 'BERT->Qwen'):
            pp = [(k[:3], v[a]['r2_q'], v[c]['r2_q']) for k, v in bl.items() if k[0] == d and a in v and c in v]
            sd2 = stat(pp); sd2['permutation_p'] = perm(pp, seed=sd); bydir[d] = sd2
        out[name]['by_direction'] = bydir
        print('%-38s 块级中位 %+.6f  正 %3d/%3d  p=%.4f | Qwen->BERT %+.6f(p=%.4f) BERT->Qwen %+.6f(p=%.4f)'
              % (name, s['median_block_delta'], s['n_positive_obs'], s['n_obs'], s['permutation_p'],
                 bydir['Qwen->BERT']['median_block_delta'], bydir['Qwen->BERT']['permutation_p'],
                 bydir['BERT->Qwen']['median_block_delta'], bydir['BERT->Qwen']['permutation_p']))
    out['comparison_v81_P1b'] = {'v81_S3_arm1_minus_arm2_r2q': -0.002455, 'v81_p': 0.9521,
                            'v82_S4P_arm1_minus_arm2_r2q': out['arm1_minus_arm2_slot_specificity']['median_block_delta'],
                            'v82_p': out['arm1_minus_arm2_slot_specificity']['permutation_p'],
                            'note': '同口径 (正文 token 级 r2_q 的 arm1-arm2), 臂间差不受两代 arm0 口径影响'}
    with open(PKG / 'results' / 'diag_readout.json', 'w', encoding='utf-8', newline='\r\n') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print()
    print('written -> results/diag_readout.json')
    print('v81 P1b 对照:', json.dumps(out['comparison_v81_P1b'], ensure_ascii=False))


if __name__ == '__main__':
    main()