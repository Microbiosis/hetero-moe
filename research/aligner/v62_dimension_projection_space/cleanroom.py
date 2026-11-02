"""净室验证：只靠 clone + 下载，检验「别人拿到能不能用」。

和 tests/test_package.py 的区别：
  test_package.py  验**实现对不对**（在作者本机、已装好依赖的环境里）
  cleanroom.py     验**交付能不能被第三方用起来**（干净目录 + 只从 GitHub/Hub 拉）

六个步骤，每步独立记录原始错误，任一失败不影响其余：
  1 从 GitHub 干净 clone（--depth 1），记录 commit
  2 子包必备文件是否都在（不依赖作者本机残留）
  3 从 clone 出来的包能否直接 import 并前向
  4 包内权威测试能否在 clone 里跑通
  5 文档声称的加载方式 AutoModel.from_pretrained(trust_remote_code=True) 是否真能用
  6 用 Hub 上的 example_inputs 复现 expected_y

用法：
    python research/aligner/v62_dimension_projection_space/cleanroom.py
    # 结果写到 stdout，同时可用 --json 输出机器可读报告
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import traceback
from typing import Any, Dict, List, Optional

REPO_URL = 'https://github.com/Microbiosis/hetero-moe.git'
HF_REPO = 'redmingwei/HeteroMoEProjSpace-v2'
SUBPKG = 'research/aligner/v62_dimension_projection_space'

#: 复现容差。expected_y 是在 transformers 5.21.0 下生成的；
#: 其它版本的骨干 forward 实现有微小数值差异，实测 5.16.1 下 ~7.9e-05。
#: 这里给 1e-3（相对量级 ~2e-6），既能抓出真错，也不因版本漂移误报。
REPRO_TOL = 1e-3

_steps: List[Dict[str, Any]] = []


def step(name: str):
    def deco(fn):
        rec: Dict[str, Any] = {'step': name, 'ok': False, 'detail': '', 'error': ''}
        try:
            d = fn()
            rec['ok'] = True
            rec['detail'] = '' if d is None else str(d)
        except Exception as e:                                   # noqa: BLE001
            rec['error'] = f'{type(e).__name__}: {e}'
            rec['traceback'] = traceback.format_exc()[-1200:]
        _steps.append(rec)
        mark = '  PASS  ' if rec['ok'] else '  FAIL  '
        print(mark + name, flush=True)
        if rec['ok'] and rec['detail']:
            print('          ' + rec['detail'][:400], flush=True)
        if not rec['ok']:
            print('          ' + rec['error'][:400], flush=True)
        return fn
    return deco


def sh(cmd: str, timeout: int = 900, cwd: Optional[str] = None) -> str:
    r = subprocess.run(['bash', '-lc', cmd], capture_output=True, text=True,
                       timeout=timeout, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError(f'命令失败 ({r.returncode}): {cmd}\n{r.stderr[-800:]}')
    return r.stdout


def main(workdir: str, json_out: Optional[str]) -> int:
    fresh = os.path.join(workdir, 'fresh')

    print('=' * 70)
    print('净室验证 — 只靠 clone + 下载')
    print('=' * 70)

    # ---------------- 1 clone
    print('\n[1] 从 GitHub 干净 clone')

    @step('1a clone hetero-moe')
    def _():
        sh(f'rm -rf {fresh} && git clone --depth 1 {REPO_URL} {fresh}')
        return 'commit ' + sh('git rev-parse --short HEAD', cwd=fresh).strip()

    @step('1b 子包必备文件齐全')
    def _():
        pkg = os.path.join(fresh, SUBPKG)
        need = ['MANIFEST.json', 'SPEC.md', 'projspace.py', 'fusion.py', 'model.py',
                'measure.py', 'results/metrics.json', 'tests/test_package.py',
                '__init__.py']
        missing = [f for f in need if not os.path.exists(os.path.join(pkg, f))]
        if missing:
            raise AssertionError(f'缺: {missing}')
        return f'{len(need)} 个文件全部到位'

    # ---------------- 2 版本
    print('\n[2] 环境版本')

    @step('2a 记录版本')
    def _():
        import numpy, torch, transformers
        return (f'torch {torch.__version__} | transformers {transformers.__version__}'
                f' | numpy {numpy.__version__}')

    # ---------------- 3 import
    print('\n[3] clone 出来的包能否直接 import')

    @step('3a import + 前向')
    def _():
        sys.path.insert(0, fresh)
        import torch
        from research.aligner.v62_dimension_projection_space.projspace import (  # noqa: E402
            MEASURED_SUBSPACES, DimProjectionSpace)
        op = DimProjectionSpace(256)
        torch.manual_seed(0)
        feats = {m: {k: torch.randn(2, 8, 256) for k in MEASURED_SUBSPACES}
                 for m in ['Qwen', 'BERT', 'ViT']}
        struct, canon, alpha = op(feats)
        return (f'struct {tuple(struct.shape)} canon {tuple(canon.shape)} '
                f'alpha {tuple(alpha.shape)} | 路由槽 {op.route_slots}')

    # ---------------- 4 包内测试
    print('\n[4] 包内权威测试能否在 clone 里跑通')

    @step('4a tests/test_package.py')
    def _():
        p = os.path.join(fresh, SUBPKG, 'tests/test_package.py')
        out = sh(f'{sys.executable} {p} 2>&1', timeout=1200, cwd=fresh)
        tail = [ln for ln in out.splitlines() if ln.startswith('合计')]
        if not tail or 'ALL PASS' not in tail[-1]:
            raise AssertionError(f'未全过: {tail}\n{out[-1200:]}')
        return tail[-1]

    # ---------------- 5 文档声称的加载方式
    print('\n[5] 文档声称的加载方式')

    @step('5a AutoModel.from_pretrained(trust_remote_code=True)')
    def _():
        from transformers import AutoModel
        m = AutoModel.from_pretrained(HF_REPO, trust_remote_code=True)
        return f'{type(m).__name__} | {m.describe().splitlines()[1].strip()}'

    # ---------------- 6 复现
    print('\n[6] 用 Hub 上的 example_inputs 复现 expected_y')

    @step('6a 复现 expected_y / expected_alpha')
    def _():
        import numpy as np
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel

        m = AutoModel.from_pretrained(HF_REPO, trust_remote_code=True).float().eval()
        npz = np.load(hf_hub_download(HF_REPO, 'example_inputs.npz'))
        inp = {k: torch.from_numpy(npz[k]) for k in npz.files}
        exp_y = torch.from_numpy(np.load(hf_hub_download(HF_REPO, 'expected_y.npy')))
        exp_a = torch.from_numpy(np.load(hf_hub_download(HF_REPO, 'expected_alpha.npy')))
        with torch.no_grad():
            out = m(**inp)
        dy = float((out.y - exp_y).abs().max())
        da = float((out.alpha - exp_a).abs().max())
        if dy > REPRO_TOL or da > REPRO_TOL:
            raise AssertionError(f'Δy={dy:.3e} Δalpha={da:.3e} 超容差 {REPRO_TOL}')
        return f'max|Δy|={dy:.3e}  max|Δalpha|={da:.3e}  （容差 {REPRO_TOL}）'

    @step('6b 缺模态可用（只给 Qwen）')
    def _():
        import numpy as np
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel

        m = AutoModel.from_pretrained(HF_REPO, trust_remote_code=True).float().eval()
        npz = np.load(hf_hub_download(HF_REPO, 'example_inputs.npz'))
        with torch.no_grad():
            out = m(qwen_input_ids=torch.from_numpy(npz['qwen_input_ids']),
                    qwen_attention_mask=torch.from_numpy(npz['qwen_attention_mask']))
        return f'纯文本输出 {tuple(out.y.shape)}'

    # ---------------- 汇总
    print('\n' + '=' * 70)
    n_ok = sum(1 for s in _steps if s['ok'])
    print(f'合计 {n_ok}/{len(_steps)}')
    for s in _steps:
        print(('  ✓ ' if s['ok'] else '  ✗ ') + s['step'])
    print('=' * 70)

    report = {'steps': _steps, 'pass': n_ok, 'total': len(_steps),
              'repro_tol': REPRO_TOL}
    if json_out:
        with open(json_out, 'w') as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        print(f'\n报告: {json_out}')

    return 0 if n_ok == len(_steps) else 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workdir', default=None, help='临时目录（默认自动创建）')
    ap.add_argument('--json', default=None, help='机器可读报告输出路径')
    args = ap.parse_args()

    wd = args.workdir or tempfile.mkdtemp(prefix='cleanroom_')
    sys.exit(main(wd, args.json))
