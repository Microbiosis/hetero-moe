"""检查 leakage_audit 的审计链：pending 的包是否其实已有审计证据。

存在的理由
----------
`leakage_audit.status = pending` 有两种完全不同的情形：

  (a) 真的没人查过          → 应保持 pending
  (b) 查过了，但状态字段没回写 → 应改成 clean / invalidated

校验器只看字段，分不出这两者，于是 (b) 会被当成负债一直挂着。
本脚本从各包自己的 `results/metrics.json` 里**重新读出判据**，
判定每个 pending 包属于哪一种，并打印依据。

它不修改任何文件 —— 判定结果由人复核后再改 MANIFEST。

用法:
    python tools/audit_leakage_chain.py
    python tools/audit_leakage_chain.py --json out.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]

#: 审计链：被审计包 -> (审计包, 该审计里对应本包的判据前缀)
#: 依据是各审计包 MANIFEST 里显式写出的审计对象
AUDIT_CHAIN = {
    '49.0': [('52.0', 'C1c')],           # v52 的 C1c 直接检验「有意义空间 margin 是否减半」
    '50.0': [('52.0', 'C1')],            # v52 是 v50 的独立复算
    '51.0': [('52.0', 'C2')],            # v52 同时复算 v51（随机目标臂）
    '52.0': [('53.0', 'D5')],            # 三审制第三审显式声明「复现 v52」
}


def load(pkg_id: str) -> Tuple[Optional[Path], Dict[str, Any]]:
    """按版本号找包目录并读 metrics.json。"""
    for man in glob.glob(str(ROOT / 'research' / '*' / 'v*' / 'MANIFEST.json')):
        try:
            d = json.loads(Path(man).read_text())
        except Exception:  # noqa: BLE001
            continue
        if d.get('version') == pkg_id:
            pkg = Path(man).parent
            mp = pkg / 'results' / 'metrics.json'
            if mp.exists():
                try:
                    return pkg, json.loads(mp.read_text())
                except Exception:  # noqa: BLE001
                    return pkg, {}
            return pkg, {}
    return None, {}


def verdicts_of(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """把各包不同形状的 verdicts 摊平成 {判据: bool}。"""
    out: Dict[str, Any] = {}
    v = metrics.get('verdicts')
    if isinstance(v, dict):
        for k, val in v.items():
            if isinstance(val, dict):
                for k2, v2 in val.items():
                    out[f'{k}.{k2}'] = v2
            else:
                out[k] = val
    if 'all_pass' in metrics:
        out['all_pass'] = metrics['all_pass']
    return out


def check_chain(pkg_id: str) -> Optional[Dict[str, Any]]:
    """该 pending 包能否由下游审计包结清。"""
    if pkg_id not in AUDIT_CHAIN:
        return None
    evid: List[Dict[str, Any]] = []
    for auditor, prefix in AUDIT_CHAIN[pkg_id]:
        apkg, ametrics = load(auditor)
        if apkg is None:
            evid.append({'auditor': auditor, 'found': False})
            continue
        vs = verdicts_of(ametrics)
        hits = {k: v for k, v in vs.items() if prefix in k.split('.')[-1]
                or k.split('.')[-1].startswith(prefix)}
        passed = [k for k, v in hits.items() if v is True]
        failed = [k for k, v in hits.items() if v is False]
        evid.append({
            'auditor': auditor,
            'metrics': str((apkg / 'results' / 'metrics.json').relative_to(ROOT)),
            'criteria_matched': sorted(hits),
            'passed': sorted(passed),
            'failed': sorted(failed),
            'conclusive': bool(passed) and not failed,
        })
    return {'audited_by': evid}


def self_audit(pkg_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """该包自己的 metrics 里有没有 held-out / 对照 这类无泄漏复测。"""
    s = json.dumps(metrics, ensure_ascii=False).lower()
    marks = {
        'has_held_out': any(k in s for k in
                            ('held_out', 'heldout', 'held-out', '留出', 'held')),
        'has_null': any(k in s for k in
                        ('shuffle', 'permut', 'null', '置换', '零分布', 'chance')),
        'has_both_arms': ('frozen' in s and ('trainable' in s or 'train_proj' in s)),
        'has_probe_selftest': ('probe' in s or '探针' in s),
    }
    return marks


def main(json_out: Optional[str]) -> int:
    pending: List[Dict[str, Any]] = []

    for man in sorted(glob.glob(str(ROOT / 'research' / '*' / 'v*' / 'MANIFEST.json'))):
        try:
            d = json.loads(Path(man).read_text())
        except Exception:  # noqa: BLE001
            continue
        la = d.get('leakage_audit') or {}
        if la.get('status') != 'pending':
            continue
        pkg = Path(man).parent
        pkg_id = d.get('version', '?')
        mp = pkg / 'results' / 'metrics.json'
        metrics = {}
        if mp.exists():
            try:
                metrics = json.loads(mp.read_text())
            except Exception:  # noqa: BLE001
                pass
        rec = {
            'version': pkg_id,
            'package': str(pkg.relative_to(ROOT)),
            'title': d.get('title', '')[:60],
            'has_metrics': mp.exists(),
            'chain': check_chain(pkg_id),
            'self_audit': self_audit(pkg_id, metrics) if metrics else {},
        }

        # 判定
        chain = rec['chain']
        if chain and chain['audited_by'] and all(e.get('conclusive') for e in chain['audited_by']):
            rec['recommend'] = 'clean'
            rec['reason'] = '下游审计包的对应判据全部通过 → 审计已实际完成，状态字段未回写'
        elif chain and any(e.get('failed') for e in chain['audited_by']):
            rec['recommend'] = 'invalidated'
            rec['reason'] = '下游审计包在对应判据上有未过项 → 部分主张被泄漏推翻'
        elif not rec['has_metrics']:
            rec['recommend'] = 'keep_pending'
            rec['reason'] = '没有 metrics.json，无法从产物复核'
        else:
            marks = rec['self_audit']
            if marks.get('has_held_out') and marks.get('has_null'):
                rec['recommend'] = 'clean_candidate'
                rec['reason'] = '本包自带 held-out 与置换零分布对照，但无下游独立复算包'
            else:
                rec['recommend'] = 'keep_pending'
                rec['reason'] = '产物里缺 held-out 或置换对照'
        pending.append(rec)

    print('=' * 78)
    print('leakage_audit 审计链检查')
    print('=' * 78)
    for r in pending:
        print(f"\nv{r['version']}  {r['package']}")
        print(f"  标题        {r['title']}")
        print(f"  metrics     {'有' if r['has_metrics'] else '无'}")
        if r['self_audit']:
            print(f"  自带对照    {r['self_audit']}")
        if r['chain']:
            for e in r['chain']['audited_by']:
                if not e.get('found', True):
                    print(f"  审计包      {e['auditor']} 未找到")
                    continue
                print(f"  审计包      {e['auditor']}  ->  {e['metrics']}")
                print(f"    命中判据  {e['criteria_matched']}")
                print(f"    通过      {e['passed']}")
                print(f"    未过      {e['failed']}")
        print(f"  → 建议      【{r['recommend']}】 {r['reason']}")

    print('\n' + '=' * 78)
    tally: Dict[str, int] = {}
    for r in pending:
        tally[r['recommend']] = tally.get(r['recommend'], 0) + 1
    print('汇总:', ', '.join(f'{k}={v}' for k, v in sorted(tally.items())))
    print('=' * 78)

    if json_out:
        Path(json_out).write_text(json.dumps(pending, indent=2, ensure_ascii=False))
        print(f'\n报告: {json_out}')
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    raise SystemExit(main(args.json))
