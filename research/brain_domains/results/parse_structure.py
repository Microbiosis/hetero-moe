#!/usr/bin/env python3
"""parse_structure.py — 解析结构勘查日志，做**嫁接可行性**的本地判定。

输入：`results/structure/{stable,dev}.log`（`hf jobs logs` 抓下来的原始日志）
输出：
1. 每个模型的结构摘要（block 数 / token 维 / token 数 / 层明细数）；
2. **嫁接可行性表**：宿主 × 供体，逐层比对 mixer / FFN 的权重形状，
   给出三值判定（compatible / needs_projection / arch_mismatch）；
3. 供体目标层的维度（取自 exp75 的逐层曲线，与阶段 D 的抽取同源）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE.parent / "models"
for _p in (str(HERE.parent), str(MODELS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import variants as V          # noqa: E402

STRUCT_DIR = HERE / "structure"
LAYERWISE = HERE / "alignment-results" / "layerwise_parsed.json"


def parse_log(path):
    """从日志里抽出 ###MODEL### 的 JSON 行。

    ⚠️ `hf jobs logs` 会把超长行**截断在 16 KB**（实测）⇒ 日志只能当兜底，
    首选是桶里的 `structure/<version>.json`。
    """
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s.startswith("###MODEL###"):
            try:
                out.append(json.loads(s[len("###MODEL###"):].strip()))
            except Exception as e:
                print(f"  ⚠ 日志行被截断，解析失败：{e}")
    return out


def load_models():
    """优先读桶里下载回来的 JSON；没有才退回日志。"""
    models = {}
    for tag in ("stable", "dev"):
        p = STRUCT_DIR / f"{tag}.json"
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                for rec in data.get("models", []):
                    models[rec["model"]] = rec
                print(f"✓ {tag}.json 解析出 {len(data.get('models', []))} 个模型")
                continue
            except Exception as e:
                print(f"  ⚠ {tag}.json 解析失败：{e}")
        lp = STRUCT_DIR / f"{tag}.log"
        if lp.exists():
            recs = parse_log(lp)
            for rec in recs:
                models[rec["model"]] = rec
            print(f"✓ {tag}.log 解析出 {len(recs)} 个模型（日志兜底）")
    return models


def donor_layer_dims():
    """exp75 逐层曲线里，供体目标层的 token 维与 RSA。"""
    if not LAYERWISE.exists():
        return {}
    d = json.loads(LAYERWISE.read_text(encoding="utf-8"))
    out = {}
    for spec in V.DONORS:
        m = d.get(spec.donor)
        if not m:
            continue
        rows = m.get("rows") if isinstance(m, dict) else m
        hit = None
        for r in (rows or []):
            if r.get("module") == spec.layer:
                hit = r
                break
        out[spec.vid] = hit
    return out


def main():
    models = load_models()

    print("\n=== 结构摘要 ===")
    print(f"{'模型':<16}{'类':<22}{'参数':>10}  {'blocks':>6} {'d_model':>8} "
          f"{'n_tok':>6} {'层明细':>6}")
    for name, r in models.items():
        print(f"{name:<16}{str(r.get('class'))[:21]:<22}"
              f"{r.get('n_params', 0):>10,}  {r.get('n_blocks', 0):>6} "
              f"{str(r.get('d_model')):>8} {str(r.get('n_tokens')):>6} "
              f"{len(r.get('layer_detail', [])):>6}")

    # --- 嫁接可行性 ---
    host = models.get(V.HOST)
    print(f"\n=== 嫁接可行性（宿主 {V.HOST}）===")
    if host is None:
        print("✗ 没有宿主的结构记录")
        return 1
    for spec in V.DONORS:
        d = models.get(spec.donor)
        if d is None:
            print(f"\n-- {spec.vid}（{spec.donor}）：本趟没有它的结构记录")
            continue
        rows = V.feasibility(host, d, max_pairs=100000)
        n_ok = sum(1 for r in rows
                   if "compatible" in (r["mixer"]["verdict"], r["ffn"]["verdict"]))
        n_proj = sum(1 for r in rows
                     if "needs_projection" in (r["mixer"]["verdict"],
                                               r["ffn"]["verdict"]))
        v = V.graft_verdict(host, d, donor_layer_name=spec.layer)
        print(f"\n-- {spec.vid}（{spec.donor}）")
        print(f"   全扫描：宿主 {len(host.get('layer_detail', []))} 层 × "
              f"供体 {len(d.get('layer_detail', []))} 层 = {len(rows)} 个可比组合")
        print(f"   compatible = {n_ok} | needs_projection = {n_proj}")
        print(f"   **结论：{v['verdict']}** — {v['why']}")
        if v.get("host"):
            print(f"      宿主槽 {v['host']} ← 供体 {v['donor']}")
            print(f"      宿主形状 {v.get('host_shapes')}")
            print(f"      供体形状 {v.get('donor_shapes')}")

    # --- 供体目标层的维度（来自 exp75）---
    print("\n=== 供体目标层（exp75 口径）===")
    for vid, row in donor_layer_dims().items():
        if not row:
            print(f"  {vid:<22} （exp75 逐层结果里没有这一层）")
            continue
        print(f"  {vid:<22} dim={row.get('dim')} tok={row.get('n_tokens')} "
              f"r={row.get('r')} p={row.get('p')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
