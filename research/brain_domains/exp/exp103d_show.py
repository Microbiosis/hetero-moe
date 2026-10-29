#!/usr/bin/env python3
"""exp103d_show.py — 把 exp103d 的结果 JSON 里的**带覆盖表**打到日志（读通道不稳时的兜底）。"""
import json
from pathlib import Path

P = Path("/domains/runs/illusionbrain-align/exp103/exp103d_btb_bands.json")
d = json.loads(P.read_text(encoding="utf-8"))

print("=== 逐被试：带覆盖（DesikanKilliany，主映射 v_main）===")
cov = d["band_coverage"]["v_main"]
for sid, v in cov.items():
    dk = v.get("DesikanKilliany", {})
    bands = dk.get("bands", {})
    print(f"{sid:>8}: n_bands={dk.get('n_bands')}  "
          f"POST={bands.get('POST', 0):>3} 电极  CENT={bands.get('CENT', 0):>3} 电极  "
          f"FRONT={bands.get('FRONT', 0):>3} 电极   (≥2电极的带={dk.get('n_bands_ge2elec')})")

print("\n=== 变体一致性（每被试的带数）===")
for vn, per in d["band_coverage"].items():
    counts = [v["DesikanKilliany"]["n_bands"] for v in per.values() if "DesikanKilliany" in v]
    print(f"{vn:>24}: {counts}  ⇒ 三带齐全 {sum(1 for x in counts if x == 3)}/10")

print("\n=== 每被试的 DK 标签清单（前 12 个，按电极数）===")
for sid, rec in d["per_subject"].items():
    labs = rec["atlases"]["DesikanKilliany"]["labels"][:12]
    print(f"{sid:>8} ({rec['n_rows']} 电极): {labs}")
    ct = rec.get("conftype")
    if ct:
        print(f"{'':>8} ConfType: {ct}")

print("\n=== 五条规格判决 ===")
for k, v in d["verdict"].items():
    print(f"--- {k} [{v.get('status')}]")
    for kk, vv in v.items():
        if kk == "status":
            continue
        s = json.dumps(vv, ensure_ascii=False)
        print(f"    {kk}: {s[:600]}")

print("\n=== 坐标表 Region 列（每被试前 8）===")
for sid, rec in d["coords_per_subject"].items():
    print(f"{sid:>8}: n_rows={rec['n_rows']} n_distinct_region={rec['n_distinct_region']} "
          f"{rec['regions'][:8]}")
