"""collect_alignment.py — 从桶里把两趟的 per-subject 结果拉回本地。

用法：python collect_alignment.py [--out <本地目录>]
桶路径：/domains/runs/illusionbrain-align/{stable,dev}/alignment_sub-XX.json
"""
import subprocess
import sys
from pathlib import Path

BUCKET = "hf://buckets/illusion-brain/brain-domains/runs/illusionbrain-align"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "alignment-results")

ok = 0
fail = []
for pas in ("stable", "dev"):
    d = OUT / pas
    d.mkdir(parents=True, exist_ok=True)
    for s in range(1, 11):
        name = f"alignment_sub-{s:02d}.json"
        src = f"{BUCKET}/{pas}/{name}"
        dst = d / name
        r = subprocess.run(["hf", "buckets", "cp", src, str(dst)],
                           capture_output=True, text=True, timeout=300)
        if r.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
            print(f"  ✓ {pas}/{name}  {dst.stat().st_size} B", flush=True)
            ok += 1
        else:
            print(f"  ✗ {pas}/{name}: {(r.stderr or r.stdout)[-120:]}", flush=True)
            fail.append(f"{pas}/{name}")
print(f"\n下载 {ok}/20，失败 {len(fail)}: {fail}")
