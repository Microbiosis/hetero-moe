#!/usr/bin/env python3
"""exp98_run_shard.py — **运行方式**的包装器（**判据一字未改**）：把某个分片的 npz 汇到一个目录，
再用**原样的** `exp98_posneg_control.py` 跑这个分片。

## 为什么要有它

被判决脚本 `exp98_posneg_control.py` **只在整轮末尾写一次结果文件**
⇒ Job 一旦超时（**平台上限 60 分钟**）⇒ 已算完的 session **全部丢失**
⇒ ⇒ 改成分片跑：每片几分钟、每片各写各的结果 ⇒ 聚合器再合并（`exp98_aggregate.py`）。

⚠️ **本脚本不碰判据**：它只做「选哪些 session → 准备目录 → 用原样的判据脚本跑」。
分片结果与整轮结果**逐位等价**（判据脚本内所有随机数由固定 seed 派生，与 session 名/分片无关）。

## 用法

```powershell
python -u exp98_run_shard.py --shard 0 --nshards 8
```
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import urllib.request

DANDI = "000021"
VER = "0.251116.2246"
BASE = f"https://api.dandiarchive.org/api/dandisets/{DANDI}/versions/{VER}"
JUDGE = "/domains/scripts/exp98_posneg_control.py"


def get_json(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.loads(r.read().decode())


def all_assets():
    out, url = [], f"{BASE}/assets/?page_size=100&fields=asset_id,path,size"
    while url:
        d = get_json(url)
        out += d.get("results", [])
        url = d.get("next")
    return out


def session_names():
    assets = all_assets()
    return sorted(a["path"].split("/")[0] for a in assets
                  if a["path"].endswith(".nwb") and "_probe-" not in a["path"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--src", default="/domains/allen_trials")
    ap.add_argument("--out", default="/domains/runs/illusionbrain-align/exp98")
    ap.add_argument("--work", default="")
    args = ap.parse_args()
    names = session_names()
    mine = [n for i, n in enumerate(names) if i % args.nshards == args.shard]
    src = pathlib.Path(args.src)
    work = pathlib.Path(args.work or f"/domains/tmp_s{args.shard}")
    work.mkdir(parents=True, exist_ok=True)
    missing = []
    for n in mine:
        s = src / f"{n}.npz"
        if not s.exists():
            missing.append(n)
            continue
        d = work / f"{n}.npz"
        if not d.exists():
            shutil.copy(s, d)
    print(f"shard {args.shard}/{args.nshards}: {len(mine)} 个 session，缺 {missing}", flush=True)
    out = pathlib.Path(args.out)
    (out / f"shard{args.shard}").mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", JUDGE, "--data", str(work), "--out", str(out / f"shard{args.shard}")]
    print("RUN " + " ".join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    print(f"===RC=== {rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
