"""fetch_results.py — 韧性下载：**每换一次出口 IP 只下一个文件**。

观察（2026-10-23）：下载通道是**按出口 IP 限流**的 ——
刚换 IP 后可以下 1 轮（7 个文件），随后同一 IP 的下载全部挂起。
⇒ 整轮批量重试是浪费（13 个文件 × 45 s 白等），
   正确做法：**一次 IP 只下一个文件，下不动就换 IP 再下一个**。
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "alignment-results"
BUCKET = "hf://buckets/illusion-brain/brain-domains/runs/illusionbrain-align"
WIFI = HERE.parent.parent / "tools" / "wifi-cycle.ps1"

WANTED = [("stable", s) for s in range(1, 11)] + [("dev", s) for s in range(1, 11)]
TIMEOUT = 50
MAX_CYCLES = 60


def missing():
    out = []
    for pas, s in WANTED:
        d = OUT / pas
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"alignment_sub-{s:02d}.json"
        if not (f.exists() and f.stat().st_size > 0):
            out.append((pas, s, f))
    return out


def wifi_cycle():
    subprocess.run(["pwsh", "-NoProfile", "-File", str(WIFI), "-DelaySeconds", "20"],
                   capture_output=True, text=True, timeout=240)


miss = missing()
print(f"缺 {len(miss)} 个文件", flush=True)
cycles = 0
while miss and cycles < MAX_CYCLES:
    # 一次 IP 连打：能下几个下几个
    for item in list(miss):
        pas, s, f = item
        try:
            r = subprocess.run(["hf", "buckets", "cp", f"{BUCKET}/{pas}/{f.name}", str(f)],
                               capture_output=True, text=True, timeout=TIMEOUT)
            ok = r.returncode == 0 and f.exists() and f.stat().st_size > 0
        except subprocess.TimeoutExpired:
            ok = False
        if ok:
            print(f"  ✓ {pas}/{f.name}", flush=True)
            miss.remove(item)
        else:
            break                      # 这个 IP 到此为止，换 IP
    if miss:
        cycles += 1
        print(f"  换 IP（第 {cycles} 次），还缺 {len(miss)}", flush=True)
        wifi_cycle()

print(f"\n最终缺失 {len(miss)}：{[f'{p}/{s:02d}' for p, s, _ in miss]}", flush=True)
