"""finalize_alignment.py — 等所有跑表 Job 结束 → 下载结果 → 合并出表。

用法：python finalize_alignment.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE / "alignment-results"

JOBS = {
    "t2-s01-stable": "6adae305c191903149a7a3d3",
    "t2-s01-dev": "6adae307925af0e27d4fa9e2",
    "t2-s02-stable": "6adae309c191903149a7a3d6",
    "t2-s02-dev": "6adae30ac191903149a7a3d8",
    "t2-s03-stable": "6adae30c925af0e27d4fa9e7",
    "t2-s03-dev": "6adae30e925af0e27d4fa9e9",
    "t2-s04-stable": "6adae310925af0e27d4fa9ef",
    "t2-s04-dev": "6adae312c191903149a7a3df",
    "t2-s05-stable": "6adae314c191903149a7a3e2",
    "t2-s05-dev": "6adae316925af0e27d4fa9f3",
    "t2-s06-stable": "6adae318c191903149a7a3e7",
    "t2-s06-dev": "6adae31ac191903149a7a3ea",
    "t2-s07-stable": "6adae31cc191903149a7a3ee",
    "t2-s07-dev": "6adae31ec191903149a7a3f0",
    "t2-s08-stable": "6adae320c191903149a7a3f5",
    "t2-s08-dev": "6adae322925af0e27d4fa9fb",
    "t2-s09-stable": "6adae324c191903149a7a3fa",
    "t2-s09-dev": "6adae326c191903149a7a3fe",
    "t2-s10-stable": "6adae328c191903149a7a402",
    "t2-s10-dev": "6adae32ac191903149a7a404",
}
TERM = ("COMPLETED", "ERROR", "FAILED", "CANCELLED")


def stages():
    r = subprocess.run(["hf", "jobs", "inspect"] + [f"redmingwei/{i}" for i in JOBS.values()]
                       + ["--format", "json"], capture_output=True, text=True,
                       timeout=240)
    try:
        return {j["id"]: j["status"]["stage"] for j in json.loads(r.stdout)}
    except Exception:
        return {}


deadline = time.time() + 2400
while time.time() < deadline:
    st = stages()
    if st:
        done = [n for n, i in JOBS.items() if st.get(i, "").startswith("COMPLETED")]
        bad = [n for n, i in JOBS.items() if st.get(i, "") in ("ERROR", "FAILED", "CANCELLED")]
        run = [n for n, i in JOBS.items() if st.get(i, "") not in TERM]
        print(f"完成 {len(done)}/{len(JOBS)}  失败 {len(bad)}  在跑 {len(run)}", flush=True)
        if not run:
            print(f"全部结束（失败 {bad}）", flush=True)
            break
    time.sleep(40)

print("\n=== 下载结果 ===", flush=True)
subprocess.run([sys.executable, str(HERE / "collect_alignment.py"), str(OUT)])

print("\n=== 合并出表 ===", flush=True)
r = subprocess.run([sys.executable, str(REPO / "exp" / "merge_alignment.py"),
                    "--dir", str(OUT / "stable"), "--dir", str(OUT / "dev"),
                    "--out", str(OUT)], capture_output=True, text=True)
print(r.stdout[-4000:])
print(r.stderr[-1500:])
