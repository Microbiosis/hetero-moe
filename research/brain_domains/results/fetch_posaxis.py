"""fetch_posaxis.py — 取 exp89 的结果 JSON 到本地（普通重试，不碰网卡）。

## 为什么要它

`hf buckets cp` 的读通道**时断时续**（实测撞 `xet-read-token`），
但 `hf jobs logs` 的读通道一直可用 ⇒ 两手都要：

- **桶里的 JSON**（`exp89_posaxis.json`）有**完整的逐通道曲线**与 `ch_names`
  ⇒ 头皮图要用它；
- **日志重建**（`parse_posaxis_from_logs.py`）有全部组级数字，
  ⇒ 即使下载不下来也能出表。

## 用法

    python research/brain_domains/results/fetch_posaxis.py
"""
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "posaxis-results"
BUCKET = "hf://buckets/illusion-brain/brain-domains/runs/illusionbrain-align"
REL = "exp89/exp89_posaxis.json"
DST = OUT / "exp89_posaxis.json"

TIMEOUT = 300
RETRIES = 5


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if DST.exists() and DST.stat().st_size > 0:
        print(f"已有 {DST}（{DST.stat().st_size} 字节）⇒ 跳过")
        return 0
    for i in range(RETRIES):
        try:
            r = subprocess.run(
                ["hf", "buckets", "cp", f"{BUCKET}/{REL}", str(DST)],
                capture_output=True, text=True, timeout=TIMEOUT)
            ok = r.returncode == 0 and DST.exists() and DST.stat().st_size > 0
            msg = (r.stderr or r.stdout or "").strip().splitlines()
            msg = msg[-1][:160] if msg else ""
        except subprocess.TimeoutExpired:
            ok, msg = False, "timeout"
        print(f"  第 {i+1} 次：{'ok' if ok else 'miss'} {msg}", flush=True)
        if ok:
            print(f"✓ 取到 {DST}（{DST.stat().st_size} 字节）")
            return 0
        if DST.exists() and DST.stat().st_size == 0:
            DST.unlink()
        time.sleep(20)
    print("✗ 没取到（走日志重建：parse_posaxis_from_logs.py）")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
