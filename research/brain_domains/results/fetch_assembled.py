"""fetch_assembled.py — 取「组件级组装」的评测与训练日志到本地。

## 为什么单独写一个（不借用 fetch_results.py）

`fetch_results.py` 的断点续传靠**换出口 IP**（`tools/wifi-cycle.ps1`）。
那是用户的红线：**WiFi 工具只有明确说了要用时才能用**，
所以本脚本**只做普通重试**，不碰网卡；下不动就如实报告缺哪些。

## 取什么

| 类型 | 桶内路径 | 本地 |
|---|---|---|
| 评测 | `exp81/{v1,v2}-ffn-<臂>/tokenizer_eval.json` | `assembled-results/{臂}.json` |
| 训练日志 | `exp86/ffn-<臂>/train_log.json`、`exp86v2/ffn-<臂>/train_log.json` | `assembled-results/train-{配方}-{臂}.json` |
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "assembled-results"
BUCKET = "hf://buckets/illusion-brain/brain-domains/runs/illusionbrain-align"

ARMS = ["none", "mirepnet", "biot", "eegdino-small"]
V2_ARMS = ARMS + ["none-nosem"]

# (桶内相对路径, 本地文件名)
TARGETS = []
for v in ARMS:
    TARGETS.append((f"exp81/v1-ffn-{v}/tokenizer_eval.json", f"v1-ffn-{v}.json"))
for v in V2_ARMS:
    TARGETS.append((f"exp81/v2-ffn-{v}/tokenizer_eval.json", f"v2-ffn-{v}.json"))
for v in ARMS:
    TARGETS.append((f"exp86/ffn-{v}/train_log.json", f"train-v1-ffn-{v}.json"))
for v in V2_ARMS:
    TARGETS.append((f"exp86v2/ffn-{v}/train_log.json", f"train-v2-ffn-{v}.json"))

TIMEOUT = 90
RETRIES = 3


def fetch(rel, name):
    dst = OUT / name
    if dst.exists() and dst.stat().st_size > 0:
        return "skip"
    try:
        r = subprocess.run(["hf", "buckets", "cp", f"{BUCKET}/{rel}", str(dst)],
                           capture_output=True, text=True, timeout=TIMEOUT)
        ok = r.returncode == 0 and dst.exists() and dst.stat().st_size > 0
    except subprocess.TimeoutExpired:
        ok = False
    if ok:
        return "ok"
    if dst.exists() and dst.stat().st_size == 0:
        dst.unlink()
    return "miss"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    got, miss = [], []
    for rel, name in TARGETS:
        state = "miss"
        for _ in range(RETRIES):
            state = fetch(rel, name)
            if state != "miss":
                break
            time.sleep(15)
        (got if state == "ok" else miss).append((rel, state))
        print(f"  {state:<5} {rel}", flush=True)
    print(f"\n取到 {len(got)} / 缺 {len(miss)}")
    if miss:
        print("缺：")
        for rel, _ in miss:
            print(f"  - {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
