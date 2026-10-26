"""parse_exp93d_level.py — exp93d（模型层读出 @ n_train=16540）的汇总。

## 它补的是哪个洞

exp92-C 当时的比较是 **n_train=1000** ⇒ 「我们自训的 tokenizer 不如手工基线」（A18）
有可能只是**训练图数不够**。exp93d 把同一批模型读出在 **n_train=16540**（全量）下重测，
看 A18 是否还成立。

**A13 口径的手工基线**（exp93a，同协议、同一批图像）：
630-d 手工 + ridge，单试次 **0.0321**（6.4× 随机，上中位，n=10）。

## 两个 Job 写**不同**的桶内目录（这次特意分开，避免互相覆盖）：

- `runs/illusionbrain-align/exp92-level-full-s1-5/level/level.json`
- `runs/illusionbrain-align/exp92-level-full-s6-10/level/level.json`

⇒ 本脚本按**被试**合并两段 ⇒ n=10。

## 统计口径

**中位 = 上中位**（`sorted[n//2]`，偶数取偏大）⇒ 右偏分布会报高 30–40%。
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp93-results"
BUCKET = "hf://buckets/illusion-brain/brain-domains/runs/illusionbrain-align"
CHANCE = 1.0 / 200.0

SEGMENTS = [
    ("exp92-level-full-s1-5", "level/level.json"),
    ("exp92-level-full-s6-10", "level/level.json"),
    ("exp93d-gpu", "exp93d_gpu.json"),     # ★ 第一版的产出（结果文件名保留为证据）
]
# 手工基线（exp93a，全量 16540，10 被试合并后）
HANDCRAFT = {"raw630": 0.0321}


def med(xs):
    """**上中位**（本项目约定）。"""
    xs = [float(x) for x in xs if x == x]
    if not xs:
        return None
    s = sorted(xs)
    return s[len(s) // 2]


def mean(xs):
    xs = [float(x) for x in xs if x == x]
    return float(sum(xs) / len(xs)) if xs else None


def fetch(seg_dir, rel, retries=5):
    dst = OUT / f"exp93d_{seg_dir.replace('exp92-level-full-', '')}.json"
    if not (dst.exists() and dst.stat().st_size > 0):
        src = f"{BUCKET}/{seg_dir}/{rel}"
        for i in range(retries):
            try:
                r = subprocess.run(["hf", "buckets", "cp", src, str(dst)],
                                   capture_output=True, text=True, timeout=420)
                if r.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
                    break
            except subprocess.TimeoutExpired:
                pass
            print(f"  重试 {i+1}/{retries}：{src}")
            time.sleep(6)
    if not (dst.exists() and dst.stat().st_size > 0):
        return None
    return json.loads(dst.read_text(encoding="utf-8"))


LOGF = HERE / "exp93d_level_job_log.txt"
RE_SUBJ = re.compile(r"^===\s*(sub-\d+)\s*===")
RE_LEVEL = re.compile(r"LEVEL\s+(?P<tag>\S+)\s+acc1=(?P<acc>[\d.]+)\s+\(零中位\s+(?P<nul>[\d.]+)\)")


def parse_log(path):
    """从 Job 日志解析逐被试的 LEVEL 行（**跑到几个算几个**）。

    exp93d 每个被试要编码 66160×2 次前向（比 exp92-C 贵 16.5 倍）⇒ 常常跑不满
    ⇒ 日志行是「已完成被试」的可靠来源。
    """
    if not path.exists():
        return {}
    cur = None
    per = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = RE_SUBJ.match(line.strip())
        if m:
            cur = m.group(1)
            per.setdefault(cur, {})
            continue
        m = RE_LEVEL.search(line)
        if m and cur:
            per[cur][m.group("tag")] = {"acc1_cos": float(m.group("acc")),
                                       "null_acc1_cos": float(m.group("nul"))}
    return per


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per = {}
    for seg_dir, rel in SEGMENTS:
        d = fetch(seg_dir, rel)
        if not d:
            print(f"（缺 {seg_dir}）")
            continue
        for r in d.get("per_subject", []):
            if "error" not in r:
                per[r["subject"]] = r.get("level") or {}
    # ---- 日志：补上 JSON 还没有的被试（只补**该被试已出现的标签**）----
    lg = parse_log(LOGF)
    added = 0
    for s, tags in lg.items():
        if s not in per:
            per[s] = tags
            added += 1
        else:
            for t, v in tags.items():
                per[s].setdefault(t, v)
    if added:
        print(f"日志补了 {added} 个被试")
    if not per:
        print("✗ 没有可用数据")
        return
    print(f"被试 {len(per)} 个：{sorted(per)}")

    tags = []
    for r in per.values():
        for t in (r or {}):
            if t not in tags:
                tags.append(t)
    rows = []
    for t in tags:
        vals = [(r or {}).get(t, {}).get("acc1_cos") for r in per.values()]
        vals = [v for v in vals if v is not None]
        nulls = [(r or {}).get(t, {}).get("null_acc1_cos") for r in per.values()]
        nulls = [v for v in nulls if v is not None]
        rows.append({"tag": t, "n": len(vals), "median_up": med(vals),
                     "mean": mean(vals), "null_median": med(nulls),
                     "ratio_to_chance": med(vals) / CHANCE if vals else None})
    rows.sort(key=lambda r: -(r["median_up"] or 0))

    print(f"\n{'读出':<22} {'n':>3} {'中位(上)':>9} {'均值':>9} {'×随机':>7} {'零中位':>8}")
    for r in rows:
        print(f"{r['tag']:<22} {r['n']:>3} {r['median_up']:>9.4f} {r['mean']:>9.4f} "
              f"{r['ratio_to_chance']:>6.1f}× {r['null_median']:>8.4f}")
    hand = HANDCRAFT["raw630"]
    print(f"\n手工 630-d 基线（同协议 exp93a，n=10）：{hand:.4f}"
          f"（{hand/CHANCE:.1f}× 随机）")

    (OUT / "exp93d_level.summary.json").write_text(
        json.dumps({"n_subjects": len(per), "rows": rows,
                    "handcraft_baseline": hand}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    L = ["# exp93d 模型层读出 @ n_train=16540（A18 的 n_train 检验）", "",
         f"被试 {len(per)} 个；**中位 = 上中位**；随机水平 1/200 = {CHANCE:.4f}", "",
         "| 读出 | n | 中位(上) | 均值 | ×随机 | 零中位 |",
         "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['tag']} | {r['n']} | {r['median_up']:.4f} | {r['mean']:.4f} "
                 f"| {r['ratio_to_chance']:.1f}× | {r['null_median']:.4f} |")
    L += ["", f"**手工 630-d 基线（同协议，exp93a，n=10）：{hand:.4f}"
          f"（{hand/CHANCE:.1f}× 随机）**", ""]
    (OUT / "exp93d_level.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n写入 {OUT/'exp93d_level.md'}")


if __name__ == "__main__":
    main()
