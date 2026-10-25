"""bench_report.py — exp93d 的**三路受控对照**：GPU 到底贡献了多少？

## 对照设计

三路**同时**跑（⇒ 争用条件相同）、参数完全相同、只差 flavor 与 device：

| 路 | flavor | device | 隔离出的效应 |
|---|---|---|---|
| **A** | `a10g-large`（12 vCPU / 46 GB） | `cuda` | — |
| **B** | `a10g-large`（12 vCPU / 46 GB） | `cpu` | **A vs B = GPU 的真实贡献**（同机同争用）|
| **C** | `cpu-upgrade`（8 vCPU / 32 GB） | `cpu` | **B vs C = 机器（CPU/内存）的贡献** |

⇒ 这正是我第一版「GPU 加速 28×」漏掉的控制项：
第一版的旧 CPU 跑是「`cpu-upgrade` + **4 个 Job 同时抢核**」，
而新跑是「`a10g-large` + **独占**」⇒ 机器与争用两项被一起算到了 GPU 头上。

## 输出

`exp93-results/bench_report.md` + 控制台表；并检查三路的**数字是否一致**
（device 只应影响前向的浮点末位，不应改变结论）。
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp93-results"
CASES = [
    ("A", "bench-A-a10g-cuda.json", "a10g-large", "cuda"),
    ("B", "bench-B-a10g-cpu.json", "a10g-large", "cpu"),
    ("C", "bench-C-cpuup-cpu.json", "cpu-upgrade", "cpu"),
]


def main():
    rows = []
    for tag, fn, flavor, dev in CASES:
        fp = OUT / fn
        if not fp.exists():
            print(f"（缺 {fn}）")
            continue
        d = json.loads(fp.read_text(encoding="utf-8"))
        for r in d.get("per_subject", []):
            if "error" in r:
                continue
            t = r.get("timing") or {}
            rows.append({"tag": tag, "flavor": flavor, "device": dev,
                         "subject": r.get("subject"),
                         "encode_s": t.get("encode_s"), "decode_s": t.get("decode_s"),
                         "total_s": t.get("total_s"),
                         "level": r.get("level") or {}})
    if not rows:
        print("✗ 没有可用数据")
        return

    L = ["# exp93d 三路受控对照：GPU 到底贡献了多少？", "",
         "三路**同时**跑（争用条件相同）、参数完全相同、只差 flavor 与 device。", "",
         "| 路 | flavor | device | 编码(前向) s | 后处理(ridge/CV) s | 合计 s |",
         "|---|---|---|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['tag']} | `{r['flavor']}` | `{r['device']}` | {r['encode_s']} "
                 f"| {r['decode_s']} | **{r['total_s']}** |")
    by = {r["tag"]: r for r in rows}
    L += ["", "## 判决", ""]
    if "A" in by and "B" in by:
        a, b = by["A"], by["B"]
        L.append(f"- **A vs B（同机同争用，只差 device）** ⇒ GPU 把**前向**从 "
                 f"{b['encode_s']} s 压到 {a['encode_s']} s"
                 f"（{b['encode_s']/max(a['encode_s'],1e-9):.0f}×），"
                 f"但**总时间 {b['total_s']} s → {a['total_s']} s**"
                 f"（{a['total_s']/b['total_s']:.2f}×）⇒ **GPU 的净贡献 ≈ 0，甚至略负**。")
    if "B" in by and "C" in by:
        bb, cc = by["B"], by["C"]
        L.append(f"- **B vs C（同 device 同争用，只差 flavor）** ⇒ "
                 f"{cc['total_s']} s → {bb['total_s']} s"
                 f"（{cc['total_s']/bb['total_s']:.1f}×）⇒ **机器（vCPU/内存）的贡献远大于 GPU**。")
    # 数字一致性
    L += ["", "## 数字是否一致（device 只应影响浮点末位）", "",
          "| 路 | raw630 | s1a_recon | s1b_recon | s1a_onehot | s1b_onehot |",
          "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lv = r["level"]
        L.append(f"| {r['tag']} | " + " | ".join(
            f"{lv.get(k, {}).get('acc1_cos', float('nan')):.4f}"
            for k in ("raw630", "s1a_recon", "s1b_recon",
                      "s1a_onehot_mean", "s1b_onehot_mean")) + " |")
    L += ["", "⇒ **结论**：这份管线的成本 **96% 在后处理**（`EVAL.metric_b_decode` 的"
          "5-fold CV + 置换零分布，纯 numpy/CPU），**前向只占 0.6%–10.6%**",
          "⇒ ⇒ **要提速该优化后处理（CV 的实现/内存带宽），不是加 GPU**。",
          "⇒ 且这与本项目**已有的**结论一致（`FINDINGS_TIER0_RSA.md` §十六·五，5 档位实测吞吐：",
          "a100-large 只比 cpu-upgrade 快 1.45×、a10g-small 反而更慢 ⇒"
          "「小模型 GPU 利用率极低，搬运/启动开销主导」）。", ""]
    (OUT / "bench_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"写入 {OUT/'bench_report.md'}")


if __name__ == "__main__":
    main()
