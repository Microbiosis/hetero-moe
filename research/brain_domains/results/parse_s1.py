"""parse_s1.py — 汇总 S1（自训 EEG 离散 tokenizer）的评测结果。

输入：`results/s1/{k1024,k256}.json`（由 exp81 写出，从桶下载）
输出：逐读出的中位 / 显著被试数 / 码本利用率，并打印**与基线表的对照**。
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
S1_DIR = HERE / "s1"

# 基线表（§十六·六）与本轮的对照数字（指标 A，10 被试中位）
BASELINE = {
    "630-d 原始特征": 0.0504,
    "cbramod（宿主）": 0.0105,
    "braintokenizer（重建输出）": 0.0155,
    "signal-jepa": 0.0260,
}


def main():
    rows = []
    for k in ("k1024", "k256"):
        fp = S1_DIR / f"{k}.json"
        if not fp.exists():
            print(f"（缺 {fp}）")
            continue
        d = json.loads(fp.read_text(encoding="utf-8"))
        print("=" * 92)
        print(f"=== 码本 {k} | 被试 {len(d['per_subject'])} | "
              f"checkpoint {d.get('ckpt','')}")
        for name, s in d["summary"].items():
            print(f"  {name:<18} 中位 A r={s['r_median']:+.4f}  "
                  f"p={s['p_median']:.3f}  显著 {s['n_sig']}/{s['n']}")
            rows.append((k, name, s))
        cb = [r.get("codebook") for r in d["per_subject"] if r.get("codebook")]
        if cb:
            u = sum(c["utilization"] for c in cb) / len(cb)
            e = sum(c["entropy_nats"] for c in cb) / len(cb)
            used = sum(c["used"] for c in cb) / len(cb)
            tot = cb[0]["total"]
            print(f"  码本利用率（全 test 集）：{used:.0f}/{tot} = {u:.1%} | "
                  f"熵 {e:.2f} nats")

    print("=" * 92)
    print("对照：基线表（§十六·六）的指标 A，10 被试中位")
    for name, v in BASELINE.items():
        print(f"  {name:<26} {v:+.4f}")
    print("  本轮 S1a 重建层（K=1024）见上表")


if __name__ == "__main__":
    main()
