"""exp98_posthoc_ladder.py — exp98 的**事后观察**（描述性，**不参与判决**）。

## 为什么有这个脚本

exp98 的**预登记判决已经出**：`C1_FAIL`（5/18 正，p=0.0963，中位 Δ=−2.0845）⇒ **判决就到此为止**。
本脚本只回答两个**事后**问题，供下一轮设计参考（**任何数字都不得回头改判据**）：

1. **阶梯**：各带对的 Δ（双向平均，与主判据同口径）随「对比跨度」怎么变？
2. **如果主判据当初选的是别的正对照会怎样**（**纯描述**：换正对照后的配对差与符号检验）。

⚠️ **这不是「换个对照再来一次」** —— 那是篡改。本脚本的输出**只写进 §三十九 的「事后观察」段**，
且必须带上「**不参与判决**」的标注。

## 用法

```powershell
python research/brain_domains/results/exp98_posthoc_ladder.py
```
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent / "exp"
for _p in (str(EXP), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RAW = HERE / "exp98-results" / "exp98_posneg_control.json"


def median_upper(xs):
    s = sorted(float(x) for x in xs if x == x)
    return s[len(s) // 2] if s else float("nan")


def sign_test(diffs):
    d = [float(x) for x in diffs if x == x]
    n = len(d)
    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    zero = n - pos - neg
    from math import comb
    k = min(pos, neg)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n)) if n else 1.0
    return n, pos, neg, p, zero


def bidir(rec, a, b):
    """双向平均 Δ（与主判据同口径）。"""
    ps = rec.get("pairs") or {}
    vals = []
    for key in (f"{a}->{b}", f"{b}->{a}"):
        d = (ps.get(key) or {}).get("delta")
        if d is not None and d == d:
            vals.append(d)
    return float(sum(vals) / len(vals)) if vals else None


def main():
    data = json.loads(RAW.read_text(encoding="utf-8"))
    per = data.get("per_session") or []
    print(f"session 记录 {len(per)} 条\n")

    # ① 阶梯（双向平均；主判据的三区齐备才进）
    partners = ("VISl", "VISam", "VISpm", "VISrl", "VISal", "CA1")
    rows = []
    for p in partners:
        vals = [(r.get("session"), bidir(r, "VISp", p)) for r in per]
        vals = [(s, v) for s, v in vals if v is not None]
        if not vals:
            continue
        vs = [v for _, v in vals]
        n, pos, neg, pval, zero = sign_test(vs)
        rows.append((f"VISp↔{p}", len(vs), pos, f"{median_upper(vs):+.4f}", f"{pval:.4f}", zero))
    print("== ① 阶梯：VISp 与各区的 Δ（双向平均）==")
    print("| 带对 | n | n_pos | 中位Δ | 符号检验 p | 零差 |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")

    # ② 事后：换正对照会怎样（**纯描述**）
    print("\n== ② 事后观察（不参与判决）：若正对照换成别的对，配对差 = Δ(VISp↔X) − Δ(VISp↔VISl) ==")
    print("| 假设的正对照 | n | n_pos | 配对差中位 | 符号检验 p |")
    print("|---|---|---|---|---|")
    out = {}
    for p in partners:
        if p == "VISl":
            continue
        diffs = []
        for r in per:
            a = bidir(r, "VISp", p)
            b = bidir(r, "VISp", "VISl")
            if a is not None and b is not None:
                diffs.append(a - b)
        if not diffs:
            continue
        n, pos, neg, pval, zero = sign_test(diffs)
        med = median_upper(diffs)
        out[p] = {"n": n, "n_pos": pos, "median": med, "p": pval}
        print(f"| VISp↔{p} | {n} | {pos} | {med:+.4f} | {pval:.4f} |")

    (HERE / "exp98-results").mkdir(parents=True, exist_ok=True)
    (HERE / "exp98-results" / "exp98_posthoc_ladder.json").write_text(
        json.dumps({"ladder": rows, "alt_positive": out}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print("\n（以上全部是**事后描述**，**不改 exp98 的判决**：`C1_FAIL`）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
