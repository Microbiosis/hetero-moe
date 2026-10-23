"""parse_assembled.py — 汇总「组件级组装」的结果，并给 S1b 判据。

## 读什么

`results/assembled-results/`（由 fetch_assembled.py 从桶下载）：

| 文件 | 是什么 |
|---|---|
| `v1-ffn-<臂>.json` | v1 配方的评测（exp81 写的 tokenizer_eval.json） |
| `v2-ffn-<臂>.json` | v2 配方的评测 |
| `train-v{1,2}-ffn-<臂>.json` | 对应的训练曲线（train_log.json） |

**臂** = 同一套自研骨架（D=64）+ 嫁接的 signal-jepa 注意力（MHA 64，逐键同形直接搬），
只换**嫁接的 FFN 供体**：`none` / `mirepnet` / `biot` / `eegdino-small`。
另有一个 `none-nosem` = **只有配方不同（无语义监督）**的对照臂。

## 三个配方

| 配方 | 语义损失 | 码本初始化 | 用途 |
|---|---|---|---|
| v1 | 挂在池化 token（**无效**：NCE 卡在 ln64） | 随机 | 历史对照 |
| v2 | 挂在**重建信号** | 用第一批数据初始化 | 主力 |
| v2-none-nosem | 无（λ=0） | 同 v2 | 隔离「语义监督」这一个因子 |

## 关键判据（S1b 的核心）

| 读出 | 测什么 | 期望 |
|---|---|---|
| `recon` | 组装模型自身的脑-语义对齐 | ≥ S1a 的 +0.0372 |
| `emb_mean` | 码本向量池化 | 上升 |
| **`onehot_mean`** | **纯离散的 one-hot 池化** | **上升 ⇒ token 身份本身带语义**（诚实的判据） |
| `code_ids` | 码号本身（顺序任意） | 仍无意义 |
| `continuous` | 不量化的 encoder 输出 | 对照 |
| `random_codebook` | 同 encoder + 随机码本 | 对照 |

⇒ **只有 `onehot_mean` 变显著，才是「统一词表」成立**（否则语义只在连续路径上）。

## 统计口径（与阶段 D 变体表一致）

配对差用**逐被试配对 Δr** + **精确二项符号检验**（H0：正负各半），
而不是「两行中位数相减」——后者把「谁更高」和「是否稳定地更高」混为一谈。
"""
import json
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
AS = HERE / "assembled-results"
S1_DIR = HERE / "s1"

ARMS = [
    ("v1-none", "v1-ffn-none"),
    ("v1-mirepnet", "v1-ffn-mirepnet"),
    ("v1-biot", "v1-ffn-biot"),
    ("v1-eegdino-small", "v1-ffn-eegdino-small"),
    ("v2-none", "v2-ffn-none"),
    ("v2-mirepnet", "v2-ffn-mirepnet"),
    ("v2-biot", "v2-ffn-biot"),
    ("v2-eegdino-small", "v2-ffn-eegdino-small"),
    ("v2-none-nosem", "v2-ffn-none-nosem"),
]
READOUTS = ["recon", "emb_mean", "onehot_mean", "code_ids", "continuous",
            "random_codebook"]


def med(vals):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def sign_test_p(n_pos, n):
    """单侧精确二项检验：H0 正负各半 ⇒ P(X >= n_pos), X~Bin(n, 0.5)。"""
    if n == 0:
        return None
    tail = sum(comb(n, k) for k in range(n_pos, n + 1))
    return tail / (2 ** n)


def load(path):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if "per_subject" not in d:
        return None
    return d


def per_subject_r(d, readout):
    """{被试: r_img}（跳过报错的被试）。"""
    out = {}
    for rec in d.get("per_subject", []):
        rows = rec.get("rows") or {}
        if readout in rows and "r_img" in rows[readout]:
            out[rec.get("subject", "?")] = float(rows[readout]["r_img"])
    return out


def paired(a, b):
    """a 减 b 的逐被试配对差 ⇒ (中位, 正向数, n, 单侧 p)。"""
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    d = [a[k] - b[k] for k in keys]
    npos = sum(1 for x in d if x > 0)
    return {"median": med(d), "n_pos": npos, "n": len(d),
            "p": sign_test_p(npos, len(d))}


def fmt_p(p):
    return "—" if p is None else f"{p:.4f}"


def main():
    # --- S1a 基线（我们自研骨架、无嫁接、无语义监督）---
    s1 = load(S1_DIR / "k1024.json")
    arms = {}
    for label, fn in ARMS:
        d = load(AS / f"{fn}.json")
        if d:
            arms[label] = d

    print("=" * 108)
    print("组件级组装：结果表（指标 A = RSA，10 被试中位；括号里是显著被试数）")
    print("=" * 108)
    head = f"{'臂':<16}" + "".join(f"{r[:12]:>15}" for r in READOUTS)
    print(head)
    print("-" * len(head))
    if s1:
        cells = []
        for r in READOUTS:
            e = s1["summary"].get(r)
            cells.append(f"{e['r_median']:>+8.4f}({e['n_sig']:>2})" if e else f"{'—':>15}")
        print(f"{'S1a（无嫁接）':<14}" + "".join(cells))
    for label, _ in ARMS:
        d = arms.get(label)
        if not d:
            print(f"{label:<16}（缺 {AS / (dict(ARMS)[label] + '.json')}）")
            continue
        cells = []
        for r in READOUTS:
            e = d["summary"].get(r)
            cells.append(f"{e['r_median']:>+8.4f}({e['n_sig']:>2})" if e else f"{'—':>15}")
        print(f"{label:<16}" + "".join(cells))

    # --- 码本利用率 ---
    print("\n码本利用率（评测时全 test 集）：")
    for label, d in arms.items():
        cbs = [r.get("codebook") for r in d.get("per_subject", []) if r.get("codebook")]
        if not cbs:
            continue
        u = sum(c["utilization"] for c in cbs) / len(cbs)
        e = sum(c["entropy_nats"] for c in cbs) / len(cbs)
        used = sum(c["used"] for c in cbs) / len(cbs)
        print(f"  {label:<16} {used:>5.0f}/{cbs[0]['total']:<5} = {u:>6.1%} | 熵 {e:.2f} nats")

    # --- 训练曲线：语义分支到底学没学 ---
    print("\n训练曲线终点（语义分支是否真的在学）：")
    print(f"  {'配方-臂':<26}{'nce_r':>9}{'nce_z':>9}{'nce_q':>9}"
          f"{'检索(重建)':>12}{'检索(码本)':>12}")
    for label, fn in ARMS:
        p = AS / f"train-{fn}.json"
        if not p.exists():
            continue
        try:
            h = json.loads(p.read_text(encoding="utf-8"))["hist"]
        except Exception:
            continue
        if not h:
            continue
        last = h[-1]
        def g(k, fmt="{:.3f}"):
            v = last.get(k)
            return "—" if v is None else fmt.format(v)
        print(f"  {label:<26}{g('nce_r'):>9}{g('nce_z'):>9}{g('nce_q'):>9}"
              f"{g('retrieval_acc'):>12}{g('retrieval_acc_q'):>12}")
    print("  （随机水平：NCE = ln64 = 4.159；检索准确率 = 1/64 = 0.016）")

    # --- 配对差：语义监督的效应（v2 vs 无语义监督对照臂）---
    ctrl = arms.get("v2-none-nosem")
    print("\n配对差（逐被试配对 Δr，精确二项符号检验）：")
    if ctrl:
        print(f"  {'臂':<16}{'读出':<16}{'Δr vs 无语义监督':>18}{'正向':>8}{'p':>9}")
        for label, d in arms.items():
            if label == "v2-none-nosem" or not label.startswith("v2-"):
                continue
            for r in ("recon", "onehot_mean", "emb_mean"):
                st = paired(per_subject_r(d, r), per_subject_r(ctrl, r))
                if not st:
                    continue
                print(f"  {label:<16}{r:<16}{st['median']:>+18.4f}"
                      f"{st['n_pos']:>4}/{st['n']:<3}{fmt_p(st['p']):>9}")
    else:
        print("  （缺 v2-none-nosem 对照臂）")

    # --- 配对差：嫁接 vs 自研骨架（S1a）---
    if s1:
        print("\n配对差（vs S1a 自研骨架，同样逐被试配对）：")
        print(f"  {'臂':<16}{'读出':<16}{'Δr vs S1a':>14}{'正向':>8}{'p':>9}")
        for label, d in arms.items():
            if not label.startswith("v2-"):
                continue
            for r in ("recon", "onehot_mean"):
                st = paired(per_subject_r(d, r), per_subject_r(s1, r))
                if not st:
                    continue
                print(f"  {label:<16}{r:<16}{st['median']:>+14.4f}"
                      f"{st['n_pos']:>4}/{st['n']:<3}{fmt_p(st['p']):>9}")

    # --- S1b 判据 ---
    print("\n" + "=" * 108)
    print("S1b 判据：离散层（onehot_mean）是否变显著")
    print("=" * 108)
    for label, d in arms.items():
        e = d["summary"].get("onehot_mean")
        if not e:
            continue
        verdict = "✅ 离散 token 带语义" if e["n_sig"] >= 5 else "❌ 语义只在连续路径"
        print(f"  {label:<16} onehot_mean 中位 {e['r_median']:+.4f}，"
              f"显著 {e['n_sig']}/{e['n']} ⇒ {verdict}")
    if s1:
        e = s1["summary"].get("onehot_mean")
        if e:
            print(f"  {'S1a（基线）':<16} onehot_mean 中位 {e['r_median']:+.4f}，"
                  f"显著 {e['n_sig']}/{e['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
