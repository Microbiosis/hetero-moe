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
    ("v2-plain", "v2-ffn-plain"),
    ("v2-plain-nosem", "v2-ffn-plain-nosem"),
    # --- S1b-v3「共享词典」三臂（码本机制 × CE 的码本路径）---
    ("v3-grad-pos", "v3-grad-pos"),
    ("v3-grad-full", "v3-grad-full"),
    ("v3-grad-none", "v3-grad-none"),
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
    # ⚠️ **v1 与 v2 的 `retrieval_acc` 记的不是同一条分支**：
    #    v1 只有两个 NCE（z / 码本），`retrieval_acc` 记的是**码本分支**；
    #    v2 有三个 NCE，`retrieval_acc`=**重建分支**、`retrieval_acc_q`=码本分支。
    #    ⇒ 按各配方实际的键位映射，否则 v1 行会被标错。
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
        is_v2 = "nce_r" in last
        recon_acc = last.get("retrieval_acc") if is_v2 else None
        code_acc = last.get("retrieval_acc_q") if is_v2 else last.get("retrieval_acc")

        def g(k):
            v = last.get(k)
            return "—" if v is None else f"{v:.3f}"

        def a(v):
            return "—" if v is None else f"{v:.3f}"

        print(f"  {label:<26}{g('nce_r'):>9}{g('nce_z'):>9}{g('nce_q'):>9}"
              f"{a(recon_acc):>12}{a(code_acc):>12}")
    print("  （随机水平：NCE = ln64 = 4.159；检索准确率 = 1/64 = 0.016）")
    print("  （v1 的「检索准确率」记的是**码本分支**；v2 才是重建/码本两条）")

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

    # --- 配对差：臂 vs 臂（零件资格赛口径：嫁接某组件到底有没有比不嫁接更好）---
    base = arms.get("v2-none")
    if base:
        print("\n配对差（vs v2-none，即「只嫁接注意力、不嫁接 FFN」的基准臂）：")
        print(f"  {'臂':<20}{'读出':<16}{'Δr':>10}{'正向':>8}{'p':>9}")
        for label, d in arms.items():
            if label in ("v2-none", "v2-none-nosem"):
                continue
            for r in ("recon", "onehot_mean", "emb_mean"):
                st = paired(per_subject_r(d, r), per_subject_r(base, r))
                if not st:
                    continue
                print(f"  {label:<20}{r:<16}{st['median']:>+10.4f}"
                      f"{st['n_pos']:>4}/{st['n']:<3}{fmt_p(st['p']):>9}")

    # --- 2×2 因子表：注意力嫁接 × 语义监督 ---
    cells = {
        ("不嫁接", "无"): "v2-plain-nosem",
        ("不嫁接", "有"): "v2-plain",
        ("嫁接", "无"): "v2-none-nosem",
        ("嫁接", "有"): "v2-none",
    }
    if all(a in arms for a in cells.values()):
        print("\n2×2 因子表（同一套配方，只有两个因子变）：")
        for ro in ("recon", "onehot_mean", "emb_mean", "continuous",
                   "random_codebook"):
            print(f"  {ro}")
            for attn in ("不嫁接", "嫁接"):
                row = []
                for sem in ("无", "有"):
                    d = arms[cells[(attn, sem)]]
                    e = d["summary"].get(ro)
                    row.append(f"{e['r_median']:+.4f}({e['n_sig']}/{e['n']})"
                               if e else "—")
                print(f"    注意力{attn:<4} 语义监督{('有' if sem=='有' else '无')}: "
                      f"{row[0]:>16} | {row[1]:>16}")
        print("  单因子效应（逐被试配对 Δr，p=0.0010 即 10/10 全正）：")
        for ro in ("recon", "onehot_mean", "emb_mean"):
            for (a1, a2), lab in (
                    (("v2-none", "v2-plain"), "嫁接注意力的净效应"),
                    (("v2-plain", "v2-plain-nosem"), "语义监督的净效应（不嫁接注意力）"),
                    (("v2-none", "v2-none-nosem"), "语义监督的净效应（嫁接注意力）")):
                st = paired(per_subject_r(arms[a1], ro),
                            per_subject_r(arms[a2], ro))
                if not st:
                    continue
                print(f"    {lab:<28}{ro:<14}{st['median']:>+10.4f}"
                      f"{st['n_pos']:>4}/{st['n']:<3}{fmt_p(st['p']):>9}")

    # --- 决定性对照：学出来的码本 vs 随机码本 ---
    # `emb_mean`（学出来的码本，64 维）与 `random_codebook`（**随机**码本，64 维）
    # 用**同一个 encoder**、同一套池化 ⇒ 唯一差别就是「码本是不是学出来的」。
    # ⇒ 这一条决定「语义在码本里」还是「语义只在连续表征里、随便量化都能带过去」。
    print("\n决定性对照：学出来的码本 vs **随机**码本（同 encoder / 同维度 / 同池化）")
    print(f"  {'臂':<20}{'emb_mean':>12}{'random_cb':>12}{'Δr(学-随机)':>14}"
          f"{'正向':>8}{'p':>9}")
    for label, d in arms.items():
        if not (label.startswith("v2-") or label.startswith("v3-")):
            continue
        a = per_subject_r(d, "emb_mean")
        b = per_subject_r(d, "random_codebook")
        st = paired(a, b)
        e = d["summary"].get("emb_mean")
        rc = d["summary"].get("random_codebook")
        if not st or not e or not rc:
            continue
        print(f"  {label:<20}{e['r_median']:>+12.4f}{rc['r_median']:>+12.4f}"
              f"{st['median']:>+14.4f}{st['n_pos']:>4}/{st['n']:<3}"
              f"{fmt_p(st['p']):>9}")

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
