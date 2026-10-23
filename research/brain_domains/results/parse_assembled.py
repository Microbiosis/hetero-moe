"""parse_assembled.py — 汇总「组件级组装」的四个变体，并与 S1a 对照。

## 读什么

四个变体（同一个自研骨架 D=64 + 嫁接的 signal-jepa 注意力 + 语义监督），
只换**嫁接的 FFN 供体**：
- `ffn-none`        不嫁接 FFN（只测注意力）
- `ffn-mirepnet`    mirepnet 的 FFN（256→1024→256）—— exp75 里层内语义最强
- `ffn-biot`        biot 的 FFN（256→1024→256）
- `ffn-eegdino`     eegdino 的 mlp（200→512→200）

## 关键判据（S1b 的核心）

| 读出 | 测什么 | 期望 |
|---|---|---|
| `recon` | 组装模型自身的脑-语义对齐 | ≥ S1a 的 +0.0372 |
| `emb_mean` | **码本向量**池化（被语义损失直接优化 ⇒ 应当上升） | 上升 |
| **`onehot_mean`** | **纯离散的 one-hot 池化** | **上升 ⇒ token 身份本身带语义**（诚实的判据） |
| `code_ids` | 码号本身（顺序任意 ⇒ 预期无意义） | 仍无意义 |
| `continuous` | 不量化的 encoder 输出 | 对照 |
| `random_codebook` | 同 encoder + 随机码本 | 对照 |

⇒ **只有 `onehot_mean` 变显著，才是「统一词表」成立**（否则语义只在连续路径上）。
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSEMBLED = HERE / "assembled-results"
S1 = HERE / "s1"

VARIANTS = ["ffn-none", "ffn-mirepnet", "ffn-biot", "ffn-eegdino-small"]
READOUTS = ["recon", "emb_mean", "onehot_mean", "code_ids", "continuous",
            "random_codebook"]
S1A = {  # S1a（K=1024，10 被试中位）
    "recon": (0.0372, 10), "emb_mean": (0.0045, 0), "onehot_mean": (0.0014, 0),
    "code_ids": (0.0002, 0), "continuous": (0.0032, 1), "random_codebook": (0.0021, 1),
}


def load(path):
    p = Path(path)
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return d.get("summary", {})


def main():
    print(f"{'变体':<18}" + "".join(f"{r[:12]:>16}" for r in READOUTS))
    print("-" * (18 + 16 * len(READOUTS)))
    print(f"{'S1a（基线）':<16}" + "".join(
        f"{S1A[r][0]:>+9.4f}({S1A[r][1]:>2})" for r in READOUTS))
    rows = {}
    for v in VARIANTS:
        s = load(ASSEMBLED / f"{v}.json")
        if not s:
            print(f"{v:<18}（缺 {ASSEMBLED / (v + '.json')}）")
            continue
        rows[v] = s
        cells = []
        for r in READOUTS:
            if r in s:
                cells.append(f"{s[r]['r_median']:>+9.4f}({s[r]['n_sig']:>2})")
            else:
                cells.append(f"{'—':>16}")
        print(f"{v:<18}" + "".join(cells))
    print("\n（格式：中位 A r（显著被试数）；S1a 行是同一读出下的基线）")

    # 码本利用率
    print("\n码本利用率（评测时全 test 集）：")
    for v in VARIANTS:
        p = ASSEMBLED / f"{v}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        cbs = [r.get("codebook") for r in d.get("per_subject", []) if r.get("codebook")]
        if cbs:
            u = sum(c["utilization"] for c in cbs) / len(cbs)
            e = sum(c["entropy_nats"] for c in cbs) / len(cbs)
            print(f"  {v:<18} 利用率 {u:.1%} | 熵 {e:.2f} nats")

    # 结论
    print("\n=== S1b 判据：离散层（onehot_mean）是否变显著 ===")
    for v, s in rows.items():
        if "onehot_mean" in s:
            m = s["onehot_mean"]
            print(f"  {v:<18} onehot_mean 中位 {m['r_median']:+.4f}，"
                  f"显著 {m['n_sig']}/{m['n']}"
                  f" ⇒ {'✅ 离散 token 带语义' if m['n_sig'] >= 5 else '❌ 语义只在连续路径'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
