"""V61 诊断：`centered` 臂的 0.188 奇偶差从哪来。

## 观察

11 处 R4 违规全部在 S=16，值恒为 0.188，三个不同架构的模型给出同一值到
小数点后四位。⇒ 由构造决定，与模型内容无关。

## 三个待区分的解释

    H1 cos 的非线性放大: Σx'=0 使非对角点积负偏置，cos 在接近 −1 时非线性
    H2 余弦本身的奇偶敏感性: 换成原始点积(不归一)后奇偶差应改变量级
    H3 与 S 的奇偶/规模有关: S=16 有、S=32 无

本脚本对比**三种观测量**在同一批数据上的奇偶差：

    cos       cos(x_i, x_j)                     （v61 用的）
    dot       原始点积 x_i·x_j                   （不归一）
    rankspear 点积在**每个样本内**的秩           （消除量纲，只看序）

若 cos 的 0.188 在 dot / rankspear 上消失 ⇒ H1/H2 成立（观测量问题）；
若三者都有 ⇒ H3（结构问题）。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from research.aligner.v60_token_locality.locality import cos_matrix

ART = os.environ.get("V61DIAG_ART_DIR", "v61diag_artifacts")
PREDICTIONS = {
    "H1/H2": "换成 dot 或 rankspear 后, 0.188 的奇偶差**消失或大幅改变** ⇒ 观测量问题",
    "H3": "三种观测量都给出 ≥0.15 的奇偶差 ⇒ 是结构本身的问题",
}


def parity_gap(M: torch.Tensor) -> float:
    S = M.shape[0]
    ev, od = [], []
    for d in range(2, S):
        vals = [float(M[i, i + d]) for i in range(S - d)]
        (ev if d % 2 == 0 else od).append(sum(vals) / len(vals))
    if not ev or not od:
        return 0.0
    return sum(ev) / len(ev) - sum(od) / len(od)


def dot_matrix(X: torch.Tensor) -> torch.Tensor:
    """原始点积（不归一）。"""
    n, S, D = X.shape
    return torch.einsum("nsd,ntd->st", X.float(), X.float()) / n


def rank_matrix(X: torch.Tensor) -> torch.Tensor:
    """逐样本：把 S×S 点积矩阵的元素秩化，再跨样本平均。消除量纲。"""
    n, S, D = X.shape
    Xf = X.float()
    acc = torch.zeros(S, S)
    for s in range(n):
        G = Xf[s] @ Xf[s].T
        r = G.flatten().argsort().argsort().float().reshape(S, S)
        acc += r
    return acc / n


def main():
    from research._primitives.scale_hetero_loaders import (
        EMB_DIR, load_small, load_tokenizer)
    from research.aligner.v61_delta_replication import TEXT_TMPL
    
    print("=" * 80)
    print("V61 诊断: centered 臂 0.188 奇偶差的来源")
    print("=" * 80)
    for k, v in PREDICTIONS.items():
        print(f"  {k}: {v}")
    print()

    N = 192
    model, D, _ = load_small("emb", EMB_DIR)
    tok = load_tokenizer(EMB_DIR)
    texts = [TEXT_TMPL.format(i=i, j=i * 7 % 97) for i in range(N)]

    rows = []
    for S in (16, 32):
        ids = tok(texts, padding="max_length", truncation=True, max_length=S,
                  return_tensors="pt")
        with torch.no_grad():
            out = model(input_ids=ids.input_ids, output_hidden_states=True)
        print(f"\n{'='*80}\nS={S}\n{'='*80}")
        for li in (3, 4, 5, 6, 7):
            X = out.hidden_states[li].float()
            Xc = X - X.mean(dim=1, keepdim=True)
            g_cos = parity_gap(cos_matrix(Xc))
            g_dot = parity_gap(dot_matrix(Xc))
            g_rnk = parity_gap(rank_matrix(Xc))
            rows.append({"S": S, "layer": li, "cos": round(g_cos, 5),
                         "dot": round(g_dot, 5), "rank": round(g_rnk, 5)})
            print(f"  layer{li}  cos={g_cos:+.5f}   dot={g_dot:+.5f}   rank={g_rnk:+.5f}")

    print("\n" + "=" * 80)
    print("判读")
    print("=" * 80)
    s16 = [r for r in rows if r["S"] == 16]
    print(f"  S=16 上 cos 的 max|奇偶差| = {max(abs(r['cos']) for r in s16):.5f}")
    print(f"  S=16 上 dot 的 max|奇偶差| = {max(abs(r['dot']) for r in s16):.5f}")
    print(f"  S=16 上 rank 的 max|奇偶差| = {max(abs(r['rank']) for r in s16):.5f}")
    h12 = max(abs(r["dot"]) for r in s16) < 0.15 or max(abs(r["rank"]) for r in s16) < 0.15
    print(f"\n  ⇒ {'H1/H2 成立（观测量问题）' if h12 else 'H3 可疑（三种观测量都有）'}")

    os.makedirs(ART, exist_ok=True)
    json.dump({"predictions": PREDICTIONS, "rows": rows,
               "verdict_H12": bool(h12)}, open(f"{ART}/diag.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"\n落盘 {ART}/diag.json")


if __name__ == "__main__":
    main()
