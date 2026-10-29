#!/usr/bin/env python3
r"""exp106_ops_diag.py — **算子级诊断**（零 GPU 成本）：exp102 的 adapter 到底塌成了什么？

## 为什么做这个（用户问的「算子呢？」）

exp102 的机制证据只有一个数：**条件 token 跨概念一致性 0.9905** ⇒ adapter 收敛成
「与概念无关的常数方向」。那到底：

1. adapter 末层权重矩阵的**秩**是多少？（是不是真的只剩 1 维？）
2. token 的**概念间散布** vs **概念内散布**，差多少倍？
3. 条件向量 z 本身的**概念信息**（输入侧）与 token 的**概念信息**（输出侧），
   在哪一步丢掉的？（MLP 第一层？第二层？）

## 这一步量什么（全部在 CPU 上，从桶里读，不往本机拉）

对 exp102 的 `adapter_final.pt` + 一个被试的条件缓存：

| 量 | 怎么算 | 读法 |
|---|---|---|
| 末层权重矩阵的秩 | `net.3.weight` 的奇异值谱 | 有效秩小 ⇒ 输出空间塌成低维 |
| token 跨概念一致性 | ‖跨概念均值‖ ÷ 平均范数（逐 token） | 复现 exp102 的 0.9905 |
| token 概念间散布 | token 在跨概念均值周围的 RMS 半径 | 与「概念内散布」比 ⇒ 越大越好 |
| **概念可分性（关键）** | 在 token 空间里，用最近邻/线性探针判「这 200 个 token 属哪个概念」 | 直接测「token 还带不带概念信息」 |
| **候选算子预演** | 见下 | 数值上判断哪个算子值得上 GPU |

## 候选算子（离线预演，零训练）

| 算子 | 做法 | 预演指标 |
|---|---|---|
| **A0 基线** | 现状：token = adapter(z) | 概念可分性 |
| **A1 去公共分量** | token ← token − 跨概念均值（逐 token 维） | 概念可分性 + 一致性 |
| **A2 去公共分量 + 缩放** | A1 后再按跨概念散布归一（让概念分量占满幅度） | 同上 + **幅度比**（token 范数 vs 文本 token 范数） |
| **A3 概念子空间投影** | 在 token 空间里投影掉「跨概念均值」与「跨概念主方向」 | 概念可分性 + 幅度比 |

⚠️ 关键点：**幅度比**（exp102 测到 token 范数中位 76.41 vs 文本 token ≈ 10⁻¹ 量级 ⇒
条件在 cross-attn 里相对文本是噪声级）⇒ 如果 A2/A3 能把幅度比拉到 1 附近，
那它们才是值得上 GPU 的算子（而不只是"可分性好看"）。

## 用法

```bash
python -u /domains/scripts/exp106_ops_diag.py --out-json /domains/runs/.../ops_diag.json
```
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

EXP102 = "/domains/runs/illusionbrain-align/exp102"
ADAPTER = f"{EXP102}/adapter_final.pt"
COND = f"{EXP102}/cache/eeg_cond_sub01.pt"
ENC_CKPT = "/domains/runs/illusionbrain-align/exp93b/encoder_best.pt"
TEST_4D = "/domains/omnibrain4d/eeg"
N_CONCEPTS = 200
IMAGES_PER_CONCEPT = 10
N_IMG = N_CONCEPTS * IMAGES_PER_CONCEPT


def log(*a):
    print(*a, flush=True)


def l2norm(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-12)


def effective_rank(s):
    """有效秩（参与率 0.9 的秩）。"""
    p = s / max(s.sum(), 1e-12)
    c = np.cumsum(p)
    return int(np.searchsorted(c, 0.9) + 1)


def concept_separability(tok, labels):
    """**概念可分性**：token 空间里 1-NN 判别同一概念的准确率（越高越好）。

    token: (n_img, K, D)；labels: (n_img,) 概念号。
    ⇒ 先把每个 token 投影到「去掉 token 维」后的 (K*D) 特征？还是逐 token 算？

    这里用**逐 token** 1-NN：对每个 token 找最近邻 token，看是不是同一概念
    ⇒ 输出「每概念 10 张图的 token 互相最近」的比例。
    """
    K, D = tok.shape[1], tok.shape[2]
    flat = tok.reshape(-1, D).astype(np.float64)          # (n_img*K, D)
    lab = np.repeat(labels, K)                           # 对应每个 token 的概念号
    fn = l2norm(flat)
    # 分块算相似度，避免 n^2 爆内存
    hits, tot = 0, 0
    step = 4096
    for i in range(0, len(fn), step):
        sim = fn[i:i + step] @ fn.T
        for r in range(sim.shape[0]):
            sim[r, i + r] = -2.0                        # 排除自己
        idx = sim.argmax(axis=1)
        hits += int((lab[idx] == lab[i:i + step]).sum())
        tot += sim.shape[0]
    return hits / max(tot, 1)


def concept_linear_probe(tok, labels, n_probe=2000):
    """概念可分性的第二种读法：逐 token 的**最近类中心**准确率（更稳）。"""
    K, D = tok.shape[1], tok.shape[2]
    flat = tok.reshape(-1, D).astype(np.float64)
    lab = np.repeat(labels, K)
    fn = l2norm(flat)
    centers = np.stack([fn[lab == c].mean(axis=0) for c in np.unique(lab)])
    centers = l2norm(centers)
    pred = np.argmax(fn @ centers.T, axis=1)
    return float((pred == lab).mean())


def consistency(tok):
    """exp102 的口径：‖跨概念均值‖ ÷ 平均范数（逐 token 维 ⇒ 再平均）。"""
    mu = tok.mean(axis=0)
    per_mean = np.linalg.norm(mu, axis=-1)
    per_avg = np.linalg.norm(tok, axis=-1).mean(axis=0)
    return float((per_mean / np.maximum(per_avg, 1e-12)).mean())


def scatter_ratio(tok):
    """**概念内散布 / 概念间散布**（越大 ⇒ 概念结构越清楚）。

    ⚠ 2026-10-28 的诊断结果说明这个量要拆成两个绝对量来看（比例会掩盖"两头都≈0"）：
      - `within`：**同一概念**的 10 张图之间的散布
      - `between`：**200 个概念簇心**之间的散布
      - 比值 alone 不可信（两头都接近 0 时比值无意义）⇒ ⇒ 三个数一起报。
    """
    labels = np.repeat(np.arange(N_CONCEPTS), IMAGES_PER_CONCEPT)
    flat = tok.reshape(-1, tok.shape[2]).astype(np.float64)
    lab = np.repeat(labels, tok.shape[1])
    centers = np.stack([flat[lab == c].mean(axis=0) for c in np.unique(lab)])
    within = 0.0
    for c in np.unique(lab):
        within += ((flat[lab == c] - flat[lab == c].mean(axis=0)) ** 2).sum()
    within /= len(flat)
    gm = centers.mean(axis=0)
    between = ((centers - gm) ** 2).sum() / len(centers)
    return {"within_rms": float(np.sqrt(within)),
            "between_rms": float(np.sqrt(between)),
            "ratio": float(np.sqrt(between / max(within, 1e-12)))}


def center_1nn(tok):
    """**簇心之间的 1-NN**：簇心最像的那个簇是不是同一概念（=200 簇是否彼此可分）。"""
    labels = np.repeat(np.arange(N_CONCEPTS), IMAGES_PER_CONCEPT)
    flat = tok.reshape(-1, tok.shape[2]).astype(np.float64)
    lab = np.repeat(labels, tok.shape[1])
    centers = np.stack([flat[lab == c].mean(axis=0) for c in np.unique(lab)])
    cn = l2norm(centers)
    sim = cn @ cn.T
    np.fill_diagonal(sim, -2.0)
    return float((np.argmax(sim, axis=1) == np.arange(len(cn))).mean())


def apply_op(tok, op, text_norm_median):
    """候选算子 ⇒ 新的 token。"""
    if op == "A0":
        return tok

    def _scale(dev):
        """按「每个 token 维的散布」归一；再（若知道文本范数）缩到文本量级。"""
        spread = np.linalg.norm(dev, axis=-1).mean(axis=0)         # (K,)
        dev = dev / np.maximum(spread, 1e-12)[:, None]
        if text_norm_median:
            dev = dev * (text_norm_median / np.sqrt(dev.shape[-1]))[..., None] * np.sqrt(dev.shape[-1])
        return dev

    if op == "A1":                      # 去公共分量（不缩放）
        return tok - tok.mean(axis=0, keepdims=True)
    if op == "A2":                      # 去公共分量 + 缩放到文本幅度
        return _scale(tok - tok.mean(axis=0, keepdims=True))
    if op == "A3":                      # 去公共分量 + 去掉跨概念主方向
        mu = tok.mean(axis=0, keepdims=True)
        dev = (tok - mu).reshape(-1, tok.shape[-1]).astype(np.float64)
        centers = np.stack([dev.reshape(N_IMG, tok.shape[1], -1)[
            np.repeat(np.arange(N_CONCEPTS), IMAGES_PER_CONCEPT) == c].reshape(-1, tok.shape[-1]).mean(axis=0)
            for c in np.arange(N_CONCEPTS)])
        u, s, vt = np.linalg.svd(centers - centers.mean(axis=0), full_matrices=False)
        v1 = vt[0]
        dev2 = dev - np.outer(dev @ v1, v1)
        return _scale(dev2.reshape(tok.shape))
    raise ValueError(op)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--cond", default=COND)
    ap.add_argument("--out-json", default="/domains/runs/illusionbrain-align/exp106/ops_diag.json")
    args = ap.parse_args()

    import torch

    import exp102_train as T

    # ---- adapter ----
    ck = torch.load(args.adapter, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    adapter = T.build_adapter()
    adapter.load_state_dict(sd)
    adapter.eval()

    # ---- 末层权重矩阵的秩 ----
    W = sd["net.3.weight"].detach().numpy()          # (K*768, 768)
    s = np.linalg.svd(W, compute_uv=False)
    rec = {"w_shape": list(W.shape),
           "w_sv_max": float(s[0]), "w_sv_min": float(s[-1]),
           "w_eff_rank_90": effective_rank(s),
           "w_cond_number": float(s[0] / max(s[-1], 1e-12)),
           "w_max_abs": float(np.abs(W).max())}

    # ---- 条件（一个被试，取 rep 0 ⇒ 与 exp102 的诊断同口径）----
    cobj = torch.load(args.cond, map_location="cpu", weights_only=False)
    cond = cobj["cond"].numpy().astype(np.float32)   # (n_img, n_reps, 1024)
    z = cond[:N_IMG, 0]
    labels = np.repeat(np.arange(N_CONCEPTS), IMAGES_PER_CONCEPT)
    log(f"[条件] {z.shape}｜概念 {N_CONCEPTS} × {IMAGES_PER_CONCEPT}")

    with torch.no_grad():
        tok = adapter(torch.from_numpy(z)).numpy().astype(np.float64)   # (n_img, K, D)
    log(f"[token] {tok.shape}")

    # ---- 文本 token 的幅度（做「幅度比」的基准）----
    txt_norm_median = None
    try:
        import exp102_train as T
        T.ensure_deps()          # ⚠ Job 镜像里没有 transformers ⇒ 必须先装
        from transformers import CLIPTextModel, CLIPTokenizer
        repo = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        tokzr = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer")
        txt = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder").eval()
        with torch.no_grad():
            ids = tokzr([T.PROMPT], padding="max_length", max_length=77, truncation=True,
                        return_tensors="pt").input_ids
            temb = txt(ids)[0].numpy().astype(np.float64)
        txt_norm_median = float(np.median(np.linalg.norm(temb, axis=-1)))
        rec["text_token_norm_median"] = txt_norm_median
        log(f"[文本 token] 范数中位 {txt_norm_median:.4f}")
    except Exception as e:                                   # noqa: BLE001
        log(f"[文本 token] 取不到（{type(e).__name__}: {str(e)[:120]}）⇒ **幅度比不可信**")
        txt_norm_median = None

    # ---- 基线 ----
    def metrics(t):
        tn = float(np.median(np.linalg.norm(t, axis=-1)))
        return {"consistency": consistency(t),
                "token_norm_median": tn,
                "amplitude_ratio": (tn / txt_norm_median) if txt_norm_median else None,
                "concept_1nn_token": concept_separability(t, labels),
                "concept_center": concept_linear_probe(t, labels),
                "center_1nn": center_1nn(t),
                **scatter_ratio(t)}

    rec["baseline"] = metrics(tok)
    log("[基线] " + json.dumps(rec["baseline"], ensure_ascii=False))

    # ---- 候选算子预演 ----
    rec["ops"] = {}
    for op in ("A0", "A1", "A2", "A3"):
        r = metrics(apply_op(tok, op, txt_norm_median))
        rec["ops"][op] = r
        log(f"[算子 {op}] " + json.dumps(r, ensure_ascii=False))

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"[写盘] {args.out_json}")
    log("OPS-DIAG-OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
