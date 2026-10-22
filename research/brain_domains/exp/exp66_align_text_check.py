#!/usr/bin/env python3
"""exp66 — 概念顺序对齐的**独立**检验（完全不碰 EEG）。

## 为什么需要这个

exp64/exp65 用「置换零分布」证明了 EEG-RDM 与 ViT-RDM 的对应不是偶然，
但那是**用 EEG 本身**做检验的：如果 EEG 与特征的顺序同时错位，置换检验同样会
给出显著结果（它检验的是「非随机」，不是「正确」）。

本脚本绕开 EEG，直接验证**特征文件的行序 == 概念序**：

  test.pt 里每条试次都带 `text`（概念文本）、`img`（图像路径）、`label`（概念 ID）。
  如果 ViT-H-14 的 `text_features` 就是按这 200 个概念的顺序抽的，那么
  「由 test.pt 的文本字符串构造的 RDM」应当与「ViT text_features 的 RDM」高度相关，
  且显著高于任何随机置换。

  这是纯文本/纯特征层的检验，与 EEG 无关，因此能独立判定顺序对不对。

## 判据

  r(字符串 RDM, ViT text RDM) > 0.5 且 > 置换零 95 分位  → 行序 == 概念序 ✅

同时输出：
  * 概念 ID 是否与行号单调对应（label[i] 是否 == i 或某种固定置换）
  * 图像路径样例（能否看出排序规则）
  * 字符串 RDM 与 ViT **image** RDM 的相关（应当也显著 —— 概念相同的图更相似）
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

EEG_PT = "/training/eeg/things-eeg/Preprocessed_data_250Hz_whiten/sub-01/test.pt"
FEAT_PT = "/training/eeg/eeg-image-decode/ViT-H-14_features_test.pt"
OUT_DIR = Path("/domains/runs/tier0-diag")
N_PERM = 200
SEED = 0


def log(*a):
    print(*a, flush=True)


def _rank(a):
    from scipy.stats import rankdata
    return rankdata(a)


def spearman(a, b):
    ra = _rank(a).astype(np.float64)
    rb = _rank(b).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def rdm_full(mat):
    m = np.asarray(mat, dtype=np.float64)
    m = m - m.mean(axis=1, keepdims=True)
    m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)
    r = m @ m.T
    np.fill_diagonal(r, 1.0)
    return 1.0 - r


def tri(R):
    return R[np.triu_indices_from(R, k=1)]


def rsa_with_null(R_a, R_b, n_perm=N_PERM, rng=None):
    rng = rng or np.random.default_rng(SEED)
    a = tri(R_a)
    r_obs = spearman(a, tri(R_b))
    null = np.empty(n_perm)
    n = R_b.shape[0]
    for i in range(n_perm):
        p = rng.permutation(n)
        null[i] = spearman(a, tri(R_b[np.ix_(p, p)]))
    return r_obs, null


def tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def tfidf_matrix(docs):
    """docs: list[str] → (n, vocab) TF-IDF（已 L2 归一）。"""
    toks = [tokenize(d) for d in docs]
    df = Counter()
    for t in toks:
        df.update(set(t))
    vocab = sorted(df.keys())
    idx = {w: i for i, w in enumerate(vocab)}
    n = len(docs)
    M = np.zeros((n, len(vocab)), dtype=np.float64)
    for i, t in enumerate(toks):
        for w in t:
            M[i, idx[w]] += 1.0
    dfv = np.array([df[w] for w in vocab], dtype=np.float64)
    idf = np.log((1.0 + n) / (1.0 + dfv)) + 1.0
    M *= idf
    nrm = np.linalg.norm(M, axis=1, keepdims=True)
    M = M / (nrm + 1e-12)
    return M


def main():
    import torch

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    report = {"schema_version": "1.0", "script": "exp66_align_text_check"}

    log("=" * 78)
    log("exp66 — 概念顺序对齐的独立检验（不碰 EEG）")
    log("=" * 78)

    d = torch.load(EEG_PT, map_location="cpu", weights_only=False)
    f = torch.load(FEAT_PT, map_location="cpu", weights_only=False)

    txt = np.asarray(d["text"])
    img = np.asarray(d["img"])
    lab = np.asarray(d["label"])
    n_img, n_rep = txt.shape
    log(f"\n  text {txt.shape} | img {img.shape} | label {lab.shape}")
    report["shapes"] = {"text": list(txt.shape), "img": list(img.shape),
                        "label": list(lab.shape)}

    # 每个概念取第 0 次重复的文本/图/标签
    texts = [str(txt[c, 0]) for c in range(n_img)]
    imgs = [str(img[c, 0]) for c in range(n_img)]
    labels = [int(lab[c, 0]) for c in range(n_img)]

    # 一致性：同一概念的所有重复是否同文本/同标签
    txt_consistent = all(len(set(map(str, txt[c]))) == 1 for c in range(n_img))
    lab_consistent = all(len(set(map(int, lab[c]))) == 1 for c in range(n_img))
    log(f"  概念内文本一致: {txt_consistent} | 概念内标签一致: {lab_consistent}")
    report["text_consistent"] = bool(txt_consistent)
    report["label_consistent"] = bool(lab_consistent)

    log("\n  样例（概念 0,1,2,198,199）:")
    for c in [0, 1, 2, 198, 199]:
        log(f"    [{c:3d}] label={labels[c]:>8}  text={texts[c][:42]!r}")
        log(f"          img={imgs[c]}")
    report["samples"] = [{"row": c, "label": labels[c], "text": texts[c],
                          "img": imgs[c]} for c in [0, 1, 2, 198, 199]]

    uniq_labels = sorted(set(labels))
    log(f"\n  label 取值: n_unique={len(uniq_labels)} "
        f"min={uniq_labels[0]} max={uniq_labels[-1]}")
    log(f"  label == 行号? {labels == list(range(n_img))}")
    log(f"  label 单调递增? {labels == sorted(labels)}")
    report["label_unique"] = len(uniq_labels)
    report["label_is_row_index"] = bool(labels == list(range(n_img)))
    report["label_sorted"] = bool(labels == sorted(labels))

    # 字符串 RDM
    log("\n" + "=" * 78)
    log("关键检验 — 字符串 RDM vs ViT 特征 RDM")
    log("=" * 78)
    M = tfidf_matrix(texts)
    R_str = rdm_full(M)
    R_txt_vit = rdm_full(f["text_features"].numpy().astype(np.float64))
    R_img_vit = rdm_full(f["img_features"].numpy().astype(np.float64))

    r_t, null_t = rsa_with_null(R_str, R_txt_vit, N_PERM, rng)
    r_i, null_i = rsa_with_null(R_str, R_img_vit, N_PERM, rng)
    pt = float((np.sum(np.abs(null_t) >= abs(r_t)) + 1) / (N_PERM + 1))
    pi = float((np.sum(np.abs(null_i) >= abs(r_i)) + 1) / (N_PERM + 1))

    log(f"  r(字符串RDM, ViT text_features RDM) = {r_t:+.4f}  p={pt:.4f}  "
        f"零分布 sd={null_t.std():.4f} 95分位={np.percentile(np.abs(null_t),95):.4f}")
    log(f"  r(字符串RDM, ViT img_features  RDM) = {r_i:+.4f}  p={pi:.4f}  "
        f"零分布 sd={null_i.std():.4f} 95分位={np.percentile(np.abs(null_i),95):.4f}")

    ok = bool(pt < 0.05 and r_t > 0.5)
    log(f"\n  → {'✅ 特征文件行序 == 概念序（对齐确认）' if ok else '❌ 未通过'}")
    report["r_text_vs_vit_text"] = {"r": r_t, "p": pt,
                                     "null_sd": float(null_t.std()),
                                     "null_p95": float(np.percentile(np.abs(null_t), 95))}
    report["r_text_vs_vit_img"] = {"r": r_i, "p": pi,
                                   "null_sd": float(null_i.std()),
                                   "null_p95": float(np.percentile(np.abs(null_i), 95))}
    report["order_confirmed"] = ok

    report["elapsed_s"] = round(time.time() - t0, 1)
    (OUT_DIR / "exp66_align_text_check.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    # 同时打到 stdout —— 防止 Job 写桶时 Xet 通道卡住导致结果丢失
    print("=== JSON BEGIN ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("=== JSON END ===")
    log(f"\n报告 → {OUT_DIR}/exp66_align_text_check.json  耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
