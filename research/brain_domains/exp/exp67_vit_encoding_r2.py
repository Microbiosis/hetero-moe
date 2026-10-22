#!/usr/bin/env python3
"""exp67 — Tier 0 第二半：ViT-H-14 特征 → EEG 脑响应的 Ridge 编码（预测分数）。

## 为什么要有这个实验

exp64/65/66 做完的是 **RSA（表征几何）**：EEG 的 RDM 与 ViT 特征的 RDM 到底对不对得上。
答案是「对得上」（去共同 ERP 后 r = +0.0798，p = 0.005，跨 10/10 被试复现）。

但 RSA 只回答「几何结构相似」，不回答「**能不能用模型特征预测脑活动**」。
ROADMAP 里 Tier 0 的产出写的是两样：

    模型特征对脑响应的预测分数 R²；表征几何对比

前半句就是本脚本。

## 三个容易做错的地方，本脚本逐个堵住

1. **交叉验证必须按「图像」切，不能按「试次」切。**
   同一张图的 80 次重复高度相关，按试次切会让测试图像出现在训练集里 → R² 虚高。
   本脚本按图像做 5-fold，同一图像的所有重复永远在同一折。

2. **必须有置换零。**
   200 张图 × 1024 维特征，R² 天然会偏乐观（高维小样本）。
   置换零的做法是打乱「特征 ↔ 图像」的对应关系后重跑整条管线，
   得到 R² 在「无对应关系」时的分布。

3. **试次平均会人为抬高 R²。**
   试次平均后的响应没有单试次噪声，R² 自然更高。
   所以主结果报**试次平均**（可与 RSA 直接对照），
   同时报**单试次**版本（现实设定：只有一次呈现的 EEG），
   两者之比就是「去噪能挽回多少」。

## 参照系：跨被试脑对脑上限（BBB）

编码领域公认的上界是「用**别的被试**的脑响应去预测这个被试的脑响应」。
本脚本用同一批 200 个概念做：

    BBB:  其余 9 个被试的试次平均响应  →  本被试的试次平均响应

ViT→EEG 的 R² 若接近 BBB，说明模型特征的信息量接近「另一个人的大脑」；
若远低于 BBB，说明模型特征只捕捉到脑表征的一小部分。

## 判据

    ViT 编码 R² 显著高于置换零（p < 0.05）且 > 0   → 特征可预测脑响应
    ViT 编码 R² / BBB 上限                          → 模型特征的信息量占比

## 用法

    python -u exp67_vit_encoding_r2.py                # 全部
    python -u exp67_vit_encoding_r2.py --subjects=3   # 只跑前 3 个被试
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

FS = 250.0
WIN = 25              # 25 样本 @250Hz = 100 ms
BASELINE = 12         # 0-50ms ≈ 12 样本
N_FOLDS = 5
N_SUBJECTS = 10
N_NULL = 20           # 置换零次数
ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
FOURD = Path("/domains/omnibrain4d/eeg")
FEAT_PT = "/training/eeg/eeg-image-decode/ViT-H-14_features_test.pt"
OUT_DIR = Path("/domains/runs/tier0-encoding")
SEED = 0


def log(*a):
    print(*a, flush=True)


# ================================================================ 工具
def make_folds(n, k, rng):
    """按样本（图像）做 k-fold。返回 [(train_idx, test_idx), ...]"""
    idx = rng.permutation(n)
    return [(np.setdiff1d(idx, te), te)
            for te in np.array_split(idx, k)]


def ridge_cv_dual(X, Y, folds, alphas):
    """对偶形式 ridge（n < d 时最省）。返回 {alpha: (R2[t], r[t])}。

    X: (n, d) 特征；Y: (n, t) 目标。
    R² 以**训练折均值**为基线（predict-the-mean），这是泛化 R² 的标准算法。

    ★ 必须对 Y 中心化：
      无截距的 ridge 惩罚的是 ||w||²，λ→大时预测被收缩到 **0**；
      而 R² 的基线是训练均值 ȳ。若不中心化，整条 alpha 曲线会被系统性压低
      （强正则端 R² 变成大负数），连 BBB 上限都会算出负值 —— 那是管线错误，
      不是"没有信号"。中心化后 λ→大时 pred→ȳ，R²→0，曲线才有意义。
    """
    n, d = X.shape
    out = {a: {"r2": np.zeros(Y.shape[1]), "r": np.zeros(Y.shape[1])} for a in alphas}
    for tr, te in folds:
        Xtr, Xte = X[tr], X[te]
        mu = Xtr.mean(axis=0, keepdims=True)
        sd = Xtr.std(axis=0, keepdims=True) + 1e-6
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
        K = Xtr @ Xtr.T
        Kte = Xte @ Xtr.T
        scale = float((Xtr ** 2).sum() / d)
        ybar = Y[tr].mean(axis=0, keepdims=True)
        Ytr = Y[tr] - ybar          # ★ 中心化：见下方说明
        Yte = Y[te]
        ss_tot = ((Yte - ybar) ** 2).sum(axis=0)
        for a in alphas:
            A = K + (scale * a) * np.eye(len(tr))
            W = np.linalg.solve(A, Ytr)            # (ntr, t)
            pred = Kte @ W + ybar   # ★ 加回训练均值
            ss_res = ((Yte - pred) ** 2).sum(axis=0)
            out[a]["r2"] += 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
            pc = pred - pred.mean(axis=0, keepdims=True)
            tc = Yte - Yte.mean(axis=0, keepdims=True)
            denom = np.sqrt((pc ** 2).sum(axis=0) * (tc ** 2).sum(axis=0))
            out[a]["r"] += np.sum(pc * tc, axis=0) / np.maximum(denom, 1e-12)
    nf = len(folds)
    for a in out:
        out[a]["r2"] /= nf
        out[a]["r"] /= nf
    return out


def _per_window(r2_flat, n_ch=63, n_bin=10):
    """把 (n_ch*n_bin,) 摊成每个 100ms 窗的中位 R²。

    目标顺序 = [窗0×63通道, 窗1×63通道, ...]（见 build_targets），
    所以 reshape(n_bin, n_ch) 与之对应。
    """
    if len(r2_flat) != n_ch * n_bin:
        return None
    M = np.asarray(r2_flat).reshape(n_bin, n_ch)
    return np.median(M, axis=1)


def pick_best(res, n_ch=63, n_bin=10):
    """选 alpha，返回 (alpha, 逐窗中位R²数组, 逐目标R²)。

    **打分用「逐窗中位的最大值」而不是全局中位**。
    原因：630 个目标里只有 100–400ms 那几个窗有信号，全局中位会被
    无信号窗压到 ~0，真实效应就看不见了。null 分布用同一套规则，
    所以比较是公平的。
    """
    best_a, best_w, best_flat = None, None, None
    for a in ALPHAS:
        flat = res[a]["r2"]
        w = _per_window(flat, n_ch, n_bin)
        if w is None:
            continue
        score = float(np.max(w))
        if best_a is None or score > float(np.max(best_w)):
            best_a, best_w, best_flat = a, w, flat
    return best_a, best_w, best_flat


def build_targets(eeg):
    """eeg: (n_img, n_rep, n_ch, T) float16 → (n_img, n_ch*n_bins) 试次平均目标。

    基线校正（0-50ms 减每试次每通道均值）+ 10 个 100ms 窗，与 RSA 同一时间栅格。
    """
    X = eeg.astype(np.float32)
    X -= X[:, :, :, :BASELINE].mean(axis=3, keepdims=True)
    n_img, n_rep, n_ch, T = X.shape
    bins = []
    for a in range(0, T, WIN):
        b = min(a + WIN, T)
        bins.append(X[:, :, :, a:b].mean(axis=3))       # (n_img, n_rep, n_ch)
    n_bin = len(bins)
    Y = np.stack(bins, axis=2)                          # (n_img, n_rep, n_bin, n_ch)
    # 注意：stack(axis=2) 的结果是 (n_img, n_rep, n_bin, n_ch)，
    # 展平后目标顺序 = [窗0×63通道, 窗1×63通道, ...]，与 _per_window 的
    # reshape(n_bin, n_ch) 一致。n_bin 必须显式取，不能用 Y.shape[3]（那是 n_ch）。
    Y = Y.reshape(n_img, n_rep, n_bin * n_ch)
    return Y.mean(axis=1), n_bin


# ================================================================ PART 0
def part0_probe():
    import torch
    log("=" * 80)
    log("PART 0 — 数据与特征")
    log("=" * 80)
    f = torch.load(FEAT_PT, map_location="cpu", weights_only=False)
    IMG = f["img_features"].numpy().astype(np.float64)
    TXT = f["text_features"].numpy().astype(np.float64)
    log(f"  ViT img_features {IMG.shape} | text_features {TXT.shape}")
    log(f"  4D 目录 {FOURD}: {sorted(p.name for p in FOURD.glob('sub-*.npz'))[:12]}")
    return IMG, TXT


# ================================================================ PART 1/2
def part12_encode(IMG, TXT, rng):
    """逐被试：ViT-img / ViT-text / BBB / 置换零 的编码性能。"""
    log("\n" + "=" * 80)
    log("PART 1/2 — Ridge 编码（5-fold 图像级 CV）")
    log("=" * 80)

    # ---- 先把所有被试的试次平均响应读进来（BBB 需要其他被试的平均）
    subj = {}
    for sid in range(1, N_SUBJECTS + 1):
        fp = FOURD / f"sub-{sid:02d}.npz"
        if not fp.exists():
            continue
        eeg = np.load(fp)["eeg"]
        Y, n_bin = build_targets(eeg)
        del eeg
        subj[sid] = Y
        log(f"  sub-{sid:02d}: 目标 {Y.shape}（{n_bin} 个 100ms 窗 × 63 通道）")

    if len(subj) < 2:
        log("  被试不足，无法算 BBB 上限")
        return {}

    rows = []
    for sid, Y in subj.items():
        t0 = time.time()
        n_img = Y.shape[0]
        folds = make_folds(n_img, N_FOLDS, rng)

        # BBB：其余被试对同一图像的平均响应 → 本被试
        others = [v for k, v in subj.items() if k != sid]
        Y_bbb = np.mean(np.stack(others, axis=0), axis=0)

        # 置换零：打乱「特征 ↔ 图像」对应关系
        def perm_null(X, n=N_NULL):
            """每次置换返回「逐窗中位 R²」数组（10 个窗），与真实值同规则。"""
            vals = []
            for _ in range(n):
                Xp = X[rng.permutation(n_img)]
                res = ridge_cv_dual(Xp, Y, folds, ALPHAS)
                _, w, _ = pick_best(res)
                vals.append(w)
            return np.array(vals)          # (n_null, n_bin)

        X_img = IMG.copy()
        X_txt = TXT.copy()

        res_img = ridge_cv_dual(X_img, Y, folds, ALPHAS)
        res_txt = ridge_cv_dual(X_txt, Y, folds, ALPHAS)
        res_bbb = ridge_cv_dual(Y_bbb, Y, folds, ALPHAS)

        a_img, w_img, r2_img = pick_best(res_img)
        a_txt, w_txt, r2_txt = pick_best(res_txt)
        a_bbb, w_bbb, r2_bbb = pick_best(res_bbb)

        null_img = perm_null(X_img)        # (N_NULL, n_bin)
        null_txt = perm_null(X_txt)

        # 判据窗：真实值最强的那个窗
        bi = int(np.argmax(w_img))
        bt = int(np.argmax(w_txt))
        null_img_at_b = null_img[:, bi]
        null_txt_at_b = null_txt[:, bt]
        p_img = float((np.sum(null_img_at_b >= w_img[bi]) + 1) / (len(null_img_at_b) + 1))
        p_txt = float((np.sum(null_txt_at_b >= w_txt[bt]) + 1) / (len(null_txt_at_b) + 1))

        rec = {
            "subject": f"sub-{sid:02d}",
            "n_img": n_img,
            "alpha_img": a_img, "alpha_txt": a_txt, "alpha_bbb": a_bbb,
            "best_win_img_ms": [bi * 100, (bi + 1) * 100],
            "best_win_txt_ms": [bt * 100, (bt + 1) * 100],
            "r2_img_best_window": float(w_img[bi]),
            "r2_txt_best_window": float(w_txt[bt]),
            "r2_bbb_best_window": float(w_bbb[bi]),
            "r2_img_mean_window": float(np.mean(w_img)),
            "r2_bbb_mean_window": float(np.mean(w_bbb)),
            "null_img_best_window_median": float(np.median(null_img_at_b)),
            "null_img_best_window_p95": float(np.percentile(null_img_at_b, 95)),
            "null_txt_best_window_median": float(np.median(null_txt_at_b)),
            "p_img": p_img, "p_txt": p_txt,
            "ratio_img_over_bbb": (float(w_img[bi]) / float(w_bbb[bi]))
            if abs(w_bbb[bi]) > 1e-9 else None,
            "per_window_r2_img": [float(x) for x in w_img],
            "per_window_r2_bbb": [float(x) for x in w_bbb],
            "per_window_null_img_p95": [float(x) for x in np.percentile(null_img, 95, axis=0)],
            "elapsed_s": round(time.time() - t0, 1),
        }
        rows.append(rec)
        log(f"  sub-{sid:02d}: 最佳窗 {bi*100}-{bi*100+100}ms  "
            f"R² img {w_img[bi]:+.4f} (置换零中位 {rec['null_img_best_window_median']:+.4f}, "
            f"p={p_img:.3f}) | txt {w_txt[bt]:+.4f} (p={p_txt:.3f}) | "
            f"BBB {w_bbb[bi]:+.4f} | img/BBB {rec['ratio_img_over_bbb']:.2f}  [{rec['elapsed_s']}s]")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "exp67_encoding.json").write_text(
            json.dumps({"per_subject": rows}, indent=2, ensure_ascii=False),
            encoding="utf-8")
    return {"per_subject": rows}


# ================================================================ PART 3
def part3_singletrial(IMG, TXT, rng):
    """sub-01 的单试次版本（现实设定），primal ridge。

    目标 = 单试次响应 (16000, 630)；特征 1024 维 > 样本量不足 → 用 primal 形式。
    同一个图像的 80 次重复必须整折进出。
    """
    log("\n" + "=" * 80)
    log("PART 3 — 单试次编码（sub-01，primal ridge）")
    log("=" * 80)
    fp = FOURD / "sub-01.npz"
    if not fp.exists():
        return {}

    eeg = np.load(fp)["eeg"]
    n_img, n_rep, n_ch, T = eeg.shape
    X = eeg.astype(np.float32)
    X -= X[:, :, :, :BASELINE].mean(axis=3, keepdims=True)
    bins = []
    for a in range(0, T, WIN):
        b = min(a + WIN, T)
        bins.append(X[:, :, :, a:b].mean(axis=3))
    Yb = np.stack(bins, axis=2)                        # (n_img, n_rep, n_bin, n_ch)
    n_bin = Yb.shape[2]
    Y = Yb.reshape(n_img * n_rep, n_bin * n_ch)         # (16000, 630)
    del X, Yb, eeg
    log(f"  单试次目标 {Y.shape}")

    folds = make_folds(n_img, N_FOLDS, rng)
    results = {}
    for name, F in (("img", IMG), ("txt", TXT)):
        per_alpha = {a: [] for a in ALPHAS}
        for tr_img, te_img in folds:
            ntr, nte = len(tr_img), len(te_img)
            # 试次级索引：图像 i 的 n_rep 次重复在 Y 里是连续的一段
            # （Y 由 (n_img, n_rep, ...) 展平而来，图像主序、重复次序）
            tr = np.concatenate([np.arange(i * n_rep, (i + 1) * n_rep)
                                 for i in tr_img])
            te = np.concatenate([np.arange(i * n_rep, (i + 1) * n_rep)
                                 for i in te_img])
            Xtr_i = F[tr_img]                       # (ntr, 1024) 图像级特征
            Xte_i = F[te_img]
            mu = Xtr_i.mean(axis=0, keepdims=True)
            sd = Xtr_i.std(axis=0, keepdims=True) + 1e-6
            Xtr_i = (Xtr_i - mu) / sd
            Xte_i = (Xte_i - mu) / sd

            # 目标按图像聚合：(n, n_rep, n_targets)
            Ytr = Y[tr].reshape(ntr, n_rep, -1).astype(np.float64)
            Yte = Y[te].reshape(nte, n_rep, -1).astype(np.float64)
            ybar = Ytr.reshape(ntr * n_rep, -1).mean(axis=0)          # (t,) 训练试次均值

            # ★ 同一图像的 n_rep 次重复共享特征 ⇒
            #     XᵀX = n_rep · X_imgᵀ X_img
            #     XᵀY = n_rep · X_imgᵀ (Σ_reps Y − n_rep·ȳ)
            #   不必真的展开 12800×1024 的矩阵。
            XtX = (Xtr_i.T @ Xtr_i).astype(np.float64) * n_rep
            Ysum = Ytr.sum(axis=1) - n_rep * ybar                     # (ntr, t)
            XtY = (Xtr_i.T @ Ysum).astype(np.float64)

            d = XtX.shape[0]
            tr_scale = float(np.trace(XtX)) / d
            Yte_flat = Yte.reshape(nte * n_rep, -1)
            ss_tot = ((Yte_flat - ybar) ** 2).sum(axis=0)
            for a in ALPHAS:
                A = XtX.copy()
                A.flat[:: d + 1] += a * tr_scale
                W = np.linalg.solve(A, XtY)
                pred_img = Xte_i @ W + ybar        # ★ XtY 用的是中心化 Y，解出的是偏差
                pred = np.repeat(pred_img, n_rep, axis=0)            # (nte*n_rep, t)
                ss_res = ((Yte_flat - pred) ** 2).sum(axis=0)
                per_alpha[a].append(1.0 - ss_res / np.maximum(ss_tot, 1e-12))
        per_alpha = {a: float(np.max(_per_window(np.mean(v, axis=0))))
                     for a, v in per_alpha.items()}
        a_best = max(per_alpha, key=lambda k: per_alpha[k])
        results[name] = {"alpha": a_best, "r2_best_window": per_alpha[a_best],
                         "curve": {str(k): float(v) for k, v in per_alpha.items()}}
        log(f"  {name}: 单试次最佳窗 R² 中位 {per_alpha[a_best]:+.4f} (alpha={a_best})")
        log(f"    alpha 曲线: " + ", ".join(f"{k}:{v:+.4f}" for k, v in per_alpha.items()))
    return results


# ================================================================ 主
def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    report = {"schema_version": "1.0", "script": "exp67_vit_encoding_r2",
              "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "design": {
                  "cv": f"{N_FOLDS}-fold over 200 IMAGES (同一图像的 80 重复整折进出)",
                  "ridge": "dual form (n=200 < d=1024); lambda 相对特征方差缩放",
                  "baselines": ["置换零（打乱特征↔图像对应）",
                                "BBB 跨被试脑对脑上限"],
                  "targets": "基线校正 0-50ms + 10 个 100ms 窗 × 63 通道 = 630 目标",
              }}

    IMG, TXT = part0_probe()
    if "--only-single" in sys.argv:
        # 只补跑 PART 3（单试次）。主结果 PART 1/2 已在 exp67_encoding.json 里，
        # 不必为此重跑 60 分钟。
        report["single_trial"] = part3_singletrial(IMG, TXT, rng)
        (OUT_DIR / "exp67_single_trial.json").write_text(
            json.dumps(report["single_trial"], indent=2, ensure_ascii=False),
            encoding="utf-8")
        print("=== SINGLE JSON BEGIN ===")
        print(json.dumps(report["single_trial"], indent=2, ensure_ascii=False))
        print("=== SINGLE JSON END ===")
        log(f"\n单试次部分完成，耗时 {time.time()-t0:.0f}s")
        return
    report["encoding"] = part12_encode(IMG, TXT, rng)
    report["single_trial"] = part3_singletrial(IMG, TXT, rng)

    # ---------------- 判定 ----------------
    log("\n" + "=" * 80)
    log("判定")
    log("=" * 80)
    v = {}
    rows = report["encoding"].get("per_subject", [])
    if rows:
        med_img = float(np.median([r["r2_img_best_window"] for r in rows]))
        med_txt = float(np.median([r["r2_txt_best_window"] for r in rows]))
        med_bbb = float(np.median([r["r2_bbb_best_window"] for r in rows]))
        med_null = float(np.median([r["null_img_best_window_median"] for r in rows]))
        n_sig = sum(1 for r in rows if r["p_img"] < 0.05)
        ratio = (med_img / med_bbb) if abs(med_bbb) > 1e-9 else None
        wins = {}
        for r in rows:
            w = r["best_win_img_ms"][0]
            wins[w] = wins.get(w, 0) + 1
        v["encoding_significant"] = {
            "ok": bool(n_sig >= max(3, len(rows) // 2)),
            "median_r2_img_best_window": med_img,
            "median_r2_txt_best_window": med_txt,
            "median_null_at_best_window": med_null,
            "n_sig_subjects": n_sig, "n_subjects": len(rows),
            "best_window_hist": wins,
            "note": "ViT-img 最佳窗编码 R² 显著高于置换零的被试数"}
        v["vs_bbb"] = {"ok": bool(ratio is not None and ratio > 0),
                       "median_r2_bbb_best_window": med_bbb,
                       "ratio_img_over_bbb": ratio,
                       "note": "ViT 编码 R² / 脑对脑上限"}
        log(f"  编码显著      {'✅' if v['encoding_significant']['ok'] else '❌'}  "
            f"最佳窗 R² 中位 img {med_img:+.4f} vs 置换零 {med_null:+.4f}，"
            f"显著被试 {n_sig}/{len(rows)}；最佳窗分布 {wins}")
        log(f"  vs BBB 上限   {'✅' if v['vs_bbb']['ok'] else '❌'}  "
            f"BBB {med_bbb:+.4f}，img/BBB = {ratio:.3f}" if ratio else "  vs BBB: n/a")
    st = report.get("single_trial", {})
    if st:
        v["single_trial"] = {"ok": bool(st.get("img", {}).get("r2_best_window", 0) > 0),
                              "r2_img": st.get("img", {}).get("r2_best_window"),
                              "note": "单试次最佳窗编码 R²（现实设定）"}
        log(f"  单试次        {'✅' if v['single_trial']['ok'] else '❌'}  "
            f"最佳窗 R² img {st.get('img', {}).get('r2_best_window')}")
    report["verdicts"] = v
    report["elapsed_s"] = round(time.time() - t0, 1)
    (OUT_DIR / "exp67_encoding_full.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== JSON BEGIN ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("=== JSON END ===")
    log(f"\n报告 → {OUT_DIR}/exp67_encoding_full.json  耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
