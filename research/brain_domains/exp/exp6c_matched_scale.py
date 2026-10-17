"""Phase 6c: 把小鼠降采样到猕猴的 recording 规模，做真正可比的对照。

## 为什么必须这样做

exp6b 发现: 小鼠 205-1240 units, 猕猴 40-192 units —— **范围零重叠**。
因此任何跨物种的 eff_dim 比较都混入了 recording 规模的差异。

## 做法

对每个小鼠 session:
  1. 从原始 spike 随机抽取 N=150 个 unit（猕猴中位数量级）
  2. 重做逐 session Gram PCA → 64 维（与原始流程一致）
  3. 分窗 8/stride 4
  4. 计算七个统计量
重复 3 次取中位（降低子采样方差）。

然后与猕猴（40-192 units）比较 —— 此时两者 recording 规模相当。

## 判据

  降采样后 ε² 大幅下降 (<0.05) → 原差异确认为规模伪影
  降采样后 ε² 仍大 (>0.14)     → 物种效应真实

## 注意

降采样会降低 SNR，可能使小鼠的统计量朝猕猴方向移动（因为两者都更 noisy），
这是"向噪声均值回归"而非"物种相似"。因此需要同时报告降采样前后的变化。
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

SEED = 0
N_SUB = 150          # 目标单元数（猕猴中位量级）
N_REPEAT = 3
WIN, STRIDE = 8, 4
PROJ_DIM = 64
MAX_WIN = 600


def stats_of(w):
    """七个统计量，逐窗算再取中位。"""
    x = w.astype(np.float64)
    n, C, T = x.shape

    a1, b1 = x[:, :, :-1], x[:, :, 1:]
    a2, b2 = x[:, :, :-2], x[:, :, 2:]

    def ac(a, b):
        num = (a * b).sum(axis=2)
        den = np.sqrt((a ** 2).sum(axis=2) * (b ** 2).sum(axis=2)) + 1e-12
        return np.median(num / den, axis=1)

    r1, r2 = ac(a1, b1), ac(a2, b2)

    spec = np.abs(np.fft.rfft(x, axis=2)) ** 2
    F = spec.shape[2]
    freq = np.fft.rfftfreq(T, d=1.0)[1:]
    sp = spec[:, :, 1:] + 1e-12
    lf = np.log(freq)[None, None, :]
    lp = np.log(sp)
    lfc = lf - lf.mean(axis=2, keepdims=True)
    slope = (lfc * (lp - lp.mean(axis=2, keepdims=True))).sum(axis=2) / (lfc ** 2).sum(axis=2)

    hi = spec[:, :, F // 2:].sum(axis=2)

    xc = x - x.mean(axis=2, keepdims=True)
    try:
        s = np.linalg.svd(xc, compute_uv=False)
        lam = s ** 2
        ed = (lam.sum(axis=1) ** 2) / ((lam ** 2).sum(axis=1) + 1e-12)
    except np.linalg.LinAlgError:
        ed = np.full(n, np.nan)

    mu = x.mean(axis=2, keepdims=True)
    sd = x.std(axis=2, keepdims=True) + 1e-12
    kurt = np.median((((x - mu) / sd) ** 4).mean(axis=2) - 3.0, axis=1)

    return {
        "ac1": float(np.nanmedian(r1)),
        "ac_ratio": float(np.nanmedian(r2 / np.maximum(np.abs(r1), 0.05))),
        "spec_slope": float(np.nanmedian(slope)),
        "hf_ratio": float(np.nanmedian(hi / (spec.sum(axis=2) + 1e-12))),
        "eff_dim": float(np.nanmedian(ed)),
        "kurtosis": float(np.nanmedian(kurt)),
    }


def process_session(spikes, rng, n_sub):
    """spikes: (n_units, n_bins_50ms)。返回统计量 dict。"""
    # 50ms -> 100ms
    nb = spikes.shape[1] // 2
    x = spikes[:, :nb * 2].reshape(spikes.shape[0], nb, 2).sum(axis=2).astype(np.float32)
    # z-score
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-6
    z = (x - mu) / sd
    # Gram PCA
    k = min(PROJ_DIM, z.shape[0] - 1, nb - 1)
    zc = z - z.mean(axis=1, keepdims=True)
    C = zc @ zc.T
    w_, U = np.linalg.eigh(C)
    o = np.argsort(-w_)[:k]
    proj = zc.T @ U[:, o]
    if proj.shape[1] < PROJ_DIM:
        proj = np.pad(proj, ((0, 0), (0, PROJ_DIM - proj.shape[1])))
    else:
        proj = proj[:, :PROJ_DIM]
    # 分窗
    starts = list(range(0, nb - WIN + 1, STRIDE))
    if not starts:
        return None
    arr = np.stack([proj[s:s + WIN].T for s in starts]).astype(np.float16)
    if len(arr) > MAX_WIN:
        idx = rng.choice(len(arr), MAX_WIN, replace=False)
        arr = arr[np.sort(idx)]
    return stats_of(arr)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/omnibrain")
    stein = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/data/steinmetz")
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/out/exp6c")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    print("=" * 74)
    print(f"Phase 6c — 小鼠降采样到 {N_SUB} units 后与猕猴比较")
    print("=" * 74)

    units = pd.read_parquet(stein / "units.parquet")
    rows = []

    # ---- 小鼠：原规模 + 降采样
    for sess in sorted(units["session"].unique()):
        f = stein / f"spikes_session_{sess:03d}.npz"
        if not f.exists():
            continue
        sp = np.load(f)["spikes"].astype(np.float32)
        n_u = sp.shape[0]

        # 原规模
        st_full = process_session(sp, rng, None)
        if st_full:
            rows.append({"species": "mouse", "group": f"sess{sess}",
                         "scale": "full", "n_units": n_u, **st_full})

        # 降采样
        if n_u > N_SUB:
            acc = []
            for _ in range(N_REPEAT):
                idx = rng.choice(n_u, N_SUB, replace=False)
                st = process_session(sp[idx], rng, None)
                if st:
                    acc.append(st)
            if acc:
                agg = {k: float(np.median([a[k] for a in acc])) for k in acc[0]}
                rows.append({"species": "mouse", "group": f"sess{sess}",
                             "scale": f"sub{N_SUB}", "n_units": N_SUB, **agg})
        print(f"  mouse sess {sess:02d}: {n_u} units → full + sub{N_SUB}")

    # ---- 猕猴
    d = root / "spike_view_macaque"
    meta = pd.read_parquet(d / "meta.parquet")
    for fname, sub in meta.groupby("file"):
        w = np.load(d / fname)["windows"]
        st = stats_of(w)
        rows.append({"species": "macaque", "group": fname, "scale": "native",
                     "n_units": int(sub["n_units"].iloc[0]), **st})

    df = pd.DataFrame(rows)
    df.to_csv(out / "phase6c.csv", index=False)

    print(f"\n  记录: {len(df)}")
    print(df.groupby(["species", "scale"]).agg(
        n=("ac1", "size"), units=("n_units", "median"),
        **{s: (s, "median") for s in ["ac1", "spec_slope", "hf_ratio", "eff_dim", "kurtosis"]}
    ).to_string())

    # ---- 比较：原规模 vs 降采样 vs 猕猴
    print("\n" + "=" * 74)
    print("关键对照: 小鼠(原) vs 小鼠(降采样) vs 猕猴")
    print("=" * 74)
    stats = ["ac1", "ac_ratio", "spec_slope", "hf_ratio", "eff_dim", "kurtosis"]
    print(f"{'统计量':<14} {'小鼠原':>10} {'小鼠降采样':>12} {'猕猴':>10} {'ε²降采样vs猕猴':>16}")
    summary = {}
    for st in stats:
        g_full = df[(df.species == "mouse") & (df.scale == "full")][st].dropna().to_numpy()
        g_sub = df[(df.species == "mouse") & (df.scale == f"sub{N_SUB}")][st].dropna().to_numpy()
        g_mac = df[df.species == "macaque"][st].dropna().to_numpy()
        if len(g_sub) < 3 or len(g_mac) < 3:
            continue
        H, p = kruskal(g_sub, g_mac)
        n = len(g_sub) + len(g_mac)
        e2 = max(0.0, (H - 2 + 1) / (n - 2))
        Hf, pf = kruskal(g_full, g_mac)
        e2_full = max(0.0, (Hf - 2 + 1) / (n - 2)) if len(g_full) else float("nan")
        summary[st] = {"mouse_full": round(float(np.median(g_full)), 4),
                       "mouse_sub": round(float(np.median(g_sub)), 4),
                       "macaque": round(float(np.median(g_mac)), 4),
                       "eps2_full": round(float(e2_full), 4),
                       "eps2_sub": round(float(e2), 4), "p_sub": float(p)}
        print(f"{st:<14} {np.median(g_full):>10.4f} {np.median(g_sub):>12.4f} "
              f"{np.median(g_mac):>10.4f} {e2:>16.4f} (原 {e2_full:.4f})")

    # 判据
    pcts = [(summary[s]["eps2_full"] - summary[s]["eps2_sub"]) /
            summary[s]["eps2_full"] * 100
            for s in summary if summary[s]["eps2_full"] > 0.01]
    med_drop = float(np.median(pcts)) if pcts else 0.0
    print(f"\n  降采样后 ε² 中位降幅: {med_drop:.1f}%")
    if med_drop > 60:
        v = "✅ 大部分差异是 recording 规模伪影"
    elif med_drop < 20:
        v = "⚠️ 差异不是规模伪影，物种效应可能真实"
    else:
        v = "⚠️ 部分为规模伪影"
    print(f"  判定: {v}")

    (out / "phase6c_report.json").write_text(json.dumps({
        "schema_version": "1.0", "n_sub": N_SUB, "n_repeat": N_REPEAT,
        "comparison": summary, "median_eps2_drop_pct": round(med_drop, 1),
        "verdict": v,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
