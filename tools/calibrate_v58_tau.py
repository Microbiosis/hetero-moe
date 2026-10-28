"""V58 阈值校准 —— 必须在远端跑, 本机不跑。

为什么需要:
  noise_tau 的系数是"两个已知正确极限之间的插值", 没有数据支撑。
  这个脚本扫 (n, d) × {纯噪声, 共享 k}, 记录:
    · 纯噪声的谱最大值 s_max  (阈值必须 > 它)
    · 共享 k 维时的恢复数 χ    (阈值必须 < 第 k 个奇异值)
  两者给出每个 (n,d) 上的**可行区间** [s_max, s_k], 校准就是让 τ 落在区间内。

判据:
  C1 每个 (n,d) 上 τ > s_max           (纯噪声 → χ=0)
  C2 每个 (n,d,k) 上 τ < s_k           (共享 k → χ≥k)
  C3 区间非空: s_max < s_1 对 k≥1 成立  (否则该 (n,d) 上根本不可分辨)
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from research.aligner.v58_entanglement_area.brane import cross_spectrum, noise_tau
from research.aligner.v57_adaptive_dimension.operators import omega

OUT = os.environ.get("V58CAL_ART_DIR", "v58cal_artifacts")
CACHE = os.environ.get("V58CAL_CACHE", "/tmp/v58cal.json")


def run():
    torch.manual_seed(0)
    print("=" * 78)
    print("V58 阈值校准: 每个 (n,d) 上的可行区间 [s_max(纯噪声), s_k(共享k)]")
    print("=" * 78)
    rows = []
    # n 从 96 到 2048; d 从远小于 n 到远大于 n, 覆盖两个极限
    for n in (96, 192, 512, 1024, 2048):
        for d in (32, 64, 128, 256, 512, 1024, 2560, 5120):
            A0, B0 = torch.randn(n, d), torch.randn(n, d)
            s_noise = float(cross_spectrum(A0, B0).max())

            # 共享 k 维 (k 取几个代表点, 不超过 d)
            rec = {}
            for k in (1, 2, 4, 8, 16):
                if k > d:
                    continue
                Z = torch.randn(n, k)
                A = torch.randn(n, d) + Z @ torch.randn(k, d)
                B = torch.randn(n, d) + Z @ torch.randn(k, d)
                spec = cross_spectrum(A, B)
                tau = noise_tau(d, d, n)
                rec[k] = {"s_k": float(spec[k - 1]), "chi": int((spec > tau).sum())}

            tau = noise_tau(d, d, n)
            beta = 1.0
            row = {
                "n": n, "d": d,
                "s_noise_max": round(s_noise, 4),
                "tau_now": round(tau, 4),
                "c1_noise_zero": tau > s_noise,
                "shared": {str(k): v for k, v in rec.items()},
            }
            # C2: τ 必须低于 s_k, 才能把 k 个通道都数出来
            c2 = {k: (tau < v["s_k"]) for k, v in rec.items()}
            row["c2_tau_below_sk"] = c2
            row["c3_interval_nonempty"] = (rec[min(rec)]["s_k"] > s_noise) if rec else None
            rows.append(row)

            flag = "OK " if (row["c1_noise_zero"] and all(c2.values())) else "BAD"
            print(f"[{flag}] n={n:<5} d={d:<5} s_noise={s_noise:8.3f} tau={tau:8.3f} "
                  f"C1={row['c1_noise_zero']} "
                  f"chi(k=1,2,4)={[rec.get(k,{}).get('chi') for k in (1,2,4)]} "
                  f"C2={list(c2.values())}")

    n_ok = sum(1 for r in rows if r["c1_noise_zero"] and all(r["c2_tau_below_sk"].values()))
    print(f"\n通过 {n_ok}/{len(rows)} 格")

    # 拟合: 找每格的可行区间中点, 反推 kappa 应该多少
    ratios = []
    for r in rows:
        k1 = r["shared"].get("1")
        if not k1:
            continue
        lo, hi = r["s_noise_max"], k1["s_k"]
        if hi > lo:
            beta = 1.0
            tau_base = (math.sqrt(r["d"] * r["d"]) / r["n"])
            mid = 0.5 * (lo + hi)
            ratios.append({"n": r["n"], "d": r["d"], "lo": lo, "hi": hi,
                           "mid": mid, "kappa_needed": mid / tau_base if tau_base else None})
    payload = {"rows": rows, "kappa_needed": ratios,
               "n_pass": n_ok, "n_total": len(rows),
               "note": "kappa_needed 是让 tau 落在区间中点所需的倍率; 若它随 (n,d) 系统性漂移, "
                       "说明凸组合形式本身不对, 需要换函数形式。"}
    os.makedirs(OUT, exist_ok=True)
    json.dump(payload, open(os.path.join(OUT, "calibration.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"\n落盘 {OUT}/calibration.json")

    print("\n=== 汇总 ===")
    c1 = sum(1 for r in rows if r["c1_noise_zero"])
    c3 = sum(1 for r in rows if r["c3_interval_nonempty"])
    print(f"  C1 纯噪声->chi=0 : {c1}/{len(rows)}")
    print(f"  C3 区间非空      : {c3}/{len(rows)}")
    print(f"  C2 tau<s_k(全部) : {n_ok}/{len(rows)}")
    print("  判定: " + ("PASS 阈值可用" if n_ok == len(rows) else "FAIL 阈值仍不可用"))

    print("\n=== kappa_needed 随 (n,d) 的走势 ===")
    for r in ratios:
        print(f"  n={r['n']:<5} d={r['d']:<5} 区间=[{r['lo']:.2f},{r['hi']:.2f}] "
              f"mid={r['mid']:.2f} kappa_needed={r['kappa_needed']:.3f}")


if __name__ == "__main__":
    run()
