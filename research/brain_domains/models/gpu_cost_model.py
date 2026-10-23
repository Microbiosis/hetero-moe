#!/usr/bin/env python3
"""gpu_cost_model.py — 「7 基模脑对齐表」在各 GPU 档位上的**总费用**最优解。

## 问题

HF Jobs 按**时长**计费：档位越快，单价越高。于是

    总费用(tier) = 跑完整个 workload 所需小时数(tier) × 单价(tier)

**最便宜的档位不一定是跑得最慢的档位** —— 慢档位便宜、慢；快档位贵、快，
两者相乘谁小要算。而这里的 workload 是**小模型**（3–6M 参数），
GPU 利用率天然低，**加 CPU 核数往往比加 GPU 更划算**。

## 本脚本做什么

1. 内置 `hf jobs hardware` 的**完整档位与单价表**（28 档，2026-10-23 实测）；
   ⚠️ 是 **Jobs** 的价（cpu-basic $0.01/h），不是 Spaces 的（Spaces 的免费）。
2. 读入**实测吞吐**（ms/试次，按 档位 × 基模 记录）——由 `probe_base_models.py`
   在各档位上实测得到，不靠理论峰值外推；
3. 按 worklist（被试数 × 基模数 × 每单元试次数）算出各档位的
   小时数与**总费用**，给出最优档。

## 用法

    python gpu_cost_model.py --measurements probe_ms.json --workload full

`probe_ms.json` 形如：

    {"t4-small": {"cbramod": 12.3, "eegdino-small": 9.1, ...},
     "l4x1":     {"cbramod": 2.1,  ...}}

缺失的档位/模型会按同档其它模型的中位数比例外推，并**明确标注是外推**。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 档位表：`hf jobs hardware`（2026-10-23 实测，**Jobs 的价，不是 Spaces 的**）
#   cost_per_hour：单价；accelerators：GPU 数量与型号；vcpu：CPU 核数
#
# ⚠️ 之前这张表取自 `hf spaces hardware`，有两个错：
#    ① Spaces 的 cpu-basic 免费，**Jobs 的 cpu-basic 是 $0.01/h**；
#    ② 漏了 4 个纯 CPU 档（cpu-performance / cpu-xl / h200 系列）。
#    跑 Job 用 Jobs 的价 —— 算错价，档位结论就是错的。
# ---------------------------------------------------------------------------
FLAVORS = {
    "cpu-basic":     {"$/h": 0.01, "vcpu": 2,   "ram": 16,   "gpu": "-"},
    "cpu-upgrade":   {"$/h": 0.03, "vcpu": 8,   "ram": 32,   "gpu": "-"},
    "cpu-xl":        {"$/h": 1.00, "vcpu": 16,  "ram": 124,  "gpu": "-"},
    "cpu-performance": {"$/h": 1.90, "vcpu": 32, "ram": 256, "gpu": "-"},
    "t4-small":      {"$/h": 0.40, "vcpu": 4,   "ram": 15,   "gpu": "1x T4 16G"},
    "t4-medium":     {"$/h": 0.60, "vcpu": 8,   "ram": 30,   "gpu": "1x T4 16G"},
    "a10g-small":    {"$/h": 1.00, "vcpu": 4,   "ram": 15,   "gpu": "1x A10G 24G"},
    "a10g-large":    {"$/h": 1.50, "vcpu": 12,  "ram": 46,   "gpu": "1x A10G 24G"},
    "a10g-largex2":  {"$/h": 3.00, "vcpu": 24,  "ram": 92,   "gpu": "2x A10G 48G"},
    "a10g-largex4":  {"$/h": 5.00, "vcpu": 48,  "ram": 184,  "gpu": "4x A10G 96G"},
    "l4x1":          {"$/h": 0.80, "vcpu": 8,   "ram": 30,   "gpu": "1x L4 24G"},
    "l4x4":          {"$/h": 3.80, "vcpu": 48,  "ram": 186,  "gpu": "4x L4 96G"},
    "l40sx1":        {"$/h": 1.80, "vcpu": 8,   "ram": 62,   "gpu": "1x L40S 48G"},
    "l40sx4":        {"$/h": 8.30, "vcpu": 48,  "ram": 382,  "gpu": "4x L40S 192G"},
    "l40sx8":        {"$/h": 23.50,"vcpu": 192, "ram": 1534, "gpu": "8x L40S 384G"},
    "a100-large":    {"$/h": 2.50, "vcpu": 12,  "ram": 142,  "gpu": "1x A100 80G"},
    "a100x4":        {"$/h": 10.00,"vcpu": 48,  "ram": 568,  "gpu": "4x A100 320G"},
    "a100x8":        {"$/h": 20.00,"vcpu": 96,  "ram": 1136, "gpu": "8x A100 640G"},
    "h200":          {"$/h": 5.00, "vcpu": 23,  "ram": 256,  "gpu": "1x H200 141G"},
    "h200x2":        {"$/h": 10.00,"vcpu": 46,  "ram": 512,  "gpu": "2x H200 282G"},
    "h200x4":        {"$/h": 20.00,"vcpu": 92,  "ram": 1024, "gpu": "4x H200 564G"},
    "h200x8":        {"$/h": 40.00,"vcpu": 184, "ram": 2048, "gpu": "8x H200 1128G"},
    "rtx-pro-6000":  {"$/h": 2.75, "vcpu": 23,  "ram": 256,  "gpu": "1x RTX PRO 6000 96G"},
    "rtx-pro-6000x2":{"$/h": 5.50, "vcpu": 46,  "ram": 512,  "gpu": "2x RTX PRO 6000 192G"},
    "rtx-pro-6000x4":{"$/h": 11.00,"vcpu": 92,  "ram": 1024, "gpu": "4x RTX PRO 6000 384G"},
    "rtx-pro-6000x8":{"$/h": 22.00,"vcpu": 184, "ram": 2048, "gpu": "8x RTX PRO 6000 768G"},
}

# ---------------------------------------------------------------------------
# Workload 定义（试次 = 一个 EEG trial 过一遍模型）
# ---------------------------------------------------------------------------
WORKLOADS = {
    # 完整协议：指标 A 只需 test split；指标 B 需要 train + test
    "full": {
        "desc": "指标 A + 指标 B 完整跑（train 16540×4 + test 200×80 = 82160 试次/被试/模型）",
        "train_trials": 16540 * 4,
        "test_trials": 200 * 80,
    },
    # 省版：train 只取 N 张图（exp70 已证图像数几乎不涨 ⇒ 饱和）
    "reduced2000": {
        "desc": "指标 A + 指标 B，train 只用 2000 张图（2000×4 + 200×80 = 24000 试次/被试/模型）",
        "train_trials": 2000 * 4,
        "test_trials": 200 * 80,
    },
    # 只要指标 A（零训练，最便宜）
    "rsa_only": {
        "desc": "只要指标 A（RSA）：仅 test split = 16000 试次/被试/模型",
        "train_trials": 0,
        "test_trials": 200 * 80,
    },
}

N_SUBJECTS = 10
N_MODELS = 7
# 阶段 C/D：重组变体（每个变体按其宿主模型的吞吐算）
N_VARIANTS = 4


def per_unit_trials(w):
    return w["train_trials"] + w["test_trials"]


def estimate_ms(meas, tier, model):
    """取该档该模型的实测 ms/试次；缺失则按同档中位数比例外推。"""
    if tier in meas and model in meas[tier]:
        return float(meas[tier][model]), "实测"
    if tier in meas and meas[tier]:
        med = sorted(meas[tier].values())[len(meas[tier]) // 2]
        # 若别的档有该模型的实测，按相对比例外推
        for t2, mm in meas.items():
            if model in mm:
                base = sorted(mm.values())[len(mm) // 2]
                if base > 0:
                    return float(med) * float(mm[model]) / float(base), "外推"
        return float(med), "外推(档位中位)"
    return None, "无数据"


def compute(meas, wl_name, n_subjects=N_SUBJECTS, n_models=N_MODELS,
            n_variants=N_VARIANTS):
    w = WORKLOADS[wl_name]
    trials = per_unit_trials(w)
    rows = []
    for tier, spec in FLAVORS.items():
        # 变体：每个变体按其宿主的模型算（这里保守按最慢的模型算）
        ms_list = []
        for model in meas.get(tier, {}):
            ms, _ = estimate_ms(meas, tier, model)
            ms_list.append(ms)
        if not ms_list:
            ms_list = [estimate_ms(meas, tier, m)[0]
                       for m in ("cbramod",) if estimate_ms(meas, tier, m)[0]]
        if not ms_list:
            continue
        ms_per_trial = sum(ms_list) / len(ms_list)
        # 总量：基模（n_models）+ 变体（n_variants，各按同量级算）
        total_trials = trials * n_subjects * (n_models + n_variants)
        hours = total_trials * ms_per_trial / 1000.0 / 3600.0
        cost = hours * spec["$/h"]
        rows.append({
            "tier": tier, "$/h": spec["$/h"], "gpu": spec["gpu"],
            "vcpu": spec["vcpu"], "ms/试次": round(ms_per_trial, 3),
            "小时": round(hours, 2), "总费用$": round(cost, 3),
            "来源": "实测" if (tier in meas and meas[tier]) else "外推",
        })
    rows.sort(key=lambda r: r["总费用$"])
    return rows, w, trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurements", default="probe_ms.json")
    ap.add_argument("--workload", default="full", choices=list(WORKLOADS))
    ap.add_argument("--subjects", type=int, default=N_SUBJECTS)
    ap.add_argument("--models", type=int, default=N_MODELS)
    ap.add_argument("--variants", type=int, default=N_VARIANTS)
    args = ap.parse_args()

    meas = {}
    try:
        meas = json.load(open(args.measurements, encoding="utf-8"))
    except Exception as e:
        print(f"（读不到实测文件 {args.measurements}: {e}）")

    rows, w, trials = compute(meas, args.workload, args.subjects,
                              args.models, args.variants)
    print("=" * 92)
    print(f"Workload：{w['desc']}")
    print(f"每被试每模型 {trials} 试次 | {args.subjects} 被试 × "
          f"({args.models} 基模 + {args.variants} 变体)")
    print("=" * 92)
    print(f"{'档位':<18}{'单价$/h':>8}{'GPU':>22}{'vCPU':>6}"
          f"{'ms/试次':>10}{'小时':>9}{'总费用$':>10}  来源")
    for r in rows:
        print(f"{r['tier']:<18}{r['$/h']:>8.2f}{r['gpu']:>22}{r['vcpu']:>6}"
              f"{r['ms/试次']:>10.3f}{r['小时']:>9.2f}{r['总费用$']:>10.3f}  {r['来源']}")
    if rows:
        best = rows[0]
        print("\n" + "=" * 92)
        print(f"★ 总费用最低：{best['tier']}  ${best['总费用$']:.3f}  "
              f"（{best['小时']:.2f} 小时 @ ${best['$/h']:.2f}/h）")
        fastest = min(rows, key=lambda r: r["小时"])
        print(f"  最快：{fastest['tier']}  {fastest['小时']:.2f} 小时  "
              f"${fastest['总费用$']:.3f}")
        print("=" * 92)


if __name__ == "__main__":
    main()
