#!/usr/bin/env python3
"""按 v47 实测结果回写 9 个包的 leakage_audit。

判据 (与 `_v47_merge_metrics.py` 一致): 比的是**本包原结论在样本外是否仍成立**,
不是「treatment 是否更好」。负向结论三个口径同为负 ⇒ clean。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "research/central_stability/v47_heldout_audit/results/metrics.json"

TARGETS = {
    "v8":  "research/routing_evolution/v8_central",
    "v12": "archive/negative_findings/v12_compose",
    "v19": "research/central_diagnostics/v19_c1_analysis",
    "v20": "research/central_diagnostics/v20_c1_position",
    "v30": "research/central_stability/v30_gate_softplus",
    "v31": "research/central_stability/v31_gate_gradient_scale",
    "v33": "research/scale_heterogeneous/v33_scale_hetero",
    "v34": "research/scale_heterogeneous/v34_dmn_subconscious",
    "v35": "research/scale_heterogeneous/v35_container_entries",
}

HEADER = """v47.0 样本外审计 (5 seeds × 3 口径, 配对逐种子中位数)。

被审计主张: {claim}

{table}

{head}
口径: in_sample 与原 runner 同构; heldout_noise = seed+1000 同刺激换噪声;
heldout_stimulus = 换文本的未见刺激。算子与判据见
`research/central_stability/v47_heldout_audit/SPEC.md`。

⚠️ 审计顺带发现本线 runner 的两个既有缺陷, 已修并记录在案:
(1) `research/_primitives/attention/attn_pool.py:107,116` 的 `P_m` / `W_o` 在
**任何种子上下文之外**随机初始化 ⇒ 同一配置跨进程不可复现 (实测同一配置
in-sample 读数在 0.23~0.61 之间跳); (2) numpy 全局 RNG 不受 `torch.manual_seed`
控制, 原 runner 的随机图像不受 seed 约束。本审计对两者都做了固定, 被审计的原
runner 没有 —— 因此上表的绝对值与原文档里的绝对值**不必逐位相等**, 但同一行内的
三口径是同源同口径的。
"""


def head_for(row):
    si = row["in_sample_median_pct"]
    noise, stim = row["heldout_noise"], row["heldout_stimulus"]
    keep_n, keep_s = noise.get("retention"), stim.get("retention")

    if row["status"] == "clean":
        direction = "增益" if si > 0 else "负向"
        s = (f"**三个口径同向**（{direction}结论在样本外复现）⇒ 原结论不依赖 "
             f"in-sample 口径，`leakage_audit` 记 clean。")
        if si > 0:
            s += (f"\n幅度上样本外只保留 {min(x for x in (keep_n, keep_s) if x is not None) * 100:.0f}%"
                  f"~{max(x for x in (keep_n, keep_s) if x is not None) * 100:.0f}% —— "
                  "方向站得住，但**效应量远小于 in-sample 的报数**。")
        return s
    if row["status"] == "invalidated":
        return "**in-sample 的增益在样本外翻转** ⇒ 原结论不成立，判为失效。"
    return ("两口径方向不稳定或逐种子分裂，**不足以判定**，保持 pending；"
            "要结清需增加 seeds 或改用更强的对照。")


def build_note(claim, row):
    table = ["| 口径 | 中位改进 | 胜出 | 算子判定 | 保留率 |",
             "|---|---:|---:|:---:|---:|"]
    table.append(f"| in_sample | {row['in_sample_median_pct']:+.1f}% | — | — | — |")
    for label, key in (("heldout_noise", "heldout_noise"), ("heldout_stimulus", "heldout_stimulus")):
        d = row[key]
        keep = "" if d.get("retention") is None else f"{d['retention'] * 100:.0f}%"
        table.append(f"| {label} | {d['median_pct']:+.1f}% | {d['wins']} | "
                     f"{d.get('operator_verdict')} | {keep} |")
    return HEADER.format(claim=claim, table="\n".join(table), head=head_for(row))


def main():
    if not METRICS.exists():
        print("缺 results/metrics.json, 先跑两个 runner + _v47_merge_metrics.py")
        return 1
    per = json.loads(METRICS.read_text(encoding="utf-8"))["per_package"]

    for key, rel in TARGETS.items():
        path = ROOT / rel / "MANIFEST.json"
        man = json.loads(path.read_text(encoding="utf-8"))
        before = man.get("leakage_audit", {}).get("status")
        row = per.get(key)
        if row is None or "blocked" in row:
            man["leakage_audit"] = {
                "status": "pending", "audits": [], "audited_by": "47.0",
                "note": ("v47 审计**已排程但未能执行** —— 阻塞在第三方 `tabldm` 包缺失: "
                         "`occams/Xiaomi-TabLDM` 仓库经 `HfApi.list_repo_files` 核实只有 10 个"
                         "文件、零个 `.py`(权重 + 文档 + 图片), 没有可安装的源码, 而本包的"
                         "入口表征完全来自 TabLDM。\n要结清需: 装上 `tabldm` 后跑 "
                         "`V47_SH_PACKAGES=v35 python3 examples/run_v47_scale_hetero_heldout.py`。\n"
                         "在此之前不能写 clean —— 本包的增益结论原本就只来自 6 个样本的 "
                         "in-sample 评估, 无任何样本外依据。"),
            }
        else:
            claim = man.get("claim", {}).get("asserts", "(未填)")
            man["leakage_audit"] = {
                "status": row["status"], "audits": [f"{key}.0"], "audited_by": "47.0",
                "note": build_note(claim, row),
            }
        path.write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{man['package']:<28} {before} -> {man['leakage_audit']['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
