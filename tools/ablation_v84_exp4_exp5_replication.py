#!/usr/bin/env python3
"""V84.0 消融复现脚本 —— 检验预训练先验多锚点/多形态设定下「v5 融合架构 vs 全部单专家无融合基线」。

背景
----
早期版本的 Exp4/Exp5 曾将类别向量直接线性写入原始输入 X (或仅含 48 个离散状态)，导致单专家无融合
模型靠残差直连或直接记忆即可达到与 v5 持平甚至更优的指标 (见 ABLATION_AUDIT.md 历史记录)。

修复后的 audit_remediation.py 引入了真实的**预训练专家先验不可替代性**:
  - Exp4: 64 个预训练锚点原型 (下游训练集仅含前 24 个锚点，测试集含未见锚点 24..63，仅预训练底座
    exp_a / exp_b 见过全部 64 个锚点) + 域 2 极性置换 β 墙;
  - Exp5: 24 词元 × 8 种表面形态变体 (下游训练集仅含变体 0..2，测试集含未见变体 3..7，仅预训练
    LM 专家 e0 / e1 见过全部 8 种形态变体)。

本脚本在完全相同的数据生成、三集切分与训练协议下，复现并对比全部消融臂，并将结果写入
`research/weight_fusion/v84_inherent_flaw_remediation/results/ablation_exp4_exp5_replication.json`。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.weight_fusion.v84_inherent_flaw_remediation.audit_remediation import (
    exp4_downstream_classification_beta_wall,
    exp5_downstream_lm_perplexity_gate,
)


def main() -> None:
    e4 = exp4_downstream_classification_beta_wall()
    e5 = exp5_downstream_lm_perplexity_gate()
    rep4 = e4["v5_gate_report"]["metrics"]
    singles4 = rep4["singles"]

    exp4_rows: List[Dict] = [
        {
            "arm": "v5_upstream_2exp_gate8_lora6",
            "test_acc": rep4["fusion"]["accuracy"],
            "cross_entropy": rep4["fusion"]["cross_entropy"],
        },
        {
            "arm": "v4_frozen_2exp_gate0_lora0_detach",
            "test_acc": e4["v4_legacy_frozen_test_acc"],
        },
    ]
    for k in (
        "single_trainable_fullinput_nofusion",
        "single_frozen_fullinput_headonly",
        "single_frozen_fullinput_lora6",
        "frozen_gelu_20d",
        "frozen_silu_28d",
    ):
        if k in singles4:
            exp4_rows.append({"arm": k, "test_acc": singles4[k]["accuracy"], "cross_entropy": singles4[k]["cross_entropy"]})

    rep5 = e5["metrics"]
    singles5 = rep5["singles"]
    exp5_rows: List[Dict] = [
        {
            "arm": "v5_upstream_2lmexp_gate6_lora4",
            "test_nll": rep5["fusion"]["nll"],
            "test_ppl": rep5["fusion"]["ppl"],
        },
    ]
    for k in (
        "single_trainable_fullinput_nofusion",
        "single_frozen_fullinput_lora4",
        "lm_expert_dom0_gelu",
        "lm_expert_dom1_silu",
    ):
        if k in singles5:
            exp5_rows.append({"arm": k, "test_nll": singles5[k]["nll"], "test_ppl": singles5[k]["ppl"]})

    out = {
        "exp4_classification": exp4_rows,
        "exp5_language_model": exp5_rows,
        "summary": {
            "exp4_v5_beats_all_single_nofusion": bool(
                rep4["fusion"]["accuracy"] > e4["v4_legacy_frozen_test_acc"] > rep4["best_single_acc"]["accuracy"]
            ),
            "exp5_v5_beats_all_single_nofusion": bool(
                rep5["fusion"]["ppl"] < rep5["best_single"]["ppl"]
            ),
        },
    }
    dst = ROOT / "research/weight_fusion/v84_inherent_flaw_remediation/results/ablation_exp4_exp5_replication.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
