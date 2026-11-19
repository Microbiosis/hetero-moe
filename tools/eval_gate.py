#!/usr/bin/env python3
"""tools/eval_gate.py — 统一防泄漏下游任务评测门禁 CLI。

用法:
    python tools/eval_gate.py --self-check   # 运行内置端到端真异构分类 + LM PPL 门禁自检
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from hetero_moe import (
    DownstreamEvalGate,
    ExpertSpec,
    HeteroFusionLayer,
    IdentifiabilityGuard,
    SplitGuard,
)


def run_self_check(seed: int = 42) -> int:
    """运行快速、确定性、零泄漏的真异构多专家分类门禁自检（证明门禁能放行真收益、拦截假收益）。"""
    torch.manual_seed(seed)
    D, N_TOTAL, N_CLS = 32, 600, 4
    tr_idx, va_idx, te_idx = SplitGuard.make_splits(N_TOTAL, 0.6, 0.2, seed=seed)
    split_info = SplitGuard.verify(tr_idx, va_idx, te_idx)

    # 构造 2 个互补子域任务: 样本所属域由特征前 4 维隐式编码，每个单专家只擅长 1 个域
    domain = torch.randint(0, 2, (N_TOTAL,))
    y = torch.randint(0, N_CLS, (N_TOTAL,))
    W_dom0 = torch.randn(N_CLS, 12)
    W_dom1 = torch.randn(N_CLS, 12)
    X = torch.randn(N_TOTAL, D) * 0.25
    for i in range(N_TOTAL):
        d_i, c_i = int(domain[i]), int(y[i])
        X[i, 0:4] = (2.0 if d_i == 0 else -2.0) + 0.1 * torch.randn(4)
        if d_i == 0:
            X[i, 4:16] += 1.6 * W_dom0[c_i]
        else:
            X[i, 16:28] += 1.6 * W_dom1[c_i]

    # 两个不同宽度的单域冻结特征提取器（真正异构，各自只看一半子空间）
    class _DomainExpert(nn.Module):
        def __init__(self, sl: slice, hidden: int, out_d: int):
            super().__init__()
            self.sl = sl
            self.net = nn.Sequential(nn.Linear(sl.stop - sl.start, hidden), nn.GELU(), nn.Linear(hidden, out_d))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x[..., self.sl])

    exp0 = _DomainExpert(slice(4, 16), 48, 20)
    exp1 = _DomainExpert(slice(16, 28), 36, 28)

    # 在训练集上各自预训练单域专家再冻结（模拟真实冻结异构底座）
    for dom_id, mod, out_d in ((0, exp0, 20), (1, exp1, 28)):
        head_tmp = nn.Linear(out_d, N_CLS)
        opt_tmp = torch.optim.Adam(list(mod.parameters()) + list(head_tmp.parameters()), lr=1e-2)
        mask = torch.tensor([i for i in tr_idx if int(domain[i]) == dom_id], dtype=torch.long)
        for _ in range(120):
            opt_tmp.zero_grad()
            loss = F.cross_entropy(head_tmp(mod(X[mask])), y[mask])
            loss.backward()
            opt_tmp.step()
        for p in mod.parameters():
            p.requires_grad_(False)

    # 训练单底座线性探针基线 (仅在 train 上训练，val 上选模，test 上报告)
    single_logits = {}
    for name, mod, out_d in (("expert_dom0", exp0, 20), ("expert_dom1", exp1, 28)):
        probe = nn.Linear(out_d, N_CLS)
        opt = torch.optim.Adam(probe.parameters(), lr=2e-2)
        best_va, best_sd = -1.0, None
        with torch.no_grad():
            f_tr, f_va, f_te = mod(X[tr_idx]), mod(X[va_idx]), mod(X[te_idx])
        for _ in range(120):
            opt.zero_grad()
            F.cross_entropy(probe(f_tr), y[tr_idx]).backward()
            opt.step()
            with torch.no_grad():
                va_acc = float((probe(f_va).argmax(-1) == y[va_idx]).float().mean())
                if va_acc > best_va:
                    best_va = va_acc
                    best_sd = {k: v.clone() for k, v in probe.state_dict().items()}
        probe.load_state_dict(best_sd)
        with torch.no_grad():
            single_logits[name] = probe(f_te)

    # 训练 v5 真异构融合层 + 分类头（train 训练，val 选模，test 零泄漏评估）
    specs = [
        ExpertSpec(name="dom0_gelu", arch="custom", d_in=D, d_out=20, freeze_base=True, lora_rank=4, module=exp0),
        ExpertSpec(name="dom1_gelu", arch="custom", d_in=D, d_out=28, freeze_base=True, lora_rank=4, module=exp1),
    ]
    layer = HeteroFusionLayer(
        d_model=D, d_ff=32, num_experts=2, top_k=1,
        quant_bits=8, gate_rank=4, expert_specs=specs, sparse_dispatch=True,
    )
    cls_head = nn.Linear(D, N_CLS)
    opt = torch.optim.Adam(list(layer.trainable_parameters) + list(cls_head.parameters()), lr=1.5e-2)
    best_va, best_sd_l, best_sd_h = -1.0, None, None
    for _ in range(160):
        opt.zero_grad()
        out, _, _ = layer(X[tr_idx].unsqueeze(1))
        loss = F.cross_entropy(cls_head(out.squeeze(1)), y[tr_idx])
        loss.backward()
        opt.step()
        with torch.no_grad():
            va_out, _, _ = layer(X[va_idx].unsqueeze(1))
            va_acc = float((cls_head(va_out.squeeze(1)).argmax(-1) == y[va_idx]).float().mean())
            if va_acc > best_va:
                best_va = va_acc
                best_sd_l = {k: v.clone() for k, v in layer.state_dict().items()}
                best_sd_h = {k: v.clone() for k, v in cls_head.state_dict().items()}
    layer.load_state_dict(best_sd_l)
    cls_head.load_state_dict(best_sd_h)
    with torch.no_grad():
        te_out, _, _ = layer(X[te_idx].unsqueeze(1))
        fusion_logits_te = cls_head(te_out.squeeze(1))
        # 阴性对照: 置换特征行
        perm_te = te_idx[torch.randperm(len(te_idx)).numpy()]
        neg_out, _, _ = layer(X[perm_te].unsqueeze(1))
        neg_logits_te = cls_head(neg_out.squeeze(1))

    gate = DownstreamEvalGate(n_boot=1000, seed=seed)
    report = gate.evaluate_classification(
        y_true_test=y[te_idx],
        fusion_logits_test=fusion_logits_te,
        single_logits_test=single_logits,
        y_train_for_majority=y[tr_idx],
        negative_control_logits_test=neg_logits_te,
        split_info=split_info,
        identifiability_info=IdentifiabilityGuard.check(len(tr_idx), N_CLS * D, max_dof_per_sample=1.0),
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Hetero-MoE 下游防泄漏评测门禁")
    parser.add_argument("--self-check", action="store_true", help="运行真异构互补基准门禁自检")
    args = parser.parse_args()
    if args.self_check:
        return run_self_check()
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
