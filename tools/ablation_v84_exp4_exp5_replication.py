#!/usr/bin/env python3
"""V85.0 消融复现脚本 —— 检验「v5 融合架构突破纯冻结 β 墙」是否归因于融合机制。

背景
----
`research/weight_fusion/v84_inherent_flaw_remediation/audit_remediation.py` 的 Exp4/Exp5
声称: 纯冻结专家组合受 β 墙锁死 (准确率上限 ~2/3), 而 hetero_moe v5.0 的低秩向量门控 +
原生 LoRA + 跨层梯度「打破 β 墙」(best_single 0.5222 -> v4 0.8667 -> v5 1.0000)。

本脚本只做一件事: 在**完全相同的数据生成、切分、专家底座与训练协议**下, 把
「专家个数 / 专家是否可训练 / 门控 / LoRA / 跨层梯度」逐个换掉, 看 v5 的能力到底来自哪。

两组实验
--------
Exp4 分类门禁 (与 audit_remediation.exp4_downstream_classification_beta_wall 同协议)
Exp5 序列语言建模门禁 (与 audit_remediation.exp5_downstream_lm_perplexity_gate 同协议)

用法
----
    cd <repo root> && python tools/ablation_v85_exp4_exp5.py

输出: JSON 数组, 每个元素 {"arm": ..., "test_acc"/"test_ppl": ..., "trainable_params": ...}
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hetero_moe import ExpertSpec, HeteroFusionLayer, SplitGuard

SEED = 42  # 与 audit_remediation.py 同值


# --------------------------------------------------------------------------- Exp4
def exp4_arms() -> List[Dict]:
    D, N_TOTAL, N_CLS = 36, 900, 4
    torch.manual_seed(SEED)
    tr_idx, va_idx, te_idx = SplitGuard.make_splits(N_TOTAL, 0.60, 0.20, seed=SEED)

    domain = torch.randint(0, 3, (N_TOTAL,))
    y = torch.randint(0, N_CLS, (N_TOTAL,))
    W0, W1, W2 = (torch.randn(N_CLS, 10) for _ in range(3))
    X = torch.randn(N_TOTAL, D) * 0.20
    for i in range(N_TOTAL):
        d_i, c_i = int(domain[i]), int(y[i])
        X[i, 0:6] = 0.0
        X[i, 2 * d_i: 2 * d_i + 2] = 2.5
        if d_i == 0:
            X[i, 6:16] += 1.8 * W0[c_i]
        elif d_i == 1:
            X[i, 16:26] += 1.8 * W1[c_i]
        else:
            # 域 2 (β 墙): 信息仅编码在 [26:36] 的符号交互中, 两个冻结切片专家都看不见
            s_sign = 1.0 if (i % 2 == 0) else -1.0
            X[i, 26:36] += s_sign * 1.9 * W2[c_i]
            X[i, 4] = 2.5 * s_sign

    class _SliceExpert(nn.Module):
        def __init__(self, sl: slice, hidden: int, out_d: int, act: str):
            super().__init__()
            self.sl, self.act = sl, act
            self.fc1 = nn.Linear(sl.stop - sl.start, hidden)
            self.fc2 = nn.Linear(hidden, out_d)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.fc1(x[..., self.sl])
            h = F.gelu(h) if self.act == "gelu" else F.silu(h)
            return self.fc2(h)

    exp_a = _SliceExpert(slice(6, 16), 40, 20, "gelu")
    exp_b = _SliceExpert(slice(16, 26), 48, 28, "silu")
    for dom_id, mod, out_d in ((0, exp_a, 20), (1, exp_b, 28)):
        tmp_h = nn.Linear(out_d, N_CLS)
        opt = torch.optim.Adam(list(mod.parameters()) + list(tmp_h.parameters()), lr=1.5e-2)
        sub = torch.tensor([i for i in tr_idx if int(domain[i]) == dom_id], dtype=torch.long)
        for _ in range(120):
            opt.zero_grad()
            F.cross_entropy(tmp_h(mod(X[sub])), y[sub]).backward()
            opt.step()
        for p in mod.parameters():
            p.requires_grad_(False)

    def train_arm(specs, num_experts, gate_rank, lora_rank, detach_input):
        torch.manual_seed(SEED)
        layer = HeteroFusionLayer(
            d_model=D, d_ff=32, num_experts=num_experts, top_k=1,
            quant_bits=8, gate_rank=gate_rank, detach_input=detach_input,
            expert_specs=specs, sparse_dispatch=True,
        )
        head = nn.Linear(D, N_CLS)
        opt = torch.optim.Adam(list(layer.trainable_parameters) + list(head.parameters()), lr=1.5e-2)
        best_va, best_sl, best_sh = -1.0, None, None
        for _ in range(260):
            opt.zero_grad()
            out, _, _ = layer(X[tr_idx].unsqueeze(1))
            F.cross_entropy(head(out.squeeze(1)), y[tr_idx]).backward()
            opt.step()
            with torch.no_grad():
                vo, _, _ = layer(X[va_idx].unsqueeze(1))
                va = float((head(vo.squeeze(1)).argmax(-1) == y[va_idx]).float().mean())
                if va > best_va:
                    best_va = va
                    best_sl = {k: v.clone() for k, v in layer.state_dict().items()}
                    best_sh = {k: v.clone() for k, v in head.state_dict().items()}
        layer.load_state_dict(best_sl)
        head.load_state_dict(best_sh)
        with torch.no_grad():
            te_log = head(layer(X[te_idx].unsqueeze(1))[0].squeeze(1))
        return {
            "test_acc": float((te_log.argmax(-1) == y[te_idx]).float().mean()),
            "best_val_acc": best_va,
            "trainable_params": sum(p.numel() for p in layer.trainable_parameters),
        }

    def sliced(lora):
        return [
            ExpertSpec("exp_a", "custom", d_in=D, d_out=20, freeze_base=True, lora_rank=lora, module=exp_a),
            ExpertSpec("exp_b", "custom", d_in=D, d_out=28, freeze_base=True, lora_rank=lora, module=exp_b),
        ]

    solo_train = [ExpertSpec("solo", "swiglu", d_in=D, d_ff=64, d_out=32, freeze_base=False, lora_rank=0)]
    solo_frozen = [ExpertSpec("solo_frozen", "swiglu", d_in=D, d_ff=64, d_out=32, freeze_base=True, lora_rank=0)]
    solo_lora = [ExpertSpec("solo_lora", "swiglu", d_in=D, d_ff=64, d_out=32, freeze_base=True, lora_rank=6)]

    out = []
    out.append({"arm": "v5_upstream_2exp_gate8_lora6", **train_arm(sliced(6), 2, 8, 6, False)})
    out.append({"arm": "v4_frozen_2exp_gate0_lora0_detach", **train_arm(sliced(0), 2, 0, 0, True)})
    out.append({"arm": "single_trainable_fullinput_nofusion", **train_arm(solo_train, 1, 0, 0, False)})
    out.append({"arm": "single_frozen_fullinput_headonly", **train_arm(solo_frozen, 1, 0, 0, False)})
    out.append({"arm": "single_frozen_fullinput_lora6", **train_arm(solo_lora, 1, 0, 6, False)})
    return out


# --------------------------------------------------------------------------- Exp5
def exp5_arms() -> List[Dict]:
    V, S, N_SEQ, D = 24, 16, 480, 32
    torch.manual_seed(SEED)
    tr_idx, va_idx, te_idx = SplitGuard.make_splits(N_SEQ, 0.60, 0.20, seed=SEED)

    seqs = torch.zeros(N_SEQ, S + 1, dtype=torch.long)
    domains = torch.zeros(N_SEQ, dtype=torch.long)
    for i in range(N_SEQ):
        dom = i % 2
        domains[i] = dom
        tok = int(torch.randint(0, V // 2, (1,)).item()) + (0 if dom == 0 else V // 2)
        seqs[i, 0] = tok
        for t in range(1, S + 1):
            prev = int(seqs[i, t - 1].item())
            if torch.rand(1).item() < 0.08:
                nxt = int(torch.randint(0, V, (1,)).item())
            elif dom == 0:
                nxt = (prev + 3) % V
            else:
                nxt = (2 * prev + 5) % V
            seqs[i, t] = nxt

    inp_tok, tgt_tok = seqs[:, :-1], seqs[:, 1:]
    emb_tok, emb_ctx = nn.Embedding(V, D), nn.Embedding(V, D)
    for p in (*emb_tok.parameters(), *emb_ctx.parameters()):
        p.requires_grad_(False)
    with torch.no_grad():
        X_all = emb_tok(inp_tok) + 0.7 * emb_ctx(inp_tok[:, :1]).expand(-1, S, -1)

    class _LMExpert(nn.Module):
        def __init__(self, d_in: int, h: int, d_out: int, use_silu: bool):
            super().__init__()
            self.f1 = nn.Linear(d_in, h)
            self.f2 = nn.Linear(h, d_out)
            self.use_silu = use_silu

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = F.silu(self.f1(x)) if self.use_silu else F.gelu(self.f1(x))
            return self.f2(h)

    e0 = _LMExpert(D, 48, 24, use_silu=False)
    e1 = _LMExpert(D, 56, 36, use_silu=True)
    for dom_id, mod, out_d in ((0, e0, 24), (1, e1, 36)):
        h_tmp = nn.Linear(out_d, V)
        opt = torch.optim.Adam(list(mod.parameters()) + list(h_tmp.parameters()), lr=2e-2)
        sub = torch.tensor([i for i in tr_idx if int(domains[i]) == dom_id], dtype=torch.long)
        for _ in range(150):
            opt.zero_grad()
            logits = h_tmp(mod(X_all[sub]))
            F.cross_entropy(logits.reshape(-1, V), tgt_tok[sub].reshape(-1)).backward()
            opt.step()
        for p in mod.parameters():
            p.requires_grad_(False)

    def train_arm(specs, num_experts, gate_rank, detach_input):
        torch.manual_seed(SEED)
        layer = HeteroFusionLayer(
            d_model=D, d_ff=32, num_experts=num_experts, top_k=1,
            quant_bits=8, gate_rank=gate_rank, detach_input=detach_input,
            expert_specs=specs, sparse_dispatch=True,
        )
        lm_head = nn.Linear(D, V)
        opt = torch.optim.Adam(list(layer.trainable_parameters) + list(lm_head.parameters()), lr=1.5e-2)
        best_va, best_sl, best_sh = float("inf"), None, None
        for _ in range(200):
            opt.zero_grad()
            y_tr, _, _ = layer(X_all[tr_idx])
            F.cross_entropy(lm_head(y_tr).reshape(-1, V), tgt_tok[tr_idx].reshape(-1)).backward()
            opt.step()
            with torch.no_grad():
                y_va, _, _ = layer(X_all[va_idx])
                va_nll = float(F.cross_entropy(lm_head(y_va).reshape(-1, V), tgt_tok[va_idx].reshape(-1)).item())
                if va_nll < best_va:
                    best_va = va_nll
                    best_sl = {k: v.clone() for k, v in layer.state_dict().items()}
                    best_sh = {k: v.clone() for k, v in lm_head.state_dict().items()}
        layer.load_state_dict(best_sl)
        lm_head.load_state_dict(best_sh)
        with torch.no_grad():
            te = lm_head(layer(X_all[te_idx])[0])
            nll = float(F.cross_entropy(te.reshape(-1, V), tgt_tok[te_idx].reshape(-1)).item())
        return {
            "test_nll": nll,
            "test_ppl": math.exp(min(50.0, nll)),
            "best_val_nll": best_va,
            "trainable_params": sum(p.numel() for p in layer.trainable_parameters),
        }

    specs5 = [
        ExpertSpec("lm_dom0", "custom", d_in=D, d_out=24, freeze_base=True, lora_rank=4, module=e0),
        ExpertSpec("lm_dom1", "custom", d_in=D, d_out=36, freeze_base=True, lora_rank=4, module=e1),
    ]
    solo_train = [ExpertSpec("solo_lm", "swiglu", d_in=D, d_ff=64, d_out=32, freeze_base=False, lora_rank=0)]
    solo_lora = [ExpertSpec("solo_lm_lora", "swiglu", d_in=D, d_ff=64, d_out=32, freeze_base=True, lora_rank=4)]

    out = []
    out.append({"arm": "v5_upstream_2lmexp_gate6_lora4", **train_arm(specs5, 2, 6, False)})
    out.append({"arm": "single_trainable_fullinput_nofusion", **train_arm(solo_train, 1, 0, False)})
    out.append({"arm": "single_frozen_fullinput_lora4", **train_arm(solo_lora, 1, 0, False)})
    return out


def main() -> None:
    print(json.dumps({
        "exp4_classification": exp4_arms(),
        "exp5_language_model": exp5_arms(),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
