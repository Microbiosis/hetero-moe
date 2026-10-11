"""V35.0-α — 大模型容器 + 异质入口潜意识层 — 端到端入口。

> 归属路线: 大模型容器 + 异质入口 (原 V6 衍生路线, 由外部快照 v9.0-α 重编号而来)
> 重编号对照: 快照 v9.0-α → 本项目 v35.0-α
> 子包: v35_container_entries (机制实现), 本脚本只负责训练/评估编排
> 结果性质: 负结果 (边界已界定), 见 docs/V35_0_SPEC.md

核心命题: 大模型作为容器(主干), 不同类型小模型作为"原生入口"(潜意识感官层),
各自处理擅长的输入类型(表格/文本/文本对), 产出表征经 P_m 投影后作为 prefix
注入大模型容器内部, 使融合优于无注入 baseline。

入口配置 (3 种不同类型小模型, 各处理原生输入):
  TabLDM 入口:    表格输入 → hook 取 representations → c_struct (D=512)
  embedding 入口: 文本     → 浅层编码                → c_sem   (D=1024)
  reranker 入口:  文本对   → 浅层编码                → c_rank  (D=1024)

用法: bash run.sh v35  (需 tabldm 可导入)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from research.scale_heterogeneous.v35_container_entries import (
    MSA_DIR, EMB_DIR, RER_DIR,
    load_backbone, load_big_container, big_forward,
    ProjHead,
)
# 2026-10-10 修正: get_tabldm_repr 由 _primitives.scale_hetero_loaders 提供(v36 转出),
# v35 自己导出的是 get_tabular_repr —— 两者不是同一个函数
from research._primitives.scale_hetero_loaders import get_tabldm_repr

D_SHARED = 256
D_BIG = 2560
B, S = 18, 16
STEPS = int(os.environ.get("V35_STEPS", "100"))
LR = 1e-2
N_SEEDS = int(os.environ.get("V35_SEEDS", "5"))


# =====================================================================
# 实验入口注入
# =====================================================================

def run_one(seed, big_model, c_struct, c_sem, c_rank, big_tok):
    torch.manual_seed(seed)
    B6 = 6
    texts = ["cat dog bird", "animal pet wild", "feline canine fowl",
             "quick brown fox", "lazy dog sleeps", "birds fly high"]
    ids = big_tok(texts, padding="max_length", truncation=True,
                  max_length=S, return_tensors="pt").input_ids  # [6, S]
    targets = torch.zeros(B6, D_SHARED)
    for i in range(B6):
        cls = i // 2
        ds = cls * (D_SHARED // 3); de = (cls + 1) * (D_SHARED // 3)
        targets[i, ds:de] = 1.0

    c_sem_m = c_sem.mean(0).detach()
    c_rank_m = c_rank.mean(0).detach()

    # --- baseline: 可学习随机 prior (非入口), 训 prior+ProjHead (公平对比) ---
    rand_prior = nn.Parameter(torch.randn(3, D_BIG) * 0.02)
    head_b = ProjHead(D_BIG, D_SHARED)
    opt_b = torch.optim.AdamW([rand_prior] + list(head_b.parameters()), lr=LR)
    for step in range(STEPS):
        opt_b.zero_grad(set_to_none=True)
        pref = rand_prior.unsqueeze(0).expand(B6, -1, -1)
        h_base = big_forward(big_model, ids, prefix=pref)
        h_tok = h_base.mean(dim=1)
        loss = F.mse_loss(head_b(h_tok), targets)
        loss.backward(); opt_b.step()
    with torch.no_grad():
        pref = rand_prior.unsqueeze(0).expand(B6, -1, -1)
        h_base = big_forward(big_model, ids, prefix=pref)
        h_tok = h_base.mean(dim=1)
        loss_base = F.mse_loss(head_b(h_tok), targets).item()

    # --- v35: prefix 直接可学习 [3,2560], 初始化自入口投影 (参数少, 好收敛) ---
    with torch.no_grad():
        Ps = torch.empty(D_BIG, c_struct.shape[-1]); nn.init.orthogonal_(Ps)
        Pe = torch.empty(D_BIG, c_sem_m.shape[-1]); nn.init.orthogonal_(Pe)
        Pr = torch.empty(D_BIG, c_rank_m.shape[-1]); nn.init.orthogonal_(Pr)
        init = torch.stack([F.linear(c_struct, Ps), F.linear(c_sem_m, Pe), F.linear(c_rank_m, Pr)]) * 0.02
    prefix = nn.Parameter(init.clone())  # [3, 2560] 可学习, 入口方向+小幅度(与rand_prior同幅度)
    head_v = ProjHead(D_BIG, D_SHARED)
    head_v.load_state_dict(head_b.state_dict())  # 同初始化公平对比
    opt_v = torch.optim.AdamW([prefix] + list(head_v.parameters()), lr=LR)
    for step in range(STEPS):
        opt_v.zero_grad(set_to_none=True)
        pref = prefix.unsqueeze(0).expand(B6, -1, -1)  # [6, 3, 2560]
        h_inj = big_forward(big_model, ids, prefix=pref)  # [6, S, 2560]
        h_tok = h_inj.mean(dim=1)  # [6, 2560]
        loss = F.mse_loss(head_v(h_tok), targets)
        loss.backward(); opt_v.step()
    with torch.no_grad():
        pref = prefix.unsqueeze(0).expand(B6, -1, -1)
        h_inj = big_forward(big_model, ids, prefix=pref)
        h_tok = h_inj.mean(dim=1)
        loss_v35 = F.mse_loss(head_v(h_tok), targets).item()

    gain = (loss_base - loss_v35) / max(loss_base, 1e-9) * 100
    return loss_base, loss_v35, gain


def main():
    print(f"{'=' * 80}")
    print("V35.0-α 大模型容器 + 异质入口潜意识层 (TabLDM+embedding+reranker → prefix 注入)")
    print(f"{'=' * 80}")
    print("入口: TabLDM(表格,D=512) + embedding(文本,D=1024) + reranker(文本,D=1024)")
    print(f"容器: MSA-4B 浅层(D={D_BIG}), prefix 注入 inputs_embeds")
    print("对比: 无注入 baseline vs 3异质入口注入 (v35)")
    print()

    # 串行加载入口 (编码后释放)
    c_struct = get_tabldm_repr()
    c_sem, D_sem = load_backbone(EMB_DIR, "embedding-0.6b", S)
    c_rank, D_rank = load_backbone(RER_DIR, "reranker-0.6b", S)
    gc.collect()
    big_tok = AutoTokenizer.from_pretrained(MSA_DIR)
    big_model = load_big_container()

    print(f"\n{'seed':>5} | {'base':>9} | {'v35_inj':>9} | {'gain':>8} | {'PASS':>4}")
    print("-" * 60)
    all_pass = True
    gs = []
    for seed in range(N_SEEDS):
        lb, li, g = run_one(seed, big_model, c_struct, c_sem, c_rank, big_tok)
        gs.append(g)
        ok = li < lb
        all_pass = all_pass and ok
        print(f"{seed:>5} | {lb:>9.4f} | {li:>9.4f} | {g:>+7.1f}% | {'✅' if ok else '❌':>4}")
    print("-" * 60)
    import statistics
    print(f"\n增益均值: {statistics.mean(gs):+.1f}%  范围: [{min(gs):+.1f}, {max(gs):+.1f}]")
    print(f"{'=' * 80}")
    if all_pass:
        print("✅ V35.0-α 异质入口注入成立: 3种类型入口 prefix 注入大模型容器优于无注入")
    else:
        n = sum(1 for g in gs if g > 0)
        print(f"⚠️ v35 优于 baseline 的种子: {n}/{N_SEEDS}")


if __name__ == "__main__":
    main()
