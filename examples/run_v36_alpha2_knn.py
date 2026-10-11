"""V36.0 — α2 跨模态验证 (大模型容器 + TabLDM 逐样本入口, k-NN 评估)。

> 归属路线: 大模型容器 + 异质入口 (由外部快照 v9.0-α2 重编号)
> 重编号对照: 快照 v9.0-α2 → 本项目 v36.0
> 子包: v36_alpha2_knn (机制实现), 本脚本只负责编排

相对 v35 的关键修复:
    1. TabLDM 表征不再 `mean(dim=(0,1))` 折叠成 [512] 常向量 (无信息)
    2. prefix 不再 `mean(dim=1, keepdim=True)` 塌缩成 [B,1,2560] 常数
    3. 评估改用无参数 k-NN (干净避免 1.3M 线性探针对 12 样本的硬记)
    4. 对照改用同结构同维度的随机张量 (而非 plain avg), 公平

任务: 跨模态 — 文本(无关) + 表格特征 → TabLDM 表格分类 (10类).
评估: 1-NN leave-one-out 准确率.

用法: bash run.sh v36
"""
import sys, os, time, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import statistics

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from research.scale_heterogeneous.v36_alpha2_knn import (
    MSA_DIR,
    get_tabldm_repr, load_big_container, encode_baseline_text,
    knn_acc,
    D_BIG, B_DEFAULT, S_DEFAULT,
)

N_SEEDS = int(os.environ.get("V36_SEEDS", "5"))
N_BIG_LAYERS = int(os.environ.get("V36_BIG_LAYERS", "1"))


def run_one(seed, big_model, c_struct, y_table, big_tok, texts):
    """单 seed: 容器预计算 h_base, 拼接 [h_base, c_struct] vs [h_base, rand] 跑 k-NN."""
    torch.manual_seed(seed)
    ids = big_tok(texts, padding="max_length", truncation=True,
                  max_length=S_DEFAULT, return_tensors="pt").input_ids  # [B, S]
    with torch.no_grad():
        h_base = encode_baseline_text(big_model, big_tok, texts, s=S_DEFAULT)  # [B, 2560]

    rand_struct = torch.randn(B_DEFAULT, c_struct.shape[-1])
    h_n = F.normalize(h_base, dim=1)
    c_n = F.normalize(c_struct, dim=1)
    r_n = F.normalize(rand_struct, dim=1)
    X_v9 = torch.cat([h_n, c_n], dim=1)
    X_ctl = torch.cat([h_n, r_n], dim=1)
    acc_v9 = knn_acc(X_v9, y_table)
    acc_ctl = knn_acc(X_ctl, y_table)
    # 公平基线: ctl=0 时以 1/B 作为最低合理基线 (1/B = 1/12)
    baseline = max(acc_ctl, 1.0 / B_DEFAULT)
    gain = (acc_v9 - acc_ctl) / baseline * 100
    return acc_ctl, acc_v9, gain


def main():
    print(f"{'='*80}")
    print("V36.0 α2 跨模态验证 — 大模型容器 + TabLDM 表格入口 (k-NN)")
    print(f"{'='*80}")
    print("任务: 文本(无关) + 表格特征 → TabLDM 表格分类 (10类)")
    print("评估: 无参数 k-NN (concat(h_base, c) vs concat(h_base, rand))")
    print()

    t0 = time.time()
    c_struct, y_table = get_tabldm_repr(n=B_DEFAULT, n_classes=10, seed=42)
    gc.collect()
    big_tok = AutoTokenizer.from_pretrained(MSA_DIR)
    big_model = load_big_container(n_layers=N_BIG_LAYERS)

    texts = ["cat dog bird", "animal pet wild", "feline canine fowl",
             "quick brown fox", "lazy dog sleeps", "birds fly high",
             "fish swim water", "tree root soil", "car wheel road",
             "book read page", "sun moon star", "rain wind cloud"]
    print(f"  容器就绪 ({time.time()-t0:.1f}s)\n")

    print(f"{'seed':>5} | {'ctl_acc':>8} | {'v9_acc':>8} | {'gain':>8} | {'PASS':>4}")
    print("-" * 60)
    all_pass = True
    acs, avs, gs = [], [], []
    for seed in range(N_SEEDS):
        ac, av, g = run_one(seed, big_model, c_struct, y_table, big_tok, texts)
        acs.append(ac); avs.append(av); gs.append(g)
        ok = av > ac
        all_pass = all_pass and ok
        print(f"{seed:>5} | {ac:>8.3f} | {av:>8.3f} | {g:>+7.1f}% | {'✅' if ok else '❌':>4}")
    print("-" * 60)
    print(f"\nk-NN 准确率均值: ctl={statistics.mean(acs):.3f}  v9={statistics.mean(avs):.3f}")
    print(f"增益均值 (v9 vs ctl): {statistics.mean(gs):+.1f}%  范围: [{min(gs):+.1f}, {max(gs):+.1f}]")
    print(f"{'='*80}")
    if all_pass:
        print("✅ V36.0 α2 跨模态成立: TabLDM 表格入口信息真正有用 (k-NN 打败随机对照)")
    else:
        n = sum(1 for g in gs if g > 0)
        print(f"⚠️ v9 优于 ctl 的种子: {n}/{N_SEEDS}")


if __name__ == "__main__":
    main()