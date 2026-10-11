"""V37.0 — β 自训练预测编码入口 (入口参与端到端梯度)。

> 归属路线: 大模型容器 + 异质入口 (由外部快照 v9.0-β 重编号)
> 重编号对照: 快照 v9.0-β → 本项目 v37.0
> 子包: v37_b_selftrained_entry (机制实现)

相对 v36 的进步:
  v36: 入口表征 c_struct = TabLDM 中间量 (冻结), 不为注入任务优化
  v37: 入口编码器 E(x_table) 可训练, 唯一监督 = 容器对目标的预测误差
       → E 必须自己发现"表格的哪些维度携带类别信息"

四臂对照 (同结构同参数量, 只差入口信息来源):
  A 无入口     h = h_base                  (训头)
  B 随机入口   c = randn 冻结              (训 P+头)
  C 被动入口   c = TabLDM 中间表征 冻结     (训 P+头)  ← v36 对照
  D 自训练入口 c = E(x_table) 可训练        (训 E+P+头) ← β 核心

任务: 表格类别 = 前三维符号模式 (10类); 文本随机 token 与表格无关.
评估: train/test 留出集准确率 (96/32) + k-NN leave-one-out 无参数佐证.

用法: bash run.sh v37
"""
import sys, os, time, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import statistics

import numpy as np
import torch
import torch.nn as nn

from research._primitives.scale_hetero_loaders import (
    EntryEncoder,
    gen_table_classification,
    tabldm_passive_repr,
    train_arm, knn_acc,
    load_big_container,
)

N_TOTAL = int(os.environ.get("V37_N", "128"))
STEPS = int(os.environ.get("V37_STEPS", "300"))
N_SEEDS = int(os.environ.get("V37_SEEDS", "5"))
SPLIT_SEED = int(os.environ.get("V37_SPLIT", "1234"))
D_ENTRY = 512
N_CLS = 10
D_BIG = 2560
S_LEN = 16


def main():
    print(f"{'='*80}")
    print("V37.0 β 自训练预测编码入口 — 入口参与梯度, 学到任务相关注入")
    print(f"{'='*80}")
    print(f"任务: 表格类别 = 前三维符号模式 (10类) | 文本随机 token (与表格无关)")
    print(f"配置: N={N_TOTAL} 留出 {N_TOTAL - 96}/96 训练, STEPS={STEPS}, seeds={N_SEEDS}")
    print("四臂: A 无入口 / B 随机入口 / C 被动TabLDM(α2) / D 自训练(β)")
    print()

    t0 = time.time()
    big_model = load_big_container()
    print(f"  容器就绪 ({time.time()-t0:.1f}s)")

    # 预计算各 seed 的容器表征 + 被动入口 (一次性, 训练循环复用)
    data = []
    for s in range(N_SEEDS):
        X, y = gen_table_classification(N_TOTAL, s, n_classes=N_CLS)
        # 随机 token 文本 → h_base (β 设计: 容器输入与表格任务无关)
        h_base = _random_token_base(big_model, N_TOTAL, S_LEN, s)
        c_passive, _ = tabldm_passive_repr(N_TOTAL, s, n_classes=N_CLS)
        data.append((s, X, y, h_base, c_passive))
        print(f"  seed {s}: 容器表征 {tuple(h_base.shape)} | 被动入口 {tuple(c_passive.shape)}")
    del big_model
    gc.collect()
    print(f"  数据准备完成 ({time.time()-t0:.1f}s)\n")

    rng = np.random.RandomState(SPLIT_SEED)
    perm = rng.permutation(N_TOTAL)
    tr_idx, te_idx = perm[:96], perm[96:]
    tr_idx, te_idx = torch.tensor(tr_idx), torch.tensor(te_idx)

    arms = ["A 无入口", "B 随机入口", "C 被动TabLDM(α2)", "D 自训练(β)"]
    results = {a: {"acc": [], "knn": []} for a in arms}

    for s, X, y, h_base, c_passive in data:
        torch.manual_seed(s)
        c_rand = torch.randn(N_TOTAL, D_ENTRY)
        accs = [
            train_arm(h_base, X, y, torch.zeros(N_TOTAL, D_ENTRY), tr_idx, te_idx,
                      D_ENTRY, D_BIG, N_CLS, STEPS, seed=s),  # A
            train_arm(h_base, X, y, c_rand, tr_idx, te_idx,
                      D_ENTRY, D_BIG, N_CLS, STEPS, seed=s),   # B
            train_arm(h_base, X, y, c_passive, tr_idx, te_idx,
                      D_ENTRY, D_BIG, N_CLS, STEPS, seed=s),  # C
            train_arm(h_base, X, y, None, tr_idx, te_idx,
                      D_ENTRY, D_BIG, N_CLS, STEPS, seed=s),   # D 自训练
        ]
        for arm, (a, k) in zip(arms, accs):
            results[arm]["acc"].append(a)
            results[arm]["knn"].append(k)

    print(f"{'臂':<22} {'test_acc':>12} {'kNN佐证':>12}")
    print("-" * 48)
    for arm in arms:
        m = statistics.mean(results[arm]["acc"])
        k = statistics.mean(results[arm]["knn"])
        print(f"{arm:<22} {m:>12.3f} {k:>12.3f}")
    print("-" * 48)

    d_acc = statistics.mean(results["D 自训练(β)"]["acc"])
    c_acc = statistics.mean(results["C 被动TabLDM(α2)"]["acc"])
    b_acc = statistics.mean(results["B 随机入口"]["acc"])
    a_acc = statistics.mean(results["A 无入口"]["acc"])
    wins = sum(1 for s in range(N_SEEDS)
               if results["D 自训练(β)"]["acc"][s] > results["C 被动TabLDM(α2)"]["acc"][s])
    print(f"\n层级关系: β={d_acc:.3f} vs α2={c_acc:.3f} vs 随机={b_acc:.3f} vs 无入口={a_acc:.3f}")
    print(f"β 打败 α2 的种子: {wins}/{N_SEEDS}")
    print(f"{'='*80}")
    if wins == N_SEEDS and d_acc > c_acc:
        print("✅ V37.0 β 成立: 自训练预测编码入口 > 被动中间量 (入口学到任务相关注入)")
    elif wins >= N_SEEDS - 1 and d_acc > c_acc:
        print(f"⚠️ β 大体成立 ({wins}/{N_SEEDS} 优于 α2), 略不稳定")
    else:
        print(f"❌ β 未成立 ({wins}/{N_SEEDS} 优于 α2) — 见 docs/V37_0_SPEC.md 边界分析")


def _random_token_base(big_model, n, s, seed):
    """随机 token 文本 → MSA-4B 浅层均值池化 (与 β 的 container_text_repr 等价)."""
    rng = np.random.RandomState(seed + 777)
    ids = torch.tensor(rng.randint(100, 20000, (n, s)), dtype=torch.long)
    with torch.no_grad():
        h = big_model(ids).last_hidden_state.float().mean(dim=1)
    return h


if __name__ == "__main__":
    main()