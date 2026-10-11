"""V39.0 — γ2 真实数据跨模态 (加性 vs 交互融合框架, 双数据集)。

> 归属路线: 大模型容器 + 异质入口 (由外部快照 v9.0-γ2 重编号)
> 重编号对照: 快照 v9.0-γ2 → 本项目 v39.0
> 子包: v39_g2_interaction (机制实现), 本脚本只负责编排

γ2 相对 γ (v38.0-α) 的修正 —— 架构层:
    γ  用加性头: logits = Linear(h_base + P(c))
        → 文本侧与表格侧贡献**线性求和**, 无法表达 XOR 交互 (经典不可分解),
          所有臂被压在同一退化天花板 (~0.77), 由此得出"三臂打平"的边界结论。
    γ2 用交互头: logits = MLP(concat[h_base, P(c)])
        → 可表达跨模态交互, 能在真实数据上真正检验 "自训练 D > 被动 C"。

五臂对照 (同结构同参数量, 只差入口信息来源):
    A 纯文本(无入口)  / B 随机入口 / C 被动TabLDM(α2) / D 自训练(β) / E 先验+自训练

数据集: adult (人口普查) | shoppers (电商会话)
融合:   add (v38 旧框架) | concat (γ2 交互框架)

用法:
    bash run.sh v39                                  # 默认 adult + concat
    V39_DATASET=shoppers bash run.sh v39             # 第二数据集
    V39_FUSION=add python3 examples/run_v39_g2_interaction.py   # 复现旧框架
"""
import sys, os, time, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import statistics

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from research.scale_heterogeneous.v39_g2_interaction import (
    MSA_DIR,
    DATASETS,
    make_head, fuse, logits_of,
    load_big_container, EntryEncoder, knn_acc,
)
from research.scale_heterogeneous.v39_g2_interaction.paths import TABLDM_CKPT

N_TOTAL = int(os.environ.get("V39_N", "512"))
STEPS = int(os.environ.get("V39_STEPS", "500"))
N_SEEDS = int(os.environ.get("V39_SEEDS", "5"))
N_BIG_LAYERS = int(os.environ.get("V39_BIG_LAYERS", "1"))
DATASET = os.environ.get("V39_DATASET", "adult").lower()
FUSION = os.environ.get("V39_FUSION", "concat").lower()
D_ENTRY = 512
N_CLS = 2
D_BIG = 2560
S_LEN = 32


def container_text_repr(big_model, tok, texts):
    """真实自然语言 → MSA-4B 浅层均值池化 (携带文本侧条件信息)。"""
    ids = tok(texts, padding="max_length", truncation=True, max_length=S_LEN,
              return_tensors="pt").input_ids
    with torch.no_grad():
        return big_model(ids).last_hidden_state.float().mean(dim=1)  # [N, D_BIG]


def tabldm_passive(X_table):
    """α2 对照: TabLDM 被动中间表征, 逐样本冻结。"""
    from tabldm import TabLDMClassifier
    clf = TabLDMClassifier(model_path=TABLDM_CKPT, device="cpu")
    d = X_table.shape[1]
    rng = np.random.RandomState(42)
    clf.fit(rng.randn(12, d).astype(np.float32), rng.randint(0, 10, 12))
    m = clf.model_
    reps = []
    h = m.icl_predictor.register_forward_hook(
        lambda mod, inp, out: reps.append(inp[0].detach() if isinstance(inp, tuple) else inp.detach()))
    clf.predict_proba(X_table.numpy())
    h.remove()
    r = reps[0]
    n = X_table.shape[0]
    rep_q = r[:, -n:, :].mean(dim=0).float()  # [N, 512]
    del clf, m
    gc.collect()
    return rep_q


def train_arm(h_base, X, y, c_frozen, tr_idx, te_idx,
              steps=STEPS, seed=0, ref=None, distill=100, mode="concat"):
    """训练单臂, 返回 (test_acc, knn).

    ref 给定时: 先 K 步把 E 蒸馏到 ref (TabLDM 先验), 再任务训练 (E 臂: 先验+自训练)。
    mode: add (v38 旧加性框架) | concat (γ2 交互框架)
    """
    torch.manual_seed(seed)
    E = EntryEncoder(X.shape[1], D_ENTRY)
    P = nn.Linear(D_ENTRY, D_BIG)
    head = make_head(mode, D_BIG, N_CLS)
    params = list(P.parameters()) + list(head.parameters()) + (
        list(E.parameters()) if c_frozen is None else [])
    opt = torch.optim.Adam(params, lr=1e-3)
    # 蒸馏预热: E 拉到 TabLDM 表征附近 (获得预训练先验)
    if ref is not None and c_frozen is None:
        for _ in range(distill):
            b = tr_idx[torch.randperm(len(tr_idx))[:64]]
            loss = nn.functional.mse_loss(E(X[b]), ref[b])
            opt.zero_grad(); loss.backward(); opt.step()
    for _ in range(steps):
        b = tr_idx[torch.randperm(len(tr_idx))[:64]]
        cin = P(c_frozen[b] if c_frozen is not None else E(X[b]))
        loss = nn.functional.cross_entropy(logits_of(head, h_base[b], cin, mode), y[b])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        cin_all = P(c_frozen if c_frozen is not None else E(X))
        acc = (logits_of(head, h_base[te_idx], cin_all[te_idx], mode).argmax(1) == y[te_idx]) \
            .float().mean().item()
        knn = knn_acc(h_base + cin_all, y)
    return acc, knn


def main():
    if DATASET not in DATASETS:
        raise SystemExit(f"未知数据集 {DATASET!r}, 可选: {list(DATASETS)}")
    load_fn, ds_name = DATASETS[DATASET]

    print("=" * 80)
    print(f"V39.0 γ2 真实数据跨模态 — {ds_name} + XOR 标签 (加性 vs 交互框架)")
    print("=" * 80)
    print("标签: y = (文本侧条件) XOR (表格侧条件) → 单侧无法解出")
    print(f"配置: 数据集={ds_name}, 融合={FUSION}, N={N_TOTAL}, "
          f"{N_TOTAL-128}/128 留出, STEPS={STEPS}, seeds={N_SEEDS}")
    print()

    t0 = time.time()
    texts, X, y = load_fn(N_TOTAL, 42, decouple=True)
    print(f"  {ds_name} 数据: {len(texts)} 样本, 正样本率 {y.float().mean():.2%}")

    big_model = load_big_container(n_layers=N_BIG_LAYERS)
    tok = AutoTokenizer.from_pretrained(MSA_DIR)
    h_base = container_text_repr(big_model, tok, texts)
    del big_model; gc.collect()
    print(f"  文本容器表征 {tuple(h_base.shape)} ({time.time()-t0:.1f}s)")

    c_passive = tabldm_passive(X)
    print(f"  TabLDM 被动入口 {tuple(c_passive.shape)} ({time.time()-t0:.1f}s)\n")

    rng = np.random.RandomState(1234)
    perm = rng.permutation(N_TOTAL)
    tr_idx = torch.tensor(perm[:N_TOTAL - 128])
    te_idx = torch.tensor(perm[N_TOTAL - 128:])

    arms = ["A 纯文本(无入口)", "B 随机入口", "C 被动TabLDM(α2)", "D 自训练(β)", "E 先验+自训练"]
    results = {a: {"acc": [], "knn": []} for a in arms}

    for s in range(N_SEEDS):
        torch.manual_seed(s)
        c_rand = torch.randn(N_TOTAL, D_ENTRY)
        accs = [
            train_arm(h_base, X, y, torch.zeros(N_TOTAL, D_ENTRY), tr_idx, te_idx, seed=s, mode=FUSION),  # A
            train_arm(h_base, X, y, c_rand, tr_idx, te_idx, seed=s, mode=FUSION),                          # B
            train_arm(h_base, X, y, c_passive, tr_idx, te_idx, seed=s, mode=FUSION),                       # C
            train_arm(h_base, X, y, None, tr_idx, te_idx, seed=s, mode=FUSION),                            # D
            train_arm(h_base, X, y, None, tr_idx, te_idx, seed=s, ref=c_passive, mode=FUSION),             # E
        ]
        for arm, (a, k) in zip(arms, accs):
            results[arm]["acc"].append(a)
            results[arm]["knn"].append(k)

    print(f"{'臂':<22} {'test_acc':>10} {'kNN佐证':>10}")
    print("-" * 46)
    for arm in arms:
        print(f"{arm:<22} {statistics.mean(results[arm]['acc']):>10.3f} "
              f"{statistics.mean(results[arm]['knn']):>10.3f}")
    print("-" * 46)

    d = statistics.mean(results["D 自训练(β)"]["acc"])
    c = statistics.mean(results["C 被动TabLDM(α2)"]["acc"])
    e = statistics.mean(results["E 先验+自训练"]["acc"])
    a = statistics.mean(results["A 纯文本(无入口)"]["acc"])
    b = statistics.mean(results["B 随机入口"]["acc"])
    wins_d_c = sum(1 for s in range(N_SEEDS)
                   if results["D 自训练(β)"]["acc"][s] > results["C 被动TabLDM(α2)"]["acc"][s])
    wins_d_a = sum(1 for s in range(N_SEEDS)
                   if results["D 自训练(β)"]["acc"][s] > results["A 纯文本(无入口)"]["acc"][s])
    wins_e_c = sum(1 for s in range(N_SEEDS)
                   if results["E 先验+自训练"]["acc"][s] > results["C 被动TabLDM(α2)"]["acc"][s])
    print(f"\n层级: D自训练={d:.3f} | C被动={c:.3f} | E先验+自训练={e:.3f} | "
          f"A纯文本={a:.3f} | B随机={b:.3f}")
    print(f"D(自训练) 打败: 被动C {wins_d_c}/{N_SEEDS} | 纯文本A {wins_d_a}/{N_SEEDS}")
    print(f"E(先验+自训练) 打败: 被动C {wins_e_c}/{N_SEEDS}")
    print("=" * 80)
    if wins_d_c >= N_SEEDS - 1 and wins_d_a == N_SEEDS:
        print("✅ V39.0 γ2 成立: 交互框架下真实数据上'自训练入口最优' (超过被动先验)")
    elif wins_d_a == N_SEEDS and wins_d_c >= 2:
        print(f"⚠️ γ2 部分成立: D 打败纯文本 {wins_d_a}/{N_SEEDS}, 多数打败被动 C ({wins_d_c}/{N_SEEDS})")
    else:
        print(f"❌ γ2 未成立 (D>A {wins_d_a}/{N_SEEDS}) — 边界分析见 docs/V39_0_SPEC.md")


if __name__ == "__main__":
    main()
