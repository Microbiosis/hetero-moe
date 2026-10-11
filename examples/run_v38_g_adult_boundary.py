"""V38.0 — γ 真实数据跨模态验证 (Adult XOR + 5 臂)。

> 归属路线: 大模型容器 + 异质入口 (由外部快照 v9.0-γ 重编号)
> 重编号对照: 快照 v9.0-γ → 本项目 v38.0
> 子包: v38_g_adult_boundary (机制实现, 含 UCI Adult 数据)

相对 v37 的扩展:
  v37: 合成符号规则任务 (类别 = 前三维模式, 简单可学)
  v38: UCI Adult 真实人口普查数据 + XOR 跨模态标签 (严格必需)

XOR 标签设计 (跨模态必需性):
  y = 1 iff (职业 ∈ 高薪集 {Exec-managerial, Prof-specialty, Tech-support} [只在文本侧])
        XOR (工时 ≥ 45 [只在表格侧])
  → AND 结构存在强泄漏 (hours<45 ⇒ y=0 确定性高), XOR 使单侧在条件独立时信息为零
  → 单侧天花板: 纯表格 ≈ 0.52, 纯文本 ≈ 0.68, 联合 ≈ 0.84

五臂对照 (同结构同参数量, 只差入口信息来源):
  A 纯文本(无入口)  h = h_base                    (训头)
  B 随机入口        c = randn 冻结                (训 P+头)
  C 被动TabLDM(α2)  c = TabLDM 中间表征 冻结      (训 P+头)  ← v36 对照
  D 自训练(β)      c = E(x_table) 可训练          (训 E+P+头) ← v37 对照
  E 先验+自训练    c = E 蒸馏初始化到 TabLDM, 再任务微调 (训 E+P+头) ← γ 核心

用法: bash run.sh v38
"""
import sys, os, time, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import statistics

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from research.scale_heterogeneous.v38_g_adult_boundary import (
    MSA_DIR,
    adult_loader,
    load_big_container,
    EntryEncoder,
    knn_acc,
)

N_TOTAL = int(os.environ.get("V38_N", "512"))
STEPS = int(os.environ.get("V38_STEPS", "300"))
N_SEEDS = int(os.environ.get("V38_SEEDS", "5"))
N_BIG_LAYERS = int(os.environ.get("V38_BIG_LAYERS", "1"))
D_ENTRY = 512
N_CLS = 2
D_BIG = 2560
S_LEN = 32


def container_text_repr(big_model, tok, texts):
    """真实自然语言 → MSA-4B 浅层均值池化 (携带职业等类别信息)。"""
    ids = tok(texts, padding="max_length", truncation=True, max_length=S_LEN,
              return_tensors="pt").input_ids
    with torch.no_grad():
        return big_model(ids).last_hidden_state.float().mean(dim=1)


def tabldm_passive_repr(X_table):
    """α2 对照: TabLDM 被动中间表征, 逐样本冻结。"""
    from tabldm import TabLDMClassifier
    from research.scale_heterogeneous.v38_g_adult_boundary import TABLDM_CKPT
    clf = TabLDMClassifier(model_path=TABLDM_CKPT, device="cpu")
    rng = np.random.RandomState(42)
    clf.fit(rng.randn(12, 6).astype(np.float32), rng.randint(0, 10, 12))
    m = clf.model_
    reps = []
    h = m.icl_predictor.register_forward_hook(
        lambda mod, inp, out: reps.append(inp[0].detach() if isinstance(inp, tuple) else inp.detach()))
    clf.predict_proba(X_table.numpy())
    h.remove()
    r = reps[0]
    n = X_table.shape[0]
    rep_q = r[:, -n:, :].mean(dim=0).float()
    del clf, m
    gc.collect()
    return rep_q


def train_arm(h_base, X, y, c_frozen, tr_idx, te_idx,
             steps=STEPS, seed=0, ref=None, distill=100):
    """ref 给定时: 先 K 步把 E 蒸馏到 ref (TabLDM 先验), 再任务训练 (E 臂: 先验+自训练)."""
    torch.manual_seed(seed)
    E = EntryEncoder(X.shape[1], D_ENTRY)
    P = nn.Linear(D_ENTRY, D_BIG)
    head = nn.Linear(D_BIG, N_CLS)
    params = list(P.parameters()) + list(head.parameters()) + (
        list(E.parameters()) if c_frozen is None else [])
    opt = torch.optim.Adam(params, lr=1e-3)
    if ref is not None and c_frozen is None:
        for _ in range(distill):
            b = tr_idx[torch.randperm(len(tr_idx))[:64]]
            loss = nn.functional.mse_loss(E(X[b]), ref[b])
            opt.zero_grad(); loss.backward(); opt.step()
    for _ in range(steps):
        b = tr_idx[torch.randperm(len(tr_idx))[:64]]
        cin = c_frozen[b] if c_frozen is not None else E(X[b])
        loss = nn.functional.cross_entropy(head(h_base[b] + P(cin)), y[b])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        cin_all = c_frozen if c_frozen is not None else E(X)
        h_all = h_base + P(cin_all)
        acc = (head(h_all[te_idx]).argmax(1) == y[te_idx]).float().mean().item()
        return acc, knn_acc(h_all, y)


def main():
    print(f"{'='*80}")
    print("V38.0 γ 真实数据跨模态验证 — Adult 数据集 + XOR 标签 (严格跨模态必需)")
    print(f"{'='*80}")
    print("标签: y = (职业∈高薪 [文本侧]) XOR (工时≥45 [表格侧]) → 单侧无法解出")
    print(f"配置: N={N_TOTAL}, 384/128 留出, STEPS={STEPS}, seeds={N_SEEDS}")
    print()

    t0 = time.time()
    texts, X, y = adult_loader.load_adult_crossmodal(N_TOTAL, 42, decouple=True)
    print(f"  Adult 数据: {len(texts)} 样本, 正样本率 {y.float().mean():.2%}")

    big_model = load_big_container(n_layers=N_BIG_LAYERS)
    big_tok = AutoTokenizer.from_pretrained(MSA_DIR)
    h_base = container_text_repr(big_model, big_tok, texts)
    del big_model; gc.collect()
    print(f"  文本容器表征 {tuple(h_base.shape)} ({time.time()-t0:.1f}s)")

    c_passive = tabldm_passive_repr(X)
    print(f"  TabLDM 被动入口 {tuple(c_passive.shape)} ({time.time()-t0:.1f}s)\n")

    rng = np.random.RandomState(1234)
    perm = rng.permutation(N_TOTAL)
    tr_idx = torch.tensor(perm[:N_TOTAL - 128])
    te_idx = torch.tensor(perm[N_TOTAL - 128:])

    arms = ["A 纯文本(无入口)", "B 随机入口", "C 被动TabLDM(α2)",
            "D 自训练(β)", "E 先验+自训练"]
    results = {a: {"acc": [], "knn": []} for a in arms}

    for s in range(N_SEEDS):
        torch.manual_seed(s)
        c_rand = torch.randn(N_TOTAL, D_ENTRY)
        accs = [
            train_arm(h_base, X, y, torch.zeros(N_TOTAL, D_ENTRY), tr_idx, te_idx, seed=s),    # A
            train_arm(h_base, X, y, c_rand, tr_idx, te_idx, seed=s),                          # B
            train_arm(h_base, X, y, c_passive, tr_idx, te_idx, seed=s),                       # C
            train_arm(h_base, X, y, None, tr_idx, te_idx, seed=s),                            # D
            train_arm(h_base, X, y, None, tr_idx, te_idx, seed=s, ref=c_passive),             # E 先验+自训练
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

    e = statistics.mean(results["E 先验+自训练"]["acc"])
    c = statistics.mean(results["C 被动TabLDM(α2)"]["acc"])
    d = statistics.mean(results["D 自训练(β)"]["acc"])
    a = statistics.mean(results["A 纯文本(无入口)"]["acc"])
    b = statistics.mean(results["B 随机入口"]["acc"])
    wins_e_c = sum(1 for s in range(N_SEEDS)
                   if results["E 先验+自训练"]["acc"][s] > results["C 被动TabLDM(α2)"]["acc"][s])
    wins_e_d = sum(1 for s in range(N_SEEDS)
                   if results["E 先验+自训练"]["acc"][s] > results["D 自训练(β)"]["acc"][s])
    wins_e_a = sum(1 for s in range(N_SEEDS)
                   if results["E 先验+自训练"]["acc"][s] > results["A 纯文本(无入口)"]["acc"][s])
    print(f"\n层级: E先验+自训练={e:.3f} | C被动={c:.3f} | D自训练={d:.3f} | A纯文本={a:.3f} | B随机={b:.3f}")
    print(f"E 打败: 被动C {wins_e_c}/{N_SEEDS} | 自训练D {wins_e_d}/{N_SEEDS} | 纯文本A {wins_e_a}/{N_SEEDS}")
    print(f"{'='*80}")
    if wins_e_c >= N_SEEDS - 1 and wins_e_a == N_SEEDS:
        print("✅ V38.0 γ 成立: 真实数据上'先验+自训练'混合入口最优")
    elif wins_e_a == N_SEEDS:
        print(f"⚠️ γ 部分成立: E 打败纯文本 {wins_e_a}/{N_SEEDS}, 但未稳定超越被动先验 ({wins_e_c}/{N_SEEDS})")
    else:
        print(f"❌ γ 未成立 (E>A {wins_e_a}/{N_SEEDS}) — 边界分析见 docs/V38_0_SPEC.md")


if __name__ == "__main__":
    main()