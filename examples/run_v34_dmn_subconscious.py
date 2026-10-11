"""V34.0 — DMN 潜意识创造性联想层 — 端到端入口。

> 归属路线: 规模异构+潜意识联想 (原 V6 衍生路线, 由外部快照 v8.0 重编号而来)
> 重编号对照: 快照 v8.0 → 本项目 v34.0; 快照 v7.0 → 本项目 v33.0 (前置)
> 子包: v34_dmn_subconscious (机制实现), 本脚本只负责训练/评估编排

核心命题: 显式路由(意识层)之外加 DMN 联想层, 跨专家×跨时刻创造性联想(涌现 r≠Σa_m),
双管注入(先验铺垫 + 路由调制)下一步意识层, 使融合超过 v33 纯线性路由。

对比: v33(无DMN) vs v34(有DMN), 同初始化公平对比, 5 seeds。
用法: bash run.sh v34
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc

import torch
import torch.nn.functional as F

from research.scale_heterogeneous.v34_dmn_subconscious import (
    MSA_DIR, EMB_DIR,
    load_msa_big, load_small, load_tokenizer, encode,
    DMNSubconscious, FusionLayer,
)

D_SHARED = 256
M, K = 3, 1
B, S = 18, 16
STEPS = int(os.environ.get("V34_STEPS", "100"))
LR = 1e-2
N_SEEDS = int(os.environ.get("V34_SEEDS", "5"))
DMN_WINDOW = int(os.environ.get("V34_WINDOW", "8"))
ETA = float(os.environ.get("V34_ETA", "0.1"))
LAM = float(os.environ.get("V34_LAM", "0.1"))
MU = float(os.environ.get("V34_MU", "0.1"))


# =====================================================================
# 单次实验: v33(无DMN) vs v34(有DMN)
# =====================================================================

def run_one(seed, big_tok, big_model, D_big, small_tok, small_model, D_small):
    torch.manual_seed(seed)
    print(f"  [seed {seed}] 编码 + 训练 (v33 vs v34)...")

    texts = ["cat dog bird", "animal pet wild", "feline canine fowl"]
    h_big = encode(texts, big_tok, big_model, S).float()
    h_sm = encode(texts, small_tok, small_model, S).float()

    modal_dims = [D_big, D_small, D_small]
    layer_v33 = FusionLayer(D_SHARED, 4 * D_SHARED, M, K, modal_dims, lam=LAM, mu=MU)
    for m in range(M):
        layer_v33.P_m[m].requires_grad_(False)
    layer_v34 = FusionLayer(D_SHARED, 4 * D_SHARED, M, K, modal_dims, lam=LAM, mu=MU)
    layer_v34.load_state_dict(layer_v33.state_dict())  # 同初始化公平对比
    for m in range(M):
        layer_v34.P_m[m].requires_grad_(False)
    dmn = DMNSubconscious(D_SHARED, M, DMN_WINDOW)

    expert_seqs, expert_indices = [], []
    for cls, h in enumerate([h_big, h_sm, h_sm]):
        n = h.shape[0]
        for i in range(6):
            # 取单样本, 而不是整个 [3,S,D] block —— 否则 xi_shared 与 tgt 广播错配
            # (target 是第 i//6 类的 one-hot, 而整块输入含 3 条文本, 输入与目标不对应)
            src = h[i % n]
            g = torch.Generator().manual_seed(seed * 10 + cls * 6 + i)
            expert_seqs.append(src + 0.1 * torch.randn(src.shape, generator=g))
            expert_indices.append(cls)

    targets = torch.zeros(B, S, D_SHARED)
    for i in range(B):
        cls = i // 6
        ds = cls * (D_SHARED // M); de = (cls + 1) * (D_SHARED // M)
        targets[i, :, ds:de] = 1.0

    def train(layer, use_dmn, use_homeostasis=False):
        params = [layer.W_router] + list(layer.gammas) + list(layer.betas) + list(layer.alphas)
        if use_dmn:
            params += list(dmn.parameters())
        opt = torch.optim.AdamW(params, lr=LR)
        for step in range(STEPS):
            opt.zero_grad(set_to_none=True)
            # 联想一次 (用当前 memory bank), r 参与梯度
            r_inj = None
            if use_dmn:
                big_summ = F.linear(h_big.mean(dim=(0, 1)), layer.P_m[0])
                r_inj = dmn.associate(big_summ)
                if use_homeostasis:
                    r_inj = dmn.inject(r_inj)   # 接 C 入前向 (v34c)
            losses = []
            for i, (xi, mod_idx) in enumerate(zip(expert_seqs, expert_indices)):
                xi_shared = F.linear(xi, layer.P_m[mod_idx])
                tgt = targets[i]                     # 与 xi_shared 同为 [S, D_SHARED], 无广播
                y = layer(xi_shared, dmn if use_dmn else None, r_inj)
                losses.append(F.mse_loss(y, tgt))
            total = sum(losses) / len(losses)
            total.backward()
            opt.step()
            # step 后: 更新 memory + 慢变 C (用 detach 的 r)
            if use_dmn:
                all_summ = torch.stack([
                    F.linear(h_big.mean(dim=(0, 1)), layer.P_m[0]),
                    F.linear(h_sm.mean(dim=(0, 1)), layer.P_m[1]),
                    F.linear(h_sm.mean(dim=(0, 1)), layer.P_m[2]),
                ]).detach()
                dmn.push_memory(all_summ)
                dmn.update_C(r_inj, ETA)

    def eval_loss(layer, use_dmn, use_homeostasis=False):
        with torch.no_grad():
            fuse_l = sbig_l = ssm_l = avg_l = 0.0
            r_inj = None
            if use_dmn:
                big_summ = F.linear(h_big.mean(dim=(0, 1)), layer.P_m[0])
                r_inj = dmn.associate(big_summ)
                if use_homeostasis:
                    r_inj = dmn.inject(r_inj)   # 与训练同一口径
            for i, (xi, mod_idx) in enumerate(zip(expert_seqs, expert_indices)):
                xi_shared = F.linear(xi, layer.P_m[mod_idx])
                tgt = targets[i]                     # 与 xi_shared 同为 [S, D_SHARED], 无广播
                d = dmn if use_dmn else None
                ri = r_inj if use_dmn else None
                fuse_l += F.mse_loss(layer(xi_shared, d, ri), tgt).item()
                sbig_l += F.mse_loss(layer.forward_single(xi_shared, 0, d, ri), tgt).item()
                ssm_l += F.mse_loss(layer.forward_single(xi_shared, 1, d, ri), tgt).item()
                avg_l += F.mse_loss(layer.forward_avg(xi_shared, d, ri), tgt).item()
            N = B
            return fuse_l / N, avg_l / N, sbig_l / N, ssm_l / N

    # v33: 无 DMN (纯线性路由)
    layer_v34c = FusionLayer(D_SHARED, 4 * D_SHARED, M, K, modal_dims, lam=LAM, mu=MU)
    layer_v34c.load_state_dict(layer_v33.state_dict())  # 三臂同初始化公平对比
    for m in range(M):
        layer_v34c.P_m[m].requires_grad_(False)

    dmn.reset()
    train(layer_v33, use_dmn=False)
    f7, a7, sb7, ss7 = eval_loss(layer_v33, use_dmn=False)
    # v34: 有 DMN, C 只累积不接前向 (原行为)
    dmn.reset()
    train(layer_v34, use_dmn=True, use_homeostasis=False)
    f8, a8, sb8, ss8 = eval_loss(layer_v34, use_dmn=True, use_homeostasis=False)
    # v34c: 有 DMN 且把 C 接进前向 (突触稳态生效)
    dmn.reset()
    train(layer_v34c, use_dmn=True, use_homeostasis=True)
    f9, a9, sb9, ss9 = eval_loss(layer_v34c, use_dmn=True, use_homeostasis=True)

    return ((f7, sb7, ss7, a7), (f8, sb8, ss8, a8), (f9, sb9, ss9, a9))


def main():
    print(f"{'=' * 90}")
    print("V34.0 DMN 潜意识创造性联想层 (跨专家×跨时刻, 大专家主导, 双管注入)")
    print(f"{'=' * 90}")
    print("大: MSA-4B浅层(D=2560) | 小: embedding/reranker-0.6b(D=1024)")
    print(f"DMN: 窗口W={DMN_WINDOW} eta={ETA} lam(先验)={LAM} mu(路由)={MU}")
    print("对比: v33(无DMN,纯线性路由) vs v34(有DMN,联想涌现)")
    print()

    big_tok = load_tokenizer(MSA_DIR)
    small_tok = load_tokenizer(EMB_DIR)
    big_model, D_big, _ = load_msa_big()
    gc.collect()
    small_model, D_small, _ = load_small("embedding-0.6b", EMB_DIR)

    print(f"\n{'seed':>5} | {'v33_fuse':>8} {'v33_gB':>7} | {'v34_fuse':>8} {'v34_gB':>7} | "
          f"{'v34c_fuse':>9} {'v34c_gB':>7} | {'Δ34':>6} {'Δ34c':>6} | {'PASS':>4}")
    print("-" * 90)
    all_pass = True
    d7s, d8s, d9s, d34s, d34cs, f8s, f9s = [], [], [], [], [], [], []
    for seed in range(N_SEEDS):
        (f7, sb7, ss7, a7), (f8, sb8, ss8, a8), (f9, sb9, ss9, a9) = run_one(
            seed, big_tok, big_model, D_big, small_tok, small_model, D_small)
        g7 = (sb7 - f7) / max(sb7, 1e-9) * 100
        g8 = (sb8 - f8) / max(sb8, 1e-9) * 100
        g9 = (sb9 - f9) / max(sb9, 1e-9) * 100
        delta = (f7 - f8) / max(f7, 1e-9) * 100
        delta_c = (f7 - f9) / max(f7, 1e-9) * 100
        d7s.append(g7); d8s.append(g8); d9s.append(g9); d34s.append(delta); d34cs.append(delta_c)
        f8s.append(f8); f9s.append(f9)
        ok = f8 < f7
        all_pass = all_pass and ok
        print(f"{seed:>5} | {f7:>8.4f} {g7:>+6.1f}% | {f8:>8.4f} {g8:>+6.1f}% | "
              f"{f9:>9.4f} {g9:>+6.1f}% | {delta:>+5.1f}% {delta_c:>+5.1f}% | {'✅' if ok else '❌':>4}")

    print("-" * 90)
    import statistics
    print(f"\n增益均值 vs 单大: v33={statistics.mean(d7s):+.1f}%  "
          f"v34={statistics.mean(d8s):+.1f}%  v34c={statistics.mean(d9s):+.1f}%")
    print(f"融合损失相对 v33 降低: v34={statistics.mean(d34s):+.1f}%  "
          f"v34c={statistics.mean(d34cs):+.1f}%")
    print(f"C 接前向的净效应 (v34c − v34): "
          f"{statistics.mean(d34cs) - statistics.mean(d34s):+.2f} 个百分点")
    n_better = sum(1 for d in d34s if d > 0)
    n_better_c = sum(1 for d in d34cs if d > 0)
    n_c_over_34 = sum(1 for a, b in zip(f9s, f8s) if a < b)
    print(f"v34 优于 v33 的种子: {n_better}/{N_SEEDS}  |  "
          f"v34c 优于 v33: {n_better_c}/{N_SEEDS}  |  v34c 优于 v34: {n_c_over_34}/{N_SEEDS}")
    print(f"{'=' * 90}")
    if all_pass:
        print("✅ V34.0 DMN 潜意识联想成立: 联想(涌现) > 纯线性路由(v33)")
    else:
        print(f"⚠️ v34 优于 v33 的种子: {n_better}/{N_SEEDS}")


if __name__ == "__main__":
    main()
