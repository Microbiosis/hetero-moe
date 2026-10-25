"""V48.0 — 投影空间 (可训练 / 条件依赖) — 诊断 + 可训练性 A/B。

> 归属路线: aligner (语义对齐器线, 接 v10/v13/v15)
> 上游: v33_scale_hetero (1大+N小, 矩形 P_m 投影 + 共享路由)
> 复现: bash run.sh v48  (或 python3 examples/run_v48_projection_space.py)

本代按顺序回答三个问题 (前一个不过就不用做后一个):

  Q1 (诊断, 无训练): **共享空间该多大?**
      "所有模型在这个空间中的并集能够刚好装下" —— 这个"刚好合适"的尺寸
      可计算: 叠放各底座表征 → 协方差谱 → MP 噪声地板以上的方向数
      = 仿射包维数 K_aff; 跨底座典型相关 → 语义冗余 → 共享维数上界。
      配套 K 网格给三个指标: 保真度 / 冗余度 / 重叠度。

  Q2 (A/B/C × 5 seeds): **投影可训练有没有用?**
      审计发现: v33 的 P_m 是 **随机正交初始化 + 冻结**, 专家 FFN 也冻结,
      全流程只训了 W_router + 每专家 (γ,β,α) ≈ 2.3K 参数。
      即旗舰结果 (+25.6%) 用的"投影空间" **完全不是学出来的**。
      三臂 **参数量完全相同**, 只改"是否被梯度更新":
        A 冻结 (v33 现状复现) / B 只解冻 P_m / C 解冻 P_m + 专家 FFN
      → 不受 v41 审计的"容量混杂"影响 (差异只能来自"是否被训练")。

  Q3 (两个任务): 任务本身是否要求"语义对齐"?
      Task-T = v33 原任务 (目标 = 专家身份块): 只考路由, 不考对齐。
      Task-S = 新任务 (目标 = 文本类别块, 同一文本在三个底座上目标相同):
             **必须跨底座对齐**才能做对 —— 这才是"投影空间"该被检验处。
      若 Task-T 上 B≈A 而 Task-S 上 B≫A, 则 v33 的收益来自路由,
      与投影语义无关; 投影空间的价值需要一个"要求对齐"的任务才显形。
"""
import sys, os, json, gc, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from research._primitives.scale_hetero_loaders import (
    MSA_DIR, EMB_DIR, RER_DIR, load_msa_big, load_small, load_tokenizer, encode,
)
from research.scale_heterogeneous.v33_scale_hetero import CrossModalFusionLayer
from research.aligner.v48_projection_space.space import (
    union_report, pairwise_canonical_correlations, k_grid_report, CONCAT_REFERENCE,
)
from research.aligner.v48_projection_space.mechanism import ARMS, configure_arm

D_SHARED = 256
M, K = 3, 1
B, S = 18, 16
STEPS = int(os.environ.get("V48_STEPS", "100"))
LR = 1e-2
LR_BIG = float(os.environ.get("V48_LR_BIG", "1e-3"))   # 大矩阵的学习率
N_SEEDS = int(os.environ.get("V48_SEEDS", "5"))
ONLY_PROTOCOLS = [x for x in os.environ.get("V48_PROTOCOLS", "").split(",") if x]
ONLY_TASKS = [x for x in os.environ.get("V48_TASKS", "").split(",") if x]
DIAG_ONLY = os.environ.get("V48_DIAG_ONLY", "") == "1"
# 设备: 默认 CPU; HF Space 上设 V48_DEVICE=cuda 走 GPU
DEVICE = torch.device(os.environ.get("V48_DEVICE", "cpu"))

# ---- 协议: 两套, 用来把"任务构造"本身的影响分离出来 ----
# P1 "v33-parity": 与 v33 逐位一致 (3 类文本, 每底座 6 样本, S=16 → 80% 位置是 padding)
# P2 "扩展":       6 类文本, 每底座 12 样本, S=4 (padding 只占 25%)
PROTOCOLS = {
    "P1_v33parity": {
        "train": ["cat dog bird", "animal pet wild", "feline canine fowl"],
        "heldout": ["kitten puppy birdie", "beast creature untamed", "tabby hound poultry"],
        "n_per_expert": 6, "S": 16,
        "note": "与 v33 逐位一致 (含 S=16 造成的 80% padding 位置)",
    },
    "P2_extended": {
        "train": ["cat dog bird", "animal pet wild", "feline canine fowl",
                  "truck boat plane", "apple pear peach", "red blue green"],
        "heldout": ["kitten puppy birdie", "beast creature untamed", "tabby hound poultry",
                    "lorry ship aircraft", "plum apricot nectarine", "crimson azure emerald"],
        "n_per_expert": 12, "S": 4,
        "note": "6 类 + S=4: 样本翻倍, padding 占比从 80% 降到 25%",
    },
}


def build_samples(h_by_expert, task, seed, n_per_expert=6, perturb=0.1, tag=0, S=S,
                 n_classes=None):
    """构造样本 (按底座分组, 批量化)。

    ★ 构造与 v33 **逐位对齐** (每底座 6 条样本, 文本按 i%3 轮转, 扰动幅度
      0.1, 扰动种子 seed*10 + m*6 + i), 因此 Task-T 的 A 臂可与 v33 已发布
      数字直接对话。

    task="T" (v33 原任务): 目标 = 专家身份块 (第 m 个专家 → 第 m 块) —— 只考路由
    task="S" (对齐任务):   目标 = 文本类别块 (同一文本在三个底座上目标相同)
                           —— 输入完全相同, 只有目标语义不同, 故 T/S 是
                           **严格配对对照**: 差值即"任务是否要求跨底座对齐"

    返回 (per_expert: [[n_m, S, D_m]], targets: [N, S, D_SHARED])
    """
    groups, targets = [], []
    if n_classes is None:
        n_classes = h_by_expert[0].shape[0]
    for m, h in enumerate(h_by_expert):
        seqs = []
        for i in range(n_per_expert):
            cls = i % n_classes
            src = h[cls]
            gseed = seed * 10 + m * 6 + i + (500 if tag else 0)
            g = torch.Generator().manual_seed(gseed)
            seqs.append(src + perturb * torch.randn(src.shape, generator=g))
            blk = m if task == "T" else cls
            tgt = torch.zeros(S, D_SHARED)
            ds = blk * (D_SHARED // M)
            de = (blk + 1) * (D_SHARED // M)
            tgt[:, ds:de] = 1.0
            targets.append(tgt)
        groups.append(torch.stack(seqs, dim=0))
    return groups, torch.stack(targets, dim=0)


def _project(layer, per_expert):
    """把各底座的样本用各自的 P_m 投影到共享空间, 再拼成一批。"""
    xs = [F.linear(h, layer.P_m[m]) for m, h in enumerate(per_expert)]
    return torch.cat(xs, dim=0)            # [N, S, D_SHARED]


def train_and_eval(layer, params, train_pack, eval_pack):
    """训练 + 评估 (in-sample 与 held-out 各一套)。

    ⚠️ 批量化说明: v33 原本逐样本循环 (18 次小算子/step), 在本机 CPU 上
       100 步要 432s (瓶颈是 100 万次小 kernel 启动的开销, 不是算力)。
       v33 的融合层本身与批维无关 (全部是逐元素 + Linear), 因此这里把
       样本拼成 [N, S, D] 一批走, **数值语义与逐样本版等价**
       (损失是逐样本 MSE 的均值 = 批内整体 MSE 均值), 但快两个数量级。
    """
    tr_x, tr_t = train_pack
    tr_x = [h.to(DEVICE) for h in tr_x]
    tr_t = tr_t.to(DEVICE)
    # 分组学习率: 路由器/仿射用主 LR; 大矩阵 (P_m / 专家 FFN) 用小 LR。
    # A 臂没有大矩阵 → 行为与 v33 完全一致 (可比性不受影响);
    # B/C 臂若用主 LR 会在 100 步内发散 (实测 C 臂 MSE → 2.7e5)。
    big_ids = {id(p) for p in params if p.numel() > 4096}
    small_ids = {id(p) for p in params if p.numel() <= 4096}
    groups = [{"params": [p for p in params if id(p) in small_ids], "lr": LR},
              {"params": [p for p in params if id(p) in big_ids], "lr": LR_BIG}]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], lr=LR, weight_decay=1e-4)
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        x_shared = _project(layer, tr_x)
        y = layer(x_shared)
        loss = F.mse_loss(y, tr_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)   # 防止 C 臂发散 (对 A/B 无影响)
        opt.step()

    out = {}
    for name, pack in (("in_sample", train_pack), ("held_out", eval_pack)):
        with torch.no_grad():
            ex, tgts = pack
            ex = [h.to(DEVICE) for h in ex]
            tgts = tgts.to(DEVICE)
            xi_shared = _project(layer, ex)
            x_det = xi_shared.detach()
            y_fuse, alpha_hat = layer.forward_with_routing(xi_shared)
            fuse = F.mse_loss(y_fuse, tgts).item()
            sbig = ssm = avg = 0.0
            route_counts = [0] * M
            for m in range(M):
                cnt = int((alpha_hat.argmax(dim=-1) == m).sum().item())
                route_counts[m] = cnt
            for m, key in ((0, "sbig"), (1, "ssm")):
                h_a = x_det * layer.gammas[m] + layer.betas[m]
                f = layer._swiglu(h_a, layer.w_gates[m], layer.w_ups[m], layer.w_downs[m])
                y_m = xi_shared + layer.alphas[m] * f
                if key == "sbig":
                    sbig = F.mse_loss(y_m, tgts).item()
                else:
                    ssm = F.mse_loss(y_m, tgts).item()
            tot = torch.zeros_like(xi_shared)
            for m in range(M):
                h_a = x_det * layer.gammas[m] + layer.betas[m]
                tot += layer._swiglu(h_a, layer.w_gates[m], layer.w_ups[m], layer.w_downs[m])
            avg = F.mse_loss(xi_shared + tot / M, tgts).item()
            # 平凡基线: 预测全零 (目标有 1/3 维是 1) → 这是 MSE 的绝对下界参照
            trivial = F.mse_loss(torch.zeros_like(tgts), tgts).item()
            out[name] = {
                "fuse": fuse, "single_big": sbig, "single_small": ssm, "avg": avg,
                "trivial_zero": trivial,
                "route": route_counts,
            }
    return out


def phase1_diagnose(h_by_expert, names):
    print("\n" + "=" * 78)
    print("阶段 1 — 空间尺寸诊断 (training-free)")
    print("=" * 78)
    rep = union_report(h_by_expert, names)
    u = rep["union"]
    print(f"\n[并集谱] N={u['n']} token  p={u['p']} 维 (叠放后)")
    print(f"  有效维数 (99% 方差) = {u['k_1pct']}   (95% 方差) = {u['k_5pct']}")
    print(f"  参与比有效秩         = {u['eff_rank']:.1f}   拐点 = {u['elbow']}")
    print(f"  [参考] MP 地板估计 K_aff = {u['k_aff']} (地板 {u['noise_floor']:.3g}) "
          f"— 对模型输出常失效, 需与相对阈值维数对照")
    ev = u["eigvals"]
    print("  前 8 个特征值: " + " ".join(f"{x:.3g}" for x in ev[:8]))
    print(f"  噪声地板 = {u['noise_floor']:.3g}  (前 8 个里 {int((ev[:8] > u['noise_floor']).sum())} 个在地板之上)")

    print("\n[每底座内在维]")
    for nm, r in rep["per_expert"].items():
        print(f"  {nm:>10}: n={r['n']:>4} p={r['p']:>5} | 99%维数={r['k_1pct']:>4} "
              f"95%维数={r['k_5pct']:>4} eff_rank={r['eff_rank']:>6.1f} 方差={r['total_var']:.3g}")

    print("\n[跨底座典型相关 (语义冗余的线性上界)]")
    ccs = pairwise_canonical_correlations(h_by_expert, top=3)
    for c in ccs:
        flag = "✔超零分布" if c.get("cka_exceeds_null") else "✘不超零分布(=噪声)"
        print(f"  {names[c['pair'][0]]} × {names[c['pair'][1]]}: "
              f"CKA={c['cka']:.4f} (零分布 max={c.get('cka_null_max',0):.4f}) {flag} | "
              f"top8PC 上 CC1={c['cc1']:.4f} (零分布 max={c.get('cc_null_max',0):.4f}) | "
              f"共享维数(95%)={c['shared_dim_95']}")
    print(f"  → 三底座 CKA 均值 = {sum(c['cka'] for c in ccs)/len(ccs):.4f} "
          f"(零分布上限均值 = {sum(c.get('cka_null_max',0) for c in ccs)/len(ccs):.4f})")

    print("\n[K 网格: 联合 PCA 共享空间的三指标]")
    grid = [8, 16, 32, 48, 64, 96]
    kg = k_grid_report(h_by_expert, grid, names)
    print(f"  {'K':>5} | {'保真度min':>9} | {'保真度(大)':>10} | {'冗余度':>7} | {'重叠度':>7} | {'重叠max':>8}")
    print("  " + "-" * 62)
    seen = set()
    for r in kg:
        if r["K"] in seen:
            continue
        seen.add(r["K"])
        print(f"  {r['K']:>5} | {r['fidelity_min']:>9.4f} | {r['fidelity'][names[0]]:>10.4f} | "
              f"{r['redundancy']:>7.4f} | {r['overlap']:>7.4f} | {r['overlap_max']:>8.4f}")
    print("  (K 受 token 行数上限约束, 超出即被截断)")
    print(f"\n  参考端点 {CONCAT_REFERENCE['name']}: K_fused={CONCAT_REFERENCE['K_fused_total']} 重叠度=0 (结构上为零)")

    return {
        "union": {k: v for k, v in u.items() if k != "eigvals"},
        "union_top_eigvals": [float(x) for x in ev[:12]],
        "per_expert": {nm: {k: v for k, v in r.items() if k != "eigvals"}
                       for nm, r in rep["per_expert"].items()},
        "canonical_correlations": [
            {"pair": list(c["pair"]), "cc": c["cc"], "shared_dim_95": c["shared_dim_95"]}
            for c in ccs],
        "k_grid": kg,
        "concat_reference": CONCAT_REFERENCE,
    }


def run_arm(arm, task, h_by_expert, heldout_by_expert, modal_dims, seeds,
            S=S, n_per_expert=6, n_classes=3):
    rows = []
    info0 = None
    for seed in range(seeds):
        torch.manual_seed(seed)
        layer = CrossModalFusionLayer(D_SHARED, 4 * D_SHARED, M, K, modal_dims).to(DEVICE)
        params, info = configure_arm(layer, arm)
        info0 = info
        tr_x, tr_t = build_samples(h_by_expert, task, seed, n_per_expert=n_per_expert,
                                   tag=0, S=S, n_classes=n_classes)
        ho_x, ho_t = build_samples(heldout_by_expert, task, seed, n_per_expert=n_per_expert,
                                   tag=1, S=S, n_classes=n_classes)
        res = train_and_eval(layer, params, (tr_x, tr_t), (ho_x, ho_t))
        rows.append({"seed": seed, **res})
    return rows, info0


def print_task_table(task, arm_rows):
    print(f"\n  -- Task-{task} --")
    print(f"  {'臂':>16} | {'fuse(in)':>9} | {'fuse(held)':>10} | {'平凡(全零)':>10} | "
          f"{'过拟合间隙':>9} | {'单大(held)':>10} | {'vs单大':>7} | {'PASS':>4}")
    print("  " + "-" * 112)
    for arm, rows in arm_rows.items():
        fi = [r["in_sample"]["fuse"] for r in rows]
        fh = [r["held_out"]["fuse"] for r in rows]
        tv = [r["held_out"]["trivial_zero"] for r in rows]
        bh = [r["held_out"]["single_big"] for r in rows]
        gb = [(b - f) / max(b, 1e-9) * 100 for b, f in zip(bh, fh)]
        gap = [max(fh[i] - fi[i], 0.0) / max(fh[i], 1e-9) * 100 for i in range(len(rows))]
        ok = sum(1 for g in gb if g > 0)
        print(f"  {ARMS[arm]['label']:>16} | {sum(fi)/len(fi):>9.4f} | {sum(fh)/len(fh):>10.4f} | "
              f"{sum(tv)/len(tv):>10.4f} | {sum(gap)/len(gap):>+8.1f}% | "
              f"{sum(bh)/len(bh):>10.4f} | {sum(gb)/len(gb):>+6.1f}% | {ok}/{len(rows)}")


def main():
    print("=" * 78)
    print("V48.0 — 投影空间 (可训练 / 条件依赖) — 诊断 + 可训练性 A/B")
    print("=" * 78)
    print("大: MSA-4B (D=2560) | 小1: embedding-0.6b (D=1024) | 小2: reranker-0.6b (D=1024)")
    print(f"STEPS={STEPS}  SEEDS={N_SEEDS}  LR={LR}  协议: {list(PROTOCOLS)}")

    # ⚠️ 内存纪律: 本机 14GB 总内存 / 可用约 3GB。三个模型同时驻留会在
    # 加载 reranker 时卡死 (实测停滞在权重加载 15%)。故改为
    # **逐个加载 → 编码 → 释放**, 峰值只留一个模型。
    big_tok = load_tokenizer(MSA_DIR)
    small_tok = load_tokenizer(EMB_DIR)
    try:
        rer_tok = load_tokenizer(RER_DIR)
    except Exception:
        rer_tok = small_tok

    JOBS = ([(pn, "train", pr["train"], pr["S"]) for pn, pr in PROTOCOLS.items()]
            + [(pn, "heldout", pr["heldout"], pr["S"]) for pn, pr in PROTOCOLS.items()])

    def _encode_on(texts, tok, model, S_p):
        """与 _primitives.encode 同义, 但把输入放到 DEVICE 上 (不改共享模块)。"""
        enc = tok(texts, padding="max_length", truncation=True, max_length=S_p,
                  return_tensors="pt")
        with torch.no_grad():
            h = model(enc["input_ids"].to(DEVICE)).last_hidden_state
        return h.detach()

    def _encode_then_free(tok, loader, *args):
        model, dim, _ = loader(*args)
        model = model.to(DEVICE)
        out = {}
        for pn, split, texts, S_p in JOBS:
            out[(pn, split)] = _encode_on(texts, tok, model, S_p)
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return out, dim

    enc_big, D_big = _encode_then_free(big_tok, load_msa_big)
    enc_s1, D_small = _encode_then_free(small_tok, load_small, "embedding-0.6b", EMB_DIR)
    try:
        enc_s2, D_rer = _encode_then_free(rer_tok, load_small, "reranker-0.6b", RER_DIR)
    except Exception as e:
        print(f"  [警告] reranker 不可用 ({e}); 小2 退化为与小1同 backbone")
        enc_s2, D_rer = enc_s1, D_small
    gc.collect()

    names = ["big", "small1", "small2"]

    # ---- 阶段 1: 诊断 (用扩展协议 P2: 6 类文本 + S=4, padding 少, 内容真实) ----
    p2 = "P2_extended"
    h_diag = [torch.cat([enc_big[(p2, "train")], enc_big[(p2, "heldout")]], dim=0),
              torch.cat([enc_s1[(p2, "train")], enc_s1[(p2, "heldout")]], dim=0),
              torch.cat([enc_s2[(p2, "train")], enc_s2[(p2, "heldout")]], dim=0)]
    diag = phase1_diagnose(h_diag, names)

    # ---- 阶段 2: A/B/C × 2 任务 × 2 协议 × 5 seeds ----
    print("\n" + "=" * 78)
    print("阶段 2 — 投影可训练性 A/B (3 臂 x 2 任务 x 2 协议 x 5 seeds)")
    print("=" * 78)
    print("  Task-T = v33 原任务 (目标=专家身份块) | Task-S = 对齐任务 (目标=文本类别块, 跨底座同目标)")
    print("  P1 = v33 逐位复现协议 | P2 = 扩展协议 (6 类 + S=4)")
    print("  阶段 2 沿用 v33 构造 (小1/小2 共用 backbone), 以便与已发布数字对话")

    results = {}
    for pn, pr in ({} if DIAG_ONLY else PROTOCOLS).items():
        if ONLY_PROTOCOLS and pn not in ONLY_PROTOCOLS:
            continue
        h_v33 = [enc_big[(pn, "train")], enc_s1[(pn, "train")], enc_s1[(pn, "train")]]
        h_v33_ho = [enc_big[(pn, "heldout")], enc_s1[(pn, "heldout")], enc_s1[(pn, "heldout")]]
        for task in ("T", "S"):
            if ONLY_TASKS and task not in ONLY_TASKS:
                continue
            arm_rows = {}
            for arm in ARMS:
                t0 = time.time()
                rows, info = run_arm(arm, task, h_v33, h_v33_ho,
                                     [D_big, D_small, D_small], N_SEEDS, S=pr["S"],
                                     n_per_expert=pr["n_per_expert"], n_classes=len(pr["train"]))
                arm_rows[arm] = rows
                results[f"{pn}:{task}:{arm}"] = {"rows": rows, "info": info}
                print(f"    [{pn} Task-{task}] {ARMS[arm]['label']} 完成 ({time.time()-t0:.0f}s)")
            print_task_table(task, arm_rows)

    # ---- 汇总判定 ----
    print("\n" + "=" * 78)
    print("判定 (held-out fuse MSE; 越低越好)")
    print("=" * 78)
    def mean_fuse(key, split):
        rs = results[key]["rows"]
        return sum(r[split]["fuse"] for r in rs) / len(rs)
    for pn in ({} if DIAG_ONLY else PROTOCOLS):
        if ONLY_PROTOCOLS and pn not in ONLY_PROTOCOLS:
            continue
        for task in ("T", "S"):
            if ONLY_TASKS and task not in ONLY_TASKS:
                continue
            a, b, c = (mean_fuse(f"{pn}:{task}:{arm}", "held_out")
                       for arm in ("A_frozen", "B_train_proj", "C_train_all"))
            tv = sum(r["held_out"]["trivial_zero"] for r in results[f"{pn}:{task}:A_frozen"]["rows"]) \
                / N_SEEDS
            gb = (a - b) / max(a, 1e-9) * 100
            gc_ = (a - c) / max(a, 1e-9) * 100
            print(f"  {pn:14s} Task-{task}: A={a:8.4f}  B={b:8.4f} ({gb:+7.1f}%)  "
                  f"C={c:8.4f} ({gc_:>+9.1f}%)   [平凡基线 {tv:.4f}]")
    print("\n  读法: B < A → 让投影可训练在留出集上也更好 (真收益);")
    print("        B 训练集远低于留出集 → 投影把空间变成了记忆器 (过拟合);")
    print("        平凡基线(预测全零)才是绝对下界参照 —— 低于它才算真的学到了。")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "research", "aligner", "v48_projection_space", "results")
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "version": "48.0",
        "package": "v48_projection_space",
        "line": "aligner",
        "protocols": {k: {kk: vv for kk, vv in v.items() if kk not in ("train", "heldout")}
                      for k, v in PROTOCOLS.items()},
        "diagnostics": diag,
        "arms": {k: v["info"] for k, v in results.items()},
        "results": {k: v["rows"] for k, v in results.items()},
        "notes": {
            "task_T": "v33 原任务: 目标 = 专家身份块 (只考路由)",
            "task_S": "新任务: 目标 = 文本类别块, 同一文本在三个底座上目标相同 (要求对齐)",
            "deviation": "阶段 1 诊断用三个真正不同的编码器 (含 reranker) + 扩展协议 P2; "
                         "阶段 2 沿用 v33 构造 (小1/小2 共用 backbone) 以与已发布数字可比",
            "gradient_path": "v33 的 _compute_output 对专家通路用 x_det=detach(), "
                             "P_m 的梯度只经残差项回传 —— 解冻后学习信号较窄",
            "batching": "训练改为批量化 (v33 原为逐样本循环): 融合层与批维无关, "
                        "损失语义等价, 速度提升约两个数量级",
        },
    }
    mpath = os.path.join(out_dir, "metrics.json")
    if os.path.exists(mpath):          # 分批跑时合并, 不覆盖已有组合
        try:
            with open(mpath, encoding="utf-8") as f:
                old = json.load(f)
            for k in ("arms", "results"):
                old.setdefault(k, {}).update(payload[k])
            old["diagnostics"] = payload["diagnostics"]
            old["notes"] = payload["notes"]
            old["protocols"] = payload["protocols"]
            payload = old
        except Exception:
            pass
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n  指标已落盘: {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
