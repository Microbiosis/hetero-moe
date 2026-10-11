"""v47.0 — 规模异构家族 (v33/v34/v35) 的样本外审计 runner.

被审计的三个 runner **全部在训练同源数据上评估**:
- ``examples/run_v33_one_big_many_small.py:104-131``  训练用 ``expert_seqs``, 评估
  又遍历同一个 ``expert_seqs``
- ``examples/run_v34_dmn_subconscious.py:111-130``  ``eval_loss`` 内嵌在 ``run_one``
  里, 同样遍历训练用的 ``expert_seqs``
- ``examples/run_v35_container_entries.py:60-115``   6 个样本训完立刻在同 6 个上评

三个 runner 均无任何 held-out 构造 (grep ``held_out|heldout`` 0 命中)。

三个口径
--------
- ``in_sample``        与原 runner 同构
- ``heldout_noise``    ``seed + 1000``: 同文本, 换噪声抽样
- ``heldout_stimulus`` **换文本**: 未见过的输入刺激 (v35 无噪声项, 只此一档)

对 v33/v34/v35 之外的同线包 (v36-v39), 原生就带留出/LOO 协议, 不在本 runner 内
重跑 —— 结论见各自 MANIFEST 的 ``leakage_audit.note``。

用法::

    python3 examples/run_v47_scale_hetero_heldout.py
    V47_SH_PACKAGES=v34 python3 examples/run_v47_scale_hetero_heldout.py
    V47_SH_SEEDS=1 python3 examples/run_v47_scale_hetero_heldout.py     # 冒烟
    V47_BIG_LAYERS=1 python3 examples/run_v47_scale_hetero_heldout.py  # 快
"""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from research.central_stability.v47_heldout_audit import (
    audit_table, leakage_verdict, paired_gain, retention,
)
from research.scale_heterogeneous.v34_dmn_subconscious import (
    MSA_DIR, EMB_DIR, DMNSubconscious, FusionLayer, load_msa_big, load_small,
    load_tokenizer, encode,
)
from research.scale_heterogeneous.v35_container_entries import (
    ProjHead, big_forward, load_backbone,
)

D_SHARED = 256
D_BIG = 2560
M, K = 3, 1
B, S = 18, 16
STEPS = int(os.environ.get("V47_SH_STEPS", "100"))
LR = 1e-2
HO_OFFSET = 1000
SH_OUT = "examples/run_v47_scale_hetero_heldout.py (5 seeds × 3 口径)"
N_BIG_LAYERS = int(os.environ.get("V47_BIG_LAYERS", "1"))
DMN_WINDOW, ETA, LAM, MU = 8, 0.1, 0.1, 0.1

TEXTS = ["cat dog bird", "animal pet wild", "feline canine fowl"]
HO_TEXTS = ["river stone cloud", "iron glass paper", "morning night noon"]
V35_TEXTS = ["cat dog bird", "animal pet wild", "feline canine fowl",
             "quick brown fox", "lazy dog sleeps", "birds fly high"]
V35_HO_TEXTS = ["morning night noon", "river stone cloud", "iron glass paper",
                "summer winter spring", "north south east", "fire earth wind"]


# =====================================================================
# 数据构造
# =====================================================================

def _blocks(scope, h_big, h_sm):
    """把三个模态的 block 变成 18 个样本 (v33/v34 用)。"""
    draw = 0 if scope == "in_sample" else HO_OFFSET
    seeds = _seed_base(scope)
    modal_seqs, modal_indices = [], []
    for cls, h in enumerate([h_big, h_sm, h_sm]):
        n = h.shape[0]
        for i in range(6):
            src = h[i % n]
            g = torch.Generator().manual_seed(seeds * 10 + draw + cls * 6 + i)
            modal_seqs.append(src + 0.1 * torch.randn(src.shape, generator=g))
            modal_indices.append(cls)
    return modal_seqs, modal_indices


_SEED_BASE = {"in_sample": 0, "heldout_noise": HO_OFFSET, "heldout_stimulus": HO_OFFSET}


def _seed_base(scope):
    return _SEED_BASE[scope]


def _scope_blocks(scope, enc):
    """heldout_noise 用**训练刺激**+新噪声; heldout_stimulus 用**未见刺激**。"""
    key = "ev" if scope == "heldout_stimulus" else "tr"
    return _blocks(scope, enc[f"h_big_{key}"], enc[f"h_sm_{key}"])


def _targets():
    targets = torch.zeros(B, S, D_SHARED)
    for i in range(B):
        cls = i // 6
        ds = cls * (D_SHARED // M)
        de = (cls + 1) * (D_SHARED // M)
        targets[i, :, ds:de] = 1.0
    return targets


# =====================================================================
# v33 / v34
# =====================================================================

def _run_v33(seed, enc, modal_seqs, modal_indices, targets, layer, dmn=None):
    """训练一次并在给定数据上评 —— v33 (无 DMN) 与 v34 (有 DMN) 共用。"""
    use_dmn = dmn is not None
    params = [layer.W_router] + list(layer.gammas) + list(layer.betas) + list(layer.alphas)
    if use_dmn:
        params += list(dmn.parameters())
    opt = torch.optim.AdamW(params, lr=LR)
    h_big = enc["h_big_tr"]
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        r_inj = None
        if use_dmn:
            r_inj = dmn.associate(F.linear(h_big.mean(dim=(0, 1)), layer.P_m[0]))
        losses = []
        for i, (xi, mod_idx) in enumerate(zip(modal_seqs, modal_indices)):
            xi_shared = F.linear(xi, layer.P_m[mod_idx])
            losses.append(F.mse_loss(layer(xi_shared, dmn if use_dmn else None, r_inj),
                                     targets[i]))
        (sum(losses) / len(losses)).backward()
        opt.step()
        if use_dmn:
            h_sm = enc["h_sm_tr"]
            all_summ = torch.stack([
                F.linear(h_big.mean(dim=(0, 1)), layer.P_m[0]),
                F.linear(h_sm.mean(dim=(0, 1)), layer.P_m[1]),
                F.linear(h_sm.mean(dim=(0, 1)), layer.P_m[2]),
            ]).detach()
            dmn.push_memory(all_summ)
            dmn.update_C(r_inj, ETA)


def _eval_fuse(layer, modal_seqs, modal_indices, targets, h_big_eval, dmn=None):
    """fuse MSE —— 与 v33 第 104-131 行 / v34 eval_loss 同式。

    ``h_big_eval`` 是**当前口径**的大模型 block, 供 DMN 联想用 —— 必须跟着口径走,
    否则 held-out 会偷偷用回训练刺激。
    """
    use_dmn = dmn is not None
    with torch.no_grad():
        r_inj = None
        if use_dmn:
            r_inj = dmn.associate(F.linear(h_big_eval.mean(dim=(0, 1)), layer.P_m[0]))
        total = 0.0
        for i, (xi, mod_idx) in enumerate(zip(modal_seqs, modal_indices)):
            xi_shared = F.linear(xi, layer.P_m[mod_idx])
            y = layer(xi_shared, dmn if use_dmn else None, r_inj)
            total += F.mse_loss(y, targets[i]).item()
        return total / B


def _eval_single_big(layer, modal_seqs, modal_indices, targets):
    """单大 (专家 0) 基线 —— 与 v33 第 116-120 行同式。"""
    with torch.no_grad():
        total = 0.0
        for i, (xi, mod_idx) in enumerate(zip(modal_seqs, modal_indices)):
            xi_shared = F.linear(xi, layer.P_m[mod_idx])
            x_det = xi_shared.detach()
            h = x_det * layer.gammas[0] + layer.betas[0]
            f = layer._swiglu(h, layer.w_gates[0], layer.w_ups[0], layer.w_downs[0])
            total += F.mse_loss(xi_shared + layer.alphas[0] * f, targets[i]).item()
        return total / B


def _new_fusion_layer(modal_dims):
    return FusionLayer(D_SHARED, 4 * D_SHARED, M, K, modal_dims, lam=LAM, mu=MU)


def _mk_pair_v33v34(kind, seed, enc, scopes):
    modal_dims = [enc["D_big"], enc["D_small"], enc["D_small"]]
    torch.manual_seed(seed)
    a = _new_fusion_layer(modal_dims)
    b = _new_fusion_layer(modal_dims)
    b.load_state_dict(a.state_dict())                 # 同初始化公平对比 (v34 原做法)
    for layer in (a, b):
        for m in range(M):
            layer.P_m[m].requires_grad_(False)

    targets = _targets()
    tr = _blocks("in_sample", enc["h_big_tr"], enc["h_sm_tr"])
    out = {s: {"treat": [], "base": []} for s in scopes}

    if kind == "v33":
        torch.manual_seed(seed)
        _run_v33(seed, enc, tr[0], tr[1], targets, a)
        a.eval()
        for s in scopes:
            ms, mi = _scope_blocks(s, enc)
            hk = "ev" if s == "heldout_stimulus" else "tr"
            out[s]["treat"].append(_eval_fuse(a, ms, mi, targets, enc[f"h_big_{hk}"]))
            out[s]["base"].append(_eval_single_big(a, ms, mi, targets))
    else:                                            # v34: DMN vs 无 DMN
        dmn = DMNSubconscious(D_SHARED, M, DMN_WINDOW)
        torch.manual_seed(seed)
        _run_v33(seed, enc, tr[0], tr[1], targets, b, dmn)
        torch.manual_seed(seed)
        _run_v33(seed, enc, tr[0], tr[1], targets, a)
        b.eval(); a.eval()
        for s in scopes:
            ms, mi = _scope_blocks(s, enc)
            hk = "ev" if s == "heldout_stimulus" else "tr"
            out[s]["treat"].append(_eval_fuse(b, ms, mi, targets, enc[f"h_big_{hk}"], dmn))
            out[s]["base"].append(_eval_fuse(a, ms, mi, targets, enc[f"h_big_{hk}"]))
    return out


# =====================================================================
# v35 — 大模型容器 + 异质入口 (只训 prefix + 头)
# =====================================================================

def _v35_targets():
    t = torch.zeros(6, D_SHARED)
    for i in range(6):
        cls = i // 2
        t[i, cls * (D_SHARED // 3):(cls + 1) * (D_SHARED // 3)] = 1.0
    return t


def _mk_pair_v35(seed, env, scopes):
    """baseline = 可学习随机 prior (无入口), treatment = 三异质入口 prefix。"""
    torch.manual_seed(seed)
    targets = _v35_targets()
    out = {s: {"treat": [], "base": []} for s in scopes}
    ids = {s: env["tok"](V35_HO_TEXTS if s == "heldout_stimulus" else V35_TEXTS,
                         padding="max_length", truncation=True, max_length=S,
                         return_tensors="pt").input_ids for s in scopes}

    torch.manual_seed(seed)
    prior = torch.nn.Parameter(torch.randn(3, D_BIG) * 0.02)
    head = ProjHead(D_BIG, D_SHARED)
    opt = torch.optim.AdamW([prior] + list(head.parameters()), lr=LR)
    tr_ids = ids["in_sample"]
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        pref = prior.unsqueeze(0).expand(len(tr_ids), -1, -1)
        h = big_forward(env["big_model"], tr_ids, prefix=pref).mean(dim=1)
        loss = F.mse_loss(head(h), targets)
        loss.backward()
        opt.step()
    head.eval()
    for s in scopes:
        with torch.no_grad():
            pref = prior.unsqueeze(0).expand(len(ids[s]), -1, -1)
            h = big_forward(env["big_model"], ids[s], prefix=pref).mean(dim=1)
            out[s]["base"].append(F.mse_loss(head(h), targets).item())

    # 入口臂: 用 baseline 训好的头做同初始化 (v35 原做法)
    init = {k: v.detach().clone() for k, v in head.state_dict().items()}
    head2 = ProjHead(D_BIG, D_SHARED)
    head2.load_state_dict(init)
    torch.manual_seed(seed)
    c_sem_m = env["c_sem"].mean(0).detach()
    c_rank_m = env["c_rank"].mean(0).detach()
    with torch.no_grad():
        Ps = torch.empty(D_BIG, env["c_struct"].shape[-1]); torch.nn.init.orthogonal_(Ps)
        Pe = torch.empty(D_BIG, c_sem_m.shape[-1]); torch.nn.init.orthogonal_(Pe)
        Pr = torch.empty(D_BIG, c_rank_m.shape[-1]); torch.nn.init.orthogonal_(Pr)
        p0 = torch.stack([F.linear(env["c_struct"], Ps), F.linear(c_sem_m, Pe),
                          F.linear(c_rank_m, Pr)]) * 0.02
    prefix = torch.nn.Parameter(p0.clone())
    opt2 = torch.optim.AdamW([prefix] + list(head2.parameters()), lr=LR)
    for _ in range(STEPS):
        opt2.zero_grad(set_to_none=True)
        pref = prefix.unsqueeze(0).expand(len(tr_ids), -1, -1)
        h = big_forward(env["big_model"], tr_ids, prefix=pref).mean(dim=1)
        loss = F.mse_loss(head2(h), targets)
        loss.backward()
        opt2.step()
    head2.eval()
    for s in scopes:
        with torch.no_grad():
            pref = prefix.unsqueeze(0).expand(len(ids[s]), -1, -1)
            h = big_forward(env["big_model"], ids[s], prefix=pref).mean(dim=1)
            out[s]["treat"].append(F.mse_loss(head2(h), targets).item())
    return out


PACKAGES = [
    ("v33", "融合 vs 单大 (MSA-4B 浅层)", "v33 1大+N小 融合优于单一最大模型"),
    ("v34", "DMN 联想层 vs 无 DMN (v33)", "DMN 联想层进一步降低融合损失"),
    ("v35", "三异质入口 prefix 注入 vs 无注入随机 prior", "异质入口注入能否带来增益"),
]


def main():
    want = os.environ.get("V47_SH_PACKAGES", "")
    wanted = {p.strip() for p in want.split(",") if p.strip()}
    n_seeds = int(os.environ.get("V47_SH_SEEDS", "5"))
    scopes = ("in_sample", "heldout_noise", "heldout_stimulus")

    print("=" * 92)
    print("v47.0 规模异构家族样本外审计 — v33 / v34 / v35")
    print("受审计包: " + ", ".join(f"{k} ({c})" for k, _, c in PACKAGES))
    print(f"3 口径 × {n_seeds} seeds | 读数 = 配对逐种子中位数")
    print("=" * 92)

    t0 = time.time()
    records, verdicts = [], {}

    need_33 = not wanted or ("v33" in wanted or "v34" in wanted)
    need_35 = not wanted or ("v35" in wanted)

    enc = v35_env = None
    v35_blocker = None
    if need_33:
        print("  加载 MSA-4B / embedding-0.6b ...")
        big_tok = load_tokenizer(MSA_DIR)
        small_tok = load_tokenizer(EMB_DIR)
        big_model, D_big, _ = load_msa_big(n_layers=N_BIG_LAYERS)
        gc.collect()
        small_model, D_small, _ = load_small("embedding-0.6b", EMB_DIR)
        enc = {"big_tok": big_tok, "small_tok": small_tok, "big_model": big_model,
               "small_model": small_model, "D_big": D_big, "D_small": D_small}
        for key, texts in (("tr", TEXTS), ("ev", HO_TEXTS)):
            enc[f"h_big_{key}"] = encode(texts, big_tok, big_model, S).float()
            enc[f"h_sm_{key}"] = encode(texts, small_tok, small_model, S).float()
        print(f"  编码器就绪 ({time.time() - t0:.1f}s)\n")
    if need_35:
        # 必须用 **v35 自己的** get_tabular_repr(), 不能用 v36 修好的 get_tabldm_repr:
        # 后者返回逐样本 [12, 512], 而 v35 的语义是 `r.mean(dim=(0,1))` 折成 [512]
        # (这正是 v36 认定为"信息量为零"的那个塌缩)。审计要复现的是 v35 自己的行为,
        # 不是修好后的版本 —— 否则测的就不是 v35 了。
        from research.scale_heterogeneous.v35_container_entries import (
            EMB_DIR as _EMB, RER_DIR, load_big_container, get_tabular_repr)
        print("  加载 TabLDM 入口 + MSA 容器 ...")
        try:
            c_struct = get_tabular_repr()
        except ImportError as e:
            # tabldm 第三方包缺失不让其余包一起挂掉 —— 记下阻塞原因, 继续。
            v35_blocker = str(e).split("\n")[0]
            print(f"  ⚠️ v35 阻塞: {v35_blocker}")
            print("     (occams/Xiaomi-TabLDM 仓库只有权重与文档, 无 Python 源码)")
        else:
            c_sem, _ = load_backbone(_EMB, "embedding-0.6b", S)
            c_rank, _ = load_backbone(RER_DIR, "reranker-0.6b", S)
            gc.collect()
            v35_env = {"c_struct": c_struct, "c_sem": c_sem, "c_rank": c_rank,
                       "big_model": load_big_container(n_layers=N_BIG_LAYERS),
                       "tok": AutoTokenizer.from_pretrained(MSA_DIR)}
            print(f"  入口就绪 ({time.time() - t0:.1f}s)\n")

    for kind, label, claim in PACKAGES:
        if wanted and kind not in wanted:
            continue
        if kind == "v35" and v35_env is None:
            print(f"[v35] 跳过 —— 依赖缺失: {v35_blocker}")
            verdicts["v35"] = {"blocked": v35_blocker}
            continue
        print(f"[{kind}] {label}")
        per_scope = {s: {"treat": [], "base": []} for s in scopes}
        for seed in range(n_seeds):
            if kind == "v35":
                r = _mk_pair_v35(seed, v35_env, scopes)
            else:
                r = _mk_pair_v33v34(kind, seed, enc, scopes)
            for s in scopes:
                per_scope[s]["treat"].extend(r[s]["treat"])
                per_scope[s]["base"].extend(r[s]["base"])
            print(f"    seed {seed}: in-sample {r['in_sample']['treat'][0]:.4f} vs "
                  f"{r['in_sample']['base'][0]:.4f} | noise {r['heldout_noise']['treat'][0]:.4f} "
                  f"| stimulus {r['heldout_stimulus']['treat'][0]:.4f}")
        records.append({"name": kind, "lower_is_better": True, **per_scope})
        g_in = paired_gain(per_scope["in_sample"]["treat"], per_scope["in_sample"]["base"])
        print(f"    → in-sample {g_in['median_pct']:+.1f}% ({g_in['wins']}/{g_in['n']})")
        verdicts[kind] = {}
        for s in scopes[1:]:
            g = paired_gain(per_scope[s]["treat"], per_scope[s]["base"])
            v = leakage_verdict(g_in, g)
            keep = retention(g_in, g)
            verdicts[kind][s] = v
            print(f"    → {s.replace('heldout_', '')} {g['median_pct']:+.1f}% "
                  f"({g['wins']}/{g['n']}) {v}"
                  f"{'' if keep is None else f'  保留 {keep * 100:.0f}%'}")

    dest = os.environ.get(
        "V47_SH_OUT", "research/central_stability/v47_heldout_audit/results/scale_hetero_heldout.json")
    # 合并而非覆盖: 按 V47_SH_PACKAGES 分批跑 (先 v33,v34 后 v35) 时, 后一批原本会把
    # 前一批的记录整个盖掉, 下游的 metrics 合并就只剩最后那一批 —— 已踩过一次。
    if os.path.exists(dest):
        try:
            with open(dest, encoding="utf-8") as fh:
                prev = json.load(fh)
        except (OSError, json.JSONDecodeError):
            prev = {}
        kept = {r["name"]: r for r in prev.get("records", [])}
        kept.update({r["name"]: r for r in records})
        records = list(kept.values())
        verdicts = {**prev.get("verdicts", {}), **verdicts}
        claims_all = {**prev.get("claims", {}), **{k: c for k, _, c in PACKAGES}}
    else:
        claims_all = {k: c for k, _, c in PACKAGES}

    print("\n" + "=" * 92)
    print(audit_table(records))
    print("=" * 92)
    out = {"scopes": list(scopes), "seeds": n_seeds, "verdicts": verdicts,
           "claims": claims_all,
        "blocked": {"v35": v35_blocker} if v35_blocker else {},
           "records": [{"name": r["name"],
                        **{s: {k: [round(x, 6) for x in v] for k, v in r[s].items()}
                           for s in scopes}} for r in records]}
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    metrics = {
        "schema_version": "1.0", "version": "47.0",
        "package": "v47_heldout_audit", "line": "central_stability",
        "metric": "fuse_mse", "direction": "lower_is_better",
        "recorded_by": SH_OUT,
        "statistic": "配对逐种子中位数 + 逐种子胜出数 (不用均值 — v40 起的纪律)",
        "scopes": list(scopes), "seeds": n_seeds,
        "audited": ["v33", "v34", "v35"],
        "verdicts": verdicts,
        "blocked": {"v35": v35_blocker} if v35_blocker else {},
        "per_package": {
            r["name"]: {
                "in_sample_median_pct": round(paired_gain(
                    r["in_sample"]["treat"], r["in_sample"]["base"])["median_pct"], 2),
                **{s: {
                    "median_pct": round(paired_gain(r[s]["treat"], r[s]["base"])["median_pct"], 2),
                    "wins": f"{paired_gain(r[s]['treat'], r[s]['base'])['wins']}/"
                            f"{paired_gain(r[s]['treat'], r[s]['base'])['n']}",
                    "verdict": verdicts.get(r["name"], {}).get(s),
                    "retention": (lambda k: None if k is None else round(k, 3))(
                        retention(paired_gain(r["in_sample"]["treat"], r["in_sample"]["base"]),
                                  paired_gain(r[s]["treat"], r[s]["base"]))),
                } for s in scopes[1:]},
            } for r in records},
    }
    mpath = os.path.join(os.path.dirname(dest), "metrics.json")
    with open(mpath, "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"结果写入 {dest} / {mpath}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
