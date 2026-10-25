"""V49.0 — 投影空间的三重检验: 非线性 × 条件依赖 × 样本量 (2×2 析因 + 样本量轴)。

> 归属路线: aligner (接 v48)
> 上游: v48_projection_space (空间尺寸诊断 + 等参数三臂可训练性 A/B)
> 复现: bash run.sh v49

v48 留下的三个悬案, v49 一并做掉 (交叉照应):

  轴 N (非线性): 冗余存在 (CKA 0.48) 但联合 PCA 抓不到 → 需**非线性**映射。
      v48 的 B 臂 P_m 是线性的, 所以它学不到那份结构。
  轴 C (条件依赖): 空间要"随着条件的改变而改变" → 投影由**内容派生条件**经
      超网络低秩调制 (而不是固定矩阵)。
  轴 S (样本量): 18-36 样本 vs 1.18M 参数必然记忆 → 样本量做成自变量
      (32 / 256 / 1024, 由组合式槽位生成器提供, 训练与留出**文本零交集**)。

交叉照应的逻辑 (三轴互相印证, 不是三个独立实验):
  · 若 B 轴的收益全是记忆 → 过拟合间隙随 S 增大而**收窄**, 且 held-out 增益
    随 S 增大而**上升**; 若增益不随 S 变化 → 它是容量假象, 不是对齐。
  · 若"非线性"真的在抓那份非线性冗余 → N 臂的增益应**随 S 增大而放大**
    (样本少时非线性只会更快地记忆); 否则 N 的作用就是多参数。
  · 若"条件依赖"真的在起作用 → **打乱条件**必须显著退化 (shuffle-c 对照),
    且该退化幅度随 S 增大而**减小** (样本多才学得出可迁移的条件调制)。
  · 若 N 与 C 是同一机制 → 两者同时开时增益**不可叠加**; 若正交 → 可叠加。

容量对齐 (v41 教训): 线性臂 d_shared×D_m vs 非线性臂 D_m·h + h·d_shared,
h 按参数量反解 → 两臂参数量差 <0.3%, N 轴的差异不能被"参数更多"解释。

算力对齐: 固定 minibatch (288/step) + 固定步数 → 样本量轴测的是
**固定算力下的数据多样性**, 不是算力。
"""
import sys, os, json, gc, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from research._primitives.scale_hetero_loaders import (
    MSA_DIR, EMB_DIR, RER_DIR, load_msa_big, load_small, load_tokenizer,
)
from research.scale_heterogeneous.v33_scale_hetero import CrossModalFusionLayer
from research.aligner.v48_projection_space.space import (
    union_report, pairwise_canonical_correlations,
)
from research.aligner.v49_projection_space.data import Split, SAMPLE_SIZES, CLASSES
from research.aligner.v49_projection_space.mechanism import (
    ARMS, FROZEN_ARMS, build_projector, param_report, permute_cond,
)

D_SHARED = 256
M, K = 3, 1
S_TOK = int(os.environ.get("V49_TOK", "8"))
BATCH = int(os.environ.get("V49_BATCH", "288"))
STEPS = int(os.environ.get("V49_STEPS", "300"))
LR = 1e-2
LR_BIG = float(os.environ.get("V49_LR_BIG", "1e-3"))
N_SEEDS = int(os.environ.get("V49_SEEDS", "5"))
N_CLASSES = len(CLASSES)
ONLY_SIZES = [x for x in os.environ.get("V49_SIZES", "").split(",") if x]
ONLY_ARMS = [x for x in os.environ.get("V49_ARMS", "").split(",") if x]
DEVICE = torch.device(os.environ.get("V49_DEVICE", "cpu"))


# ----------------------------------------------------------------------
# 数据
# ----------------------------------------------------------------------

def slice_indices(pairs, n_per_class):
    """从最大档里按类各取前 n 条 → 返回**全局下标**列表 (嵌套的小档)。

    ⚠️ pairs 是**按类分组**的 (class 0 的全部, 然后 class 1 ...), 所以
    "类内第 k 条"的全局位置 = 该类块的起点 + k。直接拿类内序号当全局下标
    会取到别的类的文本 (第一版就是这么错的)。
    """
    starts, k = {}, 0
    for c, _ in pairs:
        if c not in starts:
            starts[c] = k
        k += 1
    idx, cnt = [], {}
    for c, _ in pairs:
        j = cnt.get(c, 0)
        cnt[c] = j + 1
        if j < n_per_class:
            idx.append(starts[c] + j)
    return idx


def build_targets(labels, device):
    """目标 = 类别块 (8 类 × 32 维), 同一文本在三个底座上**同目标** (要求跨底座对齐)。"""
    blk = D_SHARED // N_CLASSES
    tgt = torch.zeros(len(labels), S_TOK, D_SHARED)
    for i, c in enumerate(labels):
        tgt[i, :, c * blk:(c + 1) * blk] = 1.0
    return tgt.to(device)


# ----------------------------------------------------------------------
# 编码
# ----------------------------------------------------------------------

def _encode_on(texts, tok, model, device):
    enc = tok(texts, padding="max_length", truncation=True, max_length=S_TOK,
              return_tensors="pt")
    with torch.no_grad():
        h = model(enc["input_ids"].to(device)).last_hidden_state
    # ⚠️ 底座是 bf16 (load_* 用 bfloat16), 而投影器/融合层是 fp32 →
    #    不转就会在 F.linear 里炸 "expected m1 and m2 to have the same dtype"
    return h.detach().float()


def encode_all(device, n_per_class):
    """加载三个底座 → 编码**当前档**的 train/heldout 文本。

    只编码本档需要的量 (档位是嵌套的, 但每个 combo 独立跑, 直接按本档建
    split 最省: 32 档只编 32 条, 1024 档才编 1024 条)。
    """
    sp = Split(n_per_class, seed=0)
    tr_texts, ho_texts = sp.texts("train"), sp.texts("heldout")
    tr_labels, ho_labels = sp.labels("train"), sp.labels("heldout")

    big_tok = load_tokenizer(MSA_DIR)
    small_tok = load_tokenizer(EMB_DIR)
    try:
        rer_tok = load_tokenizer(RER_DIR)
    except Exception:
        rer_tok = small_tok

    def load_and_encode(tok, loader, *args):
        model, dim, _ = loader(*args)
        model = model.to(device)
        a = _encode_on(tr_texts, tok, model, device)
        b = _encode_on(ho_texts, tok, model, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return a, b, dim

    out = {}
    out["big"] = load_and_encode(big_tok, load_msa_big)
    out["small1"] = load_and_encode(small_tok, load_small, "embedding-0.6b", EMB_DIR)
    try:
        out["small2"] = load_and_encode(rer_tok, load_small, "reranker-0.6b", RER_DIR)
    except Exception as e:
        print(f"  [警告] reranker 不可用 ({e}); 小2 退化为与小1同 backbone")
        out["small2"] = (out["small1"][0], out["small1"][1], out["small1"][2])
    return out, (tr_labels, ho_labels), (tr_texts, ho_texts)


# ----------------------------------------------------------------------
# 训练 / 评估
# ----------------------------------------------------------------------

def run_one(arm, size_name, n_per_class, enc, labels, seed, device):
    torch.manual_seed(seed)
    experts = ["big", "small1", "small2"]
    raw = {"big": enc["big"][0], "small1": enc["small1"][0], "small2": enc["small2"][0]}
    raw_ho = {"big": enc["big"][1], "small1": enc["small1"][1], "small2": enc["small2"][1]}
    tr_labels, ho_labels = labels
    Dm = {e: raw[e].shape[-1] for e in experts}

    projs = {e: build_projector(arm, D_SHARED, Dm[e]).to(device) for e in experts}
    for e in experts:
        if arm in FROZEN_ARMS:
            for p in projs[e].parameters():
                p.requires_grad_(False)

    layer = CrossModalFusionLayer(D_SHARED, 4 * D_SHARED, M, K,
                                   [Dm[e] for e in experts]).to(device)
    for p in layer.P_m:
        p.requires_grad_(False)

    # 每底座 n 条 (同序同文本) + 每文本一个目标块
    tr_x = {e: raw[e].to(device) for e in experts}
    ho_x = {e: raw_ho[e].to(device) for e in experts}
    tr_t = build_targets(tr_labels, device)          # [n, S, D_shared]
    ho_t = build_targets(ho_labels, device)          # [n, S, D_shared]
    n = tr_t.shape[0]

    def project(feats, shuffle_seed=None):
        """按底座投影后拼成一批: [M*n, S, D_shared]"""
        xs = []
        for e in experts:
            x = feats[e]
            p = projs[e]
            if hasattr(p, "condition_from"):
                c = p.condition_from(x)
                if shuffle_seed is not None:
                    c = permute_cond(c, shuffle_seed)
                xs.append(p(x, c))
            else:
                xs.append(p(x))
        return torch.cat(xs, dim=0)

    params = [layer.W_router] + list(layer.gammas) + list(layer.betas) + list(layer.alphas)
    for e in experts:
        params += [p for p in projs[e].parameters() if p.requires_grad]
    big_ids = {id(p) for p in params if p.numel() > 4096}
    small_ids = {id(p) for p in params if p.numel() <= 4096}
    opt = torch.optim.AdamW(
        [{"params": [p for p in params if id(p) in small_ids], "lr": LR},
         {"params": [p for p in params if id(p) in big_ids], "lr": LR_BIG}],
        lr=LR, weight_decay=1e-4)

    # ---- 训练: 固定 minibatch (每底座 bs/M 条) ----
    bs_per = max(1, BATCH // M)
    g = torch.Generator().manual_seed(seed * 977 + 13)
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        xb, tb = [], []
        for e in experts:
            loc = torch.randint(0, n, (bs_per,), generator=g).to(device)
            p = projs[e]
            if hasattr(p, "condition_from"):
                xb.append(p(tr_x[e][loc], p.condition_from(tr_x[e][loc])))
            else:
                xb.append(p(tr_x[e][loc]))
            tb.append(tr_t[loc])
        xb = torch.cat(xb, dim=0)
        tb = torch.cat(tb, dim=0)
        y = layer(xb)
        loss = F.mse_loss(y, tb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

    # ---- 评估: 目标要按底座重复 M 次 (与 project 的拼接顺序一致) ----
    def evaluate(feats, tgts, shuffle_seed=None):
        with torch.no_grad():
            x_shared = project(feats, shuffle_seed)
            t_all = torch.cat([tgts] * M, dim=0)
            y, alpha = layer.forward_with_routing(x_shared)
            fuse = F.mse_loss(y, t_all).item()
            x_det = x_shared.detach()
            sbig = avg = 0.0
            for m in range(M):
                h_a = x_det * layer.gammas[m] + layer.betas[m]
                f = layer._swiglu(h_a, layer.w_gates[m], layer.w_ups[m], layer.w_downs[m])
                if m == 0:
                    sbig = F.mse_loss(x_shared + layer.alphas[0] * f, t_all).item()
                avg += f
            avg = F.mse_loss(x_shared + avg / M, t_all).item()
            route = [int((alpha.argmax(dim=-1) == m).sum().item()) for m in range(M)]
        return {"fuse": fuse, "single_big": sbig, "avg": avg,
                "trivial": F.mse_loss(torch.zeros_like(t_all), t_all).item(),
                "route": route}

    res = {"in_sample": evaluate(tr_x, tr_t), "held_out": evaluate(ho_x, ho_t)}
    if hasattr(projs["big"], "condition_from"):
        res["held_out_shuffled_cond"] = evaluate(ho_x, ho_t, shuffle_seed=seed + 7)
    res["params"] = {e: param_report(projs[e]) for e in experts}
    res["n_train"] = n * M
    res["n_heldout"] = ho_t.shape[0] * M
    return res


# ----------------------------------------------------------------------
# 诊断 (CKA, 样本量大了统计量才站得住)
# ----------------------------------------------------------------------

def diagnose(enc, labels):
    feats = [enc["big"][0], enc["small1"][0], enc["small2"][0]]
    rep = union_report(feats, ["big", "small1", "small2"])
    ccs = pairwise_canonical_correlations(feats, top=3, n_null=20, seed=0)
    return {
        "union": {k: v for k, v in rep["union"].items() if k != "eigvals"},
        "per_expert": {nm: {k: v for k, v in r.items() if k != "eigvals"}
                       for nm, r in rep["per_expert"].items()},
        "canonical_correlations": [
            {"pair": list(c["pair"]), "cka": c["cka"],
             "cka_null_max": c["cka_null_max"], "cka_exceeds_null": c["cka_exceeds_null"],
             "cc1": c["cc1"]} for c in ccs],
        "cka_mean": sum(c["cka"] for c in ccs) / len(ccs),
        "cka_null_max_mean": sum(c["cka_null_max"] for c in ccs) / len(ccs),
    }


# ----------------------------------------------------------------------

def _partial(results, diag):
    """增量落盘 (每格算完就写) —— Space 被回收/重启时不丢已算的格。"""
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "research", "aligner", "v49_projection_space", "results")
    os.makedirs(out_dir, exist_ok=True)
    mpath = os.path.join(out_dir, "metrics.json")
    payload = {"schema_version": "1.0", "version": "49.0",
               "package": "v49_projection_space", "line": "aligner",
               "config": {"d_shared": D_SHARED, "batch": BATCH, "steps": STEPS,
                          "lr": LR, "lr_big": LR_BIG, "seeds": N_SEEDS,
                          "tokens": S_TOK, "n_classes": N_CLASSES,
                          "sample_sizes": SAMPLE_SIZES},
               "diagnostics": diag, "results": results}
    if os.path.exists(mpath):
        try:
            with open(mpath, encoding="utf-8") as f:
                old_payload = json.load(f)
            merged = dict(old_payload)
            merged.setdefault("results", {}).update(results)
            merged["diagnostics"] = diag
            payload = merged
        except Exception:
            pass
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    print("=" * 78)
    print("V49.0 — 投影空间: 非线性 × 条件依赖 × 样本量")
    print("=" * 78)
    print(f"臂: {list(ARMS)}")
    print(f"样本量档: {SAMPLE_SIZES} (每底座) | BATCH={BATCH} STEPS={STEPS} SEEDS={N_SEEDS}")
    print(f"类别: {N_CLASSES} | 目标块: {D_SHARED // N_CLASSES} 维 | tokens={S_TOK}")

    t0 = time.time()
    # 只跑选中的档 (每档独立编码, 组合可分批/可续跑)
    sizes = {k: v for k, v in SAMPLE_SIZES.items()
             if not ONLY_SIZES or k in ONLY_SIZES}
    enc, labels, texts = encode_all(DEVICE, max(sizes.values()))
    print(f"  编码完成 ({time.time()-t0:.0f}s): " +
          ", ".join(f"{k}{tuple(v[0].shape)}" for k, v in enc.items()))

    diag = diagnose(enc, labels)
    print(f"\n[诊断] CKA 均值 = {diag['cka_mean']:.4f} (零分布上限均值 {diag['cka_null_max_mean']:.4f})")
    for c in diag["canonical_correlations"]:
        print(f"   {c['pair']}: CKA={c['cka']:.4f} (null max {c['cka_null_max']:.4f}) "
              f"{'✔' if c['cka_exceeds_null'] else '✘'}")
    u = diag["union"]
    print(f"[诊断] 并集: 99%维数={u['k_1pct']} 95%维数={u['k_5pct']} 有效秩={u['eff_rank']:.1f} "
          f"(N={u['n']} token)")

    results = {}
    for size_name, n_per_class in SAMPLE_SIZES.items():
        if ONLY_SIZES and size_name not in ONLY_SIZES:
            continue
        for arm in ARMS:
            if ONLY_ARMS and arm not in ONLY_ARMS:
                continue
            t1 = time.time()
            rows = [run_one(arm, size_name, n_per_class, enc, labels, seed, DEVICE)
                    for seed in range(N_SEEDS)]
            results[f"{size_name}:{arm}"] = {"rows": rows}
            _partial(results, diag)          # 每格算完立刻落盘 (抗重启)
            held = sum(r["held_out"]["fuse"] for r in rows) / len(rows)
            ins = sum(r["in_sample"]["fuse"] for r in rows) / len(rows)
            triv = rows[0]["held_out"]["trivial"]
            sbig = sum(r["held_out"]["single_big"] for r in rows) / len(rows)
            sh = ""
            if "held_out_shuffled_cond" in rows[0]:
                shf = sum(r["held_out_shuffled_cond"]["fuse"] for r in rows) / len(rows)
                sh = f" | shuffle-c={shf:.3f} (退化 {(shf-held)/max(held,1e-9)*100:+.1f}%)"
            print(f"  [{size_name:6s}] {arm:16s} held={held:8.4f} in={ins:8.4f} "
                  f"平凡={triv:.4f} 间隙={(held-ins)/max(held,1e-9)*100:+6.1f}% "
                  f"vs单大={(sbig-held)/max(sbig,1e-9)*100:+6.1f}%{sh}  ({time.time()-t1:.0f}s)")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "research", "aligner", "v49_projection_space", "results")
    os.makedirs(out_dir, exist_ok=True)
    mpath = os.path.join(out_dir, "metrics.json")
    payload = {
        "schema_version": "1.0", "version": "49.0", "package": "v49_projection_space",
        "line": "aligner",
        "config": {"d_shared": D_SHARED, "batch": BATCH, "steps": STEPS, "lr": LR,
                   "lr_big": LR_BIG, "seeds": N_SEEDS, "tokens": S_TOK,
                   "n_classes": N_CLASSES, "sample_sizes": SAMPLE_SIZES},
        "diagnostics": diag, "results": results,
        "notes": {
            "task": "跨底座同目标 (8 类块): 同一文本在三个底座上目标相同 → 要求对齐",
            "capacity": "线性臂 d_shared×D_m vs 非线性臂 D_m·h+h·d_shared (h 反解, 参数差 <0.3%)",
            "compute": "固定 minibatch 288 + 固定步数 → 样本量轴测数据多样性, 不是算力",
            "cond_control": "shuffle-c: 打乱条件后必须退化, 否则条件是装饰",
        },
    }
    # 与已有文件合并 (分批跑 / 被重启时不丢已算的格)
    if os.path.exists(mpath):
        try:
            with open(mpath, encoding="utf-8") as f:
                old_payload = json.load(f)
            merged = dict(old_payload)
            merged.setdefault("results", {}).update(payload["results"])
            merged["diagnostics"] = payload["diagnostics"]
            merged["config"] = payload["config"]
            merged["notes"] = payload["notes"]
            payload = merged
        except Exception:
            pass
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n指标已落盘: {mpath}")


if __name__ == "__main__":
    main()
