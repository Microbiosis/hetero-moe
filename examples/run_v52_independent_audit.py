"""V52.0 — 独立复算 v50/v51 的探针结论 (审计包).

> 归属路线: aligner (接 v50/v51 的审计链)
> 复现: bash run.sh v52

本包刻意**不 import** v49/v50/v51 的数据生成器、抬升或探针 —— 全部重写:
数据 (新词表 + 奇偶划分的未见组合留出)、抬升 (MLP+LayerNorm+GELU)、探针 (闭式岭回归,
与 v50/v51 的 SGD 逻辑回归是不同估计量)、训练循环。只保留"被审计的对象"本身:
三个真实底座 (MSA-4B / embedding-0.6b / reranker-0.6b) 与 v33 的融合层。

四个臂:
  A frozen_random     冻结随机线性 (论文原配置)
  B permuted_target   目标块置换 (类别→块的映射换成固定无不动点错位)
  C learned_lift      非线性可训练抬升 (有意义空间)
  D randomized_target 目标逐样本随机 (C2 的反向对照: 探针若无效这里会露馅)

预登记判决见 research/aligner/v52_independent_audit/SPEC.md 与 audit.PREDICTIONS。
"""
import sys, os, json, gc, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from research._primitives.scale_hetero_loaders import (
    MSA_DIR, EMB_DIR, RER_DIR, load_msa_big, load_small, load_tokenizer,
)
from research.scale_heterogeneous.v33_scale_hetero import CrossModalFusionLayer
from research.aligner.v52_independent_audit.audit import (
    CLASSES, ARMS, PREDICTIONS, MLPLift, LinearLift, make_split, build_targets, probe_accuracy,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_CLASSES = len(CLASSES)
S_TOK = int(os.environ.get("V52_TOK", "8"))
N_PER_CLASS = int(os.environ.get("V52_NPC", "16"))
BATCH = int(os.environ.get("V52_BATCH", "288"))
STEPS = int(os.environ.get("V52_STEPS", "300"))
LR = 1e-2
LR_BIG = float(os.environ.get("V52_LR_BIG", "1e-3"))
SEEDS = int(os.environ.get("V52_SEEDS", "5"))
SEED_BASE = int(os.environ.get("V52_SEED_BASE", "1000"))     # 与 v49/v50/v51 错开
D_SHAREDS = [int(x) for x in os.environ.get("V52_DSHARED", "128,256,512").split(",")]
DEVICE = torch.device(os.environ.get("V52_DEVICE", "cpu"))
EXPERTS = ["big", "small1", "small2"]


def _encode_on(texts, tok, model, device):
    enc = tok(texts, padding="max_length", truncation=True, max_length=S_TOK, return_tensors="pt")
    with torch.no_grad():
        h = model(enc["input_ids"].to(device)).last_hidden_state
    return h.detach().float()


def encode_all(device, texts):
    big_tok, small_tok = load_tokenizer(MSA_DIR), load_tokenizer(EMB_DIR)
    try:
        rer_tok = load_tokenizer(RER_DIR)
    except Exception:
        rer_tok = small_tok
    out = {}
    for name, tok, loader, args in (
        ("big", big_tok, load_msa_big, ()),
        ("small1", small_tok, load_small, ("embedding-0.6b", EMB_DIR)),
        ("small2", rer_tok, load_small, ("reranker-0.6b", RER_DIR)),
    ):
        model, dim, _ = loader(*args)
        model = model.to(device)
        out[name] = _encode_on(texts, tok, model, device)
        del model
        gc.collect()
    return out


def run_arm(arm, d_shared, feats, feats_ho, labels, labels_ho, seed, device):
    lift_kind, variant = ARMS[arm]
    torch.manual_seed(seed)
    Dm = {e: feats[e].shape[-1] for e in EXPERTS}
    projs = {e: (MLPLift(d_shared, Dm[e]) if lift_kind == "mlp" else LinearLift(d_shared, Dm[e])).to(device)
             for e in EXPERTS}
    if lift_kind == "linear":
        for e in EXPERTS:
            for p in projs[e].parameters():
                p.requires_grad_(False)          # 冻结随机臂: 投影完全冻结
    layer = CrossModalFusionLayer(d_shared, 4 * d_shared, len(EXPERTS), 1,
                                  [Dm[e] for e in EXPERTS]).to(device)
    for p in layer.P_m:
        p.requires_grad_(False)

    # 目标: 类别块; B 臂做块置换, D 臂逐样本随机
    perm = None
    if variant == "permuted_target":
        perm = list(range(N_CLASSES))
        rng = np.random.default_rng(seed + 7)
        while True:
            p = rng.permutation(N_CLASSES).tolist()
            if all(p[i] != i for i in range(N_CLASSES)):
                perm = p
                break
    tgt_map = (lambda c: perm[c]) if perm else (lambda c: c)
    tr_t = build_targets([tgt_map(c) for c in labels], d_shared, N_CLASSES, device,
                         n_tokens=S_TOK)
    ho_t = build_targets([tgt_map(c) for c in labels_ho], d_shared, N_CLASSES, device,
                         randomized=(variant == "randomized_target"), seed=seed, n_tokens=S_TOK)

    def project(feats):
        return torch.cat([projs[e](feats[e]) for e in EXPERTS], dim=0)

    params = [layer.W_router] + list(layer.gammas) + list(layer.betas) + list(layer.alphas)
    if lift_kind == "mlp":
        for e in EXPERTS:
            params += list(projs[e].parameters())
    big_ids = {id(p) for p in params if p.numel() > 4096}
    small_ids = {id(p) for p in params if p.numel() <= 4096}
    opt = torch.optim.AdamW(
        [{"params": [p for p in params if id(p) in small_ids], "lr": LR},
         {"params": [p for p in params if id(p) in big_ids], "lr": LR_BIG}],
        lr=LR, weight_decay=1e-4)

    # ⚠️ 投影是按底座 cat 的 → 批次是 [M*n, S, d]; 目标必须同样按底座平铺
    tr_t_all = torch.cat([tr_t] * len(EXPERTS), dim=0)
    ho_t_all = torch.cat([ho_t] * len(EXPERTS), dim=0)
    n_total = feats[EXPERTS[0]].shape[0] * len(EXPERTS)
    g = torch.Generator().manual_seed(seed * 977 + 13)
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        xb, tb = [], []
        per = n_total // len(EXPERTS)
        for m, e in enumerate(EXPERTS):
            loc = torch.randint(0, per, (max(1, BATCH // len(EXPERTS)),), generator=g).to(device)
            xb.append(projs[e](feats[e][loc]))
            tb.append(tr_t[loc])
        loss = F.mse_loss(layer(torch.cat(xb, 0)), torch.cat(tb, 0))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

    with torch.no_grad():
        x_ho = project(feats_ho)
        y_ho, _ = layer.forward_with_routing(x_ho)
        fuse = F.mse_loss(y_ho, ho_t_all).item()
        x_det = x_ho.detach()
        tot = torch.zeros_like(x_ho)
        singles = []
        for m in range(len(EXPERTS)):
            h_a = x_det * layer.gammas[m] + layer.betas[m]
            f = layer._swiglu(h_a, layer.w_gates[m], layer.w_ups[m], layer.w_downs[m])
            tot = tot + f
            singles.append(F.mse_loss(x_ho + layer.alphas[m] * f, ho_t_all).item())
        avg = F.mse_loss(x_ho + tot / len(EXPERTS), ho_t_all).item()
        best_single = min(singles)
        # 探针 (岭回归, 独立估计量): 源专家身份 / 目标块
        y_flat = y_ho.reshape(-1, d_shared).detach()
        src = torch.tensor([m for m in range(len(EXPERTS))
                            for _ in range(feats_ho[EXPERTS[0]].shape[0] * S_TOK)]).to(device)
        blk = ho_t.shape[-1] // N_CLASSES
        tgt_idx = ho_t_all.reshape(-1, d_shared).argmax(-1) // blk   # [n*S] 目标块下标
        p_src = probe_accuracy(y_flat, src, len(EXPERTS), seed=seed)
        p_cls = probe_accuracy(y_flat, tgt_idx, N_CLASSES, seed=seed)
    return {"held_out": {"fuse": fuse, "avg": avg, "best_single": best_single},
            "probe": {"source": p_src, "class": p_cls},
            "in_sample_fuse": F.mse_loss(layer(project(feats)), tr_t_all).item(),
            "perm": perm, "d_shared": d_shared}


def _load_cells():
    """从桶里取回已算好的格 (Space 容器是临时的)。"""
    try:
        import os as _os
        from huggingface_hub import list_bucket_tree, download_bucket_files
        import tempfile
        bucket = _os.environ.get("V52_BUCKET", "redmingwei/hetero-moe")
        prefix = _os.environ.get("V52_BUCKET_PREFIX", "v52_artifacts")
        tok = _os.environ.get("HF_TOKEN")
        paths = [getattr(it, "path", None) for it in
                 list_bucket_tree(bucket, prefix=f"{prefix}/cells/", recursive=True)]
        paths = [p for p in paths if p and p.endswith(".json")]
        if not paths:
            return {}
        tmp = tempfile.mkdtemp()
        download_bucket_files(bucket, files=[(p, os.path.join(tmp, os.path.basename(p)))
                                            for p in paths], token=tok,
                              raise_on_missing_files=False)
        out = {}
        for p in paths:
            f = os.path.join(tmp, os.path.basename(p))
            if os.path.exists(f):
                d = json.load(open(f, encoding="utf-8"))
                out[d["key"]] = d
        return out
    except Exception as e:
        print(f"  (warn) 未能取回桶里已有格: {repr(e)[:110]}")
        return {}


def _push_cell(key, rows):
    """一格算完立刻进桶 (抗 Space 重建)。"""
    try:
        import os as _os
        from huggingface_hub import batch_bucket_files
        bucket = _os.environ.get("V52_BUCKET", "redmingwei/hetero-moe")
        prefix = _os.environ.get("V52_BUCKET_PREFIX", "v52_artifacts")
        tok = _os.environ.get("HF_TOKEN")
        payload = json.dumps({"key": key, "rows": rows}, ensure_ascii=False, indent=2)
        batch_bucket_files(bucket, add=[(payload.encode(), f"{prefix}/cells/{key}.json")],
                           token=tok or None)
    except Exception as e:
        print(f"  [warn] 该格未推上桶: {repr(e)[:120]}")


def main():
    print("=" * 78)
    print("V52.0 — 独立复算 v50/v51 的探针结论")
    print("=" * 78)
    print("臂:", list(ARMS))
    print(f"协议: 独立数据生成器 (8 类, 奇偶划分 → 留出是**未见组合**), "
          f"n_per_class={N_PER_CLASS}, D_shared={D_SHAREDS}")
    print(f"STEPS={STEPS} SEEDS={SEEDS} (种子基址 {SEED_BASE}, 与 v49/v50/v51 错开)")
    print("预登记判决:")
    for k, v in PREDICTIONS.items():
        print(f"   {k}: {v}")

    tr_pairs, ho_pairs = make_split(N_PER_CLASS, seed=0)
    tr_texts = [t for _, t in tr_pairs]
    ho_texts = [t for _, t in ho_pairs]
    labels = [c for c, _ in tr_pairs]
    labels_ho = [c for c, _ in ho_pairs]
    print(f"\n  训练文本 {len(tr_texts)} 条 / 留出 {len(ho_texts)} 条 "
          f"(组合交集 {len(set(tr_texts) & set(ho_texts))})")
    t0 = time.time()
    enc = encode_all(DEVICE, tr_texts)
    enc_ho = encode_all(DEVICE, ho_texts)
    print(f"  编码完成 ({time.time()-t0:.0f}s)")

    results = {}
    done_cells = _load_cells()
    if done_cells:
        print(f"  从桶里恢复 {len(done_cells)} 格已完成: {sorted(done_cells)}")
    for d_shared in D_SHAREDS:
        for arm in ARMS:
            key = f"D{d_shared}:{arm}"
            if key in done_cells:
                results[key] = done_cells[key]
                print(f"  [D{d_shared:>3}] {arm:18s} (已算过, 跳过)")
                continue
            t1 = time.time()
            rows = [run_arm(arm, d_shared, enc, enc_ho, labels, labels_ho,
                            SEED_BASE + s, DEVICE) for s in range(SEEDS)]
            results[key] = {"rows": rows}
            _push_cell(key, rows)
            n = len(rows)
            g = lambda k: sum(r["held_out"][k] for r in rows) / n
            margin = (g("best_single") - g("fuse")) / max(g("best_single"), 1e-9) * 100
            ps = sum(r["probe"]["source"] for r in rows) / n
            pc = sum(r["probe"]["class"] for r in rows) / n
            print(f"  [D{d_shared:>3}] {arm:18s} fuse={g('fuse'):8.4f} 单专家={g('best_single'):8.4f} "
                  f"margin={margin:+7.1f}% | 探针 源={ps:.3f} 目标块={pc:.3f}  ({time.time()-t1:.0f}s)",
                  flush=True)

    # 判定
    print("\n" + "=" * 78)
    print("预登记判决")
    print("=" * 78)
    verdicts = {}
    for d_shared in D_SHAREDS:
        def arm_stats(arm):
            rows = results[f"D{d_shared}:{arm}"]["rows"]
            n = len(rows)
            g = lambda k: sum(r["held_out"][k] for r in rows) / n
            margin = (g("best_single") - g("fuse")) / max(g("best_single"), 1e-9) * 100
            per_seed_margin = [(r["held_out"]["best_single"] - r["held_out"]["fuse"])
                               / max(r["held_out"]["best_single"], 1e-9) * 100 for r in rows]
            src = [r["probe"]["source"] for r in rows]
            cls = [r["probe"]["class"] for r in rows]
            return dict(margin=margin, per_seed_margin=per_seed_margin,
                        src=src, cls=cls)
        A, B, C, D = (arm_stats(a) for a in ("frozen_random", "permuted_target",
                                             "learned_lift", "randomized_target"))
        c1a = sum(1 for s, c in zip(A["src"], A["cls"]) if s - c >= 0.30) >= 4
        c1b = abs(B["margin"] - A["margin"]) <= 10.0
        c1c = C["margin"] <= A["margin"] / 2
        c2 = max(D["cls"]) <= 0.50
        print(f"D_shared={d_shared}:")
        print(f"   C1a 源−类别可解码差 ≥0.30 (≥4/5 seeds): {'通过' if c1a else '未通过'} "
              f"(每 seed 差: {[round(s-c,2) for s,c in zip(A['src'],A['cls'])]})")
        print(f"   C1b 目标置换后 margin 变化 ≤10pp: {'通过' if c1b else '未通过'} "
              f"({A['margin']:+.1f}% → {B['margin']:+.1f}%)")
        print(f"   C1c 有意义空间 margin ≤ 冻结臂一半: {'通过' if c1c else '未通过'} "
              f"({A['margin']:+.1f}% → {C['margin']:+.1f}%)")
        print(f"   C2  随机目标臂 类别可解码 ≤0.50: {'通过' if c2 else '未通过 (探针无效!)'} "
              f"(每 seed: {[round(x,2) for x in D['cls']]})")
        verdicts[f"D{d_shared}"] = dict(C1a=bool(c1a), C1b=bool(c1b), C1c=bool(c1c), C2=bool(c2))
    all_pass = all(all(v.values()) for v in verdicts.values())
    print(f"\n总判定: {'审计通过 (V1)' if all_pass else '审计未通过 — 相应主张须回退'}")
    print("  (任一 D_shared 上 C2 失败 → 探针无效 → v50/v51 的探针读数全部作废)")

    out_dir = os.path.join(ROOT, "research/aligner/v52_independent_audit/results")
    os.makedirs(out_dir, exist_ok=True)
    mpath = os.path.join(out_dir, "metrics.json")
    payload = {"schema_version": "1.0", "version": "52.0", "package": "v52_independent_audit",
               "line": "aligner",
               "config": {"d_shareds": D_SHAREDS, "steps": STEPS, "seeds": SEEDS,
                          "seed_base": SEED_BASE, "n_per_class": N_PER_CLASS,
                          "n_classes": N_CLASSES, "tokens": S_TOK},
               "predictions": PREDICTIONS, "verdicts": verdicts, "all_pass": all_pass,
               "results": results,
               "notes": {"independence": "数据/抬升/探针/训练循环全部重写, 不 import v49/v50/v51 的实现; "
                                        "只保留被审计的对象 (三个真实底座 + v33 融合层)",
                         "heldout": "留出是**未见组合** (奇偶划分), 比 v50/v51 的'换扰动种子'更严",
                         "probe": "闭式岭回归 (与 v50/v51 的 SGD 逻辑回归是不同估计量)"}}
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n指标已落盘: {mpath}")


if __name__ == "__main__":
    main()
