"""V53.0 — 第三审: 独立复算 v50/v51/v52 的结论 (第三套实现)。

> 归属路线: aligner (审计链: v50 → v51 → v52 → **v53 收口**)
> 复现: bash run.sh v53

与 v52 的差别全在"可变量"上 (数据划分/探针族/抬升实现/种子/D_shared 全部换过),
外加一个**正面量**: 跨底座一致性 (同一文本在三个底座上的输出有多一致)。

预登记判决 (research/aligner/v53_third_audit/SPEC.md 与 audit.PREDICTIONS)。
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
from research.aligner.v53_third_audit.audit import (
    CLASSES, ARMS, PREDICTIONS, ResidualLift, LinearLift, make_split, build_targets,
    probe_accuracy_all, consistency, N_CLASSES,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S_TOK = int(os.environ.get("V53_TOK", "8"))
N_PER_CLASS = int(os.environ.get("V53_NPC", "16"))
BATCH = int(os.environ.get("V53_BATCH", "288"))
STEPS = int(os.environ.get("V53_STEPS", "200"))
LR = 1e-2
LR_BIG = float(os.environ.get("V53_LR_BIG", "1e-3"))
SEEDS = int(os.environ.get("V53_SEEDS", "5"))
SEED_BASE = int(os.environ.get("V53_SEED_BASE", "2000"))
D_SHAREDS = [int(x) for x in os.environ.get("V53_DSHARED", "96,384").split(",")]
DEVICE = torch.device(os.environ.get("V53_DEVICE", "cpu"))
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
    projs = {e: (ResidualLift(d_shared, Dm[e]) if lift_kind == "residual"
                 else LinearLift(d_shared, Dm[e])).to(device) for e in EXPERTS}
    if lift_kind == "linear":
        for e in EXPERTS:
            for p in projs[e].parameters():
                p.requires_grad_(False)

    layer = CrossModalFusionLayer(d_shared, 4 * d_shared, len(EXPERTS), 1,
                                  [Dm[e] for e in EXPERTS]).to(device)
    for p in layer.P_m:
        p.requires_grad_(False)

    perm = None
    if variant == "permuted_target":
        rng = np.random.default_rng(seed + 7)
        while True:
            p = rng.permutation(N_CLASSES).tolist()
            if all(p[i] != i for i in range(N_CLASSES)):
                perm = p
                break
    tgt_map = (lambda c: perm[c]) if perm else (lambda c: c)
    tr_t = build_targets([tgt_map(c) for c in labels], d_shared, N_CLASSES, device, n_tokens=S_TOK)
    ho_t = build_targets([tgt_map(c) for c in labels_ho], d_shared, N_CLASSES, device,
                         randomized=(variant == "randomized_target"), seed=seed, n_tokens=S_TOK)
    tr_t_all = torch.cat([tr_t] * len(EXPERTS), 0)
    ho_t_all = torch.cat([ho_t] * len(EXPERTS), 0)

    def project(fs):
        return torch.cat([projs[e](fs[e]) for e in EXPERTS], 0)

    params = [layer.W_router] + list(layer.gammas) + list(layer.betas) + list(layer.alphas)
    if lift_kind == "residual":
        for e in EXPERTS:
            params += list(projs[e].parameters())
    big_ids = {id(p) for p in params if p.numel() > 4096}
    small_ids = {id(p) for p in params if p.numel() <= 4096}
    opt = torch.optim.AdamW(
        [{"params": [p for p in params if id(p) in small_ids], "lr": LR},
         {"params": [p for p in params if id(p) in big_ids], "lr": LR_BIG}],
        lr=LR, weight_decay=1e-4)

    n_total = feats[EXPERTS[0]].shape[0] * len(EXPERTS)
    per = n_total // len(EXPERTS)
    g = torch.Generator().manual_seed(seed * 977 + 13)
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        xb, tb = [], []
        for e in EXPERTS:
            loc = torch.randint(0, per, (max(1, BATCH // len(EXPERTS)),), generator=g).to(device)
            xb.append(projs[e](feats[e][loc]))
            tb.append(tr_t[loc])
        loss = F.mse_loss(layer(torch.cat(xb, 0)), torch.cat(tb, 0))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

    with torch.no_grad():
        # 逐底座的 held-out 投影与融合输出 (用于跨底座一致性 + 探针)
        proj_ho, out_ho = {}, {}
        for e in EXPERTS:
            p_ = projs[e](feats_ho[e])
            proj_ho[e] = p_
            out_ho[e] = layer(p_)          # [n, S, d]
        x_ho = torch.cat([proj_ho[e] for e in EXPERTS], 0)
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
        y_flat = y_ho.reshape(-1, d_shared).detach()
        src = torch.tensor([m for m in range(len(EXPERTS))
                            for _ in range(feats_ho[EXPERTS[0]].shape[0] * S_TOK)]).to(device)
        blk = ho_t.shape[-1] // N_CLASSES
        tgt_idx = ho_t_all.reshape(-1, d_shared).argmax(-1) // blk
        cons_in = consistency(proj_ho, None)
        cons_out = consistency(out_ho, None)
    # ⚠️ 探针必须在 no_grad 之外: MLP 探针自己要训参数
    #    (在 no_grad 里 backward 会报 "does not require grad")
    probes = {"source": probe_accuracy_all(y_flat, src, len(EXPERTS), seed=seed),
              "class": probe_accuracy_all(y_flat, tgt_idx, N_CLASSES, seed=seed)}
    return {"held_out": {"fuse": fuse, "avg": avg, "best_single": best_single},
            "probes": {k: {fam: v[0] for fam, v in d.items()} for k, d in probes.items()},
            "consistency": {"input": cons_in, "output": cons_out},
            "in_sample_fuse": F.mse_loss(layer(project(feats)), tr_t_all).item(),
            "perm": perm, "d_shared": d_shared}


def main():
    print("=" * 78)
    print("V53.0 — 第三审: 独立复算 (第三套实现)")
    print("=" * 78)
    print("臂:", list(ARMS))
    print(f"协议: 值域分割 (训练域与留出域零交集) | n_per_class={N_PER_CLASS} "
          f"| D_shared={D_SHAREDS} | 种子基址 {SEED_BASE}")
    print("预登记判决:")
    for k, v in PREDICTIONS.items():
        print(f"   {k}: {v}")

    tr_pairs, ho_pairs = make_split(N_PER_CLASS, seed=0)
    tr_texts = [t for _, t in tr_pairs]
    ho_texts = [t for _, t in ho_pairs]
    labels = [c for c, _ in tr_pairs]
    labels_ho = [c for c, _ in ho_pairs]
    print(f"\n  训练 {len(tr_texts)} 条 / 留出 {len(ho_texts)} 条 "
          f"(文本交集 {len(set(tr_texts) & set(ho_texts))})")
    t0 = time.time()
    enc = encode_all(DEVICE, tr_texts)
    enc_ho = encode_all(DEVICE, ho_texts)
    print(f"  编码完成 ({time.time()-t0:.0f}s)")

    results = _load_cells()          # 从桶里恢复已算的格 (抗 Space 重建/重复 combo)
    if results:
        print(f"  从桶里恢复 {len(results)} 格: {sorted(results)}")
    for d_shared in D_SHAREDS:
        for arm in ARMS:
            key = f"D{d_shared}:{arm}"
            if key in results:
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
            src_m = sum(np.mean([r["probes"]["source"][f] for f in r["probes"]["source"]]) for r in rows) / n
            cls_m = sum(np.mean([r["probes"]["class"][f] for f in r["probes"]["class"]]) for r in rows) / n
            cons = sum(r["consistency"]["output"] for r in rows) / n
            print(f"  [D{d_shared:>3}] {arm:18s} fuse={g('fuse'):8.4f} 单专家={g('best_single'):8.4f} "
                  f"margin={margin:+7.1f}% | 源={src_m:.3f} 类={cls_m:.3f} | 一致性(输出)={cons:+.3f} "
                  f"({time.time()-t1:.0f}s)", flush=True)

    print("\n" + "=" * 78)
    print("预登记判决")
    print("=" * 78)
    verdicts = {}
    for d_shared in D_SHAREDS:
        def st(a):
            rows = results[f"D{d_shared}:{a}"]["rows"]
            n = len(rows)
            g = lambda k: sum(r["held_out"][k] for r in rows) / n
            margin = (g("best_single") - g("fuse")) / max(g("best_single"), 1e-9) * 100
            src = {f: np.mean([r["probes"]["source"][f] for r in rows]) for f in r0["probes"]["source"]}
            cls = {f: np.mean([r["probes"]["class"][f] for r in rows]) for f in r0["probes"]["class"]}
            cons = sum(r["consistency"]["output"] for r in rows) / n
            return dict(margin=margin, src=src, cls=cls, cons=cons)
        r0 = results[f"D{d_shared}:frozen_random"]["rows"][0]
        A, B, C, D = (st(a) for a in ("frozen_random", "permuted_target", "learned_lift", "randomized_target"))
        d1a = all((A["src"][f] - A["cls"][f]) >= 0.30 for f in A["src"])
        d1b = abs(B["margin"] - A["margin"]) <= 10.0
        d1c = C["margin"] <= A["margin"] / 2
        d2 = all(D["cls"][f] <= 0.50 for f in D["cls"])
        d3 = (C["cons"] - A["cons"]) >= 0.15
        print(f"D_shared={d_shared}:")
        print(f"   D1a 源−类 ≥0.30 (三族同向): {'通过' if d1a else '未通过'}  "
              + " ".join(f"{f}:{A['src'][f]-A['cls'][f]:+.2f}" for f in A["src"]))
        print(f"   D1b 目标置换后 margin 变化 ≤10pp: {'通过' if d1b else '未通过'} "
              f"({A['margin']:+.1f}% → {B['margin']:+.1f}%)")
        print(f"   D1c 有意义空间 margin ≤ 冻结臂一半: {'通过' if d1c else '未通过'} "
              f"({A['margin']:+.1f}% → {C['margin']:+.1f}%)")
        print(f"   D2  随机目标臂 类可解码 ≤0.50 (三族): {'通过' if d2 else '未通过 (探针无效!)'}  "
              + " ".join(f"{f}:{D['cls'][f]:.2f}" for f in D["cls"]))
        print(f"   D3  有意义空间一致性 ≥ 随机空间 +0.15: {'通过' if d3 else '未通过'} "
              f"({A['cons']:+.3f} → {C['cons']:+.3f})")
        verdicts[f"D{d_shared}"] = dict(D1a=bool(d1a), D1b=bool(d1b), D1c=bool(d1c),
                                        D2=bool(d2), D3=bool(d3))
    all_pass = all(all(v.values()) for v in verdicts.values())
    print(f"\n总判定: {'第三审通过 (三审制闭合)' if all_pass else '第三审未通过 → 相应主张须回退'}")

    out_dir = os.path.join(ROOT, "research/aligner/v53_third_audit/results")
    os.makedirs(out_dir, exist_ok=True)
    mpath = os.path.join(out_dir, "metrics.json")
    payload = {"schema_version": "1.0", "version": "53.0", "package": "v53_third_audit",
               "line": "aligner",
               "config": {"d_shareds": D_SHAREDS, "steps": STEPS, "seeds": SEEDS,
                          "seed_base": SEED_BASE, "n_per_class": N_PER_CLASS,
                          "n_classes": N_CLASSES, "tokens": S_TOK},
               "predictions": PREDICTIONS, "verdicts": verdicts, "all_pass": all_pass,
               "results": results,
               "notes": {"independence": "第三套实现: 值域分割的数据 + 三族探针 + 残差抬升 + 新种子 + 不重叠的 D_shared; 只保留被审计对象",
                         "new_positive_metric": "跨底座一致性 (同一文本在三底座上的输出余弦相似度): 正面证据",
                         "chain": "v50/v51 (首审) → v52 (二审, 独立复算) → v53 (三审, 再独立复算) → 审计链闭合"}}
    if os.path.exists(mpath):
        try:
            old = json.load(open(mpath, encoding="utf-8"))
            old.setdefault("results", {}).update(payload["results"])
            old["verdicts"], old["all_pass"] = payload["verdicts"], payload["all_pass"]
            old["config"], old["notes"], old["predictions"] = payload["config"], payload["notes"], payload["predictions"]
            payload = old
        except Exception:
            pass
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n指标已落盘: {mpath}")


def _load_cells():
    try:
        import os as _os, tempfile
        from huggingface_hub import list_bucket_tree, download_bucket_files
        bucket = _os.environ.get("V53_BUCKET", "redmingwei/hetero-moe")
        prefix = _os.environ.get("V53_BUCKET_PREFIX", "v53_artifacts")
        tok = _os.environ.get("HF_TOKEN")
        paths = [getattr(it, "path", None) for it in
                 list_bucket_tree(bucket, prefix=f"{prefix}/cells/", recursive=True)]
        paths = [p for p in paths if p and p.endswith(".json")]
        if not paths:
            return {}
        tmp = tempfile.mkdtemp()
        download_bucket_files(bucket, files=[(p, os.path.join(tmp, os.path.basename(p)))
                                            for p in paths], token=tok, raise_on_missing_files=False)
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
    try:
        import os as _os
        from huggingface_hub import batch_bucket_files
        bucket = _os.environ.get("V53_BUCKET", "redmingwei/hetero-moe")
        prefix = _os.environ.get("V53_BUCKET_PREFIX", "v53_artifacts")
        tok = _os.environ.get("HF_TOKEN")
        payload = json.dumps({"key": key, "rows": rows}, ensure_ascii=False, indent=2)
        batch_bucket_files(bucket, add=[(payload.encode(), f"{prefix}/cells/{key.replace(':', '_')}.json")],
                           token=tok or None)
    except Exception as e:
        print(f"  [warn] 该格未推上桶: {repr(e)[:120]}")


if __name__ == "__main__":
    main()
