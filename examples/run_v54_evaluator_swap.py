"""V54.0 — 评价器换轨: MSE → {高斯 NLL, Cross-Entropy} (同网格, 不降级).

> 归属路线: aligner (审计链 v48 → … → v53 收口之后的第一代)
> 复现: bash run.sh v54
> 冒烟: V54_SMOKE=1 python examples/run_v54_evaluator_swap.py  (不需要模型权重)

三件事与 v53 **逐位相同** (import 同一份代码, 不是"尽量"):
  数据生成 / 值域分割  ← v53_third_audit.make_split
  三族探针             ← probe_accuracy_all
  残差/线性抬升        ← ResidualLift / LinearLift

变的是:
  · 评价器 1 档 → 3 档, 三档**配对** (同 seed → 同初始化 → 同批顺序)
  · 每个 cell 落 (config, code_sha, seed) —— v53 缺 code_sha, 事后查不出出处
  · 每档都报 `mse` 读数 (同一个输出张量的绝对误差) —— 跨档唯一可比的量
  · E4 自检: NLL 档 logvar 与平方误差的相关, 退化则该档读数作废
"""
import gc
import hashlib
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from research.aligner.v54_evaluator_swap.audit import (
    CLASSES, ARMS, N_CLASSES, EVALUATORS, PREDICTIONS_V54,
    EvaluatorHead, eval_loss, baselines, decode_accuracy,
    make_split, build_targets, probe_accuracy_all, consistency,
    ResidualLift, LinearLift,
)
from research.scale_heterogeneous.v33_scale_hetero import CrossModalFusionLayer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S_TOK = int(os.environ.get("V54_TOK", "8"))
N_PER_CLASS = int(os.environ.get("V54_NPC", "16"))
BATCH = int(os.environ.get("V54_BATCH", "288"))
STEPS = int(os.environ.get("V54_STEPS", "200"))
LR = 1e-2
LR_BIG = float(os.environ.get("V54_LR_BIG", "1e-3"))
SEEDS = int(os.environ.get("V54_SEEDS", "5"))
SEED_BASE = int(os.environ.get("V54_SEED_BASE", "2000"))
D_SHAREDS = [int(x) for x in os.environ.get("V54_DSHARED", "96,252,384").split(",")]
# 块状 one-hot 要求 D_shared 能被 n_classes 整除, 否则 block 不等宽 → reshape 会静默错。
# v52 的 256 在 8 类下可用, 在 6 类下不可用 (256/6 不整); 最近的同量级值是 252 = 42×6。
for _d in D_SHAREDS:
    if _d % N_CLASSES != 0:
        raise SystemExit(f"V54_DSHARED 含 {_d}: 不能被 n_classes={N_CLASSES} 整除 "
                         f"(块状 one-hot 要求等宽块; 最近的合法值 {(_d // N_CLASSES) * N_CLASSES} "
                         f"或 {(_d // N_CLASSES + 1) * N_CLASSES})")
DEVICE = torch.device(os.environ.get("V54_DEVICE", "cpu"))
EXPERTS = ["big", "small1", "small2"]
SMOKE = os.environ.get("V54_SMOKE", "0") == "1"
OUT_DIR = os.path.join(ROOT, "research/aligner/v54_evaluator_swap/results")
ART_DIR = os.environ.get("V54_ART_DIR", OUT_DIR)
BUCKET = os.environ.get("V54_BUCKET", "")
BUCKET_PREFIX = os.environ.get("V54_BUCKET_PREFIX", "v54_artifacts")


def code_sha() -> str:
    """产物血缘: 产生这份数字的 commit。v53 缺这个字段, 事后查不出出处。"""
    try:
        sha = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        sha = "unknown"
    try:
        dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain",
                                "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        if dirty:
            sha += "+dirty"
    except Exception:
        pass
    return sha or "unknown"


# ------------------------------------------------------------------ 编码

def _encode_on(texts, tok, model, device):
    enc = tok(texts, padding="max_length", truncation=True, max_length=S_TOK,
              return_tensors="pt")
    with torch.no_grad():
        h = model(enc["input_ids"].to(device)).last_hidden_state
    return h.detach().float()


def _fake_encoding(texts, name, dim):
    """冒烟用: 确定性伪编码, 只为把训练/判决逻辑跑通 (不用于任何结论)。"""
    g = torch.Generator().manual_seed(abs(hash(name)) % (2 ** 31))
    rows = []
    for t in texts:
        ids = [ord(c) % 89 + 1 for c in t[:S_TOK].ljust(S_TOK, "_")[:S_TOK]]
        rows.append(torch.tensor(ids, dtype=torch.float32))
    base = torch.stack(rows)                                  # [n, S]
    emb = torch.nn.functional.one_hot(base.long(), num_classes=90).float()  # [n,S,90]
    proj = torch.randn(90, dim, generator=g) / (dim ** 0.5)
    return emb @ proj                                          # [n,S,D]


def encode_all(device, texts):
    if SMOKE:
        return {n: _fake_encoding(texts, n, d)
                for n, d in (("big", 2560), ("small1", 1024), ("small2", 1024))}

    from research._primitives.scale_hetero_loaders import (
        MSA_DIR, EMB_DIR, RER_DIR, load_msa_big, load_small, load_tokenizer,
    )
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


# ------------------------------------------------------------------ 单格

def _build(arm, d_shared, Dm, seed):
    """按 seed 重建 —— 三档各自调用时权重逐位相同 (配对的关键)。"""
    lift_kind, variant = ARMS[arm]
    torch.manual_seed(seed)
    projs = {e: (ResidualLift(d_shared, Dm[e]) if lift_kind == "residual"
                 else LinearLift(d_shared, Dm[e])).to(DEVICE) for e in EXPERTS}
    if lift_kind == "linear":
        for e in EXPERTS:
            for p in projs[e].parameters():
                p.requires_grad_(False)
    layer = CrossModalFusionLayer(d_shared, 4 * d_shared, len(EXPERTS), 1,
                                  [Dm[e] for e in EXPERTS]).to(DEVICE)
    for p in layer.P_m:
        p.requires_grad_(False)
    return lift_kind, variant, projs, layer


def run_cell(arm, d_shared, evaluator, seed, feats, feats_ho, labels, labels_ho):
    Dm = {e: feats[e].shape[-1] for e in EXPERTS}
    lift_kind, variant, projs, layer = _build(arm, d_shared, Dm, seed)
    head = EvaluatorHead(evaluator, d_shared).to(DEVICE)

    perm = None
    if variant == "permuted_target":
        rng = np.random.default_rng(seed + 7)
        while True:
            p = rng.permutation(N_CLASSES).tolist()
            if all(p[i] != i for i in range(N_CLASSES)):
                perm = p
                break
    tgt_map = (lambda c: perm[c]) if perm else (lambda c: c)
    tr_t = build_targets([tgt_map(c) for c in labels], d_shared, N_CLASSES, DEVICE, n_tokens=S_TOK)
    ho_t = build_targets([tgt_map(c) for c in labels_ho], d_shared, N_CLASSES, DEVICE,
                         randomized=(variant == "randomized_target"), seed=seed, n_tokens=S_TOK)
    tr_lab = torch.tensor([tgt_map(c) for c in labels]).to(DEVICE)
    ho_lab = torch.tensor([tgt_map(c) for c in labels_ho]).to(DEVICE)
    tr_t_all = torch.cat([tr_t] * len(EXPERTS), 0)
    ho_t_all = torch.cat([ho_t] * len(EXPERTS), 0)

    params = [layer.W_router] + list(layer.gammas) + list(layer.betas) + list(layer.alphas)
    if lift_kind == "residual":
        for e in EXPERTS:
            params += list(projs[e].parameters())
    head_params = list(head.parameters())
    allp = params + head_params
    big = [p for p in allp if p.numel() > 4096]
    small = [p for p in allp if p.numel() <= 4096]
    opt = torch.optim.AdamW([{"params": small, "lr": LR}, {"params": big, "lr": LR_BIG}],
                            lr=LR, weight_decay=1e-4)

    n_total = feats[EXPERTS[0]].shape[0] * len(EXPERTS)
    per = n_total // len(EXPERTS)
    g = torch.Generator().manual_seed(seed * 977 + 13)   # 三档共用 → 配对
    last = {}
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        xb, tb, lb = [], [], []
        for e in EXPERTS:
            loc = torch.randint(0, per, (max(1, BATCH // len(EXPERTS)),), generator=g).to(DEVICE)
            xb.append(projs[e](feats[e][loc]))
            tb.append(tr_t[loc])
            lb.append(tr_lab[loc])
        out = layer(torch.cat(xb, 0))
        loss, last = eval_loss(head, out, torch.cat(tb, 0), torch.cat(lb, 0))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(allp, 1.0)
        opt.step()

    with torch.no_grad():
        proj_ho, out_ho = {}, {}
        for e in EXPERTS:
            p_ = projs[e](feats_ho[e])
            proj_ho[e] = p_
            out_ho[e] = layer(p_)
        x_ho = torch.cat([proj_ho[e] for e in EXPERTS], 0).detach()
        y_raw = layer.forward_with_routing(x_ho)[0].detach()

        # ⚠️ 读数必须过同一层评价器头: CE 档的输出被压到单纯形, 不过头就与 MSE 档不同量纲
        #    (冒烟实测未过头的 CE 臂 fuse=1.97, 过了头才落回基线附近)
        def to_eval(out_t):
            return head.forward(out_t)["pred"]

        def err(out_t):
            return float(F.mse_loss(to_eval(out_t.detach()).detach(), ho_t_all))

        fuse = err(y_raw)
        tot = torch.zeros_like(x_ho)
        single_errs, single_accs = [], []
        lab_all = ho_lab.repeat(len(EXPERTS))
        for m in range(len(EXPERTS)):
            h_a = x_ho * layer.gammas[m] + layer.betas[m]
            f = layer._swiglu(h_a, layer.w_gates[m], layer.w_ups[m], layer.w_downs[m])
            tot = tot + f
            cand = x_ho + layer.alphas[m] * f
            single_errs.append(err(cand))
            single_accs.append(decode_accuracy(to_eval(cand), lab_all))
        avg = err(x_ho + tot / len(EXPERTS))
        acc_fuse = decode_accuracy(to_eval(y_raw), lab_all)

        y_flat = to_eval(y_raw).reshape(-1, d_shared)   # 探针也在评价器空间里做
        src = torch.tensor([m for m in range(len(EXPERTS))
                            for _ in range(feats_ho[EXPERTS[0]].shape[0] * S_TOK)]).to(DEVICE)
        blk = d_shared // N_CLASSES
        tgt_idx = ho_t_all.reshape(-1, d_shared).argmax(-1) // blk
        cons_in, cons_out = consistency(proj_ho, None), consistency(out_ho, None)

    probes = {"source": probe_accuracy_all(y_flat, src, len(EXPERTS), seed=seed),
              "class": probe_accuracy_all(y_flat, tgt_idx, N_CLASSES, seed=seed)}
    return {
        "held_out": {"fuse": fuse, "avg": avg, "best_single": min(single_errs)},
        "acc": {"fuse": acc_fuse, "best_single": max(single_accs)},
        "probes": {k: {fam: v[0] for fam, v in d.items()} for k, d in probes.items()},
        "consistency": {"input": cons_in, "output": cons_out},
        "in_sample_fuse": err(layer(torch.cat([projs[e](feats[e]) for e in EXPERTS], 0))),
        "train_readout": last,
        "n_params": {"head_extra": head.extra_params(),
                     "trained": sum(p.numel() for p in allp if p.requires_grad)},
        "perm": perm, "d_shared": d_shared, "seed": seed, "evaluator": evaluator,
    }


# ------------------------------------------------------------------ 判决

def judge(results):
    """E1–E4, 全部从 rows 重算。E5 (有效秩) 在 check 阶段单独算。"""
    out = {}
    for d in D_SHAREDS:
        acc = {"E1": [], "E2": [], "E3": [], "E4": [], "E6": []}
        for ev in EVALUATORS:
            for rl, rf in zip(results.get(f"D{d}:{ev}:learned_lift", {}).get("rows", []),
                              results.get(f"D{d}:{ev}:frozen_random", {}).get("rows", [])):
                acc["E1"].append(rl["held_out"]["fuse"] < rf["held_out"]["fuse"])
                acc["E3"].append((rl["acc"]["fuse"] - rl["acc"]["best_single"]) * 100 <= 5.0)
                if ev == "ce":
                    acc["E2"].append(abs(rf["acc"]["fuse"] - 1.0 / N_CLASSES) <= 0.05)
                if ev == "gauss_nll":
                    rho = rl.get("train_readout", {}).get("logvar_err_spearman")
                    acc["E4"].append(rho is None or rho >= 0.2)
                # E6: 低于该档自己的零参数基线 (冒烟后新增, 见 audit.PREDICTIONS_V54 注释)
                base = baselines(ev)
                ref = base.get("predict_mean", base.get("uniform_pred_mse"))
                acc["E6"].append(rl["held_out"]["fuse"] < ref)
        out[f"D{d}"] = {k: (bool(np.mean(v) >= 0.8) if v else None) for k, v in acc.items()}
    return out


def _push(payload, key):
    if not BUCKET:
        return
    try:
        from huggingface_hub import batch_bucket_files
        batch_bucket_files(BUCKET,
                           add=[(json.dumps(payload, ensure_ascii=False, indent=1).encode(),
                                 f"{BUCKET_PREFIX}/cells/{key}.json")],
                           token=os.environ.get("HF_TOKEN"))
    except Exception as e:
        print(f"  (warn) 推桶失败 {key}: {repr(e)[:100]}")


def main():
    t0 = time.time()
    sha = code_sha()
    print("=" * 78)
    print("V54.0 — 评价器换轨: MSE → {高斯 NLL, CE} | 同网格, 配对, 不降级")
    print("=" * 78)
    print(f"code_sha={sha}  SMOKE={SMOKE}  device={DEVICE}")
    print(f"D={D_SHAREDS} arms={list(ARMS)} evaluators={EVALUATORS} seeds={SEEDS} steps={STEPS}")
    print("\n解析基线 (块状 one-hot, 精确值):")
    for ev in EVALUATORS:
        print(f"  {ev:10s} " + "  ".join(f"{k}={v:+.4f}" for k, v in baselines(ev).items()))

    train, held = make_split(n_per_class=N_PER_CLASS, seed=SEED_BASE)
    tr_texts = [t for _, t in train]      # make_split 返回 (label, 已拼好的文本)
    ho_texts = [t for _, t in held]
    tr_labels = [c for c, _ in train]
    ho_labels = [c for c, _ in held]
    print(f"\n训练 {len(tr_texts)} 条 / 留出 {len(ho_texts)} 条 "
          f"(文本交集 {len(set(tr_texts) & set(ho_texts))})")

    feats, feats_ho = encode_all(DEVICE, tr_texts), encode_all(DEVICE, ho_texts)
    print(f"编码完成 ({time.time() - t0:.0f}s)  "
          + " ".join(f"{k}={list(v.shape)}" for k, v in feats.items()))

    results = {}
    total = len(D_SHAREDS) * len(ARMS) * len(EVALUATORS) * SEEDS
    i = 0
    for d in D_SHAREDS:
        for arm in ARMS:
            for ev in EVALUATORS:
                key = f"D{d}:{ev}:{arm}"
                rows = []
                for s in range(SEEDS):
                    seed = SEED_BASE + s
                    i += 1
                    row = run_cell(arm, d, ev, seed, feats, feats_ho, tr_labels, ho_labels)
                    rows.append(row)
                    h = row["held_out"]
                    print(f"[{i:3d}/{total}] D{d:<4d} {ev:<10s} {arm:<18s} seed={seed} "
                          f"fuse={h['fuse']:.4f} 单专家={h['best_single']:.4f} "
                          f"acc={row['acc']['fuse']:.3f} 源={row['probes']['source']['ridge']:.3f} "
                          f"类={row['probes']['class']['ridge']:.3f} ({time.time() - t0:.0f}s)")
                cell = {"key": key, "code_sha": sha,
                        "config": {"n_classes": N_CLASSES, "tokens": S_TOK, "steps": STEPS,
                                   "evaluator": ev, "arm": arm, "d_shared": d,
                                   "seeds": SEEDS, "seed_base": SEED_BASE},
                        "rows": rows}
                results[key] = {"rows": rows, "code_sha": sha, "config": cell["config"]}
                _push(cell, key.replace(":", "_"))

    verdicts = judge(results)
    print("\n" + "=" * 78)
    print("预登记判决 E1–E4 (E5 见 tools/check_artifacts.py)")
    print("=" * 78)
    for d, v in verdicts.items():
        print(f"  {d}: " + "  ".join(f"{k}={'过' if x else '未过'}"
                                     for k, x in v.items() if x is not None))
    all_pass = all(all(x for x in v.values() if x is not None) for v in verdicts.values())

    os.makedirs(ART_DIR, exist_ok=True)
    payload = {"schema_version": "1.0", "version": "54.0", "package": "v54_evaluator_swap",
               "line": "aligner", "code_sha": sha,
               "config": {"d_shareds": D_SHAREDS, "steps": STEPS, "seeds": SEEDS,
                          "seed_base": SEED_BASE, "n_per_class": N_PER_CLASS,
                          "n_classes": N_CLASSES, "tokens": S_TOK,
                          "evaluators": list(EVALUATORS)},
               "baselines": {ev: baselines(ev) for ev in EVALUATORS},
               "predictions": PREDICTIONS_V54, "verdicts": verdicts, "all_pass": all_pass,
               "results": results,
               "notes": {"paired": "三档共用同一 seed/初始化/批顺序 → 差异只来自损失",
                         "comparable_metric": "每档都报 mse (同一输出张量的绝对误差)",
                         "lineage": "v54 起每个 cell 带 code_sha",
                         "smoke": SMOKE}}
    mpath = os.path.join(ART_DIR, "metrics.json")
    json.dump(payload, open(mpath, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n指标已落盘: {mpath}  (code_sha={sha}, {total} 格, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
