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

from research.aligner.v54_evaluator_swap.integrity import (
    expected_grid, completeness, judge_verdicts, summarize,
)
from research.aligner.v54_evaluator_swap.audit import (
    CLASSES, ARMS, N_CLASSES, EVALUATORS, PREDICTIONS_V54,
    EvaluatorHead, eval_loss, baselines, decode_accuracy, calibration,
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


def _params_sha(objs) -> str:
    """共享初始化参数的指纹 —— 配对是否成立的**产物级证据** (不是靠纪律)。

    只哈希 projs + layer (三档共用), 不含 EvaluatorHead (三档本就不同)。
    三档在 (arm, D, seed) 下必须给出同一个指纹; 不成立就说明配对没做到。
    """
    h = hashlib.sha256()
    for o in objs:
        for name in sorted(o.state_dict()):
            t = o.state_dict()[name]
            h.update(name.encode())
            h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def load_protocol_lock() -> dict | None:
    """预登记锁: 全网格开跑**之前**冻结的网格与判据。存在即必须服从。"""
    p = os.path.join(ROOT, "research/aligner/v54_evaluator_swap/protocol.lock.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def check_protocol(lock: dict | None) -> dict:
    """把当前运行的网格与预登记锁对拍。

    完整性要求: 全网格 (非冒烟) 必须与锁逐项相同。允许偏离的唯一路径是
    显式设 V54_ALLOW_DEVIATION=1, 且偏离会被**写进产物**, 不会静默。
    """
    run = {"d_shareds": D_SHAREDS, "arms": list(ARMS), "evaluators": list(EVALUATORS),
           "seeds": SEEDS, "seed_base": SEED_BASE, "steps": STEPS,
           "n_per_class": N_PER_CLASS, "tokens": S_TOK, "n_classes": N_CLASSES,
           "batch": BATCH}
    if SMOKE:
        return {"status": "smoke", "locked": lock is not None, "run": run,
                "note": "冒烟不是网格, 不与锁对拍"}
    if lock is None:
        return {"status": "no_lock", "run": run,
                "note": "无 protocol.lock.json → 这次运行无法被证明是预登记的那次"}
    grid = lock.get("grid", {})
    diff = {k: {"lock": grid.get(k), "run": v} for k, v in run.items() if grid.get(k) != v}
    if diff and os.environ.get("V54_ALLOW_DEVIATION") != "1":
        raise SystemExit(
            "✗ 本次网格与预登记锁不一致, 拒绝开跑 (完整性保护):\n"
            + "\n".join(f"    {k}: 锁={v['lock']} 运行={v['run']}" for k, v in diff.items())
            + "\n  若确属有意变更: 重生成 protocol.lock.json, 或显式设 V54_ALLOW_DEVIATION=1 "
              "(偏离会写进产物)")
    return {"status": "deviated" if diff else "match", "locked": True,
            "lock_sha": lock.get("sha256"), "diff": diff or None, "run": run}


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
    init_sha = _params_sha([*projs.values(), layer])
    return lift_kind, variant, projs, layer, init_sha


def _predictions_sha() -> str:
    """预登记判据的指纹。事后改 PREDICTIONS_V54 → 这个哈希就对不上。"""
    return hashlib.sha256(
        json.dumps(PREDICTIONS_V54, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()[:16]


def _effective_rank(x: torch.Tensor) -> float:
    """exp(H(p)), p_i = s_i² / Σs_j² (奇异值的能量分布)。

    对 d_shared 维的投影空间给一个"实际用了多少维"的连续读数。
    退化解 (所有样本挤在一个方向) → 1.0; 各向同性 → d_shared。
    """
    z = x.double() - x.double().mean(0, keepdim=True)
    try:
        sv = torch.linalg.svdvals(z)
    except Exception:
        return float("nan")
    e = sv ** 2
    tot = float(e.sum())
    if tot <= 0:
        return 0.0
    p = (e / tot).clamp_min(1e-12)
    return float(torch.exp(-(p * p.log()).sum()))


def run_cell(arm, d_shared, evaluator, seed, feats, feats_ho, labels, labels_ho):
    Dm = {e: feats[e].shape[-1] for e in EXPERTS}
    lift_kind, variant, projs, layer, init_sha = _build(arm, d_shared, Dm, seed)
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

        # E4 必须在 **held-out** 上量, 不能只用训练末读数 (训练末的 logvar 是拟合过的)
        cal_ho = None
        if evaluator == "gauss_nll":
            q_ho = head.forward(y_raw)
            err_ho = ((q_ho["pred"] - ho_t_all) ** 2).mean(dim=(1, 2))
            cal_ho = calibration(q_ho["logvar"], err_ho)

        # E5 有效秩 (Roy–Vetterli): exp(熵(归一化奇异值²))。越高 = 空间越不退化。
        eff_rank = _effective_rank(x_ho.reshape(-1, d_shared))

    probes = {"source": probe_accuracy_all(y_flat, src, len(EXPERTS), seed=seed),
              "class": probe_accuracy_all(y_flat, tgt_idx, N_CLASSES, seed=seed)}
    return {
        "held_out": {"fuse": fuse, "avg": avg, "best_single": min(single_errs)},
        "acc": {"fuse": acc_fuse, "best_single": max(single_accs)},
        "probes": {k: {fam: v[0] for fam, v in d.items()} for k, d in probes.items()},
        "consistency": {"input": cons_in, "output": cons_out},
        "in_sample_fuse": err(layer(torch.cat([projs[e](feats[e]) for e in EXPERTS], 0))),
        "train_readout": last,
        "held_out_calibration": cal_ho,
        "eff_rank": eff_rank,
        "init_sha": init_sha,
        "n_params": {"head_extra": head.extra_params(),
                     "trained": sum(p.numel() for p in allp if p.requires_grad)},
        "perm": perm, "d_shared": d_shared, "seed": seed, "evaluator": evaluator,
    }


# ------------------------------------------------------------------ 判决

def _criteria():
    """预登记判据 (可执行形式)。only_ev 也属于预登记, 开跑前固定。

    每个 fn(learned_row, frozen_row, ctx) -> bool | None
      True/False = 该观测有明确结论; None = 该观测无法判定 (会让整条判据变成"无结论")
    """
    def e1(rl, rf, ctx):
        """有意义空间的绝对误差 < 冻结随机空间 (同 seed 配对)。"""
        return rl["held_out"]["fuse"] < rf["held_out"]["fuse"]

    def e2(rl, rf, ctx):
        """冻结随机臂在 CE 下应处于 chance 水平。"""
        return abs(rf["acc"]["fuse"] - 1.0 / N_CLASSES) <= 0.05

    def e3(rl, rf, ctx):
        """融合 vs 最优单专家: CE 准确率差不超 5pp。"""
        return (rl["acc"]["fuse"] - rl["acc"]["best_single"]) * 100 <= 5.0

    def e4(rl, rf, ctx):
        """NLL 档的方差头没退化 (held-out 校准, 不是训练末读数)。"""
        rho = rl.get("held_out_calibration")
        return None if rho is None else rho >= 0.2

    def e5(rl, rf, ctx):
        """有意义空间的有效秩 > 冻结随机空间。"""
        a, b = rl.get("eff_rank"), rf.get("eff_rank")
        return None if (a is None or b is None) else a > b

    def e6(rl, rf, ctx):
        """低于该档**自己的**零参数基线 (冒烟后新增, 见 audit.PREDICTIONS_V54 注释)。"""
        base = baselines(ctx["ev"])
        ref = base.get("predict_mean", base.get("uniform_pred_mse"))
        return rl["held_out"]["fuse"] < ref

    return {"E1": {"fn": e1}, "E2": {"fn": e2, "only_ev": ("ce",)},
            "E3": {"fn": e3}, "E4": {"fn": e4, "only_ev": ("gauss_nll",)},
            "E5": {"fn": e5}, "E6": {"fn": e6}}


def judge(results):
    """判决 = 闸门 (完备 + 无未决 + 全真)。缺一格就不给结论, 见 integrity.py 的说明。"""
    grid = expected_grid(D_SHAREDS, EVALUATORS, list(ARMS), SEEDS, SEED_BASE)
    return judge_verdicts(results, grid, _criteria())


def _restore_cells(sha):
    """从桶里恢复已算的格。**不混血**: 代码/判据/步数任一不符就丢弃重算。"""
    if not BUCKET:
        return {}, []
    try:
        import tempfile
        from huggingface_hub import list_bucket_tree, download_bucket_files
        paths = [getattr(it, "path", None) for it in
                 list_bucket_tree(BUCKET, prefix=f"{BUCKET_PREFIX}/cells/", recursive=True)]
        paths = [p for p in paths if p and p.endswith(".json")]
        if not paths:
            return {}, []
        tmp = tempfile.mkdtemp()
        download_bucket_files(BUCKET,
                              files=[(p, os.path.join(tmp, os.path.basename(p))) for p in paths],
                              token=os.environ.get("HF_TOKEN"), raise_on_missing_files=False)
        out, rejected = {}, []
        cur_pred = _predictions_sha()
        for p in paths:
            f = os.path.join(tmp, os.path.basename(p))
            if not os.path.exists(f):
                continue
            try:
                d = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            key, cfg = d.get("key"), d.get("config", {}) or {}
            if not key:
                continue
            why = None
            if d.get("code_sha") != sha:
                why = f"code_sha {d.get('code_sha')} != {sha}"
            elif cfg.get("predictions_sha") != cur_pred:
                why = "判据指纹不符"
            elif cfg.get("steps") != STEPS:
                why = f"steps {cfg.get('steps')} != {STEPS}"
            if why:
                rejected.append({"key": key, "why": why})
                continue
            out[key] = {"rows": d.get("rows", []), "code_sha": d["code_sha"], "config": cfg}
        return out, rejected
    except Exception as e:  # noqa: BLE001
        print(f"  (warn) 恢复失败: {repr(e)[:110]}")
        return {}, []


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

    proto = check_protocol(load_protocol_lock())
    print(f"\n协议锁: {proto['status']}" + (f"  sha={proto.get('lock_sha')}" if proto.get('lock_sha') else ""))
    if proto.get("diff"):
        for k, v in proto["diff"].items():
            print(f"  ⚠ 偏离 {k}: 锁={v['lock']} 运行={v['run']}")

    restored, rejected = _restore_cells(sha)
    if restored or rejected:
        print(f"\n断点续跑: 恢复 {len(restored)} 格, 丢弃 {len(rejected)} 格"
              + (f" (例: {rejected[0]})" if rejected else ""))

    results = {}
    total = len(D_SHAREDS) * len(ARMS) * len(EVALUATORS) * SEEDS
    i = 0
    for d in D_SHAREDS:
        for arm in ARMS:
            for ev in EVALUATORS:
                key = f"D{d}:{ev}:{arm}"
                if key in restored and len(restored[key]["rows"]) == SEEDS:
                    results[key] = restored[key]
                    i += SEEDS
                    print(f"[{i:3d}/{total}] D{d:<4d} {ev:<10s} {arm:<18s} —— 已算, 跳过")
                    continue
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
                                   "seeds": SEEDS, "seed_base": SEED_BASE,
                                   "predictions_sha": _predictions_sha()},
                        "rows": rows}
                results[key] = {"rows": rows, "code_sha": sha, "config": cell["config"]}
                _push(cell, key.replace(":", "_"))

    verdicts, audit, all_pass = judge(results)
    comp = audit["_completeness"]
    print("\n" + "=" * 78)
    print("预登记判决 E1–E6 (闸门: only_ev 也由预登记固定)")
    print("=" * 78)
    for d, v in verdicts.items():
        cells = []
        for k, x in v.items():
            a = audit[d][k]
            tag = "过" if x is True else ("未过" if x is False else "无结论")
            cells.append(f"{k}={tag}({a['n_true']}/{a['n_expected']})")
        print(f"  {d}: " + "  ".join(cells))
    print(f"\n完整性: {summarize(audit)}")
    if not comp["ok"]:
        print(f"  缺格 {len(comp['missing_cells'])} 个: {comp['missing_cells'][:6]}")
        if comp["bad_seed_sets"]:
            print(f"  seed 集合异常 {len(comp['bad_seed_sets'])} 格")

    os.makedirs(ART_DIR, exist_ok=True)
    payload = {"schema_version": "1.0", "version": "54.0", "package": "v54_evaluator_swap",
               "line": "aligner", "code_sha": sha,
               "config": {"d_shareds": D_SHAREDS, "steps": STEPS, "seeds": SEEDS,
                          "seed_base": SEED_BASE, "n_per_class": N_PER_CLASS,
                          "n_classes": N_CLASSES, "tokens": S_TOK,
                          "evaluators": list(EVALUATORS)},
               "baselines": {ev: baselines(ev) for ev in EVALUATORS},
               "predictions": PREDICTIONS_V54,
               "predictions_sha": _predictions_sha(),
               "protocol_check": proto,
               "verdicts": verdicts, "all_pass": all_pass,
               "results": results,
               "IS_SMOKE_NOT_RESULTS": SMOKE,   # 冒烟产物必须自带这个标记, 否则会冒充结果
               "notes": {"paired": "三档共用同一 seed/初始化/批顺序 → 差异只来自损失",
                         "comparable_metric": "每档都报 mse (同一输出张量的绝对误差)",
                         "lineage": "v54 起每个 cell 带 code_sha",
                         "smoke": SMOKE}}
    payload["integrity"] = {
        "gate": audit["_gate"], "completeness": comp,
        "per_criterion": {d: audit[d] for d in audit if not d.startswith("_")},
        "restored_cells": len(restored), "rejected_cells": rejected,
        "criterion_errors": audit.get("criterion_errors", []),
    }
    os.makedirs(ART_DIR, exist_ok=True)
    if not comp["ok"]:
        # ⚠️ 网格不完备 → 只写 partial, 绝不写成 metrics.json 冒充结果
        ppath = os.path.join(ART_DIR, "metrics.partial.json")
        payload["IS_PARTIAL_NOT_RESULTS"] = True
        payload["partial_reason"] = summarize(audit)
        json.dump(payload, open(ppath, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n⛔ 网格不完备 ({comp['n_cells_have']}/{comp['n_cells_want']} 格, "
              f"{comp['n_obs_have']}/{comp['n_obs_want']} 观测)")
        print(f"   已写 {ppath} (IS_PARTIAL_NOT_RESULTS=true) —— 这不是结果, 续跑后会被取代")
        sys.exit(2)

    mpath = os.path.join(ART_DIR, "metrics.json")
    json.dump(payload, open(mpath, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n指标已落盘: {mpath}  (code_sha={sha}, {total} 格, {time.time() - t0:.0f}s)")
    print(f"闸门: {summarize(audit)} → all_pass={all_pass}")


if __name__ == "__main__":
    main()
