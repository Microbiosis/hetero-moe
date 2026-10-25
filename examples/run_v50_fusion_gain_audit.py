"""V50.0 — 「融合收益从哪来」审计 (Fusion-gain audit).

> 归属路线: aligner (接 v48/v49)
> 上游: v49_projection_space (三轴析因) — v49 发现"投影一可训练, 融合相对增益就塌到 ~0"
> 复现: bash run.sh v50

被审计的论文主张 (reconstruct.tex §6.3 + 摘要):
    "Fused beats best single expert (v33) +26.7% (5/5) ... survives" (held-out)

两种竞争解释:
  E1 假象说: margin 来自**投影空间从未被训练** (随机正交把底座搅乱 → 单专家基线弱
            → "融合"显得赢)。检验: 用 proxy 对齐过、且对目标任务不自适应的投影。
  E2 退化说: margin 来自**退化任务** (冻结随机空间让"认出样本来自哪个底座"变容易,
            而路由本来就在做这件事)。检验: 从融合输出做双探针
            (源专家身份 / 目标类别) 的 held-out 可解码性。

四臂 (在 v49 协议上: 8 类槽位文本, 训练/留出文本零交集, 三个真正不同的底座):
  A  L_static_frozen  随机线性 + 冻结            (= v49 的 A 臂, 论文的对照)
  B  NL_static        非线性 + 可训练            (= v49 的 B 臂)
  C  NL_proxy_frozen  非线性 + proxy 对齐后冻结   ← 新 (E1 的关键对照)
  D  NL_proxy_train   非线性 + proxy 对齐后可训练  ← 新

proxy 对齐是**无标签**的跨底座对齐 (同一文本在各底座的投影互相靠拢),
不引入目标任务的任何信息 → 不构成泄漏。
"""
import sys, os, json, gc, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from research._primitives.scale_hetero_loaders import (
    MSA_DIR, EMB_DIR, RER_DIR, load_msa_big, load_small, load_tokenizer,
)
from research.scale_heterogeneous.v33_scale_hetero import CrossModalFusionLayer
from research.aligner.v49_projection_space.data import Split, SAMPLE_SIZES
from research.aligner.v49_projection_space.mechanism import (
    LinearProj, NonlinearProj, param_report)
from research.aligner.v50_fusion_gain_audit.mechanism import proxy_pretrain, linear_probe

D_SHARED = 256
M, K = 3, 1
S_TOK = int(os.environ.get("V50_TOK", "8"))
BATCH = int(os.environ.get("V50_BATCH", "288"))
STEPS = int(os.environ.get("V50_STEPS", "300"))
PROXY_STEPS = int(os.environ.get("V50_PROXY_STEPS", "150"))
LR = 1e-2
LR_BIG = float(os.environ.get("V50_LR_BIG", "1e-3"))
N_SEEDS = int(os.environ.get("V50_SEEDS", "5"))
N_CLASSES = 8
DEVICE = torch.device(os.environ.get("V50_DEVICE", "cpu"))
ONLY_SIZES = [x for x in os.environ.get("V50_SIZES", "").split(",") if x]
ONLY_ARMS = [x for x in os.environ.get("V50_ARMS", "").split(",") if x]

ARMS = {
    # 名字: (抬升类型, 可训练, 先做 proxy 对齐)
    "A_linear_random_frozen": ("linear", False, False),     # ★ 论文 v33 的原配置 (随机线性 + 冻结)
    "B_nl_random_frozen":    ("nonlinear", False, False),   # 非线性但随机 + 冻结 (隔离"非线性"与"冻结")
    "C_nl_train":            ("nonlinear", True,  False),   # 非线性 + 可训练 (v49 的 NL_static)
    "D_nl_proxy_frozen":     ("nonlinear", False, True),    # 非线性 + proxy 对齐后冻结 ← E1 关键对照
    "E_nl_proxy_train":      ("nonlinear", True,  True),    # 非线性 + proxy 对齐后仍可训练
}
EXPERTS = ["big", "small1", "small2"]


def build_targets(labels, device):
    blk = D_SHARED // N_CLASSES
    tgt = torch.zeros(len(labels), S_TOK, D_SHARED)
    for i, c in enumerate(labels):
        tgt[i, :, c * blk:(c + 1) * blk] = 1.0
    return tgt.to(device)


def _encode_on(texts, tok, model, device):
    enc = tok(texts, padding="max_length", truncation=True, max_length=S_TOK, return_tensors="pt")
    with torch.no_grad():
        h = model(enc["input_ids"].to(device)).last_hidden_state
    return h.detach().float()


def encode_all(device, n_per_class):
    sp = Split(n_per_class, seed=0)
    tr, ho = sp.texts("train"), sp.texts("heldout")
    big_tok, small_tok = load_tokenizer(MSA_DIR), load_tokenizer(EMB_DIR)
    try:
        rer_tok = load_tokenizer(RER_DIR)
    except Exception:
        rer_tok = small_tok

    def one(tok, loader, *a):
        model, dim, _ = loader(*a)
        model = model.to(device)
        x = _encode_on(tr, tok, model, device)
        y = _encode_on(ho, tok, model, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return x, y, dim

    out = {"big": one(big_tok, load_msa_big),
           "small1": one(small_tok, load_small, "embedding-0.6b", EMB_DIR)}
    try:
        out["small2"] = one(rer_tok, load_small, "reranker-0.6b", RER_DIR)
    except Exception as e:
        print(f"  [警告] reranker 不可用 ({e}); 小2 退化为小1")
        out["small2"] = (out["small1"][0], out["small1"][1], out["small1"][2])
    return out, sp.labels("train"), sp.labels("heldout")


def run_one(arm, n_per_class, enc, tr_labels, ho_labels, seed, device):
    kind, trainable, use_proxy = ARMS[arm]
    torch.manual_seed(seed)
    raw = {e: enc[e][0].to(device) for e in EXPERTS}
    raw_ho = {e: enc[e][1].to(device) for e in EXPERTS}
    Dm = {e: raw[e].shape[-1] for e in EXPERTS}

    # ⚠️ build_projector 收的是 v49 的**臂名**, 不是类型串 → 这里直接用类
    proj_cls = LinearProj if kind == "linear" else NonlinearProj
    projs = {e: proj_cls(D_SHARED, Dm[e]).to(device) for e in EXPERTS}
    proxy_loss = 0.0
    if use_proxy:
        print("    [proxy] 跨底座对齐预训练 …", flush=True)
        proxy_loss = proxy_pretrain(projs, raw, EXPERTS, steps=PROXY_STEPS, log=True)
        print(f"    [proxy] L_proxy 末值 = {proxy_loss:.5f}", flush=True)
    if not trainable:
        for e in EXPERTS:
            for p in projs[e].parameters():
                p.requires_grad_(False)

    layer = CrossModalFusionLayer(D_SHARED, 4 * D_SHARED, M, K, [Dm[e] for e in EXPERTS]).to(device)
    for p in layer.P_m:
        p.requires_grad_(False)

    tr_t = build_targets(tr_labels, device)
    ho_t = build_targets(ho_labels, device)
    n = tr_t.shape[0]

    def project(feats):
        return torch.cat([projs[e](feats[e]) for e in EXPERTS], dim=0)

    params = [layer.W_router] + list(layer.gammas) + list(layer.betas) + list(layer.alphas)
    for e in EXPERTS:
        params += [p for p in projs[e].parameters() if p.requires_grad]
    big_ids = {id(p) for p in params if p.numel() > 4096}
    small_ids = {id(p) for p in params if p.numel() <= 4096}
    opt = torch.optim.AdamW(
        [{"params": [p for p in params if id(p) in small_ids], "lr": LR},
         {"params": [p for p in params if id(p) in big_ids], "lr": LR_BIG}],
        lr=LR, weight_decay=1e-4)

    bs_per = max(1, BATCH // M)
    g = torch.Generator().manual_seed(seed * 977 + 13)
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        xb, tb = [], []
        for e in EXPERTS:
            loc = torch.randint(0, n, (bs_per,), generator=g).to(device)
            xb.append(projs[e](raw[e][loc]))
            tb.append(tr_t[loc])
        loss = F.mse_loss(layer(torch.cat(xb, 0)), torch.cat(tb, 0))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

    with torch.no_grad():
        xs_tr = project(raw)
        xs_ho = project(raw_ho)
        y_tr, _ = layer.forward_with_routing(xs_tr)
        y_ho, alpha = layer.forward_with_routing(xs_ho)
        t_all_tr = torch.cat([tr_t] * M, 0)
        t_all_ho = torch.cat([ho_t] * M, 0)
        fuse = F.mse_loss(y_ho, t_all_ho).item()
        x_det = xs_ho.detach()
        sbig = avg = 0.0
        for m in range(M):
            h_a = x_det * layer.gammas[m] + layer.betas[m]
            f = layer._swiglu(h_a, layer.w_gates[m], layer.w_ups[m], layer.w_downs[m])
            if m == 0:
                sbig = F.mse_loss(xs_ho + layer.alphas[0] * f, t_all_ho).item()
            avg += f
        avg = F.mse_loss(xs_ho + avg / M, t_all_ho).item()
        trivial = F.mse_loss(torch.zeros_like(t_all_ho), t_all_ho).item()

    # ---- 双探针 (held-out) ----
    # 源专家标签: 第 m 个底座的样本块标签 = m
    src_tr = torch.cat([torch.full((n,), m, dtype=torch.long) for m in range(M)]).to(device)
    src_ho = src_tr.clone()
    cls_tr = torch.cat([torch.tensor(tr_labels) for _ in range(M)]).to(device)
    cls_ho = torch.cat([torch.tensor(ho_labels) for _ in range(M)]).to(device)
    y_tr_f = y_tr.reshape(-1, D_SHARED)
    y_ho_f = y_ho.reshape(-1, D_SHARED)
    rep_tr = src_tr.repeat_interleave(S_TOK)
    rep_ho = src_ho.repeat_interleave(S_TOK)
    cls_tr_f = cls_tr.repeat_interleave(S_TOK)
    cls_ho_f = cls_ho.repeat_interleave(S_TOK)
    probe_src_te, probe_src_tr = linear_probe(y_tr_f, rep_tr, y_ho_f, rep_ho,
                                              n_class=M, seed=seed)
    probe_cls_te, probe_cls_tr = linear_probe(y_tr_f, cls_tr_f, y_ho_f, cls_ho_f,
                                              n_class=N_CLASSES, seed=seed)
    # 输入侧的可识别性 (路由任务本身有多容易)
    in_src_te, _ = linear_probe(xs_tr.reshape(-1, D_SHARED), rep_tr,
                                xs_ho.reshape(-1, D_SHARED), rep_ho, n_class=M, seed=seed)
    return {
        "held_out": {"fuse": fuse, "single_big": sbig, "avg": avg, "trivial": trivial},
        "in_sample_fuse": F.mse_loss(y_tr, t_all_tr).item(),
        "probe": {"source_te": probe_src_te, "source_tr": probe_src_tr,
                  "class_te": probe_cls_te, "class_tr": probe_cls_tr,
                  "input_source_te": in_src_te},
        "proxy_loss": proxy_loss,
        "params": {e: param_report(projs[e]) for e in EXPERTS},
        "n": n * M,
    }


def _load_previous():
    """从桶里取回之前算好的格 (Space 容器是临时的, 本地 metrics 会丢)。"""
    try:
        import os as _os
        from huggingface_hub import batch_bucket_files, list_bucket_tree
        bucket = _os.environ.get("V50_BUCKET", "redmingwei/hetero-moe")
        prefix = _os.environ.get("V50_BUCKET_PREFIX", "v50_artifacts")
        tok = _os.environ.get("HF_TOKEN")
        paths = [getattr(it, "path", None) for it in
                 list_bucket_tree(bucket, prefix=f"{prefix}/cells/", recursive=True)]
        paths = [p for p in paths if p and p.endswith(".json")]
        if not paths:
            return {}
        import tempfile, os
        from huggingface_hub import download_bucket_files
        tmpdir = tempfile.mkdtemp()
        download_bucket_files(bucket, files=[(p, os.path.join(tmpdir, os.path.basename(p)))
                                            for p in paths], token=tok, raise_on_missing_files=False)
        out = {}
        for p in paths:
            f = os.path.join(tmpdir, os.path.basename(p))
            if os.path.exists(f):
                d = json.load(open(f, encoding="utf-8"))
                out[d["key"]] = d
        return out
    except Exception as e:
        print(f"  (warn) 未能取回桶里已有结果: {repr(e)[:110]}")
        return {}


def _push_cell(key, rows):
    """把一格的结果立刻写进桶 (Space 容器是临时的, 本地 metrics 会丢)。"""
    try:
        import os as _os
        from huggingface_hub import HfApi
        bucket = _os.environ.get("V50_BUCKET", "redmingwei/hetero-moe")
        prefix = _os.environ.get("V50_BUCKET_PREFIX", "v50_artifacts")
        tok = _os.environ.get("HF_TOKEN")
        # ⚠️ 桶不是 dataset 仓库: upload_file 对它无效 (404), 必须用桶原生 API
        from huggingface_hub import batch_bucket_files
        payload = json.dumps({"key": key, "rows": rows}, ensure_ascii=False, indent=2)
        # ⚠️ batch_bucket_files 的 add 只吃 (bytes | path, dest), **不吃文件对象**
        batch_bucket_files(bucket, add=[(payload.encode(),
                                        f"{prefix}/cells/{key.replace(':', '_')}.json")],
                           token=tok or None)
    except Exception as e:
        print(f"    [warn] 该格未推上桶: {repr(e)[:120]}", flush=True)


def main():
    print("=" * 78)
    print("V50.0 — 「融合收益从哪来」审计")
    print("=" * 78)
    print(f"臂: {list(ARMS)}")
    print(f"样本量档: {SAMPLE_SIZES} | BATCH={BATCH} STEPS={STEPS} PROXY={PROXY_STEPS} SEEDS={N_SEEDS}")
    t0 = time.time()
    sizes = {k: v for k, v in SAMPLE_SIZES.items() if not ONLY_SIZES or k in ONLY_SIZES}
    enc, tr_labels, ho_labels = encode_all(DEVICE, max(sizes.values()))
    print(f"  编码完成 ({time.time()-t0:.0f}s)")

    results = _load_previous()
    if results:
        print(f"  从桶里恢复了 {len(results)} 格已有结果: {sorted(results)}")
    for size_name, n_per_class in sizes.items():
        for arm in ARMS:
            if ONLY_ARMS and arm not in ONLY_ARMS:
                continue
            t1 = time.time()
            rows = [run_one(arm, n_per_class, enc, tr_labels, ho_labels, seed, DEVICE)
                    for seed in range(N_SEEDS)]
            results[f"{size_name}:{arm}"] = {"rows": rows}
            _push_cell(f"{size_name}:{arm}", rows)      # 每格立刻进桶 (抗 Space 重建)
            n = len(rows)
            held = sum(r["held_out"]["fuse"] for r in rows) / n
            sbig = sum(r["held_out"]["single_big"] for r in rows) / n
            ps = sum(r["probe"]["source_te"] for r in rows) / n
            pc = sum(r["probe"]["class_te"] for r in rows) / n
            pin = sum(r["probe"]["input_source_te"] for r in rows) / n
            print(f"  [{size_name:6s}] {arm:20s} held={held:8.4f} vs单大={(sbig-held)/max(sbig,1e-9)*100:+6.1f}% "
                  f"| 探针 源专家={ps:.3f} 类别={pc:.3f} 输入可识别={pin:.3f}  ({time.time()-t1:.0f}s)", flush=True)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "research", "aligner", "v50_fusion_gain_audit", "results")
    os.makedirs(out_dir, exist_ok=True)
    mpath = os.path.join(out_dir, "metrics.json")
    payload = {"schema_version": "1.0", "version": "50.0", "package": "v50_fusion_gain_audit",
               "line": "aligner",
               "config": {"d_shared": D_SHARED, "batch": BATCH, "steps": STEPS,
                          "proxy_steps": PROXY_STEPS, "seeds": N_SEEDS, "tokens": S_TOK},
               "results": results,
               "notes": {"E1": "proxy 对齐(无标签)后冻结的投影, margin 若塌 → 随机空间假象说成立",
                         "E2": "融合输出上 源专家可解码性 >> 类别可解码性 → 退化任务说成立",
                         "baselines": "随机基线: 源专家 1/3=0.333, 类别 1/8=0.125"}}
    if os.path.exists(mpath):
        try:
            old = json.load(open(mpath, encoding="utf-8"))
            old.setdefault("results", {}).update(payload["results"])
            old["config"], old["notes"] = payload["config"], payload["notes"]
            payload = old
        except Exception:
            pass
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n指标已落盘: {mpath}")


if __name__ == "__main__":
    main()
