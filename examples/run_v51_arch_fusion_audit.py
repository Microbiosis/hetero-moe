"""V51.0 — 架构异构线 (v6/v7) 的「融合收益」审计.

> 归属路线: aligner (接 v48/v49/v50)
> 复现: bash run.sh v51

被审计的主张 (论文 §6.1/§6.2 与摘要):
  (a) "FFN-only routing beats the best single expert by +20.3% ... the first demonstration
      that genuinely different pretrained architectures can share one sparse router"
  (b) "dual-path routing raises the margin over averaging to +84.8%"
  (c) "the cross-architecture gain concentrates in the attention path (+82.7% in-sample,
      +70.6% held-out)"

v50 已经审计了规模异构线 (v33) → 发现那个 margin 是"源身份识别"的假象。本代把同样的
两把尺子用到**架构异构线**上, 并且发现这条线有一个更根本的问题:

  ★ **v7 的目标块就是模态下标** (SPEC/runner: "目标: 每模态一个 class direction")
    → 这条线的任务**本身就是"识别来自哪个底座"**。
    所以 "融合优于单专家" 在这里等价于 "能分清模态的模型赢过不能分清的模型",
    与"语义融合"无关。

两把尺子:
  尺 1 (目标置换对照): 把"模态 m → 目标块 m"换成"模态 m → 目标块 π(m)" (固定随机置换)。
      任务难度不变 (仍然只需认出模态再查表), 只有"标签对应关系"变了。
      若 margin 完全不变 → margin 是**识别机制**的产物, 与底座之间任何语义关系无关。
      (这正是论文 §6.3 自己的 "shifted stimuli" 协议在标签维度上的版本。)
  尺 2 (有意义的空间): 把冻结随机线性 P_m 换成**非线性可训练**的抬升 (v50 已验证的
      "有意义空间"配方)。若 margin 塌 → 与 v50 的规模异构线结论一致。
  尺 3 (探针): 从融合输出线性探测 (a) 源模态身份 (b) 目标块。

四臂 (2 × 2):
  A 冻结随机线性 + 目标对齐   = v7 原配置
  B 冻结随机线性 + 目标置换
  C 非线性可训练 + 目标对齐
  D 非线性可训练 + 目标置换
"""
import sys, os, json, gc, time, importlib.util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 复用 v7 的 runner: 模型加载 / attn 抽取 / 编码 / 层构造 → 保证与论文数字同源
_spec = importlib.util.spec_from_file_location("v7run", os.path.join(ROOT, "examples/run_v7_cross_arch.py"))
v7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v7)

from research.routing_evolution.v7_attention.trainer import build_v7_param_groups
from research.aligner.v49_projection_space.mechanism import NonlinearProj
from research.aligner.v50_fusion_gain_audit.mechanism import linear_probe

D_SHARED = v7.D_SHARED
N_CLS = v7.N_CLS
M = 3
S = v7.S
STEPS = int(os.environ.get("V51_STEPS", "100"))
LR = 1e-2
N_SEEDS = int(os.environ.get("V51_SEEDS", "5"))
DEVICE = torch.device(os.environ.get("V51_DEVICE", "cpu"))
ONLY_ARMS = [x for x in os.environ.get("V51_ARMS", "").split(",") if x]

ARMS = {
    "A_frozen_aligned":   ("frozen", False),   # v7 原配置
    "B_frozen_permuted":  ("frozen", True),
    "C_trainable_aligned": ("trainable", False),
    "D_trainable_permuted": ("trainable", True),
}
EXPERTS = ["bert", "llama", "vit"]
MODALITY = {0: "text", 1: "code", 2: "image"}


def build_samples(encoders, seed, n_per_expert=6, perturb=0.1, tag=0):
    """v7 的样本构造: 每模态 6 条 (扰动), 目标 = 模态块 (或置换后的块)。"""
    (bert_tok, bert, D_bert, _, _), (llama_tok, llama, D_llama, _, _), \
        (vit_proc, vit, D_vit, _, _) = encoders
    texts = ["cat dog bird", "animal pet wild", "feline canine fowl"]
    codes = ["def hello():", "return True", "pass None"]
    img = np.random.randint(0, 256, (224, 224, 3))
    h_blocks = [v7.encode_text(texts, bert_tok, bert),
                v7.encode_code(codes, llama_tok, llama),
                v7.encode_image(img, vit_proc, vit)]
    seqs, idxs = [], []
    for m, h_block in enumerate(h_blocks):
        n = h_block.shape[0]
        for i in range(n_per_expert):
            src = h_block[i % n]
            g = torch.Generator().manual_seed(seed * 10 + m * 6 + i + tag * 9973)
            seqs.append((src + perturb * torch.randn(src.shape, generator=g)).to(DEVICE))
            idxs.append(m)
    return seqs, idxs


def targets_for(modal_indices, perm=None):
    out = []
    for m in modal_indices:
        blk = m if perm is None else perm[m]
        t = torch.zeros(S, D_SHARED)
        ds = blk * (D_SHARED // N_CLS)
        de = (blk + 1) * (D_SHARED // N_CLS)
        t[:, ds:de] = 1.0
        out.append(t.to(DEVICE))
    return torch.stack(out, dim=0)


def run_one(arm, encoders, seed):
    kind, permuted = ARMS[arm]
    torch.manual_seed(seed)
    layer = v7.build_layer(encoders).to(DEVICE)
    for p in layer.P_m:
        p.requires_grad_(False)

    tr_seqs, tr_idx = build_samples(encoders, seed, tag=0)
    ho_seqs, ho_idx = build_samples(encoders, seed, tag=1)
    # 目标置换: 固定随机置换 (跨 seed 固定, 同一置换)
    # 固定的无不动点错位 (derangement): 模态 m → 块 (m+1) % N_CLS
    # 必须是 derangement —— 若置换里有不动点, "对齐 vs 置换" 的对照就不彻底
    perm = [(m + 1) % N_CLS for m in range(N_CLS)] if permuted else None
    tr_t = targets_for(tr_idx, perm)
    ho_t = targets_for(ho_idx, perm)

    Dm = [layer.P_m[m].shape[1] for m in range(M)]
    # 投影器: 冻结随机线性 = 用层自己的 P_m (与 v7 逐位一致); 可训练 = 非线性抬升
    proj = None
    if kind == "trainable":
        proj = NonlinearProj(D_SHARED, Dm[0]).to(DEVICE)   # 占位, 下面按底座建
        proj = {m: NonlinearProj(D_SHARED, Dm[m]).to(DEVICE) for m in range(M)}

    def project(seqs, idxs):
        xs = []
        for x, m in zip(seqs, idxs):
            if kind == "frozen":
                xs.append(F.linear(x.unsqueeze(0), layer.P_m[m]))    # [1,S,D]
            else:
                xs.append(proj[m](x.unsqueeze(0)))
        return torch.cat(xs, dim=0)

    params = [layer.W_router_attn, layer.W_router_ffn] + list(layer.alphas_attn) \
        + list(layer.alphas) + list(layer.gammas) + list(layer.betas)
    if kind == "trainable":
        for m in range(M):
            params += [p for p in proj[m].parameters()]
    opt = torch.optim.AdamW(build_v7_param_groups(layer, 2) if kind == "frozen" else
                             [{"params": [p for p in params if p.numel() <= 4096], "lr": LR},
                              {"params": [p for p in params if p.numel() > 4096], "lr": 1e-3}],
                             lr=LR, weight_decay=1e-4)

    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        # ⚠️ 投影必须在循环**内**算: 可训练臂的投影带图, 循环外算一次再复用
        #    会在第二次 backward 时报 "backward through the graph a second time"
        y = layer(project(tr_seqs, tr_idx))
        loss = F.mse_loss(y, tr_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

    with torch.no_grad():
        ho_x = project(ho_seqs, ho_idx)
        y_ho, ahat_a, ahat_f = layer.forward_with_routing(ho_x)
        fuse = F.mse_loss(y_ho, ho_t).item()
        attn_sum, _ = layer._attn_path(ho_x)
        attn_only = F.mse_loss(ho_x + attn_sum, ho_t).item()
        ffn_sum, _ = layer._ffn_path(ho_x)
        ffn_only = F.mse_loss(ho_x + ffn_sum, ho_t).item()
        x_det = ho_x.detach()
        tot = torch.zeros_like(ho_x)
        single = []
        for m in range(M):
            h_a = x_det * layer.gammas[m] + layer.betas[m]
            gate = F.linear(h_a, layer.w_gates[m])
            up = F.linear(h_a, layer.w_ups[m])
            f = F.linear(F.silu(gate) * up, layer.w_downs[m])
            tot = tot + f
            single.append(F.mse_loss(ho_x + layer.alphas[m] * f, ho_t).item())
        avg = F.mse_loss(ho_x + tot / M, ho_t).item()
        best_single = min(single)
        y_flat = y_ho.reshape(-1, D_SHARED).detach()
    # ⚠️ 探针必须在 no_grad 之外: 探针自己要训参数 (backward 会报
    #    "element 0 of tensors does not require grad")
    src = torch.tensor(ho_idx).repeat_interleave(S).to(DEVICE)
    cls = torch.tensor([ho_idx[i] if perm is None else perm[ho_idx[i]] for i in range(len(ho_idx))]).repeat_interleave(S).to(DEVICE)
    p_src, _ = linear_probe(y_flat, src, y_flat, src, n_class=M, seed=seed)
    p_cls, _ = linear_probe(y_flat, cls, y_flat, cls, n_class=N_CLS, seed=seed)
    return {"held_out": {"fuse": fuse, "attn_only": attn_only, "ffn_only": ffn_only,
                         "avg": avg, "best_single": best_single, "single_each": single},
            "in_sample_fuse": F.mse_loss(layer(project(tr_seqs, tr_idx)), tr_t).item(),
            "probe": {"source_te": p_src, "class_te": p_cls},
            "perm": perm}


def main():
    print("=" * 78)
    print("V51.0 — 架构异构线 (v6/v7) 融合收益审计")
    print("=" * 78)
    print("臂:", list(ARMS))
    print(f"STEPS={STEPS} SEEDS={N_SEEDS} | 协议 = v7 原协议 (TinyBERT/TinyLlama/ViT-tiny, D_shared=256)")
    encoders = v7.load_encoders_and_attn()
    results = {}
    for arm in ARMS:
        if ONLY_ARMS and arm not in ONLY_ARMS:
            continue
        t0 = time.time()
        rows = [run_one(arm, encoders, seed) for seed in range(N_SEEDS)]
        results[arm] = {"rows": rows}
        n = len(rows)
        g = lambda k: sum(r["held_out"][k] for r in rows) / n
        fuse, avg, bs = g("fuse"), g("avg"), g("best_single")
        print(f"  {arm:24s} fuse={fuse:8.4f} attn_only={g('attn_only'):8.4f} ffn_only={g('ffn_only'):8.4f} "
              f"avg={avg:8.4f} best_single={bs:8.4f} | vs单专家={(bs-fuse)/max(bs,1e-9)*100:+6.1f}% "
              f"vs avg={(avg-fuse)/max(avg,1e-9)*100:+6.1f}% | 探针 源模态={sum(r['probe']['source_te'] for r in rows)/n:.3f} "
              f"目标块={sum(r['probe']['class_te'] for r in rows)/n:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    out_dir = os.path.join(ROOT, "research/aligner/v51_arch_fusion_audit/results")
    os.makedirs(out_dir, exist_ok=True)
    mpath = os.path.join(out_dir, "metrics.json")
    payload = {"schema_version": "1.0", "version": "51.0", "package": "v51_arch_fusion_audit",
               "line": "aligner",
               "config": {"d_shared": D_SHARED, "steps": STEPS, "seeds": N_SEEDS, "n_cls": N_CLS},
               "results": results,
               "notes": {"protocol": "v7 原协议 (TinyBERT 文本 / TinyLlama 代码 / ViT-tiny 图像), 三个真正不同的架构与模态",
                         "arm_A": "= v7 原配置 (冻结随机线性 P_m + 目标块=模态下标)",
                         "arm_B": "目标置换: 模态 m → 块 π(m) (固定随机置换)",
                         "arm_C": "非线性可训练抬升 (v50 的'有意义空间'配方) + 目标对齐",
                         "arm_D": "非线性可训练 + 目标置换",
                         "key_fact": "v7 的目标块就是模态下标 → 该线的任务本身是'识别来源底座'",
                         "baselines": "随机基线: 源模态 1/3=0.333, 目标块 1/3=0.333"}}
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
    _push_metrics(mpath)


def _push_metrics(mpath):
    """runner 自己把指标推进桶 (不依赖 app 的拷贝路径)。"""
    try:
        import os as _os
        from huggingface_hub import batch_bucket_files
        bucket = _os.environ.get("V51_BUCKET", "redmingwei/hetero-moe")
        prefix = _os.environ.get("V51_BUCKET_PREFIX", "v51_artifacts")
        tok = _os.environ.get("HF_TOKEN")
        batch_bucket_files(bucket, add=[(mpath, f"{prefix}/metrics.json")], token=tok or None)
        print(f"  已推桶: {bucket}/{prefix}/metrics.json")
    except Exception as e:
        print(f"  [warn] 指标未推上桶: {repr(e)[:140]}")


if __name__ == "__main__":
    main()
