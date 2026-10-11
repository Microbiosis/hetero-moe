"""v47.0 — 中枢家族 (v8/v12/v19/v20/v30/v31) 的样本外审计 runner.

被审计的六个包共用同一套数据构造 (``examples/run_v8_full.py:178-202``,
``examples/run_v19_c1_revisit.py:90-112``), 六个 runner **全部在训练同源数据上
评估** (eval 块逐字复用训练用的 ``h_by_modal`` / ``t_by_modal``), 且 grep
``held_out|heldout`` 全部 0 命中。

三个口径
--------
- ``in_sample``      与原 runner 同构 (换掉未受控的 numpy 随机源, 见下)
- ``heldout_noise``  ``seed + 1000``: 同刺激, 换噪声 + 换随机图像
- ``heldout_stimulus`` **换刺激**: 换文本 / 换代码 / 换图像, 噪声亦换

口径纪律
--------
读数一律**配对逐种子中位数**, 不报均值 (本项目自 v40 起的纪律: 均值会被单种子
主导)。判定交给 ``research.central_stability.v47_heldout_audit``。

对原 runner 的一处**偏离** (是修复, 不是改口径)
------------------------------------------------
原 runner 只调 ``torch.manual_seed(seed)``, 随后 ``np.random.randint`` 取图像 ——
numpy 的全局 RNG **不受 torch 种子控制**, 所以同一个 ``seed`` 两次运行会得到不同
图像, 逐 seed 数字本就不可复现。本 runner 在三个口径上都显式 ``np.random.seed``,
使 in-sample 读数可复现、三个口径可比。

用法::

    python3 examples/run_v47_central_heldout.py                 # 全部 6 包, 5 seeds
    V47_PACKAGES=v8,v30 python3 examples/run_v47_central_heldout.py
    V47_SEEDS=1 python3 examples/run_v47_central_heldout.py     # 冒烟
"""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from research.central_stability.v47_heldout_audit import (
    audit_table, leakage_verdict, paired_gain, retention,
)

from examples.run_v8_full import (
    load_encoders, make_pools, encode_text, encode_code, encode_image,
    D_SHARED, N_CLS, S, SEEDS,
)
from examples.run_v8_full import make_layer as make_v8_layer
from examples.run_v12_compose import make_layer as make_v12_layer, MODES as V12_MODES
from examples.run_v20_h5_compare import PositionLayer, POSITIONS
from examples.compare_v9_v30_v31 import TheoryLayerWithMode, build_param_groups

HO_OFFSET = 1000
AUDITED = ["v8", "v12", "v19", "v20", "v30", "v31"]
OUT = "examples/run_v47_central_heldout.py (5 seeds × 3 口径)"

# 训练刺激 (与被审计 runner 逐字一致)
TEXTS = ["cat dog bird", "animal pet wild", "feline canine fowl"]
CODES = ["def hello():", "return True", "pass None"]
# 留出刺激: 同为短自然语言 / 短代码, 但与训练集无交集
HO_TEXTS = ["river stone cloud", "iron glass paper", "morning night noon"]
HO_CODES = ["class Foo:", "bar = 1", "baz None"]

STEPS_V8 = 100
STEPS_V19_LONG = 500
STEPS_THEORY = 200


# =====================================================================
# 数据构造 (三个口径共用一个入口)
# =====================================================================

_ENC_CACHE: dict = {}


def _memo(key, fn):
    """block 级记忆化 —— 只缓存编码器前向, 不缓存任何训练/评估结果。"""
    if key not in _ENC_CACHE:
        _ENC_CACHE[key] = fn()
    return _ENC_CACHE[key]


def build_data(scope, seed, encoders):
    """按口径构造 18 个样本 (3 模态 × 6)。

    ``in_sample``      seed 原样, 刺激 = TEXTS/CODES
    ``heldout_noise``  seed+1000, 刺激不变 ⇒ 只换噪声抽样
    ``heldout_stimulus`` seed+1000, 刺激换成 HO_TEXTS/HO_CODES ⇒ 未见过的输入

    目标 (targets) 只由模态类别决定, 三个口径完全一致 —— 这是刻意的:
    要变的是**输入**, 不是任务定义。
    """
    draw = seed if scope == "in_sample" else seed + HO_OFFSET
    texts = TEXTS if scope != "heldout_stimulus" else HO_TEXTS
    codes = CODES if scope != "heldout_stimulus" else HO_CODES

    torch.manual_seed(draw)
    np.random.seed(draw)                      # 修复: numpy 全局 RNG 原本不受 torch 种子控制
    (bert_tok, bert, _, _), (llama_tok, llama, _, _), (vit_proc, vit, _, _) = encoders

    # 编码器前向是本 runner 的瓶颈 (ViT 每张 224×224), 而文本/代码 block **只取决于
    # 口径** —— 逐 seed 的随机性全部来自噪声与图像。故 block 级记忆化。
    h_text = _memo(("text", scope), lambda: encode_text(texts, bert_tok, bert))
    h_code = _memo(("code", scope), lambda: encode_code(codes, llama_tok, llama))
    h_img = _memo(("img", scope, seed),
                  lambda: encode_image(np.random.randint(0, 256, (224, 224, 3)), vit_proc, vit))

    modal_seqs, modal_indices = [], []
    for cls, h_block in enumerate([h_text, h_code, h_img]):
        n = h_block.shape[0]
        for i in range(6):
            src = h_block[i % n]
            g = torch.Generator().manual_seed(draw * 10 + cls * 6 + i)
            modal_seqs.append(src + 0.1 * torch.randn(src.shape, generator=g))
            modal_indices.append(cls)

    targets = []
    for cls in range(N_CLS):
        ds = cls * (D_SHARED // N_CLS)
        de = (cls + 1) * (D_SHARED // N_CLS)
        t = torch.zeros(S, D_SHARED)
        t[:, ds:de] = 1.0
        for _ in range(6):
            targets.append(t)
    return modal_seqs, modal_indices, targets


def group(modal_seqs, modal_indices, targets):
    """按模态分组 —— v19/v20/v30/v31 的 eval/train 块用的就是这一种输入。"""
    h_by = [[] for _ in range(N_CLS)]
    t_by = [[] for _ in range(N_CLS)]
    for i, m in enumerate(modal_indices):
        h_by[m].append(modal_seqs[i])
        t_by[m].append(targets[i])
    for c in range(N_CLS):
        h_by[c] = torch.stack(h_by[c], dim=0)
        t_by[c] = torch.stack(t_by[c], dim=0)
    return h_by, t_by


def per_sample(layer, modal_seqs, modal_indices, targets):
    """逐样本投影 —— v8 / v12 的 eval/train 块用的就是这一种输入。"""
    x_list = [F.linear(h.unsqueeze(0), layer.P_m[cls])
              for h, cls in zip(modal_seqs, modal_indices)]
    return x_list, [t.unsqueeze(0) for t in targets]


def eval_layer(layer, path, data):
    """fuse MSE —— 与被审计 runner 的 eval 块逐字同式 (两种前向路径各自复刻)。"""
    modal_seqs, modal_indices, targets = data
    if path == "grouped":
        h_by, t_by = group(modal_seqs, modal_indices, targets)
        with torch.no_grad():
            y = layer(h_by)
            return sum(F.mse_loss(y[s:s + 1], t_by[c][s:s + 1]).item()
                       for c in range(N_CLS) for s in range(6)) / (N_CLS * 6)
    x_list, t_list = per_sample(layer, modal_seqs, modal_indices, targets)
    with torch.no_grad():
        return sum(F.mse_loss(layer(x), t).item() for x, t in zip(x_list, t_list)) / (N_CLS * 6)


def train_layer(layer, path, data, steps, lr, grad_clip=None, use_v8_optimizer=False):
    """训练回路按各包原样复刻 (三条: V8Trainer / AdamW / theory 参数分组)。"""
    modal_seqs, modal_indices, targets = data
    if path == "grouped":
        h_by, t_by = group(modal_seqs, modal_indices, targets)
        if grad_clip is not None:                      # v30/v31: theory 参数分组 + 梯度裁剪
            opt = torch.optim.AdamW(build_param_groups(layer), lr=lr)
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                y = layer(h_by)
                loss = sum(F.mse_loss(y, t_by[c]) for c in range(N_CLS)) / N_CLS
                loss.backward()
                torch.nn.utils.clip_grad_norm_(layer.parameters(), grad_clip)
                opt.step()
            return
        if use_v8_optimizer:                            # v19: V8Trainer 只提供优化器
            from research.routing_evolution.v8_central import V8Trainer
            trainer = V8Trainer(layer, lr=lr, phase=2)
            for _ in range(steps):
                trainer.optimizer.zero_grad(set_to_none=True)
                y = layer(h_by)
                loss = sum(F.mse_loss(y, t_by[c]) for c in range(N_CLS)) / N_CLS
                loss.backward()
                trainer.optimizer.step()
            return
        opt = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad], lr=lr)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            y = layer(h_by)
            loss = sum(F.mse_loss(y, t_by[c]) for c in range(N_CLS)) / N_CLS
            loss.backward()
            opt.step()
        return

    x_list, t_list = per_sample(layer, modal_seqs, modal_indices, targets)
    from research.routing_evolution.v7_attention import CrossArchAttnFFNFusionLayer, V7Trainer
    from research.routing_evolution.v8_central import V8Trainer
    trainer = (V7Trainer(layer, lr=lr, phase=2) if isinstance(layer, CrossArchAttnFFNFusionLayer)
               else V8Trainer(layer, lr=lr, phase=2))
    for _ in range(steps):
        trainer.step(x_list, t_list)


# =====================================================================
# 各包的 (treatment, baseline, 训练配置)
# =====================================================================


def _pair(kind, seed, attn_pools, encoders, lr, scopes, treat_mode, base_mode):
    """训练 treatment / baseline 各一次, 三个口径各评一次。返回 ``{scope: {slot: [值]}}``。"""
    dims = [e[2] for e in encoders]
    out = {s: {"treat": [], "base": []} for s in scopes}
    tr = build_data("in_sample", seed, encoders)

    for slot, mode in (("treat", treat_mode), ("base", base_mode)):
        torch.manual_seed(seed)
        path, steps, clip, v8opt = CONFIGS[kind]

        if kind == "v8":
            layer = make_v8_layer(mode, attn_pools, encoders)
        elif kind == "v12":
            layer = make_v12_layer(mode, attn_pools, encoders)
            path = "per_sample" if mode == "v9-linear-gate" else "grouped"
        elif kind == "v19":
            layer = _v19_layer(attn_pools, dims)
            steps = STEPS_V19_LONG if mode.endswith("-500") else STEPS_V8
        elif kind == "v20":
            layer = PositionLayer(d_shared=D_SHARED, num_experts_attn=3, num_experts_ffn=3,
                                  modal_dims=dims, attn_pools=attn_pools, position=mode)
        else:                                            # v30 / v31
            layer = TheoryLayerWithMode(d_shared=D_SHARED, num_experts=3, modal_dims=dims,
                                        attn_pools=attn_pools, mode=mode)

        train_layer(layer, path, tr, steps, lr, grad_clip=clip, use_v8_optimizer=v8opt)
        layer.eval()
        for scope in scopes:
            out[scope][slot].append(eval_layer(layer, path, build_data(scope, seed, encoders)))
    return out


def _v19_layer(attn_pools, dims):
    """v19 的层 —— 与 run_v19_c1_revisit.run_seed 的构造逐行一致。"""
    from research.aligner.v10_embedding import AlignedFusionLayer, CrossArchAttnAligner
    return AlignedFusionLayer(
        d_shared=D_SHARED, d_ff=4 * D_SHARED, attn_pools=attn_pools, modal_dims=dims,
        aligner=CrossArchAttnAligner(modal_dims=dims, d_shared=D_SHARED, num_heads=4),
        c1_alpha=0.0, c1_mode="broadcast")


# kind -> (前向路径, 训练步数, 梯度裁剪, 是否借用 V8Trainer 的优化器)
CONFIGS = {
    "v8":  ("per_sample", STEPS_V8, None, False),
    "v12": ("grouped",    STEPS_V8, None, False),
    "v19": ("grouped",    STEPS_V8, None, True),
    "v20": ("grouped",    STEPS_V8, None, False),
    "v30": ("grouped",    STEPS_THEORY, 1.0, False),
    "v31": ("grouped",    STEPS_THEORY, 1.0, False),
}

PACKAGES = [
    ("v8",  "C-1 中枢 vs 无中枢", "c1", "baseline",
     "C-1 中枢广播单独启用是否改善融合"),
    ("v12", "attn 对齐器 + gate 中枢 vs 单独 gate", "v12-attn-gate", "v9-linear-gate",
     "v9 + v10 是否叠加增益"),
    ("v19", "C-1 长训练 (500 步) vs 短训练 (100 步)", "c1-500", "c1-100",
     "H3: 训练步数不足是否为根因"),
    ("v20", "中枢放 attn vs 无中枢", "attn", "none",
     "中枢 broadcast 的位置是否影响效果"),
    ("v30", "softplus gate vs 纯 tanh", "v30", "v9",
     "softplus 能否修复梯度饱和"),
    ("v31", "梯度重缩放 gate vs 纯 tanh", "v31", "v9",
     "梯度重缩放能否修复饱和"),
]


def main():
    want = os.environ.get("V47_PACKAGES", "")
    wanted = {p.strip() for p in want.split(",") if p.strip()}
    n_seeds = int(os.environ.get("V47_SEEDS", str(SEEDS)))
    scopes = ("in_sample", "heldout_noise", "heldout_stimulus")

    print("=" * 92)
    print("v47.0 中枢家族样本外审计 — v8 / v12 / v19 / v20 / v30 / v31")
    print("受审计包: " + ", ".join(f"{k} ({c})" for k, _, _, _, c in PACKAGES))
    print(f"3 口径 × {n_seeds} seeds | 读数 = 配对逐种子中位数")
    print("=" * 92)

    t0 = time.time()
    encoders = load_encoders()
    # 修复: make_pools 里 AttnPool 的 P_m / W_o 是 nn.init.orthogonal_ 随机初始化,
    # 而 run_v8_full 只在 run_seed 内部调 torch.manual_seed(seed) —— attn 专家的输出
    # 投影因而**不受 seed 控制**, 跨进程逐位不同 (实测同一配置多次运行 in-sample
    # 读数在 0.23 ~ 0.61 之间跳)。这里给 attn pool 一个固定种子, 使本审计可复现。
    # 注意: 被审计的原 runner 没有这一步, 它们的数字本身带这一层未受控变异。
    torch.manual_seed(0)
    attn_pools = make_pools(encoders)
    print(f"  编码器就绪 ({time.time() - t0:.1f}s)\n")

    records, verdicts = [], {}
    for kind, label, treat_mode, base_mode, _claim in PACKAGES:
        if wanted and kind not in wanted:
            continue
        print(f"[{kind}] {label}  (treat={treat_mode} base={base_mode})")
        per_scope = {s: {"treat": [], "base": []} for s in scopes}
        for seed in range(n_seeds):
            r = _pair(kind, seed, attn_pools, encoders, 1e-2, scopes, treat_mode, base_mode)
            for s in scopes:
                per_scope[s]["treat"].extend(r[s]["treat"])
                per_scope[s]["base"].extend(r[s]["base"])
            print(f"    seed {seed}: in-sample {r['in_sample']['treat'][0]:.4f} vs "
                  f"{r['in_sample']['base'][0]:.4f} | noise {r['heldout_noise']['treat'][0]:.4f} | "
                  f"stimulus {r['heldout_stimulus']['treat'][0]:.4f}")

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

    print("\n" + "=" * 92)
    print(audit_table(records))
    print("=" * 92)
    out = {
        "scopes": list(scopes), "seeds": n_seeds, "verdicts": verdicts,
        "claims": {k: c for k, _, _, _, c in PACKAGES},
        "records": [{"name": r["name"],
                     **{s: {k: [round(x, 6) for x in v] for k, v in r[s].items()}
                        for s in scopes}} for r in records],
    }
    dest = os.environ.get(
        "V47_OUT", "research/central_stability/v47_heldout_audit/results/central_heldout.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    metrics = {
        "schema_version": "1.0", "version": "47.0",
        "package": "v47_heldout_audit", "line": "central_stability",
        "metric": "fuse_mse", "direction": "lower_is_better",
        "recorded_by": OUT,
        "statistic": "配对逐种子中位数 + 逐种子胜出数 (不用均值 — v40 起的纪律)",
        "scopes": list(scopes), "seeds": n_seeds,
        "audited": AUDITED,
        "verdicts": verdicts,
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
