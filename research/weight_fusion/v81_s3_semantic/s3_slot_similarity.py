"""s3_slot_similarity.py — **post_hoc_only** 的「同层 Q 槽 vs O 槽可互换性」诊断 (v81, S3).

⚠️ **这不是判据**, 不参与 S3 的任何裁决, 也不推翻任何已写死的阈值。
   它回答一个**对照是否有区分力**的问题。

背景: P1b (槽特异) 与 P3② (选择性) 都在预登记中**失败**。失败有两种可能:
  (a) 机制上真的没有槽对应 (注入 Q 槽与注入 O 槽的效果一样);
  (b) **对照本身没有区分力** —— 若同一层内 a_Q 与 a_O 本就高度可互换,
      那「注入 O 槽也能让 Q 槽更可读」是同义反复, 不能据此断言机制无选择性。

本脚本测的是 (b) 需要的那个量: **a_Q 能不能线性预测 a_O**。
  - R²(a_Q → a_O) 高 ⇒ 两槽在该层高度可互换 ⇒ arm1/arm2 的注入内容近乎等价
    ⇒ P1b / P3② 读不出差别是**对照无区分力**, 不是机制无选择性;
  - R²(a_Q → a_O) 低 ⇒ 两槽内容确实不同 ⇒ arm1 ≠ arm2 的注入内容是实质性的
    ⇒ P1b 的失败就**不能**用同义反复解释。

口径:
  - 源侧槽激活按**同一口径**重算 (样本级 mean-pool; 与 s3_flow.py 的 src_cache 同款);
  - 逐层用**留出集**线性预测 (ridge, lambda=10) 测 R²; 切分与 s3_flow.py 一致
    (文本前 64 训 / 后 64 测; ViT 按位置前 32 训 / 后 32 测 —— ⚠️ 见下方说明);
  - 输出 results/qo_similarity.json。

⚠️ **ViT 的切分是位置切分**, 而 fetch_vit_images 的图序是**按类排列**的
   ⇒ ViT 的 train 类别 {0,1} 与 test 类别 {2,3} **完全不相交**
   ⇒ ViT 的 R² 是**跨类别泛化**读数, 比文本臂更严苛, 不可与文本臂直接比大小。
   (这一条同时是 v81 的**已知缺陷**, 见 SPEC.md §4.5; 只影响 ViT 臂的标签可读性
   探针与 P1c, **不影响**主裁决。)
"""
import gc
import importlib.util as ilu
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_PKG = Path(__file__).resolve().parent


def _load_by_path(path: Path, name: str):
    spec = ilu.spec_from_file_location(name, str(path))
    mod = ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_S3 = _load_by_path(_PKG / "s3_flow.py", "s3_flow_v81_by_path")

MEAS = _S3.MEAS
Backbone = _S3.Backbone
capture_slots = _S3.capture_slots
fetch_models = _S3.fetch_models
fetch_texts = _S3.fetch_texts
fetch_vit_images = _S3.fetch_vit_images
make_batches = _S3.make_batches

L_Q, L_B, L_V = _S3.L_Q, _S3.L_B, _S3.L_V
N_TEXT, N_VIT = _S3.N_TEXT, _S3.N_VIT
TEXT_SPLIT_TRAIN = _S3.TEXT_SPLIT_TRAIN
VIT_SPLIT_TRAIN = _S3.VIT_SPLIT_TRAIN

RIDGE_LAM = 10.0


def main() -> int:
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(min(16, os.cpu_count() or 8))

    fetch_models()
    texts = fetch_texts(N_TEXT)
    imgs, meta = fetch_vit_images()

    out = {}
    for name, layers in (("Qwen", L_Q), ("BERT", L_B), ("ViT", L_V)):
        bb = Backbone(name)
        batches = make_batches(bb, name, texts, imgs)
        want = [(ly, "Q") for ly in layers] + [(ly, "O") for ly in layers]
        acc = {k: [] for k in want}
        for b in batches:
            st = capture_slots(bb, b, want)
            for k in want:
                acc[k].append(st[k].mean(dim=1))
        acc = {k: torch.cat(v, 0) for k, v in acc.items()}
        n = N_VIT if name == "ViT" else N_TEXT
        split = VIT_SPLIT_TRAIN if name == "ViT" else TEXT_SPLIT_TRAIN
        rows = []
        for ly in layers:
            aQ, aO = acc[(ly, "Q")], acc[(ly, "O")]
            cos = torch.nn.functional.cosine_similarity(aQ, aO, dim=1)
            tr, te = slice(0, split), slice(split, n)
            Xtr, Xte = aQ[tr], aQ[te]
            Ytr, Yte = aO[tr], aO[te]
            mu = Xtr.mean(0, keepdim=True)
            sd = Xtr.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
            Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
            d = Xtr.shape[1]
            A = torch.linalg.solve(Xtr.t() @ Xtr + RIDGE_LAM * torch.eye(d), Xtr.t() @ Ytr)
            ym = Ytr.mean(0, keepdim=True)
            pred = Xte @ A
            ssr = float((pred - (Yte - ym)).pow(2).sum())
            sst = float((Yte - ym).pow(2).sum())
            r2 = 1.0 - ssr / (sst + 1e-12)
            rows.append({
                "layer": int(ly), "D": int(aQ.shape[1]),
                "cos_raw_mean": round(float(cos.mean()), 6),
                "cos_raw_std": round(float(cos.std()), 6),
                "r2_Q_predicts_O": round(r2, 6),
            })
        out[name] = rows
        for r in rows:
            print("%-5s layer=%-3d D=%-4d cos_raw=%+.4f+-%.4f  R2(a_Q->a_O)=%+.4f"
                  % (name, r["layer"], r["D"], r["cos_raw_mean"], r["cos_raw_std"],
                     r["r2_Q_predicts_O"]), flush=True)
        del bb, batches, acc
        gc.collect()

    p = _PKG / "results" / "qo_similarity.json"
    p.write_text(json.dumps({
        "role": "post_hoc_only —— 对照区分力诊断, 不参与 S3 任何裁决",
        "question": "同层内 Q 槽与 O 槽是否高度可互换 (若是, P1b/P3② 的对照无区分力)",
        "metric": "留出集 R2(a_Q -> a_O) (ridge lambda=%.g)" % RIDGE_LAM,
        "caveat_vit_split": "ViT 用位置切分且图序按类排列 ⇒ train 类别 {0,1} 与 test 类别 {2,3} 不相交 ⇒ ViT 读数是跨类别泛化, 不可与文本臂比大小",
        "by_model_layer": out,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[qo] 已写出 %s" % p, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
