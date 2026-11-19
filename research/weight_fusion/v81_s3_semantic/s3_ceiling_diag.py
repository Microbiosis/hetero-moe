"""s3_ceiling_diag.py — **post_hoc_only** 的读数解读诊断 (v81, S3).

⚠️ **这不是判据**, 不参与 S3 的任何裁决, 也不推翻任何已写死的阈值。
   它只回答一个**读数解读**问题, 并且只做一件事: 把绝对 R² 换成能读懂的量。

背景 (两条已经实测的事实):
  (1) **注入只带源侧激活的 256 维子空间**: Δ_B = a_s @ P_Aᵀ @ P_B, P_A 是 256 维
      正交投影 ⇒ 注入只携带 a_s 的 256 维子空间。
  (2) **绝对 R² 被共享成分占满**: 中途盘点实测, 不注入时 (arm0) 探针已能读出
      Qwen 的 Q 槽 (Qwen→BERT 方向 arm0 中位 R² ≈ 0.15, 最高 0.40) —— 这是两个
      模型共享的语义成分, 与注入无关。

  ⇒ 绝对 R² 不是「穿透了多少」; **Δ = arm1 − arm0 才是**。

本脚本算三样东西:
  r2_project  = Var(P_A P_Aᵀ a_s) / Var(a_s)  —— 注入**本来就能带**的方差比例
                (「注入的信息天花板」; 测法: 用注入子空间对 a_s 做最优线性重构)
  delta       = 中位(arm1) − 中位(arm0)       —— 注入带来的**新增**可读性
  delta_over_project = delta / r2_project     —— 传输效率的**下界**

为什么是下界: Δ ≤ r2_project 恒成立 (可读性增益不会超过注入进来的方差);
  即便目标模型**完美无损**地转发了注入, 注入子空间与「已经被读出的共享成分」仍有
  重叠, 那部分不会变成**新增**可读性 ⇒ Δ/r2_project ≤ 真实传输效率。

实现说明:
  - 源侧槽激活按**同一口径**重算 (样本级 mean-pool; 与 s3_flow.py 的 src_cache 同款);
  - 投影矩阵用 v62 的 projections(seed), 与 s3_flow.py 逐位同源;
  - 训练/测试切分与 s3_flow.py 一致 (文本前 64 训 / 后 64 测; ViT 按类别分层 32/32);
  - 输出 results/ceiling_diag.json。
"""
import gc
import importlib.util as ilu
import json
import os
import statistics as st
import sys
from collections import defaultdict
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

SEEDS = _S3.SEEDS
L_Q, L_B, L_V = _S3.L_Q, _S3.L_B, _S3.L_V
TEXT_SPLIT_TRAIN = _S3.TEXT_SPLIT_TRAIN
VIT_SPLIT_TRAIN = _S3.VIT_SPLIT_TRAIN
N_TEXT, N_VIT = _S3.N_TEXT, _S3.N_VIT


def _r2_from_residual(resid, target, train_mask):
    """与探针同口径的留出集 R²: 用训练集均值中心化，仅在留出集 (~train_mask) 上评估。"""
    test_mask = ~np.asarray(train_mask, dtype=bool)
    m = target[train_mask].mean(0, keepdim=True)
    b = resid[train_mask].mean(0, keepdim=True)
    r = resid[test_mask] - b
    y = target[test_mask] - m
    ss_res = float(r.pow(2).sum())
    ss_tot = float(y.pow(2).sum())
    return 1.0 - ss_res / (ss_tot + 1e-12)


def main() -> int:
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(min(16, os.cpu_count() or 8))

    metrics_path = _PKG / "results" / "metrics.json"
    if not metrics_path.exists():
        print("[skip] results/metrics.json 还不存在 (跑批未完成)", flush=True)
        return 0
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    fetch_models()
    texts = fetch_texts(N_TEXT)
    imgs, vit_meta = fetch_vit_images()
    proj_cache = {s: MEAS.projections(s) for s in SEEDS}

    # 源侧槽激活 (样本级 mean-pool), 与 s3_flow.py 同口径
    src = {}
    for name, layers in (("Qwen", L_Q), ("BERT", L_B), ("ViT", L_V)):
        bb = Backbone(name)
        batches = make_batches(bb, name, texts, imgs)
        want = [(ly, "Q") for ly in layers]
        acc = {k: [] for k in want}
        for b in batches:
            stt = capture_slots(bb, b, want)
            for k in want:
                acc[k].append(stt[k].mean(dim=1))
        src[name] = {k: torch.cat(v, 0) for k, v in acc.items()}
        del bb, batches, acc
        gc.collect()
        print("[source] %s 槽激活重算完成" % name, flush=True)

    # 每 (源模型, 源层, 种子) 的信息天花板 r2_project
    ceiling = {}
    for name in ("Qwen", "BERT", "ViT"):
        layers = {"Qwen": L_Q, "BERT": L_B, "ViT": L_V}[name]
        is_vit = (name == "ViT")
        n = N_VIT if is_vit else N_TEXT
        split = VIT_SPLIT_TRAIN if is_vit else TEXT_SPLIT_TRAIN
        train_mask = np.zeros(n, dtype=bool)
        train_mask[:split] = True
        ceiling[name] = {}
        for ly in layers:
            a = src[name][(ly, "Q")]
            ceiling[name][str(ly)] = {}
            for seed in SEEDS:
                P = proj_cache[seed][name]
                z = a @ P.t()
                resid = a - (z @ P)
                ceiling[name][str(ly)][str(seed)] = _r2_from_residual(resid, a, train_mask)

    # 逐块对账: 臂间中位差 + 天花板
    rows = metrics["blocks"]
    per_arm = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per_arm[(r["direction"], r["source_layer"], r["target_layer"])][r["arm"]].append(r)

    per_block = []
    for key, arms in sorted(per_arm.items()):
        med = {a: st.median([x["r2_q"] for x in v]) for a, v in arms.items()}
        dname, ly, j = key
        s_model = rows[0]["source_model"]
        for r in rows:
            if (r["direction"], r["source_layer"], r["target_layer"]) == key:
                s_model = r["source_model"]
                break
        cap = ceiling[s_model][str(ly)]
        cap_med = st.median(list(cap.values()))
        delta = med["arm1_slot_Q"] - med["arm0_no_inject"]
        per_block.append({
            "direction": dname, "source_model": s_model,
            "source_layer": int(ly), "target_layer": int(j),
            "r2_q_arm0_baseline": med["arm0_no_inject"],
            "r2_q_arm1": med["arm1_slot_Q"],
            "r2_q_arm2_wrong_slot": med["arm2_wrong_slot_O"],
            "r2_q_arm3_random": med["arm3_random_dir"],
            "delta_arm1_minus_arm0": delta,
            "delta_arm1_minus_arm3": med["arm1_slot_Q"] - med["arm3_random_dir"],
            "r2_project_ceiling": cap_med,
            "delta_over_project": (delta / cap_med) if cap_med > 1e-9 else None,
        })

    def _agg(sel, field):
        vals = [b[field] for b in per_block if sel(b) and b[field] is not None]
        if not vals:
            return None
        return {"n": len(vals), "median": float(st.median(vals)),
                "mean": float(st.mean(vals)),
                "min": float(min(vals)), "max": float(max(vals)),
                "n_positive": int(sum(1 for v in vals if v > 0))}

    summary = {}
    for name, sel in (("all_text_blocks", lambda b: b["direction"] != "ViT->ViT"),
                      ("BERT->Qwen", lambda b: b["direction"] == "BERT->Qwen"),
                      ("Qwen->BERT", lambda b: b["direction"] == "Qwen->BERT"),
                      ("ViT->ViT", lambda b: b["direction"] == "ViT->ViT")):
        summary[name] = {
            "arm0_baseline_r2": _agg(sel, "r2_q_arm0_baseline"),
            "delta_arm1_minus_arm0": _agg(sel, "delta_arm1_minus_arm0"),
            "r2_project_ceiling": _agg(sel, "r2_project_ceiling"),
            "delta_over_project_lower_bound": _agg(sel, "delta_over_project"),
        }

    out = {
        "role": "post_hoc_only —— **读数解读诊断**, 不参与 S3 任何裁决, 不推翻任何阈值",
        "question": "绝对 R² 被共享成分占满; 把 R² 换成「注入带来的新增可读性 / 注入能带的信息量」",
        "definition": {
            "r2_project_ceiling": "Var(P_A P_Aᵀ a_s) / Var(a_s) —— 注入的 256 维子空间能带走的方差比例",
            "delta_arm1_minus_arm0": "中位(arm1) − 中位(arm0) —— 注入带来的**新增**可读性 (绝对 R² 不是穿透量)",
            "delta_over_project_lower_bound": "delta / r2_project —— 传输效率的**下界** (Δ ≤ r2_project 恒成立; "
                                             "即便完美转发, 注入子空间与共享成分的重叠也不会变成新增可读性)",
        },
        "ceiling_by_source_layer": ceiling,
        "summary": summary,
        "per_block": per_block,
        "note": "本文件只做读数解读; S3 的判据 (P1a/P1b/P2/P3①/P3②/P4) 与阈值一字未动。",
    }
    p = _PKG / "results" / "ceiling_diag.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[ceiling] 已写出 %s" % p, flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
