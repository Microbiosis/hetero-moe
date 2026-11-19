"""summarize_v82.py — v82 结果汇总 (临时脚本, 跑批后用来写文档).

用法: python summarize_v82.py
"""
import collections
import json
import pathlib
import statistics
import sys

PKG = pathlib.Path(__file__).resolve().parent


def main(mname: str = "metrics.json"):
    m = json.loads((PKG / "results" / mname).read_text(encoding="utf-8"))
    cal_path = PKG / "results" / "calib.json"
    cal = json.loads(cal_path.read_text(encoding="utf-8")) if cal_path.exists() else {}
    rows = m["blocks"]
    crit = m["criteria"]

    print("=" * 78)
    print("verdict:", m["verdict"], "| failed:", m["verdict_detail"]["failed_criteria"])
    print("lambda*:", cal.get("selected_lambda"), "| runtime: %.0fs" % m["runtime_sec"])
    print("=" * 78)

    for k in ("P5a", "P5b", "P5c", "P5d", "P2_prime", "P3_1"):
        c = crit[k]
        print("[%s] passed=%s" % (k, c.get("passed")))
        for kk in ("n_blocks", "n_obs", "n_positive", "median_delta", "mean_delta",
                   "permutation_p", "n_perm", "median_same_layer", "median_cross_layer",
                   "n_same_layer", "n_cross_layer", "n_rows", "n_violations",
                   "min_rel_perturbation", "max_rel_perturbation"):
            if kk in c:
                print("     %-24s %s" % (kk, c[kk]))
        if "content_specific_vs_random" in c:
            print("     vs_random:", c["content_specific_vs_random"])
        if "rule" in c:
            print("     rule:", c["rule"])

    print("\n--- 分方向 (P5c: arm1-arm0 的 r2_q 增量, 按块中位) ---")
    for d in ("Qwen->BERT", "BERT->Qwen"):
        blk = collections.defaultdict(dict)
        for r in rows:
            if r["direction"] == d:
                blk[(r["source_layer"], r["target_layer"])][(r["arm"], r["seed"])] = r
        deltas = []
        for key, arms in sorted(blk.items()):
            ds = []
            for seed in sorted(set(s for (a, s) in arms)):
                a1 = arms.get(("arm1_slot_Q", seed))
                a0 = arms.get(("arm0_pipe_zero", seed))
                if a1 and a0:
                    ds.append(a1["r2_q"] - a0["r2_q"])
            if ds:
                deltas.append((key, statistics.median(ds)))
        if not deltas:
            continue
        pos = sum(1 for _, v in deltas if v > 0)
        vals = sorted(v for _, v in deltas)
        print("  %-12s 块数=%d 正号=%d 中位=%+.6f 均值=%+.6f  min=%+.6f max=%+.6f"
              % (d, len(deltas), pos, statistics.median(vals),
                 sum(vals) / len(vals), vals[0], vals[-1]))

    print("\n--- 分方向 (P5d: 非目标槽 r2_o 增量) ---")
    for d in ("Qwen->BERT", "BERT->Qwen"):
        blk = collections.defaultdict(dict)
        for r in rows:
            if r["direction"] == d:
                blk[(r["source_layer"], r["target_layer"])][(r["arm"], r["seed"])] = r
        deltas = []
        for key, arms in sorted(blk.items()):
            ds = []
            for seed in sorted(set(s for (a, s) in arms)):
                a1 = arms.get(("arm1_slot_Q", seed))
                a0 = arms.get(("arm0_pipe_zero", seed))
                if a1 and a0:
                    ds.append(a1["r2_o"] - a0["r2_o"])
            if ds:
                deltas.append((key, statistics.median(ds)))
        if not deltas:
            continue
        pos = sum(1 for _, v in deltas if v > 0)
        vals = sorted(v for _, v in deltas)
        print("  %-12s 块数=%d 正号=%d 中位=%+.6f 均值=%+.6f"
              % (d, len(deltas), pos, statistics.median(vals), sum(vals) / len(vals)))

    print("\n--- 深度剖面: 按读出层 j 的 P5c 增量中位 ---")
    a0 = {}
    for r in rows:
        if r["arm"] == "arm0_pipe_zero":
            a0[(r["source_layer"], r["target_layer"], r["direction"], r["seed"])] = r
    agg = collections.defaultdict(list)
    for r in rows:
        if r["arm"] != "arm1_slot_Q":
            continue
        b = a0.get((r["source_layer"], r["target_layer"], r["direction"], r["seed"]))
        if b:
            agg[(r["direction"], r["target_layer"])].append(r["r2_q"] - b["r2_q"])
    for key in sorted(agg):
        v = sorted(agg[key])
        print("  %-12s j=%-3d n=%-4d 中位=%+.6f 均值=%+.6f 正号=%d/%d"
              % (key[0], key[1], len(v), statistics.median(v), sum(v) / len(v),
                 sum(1 for x in v if x > 0), len(v)))

    print("\n--- 注意力 (第 0 层, 逐样本中位) ---")
    att = m["attention_summary"]
    by = collections.defaultdict(dict)
    for r in att:
        by[(r["direction"], r["source_layer"])][(r["arm"], r["seed"])] = r
    for key in sorted(by):
        arms = by[key]
        a1 = [v for (a, s), v in arms.items() if a == "arm1_slot_Q"]
        a0v = [v for (a, s), v in arms.items() if a == "arm0_pipe_zero"]
        a2 = [v for (a, s), v in arms.items() if a == "arm2_wrong_slot_O"]
        a3 = [v for (a, s), v in arms.items() if a == "arm3_random_dir"]
        if not (a1 and a0v and a2 and a3):
            continue
        f1 = statistics.median(v["flip_med"] for v in a1)
        f2 = statistics.median(v["flip_med"] for v in a2)
        p1 = statistics.median(v["attn_pipe_med"] for v in a1)
        p0 = statistics.median(v["attn_pipe_med"] for v in a0v)
        p3 = statistics.median(v["attn_pipe_med"] for v in a3)
        print("  %-12s ly=%-3d attn_pipe: arm1=%.4e arm0=%.4e arm3=%.4e (Δ1-0=%+.3e Δ1-3=%+.3e) | flip: arm1=%+.3e arm2=%+.3e (Δ=%+.3e)"
              % (key[0], key[1], p1, p0, p3, p1 - p0, p1 - p3, f1, f2, f1 - f2))

    print("\n--- ViT 臂 (跑前整臂撤出, 只留撤出记录) ---")
    print(json.dumps(crit.get("vit_parallel", {}), ensure_ascii=False, indent=2))
    print("\n--- 辅助读数 (样本级 mean-pool, 不参与裁决) ---")
    print(json.dumps(crit["aux_pooled"], ensure_ascii=False, indent=2))
    print("\n--- 探针 lambda ---")
    for d, v in m["protocol"]["probe_lambda_by_direction"].items():
        print("  %-12s probe_lambda=%g cv_r2=%.6f cv_scores=%s"
              % (d, v["probe_lambda"], v["cv_r2"],
                 {k: round(x, 4) for k, x in v["cv_scores"].items()}))
    print("\n--- 守门: 注入臂扰动分方向 ---")
    for d in ("Qwen->BERT", "BERT->Qwen"):
        v = [r["rel_perturb_final"] for r in rows if r["direction"] == d and r["arm"] != "arm0_pipe_zero"]
        if v:
            print("  %-12s min=%.4f max=%.4f mean=%.4f" % (d, min(v), max(v), sum(v) / len(v)))
    print("\n--- 布局 ---")
    print(json.dumps(m["protocol"]["layout_by_direction"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "metrics.json")
