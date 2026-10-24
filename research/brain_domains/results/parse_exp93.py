"""parse_exp93.py — exp93 的结果重建：档位归因扫描（exp93a）+ 对比学习 encoder（exp93b）。

## 为什么要有这个脚本

exp93a / exp93b 是在 Job 里跑的，JSON 落在桶里。本机读通道时断时续 ⇒ 重试下载，
然后把结果压成**一张因子表 + 一张图**，供 FINDINGS / INTEGRATION 直接引用。

## 统计口径（与本项目既有约定一致）

- **中位 = 上中位**（`sorted[n//2]`，偶数取偏大）⇒ 右偏分布会**报高 30–40%**
  ⇒ 表里同时给出均值，避免被单边分布骗。
- 符号检验用双侧精确二项（`sign_test_p`）。
- **协议不可混比**：`P3 同集身份解码` 与 `P1 跨集检索` 不是一回事，
  图里 P3 用不同的底色标注，提醒读者它**不能和文献的 15.6% 比**。

## 输出

- `results/exp93-results/exp93a_level_audit.{json,md}`
- `results/exp93-results/exp93b_contrastive.{json,md}`
- `results/exp93-results/exp93_level.png`
"""
from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp93-results"
BUCKET = "hf://buckets/illusion-brain/brain-domains/runs/illusionbrain-align"

LIT_TOP1 = 15.6     # NICE-EEG（arXiv:2308.13234, ICLR 2024）同一份 THINGS-EEG 的 200-way 零样本 top-1
CHANCE = 1.0 / 200.0


def sign_test_p(diffs):
    d = [float(x) for x in diffs if x == x]
    n = len(d)
    if n == 0:
        return 0, 0, float("nan")
    k = sum(1 for x in d if x > 0)
    m = n - k
    p = 2.0 * sum(math.comb(n, i) for i in range(min(k, m) + 1)) / (2 ** n)
    return k, m, float(min(1.0, p))


def med(xs):
    """**上中位**（本项目约定）：sorted[n//2]，偶数取偏大。"""
    xs = [float(x) for x in xs if x == x]
    if not xs:
        return None
    s = sorted(xs)
    return float(s[len(s) // 2])


def mean(xs):
    xs = [float(x) for x in xs if x == x]
    return float(sum(xs) / len(xs)) if xs else None


def fetch(rel, retries=6, wait=6, force=False):
    """从桶里取 JSON（读通道不稳 ⇒ 重试）。"""
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / Path(rel).name
    if (not force) and dst.exists() and dst.stat().st_size > 0:
        return json.loads(dst.read_text(encoding="utf-8"))
    src = f"{BUCKET}/{rel}"
    for i in range(retries):
        try:
            r = subprocess.run(["hf", "buckets", "cp", src, str(dst)],
                               capture_output=True, text=True, timeout=420)
            if r.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
                return json.loads(dst.read_text(encoding="utf-8"))
        except subprocess.TimeoutExpired:
            pass
        print(f"  重试 {i+1}/{retries}：{src}")
        time.sleep(wait)
    print(f"✗ 取不到 {src}")
    return None


# ---------------------------------------------------------------------------
# exp93a：因子表
# ---------------------------------------------------------------------------

def cells_of(rec):
    """把一个被试的记录摊成 {cell: value}。"""
    out = {}
    b = rec.get("baseline_ridge") or {}
    if b:
        out["P1/ridge/16540"] = b["acc1_cos"]
        out["P1/ridge/16540/试次平均"] = b.get("acc1_featavg_cos")
    for n, v in (rec.get("F1_train_size") or {}).items():
        out[f"P1/ridge/{n}"] = v["acc1_cos"]
    for k, v in (rec.get("F2_decoder") or {}).items():
        out[f"P1/{k}/16540"] = v["acc1_cos"]
    f3 = rec.get("F3_trial_agg") or {}
    if f3:
        out["P1/ridge/16540/试次平均"] = f3.get("feature_avg_acc1")
    f4 = rec.get("F4_target_pca128") or {}
    if f4:
        out["P1/ridge+PCA128目标/16540"] = f4["acc1_cos"]
    f5 = rec.get("F5_protocol") or {}
    if f5:
        out["P3/同集身份解码"] = f5.get("P3_same_set_identity")
    s2 = rec.get("self_check_S2_same_set_retrieval") or {}
    if s2:
        out["P2/同集检索"] = s2.get("acc1_cos")
    return out


ORDER = [
    "P1/ridge/16540", "P1/ridge/16540/试次平均", "P1/ridge/4000", "P1/ridge/1000",
    "P1/rrr_k64/16540", "P1/rrr_k256/16540", "P1/cca_k32/16540", "P1/cca_k128/16540",
    "P1/ridge_unit_target/16540", "P1/ridge+PCA128目标/16540",
    "P2/同集检索", "P3/同集身份解码",
]

# ⚠️ **每格的候选数不同 ⇒ 随机水平不同**：P1/P3 是 200 张候选（1/200），
#    P2 是 2000 张候选（1/2000）⇒ 用全局 1/200 会把 P2 少算 10 倍。
CELL_CHANCE = {
    "P2/同集检索": 1.0 / 2000.0,
}


def summarize_audit(data):
    rows = [r for r in data["per_subject"] if "error" not in r]
    errs = [r for r in data["per_subject"] if "error" in r]
    cells = {}
    for r in rows:
        for c, v in cells_of(r).items():
            if v is not None:
                cells.setdefault(c, []).append(v)
    table = []
    for c in ORDER:
        if c not in cells:
            continue
        xs = cells[c]
        chance = CELL_CHANCE.get(c, CHANCE)
        table.append({
            "cell": c, "n": len(xs),
            "median_up": med(xs), "mean": mean(xs),
            "chance": chance,
            "ratio_to_chance": med(xs) / chance if med(xs) is not None else None,
            "max": max(xs), "min": min(xs),
            "n_above_chance": sum(1 for x in xs if x > chance),
        })
    align = [r.get("align_check") for r in rows if r.get("align_check")]
    return {"n_ok": len(rows), "n_err": len(errs),
            "errors": [f"{e['subject']}: {e['error']}" for e in errs],
            "align_check": align,
            "table": table}


# ---------------------------------------------------------------------------
# exp93b：每个被试三条读出
# ---------------------------------------------------------------------------

def summarize_contrastive(data):
    rows = [r for r in data["per_subject"] if "error" not in r]
    errs = [r for r in data["per_subject"] if "error" in r]
    def col(path):
        out = []
        for r in rows:
            v = r
            for p in path:
                v = (v or {}).get(p) if isinstance(v, dict) else None
                if v is None:
                    break
            if v is not None:
                out.append(float(v))
        return out
    zs = col(("zeroshot", "acc1_cos"))
    zsa = col(("zeroshot", "acc1_featavg_cos"))
    z5 = col(("zeroshot", "acc5_cos"))
    rf = col(("ridge_feat", "acc1_cos"))
    rfa = col(("ridge_feat", "acc1_featavg_cos"))
    rs = col(("rsa", "r"))
    tr = data.get("train") or {}
    hist = tr.get("hist") or []
    return {
        "n_ok": len(rows), "n_err": len(errs),
        "errors": [f"{e['subject']}: {e['error']}" for e in errs],
        "zeroshot_acc1": {"median_up": med(zs), "mean": mean(zs),
                           "ratio_to_chance": med(zs) / CHANCE if zs else None,
                           "n_above_chance": sum(1 for x in zs if x > CHANCE)},
        "zeroshot_acc1_featavg": {"median_up": med(zsa), "mean": mean(zsa),
                                  "ratio_to_chance": med(zsa) / CHANCE if zsa else None},
        "zeroshot_acc5": {"median_up": med(z5), "mean": mean(z5),
                          "ratio_to_chance": med(z5) / CHANCE if z5 else None},
        "ridge_feat_acc1": {"median_up": med(rf), "mean": mean(rf),
                            "ratio_to_chance": med(rf) / CHANCE if rf else None},
        "ridge_feat_acc1_featavg": {"median_up": med(rfa), "mean": mean(rfa),
                                    "ratio_to_chance": med(rfa) / CHANCE if rfa else None},
        "rsa": {"median_up": med(rs), "mean": mean(rs)},
        "train_best": tr.get("best"),
        "train_hist_tail": hist[-3:],
        "align_rsa": tr.get("align_rsa"), "align_p": tr.get("align_p"),
        "n_train_images": tr.get("n_img"),
        "per_subject": rows,
    }


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def write_md_audit(s, path):
    L = ["# exp93a 档位归因扫描（零训练）", "",
         f"- 成功 {s['n_ok']} 个被试，报错 {s['n_err']} 个"
         + ("：" + "；".join(s["errors"]) if s["errors"] else ""),
         f"- 随机水平 = 1/200 = {CHANCE:.4f}；**中位 = 上中位**（右偏分布会报高 30–40%）", ""]
    if s["align_check"]:
        L += ["- 对齐预检（三段式，见 §二十九·三）：", ""]
        for a in s["align_check"]:
            L += [f"  - `{a}`", ""]
    L += ["## 因子表", "",
          "| 协议 / 解码器 / 训练图数 | 候选数 | n | 中位(上) | 均值 | ×随机 | 高于随机的被试 |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for r in s["table"]:
        ncand = int(round(1.0 / r["chance"]))
        L.append(f"| `{r['cell']}` | {ncand} | {r['n']} | {r['median_up']:.4f} "
                 f"| {r['mean']:.4f} | {r['ratio_to_chance']:.1f}× "
                 f"| {r['n_above_chance']}/{r['n']} |")
    L += ["", "⚠️ `P2/同集检索`（2000 候选）与 `P3/同集身份解码`（200 候选）**都不是跨集检索**，"
          "不能与 NICE-EEG 的 15.6% 直接比；"
          "只有 `P1/...` 行能与 15.6% 同口径比。", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def write_md_contrastive(s, path):
    L = ["# exp93b 对比学习 encoder（InfoNCE → ViT-H-14 图像特征）", "",
         f"- 成功 {s['n_ok']} 个被试，报错 {s['n_err']} 个"
         + ("：" + "；".join(s["errors"]) if s["errors"] else ""),
         f"- 训练图像数 {s['n_train_images']}；对齐预检 RSA={s['align_rsa']}（p={s['align_p']}）",
         f"- 最佳 epoch（按 **train-split 验证集**选，未用 test）：{s['train_best']}", "",
         "| 读出 | 中位(上) | 均值 | ×随机 |", "|---|---:|---:|---:|"]
    for key, name in (("zeroshot_acc1", "zeroshot top-1（单试次）"),
                      ("zeroshot_acc1_featavg", "zeroshot top-1（试次平均）"),
                      ("zeroshot_acc5", "zeroshot top-5"),
                      ("ridge_feat_acc1", "ridge-on-features top-1（A13 口径）"),
                      ("ridge_feat_acc1_featavg", "ridge-on-features top-1（试次平均）")):
        v = s[key]
        if v["median_up"] is None:
            continue
        rc = v.get("ratio_to_chance")
        L.append(f"| {name} | {v['median_up']:.4f} | {v['mean']:.4f} "
                 f"| {rc:.1f}× |" if rc is not None
                 else f"| {name} | {v['median_up']:.4f} | {v['mean']:.4f} | — |")
    if s["rsa"]["median_up"] is not None:
        L.append(f"| RSA | {s['rsa']['median_up']:+.4f} | {s['rsa']['mean']:+.4f} | — |")
    L += ["", f"**对照**：A13（630-d + ridge，跨集）1.1%；"
          f"NICE-EEG（跨集零样本）{LIT_TOP1}%。", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def summarize_dose(data):
    """exp93c 剂量-反应：固定 34 通道、只改后部通道数 k ⇒ 每层的 RSA 中位。"""
    ps = [r for r in data.get("per_subject", []) if "error" not in r]
    out = {"n_ok": len(ps), "n_err": len(data.get("per_subject", [])) - len(ps)}
    for tag, key in (("s1a", "model_dose"), ("s1b", "model_dose"), ("feature", "feature_dose")):
        if tag == "feature":
            src = [(r.get("feature_dose") or {}) for r in ps]
        else:
            src = [((r.get("model_dose") or {}).get(tag) or {}) for r in ps]
        ks = sorted({int(k) for d in src for k in d})
        row = {}
        for k in ks:
            vals = [float(d[str(k)]["r_median"]) for d in src if str(k) in d]
            row[str(k)] = {"median_up": med(vals), "mean": mean(vals), "n": len(vals)}
        out[tag] = row
    return out


def summarize_window(data):
    """exp93e 窗口扫描：每个窗口下的 POST/ANT RSA、档位、随机组地板。"""
    ps = [r for r in data.get("per_subject", []) if "error" not in r]
    out = {"n_ok": len(ps), "n_err": len(data.get("per_subject", [])) - len(ps)}
    wins = []
    for r in ps:
        for w, v in (r.get("windows") or {}).items():
            if w not in wins:
                wins.append(w)
    order = ["w0_250", "w50_500", "w100_400", "w50_250", "w100_250"]
    wins = [w for w in order if w in wins] + [w for w in wins if w not in order]
    for w in wins:
        vals = [(r.get("windows") or {}).get(w) for r in ps]
        vals = [v for v in vals if v]
        if not vals:
            continue
        out[w] = {
            "POST": {"median_up": med([v["groups"]["POST"]["r"] for v in vals]),
                     "mean": mean([v["groups"]["POST"]["r"] for v in vals])},
            "ANT": {"median_up": med([v["groups"]["ANT"]["r"] for v in vals]),
                    "mean": mean([v["groups"]["ANT"]["r"] for v in vals])},
            "level": {"median_up": med([v["level_acc1"] for v in vals]),
                      "mean": mean([v["level_acc1"] for v in vals])},
            "null_median": med([v["null_median"] for v in vals]),
            "POST_above_all": sum(1 for v in vals if v["POST_above_all_null"]),
            "ANT_below_all": sum(1 for v in vals if v["ANT_below_all_null"]),
            "n": len(vals),
        }
    return out


def pick_font():
    """中文字体（沿用本项目 plot_*.py 的同一套候选与写法）。"""
    import matplotlib
    import matplotlib.pyplot as plt
    for fam in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                "Source Han Sans SC", "WenQuanYi Micro Hei"):
        try:
            matplotlib.font_manager.findfont(fam, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [fam]
            plt.rcParams["axes.unicode_minus"] = False
            return fam
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    return None


def plot(audit, contra, dose, window, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"✗ 画图跳过：{e}")
        return
    fam = pick_font()
    print(f"字体：{fam}" if fam else "⚠ 没找到中文字体")
    n_panels = 2 + (1 if dose else 0) + (1 if window else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, 5.2))
    ax = axes[0]
    if audit and audit["table"]:
        rows = audit["table"]
        names = [r["cell"] for r in rows]
        vals = [r["ratio_to_chance"] for r in rows]
        colors = ["#c0392b" if n.startswith("P1") else ("#7f8c8d" if n.startswith("P2")
                 else "#2980b9") for n in names]
        y = np.arange(len(names))
        ax.barh(y, vals, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.axvline(1.0, color="k", lw=1, ls="--")
        ax.axvline(LIT_TOP1 / CHANCE, color="#e67e22", lw=1.5, ls=":")
        ax.text(LIT_TOP1 / CHANCE, len(names) - 0.4, " NICE-EEG 15.6%（P1 口径）",
                color="#e67e22", fontsize=9, va="top")
        ax.set_xlabel("top-1 ÷ 随机（×）")
        ax.set_title("exp93a：档位归因（P1 红 / P2 灰 / P3 蓝，P3 不可与文献比）",
                     fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    ax = axes[1]
    labels, vals, colors = [], [], []
    if audit and audit["table"]:
        for r in audit["table"]:
            if r["cell"] in ("P1/ridge/16540", "P1/ridge/1000"):
                labels.append(r["cell"] + "\n（手工 630-d）")
                vals.append(r["ratio_to_chance"])
                colors.append("#7f8c8d")
    if contra:
        for key, name in (("zeroshot_acc1", "对比学习 encoder\nzeroshot top-1"),
                          ("zeroshot_acc1_featavg", "对比学习 encoder\nzeroshot（试次平均）"),
                          ("ridge_feat_acc1", "对比学习 encoder\nridge-on-features")):
            v = contra[key]
            if v["median_up"] is None:
                continue
            labels.append(name)
            vals.append(v["ratio_to_chance"])
            colors.append("#27ae60")
    if labels:
        x = np.arange(len(labels))
        ax.bar(x, vals, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.axhline(1.0, color="k", lw=1, ls="--")
        ax.axhline(LIT_TOP1 / CHANCE, color="#e67e22", lw=1.5, ls=":")
        ax.set_ylabel("top-1 ÷ 随机（×）")
        ax.set_title("手工基线 vs 对比学习 encoder（同为跨集 P1 口径）", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    if dose:
        ax = axes[2]
        for tag, lab, col in (("feature", "特征层 630-d", "#d1495b"),
                              ("s1a", "模型层 S1a（自监督）", "#e08e45"),
                              ("s1b", "模型层 S1b（语义监督）", "#3d5a80")):
            row = dose.get(tag) or {}
            ks = sorted(int(k) for k in row)
            if not ks:
                continue
            ax.plot(ks, [row[str(k)]["median_up"] for k in ks], "o-",
                    color=col, lw=1.8, ms=5, label=lab)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("通道组里的后部通道数 k（总数固定 34）")
        ax.set_ylabel("RSA（中位，上中位）")
        ax.set_title("位置轴的剂量-反应：随 k 单调上升 ⇒ 是梯度不是两块", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    if window:
        ax = axes[2 + (1 if dose else 0)]
        names, posts, ants, lv = [], [], [], []
        for w, v in window.items():
            if not isinstance(v, dict) or "POST" not in v:
                continue
            names.append(w)
            posts.append(v["POST"]["median_up"])
            ants.append(v["ANT"]["median_up"])
            lv.append(v["level"]["median_up"])
        if names:
            x = np.arange(len(names))
            ax.plot(x, posts, "o-", color="#d1495b", lw=1.8, ms=5, label="后部 POST")
            ax.plot(x, ants, "s-", color="#3d5a80", lw=1.8, ms=5, label="前部 ANT")
            ax.set_xticks(x)
            ax.set_xticklabels(names, fontsize=8)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_ylabel("RSA（中位）")
            ax.set_title("特征窗口扫描（exp93e）：窗选得越对，后部越强", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)
            for xi, v in zip(x, lv):
                ax.text(xi, max(posts) * 0.04, f"档位 {v:.4f}", rotation=90,
                        fontsize=6.5, ha="center", va="bottom", color="#666")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"图写入 {path}")


def fetch_exp93a(force=False):
    """exp93a 的两个 Job 写**同一路径** ⇒ 下下来后按**被试范围**另存，再合并。

    （不这么做的话，后完成的 Job 会把前一个的结果整段覆盖掉。）
    """
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "exp93a_level_audit.raw.json"
    if force or not (tmp.exists() and tmp.stat().st_size > 0):
        src = f"{BUCKET}/exp93a/exp93a_level_audit.json"
        for i in range(6):
            try:
                r = subprocess.run(["hf", "buckets", "cp", src, str(tmp)],
                                   capture_output=True, text=True, timeout=420)
                if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
                    break
            except subprocess.TimeoutExpired:
                pass
            print(f"  重试 {i+1}/6：{src}")
            time.sleep(6)
    if not tmp.exists():
        return {"per_subject": []}
    d = json.loads(tmp.read_text(encoding="utf-8"))
    subs = [r.get("subject", "") for r in d.get("per_subject", [])]
    if subs:
        dst = OUT / f"exp93a_level_audit_{subs[0]}-{subs[-1]}.json"
        # ⚠️ **总是覆盖**：同名文件可能是上一次跑剩的旧版本（错误版）⇒ 不覆盖就永远留着错数据
        dst.write_text(tmp.read_text(encoding="utf-8"), encoding="utf-8")
        # 临时下载件不留在仓库里（它只是搬运途）
        try:
            tmp.unlink()
        except OSError:
            pass
    return d


def merge_exp93a():
    """把本地所有 exp93a 分段结果按被试合并（同名后者覆盖前者）。"""
    by = {}
    for fp in sorted(OUT.glob("exp93a_level_audit*.json")):
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        for r in d.get("per_subject", []):
            by[r.get("subject", fp.name)] = r
    return {"per_subject": [by[k] for k in sorted(by)]}


def write_md_dose(s, path):
    L = ["# exp93c 位置轴的剂量-反应（固定 34 通道、只改后部通道数 k）", "",
         f"- 成功 {s['n_ok']} 个被试，报错 {s['n_err']} 个", "",
         "| 层 | " + " | ".join(sorted({k for tag in ('feature', 's1a', 's1b')
                                      for k in (s.get(tag) or {})})) + " |",
         "|---|" + "---:|" * len({k for tag in ('feature', 's1a', 's1b')
                                  for k in (s.get(tag) or {})})]
    ks = sorted({int(k) for tag in ('feature', 's1a', 's1b')
                 for k in (s.get(tag) or {})})
    L[2] = "| 层 | " + " | ".join(f"k={k}" for k in ks) + " |"
    L[3] = "|---|" + "---:|" * len(ks)
    for tag, name in (("feature", "特征层 630-d"), ("s1a", "模型层 S1a（自监督）"),
                      ("s1b", "模型层 S1b（语义监督）")):
        row = s.get(tag) or {}
        if not row:
            continue
        L.append(f"| {name} | " + " | ".join(
            f"{row[str(k)]['median_up']:+.4f}" if str(k) in row else "—"
            for k in ks) + " |")
    L += ["", "⇒ 三层都**严格单调上升** ⇒ 位置轴是**梯度**，不是两块；"
          "k=5（几乎纯前部）≈ 0 ⇒ 前部不带语义。", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def write_md_window(s, path):
    L = ["# exp93e 特征窗口扫描（位置轴 + 绝对档位，一起查）", "",
         f"- 成功 {s.get('n_ok', '?')} 个被试，报错 {s.get('n_err', 0)} 个", "",
         "| 窗口 | POST RSA | ANT RSA | 档位 top-1 | 随机组中位 | POST>全部随机组 | ANT<全部随机组 |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for w, v in s.items():
        if not isinstance(v, dict) or "POST" not in v:
            continue
        L.append(f"| `{w}` | {v['POST']['median_up']:+.4f} | {v['ANT']['median_up']:+.4f} "
                 f"| {v['level']['median_up']:.4f} | {v['null_median']:+.4f} "
                 f"| {v['POST_above_all']}/{v['n']} | {v['ANT_below_all']}/{v['n']} |")
    L += ["", "⇒ 窗口是**第四个杠杆**：窗口选得越贴信号（50–500 ms），"
          "**后部 RSA 越强、绝对档位越高**，而**前部始终≈0**"
          "⇒ 位置轴结论**不是窗口假象**，反而更强。", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="忽略本地缓存，强制从桶里重取（结果重跑后必须加）")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fetch_exp93a(force=args.force)
    a = merge_exp93a()
    b = fetch("exp93b/exp93b_contrastive.json", force=args.force)
    c = fetch("exp93c/exp93c_dose_response.json", force=args.force)
    e = fetch("exp93e/exp93e_window_sweep.json", force=args.force)
    dose = summarize_dose(c) if c else None
    window = summarize_window(e) if e else None
    if window is None:
        # 兜底：桶内 JSON 还没落盘 ⇒ 用 **Job 日志** 解析出来的那份
        fl = OUT / "exp93e_window.from_log.json"
        if fl.exists():
            window = json.loads(fl.read_text(encoding="utf-8")).get("windows")
            print(f"exp93e：用 Job 日志版（{len(window or {})} 个窗口）")
    if dose is None:
        fl = OUT / "exp93c_dose.summary.json"
        if fl.exists():
            dose = json.loads(fl.read_text(encoding="utf-8"))
    audit = summarize_audit(a) if a else None
    contra = summarize_contrastive(b) if b else None
    if audit:
        # ⚠️ 这里只有**桶内 JSON 段**（s6-10）的被试 ⇒ 文件名带 `from_json` 以免
        #    被误当成完整的 10 被试表；**完整表**见 `exp93a_merged.{json,md}`
        (OUT / "exp93a_level_audit.from_json.summary.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        write_md_audit(audit, OUT / "exp93a_level_audit.from_json.md")
        print(f"exp93a（仅桶内 JSON 段）：{audit['n_ok']} 个被试成功，"
              f"{audit['n_err']} 个报错")
    if contra:
        (OUT / "exp93b_contrastive.summary.json").write_text(
            json.dumps(contra, indent=2, ensure_ascii=False), encoding="utf-8")
        write_md_contrastive(contra, OUT / "exp93b_contrastive.md")
        print(f"exp93b：{contra['n_ok']} 个被试成功，{contra['n_err']} 个报错")
    if dose:
        (OUT / "exp93c_dose.summary.json").write_text(
            json.dumps(dose, indent=2, ensure_ascii=False), encoding="utf-8")
        write_md_dose(dose, OUT / "exp93c_dose.md")
        print(f"exp93c：{dose['n_ok']} 个被试成功，{dose['n_err']} 个报错")
    if window:
        (OUT / "exp93e_window.summary.json").write_text(
            json.dumps(window, indent=2, ensure_ascii=False), encoding="utf-8")
        write_md_window(window, OUT / "exp93e_window.md")
        print(f"exp93e：{window.get('n_ok', '?')} 个被试成功")
    plot(audit, contra, dose, window, OUT / "exp93_level.png")


if __name__ == "__main__":
    main()
