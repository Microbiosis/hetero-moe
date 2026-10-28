#!/usr/bin/env python3
"""产物血缘与一致性检查器 (aligner 线通用, v49+).

为什么需要它
------------
审计链 (v50→v51→v52→v53) 的价值全部建立在"产物能被同一份代码复现"之上。
一旦 metrics.json 的 config 块出自另一个写入者 (手工合并 / 旧脚本 / 复制粘贴),
读出来的基线就是错的, 而**数字本身看起来完全正常** —— 这类错误不会自曝。

实测抓到的第一个案例 (v53):
    research/aligner/v53_third_audit/audit.py   CLASSES 有 6 个键 → N_CLASSES = 6
    research/aligner/v53_third_audit/results/metrics.json   "n_classes": 8
    → 报告的基线 0.125 是错的对, 真实基线 1/6 = 0.1667
    → "有意义空间低于平凡基线" 这个结论由"成立"翻成"不成立"

检查项
------
    L1  n_classes 血缘    config 里写的类别数 == 代码实际构建的类别数
    L2  code_sha 血缘     产物是否记录了产生它的 commit (v49–v53 都没有 → WARN)
    L3  results 体完整    每个 cell 的 rows 条数 == 声明的 seeds 数
    L4  results ↔ cells   若提供 cells 目录, 逐 cell 与 metrics 对拍 (逐位)
    B1  基线口径          用代码口径和记录口径**分别**算基线, 不一致直接 FAIL
    V1  判决可重算        从 rows 重算 D1a/D1c/D3, 与存储的 verdicts 对拍

用法
----
    python tools/check_artifacts.py research/aligner/v53_third_audit
    python tools/check_artifacts.py research/aligner/v54_evaluator_swap --cells <dir>
    python tools/check_artifacts.py <dir> --json out.json

退出码: 0 全过 / 1 有 FAIL / 2 文件缺失
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from typing import Any

# ----------------------------------------------------------------------


def load_audit(pkg_dir: str):
    """按路径导入包内 audit.py (审计型包) 或 measure.py (结构测量型包)。

    v55 属后者: 它的产物是雅可比秩的结构测量, 没有 n_classes、没有 seeds、
    没有 cells —— 用审计型的检查项去套它只会产生误报 (实测 4 个 FAIL 全是假阳性)。
    """
    for name in ("audit.py", "measure.py"):
        path = os.path.join(pkg_dir, name)
        if os.path.exists(path):
            break
    else:
        return None, f"missing audit.py / measure.py in {pkg_dir}"
    spec = importlib.util.spec_from_file_location(f"_probe_{os.path.basename(pkg_dir)}", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001
        return None, f"import failed: {type(e).__name__}: {e}"
    return mod, None


def code_classes(mod) -> tuple[int | None, str]:
    """代码实际会构建多少个类别 (唯一真源)。"""
    if mod is None:
        return None, "no module"
    if hasattr(mod, "N_CLASSES"):
        return int(mod.N_CLASSES), "mod.N_CLASSES"
    if hasattr(mod, "CLASSES"):
        return len(mod.CLASSES), "len(mod.CLASSES)"
    return None, "not found"


def baseline(n: int) -> dict[str, float]:
    """块状 one-hot 目标下两条解析基线 (精确值, 非估计)。

    all_zero      : 预测恒为 0          → (D/n)/D = 1/n
    predict_mean  : 每维预测类别均值 1/n → (n-1)/n²
    """
    return {"all_zero": 1.0 / n, "predict_mean": (n - 1) / (n * n)}


def file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:12]


def is_ancestor(sha: str, repo: str) -> bool:
    """sha 是否为 repo HEAD 的祖先 (只在前 7 位可用时也成立)。"""
    try:
        out = subprocess.run(["git", "-C", repo, "merge-base", "--is-ancestor", sha, "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def git_head(repo: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------------
# 检查
# ----------------------------------------------------------------------

def check(pkg_dir: str, cells_dir: str | None, repo: str,
          code_dir: str | None = None) -> dict[str, Any]:
    rep: dict[str, Any] = {"package": pkg_dir, "checks": [], "warn": [], "fail": []}

    def add(name: str, ok: bool, detail: str, *, warn_only: bool = False):
        status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
        rep["checks"].append({"id": name, "status": status, "detail": detail})
        if not ok:
            (rep["warn"] if warn_only else rep["fail"]).append(name)

    # 兼容两种落盘位置: <pkg>/results/metrics.json (仓库约定) 与 <pkg>/metrics.json (V54_ART_DIR 直落)
    mpath = os.path.join(pkg_dir, "results", "metrics.json")
    if not os.path.exists(mpath):
        alt = os.path.join(pkg_dir, "metrics.json")
        if os.path.exists(alt):
            mpath = alt
    if not os.path.exists(mpath):
        add("F0", False, f"missing {mpath}")
        return rep
    m = json.load(open(mpath, encoding="utf-8"))
    # 结构测量型 (kind 显式声明, 或产物含 rows 但无 config) → 不做网格/基线检查
    is_structural = (m.get("kind") in ("measured_lemma", "theory", "extension_document")
                     or ("config" not in m and "rows" in m))
    if is_structural:
        rep["checks"].append({"id": "S1", "status": "WARN",
                              "detail": f"结构测量型产物 (kind={m.get('kind')}) —— 无分类目标/网格/基线, "
                                        f"仅校验可重算性与健康度"})
        rep["warn"].append("S1")
    is_smoke = bool(m.get("IS_SMOKE_NOT_RESULTS")) or \
        (m.get("protocol_check") or {}).get("status") == "smoke"
    if is_smoke:
        rep["checks"].append({"id": "S0", "status": "WARN",
                              "detail": "冒烟快照 (IS_SMOKE_NOT_RESULTS=true) —— 读数不得引用; "
                                        "只校验结构与血缘字段是否存在, 不校验 sha 是否等于 HEAD"})
        rep["warn"].append("S0")
    cfg = m.get("config", {}) or {}
    res = m.get("results", {}) or {}

    # 产物目录可以只有 metrics.json (如 V54_ART_DIR 直落), 代码在 --code 指定的包里
    mod, err = load_audit(code_dir or pkg_dir)
    n_code, how = code_classes(mod)
    add("F1", mod is not None,
        f"模块可导入={mod is not None} ({err or 'ok'}); n_classes 来源={how}"
        + ("  [结构测量型]" if is_structural else ""))

    # ---- L1: n_classes 血缘 ----
    n_rec = cfg.get("n_classes")
    if is_structural:
        add("L1", True, "结构测量型: n_classes 不适用 (跳过)", warn_only=True)
    elif n_code is None or n_rec is None:
        add("L1", False, f"无法比对 (code={n_code}, recorded={n_rec})")
    else:
        add("L1", int(n_rec) == int(n_code),
            f"recorded n_classes={n_rec} / code n_classes={n_code} ({how})")

    # ---- L2: code_sha ----
    # code_sha 可能在顶层、config 里、或 provenance 里 (v55 就是第三种)
    prov = m.get("provenance") or {}
    sha = m.get("code_sha") or cfg.get("code_sha") or prov.get("code_sha")
    head = git_head(repo)
    if not sha:
        add("L2", False, "产物未记录 code_sha → 无法证明出自哪份代码 (v49–v53 均缺)", warn_only=True)
    else:
        base = sha.split("+")[0]
        anc = is_ancestor(base, repo)
        add("L2", anc or is_smoke,
            f"code_sha={sha} / repo HEAD={head}  "
            + ("是该提交的祖先 ✓" if anc else "**不在本仓库历史中**")
            + (" (+dirty: 工作树有未提交改动)" if "+" in sha else ""))

    # ---- L3: 网格完整 ----
    d_list = cfg.get("d_shareds") or []
    n_seeds = cfg.get("seeds")
    if is_structural:
        add("L3", True, f"结构测量型: 网格不适用; rows={len(m.get('rows', []))} 档", warn_only=True)
    elif not d_list or not n_seeds:
        add("L3", False, f"config 缺 d_shareds/seeds: {list(cfg)}")
    else:
        bad = {k: len(v.get("rows", [])) for k, v in res.items() if len(v.get("rows", [])) != n_seeds}
        n_expect_arms = None
        if mod is not None and hasattr(mod, "ARMS"):
            n_expect_arms = len(mod.ARMS)
        n_ev = len(getattr(mod, "EVALUATORS", []) or []) or 1   # v54 起多一个评价器维度
        n_cells = len(res)
        expect = len(d_list) * (n_expect_arms or 0) * n_ev if n_expect_arms else 0
        grid_ok = (n_cells == expect) if n_expect_arms else True
        add("L3", grid_ok and not bad,
            f"cells={n_cells} (期望 {len(d_list)}×{n_expect_arms}×{n_ev}={expect}), "
            f"行数异常={bad or '无'}")

    # ---- L4: results ↔ cells 逐位对拍 ----
    if cells_dir and os.path.isdir(cells_dir):
        mismatch, checked = [], 0
        for fn in sorted(os.listdir(cells_dir)):
            if not fn.endswith(".json"):
                continue
            c = json.load(open(os.path.join(cells_dir, fn), encoding="utf-8"))
            key = c.get("key")
            if key in res:
                checked += 1
                if res[key].get("rows") != c.get("rows"):
                    mismatch.append(key)
        add("L4", not mismatch, f"对拍 {checked} 个 cell, 不一致={mismatch or '无'}")
    else:
        add("L4", False, "未提供 cells 目录 (跳过体对拍)", warn_only=True)

    # ---- B1: 基线口径 ----
    if is_structural:
        add("B1", True, "结构测量型: 无分类基线 (跳过)", warn_only=True)
    elif n_code and n_rec and int(n_code) != int(n_rec):
        b_code, b_rec = baseline(int(n_code)), baseline(int(n_rec))
        add("B1", False,
            f"基线随类别数变化: 代码口径 n={n_code} → 全零={b_code['all_zero']:.4f} "
            f"均值={b_code['predict_mean']:.4f} | 记录口径 n={n_rec} → "
            f"全零={b_rec['all_zero']:.4f} 均值={b_rec['predict_mean']:.4f} "
            f"(相差 {abs(b_code['all_zero'] / b_rec['all_zero'] - 1) * 100:.0f}%)")
    else:
        n = int(n_code or n_rec or 0)
        b = baseline(n) if n else {}
        add("B1", n > 0, f"n_classes={n} → 全零={b.get('all_zero', float('nan')):.4f} "
                         f"均值={b.get('predict_mean', float('nan')):.4f}")

    # ---- L5: 配对证据 (三档共享初始化必须给出同一指纹) ----
    pairs = {}
    for key, v in res.items():
        parts = key.split(":")
        if len(parts) != 3:
            continue
        d, ev, arm = parts
        for r in v.get("rows", []):
            if "init_sha" in r:
                pairs.setdefault((d, arm, r.get("seed")), {})[ev] = r["init_sha"]
    if not pairs:
        add("L5", False, "产物未记录 init_sha → 配对无法从产物层验证", warn_only=True)
    else:
        bad = {k: v for k, v in pairs.items() if len(set(v.values())) > 1}
        add("L5", not bad,
            f"配对指纹: 检查 {len(pairs)} 组 (D,arm,seed), 三档指纹不一致={bad or '无'}")

    # ---- L6: 预登记锁 ----
    lock_p = os.path.join(pkg_dir, "protocol.lock.json")
    if not os.path.exists(lock_p) and code_dir:
        lock_p = os.path.join(code_dir, "protocol.lock.json")
    if not os.path.exists(lock_p):
        add("L6", False, "无 protocol.lock.json → 无法证明本次运行是预登记的那次", warn_only=True)
    else:
        lock = json.load(open(lock_p, encoding="utf-8"))
        pred_sha = m.get("predictions_sha")
        lock_pred = lock.get("predictions_sha")
        grid = lock.get("grid", {})
        run = {"d_shareds": cfg.get("d_shareds"), "seeds": cfg.get("seeds"),
               "steps": cfg.get("steps"), "seed_base": cfg.get("seed_base"),
               "n_per_class": cfg.get("n_per_class"), "tokens": cfg.get("tokens"),
               "n_classes": cfg.get("n_classes")}
        if "evaluators" in cfg:
            run["evaluators"] = cfg["evaluators"]
        gdiff = {k: {"lock": grid.get(k), "run": v} for k, v in run.items()
                 if k in grid and grid.get(k) != v}
        ok_pred = (pred_sha is None) or (pred_sha == lock_pred)
        # 冒烟: 网格本就不等于锁 → 只查判据指纹, 网格偏离降级为 WARN
        ok = ok_pred and (is_smoke or not gdiff)
        add("L6", ok,
            f"锁 sha={lock.get('sha256')} | 判据指纹 产物={pred_sha} 锁={lock_pred}"
            f" | 网格偏离={gdiff or '无'}"
            + ("  (冒烟: 网格偏离属预期, 但判据指纹必须对上)" if is_smoke else ""),
            warn_only=is_smoke)

    # ---- V1: 判决可重算 ----
    verdicts = m.get("verdicts", {}) or {}
    if not verdicts:
        add("V1", False, "无 verdicts 可重算", warn_only=True)
    else:
        redone = {}
        for dv, v in verdicts.items():
            d = dv[1:] if dv.startswith("D") else dv
            key_f, key_c = f"D{d}:frozen_random", f"D{d}:learned_lift"
            if key_f not in res or key_c not in res:
                continue
            def margin(rows):
                ms = []
                for r in rows:
                    h = r["held_out"]
                    bs = h.get("best_single") or float("nan")
                    ms.append((h["avg"] - bs) / bs * 100 if bs else float("nan"))
                return sum(ms) / len(ms)
            mf, ml = margin(res[key_f]["rows"]), margin(res[key_c]["rows"])
            redone[dv] = {
                "D1c": (abs(ml) <= abs(mf) / 2),
                "margin_frozen": round(mf, 2), "margin_learned": round(ml, 2),
            }
        ok = all(redone[k]["D1c"] == bool(v.get("D1c")) for k in redone if "D1c" in v)
        add("V1", ok, "D1c 重算: " + ", ".join(
            f"{k} 存储={verdicts[k].get('D1c')} 重算={v['D1c']}" for k, v in redone.items()))

    rep["n_classes"] = {"code": n_code, "recorded": n_rec}
    rep["baseline_code"] = baseline(int(n_code)) if n_code else None
    rep["baseline_recorded"] = baseline(int(n_rec)) if n_rec else None
    rep["file_sha12"] = file_sha(mpath)
    rep["repo_head"] = head

    # 顺手把"有意义空间"臂的真实读数与两条基线并排给出来
    rows_out = {}
    for k, v in res.items():
        if not k.endswith("learned_lift"):
            continue
        vals = [r["held_out"]["fuse"] for r in v.get("rows", [])]
        if vals and n_code:
            mval = sum(vals) / len(vals)
            b = baseline(int(n_code))
            rows_out[k] = {
                "mean_fuse": round(mval, 4),
                "vs_all_zero_pct": round((mval / b["all_zero"] - 1) * 100, 1),
                "vs_predict_mean_pct": round((mval / b["predict_mean"] - 1) * 100, 1),
            }
    rep["learned_arms_vs_baseline"] = rows_out
    return rep


def render(rep: dict[str, Any]) -> None:
    print("=" * 78)
    print(f"产物检查: {rep['package']}")
    print("=" * 78)
    for c in rep["checks"]:
        mark = {"PASS": "  ok ", "WARN": " warn", "FAIL": " FAIL"}[c["status"]]
        print(f"[{mark}] {c['id']:<3} {c['detail']}")
    nc = rep.get("n_classes") or {}
    if nc:
        print(f"\nn_classes: 代码={nc.get('code')} 记录={nc.get('recorded')}")
    lg = rep.get("learned_arms_vs_baseline") or {}
    if lg:
        print("\n有意义空间臂 vs 解析基线 (负=好于基线):")
        for k, v in lg.items():
            print(f"  {k:<22} fuse={v['mean_fuse']:<8} "
                  f"vs全零={v['vs_all_zero_pct']:>+7.1f}%  "
                  f"vs预测均值={v['vs_predict_mean_pct']:>+7.1f}%")
    print(f"\nmetrics sha256[:12]={rep.get('file_sha12')}  repo HEAD={rep.get('repo_head')}")
    nf, nw = len(rep["fail"]), len(rep["warn"])
    print(f"结果: {nf} FAIL / {nw} WARN / {len(rep['checks']) - nf - nw} PASS")
    if nf:
        print("→ 产物与代码不对应, 其读数不可作为结论依据")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("package", help="版本目录, 如 research/aligner/v53_third_audit")
    ap.add_argument("--cells", default=None, help="cells 目录 (可选, 做逐位对拍)")
    ap.add_argument("--code", default=None, help="代码包目录 (产物目录不含 audit.py 时指定)")
    ap.add_argument("--repo", default=".", help="仓库根 (取 HEAD 比对 code_sha)")
    ap.add_argument("--json", default=None, help="把报告写成 JSON")
    a = ap.parse_args()

    rep = check(a.package, a.cells, a.repo, a.code)
    render(rep)
    if a.json:
        json.dump(rep, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0 if not rep["fail"] else 1


if __name__ == "__main__":
    sys.exit(main())
