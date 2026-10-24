"""parse_exp93_logs.py — 从 **Job 日志** 重建 exp93d / exp93e 的表。

## 为什么需要

exp93d（模型层读出 @ n_train=16540）与 exp93e（窗口扫描）在 Job 里跑，
本机写通道坏 ⇒ 结果 JSON 落在桶里；而桶里的读通道时断时续，
且三个 CPU Job 互相抢核导致**推进极慢**。

⇒ 先从 **Job 日志**（`hf jobs logs`）把已完成的部分解析出来，
JSON 落盘后再用 `parse_exp93.py --force` 覆盖（两者口径一致，可对拍）。

## 解析的两张表

- **exp93e 窗口表**：日志行形如
  `  [w50_500  ] POST=+0.1254 ANT=+0.0103 | level acc1=0.0495 (9.9×) | 随机组中位 +0.0788 最大 +0.0986 | POST>+0.0986=True ANT<+0.0986=True [70.5s]`
- **exp93d 档位表**：日志行形如
  `  LEVEL s1a_onehot_mean acc1=0.0057 (零中位 0.0051)`

## 统计口径

**中位 = 上中位**（`sorted[n//2]`，偶数取偏大）⇒ 右偏分布会报高 30–40%。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG_E = HERE / "exp93e_window_job_log.txt"
LOG_D = HERE / "exp93d_level_job_log.txt"
LOG_A = HERE / "exp93a_s1-5_job_log.txt"
OUT = HERE / "exp93-results"

RE_SUBJ = re.compile(r"^===\s*(sub-\d+)\s*===")
RE_WIN = re.compile(
    r"\[w(?P<w>[0-9_]+)\s*\]\s*POST=(?P<post>[+-][\d.]+)\s+ANT=(?P<ant>[+-][\d.]+)"
    r"\s*\|\s*level acc1=(?P<lvl>[\d.]+)\s*\((?P<lvlx>[\d.]+)×\)"
    r"\s*\|\s*随机组中位\s+(?P<nmed>[+-][\d.]+)\s+最大\s+(?P<nmax>[+-][\d.]+)"
    r"\s*\|\s*POST>(?P<nmax2>[+-][\d.]+)=(?P<pabove>True|False)"
    r"\s+ANT<(?P<nmax3>[+-][\d.]+)=(?P<pbelow>True|False)")
RE_LEVEL = re.compile(r"LEVEL\s+(?P<tag>\S+)\s+acc1=(?P<acc>[\d.]+)\s+\(零中位\s+(?P<nul>[\d.]+)\)")
RE_BASE = re.compile(r"\[基准 P1/ridge/16540\] acc1=(?P<acc>[\d.]+)")
RE_F1 = re.compile(r"\[F1 n_train=(?P<n>\d+)\] acc1=(?P<acc>[\d.]+)")
RE_F2 = re.compile(r"\[F2 (?P<tag>\S+)\s*\]\s*acc1=(?P<acc>[\d.]+)")
RE_P3 = re.compile(r"\[F5 P3 同集身份解码\] acc=(?P<acc>[\d.]+)")
RE_P2 = re.compile(r"\[S2 P2 同集检索\] acc1=(?P<acc>[\d.]+)")


def med(xs):
    """**上中位**（本项目约定）：sorted[n//2]，偶数取偏大。"""
    xs = [float(x) for x in xs if x == x]
    if not xs:
        return None
    s = sorted(xs)
    return s[len(s) // 2]


def mean(xs):
    xs = [float(x) for x in xs if x == x]
    return float(sum(xs) / len(xs)) if xs else None


def parse_window(path):
    if not path.exists():
        return None
    cur = None
    per = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = RE_SUBJ.match(line.strip())
        if m:
            cur = m.group(1)
            continue
        m = RE_WIN.search(line)
        if not m or cur is None:
            continue
        d = m.groupdict()
        per.setdefault(cur, {})[d["w"]] = {
            "post": float(d["post"]), "ant": float(d["ant"]),
            "level": float(d["lvl"]),
            "null_median": float(d["nmed"]), "null_max": float(d["nmax"]),
            "post_above": d["pabove"] == "True", "ant_below": d["pbelow"] == "True"}
    if not per:
        return None
    wins = []
    for v in per.values():
        for w in v:
            if w not in wins:
                wins.append(w)
    order = ["0_250", "50_500", "100_400", "50_250", "100_250"]
    wins = [w for w in order if w in wins] + [w for w in wins if w not in order]
    out = {"n_ok": len(per), "source": "job_log",
           "subjects": sorted(per), "windows": {}}
    for w in wins:
        vals = [v[w] for v in per.values() if w in v]
        if not vals:
            continue
        out["windows"][w] = {
            "POST": {"median_up": med([x["post"] for x in vals]),
                     "mean": mean([x["post"] for x in vals])},
            "ANT": {"median_up": med([x["ant"] for x in vals]),
                    "mean": mean([x["ant"] for x in vals])},
            "level": {"median_up": med([x["level"] for x in vals]),
                      "mean": mean([x["level"] for x in vals])},
            "null_median": med([x["null_median"] for x in vals]),
            "POST_above_all": sum(1 for x in vals if x["post_above"]),
            "ANT_below_all": sum(1 for x in vals if x["ant_below"]),
            "n": len(vals)}
    return out


def parse_level(path):
    if not path.exists():
        return None
    cur = None
    per = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = RE_SUBJ.match(line.strip())
        if m:
            cur = m.group(1)
            continue
        m = RE_LEVEL.search(line)
        if not m or cur is None:
            continue
        per.setdefault(cur, {})[m.group("tag")] = {
            "acc1": float(m.group("acc")), "null": float(m.group("nul"))}
    if not per:
        return None
    tags = []
    for v in per.values():
        for t in v:
            if t not in tags:
                tags.append(t)
    out = {"n_ok": len(per), "source": "job_log", "subjects": sorted(per),
           "tags": {}}
    for t in tags:
        vals = [v[t]["acc1"] for v in per.values() if t in v]
        out["tags"][t] = {"median_up": med(vals), "mean": mean(vals),
                          "n": len(vals)}
    return out


def parse_audit(path):
    if not path.exists():
        return None
    cur = None
    per = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = RE_SUBJ.match(line.strip())
        if m:
            cur = m.group(1)
            per.setdefault(cur, {})
            continue
        if cur is None:
            continue
        for rx, key in ((RE_BASE, "base"), (RE_F1, "f1"), (RE_F2, "f2"),
                        (RE_P3, "p3"), (RE_P2, "p2")):
            mm = rx.search(line)
            if mm:
                d = mm.groupdict()
                if key == "f1":
                    per[cur].setdefault("f1", {})[d["n"]] = float(d["acc"])
                elif key == "f2":
                    per[cur].setdefault("f2", {})[d["tag"]] = float(d["acc"])
                else:
                    per[cur][key] = float(d["acc"])
    if not per:
        return None
    return {"n_ok": len(per), "source": "job_log", "subjects": sorted(per),
            "per_subject": per}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    w = parse_window(LOG_E)
    if w:
        (OUT / "exp93e_window.from_log.json").write_text(
            json.dumps(w, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"exp93e（日志）：{w['n_ok']} 个被试，窗口 {list(w['windows'])}")
        for k, v in w["windows"].items():
            print(f"  {k:<9} POST {v['POST']['median_up']:+.4f}  ANT {v['ANT']['median_up']:+.4f}"
                  f"  档位 {v['level']['median_up']:.4f}"
                  f"  POST>{v['POST_above_all']}/{v['n']}"
                  f"  ANT<{v['ANT_below_all']}/{v['n']}")
    else:
        print("exp93e：日志里没解析出东西")
    d = parse_level(LOG_D)
    if d:
        (OUT / "exp93d_level.from_log.json").write_text(
            json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"exp93d（日志）：{d['n_ok']} 个被试")
        for t, v in d["tags"].items():
            print(f"  {t:<18} acc1 中位 {v['median_up']:.4f}（n={v['n']}）")
    a = parse_audit(LOG_A)
    if a:
        (OUT / "exp93a_s1-5.from_log.json").write_text(
            json.dumps(a, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"exp93a s1-5（日志）：{a['n_ok']} 个被试")
        for s, v in a["per_subject"].items():
            print(f"  {s}: base {v.get('base')} f1 {v.get('f1')} p3 {v.get('p3')}")


if __name__ == "__main__":
    main()
