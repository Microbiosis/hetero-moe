#!/usr/bin/env python3
"""exp98_scan.py — **可得性扫描 v2**：逐 session 的**每脑区单元数** + 自然图像刺激块结构。

为什么需要：判据① 的正/负对照要**同一 session 内**配对（VISp↔VISl 负 / VISp↔VISam 正），
且每带要有足够单元（我们沿用 exp97c 的 ≥20 单元/带）⇒ ⇒ 必须先知道
**每个 session 每个脑区到底有多少单元**（这是**数据事实**，不是结果）。

做法：DANDI 000021 的 **HTTP 范围读**（h5py + fsspec 挂远程）⇒ 只读
- `general/extracellular_ephys/electrodes/{id,location}`（小）
- `units/{id,peak_channel_id}`（小；**不读** 1 GB 的 spike_times ⇒ 这一步是廉价的）
- `intervals/natural_scenes_presentations/{start_time,stop_time,stimulus_name,stimulus_index}`（小）

输出：`/domains/allen_probe/units_by_area.json` + stdout 紧凑摘要。
"""
import json
import pathlib
import subprocess
import sys
import urllib.request

DANDI = "000021"
VER = "0.251116.2246"
OUT = pathlib.Path("/domains/allen_probe")
BASE = f"https://api.dandiarchive.org/api/dandisets/{DANDI}/versions/{VER}"


def ensure_deps():
    for mod, pkg in (("h5py", "h5py"), ("fsspec", "fsspec")):
        try:
            __import__(mod)
        except Exception:
            print(f"installing {pkg} ...", flush=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                            "--no-cache-dir", pkg], check=True)


def get_json(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.loads(r.read().decode())


def all_assets():
    out, url = [], f"{BASE}/assets/?page_size=100&fields=asset_id,path,size"
    while url:
        d = get_json(url)
        out += d.get("results", [])
        url = d.get("next")
    return out


def dec(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", "replace")
    return str(x)


def read_session(asset_id):
    """⇒ (每脑区单元数, 自然图像块信息)"""
    import fsspec
    import h5py
    url = f"{BASE}/assets/{asset_id}/download/"
    fs = fsspec.filesystem("http")
    out = {}
    with fs.open(url, "rb", block_size=1 << 20, cache_type="readahead") as f:
        with h5py.File(f, "r") as h:
            el = h["general/extracellular_ephys/electrodes"]
            eid = el["id"][:]
            loc = [dec(x) for x in el["location"][:]]
            e2loc = {int(i): l for i, l in zip(eid, loc)}
            units = h["units"]
            peak = units["peak_channel_id"][:]
            per_area = {}
            n_no_peak = 0
            for p in peak:
                try:
                    pi = int(p)
                except Exception:
                    n_no_peak += 1
                    continue
                l = e2loc.get(pi)
                if l is None:
                    n_no_peak += 1
                    continue
                per_area[l] = per_area.get(l, 0) + 1
            out["units_by_area"] = per_area
            out["n_units"] = int(len(peak))
            out["n_units_unmapped"] = n_no_peak
            ns = h["intervals/natural_scenes_presentations"]
            st = ns["start_time"][:]
            sp = ns["stop_time"][:]
            names = [dec(x) for x in ns["stimulus_name"][:]]
            out["n_ns_presentations"] = int(len(st))
            out["n_ns_classes"] = len(set(names))
            out["ns_duration_s"] = [float(sp[0] - st[0]), float(sp[-1] - st[0])]
            out["ns_class_examples"] = sorted(set(names))[:3]
    return out


def main():
    ensure_deps()
    OUT.mkdir(parents=True, exist_ok=True)
    assets = all_assets()
    sessions = [a for a in assets
                if a["path"].endswith(".nwb") and "_probe-" not in a["path"]]
    print(f"sessions={len(sessions)}", flush=True)
    table = []
    for a in sessions:
        sess = a["path"].split("/")[0]
        try:
            rec = read_session(a["asset_id"])
        except Exception as e:
            rec = {"err": f"{type(e).__name__}: {e}"}
        rec["session"] = sess
        table.append(rec)
        keep = {k: v for k, v in (rec.get("units_by_area") or {}).items()
                if k and (k.startswith("VIS") or k in ("CA1", "CA2", "CA3", "SUB",
                                                      "DG", "HPF", "LGd", "LGv", "LP",
                                                      "APN", "TH"))}
        rec["units_by_area_kept"] = keep
        print("===S===" + json.dumps({"session": sess,
                                      "n_units": rec.get("n_units"),
                                      "unmapped": rec.get("n_units_unmapped"),
                                      "ns_pres": rec.get("n_ns_presentations"),
                                      "ns_classes": rec.get("n_ns_classes"),
                                      "ns_dur": rec.get("ns_duration_s"),
                                      "areas": keep,
                                      "err": rec.get("err")},
                                     ensure_ascii=False), flush=True)
    (OUT / "units_by_area.json").write_text(json.dumps(table, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    print("===DONE===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
