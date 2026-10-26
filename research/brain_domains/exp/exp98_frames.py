#!/usr/bin/env python3
"""exp98_frames.py — 探针：**自然图像的类别标签到底在哪一列、类数是多少**。

为什么需要：exp98 的 C0′（可读性）与码本都锚在「刺激身份（自然图像）」上。
已知的 NWB 结构事实：`intervals/natural_scenes_presentations` 有
`frame` / `stimulus_index` / `stimulus_name` / `start_time` / `stop_time`，
其中 `stimulus_name` 全同（1 类，**不可用**）⇒ 类别必须来自 `frame` 或 `stimulus_index`。
本探针只读这一个小表 ⇒ 便宜；**不读 spike_times**。

输出：
- stdout：`===F===` 每 session 的类数统计
- 桶内：`/domains/allen_probe/frames_<session>.json`（第一个 session 的逐 trial 表：start/stop/frame）
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


def read_frames(asset_id):
    import fsspec
    import h5py
    import numpy as np
    url = f"{BASE}/assets/{asset_id}/download/"
    fs = fsspec.filesystem("http")
    with fs.open(url, "rb", block_size=1 << 20, cache_type="readahead") as f:
        with h5py.File(f, "r") as h:
            ns = h["intervals/natural_scenes_presentations"]
            st = ns["start_time"][:]
            sp = ns["stop_time"][:]
            fr = ns["frame"][:]
            si = ns["stimulus_index"][:]
            nm = [dec(x) for x in ns["stimulus_name"][:]]
            out = {
                "n": int(len(st)),
                "frame_unique": int(len(np.unique(fr))),
                "frame_min": float(np.min(fr)), "frame_max": float(np.max(fr)),
                "stim_index_unique": int(len(np.unique(si))),
                "stim_index_min": float(np.min(si)), "stim_index_max": float(np.max(si)),
                "name_unique": len(set(nm)),
                "name_examples": sorted(set(nm))[:3],
                "dur_min": float(np.min(sp - st)), "dur_max": float(np.max(sp - st)),
            }
            # frame ↔ stimulus_index 是否一一对应
            pairs = set(zip(fr.tolist(), si.tolist()))
            out["frame_index_pairs"] = len(pairs)
            # 每类次数分布
            u, c = np.unique(fr, return_counts=True)
            out["frame_count_min"] = int(c.min())
            out["frame_count_max"] = int(c.max())
            out["frame_n"] = int(len(u))
            out["frames"] = u.tolist()
            out["counts"] = c.tolist()
            out["trial"] = {"start": st.tolist(), "stop": sp.tolist(),
                            "frame": fr.tolist(), "stim_index": si.tolist()}
            return out


def main():
    ensure_deps()
    OUT.mkdir(parents=True, exist_ok=True)
    assets = all_assets()
    sessions = [a for a in assets
                if a["path"].endswith(".nwb") and "_probe-" not in a["path"]]
    print(f"sessions={len(sessions)}", flush=True)
    for i, a in enumerate(sessions):
        sess = a["path"].split("/")[0]
        try:
            r = read_frames(a["asset_id"])
            err = None
        except Exception as e:
            r, err = {}, f"{type(e).__name__}: {e}"
        rec = {"session": sess, "err": err}
        rec.update({k: v for k, v in r.items() if k != "trial"})
        print("===F===" + json.dumps(rec, ensure_ascii=False), flush=True)
        if i == 0 and r:
            (OUT / f"frames_{sess}.json").write_text(
                json.dumps(r["trial"], ensure_ascii=False), encoding="utf-8")
            print(f"  wrote {OUT}/frames_{sess}.json", flush=True)
    print("===DONE===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
