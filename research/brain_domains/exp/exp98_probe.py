#!/usr/bin/env python3
"""exp98_probe.py — **可得性扫描**（不是实验，不出判据结果）。

目的：把 Allen Visual Coding–Neuropixels（DANDI 000021）**每个 session 同时录到了哪些脑区**
列出来 ⇒ 这决定判据① 的**正/负对照**能不能配对（§三十七·三：负对照 = 同层同功能、正对照 = 跨层/跨类型）。

做法：走 DANDI 的 **HTTP 范围读**（h5py + fsspec 挂在远程文件上）⇒ **只读 HDF5 的元数据与小表**
⇒ **不下载整个 NWB**（每个 probe 文件 1.3–2.5 GB，全下不可行）。

输出：
- `/domains/allen_probe/tree_<session>.txt`：第一个 session 的 HDF5 结构（深度受限）
- `/domains/allen_probe/areas.json`：每个 session 的 {session, areas:{area:n_electrodes}}
- stdout：紧凑摘要（供 Job 日志取回）
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


def read_remote(asset_id, want_tree=False):
    """在远程 NWB 上只读元数据/小表 ⇒ 返回 (电极位置列表, 结构文本)。

    ⚠️ fsspec 的 HTTP 文件必须在 h5py 用完之前保持打开 ⇒ 读取全部在 `with` 内部完成。
    """
    import fsspec
    import h5py
    url = f"{BASE}/assets/{asset_id}/download/"
    fs = fsspec.filesystem("http")
    locs, tree = None, None
    with fs.open(url, "rb", block_size=1 << 20, cache_type="readahead") as f:
        with h5py.File(f, "r") as h:
            for path in ("general/extracellular_ephys/electrodes",
                         "general/extracellular_ephys",
                         "general/extracellular_ephys/electrodes/location"):
                if path not in h:
                    continue
                g = h[path]
                if "location" in g:
                    locs = [x.decode() if isinstance(x, bytes) else str(x)
                            for x in g["location"][:]]
                    break
                for k in list(g.keys()):
                    sub = g[k]
                    if hasattr(sub, "keys") and "location" in sub:
                        locs = [x.decode() if isinstance(x, bytes) else str(x)
                                for x in sub["location"][:]]
                        break
                if locs:
                    break
            if want_tree:
                lines = []

                def visit(name, obj):
                    if len(lines) >= 500:
                        return
                    if isinstance(obj, h5py.Dataset):
                        lines.append(f"D {name} shape={obj.shape} dtype={obj.dtype}")
                    else:
                        lines.append(f"G {name}")
                h.visititems(visit)
                tree = "\n".join(lines)
    return locs, tree


def main():
    ensure_deps()
    OUT.mkdir(parents=True, exist_ok=True)
    assets = all_assets()
    sessions = [a for a in assets
                if a["path"].endswith(".nwb") and "_probe-" not in a["path"]]
    probes = {}
    for a in assets:
        if a["path"].endswith(".nwb") and "_probe-" in a["path"]:
            probes.setdefault(a["path"].split("/")[0], []).append(a)
    print(f"assets={len(assets)} sessions={len(sessions)}", flush=True)

    table = []
    for i, a in enumerate(sessions):
        sess = a["path"].split("/")[0]
        areas, err = {}, None
        try:
            locs, tree = read_remote(a["asset_id"], want_tree=(i == 0))
            if tree:
                (OUT / ("tree_" + sess + ".txt")).write_text(tree, encoding="utf-8")
                print(f"  tree saved: {sess}", flush=True)
            if locs:
                for v in locs:
                    areas[v] = areas.get(v, 0) + 1
            else:
                err = "no electrodes in session file"
        except Exception as e:
            err = f"session {type(e).__name__}: {e}"
        if not areas:
            for pa in probes.get(sess, []):
                try:
                    locs, _ = read_remote(pa["asset_id"])
                    if not locs:
                        continue
                    for v in locs:
                        areas[v] = areas.get(v, 0) + 1
                except Exception as e:
                    err = f"probe {pa['path'].split('/')[-1]}: {type(e).__name__}: {e}"
        rec = {"session": sess, "n_electrodes": sum(areas.values()),
               "areas": areas, "err": err}
        table.append(rec)
        print("===SESS===" + json.dumps(rec, ensure_ascii=False), flush=True)

    (OUT / "areas.json").write_text(json.dumps(table, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    print("===DONE===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
