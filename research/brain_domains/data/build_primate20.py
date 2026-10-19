#!/usr/bin/env python3
"""Two sources, one time grid, no resampling.

    macaque_nlb          /domains/primate/macaque_nlb/spikes_*.npz      72 segments
    human_intracortical  HF eminorhan/h1, eminorhan/h2                 103 segments

Both are spike counts in 20 ms bins, so nothing is resampled in time.

    per-unit z-score over the whole segment
    per-segment PCA to 64 components (Gram matrix + eigh)
    window 40 bins = 800 ms, stride 20 bins = 50%

Output (N, 64, 40) float16.
"""
import gc, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "32")

WIN, STRIDE, DIM, EPS = 40, 20, 64, 1e-6
SEG = Path("/domains/primate/macaque_nlb")
OUT = Path("/tmp/two")
OUT.mkdir(parents=True, exist_ok=True)

# NLB dataset -> (region_raw, region, region_kind, shared_node, mapping_status, note)
# Verified against each NWB's general/extracellular_ephys/electrodes:location:
#   000127 area2-bump -> b"S1 Area 2"   000130 dmfc-rsg -> b"DMFC"
#   000070 churchland -> M1 and PMd (source paper)
# segments.parquet still carries "S1_M1" for area2-bump; that is a stale label
# from an earlier build and is deliberately not read here.
MACAQUE = {
    "area2-bump": ("Area2",  "S1",  "anatomy", "S1",  "exact",
                   "Brodmann area 2, within primary somatosensory cortex"),
    "dmfc-rsg":   ("DMFC",   "PFC", "anatomy", "PFC", "coarse",
                   "dorsomedial frontal cortex, resolved to prefrontal cortex"),
    "churchland": ("M1_PMd", "M1",  "anatomy", "M1",  "compound",
                   "primary motor and dorsal premotor cortex, not separable per unit"),
}
HUMAN = {
    "h1": "FALCON H1, 7DoF reach and grasp",
    "h2": "FALCON H2, handwriting",
}
DOMAIN = {"Area2": "sensory", "DMFC": "association", "M1_PMd": "motor", "M1": "motor"}


def pad_units(sc):
    arrs = [np.asarray(u, dtype=np.float32).ravel() for u in sc]
    L = max(len(a) for a in arrs)
    m = np.zeros((len(arrs), L), dtype=np.float32)
    for i, a in enumerate(arrs):
        m[i, :len(a)] = a
    return m, [len(a) for a in arrs]


def windows(mat):
    nb = mat.shape[1]
    if nb < WIN or mat.shape[0] < 5:
        return None, 0, 0.0
    mu = mat.mean(axis=1, keepdims=True)
    sd = mat.std(axis=1, keepdims=True) + EPS
    z = (mat - mu) / sd
    k = min(DIM, z.shape[0] - 1, nb - 1)
    if k < 2:
        return None, 0, 0.0
    zc = z - z.mean(axis=1, keepdims=True)
    w, U = np.linalg.eigh(zc @ zc.T)
    order = np.argsort(-w)[:k]
    proj = zc.T @ U[:, order]
    if proj.shape[1] < DIM:
        proj = np.pad(proj, ((0, 0), (0, DIM - proj.shape[1])))
    else:
        proj = proj[:, :DIM]
    ev = float(np.clip(w[order], 0, None).sum() / max(w.sum(), EPS))
    starts = list(range(0, nb - WIN + 1, STRIDE))
    arr = np.stack([proj[s:s + WIN].T for s in starts]).astype(np.float16)
    return arr, arr.shape[0], ev


def main():
    t0 = time.time()
    rows, chunks = [], []
    seg_no = 0

    segs = pd.read_parquet(SEG / "segments.parquet")
    print(f"macaque segments: {len(segs)}", flush=True)
    for i, r in segs.iterrows():
        sp = np.load(SEG / r["file"])["spikes"].astype(np.float32)
        arr, n, ev = windows(sp)
        nb = sp.shape[1]
        del sp
        if arr is None:
            print(f"  [skip] {r['file']} too short", flush=True)
            continue
        raw, reg, kind, node, status, note = MACAQUE[str(r["dataset"])]
        chunks.append(arr)
        rows.append(pd.DataFrame({
            "source": "macaque_nlb", "species": "macaca_mulatta",
            "modality": "extracellular_spike", "group_key": str(r["session"]),
            "segment_idx": seg_no, "file": str(r["file"]),
            "window_idx": np.arange(n), "win_in_segment": np.arange(n),
            "region_raw": raw, "region": reg, "region_kind": kind,
            "shared_node": node, "mapping_status": status, "region_note": note,
            "n_units": int(r["n_units"]), "n_bins_20ms": int(nb),
            "t0_bin20": np.arange(n) * STRIDE,
            "explained_var": round(ev, 4), "domain_l1": DOMAIN[raw],
        }))
        seg_no += 1
        if i % 20 == 0:
            print(f"  [{i+1}/{len(segs)}] {r['file']} {n} win [{time.time()-t0:.0f}s]",
                  flush=True)
        gc.collect()

    from huggingface_hub import hf_hub_download
    for ds in ("h1", "h2"):
        p = hf_hub_download(f"eminorhan/{ds}",
                            "data/train-00000-of-00001.parquet", repo_type="dataset")
        df = pd.read_parquet(p)
        print(f"{ds}: {len(df)} segments ({HUMAN[ds]})", flush=True)
        for i in range(len(df)):
            mat, lens = pad_units(df.iloc[i]["spike_counts"])
            arr, n, ev = windows(mat)
            nb = mat.shape[1]
            nu = mat.shape[0]
            del mat
            if arr is None:
                continue
            chunks.append(arr)
            rows.append(pd.DataFrame({
                "source": "human_intracortical", "species": "homo_sapiens",
                "modality": "extracellular_spike",
                "group_key": f"{ds}:{df.iloc[i]['session_id']}",
                "segment_idx": seg_no, "file": f"{ds}_{i:03d}",
                "window_idx": np.arange(n), "win_in_segment": np.arange(n),
                "region_raw": f"{ds}_intracortical", "region": "M1",
                "region_kind": "anatomy", "shared_node": "M1",
                "mapping_status": "task_code_reinterpreted",
                "region_note": f"hand-area M1 Utah array, {HUMAN[ds]}",
                "n_units": int(nu), "n_bins_20ms": int(nb),
                "t0_bin20": np.arange(n) * STRIDE,
                "explained_var": round(ev, 4), "domain_l1": "motor",
            }))
            seg_no += 1
        gc.collect()

    idx = pd.concat(rows, ignore_index=True)
    # window_idx is a per-source counter, the same convention as the other tables
    for src in idx.source.unique():
        m = (idx.source == src).to_numpy()
        idx.loc[m, "window_idx"] = np.arange(m.sum())

    W = np.concatenate(chunks, axis=0)
    del chunks
    assert len(W) == len(idx), (len(W), len(idx))
    print(f"\ntensor {W.shape} {W.dtype} ({W.nbytes/1e9:.2f} GB)", flush=True)
    np.save(OUT / "timeseries.npy", W)
    idx.to_parquet(OUT / "index_timeseries.parquet", index=False)
    print(idx.groupby(["source", "region"]).size().to_string(), flush=True)
    print(f"segments {seg_no} | done [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
