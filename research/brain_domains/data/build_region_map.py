#!/usr/bin/env python3
"""Build a region mapping table for illusion-brain/omnibrain-multi.

Inputs (downloaded once, cached next to this script):
    index_timeseries.parquet, index_graph.parquet  -- from the dataset repo
    mba_struct.csv     -- Allen mouse CCF structure table (api.brain-map.org)
    mba2uberon.json    -- MBA -> UBERON from uberon-bridge-to-mba.owl

Outputs:
    region_map.csv        one row per (source, region_raw)
    index_timeseries.parquet  columns added: region_raw, region, region_kind,
                              structure_id, uberon_id, shared_node, mapping_status
    index_graph.parquet       same added columns
    region_groups.npz     window indices for each cross-species node

The point of the exercise: `region` in v1 mixes four different kinds of label.
Column `region_kind` states which kind, so the four never get pooled again.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "illusion-brain/omnibrain-multi"
HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- fetch
def _get(name, repo_type="dataset"):
    local = HERE / name
    if local.exists():
        return local
    p = hf_hub_download(REPO, name, repo_type=repo_type, local_dir=str(HERE))
    return Path(p)


ts = pd.read_parquet(_get("index_timeseries.parquet"))
gr = pd.read_parquet(_get("index_graph.parquet"))

# ---------------------------------------------------------------- UBERON ids
# Verified against OLS4 (ontology=uberon) on 2026-10-19.
UBERON = {
    "M1":   ("UBERON:0001384", "primary motor cortex"),
    "PMd":  ("UBERON:0016634", "premotor cortex"),
    "S1":   ("UBERON:0008933", "primary somatosensory cortex"),
    "PFC":  ("UBERON:0000451", "prefrontal cortex"),
}

# Canonical id of each cross-species node. Per-row uberon_id may be finer
# (a mouse prelimbic area under the PFC node) or coarser (a compound macaque
# label); the node id is the level at which the species are actually joined.
NODE = {
    "M1":  ("UBERON:0001384", "primary motor cortex"),
    "S1":  ("UBERON:0008933", "primary somatosensory cortex"),
    "PFC": ("UBERON:0000451", "prefrontal cortex"),
}

# Allen CCF acronym -> structure_id. Restricting to the adult CCF (root 997) is
# required: abbreviations are only unique there.  PL/RSP/SSp collide elsewhere.
mba = pd.read_csv(HERE / "mba_struct.csv", dtype={"id": str, "parent_structure_id": str})
adult = mba[mba.structure_id_path.astype(str).str.startswith("/997/")].copy()
adult = adult.drop_duplicates("acronym")
acro2id = dict(zip(adult.acronym, adult.id))
acro2name = dict(zip(adult.acronym, adult["name"]))

mba2uberon = {k: v[0] for k, v in json.loads((HERE / "mba2uberon.json").read_text()).items()}

# Explicit cross-species assignment for the three nodes present in >= 2 species.
# Anything not listed is single-species and gets shared_node = "".
SHARED = {
    # mouse
    "MOp": "M1", "SSp": "S1", "PL": "PFC",
    # macaque
    "M1_PMd": "M1", "Area2": "S1", "DMFC": "PFC",
    # human intracortical
    "h1_intracortical": "M1", "h2_intracortical": "M1",
}

MOUSE = set(acro2id)
MACAQUE = {"M1_PMd", "Area2", "DMFC"}
HUMAN_IC = {"h1_intracortical", "h2_intracortical"}

# FBbt lookups confirmed on OLS4 (ontology=fbbt).
FBbt = {
    "ME": ("FBbt:00003748", "medulla", "neuropil"),
    "LO": ("FBbt:00003852", "lobula", "neuropil"),
    "LA": ("FBbt:00003708", "lamina", "neuropil"),
    "LOP": ("FBbt:00003885", "lobula plate", "neuropil"),
    "CX": ("FBbt:00003632", "adult central complex", "neuropil"),
    "ALPN": ("FBbt:00007422", "antennal lobe projection neuron", "cell_type"),
    "Kenyon_Cell": ("FBbt:00003686", "Kenyon cell", "cell_type"),
    "MBON": ("FBbt:00047953", "mushroom body output neuron", "cell_type"),
    "DAN": ("FBbt:00005131", "dopaminergic neuron", "cell_type"),
    "pars_intercerebralis": ("FBbt:00005802", "pars intercerebralis", "region"),
    "pars_lateralis": (None, "pars lateralis", "region"),
    "optic_lobes": (None, "optic lobes", "region"),
}


def map_row(source, raw):
    """-> (region_kind, structure_id, uberon_id, shared_node, status, note)"""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        if source == "flywire":
            return ("none", "", "", "", "unassigned",
                    "cell without a FlyWire region annotation")
        return ("none", "", "", "", "unassigned",
                "upstream index carries no region; scalp EEG has no focal site")

    if source == "mouse_steinmetz":
        sid = acro2id.get(raw, "")
        if not sid:
            return ("anatomy", "", "", "", "unmapped", "not in adult Allen CCF")
        u = mba2uberon.get(sid, "")
        u = ("UBERON:" + u) if u else ""
        shared = SHARED.get(raw, "")
        if u and shared:
            st = "exact" if u == NODE[shared][0] else "rollup"
        elif u:
            st = "exact_single_species"
        else:
            st = "uberon_missing"
        note = acro2name.get(raw, "")
        return ("anatomy", "MBA:" + sid, u, shared, st, note)

    if source == "macaque_nlb":
        if raw == "M1_PMd":
            return ("anatomy", "", f"{UBERON['M1'][0]}|{UBERON['PMd'][0]}",
                    "M1", "compound",
                    "recordings span primary motor and dorsal premotor cortex; not separable in upstream index")
        if raw == "Area2":
            return ("anatomy", "", UBERON["S1"][0], "S1", "coarse",
                    "Brodmann area 2, a subarea of primary somatosensory cortex")
        if raw == "DMFC":
            return ("anatomy", "", UBERON["PFC"][0], "PFC", "coarse",
                    "dorsomedial frontal cortex, resolved to prefrontal cortex")
        return ("anatomy", "", "", "", "unmapped", "")

    if source == "human_intracortical":
        return ("anatomy", "", UBERON["M1"][0], "M1", "task_code_reinterpreted",
                f"upstream value {raw} is a FALCON task code (H1 7DoF reach/grasp, "
                "H2 handwriting); both are hand-area M1 Utah arrays")

    if source == "human_eeg":
        return ("none", "", "", "", "unassigned", "64-channel scalp EEG, no focal site")

    if source == "flywire":
        if raw in FBbt:
            fid, label, kind = FBbt[raw]
            return (kind, fid or "", "", "", "no_vertebrate_homolog" if fid else "fbbt_unresolved",
                    "FBbt: " + label)
        return ("cell_type", "", "", "", "no_vertebrate_homolog",
                "FlyWire cell class / superclass label, not a vertebrate-homologous region")

    return ("unknown", "", "", "", "unmapped", "")


def annotate(df):
    df = df.copy()
    df["region_raw"] = df["region"]
    recs = [map_row(s, r) for s, r in zip(df["source"], df["region_raw"])]
    df["region_kind"] = [r[0] for r in recs]
    df["structure_id"] = [r[1] for r in recs]
    df["uberon_id"] = [r[2] for r in recs]
    df["shared_node"] = [r[3] for r in recs]
    df["mapping_status"] = [r[4] for r in recs]
    # `region` is now the harmonised label: the raw value except for the two
    # human task codes, which were never anatomical labels.
    df["region"] = [
        "M1" if (s == "human_intracortical") else r
        for s, r in zip(df["source"], df["region_raw"])
    ]
    cols = list(df.columns)
    head = ["source", "species", "modality", "group_key", "window_idx", "domain_l1",
            "region", "region_raw", "region_kind", "structure_id", "uberon_id",
            "shared_node", "mapping_status", "region_level", "area", "n_units"]
    return df[[c for c in head if c in cols] + [c for c in cols if c not in head]]


ts2 = annotate(ts)
gr2 = annotate(gr)

# ---------------------------------------------------------------- map table
rows = []
for (src, raw), sub in pd.concat([ts2, gr2]).groupby(["source", "region_raw"], dropna=False):
    r = sub.iloc[0]
    rows.append({
        "source": src,
        "region_raw": raw if isinstance(raw, str) else "",
        "n_rows": len(sub),
        "region_kind": r["region_kind"],
        "structure_id": r["structure_id"],
        "uberon_id": r["uberon_id"],
        "shared_node": r["shared_node"],
        "mapping_status": r["mapping_status"],
        "note": map_row(src, raw)[5],
    })
rmap = pd.DataFrame(rows).sort_values(["source", "n_rows"], ascending=[True, False])
rmap.to_csv(HERE / "region_map.csv", index=False)

# ---------------------------------------------------------------- groups
groups = {}
for node in ["M1", "S1", "PFC"]:
    idx = np.flatnonzero((ts2["shared_node"] == node).to_numpy())
    groups[node] = idx.astype(np.int32)
    print(f"{node}: {len(idx)} windows")
np.savez_compressed(HERE / "region_groups.npz", **groups)

# ---------------------------------------------------------------- node table
srcs = ["mouse_steinmetz", "macaque_nlb", "human_intracortical", "human_eeg"]
srows = []
for node in ["M1", "S1", "PFC"]:
    g = ts2[ts2["shared_node"] == node]
    d = {"node": node, "uberon_id": NODE[node][0], "label": NODE[node][1]}
    for s in srcs:
        d[s] = int((g["source"] == s).sum())
    d["total"] = len(g)
    d["n_species"] = sum(d[s] > 0 for s in srcs)
    srows.append(d)
pd.DataFrame(srows).to_csv(HERE / "shared_nodes.csv", index=False)

ts2.to_parquet(HERE / "index_timeseries.parquet", index=False)
gr2.to_parquet(HERE / "index_graph.parquet", index=False)

print()
print(rmap.to_string(index=False))
print()
print("region_kind x n:")
print(pd.concat([ts2, gr2]).groupby(["source", "region_kind"]).size().to_string())
