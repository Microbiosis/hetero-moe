"""构建 OmniBrain：跨数据集统一脑域数据集。

设计见同目录 OMNIBRAIN_DESIGN.md。三个视图各自独立，共享的只有
bin 宽（100 ms）与 L1 功能域本体，不假设 EEG 与 spike 同质。

## 视图

eeg_view   THINGS-EEG (人类, 63 电极)  → (N, 64, 8) float16
spike_view Steinmetz (小鼠, 变长单元)   → 逐 session (n_win, 64, 8) float16
graph_view FlyWire (果蝇, 连接组)      → (M, 64) float32 邻接谱嵌入

## 关键处理

1. **时间重采样**: 统一到 100 ms bin
   - EEG: 250 Hz, 25 samples/bin, 取均值
   - 小鼠: 50 ms, 2 bins/bin, 取和 (spike counts 是可加的)
2. **通道投影**: 每视图独立
   - EEG: 63 → 64, 补一个零通道 (可逆, 不引入旋转)
   - 小鼠: 逐 session PCA, n_units → 64 (不可逆, 记录 explained_var)
   - 果蝇: 稀疏邻接矩阵 SVD, 取前 64 个左奇异向量
3. **归一化**: 逐通道 z-score (必需); per-domain 去均值默认关闭
4. **域标签**: 窗内活动加权主导域 + 置信度

## 用法

    python build_omnibrain.py <数据根> <输出根>

数据根需含:
    steinmetz/{spikes_session_*.npz, units.parquet}
    primate/human_eeg/{eeg_sub-*_{train,val,test}.npz, trials.parquet, channels.csv}
    flywire/{node_features.parquet, connections.parquet}   (connections 可选)
"""

from __future__ import annotations

import os

# ⚠️ 必须在 import numpy 之前设置: 单线程 BLAS 在这类宽矩阵乘法上只有
# 0.2 GFLOPS, 8 线程可达 ~17 GFLOPS (实测 600x)。这是本构建器最大的性能开关。
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import gc
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------- 全局参数 ----------------
BIN_MS = 100.0            # 统一时间分辨率
WIN_BINS = 8              # 窗长 (0.8 s) —— EEG 单 trial 仅 10 bins, 取交集
STRIDE_BINS = 4           # 50% 重叠
PROJ_DIM = 64             # 统一通道维度
SEED = 0
EPS = 1e-6


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def atomic_save_np(path: Path, **arrays):
    """原子写 numpy (先 .tmp 后 rename); np.savez 会自动加 .npz, 故用句柄。"""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def atomic_savez(path: Path, **arrays):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


# ---------------------------------------------------------------- 本体
# L1 功能域: 6 类。每条映射附证据等级 (strong / medium / weak)
L1_DOMAINS = ["sensory", "motor", "memory", "association", "subcortical", "other"]

# 小鼠 (Steinmetz, Allen CCF 前缀) → L1
STEINMETZ_L1 = {
    "VIS": ("sensory", "strong"), "AUD": ("sensory", "strong"),
    "SSp": ("sensory", "strong"), "SSs": ("sensory", "strong"),
    "MOp": ("motor", "strong"), "MOs": ("motor", "strong"),
    "ACA": ("association", "medium"), "PL": ("association", "medium"),
    "ILA": ("association", "medium"), "ORB": ("association", "medium"),
    "FRP": ("association", "medium"),
    "CA1": ("memory", "strong"), "CA2": ("memory", "strong"),
    "CA3": ("memory", "strong"), "DG": ("memory", "strong"),
    "SUB": ("memory", "strong"), "ProS": ("memory", "strong"),
    "LGd": ("sensory", "strong"), "LP": ("association", "medium"),
    "LD": ("association", "medium"), "MD": ("association", "medium"),
    "PO": ("sensory", "strong"), "VPM": ("sensory", "strong"),
    "VPL": ("sensory", "strong"), "MG": ("sensory", "strong"),
    "VAL": ("subcortical", "medium"),
    "CP": ("motor", "medium"), "ACB": ("subcortical", "medium"),
    "STR": ("motor", "medium"),
    "LS": ("subcortical", "medium"),
    "TH": ("subcortical", "medium"), "ZI": ("subcortical", "medium"),
    "MRN": ("subcortical", "medium"), "MB": ("subcortical", "medium"),
    "SNr": ("subcortical", "medium"), "VTA": ("subcortical", "medium"),
    "SC": ("sensory", "medium"), "IC": ("sensory", "medium"),
    "MY": ("subcortical", "medium"), "PAG": ("subcortical", "medium"),
    "RN": ("subcortical", "medium"),
}

# 人类 EEG 头皮区 → L1（弱: 仅按电极位置归类，不反映皮层功能）
EEG_REGION_L1 = {
    "frontal": ("association", "weak"),
    "central": ("motor", "weak"),
    "temporal": ("sensory", "weak"),
    "parietal": ("association", "weak"),
    "occipital": ("sensory", "weak"),
}

# 果蝇原始 region 前缀 → 功能域 (读上游 raw nodes.parquet 时需要)
FLYWIRE_REGION_TO_FUNC = {
    "ME": "visual", "LO": "visual", "LOP": "visual", "visual": "visual",
    "LA": "olfactory", "olfactory": "olfactory",
    "Kenyon_Cell": "memory", "mushroom_body": "memory",
    "CX": "central_complex",
    "mechanosensory": "somatosensory", "gustatory": "somatosensory",
    "AN": "ascending", "DAN": "descending",
    "ALPN": "olfactory", "ALLN": "olfactory",
}

# 果蝇 (FlyWire) 功能域 → L1
FLYWIRE_L1 = {
    "visual": ("sensory", "strong"),
    "olfactory": ("sensory", "strong"),
    "somatosensory": ("sensory", "strong"),
    "memory": ("memory", "medium"),
    "central_complex": ("association", "medium"),
    "ascending": ("association", "weak"),
    "descending": ("motor", "medium"),
    "other": ("other", "weak"),
}


def l1_of_steinmetz(region: str) -> tuple[str, str]:
    r = str(region)
    for prefix, (dom, ev) in STEINMETZ_L1.items():
        if r.startswith(prefix):
            return dom, ev
    return "other", "weak"


# ---------------------------------------------------------- 通用: 分窗
def span_windows(n_bins: int, win: int = WIN_BINS, stride: int = STRIDE_BINS):
    """返回 [(start, end), ...]，末尾不足一窗则丢弃。"""
    out = []
    s = 0
    while s + win <= n_bins:
        out.append((s, s + win))
        s += stride
    return out


def dominant_domain(z: np.ndarray, ch_dom_idx: np.ndarray, n_dom: int):
    """窗内活动加权主导域。

    z: (n_ch, n_bins) 已 z-score
    ch_dom_idx: (n_ch,) 每通道的 L1 域索引
    返回 (domain_idx, confidence)
    """
    energy = np.abs(z).mean(axis=1)                    # (n_ch,) 每通道平均幅度
    score = np.zeros(n_dom, dtype=np.float64)
    np.add.at(score, ch_dom_idx, energy)
    tot = score.sum()
    if tot <= EPS:
        return int(np.argmax(score)), 0.0
    return int(np.argmax(score)), float(score.max() / tot)


# ------------------------------------------------------------ spike_view
def build_spike_view(src: Path, out: Path) -> dict:
    """Steinmetz → 逐 session 的 (n_win, 64, 8) 窗口。

    投影: 逐 session PCA (单元数随会话变化, 不能跨会话共享矩阵)。
    域:   每单元的 L1 域经 activity-weighted 投票得到窗标签。
    """
    out.mkdir(parents=True, exist_ok=True)
    units_all = pd.read_parquet(src / "units.parquet")

    dom2i = {d: i for i, d in enumerate(L1_DOMAINS)}
    stats, all_meta = [], []

    for sess in sorted(units_all["session"].unique()):
        f = src / f"spikes_session_{sess:03d}.npz"
        if not f.exists():
            print(f"    [skip] session {sess}: 文件缺失")
            continue
        spikes = np.load(f)["spikes"].astype(np.float32)       # (n_units, n_bins) @50ms
        u = units_all[units_all["session"] == sess].reset_index(drop=True)
        if spikes.shape[0] != len(u):
            print(f"    [skip] session {sess}: {spikes.shape[0]} 行 != {len(u)} units")
            continue

        # 1) 50ms → 100ms: 相邻两 bin 求和 (spike counts 可加)
        n_bins_100 = spikes.shape[1] // 2
        x = spikes[:, :n_bins_100 * 2].reshape(spikes.shape[0], n_bins_100, 2).sum(axis=2)

        # 2) 逐单元 z-score (时间轴)
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True) + EPS
        z = (x - mu) / sd

        # 3) 逐 session PCA → 64 维（Gram 矩阵法）
        #
        # 为什么不用 svds/SVD：
        #   完整 np.linalg.svd 对 (1000, 40000) 是 O(n²m)≈4e10 flops，且要 160MB 中间量；
        #   scipy svds(ARPACK) 在这类"宽矩阵"上收敛极慢 —— 实测 CPU 打满 8 分钟无输出。
        #
        # Gram 矩阵法（当 n_units << n_bins 时最优）：
        #   C = zc @ zc.T 是 (n_units, n_units) 小矩阵 → eigh 精确求解 → U
        #   proj = zc.T @ U[:, :k]  得到每个时间 bin 的 k 维表示
        #   代价: n_units² × n_bins ≈ 4e10 但全在 BLAS 里, 比 ARPACK 快两个数量级。
        k = min(PROJ_DIM, z.shape[0] - 1)
        zc = z - z.mean(axis=1, keepdims=True)                  # (d, bins)
        C = zc @ zc.T                                            # (d, d)
        w_eig, U = np.linalg.eigh(C)                             # 升序
        order = np.argsort(-w_eig)[:k]
        U = U[:, order]
        w_top = np.clip(w_eig[order], 0, None)
        proj = zc.T @ U                                          # (bins, k)
        comps = U.T                                              # (k, d) 成分 = 单元载荷
        total_sq = float(w_eig.sum())
        explained = float(w_top.sum() / max(total_sq, EPS))

        # 补齐/截断到 PROJ_DIM: 若 k < 64 则零填充
        if proj.shape[1] < PROJ_DIM:
            proj = np.pad(proj, ((0, 0), (0, PROJ_DIM - proj.shape[1])))
        else:
            proj = proj[:, :PROJ_DIM]

        # 4) 分窗
        wins = span_windows(n_bins_100)
        if not wins:
            print(f"    [skip] session {sess}: 仅 {n_bins_100} bins, 不足一窗")
            continue
        arr = np.stack([proj[s:e].T for s, e in wins]).astype(np.float16)   # (n_win, 64, 8)

        # 5) 域标签
        unit_l1 = np.array([dom2i[l1_of_steinmetz(r)[0]] for r in u["region"]], dtype=np.int64)
        rows = []
        for (s, e) in wins:
            zwin = z[:, s:e]
            dom, conf = dominant_domain(zwin, unit_l1, len(L1_DOMAINS))
            rows.append({"session": int(sess), "t0_bin": int(s), "domain_l1": L1_DOMAINS[dom],
                         "domain_conf": round(conf, 4),
                         "data_source": "steinmetz"})
        atomic_save_np(out / f"windows_sess{sess:03d}.npz", windows=arr)
        atomic_savez(out / f"encoder_sess{sess:03d}.npz", comps=comps.astype(np.float32),
                     mu=mu.astype(np.float32), sd=sd.astype(np.float32))
        all_meta.extend(rows)
        st = {"session": int(sess), "n_units": int(spikes.shape[0]),
              "n_bins_50ms": int(spikes.shape[1]), "n_bins_100ms": int(n_bins_100),
              "n_windows": int(len(wins)), "pca_k": int(k),
              "explained_var": round(explained, 4)}
        stats.append(st)
        print(f"    sess {sess:02d}: {spikes.shape[0]:4d} units × {n_bins_100:5d} bins "
              f"→ {len(wins):4d} 窗 (k={k}, EV={explained:.3f})")

    meta = pd.DataFrame(all_meta)
    meta.to_parquet(out / "meta.parquet", index=False)
    dist = meta.groupby("domain_l1").size().rename("n_windows").reset_index()

    return {
        "view": "spike_view", "source": "steinmetz", "species": "mus_musculus",
        "n_sessions": len(stats), "n_windows": int(len(meta)),
        "shape": f"(n_win, {PROJ_DIM}, {WIN_BINS}) float16",
        "bin_ms": BIN_MS, "win_bins": WIN_BINS,
        "domain_distribution": dist.to_dict("records"),
        "mean_confidence": round(float(meta["domain_conf"].mean()), 4) if len(meta) else 0,
        "projection": "per-session PCA on z-scored unit activity (SVD), 0-padded to 64",
        "sessions": stats,
    }


# -------------------------------------------------------------- eeg_view
def build_eeg_view(src: Path, out: Path) -> dict:
    """THINGS-EEG → (N, 64, 8)。

    投影: 63 → 64, 直接补一个零通道 (可逆、不引入旋转, 便于回溯到电极空间)
    域:   电极所属头皮区经活动加权投票
    """
    out.mkdir(parents=True, exist_ok=True)
    ch = pd.read_csv(src / "channels.csv")
    trials = pd.read_parquet(src / "trials.parquet")

    # 通道 → L1 域索引
    dom2i = {d: i for i, d in enumerate(L1_DOMAINS)}
    ch_l1 = np.array([dom2i[EEG_REGION_L1.get(r, ("other", "weak"))[0]] for r in ch["region"]],
                     dtype=np.int64)

    all_win, all_meta = [], []
    stats = []
    for sub in sorted(trials["subject"].unique()):
        for split in ("train", "val", "test"):
            f = src / f"eeg_{sub}_{split}.npz"
            if not f.exists():
                continue
            eeg = np.load(f)["eeg"].astype(np.float32)         # (N, 63, 250)
            t = trials[(trials.subject == sub) & (trials.split == split)]
            if len(t) != eeg.shape[0]:
                print(f"    [skip] {sub}/{split}: {eeg.shape[0]} vs 表 {len(t)}")
                continue

            # 1) 250 Hz (4ms) → 100ms: 25 samples/bin, 取均值
            step = int(BIN_MS / 4.0)                            # 25
            n_bins = eeg.shape[2] // step
            x = eeg[:, :, :n_bins * step].reshape(eeg.shape[0], eeg.shape[1], n_bins, step).mean(axis=3)

            # 2) 逐通道 z-score (跨 trial 时间样本统计)
            mu = x.mean(axis=(0, 2), keepdims=True)
            sd = x.std(axis=(0, 2), keepdims=True) + EPS
            z = (x - mu) / sd                                   # (N, 63, n_bins)

            # 3) 分窗 (每 trial 通常 1 窗: 10 bins, win 8, stride 4)
            wins = span_windows(n_bins)
            if not wins:
                continue
            # 4) 补零到 64 通道
            arr = np.zeros((len(wins) * eeg.shape[0], PROJ_DIM, WIN_BINS), dtype=np.float16)
            rows = []
            idx = 0
            for ti in range(eeg.shape[0]):
                for (s, e) in wins:
                    arr[idx, :63] = z[ti, :, s:e].astype(np.float16)
                    dom, conf = dominant_domain(z[ti, :, s:e], ch_l1, len(L1_DOMAINS))
                    rows.append({"subject": sub, "split": split,
                                 "trial_idx": int(t.iloc[ti]["trial_idx"]),
                                 "t0_bin": int(s), "domain_l1": L1_DOMAINS[dom],
                                 "domain_conf": round(conf, 4),
                                 "data_source": "things_eeg"})
                    idx += 1
            all_win.append(arr)
            all_meta.extend(rows)
            stats.append({"subject": sub, "split": split, "n_trials": int(eeg.shape[0]),
                          "n_windows": int(len(wins) * eeg.shape[0]),
                          "n_bins_100ms": int(n_bins)})
            print(f"    {sub}/{split}: {eeg.shape[0]:5d} trials × {n_bins} bins "
                  f"→ {len(wins)*eeg.shape[0]:5d} 窗")

    if not all_win:
        raise RuntimeError("eeg_view 无数据")

    windows = np.concatenate(all_win, axis=0)
    atomic_save_np(out / "windows.npz", windows=windows)
    meta = pd.DataFrame(all_meta)
    meta.to_parquet(out / "meta.parquet", index=False)

    # 投影矩阵: 63 → 64 的补零嵌入 (显式保存, 可逆)
    P = np.zeros((PROJ_DIM, len(ch)), dtype=np.float32)
    P[:len(ch), :] = np.eye(len(ch))
    atomic_savez(out / "encoder.npz", P=P,
                 ch_l1=ch_l1, ch_names=np.array(ch["channel"].tolist()))

    dist = meta.groupby("domain_l1").size().rename("n_windows").reset_index()
    return {
        "view": "eeg_view", "source": "things_eeg", "species": "homo_sapiens",
        "n_windows": int(len(windows)),
        "shape": f"({len(windows)}, {PROJ_DIM}, {WIN_BINS}) float16",
        "bin_ms": BIN_MS, "win_bins": WIN_BINS,
        "domain_distribution": dist.to_dict("records"),
        "mean_confidence": round(float(meta["domain_conf"].mean()), 4),
        "projection": "63→64 zero-pad identity embedding (invertible, no rotation)",
        "subjects": sorted(meta["subject"].unique().tolist()),
        "splits": sorted(meta["split"].unique().tolist()),
        "per_group": stats,
    }


# ------------------------------------------------------------ graph_view
def build_graph_view(src: Path, out: Path, k: int = PROJ_DIM, n_prop: int = 8) -> dict:
    """FlyWire → 每神经元的连通性嵌入 (k 维)，**不用 SVD**。

    ## 为什么放弃谱嵌入
    107,265 个神经元。`scipy.svds(An, k=64)` 走 ARPACK, 每次迭代要 O(E) 的
    稀疏矩阵乘, 且需要多次重正交化 —— 实测在 cpu-upgrade 上跑 20 分钟无输出。
    本项目的目标是"给每个神经元一个与拓扑相关、但与域标签不直接同源的表示",
    并不需要精确的主成分。

    ## 采用的替代: 图信号传播 (graph signal propagation)
    对每个通道 c 构造一个随机初始信号 s_c ∈ R^M, 在归一化邻接上迭代传播:
        s ← α·Â·s + (1−α)·s₀        (α=0.85, 类似 PageRank 的阻尼)
    迭代 n_prop 次后, [s_1, ..., s_k] 就是 k 维嵌入。
    成本: n_prop × k 次稀疏矩阵-向量乘 = 8 × 64 = 512 次 O(E),
          E ≈ 3000 万 → 约 1.5e10 次乘加, 在 8 线程下秒级完成。

    ## 与域标签的关系
    嵌入只依赖**拓扑与随机种子**, 不读任何解剖/功能标签。因此
    "用嵌入预测域" 是一个有意义的实验 (不像递质加权特征那样同义反复)。
    """

    from scipy.sparse import coo_matrix, eye as sp_eye

    out.mkdir(parents=True, exist_ok=True)
    nodes_f = src / "node_features.parquet"
    if not nodes_f.exists():
        nodes_f = src / "nodes.parquet"
    nodes = pd.read_parquet(nodes_f)

    if "functional_domain" not in nodes.columns:
        cc = nodes["cell_class"].astype("string")
        nodes["region"] = cc.str.split(">").str[0].str.split(".").str[0]
        nodes["functional_domain"] = nodes["region"].map(
            lambda r: FLYWIRE_REGION_TO_FUNC.get(str(r), "other"))
    nodes = nodes[nodes["functional_domain"].notna()].reset_index(drop=True)

    dom2i = {d: i for i, d in enumerate(L1_DOMAINS)}
    node_l1 = np.array([dom2i[FLYWIRE_L1.get(r, ("other", "weak"))[0]]
                        for r in nodes["functional_domain"]], dtype=np.int64)

    conn_f = src / "connections.parquet"
    if not conn_f.exists():
        raise FileNotFoundError(f"缺少 {conn_f}")
    conn = pd.read_parquet(conn_f, columns=["pre", "post", "syn_count"])
    conn = conn[conn["syn_count"] > 0]

    ids = nodes["root_id"].to_numpy()
    M = len(ids)
    order_ids = np.argsort(ids)
    sorted_ids = ids[order_ids]

    def remap(vals):
        pos = np.clip(np.searchsorted(sorted_ids, vals), 0, M - 1)
        return order_ids[pos], sorted_ids[pos] == vals

    pre_idx, pre_ok = remap(conn["pre"].to_numpy())
    post_idx, post_ok = remap(conn["post"].to_numpy())
    keep = pre_ok & post_ok
    r, c = pre_idx[keep], post_idx[keep]
    w = conn["syn_count"].to_numpy(dtype=np.float32)[keep]
    print(f"    连接 {len(conn):,} → 保留 {int(keep.sum()):,} 条 (两端都有域标签)")

    A = coo_matrix((w, (r, c)), shape=(M, M)).tocsr()
    A = A + A.T
    deg = np.asarray(A.sum(axis=1)).ravel() + EPS
    Dinv = coo_matrix((1.0 / np.sqrt(deg), (np.arange(M), np.arange(M))), shape=(M, M)).tocsr()
    Ah = (Dinv @ A @ Dinv).tocsr()

    rng = np.random.default_rng(SEED)
    alpha = 0.85
    emb = np.zeros((M, k), dtype=np.float32)
    print(f"    图信号传播: M={M:,}, k={k}, iters={n_prop}, alpha={alpha}")
    t0 = time.time()
    for j in range(k):
        s0 = rng.standard_normal(M).astype(np.float32)
        s0 /= (np.linalg.norm(s0) + EPS)
        s = s0.copy()
        for _ in range(n_prop):
            s = alpha * (Ah @ s) + (1 - alpha) * s0
        emb[:, j] = s
    print(f"    传播完成 {time.time()-t0:.1f}s")

    # 逐维 z-score (与另两个视图的归一化一致)
    mu = emb.mean(axis=0, keepdims=True)
    sd = emb.std(axis=0, keepdims=True) + EPS
    emb = ((emb - mu) / sd).astype(np.float32)

    atomic_save_np(out / "embeds.npz", embeds=emb)
    meta = nodes[["root_id", "region", "functional_domain"]].copy()
    meta["domain_l1"] = np.array(L1_DOMAINS, dtype=object)[node_l1]
    meta.to_parquet(out / "meta.parquet", index=False)

    dist = meta.groupby("domain_l1").size().rename("n_neurons").reset_index()
    return {
        "view": "graph_view", "source": "flywire", "species": "drosophila",
        "n_nodes": int(M), "n_edges_used": int(keep.sum()),
        "embedding_dim": int(k),
        "domain_distribution": dist.to_dict("records"),
        "projection": (f"graph signal propagation on sym-normalized adjacency "
                       f"({n_prop} iters, alpha={alpha}), per-dim z-score"),
        "note": ("嵌入仅依赖连接拓扑与随机种子, 不读解剖/功能标签; "
                 "故「用嵌入预测域」是有意义的实验 (区别于递质加权特征)。"),
    }


# ------------------------------------------------------------------ main
def _resolve_flywire(data_root: Path) -> Path:
    """定位 FlyWire 目录 (含 node_features.parquet 与 connections.parquet)。

    优先级: BD_FLYWIRE 环境变量 > 若干约定路径。每个候选都会打印检查结果,
    避免"看起来找到了但其实是空目录"这种静默错误。
    """
    env = os.environ.get("BD_FLYWIRE", "").strip()
    cands = []
    if env:
        cands.append(Path(env))
    cands += [
        data_root / "flywire",
        data_root.parent / "connectome" / "flywire-connectome",
        data_root / "connectome" / "flywire-connectome",
    ]
    print("  FlyWire 候选路径检查:")
    for c in cands:
        has_nodes = (c / "node_features.parquet").exists() or (c / "nodes.parquet").exists()
        has_conn = (c / "connections.parquet").exists()
        print(f"    {c}: nodes={has_nodes} conn={has_conn}")
        if has_nodes and has_conn:
            print(f"    → 采用 {c}")
            return c
    raise FileNotFoundError(
        "未找到完整的 FlyWire 数据 (需 node_features.parquet/nodes.parquet + connections.parquet)。"
        "已尝试:\n  " + "\n  ".join(f"{c}" for c in cands))


def build_ontology() -> pd.DataFrame:
    rows = []
    for prefix, (l1, ev) in STEINMETZ_L1.items():
        rows.append({"source": "steinmetz", "species": "mus_musculus",
                     "raw": prefix, "l1": l1, "evidence": ev,
                     "basis": "Allen CCF 解剖名称 → 功能系统"})
    for region, (l1, ev) in EEG_REGION_L1.items():
        rows.append({"source": "things_eeg", "species": "homo_sapiens",
                     "raw": region, "l1": l1, "evidence": ev,
                     "basis": "10-10 电极位置 → 头皮区 (非皮层功能)"})
    for region, (l1, ev) in FLYWIRE_L1.items():
        rows.append({"source": "flywire", "species": "drosophila",
                     "raw": region, "l1": l1, "evidence": ev,
                     "basis": "果蝇功能系统命名"})
    return pd.DataFrame(rows)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    data_root, out_root = Path(sys.argv[1]), Path(sys.argv[2])
    out_root.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("=" * 72)
    print(f"OmniBrain 构建 | bin={BIN_MS}ms win={WIN_BINS}bins dim={PROJ_DIM}")
    print(f"数据 {data_root} → 输出 {out_root}")
    print("=" * 72)

    manifest = {"schema_version": "1.0", "bin_ms": BIN_MS, "win_bins": WIN_BINS,
                "stride_bins": STRIDE_BINS, "proj_dim": PROJ_DIM, "seed": SEED,
                "l1_domains": L1_DOMAINS, "views": {}}

    # 本体表
    ont = build_ontology()
    ont.to_csv(out_root / "ontology.csv", index=False)
    print(f"\n本体: {len(ont)} 条映射 "
          f"(strong {(ont.evidence=='strong').sum()} / "
          f"medium {(ont.evidence=='medium').sum()} / "
          f"weak {(ont.evidence=='weak').sum()})")

    stages = [
        ("spike_view", lambda: build_spike_view(data_root / "steinmetz", out_root / "spike_view")),
        ("eeg_view", lambda: build_eeg_view(data_root / "primate/human_eeg", out_root / "eeg_view")),
        ("graph_view", lambda: build_graph_view(_resolve_flywire(data_root), out_root / "graph_view")),
    ]
    for name, fn in stages:
        print(f"\n--- {name} ---")
        if os.environ.get(f"SKIP_{name.upper()}") == "1":
            print("  (跳过)")
            continue
        try:
            manifest["views"][name] = fn()
        except Exception as e:
            import traceback; traceback.print_exc()
            manifest["views"][name] = {"error": f"{type(e).__name__}: {e}"}

    wall = time.time() - t0
    manifest["resources"] = {"wall_s": round(wall, 1),
                             "peak_rss_gb": round(peak_rss_gb(), 2)}
    manifest["known_limitations"] = [
        "EEG 与 spike 分属不同视图，未假设二者同质；跨视图迁移是研究对象而非前提",
        "跨物种 L1 映射为人工标注，证据等级见 ontology.csv",
        "NLB 猕猴数据集未纳入（上游 bin 宽未核实）",
        "per-domain 去均值默认关闭；开启会削弱域可分性",
        f"窗长 {WIN_BINS} bins 是 EEG(10 bins) 与 spike(数百 bins) 的折中，非理想对齐",
    ]
    (out_root / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72)
    for k, v in manifest["views"].items():
        if "error" in v:
            print(f"  ❌ {k}: {v['error'][:100]}")
        else:
            n = v.get("n_windows") or v.get("n_nodes")
            print(f"  ✅ {k}: {n:,} 样本, {len(v.get('domain_distribution', []))} 个 L1 域")
    print(f"墙钟 {wall:.1f}s | 峰值 RSS {peak_rss_gb():.1f}GB")
    print(f"完成 → {out_root}")


if __name__ == "__main__":
    main()
