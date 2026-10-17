"""构建灵长类脑域数据集 (v2 扩展) — 人类 EEG + 猕猴皮层内记录。

## A. human_eeg — THINGS-EEG（人类，10 被试）

来源: 桶内 `eeg/things-eeg/Preprocessed_data_250Hz_whiten/sub-0X/{train,test}.pt`
  每个 .pt 是 dict: eeg (N, 4 reps, 63 ch, 250 t), label, img, text, session, ch_names, times
  采样率 250 Hz, 时间窗 -0.2 → 1.0 s（基线 + 诱发）

**域定义**: 电极按 10-10 布局归并为 5 个头皮区域
  frontal(15) / central(14) / temporal(10) / parietal(16) / occipital(8)

⚠️ **头皮 ≠ 皮层**: EEG 电极测的是头皮电位，区域归属是**解剖投影的粗近似**，
不是像 Steinmetz 那样由记录位点直接确定的脑区。任何「域结构」结论必须
带上这一限定；域效应可能反映的是容积传导与参考电极几何，而非皮层功能分化。

**切分**: THINGS-EEG 的 `eeg` 形状是 `(n_trials, 4, 63, 250)`，第二维 **就是 session 轴**
（`session` 字段每行恒为 `[0,1,2,3]`，同一张图在 4 个 session 里各呈现一次）。
本数据集保留 **session 0（训练）与 session 3（测试）**，按被试拆成两个 npz，
天然形成 session-held-out 切分，杜绝随机划分泄漏。

**采样**: 每被试随机保留 SAMPLE_TRIALS 张图（种子固定）× 2 个 session，
float16 存储。上游全量 16,540 图 × 4 session 可按需回到桶内原始数据。

## B. macaque_nlb — 猕猴皮层内 spike（NLB 系）

来源: `eminorhan/{area2-bump, dmfc-rsg, churchland}`
  spike_counts 是「每 unit 一个变长数组」的列表（非矩阵），需补齐为 padded 矩阵

**域定义**: 按数据集对应的记录区域（文献确定）
  area2-bump → 初级躯体感觉 + 初级运动 (S1/M1, 混合)
  dmfc-rsg   → 背内侧前额叶 (DMFC)
  churchland → 运动/前运动皮层 (M1/PMd)

⚠️ **域标签是数据集级而非神经元级**: 上游不提供逐 unit 的区域标注，
area2-bump 甚至混合了 S1 与 M1。因此这里的 domain 实际上是
「记录区域 × 实验范式 × 个体」的混合因子，**不能**解释为纯粹的皮层区域效应。
本数据集定位为「跨记录位点泛化」的对照数据，不是主实验材料。

## 输出

  primate/
  ├── human_eeg/
  │   ├── eeg_sub-01.npz …                    (n_trials, 63, 250) float16
  │   ├── channels.csv                        channel, region, domain_id
  │   ├── trials.parquet                      subject, trial_idx, session, label, text
  │   └── meta.json                           被试/试次/切分说明
  └── macaque_nlb/
      ├── spikes_<dataset>_<session>.npz      (n_units, n_bins) float16 (零填充)
      ├── units.parquet                       dataset, session, unit_idx, area, domain_id
      └── meta.json                           区域归属与混淆说明
"""

from __future__ import annotations

import gc
import json
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_TRIALS = 1500          # 每被试保留的训练图数 (train/val 各一份)
SEED = 0
COMPRESS = False              # False = np.savez (快, 体积 1.5-2x); True = zlib 压缩 (慢, 单线程)
WORKERS = int(os.environ.get("BD_WORKERS", "0"))   # 0 = 自动取 min(8, 任务数)


def _saver(out: Path, name: str, **arrays):
    """统一保存入口。压缩是单线程 CPU 瓶颈, 默认关闭。

    注意 np.savez 会自动补 ".npz" 后缀 —— 原子写入时不能直接传 ".tmp" 结尾的
    路径, 否则实际落盘成 "xxx.npz.tmp.npz"。原子写入改用 _saver_fh。
    """
    if COMPRESS:
        np.savez_compressed(out / name, **arrays)
    else:
        np.savez(out / name, **arrays)


def _saver_fh(path: Path, **arrays):
    """写到显式路径（用文件句柄, 避免 np.savez 自动加后缀）。"""
    with open(path, "wb") as f:
        if COMPRESS:
            np.savez_compressed(f, **arrays)
        else:
            np.savez(f, **arrays)


def peak_rss_gb() -> float:
    """本进程峰值常驻内存 (GB)。"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def cpu_time_s() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime

# 10-10 电极前缀 → 头皮区域。长前缀必须先匹配 (AF/FC/FT/CP/TP/PO)
REGION_PREFIXES = [
    ("AF", "frontal"), ("FC", "central"), ("FT", "temporal"),
    ("CP", "parietal"), ("TP", "temporal"), ("PO", "occipital"),
    ("Fp", "frontal"), ("F", "frontal"),
    ("C", "central"), ("T", "temporal"),
    ("P", "parietal"), ("O", "occipital"),
]

REGION_ORDER = ["frontal", "central", "temporal", "parietal", "occipital"]

# NLB 数据集 → 记录区域（文献确定，供 domain_id 使用）
# ⚠️ 2026-10-17 更正: area2-bump 原标注 "S1_M1 混合" 是错的。
#    NLB 官方 (neurallatents.github.io/datasets.html):
#      Area2_Bump — Brodmann's area 2 of the somatosensory cortex
#      DMFC_RSG   — dorsomedial frontal cortex (ready-set-go 计时任务)
#      Churchland — M1/PMd  (dandi:000070)
NLB_AREAS = {
    "area2-bump": {"area": "Area2", "desc": "Brodmann 区 2（躯体感觉皮层）",
                   "species": "macaca_mulatta", "l1": "sensory"},
    "dmfc-rsg": {"area": "DMFC", "desc": "背内侧额叶皮层（计时任务）",
                 "species": "macaca_mulatta", "l1": "association"},
    "churchland": {"area": "M1_PMd", "desc": "初级运动 / 背侧前运动皮层",
                   "species": "macaca_mulatta", "l1": "motor"},
}

NLB_FILES = {
    "area2-bump": "data/train-00000-of-00001.parquet",
    "dmfc-rsg": "data/train-00000-of-00001.parquet",
    "churchland": "data/train-00000-of-00002.parquet",
}


def classify_electrode(name: str) -> str:
    for prefix, region in REGION_PREFIXES:
        if name.startswith(prefix):
            return region
    return "other"


def _to_np(x):
    """torch tensor / numpy 数组 / list 统一转 numpy。"""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ------------------------------------------------------------ 人类 EEG
def _process_one_subject(args):
    """单个被试的完整处理 (供 ProcessPoolExecutor 调用)。

    返回 (subject, stats, trial_rows, ch_names, timing, peak_rss_gb)。
    每个 worker 独立进程: torch.load 的 2.1GB 不跨任务累积。
    """
    import torch

    sub, data_root_s = args
    data_root = Path(data_root_s)
    t0 = time.time()

    f_tr = data_root / sub / "train.pt"
    f_te = data_root / sub / "test.pt"
    d_tr = torch.load(f_tr, map_location="cpu", weights_only=False)
    ch_names = [str(c) for c in d_tr["ch_names"]]

    eeg_tr = d_tr["eeg"]
    eeg_tr = eeg_tr if isinstance(eeg_tr, np.ndarray) else np.asarray(eeg_tr)
    lab_tr = np.asarray(d_tr["label"])
    txt_tr = np.asarray(d_tr["text"])
    n_img = eeg_tr.shape[0]

    rng = np.random.default_rng(SEED + hash(sub) % 10000)
    take = min(SAMPLE_TRIALS, n_img)
    sel = np.sort(rng.choice(n_img, size=take, replace=False))

    rows = []
    # train: rep 0（session 0）
    arr_tr = np.ascontiguousarray(eeg_tr[sel, 0])
    for k, j in enumerate(sel):
        rows.append({"subject": sub, "split": "train", "session": 0,
                     "trial_idx": int(k), "src_index": int(j),
                     "label": str(lab_tr[j][0]), "text": str(txt_tr[j][0])})

    # val: rep 3（session 3，同批图）
    arr_va = np.ascontiguousarray(eeg_tr[sel, 3])
    for k, j in enumerate(sel):
        rows.append({"subject": sub, "split": "val", "session": 3,
                     "trial_idx": int(k), "src_index": int(j),
                     "label": str(lab_tr[j][3]), "text": str(txt_tr[j][3])})
    del eeg_tr, d_tr
    gc.collect()

    # test: 80 rep 均值（session 3）
    n_test = 0
    if f_te.exists():
        d_te = torch.load(f_te, map_location="cpu", weights_only=False)
        eeg_te = d_te["eeg"]
        eeg_te = eeg_te if isinstance(eeg_te, np.ndarray) else np.asarray(eeg_te)
        lab_te = np.asarray(d_te["label"])
        txt_te = np.asarray(d_te["text"])
        # 分块求均值，避免 float32 中间量占 ~1GB
        n, reps = eeg_te.shape[0], eeg_te.shape[1]
        out_arr = np.empty((n, eeg_te.shape[2], eeg_te.shape[3]), dtype=np.float16)
        CHUNK = 50
        for s in range(0, n, CHUNK):
            e = min(s + CHUNK, n)
            out_arr[s:e] = eeg_te[s:e].astype(np.float32).mean(axis=1).astype(np.float16)
            del s, e
        for k in range(n):
            lab = lab_te[k][0] if lab_te.ndim > 1 else lab_te[k]
            txt = txt_te[k][0] if txt_te.ndim > 1 else txt_te[k]
            rows.append({"subject": sub, "split": "test", "session": 3,
                         "trial_idx": int(k), "src_index": int(k),
                         "label": str(lab), "text": str(txt)})
        n_test = n
        arr_te = out_arr
        del eeg_te, d_te
        gc.collect()
    else:
        arr_te = None

    stats = {"subject": sub, "n_train_rows": take, "n_val_rows": take,
             "n_test_rows": n_test, "n_images_total": int(n_img),
             "proc_sec": round(time.time() - t0, 1),
             "peak_rss_gb": round(peak_rss_gb(), 2)}
    return sub, stats, rows, ch_names, arr_tr, arr_va, arr_te


def build_human_eeg(data_root: Path, out: Path, only=None) -> dict:
    """THINGS-EEG 构建 —— **严格串行**, 每被试跑完立刻落盘。

    为什么串行 (踩坑记录):
      上一版用 8 进程并行, 输出直接写共享目录, 结果 sub-05/06/08/09 的
      train/val 文件字节数完全相同 —— 并发写冲突, 数据损坏。
      而 trials.parquet 恰好只记下那 4 个坏被试。整个子集不可用。

      并行在这里也没有收益: 实测 CPU 利用率 0.6% (11s CPU / 1787s 墙钟),
      瓶颈是 26GB 的网络下载, 不是计算。多进程只是把 IO 竞争放大。

    本版约束:
      - 单进程, 一次一个被试
      - 写入用 <file>.tmp 再 os.replace 原子落盘, 中途失败不留半成品
      - 每被试完成即把该被试的 rows 追加进内存表, 结尾统一写 trials.parquet
      - only: 只处理指定被试 (增量补齐时用)

    返回 meta dict。
    """
    out.mkdir(parents=True, exist_ok=True)
    import torch

    subjects = sorted([p.name for p in data_root.glob("sub-*") if p.is_dir()])
    if not subjects:
        raise FileNotFoundError(f"在 {data_root} 下未找到 sub-* 目录")
    if only:
        subjects = [s for s in subjects if s in set(only)]
        if not subjects:
            raise ValueError(f"only={only} 未匹配任何被试")

    print(f"  被试 {len(subjects)} 个 | 串行 | 压缩 {COMPRESS}")

    t_start = time.time()
    cpu_start = cpu_time_s()
    all_rows, stats, ch_names = [], [], None

    for i, s in enumerate(subjects, 1):
        try:
            sub, st, rows, chn, arr_tr, arr_va, arr_te = _process_one_subject(
                (s, str(data_root)))
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ❌ [{i}/{len(subjects)}] {s}: {type(e).__name__}: {e}")
            continue

        # 原子落盘: 先写 .tmp 再 rename
        for suffix, arr in (("train", arr_tr), ("val", arr_va), ("test", arr_te)):
            if arr is None:
                continue
            target = out / f"eeg_{sub}_{suffix}.npz"
            tmp = target.with_name(target.name + ".tmp")
            _saver_fh(tmp, eeg=arr)
            os.replace(tmp, target)
        del arr_tr, arr_va, arr_te
        gc.collect()

        stats.append(st)
        all_rows.extend(rows)
        if ch_names is None:
            ch_names = chn
        print(f"  [{i}/{len(subjects)}] {sub}: train {st['n_train_rows']} + "
              f"val {st['n_val_rows']} + test {st['n_test_rows']} | "
              f"{st['proc_sec']}s | RSS {st['peak_rss_gb']}GB")

    wall = time.time() - t_start
    cpu = cpu_time_s() - cpu_start
    stats.sort(key=lambda d: d["subject"])

    # 通道表
    ch_rows = []
    for i, name in enumerate(ch_names or []):
        ch_rows.append({"channel_idx": i, "channel": name,
                        "region": classify_electrode(name)})
    ch = pd.DataFrame(ch_rows)
    doms = [r for r in REGION_ORDER if r in set(ch["region"])]
    d2i = {d: i for i, d in enumerate(doms)}
    ch["domain_id"] = ch["region"].map(d2i)
    ch.to_csv(out / "channels.csv", index=False)

    # 与已有 trials 合并 (增量补齐场景)
    tp = out / "trials.parquet"
    trials_new = pd.DataFrame(all_rows)
    if tp.exists():
        old = pd.read_parquet(tp)
        done = set(trials_new["subject"].unique())
        old = old[~old["subject"].isin(done)]     # 新数据覆盖旧的
        trials = pd.concat([old, trials_new], ignore_index=True)
    else:
        trials = trials_new
    trials = trials.sort_values(["subject", "split", "trial_idx"]).reset_index(drop=True)
    trials.to_parquet(tp, index=False)

    # 一致性: npz 行数 == trials 行数
    print("\n  一致性校验 (npz 行数 vs trials 行数):")
    bad = []
    for sub in sorted(trials["subject"].unique()):
        for sp in ("train", "val", "test"):
            f = out / f"eeg_{sub}_{sp}.npz"
            n_tab = len(trials[(trials.subject == sub) & (trials.split == sp)])
            if not f.exists():
                if n_tab: bad.append(f"{sub}/{sp}: 表有 {n_tab} 行但文件缺失")
                continue
            n_file = np.load(f)["eeg"].shape[0]
            if n_file != n_tab:
                bad.append(f"{sub}/{sp}: 文件 {n_file} vs 表 {n_tab}")
    if bad:
        for b in bad: print(f"    ❌ {b}")
    else:
        print(f"    ✅ 全部 {len(trials['subject'].unique())} 被试 × 3 切分一致")

    dist = ch.groupby(["domain_id", "region"]).size().rename("n_channels").reset_index()
    split_counts = trials.groupby("split").size().rename("n_rows").reset_index()

    meta = {
        "name": "human_eeg_things_v1",
        "species": "homo_sapiens",
        "source": "桶内 eeg/things-eeg (THINGS-EEG2, 250 Hz 白化预处理)",
        "n_subjects": int(trials["subject"].nunique()),
        "subjects": sorted(trials["subject"].unique().tolist()),
        "n_rows_total": int(len(trials)),
        "split_counts": split_counts.to_dict("records"),
        "n_channels": len(ch),
        "n_domains": len(doms),
        "domains": doms,
        "domain_distribution": dist.to_dict("records"),
        "shape": "[n_trials, 63, 250] float16; 每被试三个文件 eeg_<sub>_{train,val,test}.npz",
        "split": ("train = train 图 × rep0 (session 0); "
                  "val = 同一批 train 图 × rep3 (session 3, 测 session 偏移); "
                  "test = test 图 × 80rep 均值 (session 3, 图像训练时未见过)。"),
        "caveat": ("EEG 是头皮电位, 区域归属为解剖投影的粗近似, 非皮层记录位点。"),
        "build": {"mode": "serial", "compress": COMPRESS,
                  "atomic_write": True, "resume_supported": True},
        "resources": {"wall_s": round(wall, 1), "cpu_s": round(cpu, 1),
                      "cpu_util_pct": round(100 * cpu / wall, 1) if wall else 0,
                      "peak_rss_gb": round(peak_rss_gb(), 2)},
        "per_subject": stats,
        "consistency_errors": bad,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[human_eeg] {meta['n_subjects']} 被试, {len(trials):,} 行, {len(ch)} 通道")
    print(f"  墙钟 {wall:.1f}s | CPU {cpu:.1f}s | 峰值 RSS {peak_rss_gb():.1f}GB")
    print(split_counts.to_string(index=False))
    return meta


# ---------------------------------------------------------- 猕猴 NLB
def _pad_units(sc, dtype=np.float16):
    """把「每 unit 一个变长数组」补零成 (n_units, max_len) 矩阵。"""
    arrs = [np.asarray(u, dtype=np.float32).ravel() for u in sc]
    n = len(arrs)
    if n == 0:
        return np.zeros((0, 0), dtype=dtype), [0]
    L = max(len(a) for a in arrs)
    out = np.zeros((n, L), dtype=dtype)
    for i, a in enumerate(arrs):
        out[i, :len(a)] = a
    return out, [len(a) for a in arrs]


def build_macaque(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    areas = sorted({v["area"] for v in NLB_AREAS.values()})
    a2i = {a: i for i, a in enumerate(areas)}

    unit_rows = []      # 每个 (dataset, session, unit_idx) 一行 —— 去重
    seg_rows = []       # 每个片段文件一行
    stats = []
    for ds, area_info in NLB_AREAS.items():
        f = NLB_FILES.get(ds)
        if not f:
            print(f"  [skip] {ds}: 未配置文件路径")
            continue
        try:
            p = hf_hub_download(f"eminorhan/{ds}", f, repo_type="dataset")
            df = pd.read_parquet(p)
        except Exception as e:
            print(f"  [skip] {ds}: {e}")
            continue

        seen_units = set()          # (session, unit_idx)
        for i, r in df.iterrows():
            mat, lens = _pad_units(r["spike_counts"])
            if mat.size == 0:
                continue
            sess = str(r["session_id"]).replace("/", "_")
            fn = f"spikes_{ds}_{i:02d}.npz"
            _saver(out, fn, spikes=mat)

            seg_rows.append({"dataset": ds, "file": fn, "session": sess,
                             "n_units": int(mat.shape[0]), "n_bins": int(mat.shape[1]),
                             "bins_per_unit_min": int(min(lens)),
                             "bins_per_unit_max": int(max(lens)),
                             "area": area_info["area"]})
            # 同一 session 的同一 unit 在多个片段里重复出现, 只记一次
            for u in range(mat.shape[0]):
                key = (sess, u)
                if key in seen_units:
                    continue
                seen_units.add(key)
                unit_rows.append({"dataset": ds, "session": sess, "unit_idx": u,
                                  "area": area_info["area"],
                                  "species": area_info["species"],
                                  "domain_id": a2i[area_info["area"]]})

            stats.append({"dataset": ds, "file": fn, "session": sess,
                          "n_units": int(mat.shape[0]), "n_bins": int(mat.shape[1]),
                          "area": area_info["area"]})
            if i < 3 or i % 20 == 0:
                print(f"  {ds}[{i}]: {mat.shape} units×bins, session={sess[:40]}")

    units = pd.DataFrame(unit_rows)
    segs = pd.DataFrame(seg_rows)
    units.to_parquet(out / "units.parquet", index=False)
    segs.to_parquet(out / "segments.parquet", index=False)

    dist = (units.groupby(["domain_id", "area"]).size().rename("n_units").reset_index()
            if len(units) else pd.DataFrame())
    seg_stats = (segs.groupby(["dataset", "area"]).agg(
        n_segments=("file", "size"), n_sessions=("session", "nunique"),
        n_bins_total=("n_bins", "sum")).reset_index() if len(segs) else pd.DataFrame())

    meta = {
        "name": "macaque_nlb_v1",
        "species": "macaca_mulatta",
        "sources": [f"eminorhan/{k}" for k in NLB_AREAS],
        "n_segments": len(segs),
        "n_unique_units": int(len(units)),
        "n_sessions": int(units["session"].nunique()) if len(units) else 0,
        "n_domains": len(areas),
        "domains": areas,
        "domain_distribution": dist.to_dict("records") if len(dist) else [],
        "segment_stats": seg_stats.to_dict("records") if len(seg_stats) else [],
        "shape": "[n_units, n_bins] float16, 变长序列补零; 每片段一个 npz",
        "counting_note": ("units.parquet 按 (dataset, session, unit_idx) 去重 —— "
                          "上游同一 session 的同一批神经元会出现在多个片段文件里, "
                          "按文件累加会把 unit 数严重高估 (churchland: 67 片段 × 192 = 12864, "
                          "实际仅 6 session × 192 = 1152)。"),
        "caveat": ("域标签为**数据集级**而非神经元级: 上游不提供逐 unit 区域标注, "
                   "area2-bump 混合了 S1 与 M1。因此 domain 实际是"
                   "「记录区域 × 实验范式 × 个体」的混合因子, 不能解释为纯皮层区域效应。"
                   "定位为跨记录位点泛化的对照数据, 不作主实验材料。"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[macaque_nlb] {len(segs)} 片段, {len(units)} 唯一 units, "
          f"{meta['n_sessions']} sessions, {len(areas)} 域")
    if len(dist):
        print(dist.to_string(index=False))
    return meta


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    data_root, out_root = Path(sys.argv[1]), Path(sys.argv[2])
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("灵长类脑域数据集构建 v2")
    print(f"进程数 {WORKERS or 'auto'} | 压缩 {COMPRESS} | 采样 {SAMPLE_TRIALS}/被试")
    print("=" * 70)

    t0 = time.time()
    c0 = cpu_time_s()
    manifest = {"schema_version": "1.0", "built_from": str(data_root),
                "sample_trials_per_subject": SAMPLE_TRIALS, "seed": SEED,
                "compress": COMPRESS, "workers": WORKERS, "datasets": {}}

    print("\n--- A. human_eeg (THINGS-EEG, 人类) ---")
    if os.environ.get("BD_SKIP_HUMAN") != "1":
        try:
            only_env = os.environ.get("BD_ONLY", "").strip()
            only = [s.strip() for s in only_env.split(",") if s.strip()] or None
            if only:
                print(f"  增量模式: 只处理 {only}")
            manifest["datasets"]["human_eeg"] = build_human_eeg(
                data_root / "eeg/things-eeg/Preprocessed_data_250Hz_whiten",
                out_root / "human_eeg", only=only)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ❌ {type(e).__name__}: {e}")
    else:
        print("  (跳过: BD_SKIP_HUMAN=1)")

    print("\n--- B. macaque_nlb (猕猴, NLB) ---")
    if os.environ.get("BD_SKIP_MACAQUE") != "1":
        try:
            manifest["datasets"]["macaque_nlb"] = build_macaque(out_root / "macaque_nlb")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ❌ {type(e).__name__}: {e}")

    wall = time.time() - t0
    cpu = cpu_time_s() - c0
    manifest["resources"] = {"wall_s": round(wall, 1), "cpu_s": round(cpu, 1),
                             "cpu_util_pct": round(100 * cpu / wall, 1) if wall else 0,
                             "peak_rss_gb": round(peak_rss_gb(), 2),
                             "workers": WORKERS, "compress": COMPRESS}
    (out_root / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 70)
    for k, v in manifest["datasets"].items():
        print(f"  {k}: {v.get('n_domains')} domains")
    print(f"总墙钟 {wall:.1f}s | CPU {cpu:.1f}s | 利用率 {100*cpu/wall:.0f}% | "
          f"峰值 RSS {peak_rss_gb():.1f}GB")
    print(f"完成 → {out_root}")
    print("=" * 70)


if __name__ == "__main__":
    main()
