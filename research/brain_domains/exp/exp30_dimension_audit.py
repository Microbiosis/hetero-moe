"""Phase 30: 维度审计 —— 原始数据到底有哪些轴？我丢掉了哪些？

## 为什么做这个

用户指出：整个项目一直在把数据压成二维（样本 × 特征）。
这是对的。证据在代码里：

    w = np.load(f)["windows"]      # (n_win, 64, 8)
    X = w.mean(axis=2).T           # (64, n_win)   ← 时间轴被平均

几乎每个实验都做了这一步。

## 神经数据本应有的维度

    1. 单元/通道   space      （已用，64）
    2. 时间       time       （被平均掉，只剩 8 bin 的均值）
    3. 频率       frequency  （完全没用）
    4. 试次/条件   trial      （完全没用）
    5. 层/深度     layer      （未用）
    6. 被试        subject    （只当分组键）

## 本脚本要回答

对每个数据源，逐个检查：
  - 原始文件里有哪些轴？
  - 采样率支持到多高的频率（Nyquist）？
  - 试次结构还在吗？还是已经被上游拼成连续序列？

这决定了"三维/四维数据集"在物理上是否可能，还是只能造假。

## 诚实原则

若某维度在上游数据里已经不存在（如 neural-pile 把 trials 拼成了连续序列），
就如实报告"不可恢复"，不假装构造出来。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

DATA = Path("/data")


def probe_npz(p: Path, keys=None, max_show=6):
    try:
        with np.load(p) as z:
            info = {}
            for k in (keys or z.files)[:max_show]:
                a = z[k]
                info[k] = {"shape": list(a.shape), "dtype": str(a.dtype)}
            return info
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:100]}"}


def probe_parquet(p: Path):
    try:
        df = pd.read_parquet(p)
        out = {"n_rows": int(len(df)), "columns": list(df.columns)}
        # 每列的类型与示例
        sample = {}
        for c in df.columns[:12]:
            v = df[c].iloc[0] if len(df) else None
            if isinstance(v, np.ndarray):
                sample[c] = f"ndarray{list(v.shape)}"
            elif isinstance(v, (list, tuple)):
                sample[c] = f"{type(v).__name__}[{len(v)}]"
            else:
                sample[c] = type(v).__name__
        out["dtypes_sample"] = sample
        # 判断有无 trial / 条件列
        out["trial_like_cols"] = [c for c in df.columns
                                  if any(t in c.lower() for t in
                                         ("trial", "condition", "stim", "label",
                                          "image", "category", "epoch"))]
        return out
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:100]}"}


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/out/exp30")
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print("Phase 30 — 维度审计：原始数据有哪些轴，我丢了什么")
    print("=" * 84)

    report = {}

    # ---------------- 1. Steinmetz（小鼠）
    print("\n" + "-" * 84)
    print("1. Steinmetz (小鼠, 侵入式)")
    print("-" * 84)
    st = DATA / "steinmetz"
    stein = {}
    if st.exists():
        npzs = sorted(st.glob("spikes_session_*.npz"))
        print(f"  spikes 文件: {len(npzs)}")
        if npzs:
            info = probe_npz(npzs[0], max_show=6)
            print(f"    {npzs[0].name}: {json.dumps(info, ensure_ascii=False)}")
            stein["spikes_probe"] = info
        up = st / "units.parquet"
        if up.exists():
            info = probe_parquet(up)
            print(f"    units.parquet: {info.get('n_rows')} 行, "
                  f"列 {info.get('columns')}")
            print(f"    trial 类列: {info.get('trial_like_cols')}")
            stein["units"] = info
        # 采样率
        if npzs:
            with np.load(npzs[0]) as z:
                shp = z["spikes"].shape if "spikes" in z.files else None
            if shp:
                stein["n_units"] = int(shp[0])
                stein["n_bins_50ms"] = int(shp[1])
                stein["duration_s"] = round(shp[1] * 0.05, 1)
                stein["nyquist_hz"] = 10.0     # 50ms → 1/(2*0.05)
                print(f"    维度: ({shp[0]} units, {shp[1]} bins @50ms)")
                print(f"    时长 {stein['duration_s']}s | Nyquist {stein['nyquist_hz']} Hz")
                print(f"    ⚠️ 只有 2 个轴（unit × time），trials 上游已拼成连续序列")
    report["steinmetz"] = stein

    # ---------------- 2. THINGS-EEG（人类 EEG）
    print("\n" + "-" * 84)
    print("2. THINGS-EEG (人类 EEG, 4ms)")
    print("-" * 84)
    eeg = {}
    for cand in [DATA / "eeg" / "things-eeg", DATA / "things-eeg"]:
        if cand.exists():
            print(f"  路径: {cand}")
            files = sorted(cand.rglob("*"))[:20]
            for f in files:
                if f.is_file():
                    print(f"    {f.relative_to(cand)}  ({f.stat().st_size/1024**2:.1f} MB)")
            # 找 npz/pt
            npzs = list(cand.rglob("*.npz"))[:3]
            pts = list(cand.rglob("*.pt"))[:3]
            eeg["path"] = str(cand)
            eeg["n_npz"] = len(list(cand.rglob("*.npz")))
            eeg["n_pt"] = len(list(cand.rglob("*.pt")))
            if npzs:
                info = probe_npz(npzs[0])
                print(f"    npz 探针 {npzs[0].name}: {json.dumps(info, ensure_ascii=False)[:300]}")
                eeg["npz_probe"] = info
            if pts:
                print(f"    .pt 文件: {[p.name for p in pts]}")
                eeg["pt_files"] = [str(p.name) for p in pts]
            break
    if not eeg:
        print("  ❌ 未找到 THINGS-EEG 原始目录（可能只在桶里）")
        eeg["note"] = "只在桶里，本地不可达"
    report["things_eeg"] = eeg

    # ---------------- 3. CineBrain（多模态）
    print("\n" + "-" * 84)
    print("3. CineBrain (fMRI + EEG + ECG)")
    print("-" * 84)
    cine = {}
    for cand in [DATA / "multimodal" / "cinebrain", DATA / "cinebrain"]:
        if cand.exists():
            print(f"  路径: {cand}")
            for f in sorted(cand.rglob("*"))[:25]:
                if f.is_file():
                    print(f"    {f.relative_to(cand)}  ({f.stat().st_size/1024**2:.1f} MB)")
            cine["path"] = str(cand)
            break
    if not cine:
        print("  ❌ 未找到")
    report["cinebrain"] = cine

    # ---------------- 4. Algonauts（模型特征）
    print("\n" + "-" * 84)
    print("4. Algonauts 2025（模型层特征）")
    print("-" * 84)
    alg = {}
    for cand in [DATA / "model-features" / "algonauts-2025",
                 DATA / "algonauts-2025"]:
        if cand.exists():
            print(f"  路径: {cand}")
            subs = sorted([d for d in cand.iterdir() if d.is_dir()])[:8]
            print(f"    子目录: {[d.name for d in subs]}")
            # 探针
            for s in subs[:2]:
                fs = sorted(s.rglob("*"))[:6]
                for f in fs:
                    if f.is_file():
                        print(f"      {f.relative_to(cand)} ({f.stat().st_size/1024**2:.1f} MB)")
            alg["path"] = str(cand)
            alg["subdirs"] = [d.name for d in subs]
            npzs = list(cand.rglob("*.npz"))[:2]
            for n in npzs:
                info = probe_npz(n)
                print(f"      npz {n.name}: {json.dumps(info, ensure_ascii=False)[:200]}")
                alg.setdefault("npz_probe", {})[n.name] = info
            break
    if not alg:
        print("  ❌ 未找到")
    report["algonauts"] = alg

    # ---------------- 5. OmniBrain 现有视图的维度
    print("\n" + "-" * 84)
    print("5. OmniBrain 现有视图（我造出来的）")
    print("-" * 84)
    omni = {}
    for name in ["spike_view", "spike_view_macaque", "spike_view_human",
                 "eeg_view", "graph_view"]:
        d = DATA / "omnibrain" / name
        if not d.exists():
            continue
        npzs = sorted(d.glob("*.npz"))
        if npzs:
            info = probe_npz(npzs[0])
            key = list(info.keys())[0] if info else None
            shape = info.get(key, {}).get("shape") if key else None
            omni[name] = {"sample": npzs[0].name, "shape": shape,
                          "n_files": len(npzs)}
            print(f"  {name}: {shape}  ({len(npzs)} 个文件)")
        else:
            mp = d / "meta.parquet"
            if mp.exists():
                info = probe_parquet(mp)
                omni[name] = {"meta_columns": info.get("columns")}
                print(f"  {name}: 只有 meta, 列 {info.get('columns')}")
    report["omnibrain"] = omni

    # ---------------- 总结：可行的维度
    print("\n" + "=" * 84)
    print("结论：哪些维度可恢复")
    print("=" * 84)

    verdict = []

    # 频率维
    nyq = stein.get("nyquist_hz")
    verdict.append({
        "dimension": "频率 frequency",
        "steinmetz": f"Nyquist {nyq} Hz（仅 delta/theta）" if nyq else "不可用",
        "things_eeg": "Nyquist 125 Hz（1-45 Hz 全可解析）",
        "feasible": "受采样率限制；EEG 可做完整频带分解",
    })

    # 试次维
    verdict.append({
        "dimension": "试次 trial",
        "steinmetz": "上游已拼成连续序列，不可恢复",
        "things_eeg": "若原始 .pt 按试次存，可恢复",
        "feasible": "取决于上游格式，需实测",
    })

    # 时间维
    verdict.append({
        "dimension": "时间 time",
        "steinmetz": "50ms × 数万 bin，但我在分析里平均掉了",
        "things_eeg": "4ms × 数百 ms，完整保留",
        "feasible": "✅ 完全可恢复，只需不平均",
    })

    # 层维
    verdict.append({
        "dimension": "层/深度 layer",
        "steinmetz": "units.parquet 无 layer 列（只有 region）",
        "things_eeg": "EEG 无层概念",
        "feasible": "❌ 数据里没有",
    })

    for v in verdict:
        print(f"\n  {v['dimension']}")
        print(f"    小鼠: {v['steinmetz']}")
        print(f"    EEG : {v['things_eeg']}")
        print(f"    可行: {v['feasible']}")

    report["dimensionality_verdict"] = verdict

    (out / "phase30_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  报告 → {out}/phase30_report.json")


if __name__ == "__main__":
    main()
