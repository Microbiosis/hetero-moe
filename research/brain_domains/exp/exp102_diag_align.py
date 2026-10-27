#!/usr/bin/env python3
r"""exp102_diag_align.py — **诊断 exp102 的「对齐闸门」为什么只在某些被试上炸**。

## 现象（2026-10-27 夜）

生成跑到 sub-05 时，`exp102_gen.py` 的对齐闸门炸了：

    [对齐] sub-02: 200-way top-1=0.0550（随机 0.0050｜exp101 参照 0.0589）｜条件侧 2-way=0.7500
    [对齐] sub-04: 200-way top-1=0.0600（随机 0.0050｜exp101 参照 0.0589）｜条件侧 2-way=0.7800
    [对齐] sub-05: 200-way top-1=0.0000（随机 0.0050｜exp101 参照 0.0589）｜条件侧 2-way=0.6650
    RuntimeError: 对齐闸门未过：200-way=0.0000 ≈ 随机 ⇒ 真图顺序与 EEG 行序对不上

⇒ **条件侧 2-way 还有 0.665（不是 0.5）** ⇒ 条件本身不是噪声，概念集合也不是另一套
⇒ 但 200-way **恰好 0.0000**（不是 ~0.005 的随机水平）⇒ **系统性错位**。

## 假设（要验的就是它）

- 4D 的行序来自**各被试自己的** `test.pt`（`build_4d_eeg.py` 原样存 `obj["eeg"]`）；
- `exp102_gen.find_test_image_paths()` **自发现** test 图并**按概念号排序**，
  ⇒ 于是它默认「4D 行号 i ↔ 概念号 i」；
- 而 `omnibrain4d/eeg/meta.parquet` 里**记着每一行对应哪张图**（`img` 列，来自同一个 `test.pt`）
  ⇒ ⇒ **它是权威对应**，不是猜的。

⇒ 本脚本：对每个被试
  ① 用 meta.parquet 求出「行号 → 概念号」的真实映射；
  ② 用它把条件重排到「自发现顺序」上；
  ③ 重算 200-way top-1 ⇒ **若回到 ~0.05 ⇒ 假设成立 ⇒ 修 exp102_gen 的行序**；
  ④ 同时报未重排时的命中偏移直方图（判「整体平移」还是「任意置换」）。

## 结论怎么用

- **重排后 200-way 回到 ~0.05** ⇒ **管线索引错** ⇒ 修 `exp102_gen.py`（用 meta.parquet 对齐）
  ⇒ **不改判据、不改阈值**；
- **重排后仍 ~0** ⇒ 该被试的 4D 数据与 test 图对不上 ⇒ **如实剔除并写进留痕**，
  **不许**用「反正 9 个被试也够 8 个」这种事后挑人的说法。

## 用法

```powershell
python research\\brain_domains\\exp\\exp102_diag_align.py            # 本地
python -u /domains/scripts/exp102_diag_align.py                      # 云端（CPU 够）
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

EXP_DIR = Path(__file__).resolve().parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

TEST_4D = Path("/domains/omnibrain4d/eeg")
META_PARQUET = TEST_4D / "meta.parquet"
ENCODER_CKPT = "/domains/runs/illusionbrain-align/exp93b/encoder_best.pt"
DEFAULT_OUT = Path("/domains/runs/illusionbrain-align/exp102/gen2")
N_IMG = 200
TEST_SUBJECTS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)


def log(*a):
    print(*a, flush=True)


def l2norm(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-12)


def ensure_pandas():
    """parquet 读取要 pandas；镜像里不一定有 ⇒ 缺就装（**缺资源即装，不许静默跳过**）。"""
    try:
        import pandas as pd  # noqa: F401
        return True
    except Exception:
        pass
    log("[依赖] 缺 pandas ⇒ pip install pandas pyarrow")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pandas", "pyarrow"],
                       check=True, timeout=900)
        import pandas as pd  # noqa: F401
        return True
    except Exception as e:                                       # noqa: BLE001
        log(f"!! pandas 装不上：{type(e).__name__}: {e}")
        return False


def concept_of(img_str):
    """meta 的 ``img`` 字符串 ⇒ 概念号（目录名 ``<概念号>_<概念名>``）。"""
    s = str(img_str).replace("\\", "/")
    parts = s.split("/")
    if len(parts) < 2:
        return None
    try:
        return int(parts[-2].split("_")[0])
    except Exception:
        return None


def row_to_concept_map():
    """meta.parquet ⇒ {被试: {行号: 概念号}}（权威对应，来自各被试自己的 test.pt）。"""
    import pandas as pd
    df = pd.read_parquet(META_PARQUET, columns=["subject", "image_idx", "img"])
    out = {}
    for (subj, iidx), grp in df.groupby(["subject", "image_idx"], sort=False):
        cons = {concept_of(v) for v in grp["img"].tolist()}
        cons.discard(None)
        if len(cons) != 1:
            raise RuntimeError(f"meta 里 (subject={subj}, image_idx={iidx}) 的概念不唯一：{cons}")
        out.setdefault(str(subj), {})[int(iidx)] = int(cons.pop())
    return out


def load_true_clip(out_dir: Path):
    tp = out_dir / "true_clip.npy"
    if tp.exists():
        arr = np.load(tp).astype(np.float64)
        log(f"[真图] 用已存的 {tp}（{arr.shape}）")
        return l2norm(arr)
    raise SystemExit(f"!! {tp} 不存在 ⇒ 先跑一次 --step generate")


def concept_order(out_dir: Path):
    """自发现顺序里「第 k 张图」的概念号（生成那一步存过路径表）⇒ 概念号 → 行号。

    ⇒ 不假设「概念号 == 行号」⇒ 概念号可能是任意值 ⇒ 必须查表。
    """
    jp = out_dir / "true_clip_paths.json"
    if not jp.exists():
        log(f"!! {jp} 不存在 ⇒ 只能假设概念号 == 行号")
        return None
    paths = json.loads(jp.read_text(encoding="utf-8"))
    out = {}
    for k, p in enumerate(paths):
        con = concept_of(p)
        if con is None:
            log(f"!! 路径 {p} 解析不出概念号")
            return None
        out[con] = k
    log(f"[真图] 路径表 {jp.name} ⇒ 概念号 {min(out)}..{max(out)}，共 {len(out)} 个")
    return out


def encode_subject(sid: int, device: str):
    """4D EEG 的前 200 个试次（rep 0）⇒ 冻结 exp93b 编码器 ⇒ 条件 z。"""
    import torch
    import exp93b_contrastive as E93B

    eeg = np.load(TEST_4D / f"sub-{sid:02d}.npz")["eeg"]
    flat = np.asarray(eeg[:N_IMG, 0], dtype=np.float32)
    ck = torch.load(ENCODER_CKPT, map_location="cpu", weights_only=False)
    enc = E93B.build_model(n_ch=ck["n_ch"], n_times=ck["n_times"], out_dim=ck["out_dim"])
    enc.load_state_dict(ck["model"] if "model" in ck else ck["state_dict"])
    enc = enc.to(device).eval()
    with torch.no_grad():
        z = E93B.encode_all(enc, flat, device, 16)
    return z.astype(np.float64), tuple(eeg.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    true_clip = load_true_clip(out_dir)
    n = min(N_IMG, true_clip.shape[0])
    log(f"[真图] 参与比较的真图数 = {n}（自发现顺序＝按概念号升序）")

    have_meta = ensure_pandas()
    con2idx = concept_order(out_dir)
    r2c = {}
    if have_meta:
        try:
            r2c = row_to_concept_map()
            log(f"[meta] {META_PARQUET} ⇒ 逐被试行号→概念号（"
                f"{ {k: len(v) for k, v in list(r2c.items())[:3]} } …）")
        except Exception as e:                                   # noqa: BLE001
            log(f"!! meta 解析失败：{type(e).__name__}: {e}")
            r2c = {}

    report = []
    for sid in TEST_SUBJECTS:
        try:
            cond, eeg_shape = encode_subject(sid, args.device)
        except Exception as e:                                   # noqa: BLE001
            log(f"[sub-{sid:02d}] 读/编码失败：{type(e).__name__}: {e}")
            report.append({"subject": sid, "error": f"{type(e).__name__}: {e}"})
            continue
        cz = l2norm(cond[:n].astype(np.float64))
        sim = cz @ true_clip[:n].T
        am = sim.argmax(axis=1)
        idx = np.arange(n)
        acc_raw = float((am == idx).mean())
        off = Counter((am - idx).tolist())
        best_k, best_h = off.most_common(1)[0]
        neg = (idx + 37) % n
        acc2 = float((sim[idx, idx] > sim[idx, neg]).mean())

        rec = {"subject": sid, "eeg_shape": list(eeg_shape),
               "acc200_raw": acc_raw, "acc2_raw": acc2,
               "top_offset": int(best_k), "top_offset_hits": int(best_h)}

        # ---- 用 meta 的权威对应重排 ----
        key = f"sub-{sid:02d}"
        mp = r2c.get(key)
        if mp and len(mp) >= n and con2idx:
            # 行 r 的概念 c ⇒ perm[c] = 那一行（c 是**真图行号**，由路径表查得）
            perm = np.full(n, -1, dtype=np.int64)
            okmap = True
            for r in range(n):
                c = mp.get(r)
                if c is None or c not in con2idx:
                    okmap = False
                    break
                k = con2idx[c]
                if perm[k] != -1:
                    okmap = False
                    break
                perm[k] = r
            if okmap:
                acc_perm = float((am[perm] == idx).mean())
                rec.update({"acc200_perm": acc_perm,
                            "perm_is_identity": bool(np.array_equal(perm, idx)),
                            "perm_first10": [int(v) for v in perm[:10]]})
                log(f"[sub-{sid:02d}] 原始 200-way={acc_raw:.4f}（偏移众数 k={best_k:+d}，"
                    f"{best_h}/{n} 命中）｜**按 meta 重排后** 200-way={acc_perm:.4f}"
                    f"｜permutation 为恒等？{rec['perm_is_identity']}")
            else:
                log(f"[sub-{sid:02d}] meta 映射不完整 ⇒ 跳过重排")
        else:
            log(f"[sub-{sid:02d}] 无 meta 映射 ⇒ 只报原始值")

        norms = np.linalg.norm(cond[:n], axis=1)
        rec["cond_norm_med"] = float(np.median(norms))
        rec["nan"] = bool(np.isnan(cz).any())
        report.append(rec)
        log(f"           2-way(偏移37)={acc2:.4f}｜‖z‖ 中位 {rec['cond_norm_med']:.3f}"
            f"｜NaN={rec['nan']}")

    ok = [r for r in report if "acc200_raw" in r]
    log("")
    log("=== 结论 ===")
    for r in ok:
        log(f"  sub-{r['subject']:02d}: 原始 {r['acc200_raw']:.4f}"
            + (f" → 重排后 {r.get('acc200_perm', float('nan')):.4f}" if "acc200_perm" in r else "")
            + f"｜恒等？{r.get('perm_is_identity')}")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
        log(f"[写盘] {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
