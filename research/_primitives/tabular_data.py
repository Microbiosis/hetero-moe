"""Adult / Online Shoppers 数据集的**唯一**加载入口.

背景
----
``v38_g_adult_boundary`` 与 ``v39_g2_interaction`` 各自带了一份**逐字节完全相同**的
``adult_loader.py`` + ``adult.data`` + ``adult.test``(UCI Adult, 6 MB 重复),
并且都会把解析结果缓存成 ``adult_parsed.npz``。

2026-11 起这条线解除 gitignore 恢复入库, 重复必须消除, 否则公开仓库里
会实打实多出 6 MB 冗余, 且两份数据一旦分叉, v38 与 v39 的结果就不可比。

做法
----
本模块是唯一实现:

1. 数据文件按顺序查找 —— 本包 data 目录 → 环境变量 ``ADULT_DATA_DIR``
   → v38 的 data 目录 → UCI 官方下载地址
2. 解析结果缓存写到 ``HF_HOME``/临时目录等**可写位置**, 不再往包里写派生物

两个包的 ``adult_loader.py`` 退化为薄转发层(见各包内同名文件)。
"""
from __future__ import annotations

import os
import tempfile
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "adult_data_dir",
    "load_adult",
    "load_adult_crossmodal",
    "load_online_shoppers",
]

# UCI Adult 官方地址(公开数据集)
_UCI = {
    "adult.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "adult.test": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
}
_UCI_FALLBACK = {
    "adult.data": "https://archive.ics.uci.edu/static/public/2/adult.zip",
}

_NUM_COLS = [0, 2, 4, 10, 11, 12]   # age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
_LABEL = 14                          # >50K / <=50K

_HERE = os.path.dirname(os.path.abspath(__file__))
# _HERE = <repo>/research/_primitives  ⇒ 退一层到 <repo>/research
_REPO_RESEARCH = os.path.abspath(os.path.join(_HERE, ".."))
_V38_DATA = os.path.join(_REPO_RESEARCH, "scale_heterogeneous", "v38_g_adult_boundary", "data")
_V39_DATA = os.path.join(_REPO_RESEARCH, "scale_heterogeneous", "v39_g2_interaction", "data")


def _cache_dir() -> str:
    """派生缓存的落盘位置 —— 必须是**可写**且不在版本库里。"""
    for key in ("ADULT_CACHE_DIR", "HF_HOME", "XDG_CACHE_HOME"):
        v = os.environ.get(key)
        if v:
            d = os.path.join(v, "hetero_moe", "adult") if key != "ADULT_CACHE_DIR" else v
            os.makedirs(d, exist_ok=True)
            return d
    d = os.path.join(tempfile.gettempdir(), "hetero_moe", "adult")
    os.makedirs(d, exist_ok=True)
    return d


def adult_data_dir() -> str:
    """返回含 ``adult.data`` / ``adult.test`` 的目录。

    查找顺序: 环境变量 ``ADULT_DATA_DIR`` → v38 包 → v39 包。
    若都不存在, 由 :func:`_ensure_files` 从 UCI 下载到缓存目录。
    """
    env = os.environ.get("ADULT_DATA_DIR")
    if env and all(os.path.exists(os.path.join(env, f)) for f in _UCI):
        return env
    for cand in (_V38_DATA, _V39_DATA):
        if all(os.path.exists(os.path.join(cand, f)) for f in _UCI):
            return cand
    return _ensure_files()


def _ensure_files() -> str:
    """从 UCI 下载 Adult 原始文件到缓存目录(公开数据集, 无需授权)。"""
    d = _cache_dir()
    try:
        for name, url in _UCI.items():
            path = os.path.join(d, name)
            if not os.path.exists(path):
                urllib.request.urlretrieve(url, path)
        return d
    except Exception as exc:  # pragma: no cover - 网络不可达时给出可操作的提示
        raise RuntimeError(
            "未找到 UCI Adult 数据文件, 且自动下载失败。"
            f"请手动下载 adult.data / adult.test 到任意目录, 然后设置环境变量 "
            f"ADULT_DATA_DIR 指向该目录 (原始地址: {_UCI['adult.data']}). "
            f"原始错误: {exc}") from exc


def _parse(path: str) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[List[float]] = []
    ys: List[int] = []
    with open(path, encoding="ascii", errors="ignore") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().rstrip(".").split(",")]
            if len(parts) < 15:
                continue
            try:
                xs.append([float(parts[c]) for c in _NUM_COLS])
            except ValueError:
                continue          # 含 '?' 的行
            ys.append(1 if parts[_LABEL] == ">50K" else 0)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def _cached() -> Tuple[np.ndarray, np.ndarray]:
    cache = os.path.join(_cache_dir(), "adult_parsed.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["X"], d["y"]
    src = adult_data_dir()
    Xa, ya = _parse(os.path.join(src, "adult.data"))
    Xt, yt = _parse(os.path.join(src, "adult.test"))
    X = np.concatenate([Xa, Xt])
    y = np.concatenate([ya, yt])
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)      # 标准化, 训练/推理一致
    try:
        np.savez(cache, X=X, y=y)
    except OSError:
        pass                                       # 只读环境: 直接用内存结果
    return X, y


def load_adult(n: int = 512, seed: int = 0, balance: bool = True):
    """返回 ``[n,6] float32`` 特征 + ``[n] long`` 标签(torch 张量)。

    ``balance=True`` 时正负样本各半(Adult 原始不平衡 ~24% 正)。
    """
    import torch

    X, y = _cached()
    rng = np.random.RandomState(seed)
    if balance:
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        half = n // 2
        idx = np.concatenate([rng.choice(pos, half, replace=len(pos) < half),
                              rng.choice(neg, n - half, replace=len(neg) < n - half)])
    else:
        idx = rng.choice(len(y), n, replace=len(y) < n)
    idx = rng.permutation(idx)
    return torch.tensor(X[idx]), torch.tensor(y[idx])


_CAT_COLS = [1, 3, 5, 6, 7, 8, 9, 13]   # workclass, education, marital, occupation,
                                         # relationship, race, sex, native-country
# 高薪职业(Adult 中 >50K 比例最高的职业集) — 跨模态标签只用**文本侧**的信息
_HIGH_PAY = {"Exec-managerial", "Prof-specialty", "Tech-support"}
_HOURS_THR = 45        # 工时阈值 — 只在**表格数值侧**


def load_adult_crossmodal(n: int = 512, seed: int = 42, decouple: bool = True):
    """跨模态构造: 文本(类别特征) + 表格(数值特征), 标签需**两侧联合**。

    ``y = 1`` iff ``(职业 ∈ 高薪集) XOR (工时 ≥ 45)``
    → 纯文本 / 纯表格单侧都有预测天花板, 跨模态融合才解锁。

    **标签结构是 XOR 而非 AND**: AND 结构存在强泄漏(hours<45 ⇒ y=0 确定性高,
    单侧可分); XOR 使任何单侧在条件独立时信息为零, 从而保证**严格的
    "跨模态必需"**。

    ``decouple=True``: 非工时数值列做样本间随机置换, 保留真实边缘分布但切断与
    职业的联合分布(Adult 的 capital-gain / education-num 与职业强相关, 否则表格侧
    能间接泄漏职业信息, 单侧天花板虚高)。

    返回 ``(texts, X, y)`` —— texts 含 ``"working as <职业>"`` 模板。
    """
    import torch

    src = adult_data_dir()
    rows = []
    with open(os.path.join(src, "adult.data"), encoding="ascii", errors="ignore") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().rstrip(".").split(",")]
            if len(parts) < 15 or "?" in parts:
                continue
            try:
                nums = [float(parts[c]) for c in _NUM_COLS]
            except ValueError:
                continue
            cat = [parts[c] for c in _CAT_COLS]
            occ = cat[3]
            hours = nums[5]
            y = 1 if (occ in _HIGH_PAY) != (hours >= _HOURS_THR) else 0
            rows.append((cat, nums, y))

    pos = [r for r in rows if r[2] == 1]
    neg = [r for r in rows if r[2] == 0]
    rng = np.random.RandomState(seed)
    half = n // 2
    pi = rng.choice(len(pos), half, replace=len(pos) < half)
    ni = rng.choice(len(neg), n - half, replace=len(neg) < n - half)
    sel = [pos[i] for i in pi] + [neg[i] for i in ni]
    order = rng.permutation(len(sel)).tolist()
    sel = [sel[i] for i in order]

    texts, Xs, ys = [], [], []
    for cat, nums, y in sel:
        t = (f"A {cat[4].lower()} {cat[5].lower()} working as {cat[3].lower()} "
             f"in {cat[0].lower()}, {cat[1].lower()} educated, {cat[2].lower()}, "
             f"{cat[6].lower()}, from {cat[7].lower()}.")
        texts.append(t); Xs.append(list(nums)); ys.append(y)
    Xa = np.asarray(Xs, dtype=np.float32)
    if decouple:
        # hours(列 5)保持不动; 其余 5 列在样本间随机置换 → 切断与职业的关联
        for c in range(6):
            if c != 5:
                Xa[:, c] = Xa[rng.permutation(len(Xa)), c]
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
    X = (Xa - mu) / sd
    return texts, torch.tensor(X), torch.tensor(ys, dtype=torch.long)


def load_online_shoppers(n: int = 512, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Online Shoppers Intention(公开数据集)→ ``([n,6], [n])``。"""
    for cand in (_V39_DATA, _V38_DATA):
        p = os.path.join(cand, "online_shoppers_intention.csv")
        if os.path.exists(p):
            rows = [r for r in open(p, encoding="utf-8", errors="ignore").read().splitlines()[1:]]
            header = ["bounce", "session", "dow", "month", "year", "special"]
            data, ys = [], []
            for r in rows:
                parts = r.split(",")
                if len(parts) < 18:
                    continue
                try:
                    data.append([float(parts[header.index(h)]) for h in header])
                except ValueError:
                    continue
                ys.append(1 if parts[17] == "TRUE" else 0)
            X = np.asarray(data, dtype=np.float32); y = np.asarray(ys, dtype=np.int64)
            X = (X - X.mean(0)) / (X.std(0) + 1e-6)
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(y), min(n, len(y)), replace=False)
            return X[idx], y[idx]
    raise RuntimeError(
        "未找到 online_shoppers_intention.csv; 它随 v39 包提供或可用 ADULT_DATA_DIR 指定目录。"
        "公开地址: https://archive.ics.uci.edu/ml/machine-learning-databases/00468/"
        "Online_Shoppers_Intention.csv")
