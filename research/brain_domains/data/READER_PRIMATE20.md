# 读取方式

数据集是**索引族**（indexed family），不是矩阵。

```
D = ⊔_{i=1..175} X_i,     X_i ∈ R^{n_bins_i × n_units_i}
```

每个 `X_i` 是独立的集合，有自己的形状。**不存在跨片段的共同单元维度**：
`X_1` 的第 7 列和 `X_2` 的第 7 列是两个不同神经元的记录。

`counts.npy` 与 `counts_ragged.npy` 的差别就在这里——前者把每个片段补零到
192 列，制造了一个不存在的共同维度；后者按原生长度首尾相接。

## 读取

```python
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "illusion-brain/primate-spike-20ms"

# 1) 全部成员首尾相接的一维数组
C = np.load(hf_hub_download(REPO, "counts_ragged.npy", repo_type="dataset"),
            mmap_mode="r")                     # (998102320,) uint8

# 2) 索引：每个成员的偏移、形状、bin 宽
idx = pd.read_parquet(hf_hub_download(REPO, "counts_index.parquet",
                                      repo_type="dataset"))
```

单个成员：

```python
r = idx.iloc[0]
x = C[r.offset : r.offset + r.n_elements].reshape(int(r.n_bins_20ms),
                                                  int(r.n_units))
# x.shape == (3425, 49), dtype uint8, 单位：每 20 ms 的放电计数
```

## 不能做的事

```python
# ✗ 不存在共同单元维度
C.reshape(175, -1)          # 形状不同，不能这样切

# ✗ 不同成员的 bin 宽可能不同（见 bin_ms 列）
#   目前 175 个成员全部是 20 ms，但结构允许混合
```

## 若成员有不同的 bin 宽

`bin_ms` 是每行自己的属性，不是数据集的全局属性。混合时不要重采样到共同栅格，
而是各自在自己的栅格上算，再把无量纲量放在一起比较。

## 派生的窗口版

`timeseries.npy` 是从 `counts_ragged.npy` 派生的一个视图
（逐片段 PCA 到 64 维、800 ms 窗、50% 重叠），形状 `(273918, 64, 40)` float16。
它比原始计数少了信息，只在需要与旧实验对齐时使用。
