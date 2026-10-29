# v57 产物状态

| 文件 | 是什么 | 能不能引用 |
|---|---|---|
| （暂无 `metrics.json`）| 全网格**尚未在本仓库落盘** | — |

## 已完成的验证

| 项 | 结果 |
|---|---|
| 单元测试 | **10/10 PASS** (`tests/test_v57_adaptive_dimension.py`) |
| 数学标定 | σ̂ 在 σ∈{0.5,1,3} 上还原到 8% 内; τ 高于噪声谱上边缘 (三组形状) |
| 秩精确性 | 真实秩 {0,3,8,20} 全部精确估出 |
| 生长 | 固定秩-3 数据流 → r=3 (不多不少) |
| 扩张 | 新方向出现 → r 3 → 8 |
| **坍缩** | 信号消失 → r 8 → **0** (died 合计 8) |
| 算子 | `P²=P`, 子空间外残差 3.4e-07, 梯度非零 |

## 为什么没有 metrics.json

`examples/run_v57_adaptive_dimension.py` 需要 v56 的表征缓存 (`h_base/c_passive` npz),
本机无该缓存时会在 Space 上重算 4B 容器。**这是本地不该跑的活。**
该实验已在 Space 上跑过一次，但因容器重启未落桶; 重跑命令:

```bash
V57_CACHE=/app/v56_repr.npz bash run.sh v57     # 在 HF Space 上
```

落桶后应把 `metrics.json` 放回本目录，并重跑
`python tools/check_artifacts.py research/aligner/v57_adaptive_dimension`。
