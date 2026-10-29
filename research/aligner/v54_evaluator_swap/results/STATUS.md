# v54 产物状态

| 文件 | 是什么 | 能不能引用 |
|---|---|---|
| `metrics.json` | **全网格正式产物**: 36 格 × 5 seeds = 180 观测, 完备 | ✅ 可引用 |
| `run_full.log` | 该次运行的完整日志 | ✅ 佐证 |

- `code_sha = a32eda6`, `code_fp = aeafb1d1f4fc54c8`, `predictions_sha = 20c6ae18d313b4e3`
- `device = cpu`（`n_cpu=64` 是容器看到的核数, 非实际配额; `torch_cuda_build = None`）
- 桶: `redmingwei/hetero-moe-jobs/v54-a32eda6-cpu8/`
- 硬件: HF Space `cpu-upgrade` (8 vCPU / 32 GB), 全程 **49 分钟** 跑完 36 格
  （对照 `cpu-basic` 2 vCPU: 75 分钟只到 28 格, 约 **2.4 倍**）

## 本地独立复核 (`tools/verify_v54_run.py`)

```
C1a code_fp  产物=aeafb1d1f4fc54c8 本地=aeafb1d1f4fc54c8   ok
C1b 判据指纹 产物=20c6ae18d313b4e3 本地=20c6ae18d313b4e3   ok
C2  网格     36/36 格, 180/180 观测                        ok
C3  配对     30 组 (arm,D,seed) 三档 init_sha 不一致 0 组    ok
C4  判决     逐条对拍 0 处不一致                            ok
C5  闸门     all_pass 产物=False 本地重算=False             ok
→ 9/9 PASS, 远端产物在本地代码下逐条复现
```

## 闸门状态

`grid_complete=True` / `no_undecided=True` / `all_true=False` → **`all_pass=False`**

**这是完备的负面结果, 不是"没跑成"。** 六条判据里四条未过, 原因见 `FINDINGS.md`。

## 与冒烟的关系

首轮冒烟产物 (出自 `4023ee5+dirty`) 已删除; 本目录只保留正式产物。
后续任何冒烟运行必须落在 `V54_ART_DIR` 指向的临时目录, 不得写进 `results/`。
