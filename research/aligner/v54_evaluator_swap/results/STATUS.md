# v54 产物状态

| 文件 | 是什么 | 能不能引用 |
|---|---|---|
| `metrics.json` | **冒烟快照** (D=96 / 1 seed / 8 steps / 伪编码) | ❌ 不能。顶部有 `IS_SMOKE_NOT_RESULTS: true` |

全网格 (180 格 = 3 D × 4 臂 × 3 评价器 × 5 seeds, 200 steps, 真实底座编码) **尚未运行**。
计划: HF Space `redmingwei/hetero-moe` 升到 t4-small, `V54_DEVICE=cuda V54_MAX_MINUTES=120 bash run.sh v54`,
产物按 `v54-<sha>/` 落桶 `redmingwei/hetero-moe-jobs`。

跑完后应:
1. 用真实全网格 metrics 覆盖本目录的 `metrics.json` (去掉 `IS_SMOKE_NOT_RESULTS`);
2. `python tools/check_artifacts.py research/aligner/v54_evaluator_swap` 应 0 FAIL;
3. `bash run.sh generations` 的 `[MISSING] results/metrics.json` 应消失。
