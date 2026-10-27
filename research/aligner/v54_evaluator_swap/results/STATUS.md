# v54 产物状态

> **本目录的 `metrics.json` 是冒烟快照, 不是实验结果。**

| 项 | 值 |
|---|---|
| 性质 | 冒烟 (链路验证) |
| 产物自述 | `protocol_check.status == "smoke"` (runner 写入, 检查器据此放行 L6) |
| 网格 | D=96 × 4 臂 × 3 评价器 × **1 seed** × **2 steps** |
| 编码器 | **伪编码** (确定性哈希投影, 不用底座权重) |
| code_sha | `2f4119a` (干净树, 对应真实提交) |
| predictions_sha | `20c6ae18d313b4e3` (= `protocol.lock.json`, 判据未被事后改动) |
| 能否引用 | **否** |

## 全网格未跑

180 格 = 3 D (96/252/384) × 4 臂 × 3 评价器 × 5 seeds, 200 steps, 真实底座编码。
本机不可行: 沙箱剩 ~350 MB 可用内存、swap 已用 12.4 GB, 冒烟 12 格就要 228 s。

计划:
```bash
V54_DEVICE=cuda V54_MAX_MINUTES=120 bash run.sh v54     # HF Space t4-small
```

## 全网格跑完后必须做的三件事

1. 用真实 metrics 覆盖本文件所在目录的 `metrics.json`; 冒烟版另存 `metrics.smoke.json`
2. `python tools/check_artifacts.py research/aligner/v54_evaluator_swap` → 必须 0 FAIL
3. `bash run.sh generations` 的 `[MISSING] results/metrics.json` 应消失; `selfcontain` 应保持 46/46

## 完整性四件套 (全网格开跑前已就位)

| 机制 | 防的是什么 | 检查项 |
|---|---|---|
| `protocol.lock.json` | 看到结果后偷改网格 (seeds/steps/D/评价器) | L6 |
| `predictions_sha` | 看到结果后偷改预登记判据 | L6 |
| `init_sha` (逐 arm,D,seed) | "三档配对"只是嘴上说说 | L5 |
| `code_sha` + `+dirty` 标记 | 产物出处不明 (v53 的病) | L2 |
