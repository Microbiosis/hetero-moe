# V53.0 — 产物血缘更正 (2026-10-27)

> **结论**: v53 的 `metrics.json` 里 `n_classes` 记错了 (8), 真实是 **6**。
> 判决 (D1a/D1b/D1c/D3) 全部不受影响; 但**基线数字与"低于平凡基线"的说法作废**。

## 1. 证据链

| # | 证据 | 读数 |
|---|---|---|
| 1 | `research/aligner/v53_third_audit/audit.py` 的 `CLASSES` | **6** 个键 (cavity/pollen/cable/bead/timber/tide) |
| 2 | `examples/run_v53_third_audit.py:257` | 写的是 `"n_classes": N_CLASSES` (即 6) |
| 3 | HF Space `redmingwei/hetero-moe` 上的 `audit.py` / runner | 与仓库版 **逐字节相同** (diff 为空) |
| 4 | Space 的 `v53_artifacts/logs/run_D96_audit.log` | 数字与 cells 逐位一致 → 该日志确由这套代码产生 |
| 5 | 桶里的 `v53-v53_artifacts/metrics.json` 与仓库版 | 逐字节相同 (32431 B, 内容一致) |
| 6 | `metrics[D96:learned_lift].rows == cells/D96_learned_lift.rows` | **True** |

⇒ **results 体来自那次真实运行; 只有 `config` 块出自另一个写入者。**
一份产物里两个来源, 是这次错误的入口。

## 2. 影响的数字

块状 one-hot 目标下 `MSE(预测全零) = 1/n_classes` 是**精确等式**, 不是估计。

| 口径 | n_classes | 全零基线 | 预测类别均值基线 `(n-1)/n²` |
|---|---|---|---|
| 错误的记录值 | 8 | 0.1250 | 0.1094 |
| **真实值** | **6** | **0.1667** | **0.1389** |

| 臂 | held-out fuse | 真实口径下 |
|---|---|---|
| D96 `learned_lift` | 0.1617 | 比全零**好 3.0%**, 比预测均值**差 16.4%** |
| D384 `learned_lift` | 0.1545 | 比全零**好 7.3%**, 比预测均值**差 11.3%** |

**必须改写的一句**: "有意义空间把误差降到平凡基线以下" —— 在 v53 **不成立**。
正确表述是: 有意义空间**勉强好于预测全零 (3–7%)，但明显差于"预测类别均值"这个
零参数基线 (11–16%)**。也就是说 v53 的有意义空间臂**还没有真正学到类别结构**。

这与 v52 形成对比 (v52 的 0.071 好于 0.109 的预测均值基线) ——
两台独立的实现, 一台过了零参数基线, 一台没过。这条差异需要在 v54 里被解释。

## 3. 不受影响的部分

D1a (源−类可解码 ≥0.30) / D1b (置换后 margin 变化) / D1c (有意义空间 margin ≤ 冻结臂一半) /
D3 (跨底座一致性) 全部是**相对比较**, 与类别数无关。`tools/check_artifacts.py` 的 V1 项
从 rows 重算了 D1c, 与存储值逐档一致。

冻结随机臂"比全零差 46–57 倍"在新口径下是 **45–57 倍**, 量级不变 ——
这一条反而更稳 (无论 6 还是 8 类, 它都远离基线)。

## 4. 已做的更正

1. `results/metrics.json`: `config.n_classes` 8 → **6**, 原值保留在
   `config.n_classes_recorded_originally`, 并新增 `corrections` 块说明缘由;
   `results` 体**逐字节未改** (已校验)。
2. `tools/check_artifacts.py` (新增): 通用产物检查器, 7 项 (F1/L1/L2/L3/L4/B1/V1),
   对 v53 报 **2 FAIL** (L1 n_classes 不一致, B1 基线口径相差 33%) + 1 WARN (无 code_sha)。
3. v54 起强制: 每个 cell 落 `(config, code_sha, seed)`, `n_classes` 只在 audit 模块定义一次。

## 5. 复现检查

```bash
python tools/check_artifacts.py research/aligner/v53_third_audit --cells /tmp/v53_bucket
```

（`--cells` 指向从桶 `redmingwei/hetero-moe-jobs/v53-v53_artifacts/cells/` 下载的目录；
不提供时 L4 降级为 WARN。）
