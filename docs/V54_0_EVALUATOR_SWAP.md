# V54.0 — 评价器换轨: MSE → {高斯 NLL, Cross-Entropy}

> **本文件是薄指针 (thin pointer), 非权威副本.**
> 权威正文: [`research/aligner/v54_evaluator_swap/SPEC.md`](../research/aligner/v54_evaluator_swap/SPEC.md)
> 机器可读声明: [`research/aligner/v54_evaluator_swap/MANIFEST.json`](../research/aligner/v54_evaluator_swap/MANIFEST.json)
> 实测快照: `research/aligner/v54_evaluator_swap/results/metrics.json`
> 复现: `bash run.sh v54` / `bash run.sh v54-smoke`

**一句话**: v49–v53 的全部结论都建立在一个评价器上; 本代不缩小实验, 只把尺子从一个变成三个,
数据/探针/抬升 import 同一份代码, 三档配对 (同 seed 同初始化)。

**出发点**: v53 的块状 one-hot 目标下 `MSE(预测全零) = 1/n_classes` 是精确等式。
v53 真实 `n_classes = 6` → 全零基线 **0.166667**、预测类别均值基线 **0.138889**。
v53 有意义空间臂 0.1617 只是勉强好于全零 (3.0%), **比零参数的"预测类别均值"还差 16.4%**。

**已实现并自检**:
- 三档评价器 + 参数等量 (`mse` +0 / `ce` +1 / `gauss_nll` +D+1), 9/9 单测通过
- CE 输出压到单纯形, 否则其 MSE 与 MSE 档不可比 (冒烟: 1.97 → 0.1528)
- 推理读数与探针输入都过同一层评价器头
- 内部一致性: CE 均匀预测的 MSE == `predict_mean` 基线 = 5/36; CE loss == ln 6
- `D_shared` 整除约束 (6 类下 256 不可用, 用 252)
- 血缘: 每格带 `code_sha`, 配合 `tools/check_artifacts.py` 七项检查

**预登记**: E1–E6 见 SPEC §3。E4 (方差头没退化) 必须在读 E2/E3 之前过。

**冒烟阶段的两个结构性发现** (已写入 SPEC §7):
1. CE 输出必须压到单纯形 **且** 推理读数必须过同一层头 —— 否则 CE 臂读数是 1.97 (基线 0.1667)。
   修好后三档配对读数: `0.1504 / 0.1507 / 0.1500`, 差 <0.5%。
2. CE 下"随机冻结臂"会饱和到均匀分布 = 零参数最优常数预测器 (0.1389 = predict_mean 基线),
   于是 E1 的 "learned < frozen" 对 CE 档结构上不成立 → **新增 E6**(低于该档自己的零参数基线),
   E1 保留不改。
