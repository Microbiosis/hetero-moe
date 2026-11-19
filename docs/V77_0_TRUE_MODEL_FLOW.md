# V77.0 — 路线 β：真 HF 模型的逐层驱动与层边界注入（S1β）

> 本文件是指针。权威规范在
> [`research/weight_fusion/v77_true_model_flow/SPEC.md`](../research/weight_fusion/v77_true_model_flow/SPEC.md)

**一句话**：在**真模型**上把「打断 → 注入 → 续跑」跑通并**逐位验真**；同时**正面确证**了 v76 的证伪，但「槽身份可分辨」这个对照测量**功效不足**。

| 入口 | 位置 |
|---|---|
| 权威规范 | `research/weight_fusion/v77_true_model_flow/SPEC.md` |
| 机器可读声明 | `research/weight_fusion/v77_true_model_flow/MANIFEST.json` |
| 指标快照 | `research/weight_fusion/v77_true_model_flow/results/metrics.json` |
| 权威测试 | `research/weight_fusion/v77_true_model_flow/tests/test_package.py` |

## ⚠️ 本代是 4/5 判据通过（机器裁决 `S1B_FAIL`）

- **机制层 B1–B4 全过**，且 B1/B2 是**位级**证据：切片权重与真模型参数 **72/72 块逐位相同**（rel_err = 0.0）；零注入的「打断→续跑」与原前向逐层**位级相同**（rel_err = 0.0）。
- 唯一未过的是 **B5**（**对照测量**，不是 P1b）：方向为正（占优 12/16、中位差 +0.0738）但**功效不足**（双侧符号检验 p = 0.0768；n=16 时需 ≥13 个正号）⇒ 读作「**不可判定**」，**不是**「槽身份在真模型上不可分辨」的否证。阈值一条未放宽。
- **事后拆到槽看**（快照里明确标注 `post_hoc_only`，不参与裁决）：**Q 槽 8/8 占优**（p = 0.0078），O 槽 4/8 ⇒ 判据粒度必须跟机制粒度对齐（这是 S3 的直接输入）。
- **v76 的证伪被正面确证**：路线 α 无信号（11/24，中位差 −0.00008）vs 路线 β 有信号（12/16，+0.0738）⇒ 语义层确实必须走真模型真输入这条路。
- **ViT 在真模型上没有 v76 的缺陷**（最大范数比 3.52 vs 路线 α 的 20.11）⇒ v76 的根因假设（切片缺 norm gain / embedding 尺度）成立。
- **Qwen 深层 hidden state 有效秩塌到个位数**（massive activations）⇒ S3 的线性探针必须控制有效秩。
- **注入方向在深层不可靠**（Qwen 22 层后 alignment = −0.162）⇒ S3 不能只靠方向对齐判定「信息流过去了」。

引用本代必须带上「机制层过 / 对照不可判定」这两半，以及「S2–S4 未做 ⇒ A1 整体未判定」。

复现需要真模型权重（约 1.3 GB），步骤见 SPEC.md §8。
