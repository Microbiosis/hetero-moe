# V84.0 附录 — Exp4/Exp5 消融复现与「β 墙」归因审计

> 本文件是**薄指针**，权威正文在
> [`research/weight_fusion/v84_inherent_flaw_remediation/ABLATION_AUDIT.md`](../research/weight_fusion/v84_inherent_flaw_remediation/ABLATION_AUDIT.md)

**一句话**：外部复现证明 v84 的「融合突破 β 墙」归因不成立——同数据同协议下，**单专家、
零融合**即可达到与 v5 相同的分类准确率（`1.0000`）且 LM PPL 更优（`2.198` < `2.311`）；
同时上游自己的 K=1 路由梯度修复把「冻结基线」从 `0.7667` 抬到 `0.8667`，对外宣称的
Δ `+22.8%` 实为 `+13.3pp`（缩水 42%）。Exp4 两个官方臂逐位复现，Exp5 PPL 存在 +0.26% 偏差。

| 入口 | 位置 |
|---|---|
| 权威正文 | [`research/weight_fusion/v84_inherent_flaw_remediation/ABLATION_AUDIT.md`](../research/weight_fusion/v84_inherent_flaw_remediation/ABLATION_AUDIT.md) |
| 复现脚本 | [`tools/ablation_v84_exp4_exp5_replication.py`](../tools/ablation_v84_exp4_exp5_replication.py) |
| 原始输出 | [`research/weight_fusion/v84_inherent_flaw_remediation/results/ablation_exp4_exp5_replication.json`](../research/weight_fusion/v84_inherent_flaw_remediation/results/ablation_exp4_exp5_replication.json) |
