# V84.0 附录 — Exp4/Exp5 消融复现与「β 墙」归因审计及闭环修复

> 本文件是**薄指针**，权威正文在
> [`research/weight_fusion/v84_inherent_flaw_remediation/ABLATION_AUDIT.md`](../research/weight_fusion/v84_inherent_flaw_remediation/ABLATION_AUDIT.md)

**一句话**：针对初版消融审计发现的「合成数据未赋予预训练专家不可替代先验导致单专家零融合可打平」及三项跨平台/干净克隆缺陷，已在 `v84` 中引入 **64 锚点预训练分类先验**与 **8 形态变体预训练 LM 先验**，并将全部单专家无融合臂（`single_trainable` / `single_frozen` / `single_lora`）直接纳入 `DownstreamEvalGate` 强制对比：修复后 `v5` 在分类上达 **`0.9111`**（严格击败 `v4` `0.5889`、`single_trainable` `0.4778`、`single_frozen` `0.2611`），在 LM 上达 **`PPL = 4.259`**（严格优于 `best_single` `8.054` 与 `single_trainable` `15.347`），同时修复了 CRLF 哈希、`.venv` 干净克隆与 `float64 pinv` 跨平台一致性。

| 入口 | 位置 |
|---|---|
| 权威正文 | [`research/weight_fusion/v84_inherent_flaw_remediation/ABLATION_AUDIT.md`](../research/weight_fusion/v84_inherent_flaw_remediation/ABLATION_AUDIT.md) |
| 复现脚本 | [`tools/ablation_v84_exp4_exp5_replication.py`](../tools/ablation_v84_exp4_exp5_replication.py) |
| 原始输出 | [`research/weight_fusion/v84_inherent_flaw_remediation/results/ablation_exp4_exp5_replication.json`](../research/weight_fusion/v84_inherent_flaw_remediation/results/ablation_exp4_exp5_replication.json) |
