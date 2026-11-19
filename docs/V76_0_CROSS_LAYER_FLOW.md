# V76.0 — A1 跨层信息流：从权重切片重建逐层前向（S1）

> 本文件是指针。权威规范在
> [research/weight_fusion/v76_cross_layer_flow/SPEC.md](../research/weight_fusion/v76_cross_layer_flow/SPEC.md)

**一句话**：A1 的第一步 —— 先证明「打断 → 注入 → 续跑」在计算图层面真的能这么干；顺带证明「随机输入这条路测不了语义」。

| 入口 | 位置 |
|---|---|
| 权威规范 | research/weight_fusion/v76_cross_layer_flow/SPEC.md |
| 机器可读声明 | research/weight_fusion/v76_cross_layer_flow/MANIFEST.json |
| 指标快照 | research/weight_fusion/v76_cross_layer_flow/results/metrics.json |
| 权威测试 | research/weight_fusion/v76_cross_layer_flow/tests/test_package.py |

## ✅ S1 PASS（五项机械自检全过）

C1 结构合法（Qwen GQA 14 头 / 2 KV 头 × 64，K=[128,896]）· C2 口径同源（相对差 5.16e-8，n=1728，p=0.0002）· C3 层栈健康（Qwen/BERT）· C4 注入通路（扰动 0.63~0.68，alignment 0.63~0.83，方向 100% 为正）· C5 跨模型维数通路（正交残差 < 1e-5）。

## ⚠️ 本代只确认机制，不确认语义

- S2（注入算子）/ S3（语义判据）/ S4（真模型复现）**未做** ⇒ **A1 整体未判定**。
- **路线 α 测不了语义层**：随机输入下匹配槽 R² 0.964~0.992，但错槽几乎相同（差值中位数 −0.0000754，正号 11/24 = 45.8%）⇒ P1a/P1b/P2/P3 必须上路线 β（真模型逐层驱动 + 真输入 + 标签）。
- **ViT 进不了首代范围**：layer0 残差膨胀 20.11×（判据上限 10×），有效秩 744→64，根因是 FFN 分支 RMS 20.84 而非仿射 RMSNorm 压不住 ⇒ 修好之前不纳入。

引用本代必须带上「只证机制不证语义」与「S2–S4 未做 ⇒ A1 整体未判定」这两条。

复现需要权重切片缓存，步骤见 SPEC.md §7。
