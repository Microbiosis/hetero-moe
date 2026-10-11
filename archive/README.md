# archive/ — 负面发现 + 前基线

不进入主线迭代的实验归档。

## pre_baseline/ — 前基线 (v6 之前的实验)

### v5_alpha/
- **状态**: 与 v6-v9 不同范式的前基线 (序列级路由 + 嵌入场景)
- **入栈原因**: 与 v6 起的"跨架构族融合"是不同研究问题, 不再随主线演进
- **保留**: 后续 v7/v10 路由实验可能复用其序列级 STE 思想

## negative_findings/ — 负面发现 (实验执行了, 但结论为"地图边界")

### v12_compose/ (8 PASS test)
- **实验**: v9 gate-style 中枢 + v10 attn 对齐器组合
- **结论**: 0 叠加增益 — "组合多版本不总是更好"
- **保留**: 警示后续不要再做"无脑叠加"

### v23b_hard_vqa/ (8 PASS test)
- **实验**: 8 类陷阱 toy VQA (从 v23_vqa/hard_dataset.py 抽出)
- **结论**: 所有模型 (含 multimodal) 仍 50% = 随机水平 — "toy 数据集根本不可行"
- **保留**: 警示后续 toy 任务规模需要 > 24 样本 + 真实模态差异

### v27_router_norm_fix/ — 注意: 此包**仍在主研究线** (research/corpus_scale/)
- v27 是真实实验, 4 patch × 4 mode × 5 seeds = 80 run, 全部失败
- 包含**最有信息量的负结果之一**: v22 patch 已局部最优
- 不进 archive, 因为是研究线 7 的合法一员

### v28_architectural_fix/ — 同样仍在主研究线 (research/corpus_scale/)
- v28 是真实实验, 11 架构变体 × 5 seeds = 55 run, 7 变体全部失败
- 包含另一最有信息量的负结果: 架构层救不了数据规模问题