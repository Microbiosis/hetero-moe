# 研究线 8: scale-heterogeneous (规模异构 + 大模型容器范式)

**研究问题**: 与 v6-v9 路由融合**完全不同的融合范式**——"1大+N小"规模异构 + "大模型容器 + 异质入口注入"。

**父版本**: 独立范式, 无父版本 (与 v6-v9 平行)
**子版本**: 自演进 (v33 → v34 → v35 → v36 → v37 → v38 → v39)

**注**: 本线使用的"大模型主干"和"小模型入口"是机构内部的辅助模型（proprietary auxiliary models）；公开仓库中的代码以泛化名 `BIG_LM` / `TAB_LM` / `EMBED_LM` / `RERANK_LM` 引用这些模型，模型权重可在评审需要时向作者索取。

## ⚠️ 范式独立性

这条线**完全独立于** research/routing_evolution/、research/aligner/ 等其他所有研究线。
- 不复用 v7 的双路由
- 不复用 v9 的 gate-style 中枢
- 不复用 v13 的 MixtureAligner
- **复用关系**: v34←v33, v37←v36, v38/v39←v36+v37 (本线内部复用)

## 成员

| 版本 | 范式 | 关键命题 | 关键模块 |
|---|---|---|---|
| v33 | "1大+N小" 规模异构 | 高容量大模型 + 多个低容量小模型, 各自独立前向 → 路由融合 | `paths.BIG_LM_DIR/EMBED_LM_DIR`, `encoders.load_big_lm/load_small`, `fusion.CrossModalFusionLayer` |
| v34 | DMN 潜意识创造性联想 (v33 之上的"涌现"层) | 显式路由(意识) + DMN 联想层(潜意识) → 双管注入 | `dmn.DMNSubconscious`, `fusion.FusionLayer` |
| v35 | 大模型容器 + 异质入口注入 | 大模型作为主干, 不同类型小模型作为"原生入口" → prior 注入 | `entries.get_tabular_repr`, `load_big_container`, `big_forward`, `inject.ProjHead` |
| v36 | α2 表格-语言跨模态 + kNN 评估 | 修复 v35 prefix 塌缩; 入口表征逐样本 | `entries.encode_baseline_text`, `knn_acc`, `concat_norm_then_knn` |
| v37 | β 自训练预测编码入口 | 入口编码器 E 可训练, 唯一监督 = 容器对目标的预测误差 | `entry.EntryEncoder` |
| v38 | γ Adult XOR + 5 臂 | 真实数据上自训练/被动/先验+自训练打平 (β 边界) | `data.adult_loader.load_adult_crossmodal` |
| v39 | γ2 加性 vs 交互融合框架 | 修正 v38 架构层假边界, 加性 vs 交互融合 | `heads.make_head/fuse/logits_of` |

## 关键发现

### v33-v34: 规模异构 + 涌现

- 1大+N小 线性融合可行
- v34 DMN 联想层 > v33 纯线性路由 ("灵光一现" > "理性选择")

### v35-v39: 大模型容器范式 (含 4 个连续发现)

- **v35.0-α: 负结果 (-19.5%)** — 简单任务 + 强容器下入口注入是扰动非增量
- **v36.0 α2: 修复 v35 prefix 塌缩 bug** — 表格-语言中间量不再 mean 折叠成 [512]
- **v37.0 β: 自训练入口预测编码闭环** — 4 臂对照 (无/随机/被动/自训练)
- **v38.0 γ: Adult XOR 5 臂边界结果** — 自训练/被动/先验+自训练在 0.755-0.767 打平 (β 在真实复杂数据上不成立)
- **v39.0 γ2: 架构层修正** — 加性 vs 交互融合框架, 双数据集 (Adult + Shoppers)

## 边界与负结果

- **v35.0-α: -19.5% 入口注入负结果** — 简单任务上容器范式不增量
- **v36 α2 prefix 塌缩 bug** — v35 评估失误根因, 由 v36 修复
- **v38 边界** — β 自训练 vs 被动在真实数据上不成立 (C/D/E 打平)

## 启示

- **容器范式**与**路由融合范式**是两条路径, 不是同一问题的两个答案
- 容器范式的成功条件更苛刻: 需要真正的跨模态任务 + 入口具有任务相关信息
- 简单玩具任务下容器范式退步 (v35 负结果)

## 测试

- test_v33_scale_hetero.py (10 PASS)
- test_v34_dmn_subconscious.py (10 PASS)
- test_v35_container_entries.py (10 PASS)
- test_v36_alpha2_knn.py (10 PASS)
- test_v37_b_selftrained_entry.py (10 PASS)
- test_v38_g_adult_boundary.py (10 PASS)
- test_v39_g2_interaction.py (10 PASS)