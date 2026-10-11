<!-- 权威副本. 本文件是 research/scale_heterogeneous/v35_container_entries/SPEC.md; docs/V35_0_SPEC.md 是指向本文件的薄指针. -->
<!-- 机器可读声明见同目录 MANIFEST.json; 指标快照见 results/metrics.json -->

# V35.0-α 大模型容器 + 异质入口注入规范

> **版本**: v35.0-α | **状态**: **Negative Result (负结果, 边界已界定)** (2026-09-08, 原快照)
> **归属路线**: 大模型容器 + 异质入口 (原 V6 衍生路线; 由外部快照 v9.0-α 重编号而来)
> **重编号对照**: 快照 v9.0-α → 本项目 v35.0-α (本项目 v9.0 已占用: gate-style 中枢)
> **核心命题**: 大模型作为容器, 不同类型小模型作为原生入口, 表征注入大模型内部
> **结果**: 负增益 (-19.5%), 入口注入在简单任务+强容器下是扰动非增量
> **复现命令**: `bash run.sh v35` (需 tabldm 可导入; 本环境已恢复 /workspace/xiaomi-tabldm 源码并装齐依赖)
> **本环境复现** (2026-09-11): seed 0 / 50 步 → base 1.6799 vs inj 1.7726, **gain -5.5%** (同方向; 100 步更充分时负得更多)

---

## 1. 核心命题

**命题 V35-α**: 大模型作为容器(主干), 不同类型小模型(TabLDM表格/embedding语义/reranker排序)
作为"原生入口"各自处理擅长的输入, 产出表征经投影后作为 prior 注入大模型容器内部
(inputs_embeds), 使融合优于随机 prior baseline。

---

## 2. 架构

```
TabLDM 入口:    表格 → hook 取 representations → c_struct (D=512)
embedding 入口: 文本 → 浅层编码                → c_sem   (D=1024)
reranker 入口:  文本 → 浅层编码                → c_rank  (D=1024)
                                              ↓ 各投影 → D_BIG=2560
                                         prefix [3, 2560]
                                              ↓ mean → prior
大模型容器 (MSA-4B 浅层):  embed + prior → 前向 → h
                                              ↓
                                         ProjHead → target
```

注入方式: prior 加到 embed (不增序列长度, 不干扰 attention), 非拼序列 prefix。

---

## 3. 负结果 (6 种配置全部为负)

| 配置 | gain |
|---|---|
| cat prefix 拼序列, 冻结 P_m | -11.8% |
| cat prefix 拼序列, P_m 可训练 | -47.4% |
| prior 加 embed, P_m 可训练 | -26.0% |
| prior 加 embed, 可学习 prefix (30步) | -25.9% |
| 公平对比 (随机prior vs 入口prior, 30步) | -14.0% |
| **公平对比 + 同幅度 + 100步充分收敛** | **-19.5%** |

100 步充分收敛: base=0.0166, v35=0.0199。两者都收敛到极低, 入口 prior 略差于随机。

---

## 4. 根因分析

1. **任务太简单**: class direction (ProjHead 2560→256 线性即可拟合), 大模型裸前向 + 任何 prior 都能收敛到 0.02。
2. **强容器 baseline**: MSA-4B 浅层裸前向 h_base 已是足够好的 2560 维表征, 入口 prior 是小扰动, 不提供信息增量。
3. **入口信息与任务无关**: TabLDM 表格/embedding 语义/reranker 排序的表征, 对"文本 class direction"任务无相关信息, 注入的是噪声方向。

---

## 5. 价值边界 (本负结果界定了 v35 的工作前提)

| 场景 | 入口注入价值 |
|---|---|
| 简单单模态任务 + 强大模型容器 | ✗ 无增量 (扰动) — **本实验证实** |
| 跨模态任务 (表格+文本, 单大模型拟合不了) | ✓ 待验证 — 入口必需 |
| 大模型容量不足 (浅层/小模型容器) | ✓ 待验证 — 入口补充 |
| v35-β 自训练 (预测编码) | ✓ 待验证 — 入口学到有意义注入 |

**结论**: "大模型容器入口注入"的价值不在简单任务, 而在**入口信息为任务所必需**的场景。
本负结果有意义: 证明了入口注入不是"万能增益", 需要任务-入口匹配。

---

## 6. 诚实声明

- v35-α 只验证"容器+入口注入"机制, 未做自训练(v35-β)和跨入口推动。
- 负结果不否定构想, 而是界定其前提: 需跨模态任务或自训练。
- TabLDM 经 forward hook 真实取 representations (D=512), 非代理。
- MSA-4B 浅层 1 层, 未体现完整 4B 容量。

---

## 7. 工程产出

- 子包: `v35_container_entries/` — `paths.py`(路径 + tabldm 源路径注入) · `entries.py`(get_tabldm_repr/load_backbone/load_big_container/big_forward) · `inject.py`(ProjHead)
- 示例: `examples/run_v35_container_entries.py` (薄编排层, `from v35_container_entries import ...`)
- 规范: `docs/V35_0_SPEC.md` (本文档, 负结果)
- 命令: `bash run.sh v35`
- 单测: `python3 tests/test_v35_container_entries.py` (10 项)
- 依赖: 需 `tabldm` (Xiaomi-TabLDM) 可导入; 脚本支持 `V35_TABLDM_SRC` 指定源码目录
  （本环境源码位于 `/workspace/xiaomi-tabldm`，依赖 scikit-learn/scipy/einops/psutil/pandas 已装）
- 权重: MSA-4B (/workspace/models/MSA-4B/), TabLDM ckpt (/workspace/models/Xiaomi-TabLDM/checkpoints/)
- **来源**: 外部快照 `hetero_fusion.zip` 的 v9.0-α (见 `docs/V33_V35_INTEGRATION.md`)

---

## 8. 系统演进总览 (本项目编号)

> **重要更正 (2026-09-11)**: 本规范的 §4 "根因分析"列了 3 条负结果根因, 但**真正的根因是设计层缺陷而非任务本身** —
> 入口表征被 `mean(dim=(0,1))` 折叠成常向量、prefix prior 被 `mean(dim=1, keepdim=True)` 塌缩成 `[B,1,2560]` 常数,
> 信息量为零。这一塌缩 bug 在外部快照 v9.0-α2 的修复版 (本项目 **v36.0**) 中被纠正:
> 逐样本入口 + 无参数 k-NN 评估后, 5/5 PASS +∞/+100%。详见 [`V36_0_SPEC.md`](V36_0_SPEC.md) §1。
> **本负结果作为 v9.0-α 的历史快照**仍保留 (序列完整), 但不应被引为"入口注入有害"的证据。

| 版本 | 范围 | 核心原理 | 状态 |
|---|---|---|---|
| v4.0 | 同构族 LM-FFN | token routing, STE | Baselined |
| v5.0-α | Embedding, sequence routing | errata V1/V2 | Verified |
| v5.1 | 路由粒度匹配 | N_dec ≈ N_sig | Baselined |
| v6.0 | 跨架构族融合 | 原生编码 + 矩形 P_m | Verified |
| v7.0–v32.0 | (本项目平行路线) | Attn+FFN 双路由 / 中枢 / 对齐器 / 蒸馏 / 多模态 … | 见 EVOLUTION.md |
| v33.0 | 规模异构融合 | 大维度差异 + 共享路由 | Verified |
| v34.0 | DMN 潜意识联想 | 跨专家×跨时刻联想 + 双管注入 | Verified |
| **v35.0-α** | **大模型容器+异质入口 (含塌缩 bug)** | **入口 prior 注入** | 负结果, **后由 v36 修复** |
| **v36.0** | **α2 跨模态验证 (修复 v35 塌缩)** | **逐样本入口 + k-NN** | **✅ Verified** |
| **v37.0** | **β 自训练预测编码入口** | **E 参与梯度 + 4 臂** | **✅ Verified** |
| **v38.0-α** | **γ 真实数据跨模态 (Adult XOR)** | **5 臂 + 蒸馏预热** | **⚠ 边界结果** |
