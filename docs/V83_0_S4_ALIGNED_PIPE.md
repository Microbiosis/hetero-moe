# V83.0 — S4-AP：Q-K 接收场正交对齐 + 源侧同层正交解耦管道

> 本文件是指针。权威规范在
> [`research/weight_fusion/v83_s4_aligned_pipe/SPEC.md`](../research/weight_fusion/v83_s4_aligned_pipe/SPEC.md)

**一句话**：v82（S4-P）暴露出的两个机制失败项——**P5a 管道不被按内容读取**（`arm1−arm3 = −0.000983, p = 0.9053`）与 **P5d 非目标槽普涨**（`r2_o 增量 +0.001921, p = 0.0`）——根因分别在于：① 载体 `Δ = (a_s @ P_Aᵀ) @ P_B` 使用未对齐的独立随机正交投影，注入方向与目标模型第 0 层注意力的 Query-Key 接收锥正交；② `v77–v82` 对 O 槽抓取的是 `o_proj` 的输入而非输出，且样本级 `mean-pool` 使同层 Q/O 共享句子主成分。本代用**纯前向闭式算子 + 训练集内正交 Procrustes 对齐**（`Ω_s ∈ O(256)`，严格等距、零测试集泄漏）与**真 O 槽输出 + 训练集内同层正交解耦**（`a_{Q⊥O}, a_{O⊥Q}`）同时修复这两处缺陷，使六项主判据从 v82 的 **3/6 通过**推进到 **5/6 通过**（**P5a / P5b / P5c / P5d / P3① 全过**，正文侧槽特异 `arm1−arm2` 在 **BERT↔Qwen 双向均以 p = 0.0 通过**，逐样本通道寻址从 v82 的 `p = 0.787` 跃升至 `p < 1e-100`），并定量证明唯一未过项 **P2′（深度语义）** 受制于 Qwen 第 0 层 GQA Value-Output 算子 `M_vo^(0)` 的 128 维秩瓶颈与 43 倍范数衰减。

| 入口 | 位置 |
|---|---|
| 权威规范 | [`research/weight_fusion/v83_s4_aligned_pipe/SPEC.md`](../research/weight_fusion/v83_s4_aligned_pipe/SPEC.md) |
| 机器可读声明 | [`research/weight_fusion/v83_s4_aligned_pipe/MANIFEST.json`](../research/weight_fusion/v83_s4_aligned_pipe/MANIFEST.json) |
| 指标快照（正本） | [`research/weight_fusion/v83_s4_aligned_pipe/results/metrics.json`](../research/weight_fusion/v83_s4_aligned_pipe/results/metrics.json) |
| 逐块 checkpoint | [`research/weight_fusion/v83_s4_aligned_pipe/results/blocks.jsonl`](../research/weight_fusion/v83_s4_aligned_pipe/results/blocks.jsonl) |
| λ 标定（正本） | [`research/weight_fusion/v83_s4_aligned_pipe/results/calib.json`](../research/weight_fusion/v83_s4_aligned_pipe/results/calib.json) |
| 双向槽特异 + GQA `M_vo` 谱瓶颈诊断（post_hoc_only） | [`research/weight_fusion/v83_s4_aligned_pipe/results/diag_mvo_readout.json`](../research/weight_fusion/v83_s4_aligned_pipe/results/diag_mvo_readout.json) |
| 权威测试 | [`research/weight_fusion/v83_s4_aligned_pipe/tests/test_package.py`](../research/weight_fusion/v83_s4_aligned_pipe/tests/test_package.py) |
