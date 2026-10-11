<!-- 权威副本. 本文件是 research/routing_evolution/v46_attn_path_audit/SPEC.md -->
<!-- 机器可读声明见同目录 MANIFEST.json; 指标快照见 results/metrics.json -->

# V46.0 — 「增益集中在 attention 通路」的样本外审计

> **日期**: 2026-10-11
> **状态**: ✅ Verified (7/7 单元测试 PASS, 5 seeds × 2 口径)
> **复现命令**: `python3 examples/run_v46_attn_path_audit.py` (CPU 约 1 分钟)
> **审计对象**: `microfusion.tex:77/151` 的通路分解主张
> **一句话**: **主张成立, 且在样本外同样成立** —— 本轮审计中少数未被推翻的结果。

---

## 1. 主张与判定形式

`microfusion.tex:77`:

> *"a path decomposition shows the gain concentrates in the attention path:
> cross-architecture differences surface mainly in attention patterns"*

对应 v7 的四个通路分解量 `fuse / attn_only / ffn_only / avg`。
可判定形式: **attn_only 的误差显著小于 ffn_only**。

## 2. 原口径的问题

`examples/run_v7_cross_arch.py` 的四个量**全部在训练同源数据上评估**
(第 100–122 行与训练共用 `x_shared_list` / `targets`),
且该 runner **没有任何 held-out 构造**(grep 计数 0)。

本代复用 v7 全部构件(层构造 / 训练器 / 编码 / 数据构造), 只把评估扩到两个口径。

## 3. 结果(5 seeds,配对逐样本中位数)

| 口径 | fuse | attn_only | ffn_only | avg | 通路差(attn−ffn) |
|---|---:|---:|---:|---:|---:|
| in-sample | 1.1061 | 1.2784 | 7.2002 | 7.1677 | **+82.7% (5/5)** |
| **held-out** | 2.0626 | 2.1330 | 7.1651 | 7.1947 | **+70.6% (5/5)** |

逐 seed 的 held-out 通路差: `+3.9 / +30.4 / +74.4 / +46.1 / +1.7` —— 5/5 全胜。

## 4. 裁决

> **主张成立。** attn 通路相对 ffn 通路的优势在两个口径下同向且都是 5/5:
> in-sample **+82.7%**, held-out **+70.6%**。
>
> 样本外的优势**略小但方向不变**, 因此这项主张**不依赖 in-sample 评估口径**。

## 5. 必须一起说的限定

`fuse` 的绝对值从 in-sample 的 1.1061 劣化到 held-out 的 2.0626(**+86%**)。
也就是说**通路分解本身的绝对性能泛化不佳**。

但通路之间的**相对次序**稳定, 而"增益集中在 attention 通路"是一句**相对**主张,
所以不受影响。

> 这个区分很重要: 若论文写的是"attention 通路 achieves X", 那是绝对主张, 会被这 86% 推翻;
> 它写的是 "gain concentrates in", 是相对主张, 站得住。

## 6. 局限

- CPU 沙箱: `D_shared=256`, 序列长 16, 合成 class-direction 目标
- 只审 v7-A(跨架构路径); **v7-B(同架构多任务)未审**
- `avg`(无路由 STE 的等权基线)在两个口径下都劣于 fuse(7.17 / 7.19 vs 1.11 / 2.06),
  说明路由本身有增益 —— 但 `avg` 与 `ffn_only` 几乎相等, 两者都退化为"仅 FFN 通路",
  这一项的区分度有限。

## 7. 工程产出

- `v46_attn_path_audit/path_audit.py` — 配对通路差 + 三态裁决 + 表格渲染
- `v46_attn_path_audit/tests/test_package.py` — 7 项单元测试(权威副本)
- `tests/test_v46_attn_path_audit.py` — 根转发层
- `examples/run_v46_attn_path_audit.py`
- `results/metrics.json` / `results/raw.json`
