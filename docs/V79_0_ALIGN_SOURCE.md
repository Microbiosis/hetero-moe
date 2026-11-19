# V79.0 — S2-0：定死 align[s|A→B] 的来源（**负结果：没有可用来源**）

> 本文件是指针。权威规范在
> [`research/weight_fusion/v79_align_source/SPEC.md`](../research/weight_fusion/v79_align_source/SPEC.md)

**一句话**：S3 的注入算子需要一个跨模型槽对齐映射；四个候选来源在**留出集**上没有一个能超过「什么都不做」⇒ 这个对象在真模型激活上不存在可用的版本，**S3 的注入算子口径必须先修订才能开跑**。

| 入口 | 位置 |
|---|---|
| 权威规范 | `research/weight_fusion/v79_align_source/SPEC.md` |
| 机器可读声明 | `research/weight_fusion/v79_align_source/MANIFEST.json` |
| 指标快照 | `research/weight_fusion/v79_align_source/results/metrics.json` |
| 块级映射（工程件） | `research/weight_fusion/v79_align_source/results/align_maps.pt`（64 张，**post_hoc 依据、非裁决产物**） |
| 权威测试 | `research/weight_fusion/v79_align_source/tests/test_package.py` |

## ❌ 本代是负结果（机器裁决 `ALIGN_SOURCE_NOT_FOUND`）

- **功效闸门过**（32 块/候选/槽）、**A1 过**（不对齐基线上同槽 0.6958 vs 错槽 0.4112，29/32，p<1e-4 ⇒ 槽的差别跨模型可见）。
- **A2 对全部候选不过**：matched − no-align 中位差 = none 0.0000 / procrustes 0.0000 / v75_synth −0.0129 / ridge −0.0831 / lowrank16 −0.2772 ⇒ **没有一个候选能超过「什么都不做」**。A3（λ 稳健）也不过。阈值一条未放宽。
- **不是过拟合**：池化映射在**训练集**上也不如基线（0.5082 < 0.5990）⇒ 是没东西可学。
- **机理（post_hoc）**：Q 槽 mean-pool 后有效秩只有 4.84–38.74（O 槽 58–69）⇒ 跨模型的高 CKA 主要来自**低秩共同成分**；去掉 top1 PC 后，连块级映射的优势也消失（A2 −0.0261，p=1.0）。
- **对主线的压力**：v77 的 B5（matched R² 0.72 vs wrong 0.518）与 v78 的 Q 槽 8/8，很可能是同一个共同成分现象 —— 本代**不能直接否证**（post_hoc），但必须重新读。
- **对 S3 的后果**：协议 §2.3 的「`z~ = align[s|A→B]·z`」不能按原文开跑；且对齐映射只能在**样本级**拟合（token 网格不可配对）⇒ 与 S3 拍板的 token 级口径不相容 ⇒ **注入算子口径已于 2026-11-17 拍板：保留共享空间载体、不学任何映射**（协议已修订）。

引用本代必须带上「负结果 + 机理 + post_hoc 不参与裁决」这三半，以及「S3 仍未开跑、A1 整体仍未判定」。

复现步骤见 SPEC.md §7（本机 CPU 约 50 秒；退出码 1 属契约内，不是崩溃）。
