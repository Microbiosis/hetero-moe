# V51.0 — 架构异构线 (v6/v7) 融合收益审计

> **薄指针层**: 权威正文在 `research/aligner/v51_arch_fusion_audit/SPEC.md`。
>
> - 规范: [`research/aligner/v51_arch_fusion_audit/SPEC.md`](../research/aligner/v51_arch_fusion_audit/SPEC.md)
> - 机器可读声明: `research/aligner/v51_arch_fusion_audit/MANIFEST.json`
> - 指标快照: `research/aligner/v51_arch_fusion_audit/results/metrics.json`
> - 复现: `bash run.sh v51` / HF Space `redmingwei/hetero-moe`
>
> **一句话**: 这条线的**目标块就是模态下标** → 任务是"识别来自哪个底座"; 目标置换后
> margin 纹丝不动 (+70.4% → +72.7%), 换成有意义空间后塌到 +2.3% (绝对误差降 68 倍),
> **attention 通路的领先也随之从 +69% 降到 3%** → 论文的"+20.3%"与"增益集中在
> attention 通路"两条主张, 都是冻结随机空间下的产物。
