# V50.0 — 「融合收益从哪来」审计

> **薄指针层**: 权威正文在 `research/aligner/v50_fusion_gain_audit/SPEC.md`。
>
> - 规范: [`research/aligner/v50_fusion_gain_audit/SPEC.md`](../research/aligner/v50_fusion_gain_audit/SPEC.md)
> - 机器可读声明: `research/aligner/v50_fusion_gain_audit/MANIFEST.json`
> - 指标快照: `research/aligner/v50_fusion_gain_audit/results/metrics.json`
> - 复现: `bash run.sh v50` / HF Space `redmingwei/hetero-moe`
>
> **一句话**: 审计论文的"+26.7% survives" → 在论文自己的配置下, 融合输出对**源专家身份**
> 的 held-out 可解码性 = 1.000 而目标类别只有 0.279 (随机基线 0.125) → 那个 margin
> 测的是"认出这是哪个底座的"再加分派, 不是语义融合; 空间一旦有意义 (非线性可训练 /
> proxy 对齐), margin 塌到 +4.1%~+12.0%, 绝对误差同时降到平凡基线以下。
